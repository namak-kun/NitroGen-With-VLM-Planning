"""
PoC: Plan-conditioned NitroGen.

Wires a frozen Qwen3.5-0.8B VLM planner -> PlanResampler (K learned queries) ->
PlanAdapter (1024->1024) -> injects K "plan tokens" into NitroGen's VL cross-
attention stream alongside the 256 SigLIP image tokens, then runs the real
NitroGen DiT flow-matching forward + backward on random data.

Goal of this PoC (NOT training):
  1. Prove the architecture wires end-to-end against the REAL ng.pt + REAL Qwen3.5.
  2. Prove gradients flow into the new modules (resampler/adapter/null_plan).
  3. Prove the plan channel has EFFECT: null-plan vs plan produce different
     predicted velocities for the same frame/noise/timestep.

Run:
  cd /home/t-nagupta/NitroGen-With-VLM-Planning && source .venv/bin/activate && \
  python3 files/poc_planner.py   # (path adjusted; see __main__)
"""
import os, sys, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
NG_CKPT = os.path.join(REPO, "ckpts/nitrogen/ng.pt")
QWEN_DIR = os.path.join(REPO, "ckpts/qwen35-0.8b")
sys.path.insert(0, REPO)

# ---------------------------------------------------------------------------
# Env-compat shim: transformers>=5.x SiglipVisionModel no longer exposes
# `.vision_model` (it IS the vision model). NitroGen.__init__ does
# `self.vision_encoder = model.vision_model`. Add a property that returns self
# so the existing repo code loads unmodified. State-dict keys (vision_encoder.*)
# still match because the child module names are identical.
# ---------------------------------------------------------------------------
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda self: self)

from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig


# ---------------------------------------------------------------------------
# New trainable modules (the actual proposal)
# ---------------------------------------------------------------------------
class PlanResampler(nn.Module):
    """K learned queries cross-attend to VLM hidden states -> (B, K, d)."""
    def __init__(self, dim=1024, num_queries=8, num_heads=8, num_layers=2, ff_mult=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(dim),
                "ln_kv": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, num_heads, batch_first=True),
                "ln_ff": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(),
                                    nn.Linear(dim * ff_mult, dim)),
            }))

    def forward(self, h, key_padding_mask=None):
        # h: (B, L, d); key_padding_mask: (B, L) True = pad/ignore
        B = h.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        for blk in self.layers:
            qn = blk["ln_q"](q)
            kv = blk["ln_kv"](h)
            attn_out, _ = blk["attn"](qn, kv, kv, key_padding_mask=key_padding_mask,
                                      need_weights=False)
            q = q + attn_out
            q = q + blk["ff"](blk["ln_ff"](q))
        return q  # (B, K, d)


class PlanAdapter(nn.Module):
    """1024 -> 1024 map into NitroGen's vision_hidden_size space."""
    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.net(x)


class PlanHead(nn.Module):
    """Resampler + adapter + a learned null-plan token for plan-dropout / CFG."""
    def __init__(self, dim=1024, num_queries=8):
        super().__init__()
        self.K = num_queries
        self.resampler = PlanResampler(dim=dim, num_queries=num_queries)
        self.adapter = PlanAdapter(dim=dim)
        self.null_plan = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

    def forward(self, vlm_hidden, key_padding_mask=None, dropped=None):
        """vlm_hidden: (B,L,d). dropped: (B,) bool -> use null plan for those."""
        B = vlm_hidden.shape[0]
        plan = self.adapter(self.resampler(vlm_hidden, key_padding_mask))  # (B,K,d)
        null = self.null_plan.unsqueeze(0).expand(B, -1, -1)               # (B,K,d)
        if dropped is not None:
            m = dropped.view(B, 1, 1).to(plan.dtype)
            plan = (1 - m) * plan + m * null
        return plan, null


# ---------------------------------------------------------------------------
# Plan encoder (frozen Qwen3.5-0.8B). Returns last-layer hidden states.
# ---------------------------------------------------------------------------
def load_planner(device, dtype):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN_DIR)
    vlm = AutoModelForImageTextToText.from_pretrained(
        QWEN_DIR, dtype=dtype, low_cpu_mem_usage=True)
    vlm.to(device).eval()
    for p in vlm.parameters():
        p.requires_grad_(False)
    return vlm, tok


@torch.no_grad()
def encode_plan_text(vlm, tok, texts, device):
    """Text-only encode -> (B, L, 1024) last hidden states + key_padding_mask."""
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    # text backbone path of the VLM
    lm = vlm.model if hasattr(vlm, "model") else vlm
    out = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
             output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else out.last_hidden_state
    key_padding_mask = enc["attention_mask"] == 0  # True = pad
    return h.float(), key_padding_mask


# ---------------------------------------------------------------------------
# Load real NitroGen
# ---------------------------------------------------------------------------
def load_nitrogen(device):
    ckpt = torch.load(NG_CKPT, map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    model = NitroGen(config=cc.model_cfg, game_mapping=None)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    # unexpected = siglip text_model keys (we only keep the vision tower) -> fine
    real_missing = [k for k in missing if not k.startswith("vision_encoder.head")]
    model.to(device).eval()
    return model, cc, real_missing, unexpected


def main():
    t0 = time.time()
    device = "cuda"
    dtype = torch.bfloat16
    log = []
    def out(s):
        print(s); log.append(str(s))

    out("=" * 70)
    out("PoC: Plan-conditioned NitroGen  (real ng.pt + real Qwen3.5-0.8B)")
    out("=" * 70)

    # ---- load models
    out("\n[1] Loading real NitroGen from ng.pt ...")
    ng, cc, missing, unexpected = load_nitrogen(device)
    K = 8
    d = cc.model_cfg.hidden_size
    out(f"    hidden_size d        = {d}")
    out(f"    vision_hidden_size   = {cc.model_cfg.vision_hidden_size}")
    out(f"    action_dim/horizon   = {cc.model_cfg.action_dim} / {cc.model_cfg.action_horizon}")
    out(f"    DiT layers/interleave= {cc.model_cfg.diffusion_model_cfg.num_layers} / "
        f"{cc.model_cfg.diffusion_model_cfg.interleave_self_attention}")
    out(f"    state_dict load: {len(missing)} real-missing, {len(unexpected)} unexpected (siglip text head, expected)")
    assert len(missing) == 0, f"unexpected missing keys: {missing[:10]}"

    out("\n[2] Loading frozen Qwen3.5-0.8B planner ...")
    vlm, tok = load_planner(device, dtype)
    n_vlm = sum(p.numel() for p in vlm.parameters())
    out(f"    Qwen3.5 params       = {n_vlm/1e9:.2f}B (frozen)")

    out("\n[3] Building PlanHead (resampler + adapter + null) ...")
    plan_head = PlanHead(dim=d, num_queries=K).to(device).float()
    n_new = sum(p.numel() for p in plan_head.parameters())
    out(f"    K plan tokens        = {K}")
    out(f"    NEW trainable params = {n_new/1e6:.2f}M")

    # ---- fake batch
    B = 2
    H = cc.model_cfg.action_horizon       # 18
    A = cc.model_cfg.action_dim           # 25
    out(f"\n[4] Fake batch: B={B}, frame=256x256, action target ({B},{H},{A})")
    frame = torch.rand(B, 1, 3, 256, 256, device=device)  # (B, num_frames=1, C,H,W)
    actions = torch.randn(B, H, A, device=device)
    embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

    plan_texts = ["go left immediately, move to the left side",
                  "do nothing, stay idle and wait"]

    # ---- encode plan with frozen VLM
    out("\n[5] Encoding plan text with frozen Qwen3.5 (text-only path) ...")
    h_plan, kpm = encode_plan_text(vlm, tok, plan_texts, device)
    out(f"    VLM hidden states    = {tuple(h_plan.shape)}  (B, L, d)")
    assert h_plan.shape[-1] == d, f"VLM dim {h_plan.shape[-1]} != NitroGen dim {d}"
    out(f"    -> dims match ({d}); no projector needed.  [CONFIRMS warm-start claim]")

    # ---- build plan tokens (no dropout = plan present) and null tokens
    plan_tokens, null_tokens = plan_head(h_plan, key_padding_mask=kpm,
                                         dropped=torch.zeros(B, device=device))
    out(f"    plan tokens          = {tuple(plan_tokens.shape)}  (B, K, d)")

    def run_dit(vl_extra_tokens):
        """Replicate NitroGen.forward's core with extra VL tokens prepended."""
        with torch.autocast(device_type="cuda", dtype=dtype):
            vis = ng.encode_images(frame)                 # (B, 1, 256, d)
            vis = vis.reshape(B, -1, d)                   # (B, 256, d)
            vl = torch.cat([vl_extra_tokens.to(vis.dtype), vis], dim=1)  # (B, K+256, d)
            vl = ng.vl_self_attention_model(vl)

            noise = torch.randn_like(actions)
            t = ng.sample_time(B, device=device, dtype=actions.dtype)[:, None, None]
            noisy = (1 - t) * noise + t * actions
            velocity = actions - noise
            t_disc = (t[:, 0, 0] * ng.num_timestep_buckets).long()
            act_feat = ng.action_encoder(noisy, t_disc, embodiment_id)
            model_out = ng.model(hidden_states=act_feat, encoder_hidden_states=vl,
                                 encoder_attention_mask=None, timestep=t_disc)
            pred = ng.action_decoder(model_out, embodiment_id)
            pred_v = pred[:, -H:]
            loss = F.mse_loss(pred_v.float(), velocity.float())
        return loss, pred_v.float(), (noise, t)

    out("\n[6] Forward + backward (plan present) ...")
    loss, pred_v_plan, _ = run_dit(plan_tokens)
    out(f"    flow-matching loss   = {loss.item():.4f}")
    loss.backward()

    # ---- check grads on the NEW modules
    grad_norms = {}
    for name, mod in [("resampler.queries", plan_head.resampler.queries),
                      ("null_plan", plan_head.null_plan)]:
        g = mod.grad
        grad_norms[name] = None if g is None else g.norm().item()
    adapter_g = sum((p.grad.norm().item() for p in plan_head.adapter.parameters()
                     if p.grad is not None))
    resampler_g = sum((p.grad.norm().item() for p in plan_head.resampler.parameters()
                       if p.grad is not None))
    out(f"    grad norm resampler  = {resampler_g:.4e}")
    out(f"    grad norm adapter    = {adapter_g:.4e}")
    out(f"    grad norm null_plan  = {grad_norms['null_plan']}")
    grads_ok = resampler_g > 0 and adapter_g > 0
    out(f"    GRAD FLOW INTO NEW MODULES: {'PASS' if grads_ok else 'FAIL'}")

    # ---- effect of plan: null vs plan velocity, same frame/noise/t
    out("\n[7] Plan EFFECT test: null-plan vs plan velocity (same frame) ...")
    torch.manual_seed(0)
    with torch.no_grad():
        # fix noise & t by seeding inside run; instead compute both with same seed
        def velocity_for(tokens, seed):
            torch.manual_seed(seed)
            with torch.autocast(device_type="cuda", dtype=dtype):
                vis = ng.encode_images(frame).reshape(B, -1, d)
                vl = torch.cat([tokens.to(vis.dtype), vis], dim=1)
                vl = ng.vl_self_attention_model(vl)
                noise = torch.randn_like(actions)
                t = ng.sample_time(B, device=device, dtype=actions.dtype)[:, None, None]
                noisy = (1 - t) * noise + t * actions
                t_disc = (t[:, 0, 0] * ng.num_timestep_buckets).long()
                af = ng.action_encoder(noisy, t_disc, embodiment_id)
                mo = ng.model(hidden_states=af, encoder_hidden_states=vl,
                              encoder_attention_mask=None, timestep=t_disc)
                return ng.action_decoder(mo, embodiment_id)[:, -H:].float()
        v_plan = velocity_for(plan_tokens, 123)
        v_null = velocity_for(null_tokens, 123)
        diff = (v_plan - v_null).norm().item()
        rel = diff / (v_null.norm().item() + 1e-6)
    out(f"    ||v_plan - v_null||  = {diff:.4f}   (rel {rel:.3f})")
    effect_ok = diff > 1e-3
    out(f"    PLAN CHANNEL HAS EFFECT: {'PASS' if effect_ok else 'FAIL'}")

    # ---- bonus: confirm Qwen3.5 accepts a real image (frames-into-planner)
    out("\n[8] Bonus: confirm Qwen3.5 ingests an image natively ...")
    img_ok = False
    try:
        from transformers import AutoProcessor
        from PIL import Image
        import numpy as np
        proc = AutoProcessor.from_pretrained(QWEN_DIR)
        img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "describe the plan"}]}]
        chat = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        binp = proc(text=[chat], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            o = vlm(**binp, output_hidden_states=True, use_cache=False)
        hh = o.hidden_states[-1]
        out(f"    image+text hidden    = {tuple(hh.shape)}  -> Qwen3.5 sees frames. PASS")
        img_ok = True
    except Exception as e:
        out(f"    (image path skipped: {type(e).__name__}: {str(e)[:120]})")
        out("    NOTE: text path already proves the wiring; image path is a nicety.")

    # ---- summary
    out("\n" + "=" * 70)
    allpass = grads_ok and effect_ok
    out(f"RESULT: wiring={'OK' if len(missing)==0 else 'BAD'} | "
        f"grad_flow={'PASS' if grads_ok else 'FAIL'} | "
        f"plan_effect={'PASS' if effect_ok else 'FAIL'} | "
        f"vlm_image={'PASS' if img_ok else 'n/a'}")
    out(f"OVERALL: {'*** PoC PASS ***' if allpass else '!!! PoC FAIL !!!'}")
    out(f"elapsed {time.time()-t0:.1f}s")
    out("=" * 70)

    with open(os.path.join(os.path.dirname(__file__), "poc_results.txt"), "w") as f:
        f.write("\n".join(log))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
