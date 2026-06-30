"""Attention A/B eval for SEQ ordering: compares self-attn OFF vs ON resampler on the
within-chunk temporal task. CRITICAL: loads each checkpoint with the correct
resampler_query_self_attn flag (else the ON ckpt's self-attn params are silently dropped).
Reports per-plan first/second-half split (ordering quality) + a token-diversity metric
(mean pairwise cosine among the K plan tokens; lower = more diverse/specialized).
"""
import os, sys, glob
import numpy as np
import torch
import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO)
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda self: self)
from PIL import Image
from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22; TS = [200, 500, 800]
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[-8:]
fn = torch.tensor(np.random.RandomState(1).randn(1, H, 25), dtype=torch.float32, device=device)

SEQ = {
    "left->right": ("go left then right", "x", (-1, +1)),
    "right->left": ("go right then left", "x", (+1, -1)),
    "up->down":    ("go up then down", "y", (-1, +1)),
    "down->up":    ("go down then up", "y", (+1, -1)),
}


def load(path, which, self_attn):
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"
    mc.planner_cfg.resampler_query_self_attn = self_attn
    m = NitroGen(config=mc, game_mapping=None)
    miss, unexp = m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)[which], strict=False)
    # sanity: no self-attn params should be silently missing/unexpected
    sa_bad = [k for k in (list(miss) + list(unexp)) if "self_attn" in k or "ln_sa" in k]
    if sa_bad:
        print(f"  WARN self-attn param mismatch ({len(sa_bad)}): {sa_bad[:2]}")
    return m.to(device).eval()


def plan_tokens(m, text):
    h, kpm = cache.get(text if text else ".")
    d = {"plan_hidden": h.unsqueeze(0).to(device), "plan_key_padding_mask": kpm.unsqueeze(0).to(device),
         "plan_dropped": torch.tensor([False], device=device)}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pt, _ = m.compute_plan_tokens(d)
    return pt[0].float().cpu().numpy()  # (K, d)


def token_diversity(m):
    """Mean pairwise cosine among K plan tokens, averaged over SEQ plans. Lower=more
    diverse (tokens specialize); high=collapsed."""
    cos = []
    for name, (text, _, _) in SEQ.items():
        t = plan_tokens(m, text)
        tn = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)
        S = tn @ tn.T
        iu = np.triu_indices(K, 1)
        cos.append(S[iu].mean())
    return float(np.mean(cos))


def vel(m, png, text, dr, t):
    fr = np.asarray(Image.open(png).convert("RGB")); pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device) for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device); d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else "."); d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device); d["plan_dropped"] = torch.tensor([dr], dtype=torch.bool, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vis = m.encode_images(d["images"]); pt, pdp = m.compute_plan_tokens(d)
        vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, m.action_encoder(fn, torch.tensor([t], device=device), d["embodiment_id"]), d["dropped_images"], game_ids=d["game_ids"], plan_tokens=pt)
        vlm = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp); sm = m._additive_key_mask(vlm, vl.dtype)
        vl = m.vl_self_attention_model(vl, attention_mask=sm)
        mo = m.model(hidden_states=sa, encoder_hidden_states=vl, encoder_attention_mask=vlm, timestep=torch.tensor([t], device=device))
        return m.action_decoder(mo, d["embodiment_id"])[0, -H:].float().cpu().numpy()


def evaluate(tag, path, which, self_attn):
    m = load(path, which, self_attn)
    print(f"\n=== {tag} ===")
    half = H // 2; oks = 0; splits = []
    for name, (text, axis, (s1, s2)) in SEQ.items():
        ax = JLX if axis == "x" else JLY
        d1 = d2 = 0.0
        for png in pngs:
            for t in TS:
                dv = vel(m, png, text, False, t) - vel(m, png, "", True, t)
                d1 += dv[:half, ax].mean(); d2 += dv[half:, ax].mean()
        n = len(pngs) * len(TS); d1 /= n; d2 /= n
        ok = (np.sign(d1) == s1) and (np.sign(d2) == s2) and ((d2 - d1) * (s2 - s1) > 0)
        oks += ok; splits.append(abs(d2 - d1))
        print(f"  {name:12s} 1st={d1:+.3f} 2nd={d2:+.3f} split={d2-d1:+.3f}  {'OK' if ok else 'x'}")
    div = token_diversity(m)
    print(f"  ORDERING: {oks}/4 | mean|split|={np.mean(splits):.3f} | token-cos(diversity)={div:.3f}")
    del m; torch.cuda.empty_cache()


runs = [
    ("REF seq_pertoken (orig, OFF)", "runs/seq_pertoken/plan_stage1_1500.pt", "model_ema", False),
    ("SEQ attn-OFF EMA", "runs/attn_seq_off/plan_stage1_1500.pt", "model_ema", False),
    ("SEQ attn-OFF raw model", "runs/attn_seq_off/plan_stage1_1500.pt", "model", False),
    ("SEQ attn-ON  (q-former self-attn)", "runs/attn_seq_on/plan_stage1_1500.pt", "model_ema", True),
]
for tag, ckp, which, sa in runs:
    full = os.path.join(REPO, ckp)
    if os.path.exists(full):
        evaluate(tag, full, which, sa)
    else:
        print(f"(missing {ckp})")
