"""Package integration test for plan-conditioned NitroGen.

Loads the real ng.pt through the *modified package* with plan conditioning
enabled, builds a batch via the real NitrogenTokenizer (now emitting _PLAN_TOKEN
placeholders), runs forward()+backward(), and checks:
  - state_dict load: only plan_head.* is "missing" (new params), nothing else
  - flow-matching loss is finite and backward populates plan_head grads
  - null-plan vs plan produce different losses (plan channel has effect)

Run:
  cd /home/t-nagupta/NitroGen && source .venv/bin/activate && \
  python3 planner_poc/test_integration.py
"""
import os, sys
import numpy as np
import torch

REPO = "/home/t-nagupta/NitroGen"
NG_CKPT = os.path.join(REPO, "ckpts/nitrogen/ng.pt")
sys.path.insert(0, REPO)

# transformers>=5 SigLIP shim is now handled inside the package, but keep a
# belt-and-suspenders guard for older call sites.
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda self: self)

from nitrogen.flow_matching_transformer.nitrogen import NitroGen, _PLAN_TOKEN, _IMG_TOKEN
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.cfg import CkptConfig


def main():
    device = "cuda"
    K = 8
    D = 1024
    print("=" * 70)
    print("Package integration test: plan-conditioned NitroGen")
    print("=" * 70)

    # ---- load ckpt config, enable planner
    ckpt = torch.load(NG_CKPT, map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    cc.model_cfg.planner_cfg.enabled = True
    cc.model_cfg.planner_cfg.num_plan_tokens = K
    cc.model_cfg.planner_cfg.plan_hidden_size = D
    cc.model_cfg.tune_plan_head = True

    model = NitroGen(config=cc.model_cfg, game_mapping=None)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    missing_non_plan = [k for k in missing if not k.startswith("plan_head.")]
    plan_missing = [k for k in missing if k.startswith("plan_head.")]
    print(f"[load] non-plan missing = {len(missing_non_plan)} (want 0)")
    print(f"[load] plan_head missing = {len(plan_missing)} (new params, expected > 0)")
    print(f"[load] unexpected        = {len(unexpected)} (want 0)")
    assert len(missing_non_plan) == 0, missing_non_plan[:10]
    assert len(unexpected) == 0, unexpected[:10]
    assert len(plan_missing) > 0
    model.to(device).train()

    # ---- build a batch through the real tokenizer (with plan placeholders)
    tok_cfg = cc.tokenizer_cfg
    tok_cfg.training = True
    tok_cfg.num_plan_tokens = K
    # VL stream now needs room for K plan tokens + 256 image tokens.
    tok_cfg.max_sequence_length = 256 + K
    tok = NitrogenTokenizer(tok_cfg)

    H = cc.model_cfg.action_horizon
    B = 2
    examples = []
    for _ in range(B):
        ex = {
            "frames": np.random.rand(1, 3, 256, 256).astype(np.float32),
            "dropped_frames": np.zeros((1,), dtype=bool),
            "buttons": np.random.randint(0, 2, (1, H, 17)).astype(np.float32),
            "j_left": (np.random.rand(1, H, 2) * 2 - 1).astype(np.float32),
            "j_right": (np.random.rand(1, H, 2) * 2 - 1).astype(np.float32),
        }
        examples.append(tok.encode(ex))

    def stack(key):
        return torch.stack([torch.as_tensor(np.asarray(e[key])) for e in examples], dim=0)

    vl_token_ids = stack("vl_token_ids").to(device)
    sa_token_ids = stack("sa_token_ids").to(device)
    print(f"[tok] vl_token_ids shape = {tuple(vl_token_ids.shape)}; "
          f"_PLAN_TOKEN count/example = {(vl_token_ids[0]==_PLAN_TOKEN).sum().item()} (want {K})")
    print(f"[tok] _IMG_TOKEN count/example = {(vl_token_ids[0]==_IMG_TOKEN).sum().item()} (want 256)")
    assert (vl_token_ids[0] == _PLAN_TOKEN).sum().item() == K

    data = {
        "images": stack("images").to(device).float(),
        "dropped_images": stack("dropped_images").to(device),
        "vl_token_ids": vl_token_ids,
        "sa_token_ids": sa_token_ids,
        "vl_attn_mask": stack("vl_attn_mask").to(device),
        "actions": stack("actions").to(device).float(),
        "actions_mask": stack("actions_mask").to(device),
        "embodiment_id": torch.zeros(B, dtype=torch.long, device=device),
        "has_real_action": torch.ones(B, device=device),
        # plan input: precomputed VLM hidden states (random stand-in here)
        "plan_hidden": torch.randn(B, 12, D, device=device),
        "plan_key_padding_mask": torch.zeros(B, 12, dtype=torch.bool, device=device),
    }

    # ---- forward + backward (plan forced on, no dropout)
    data["plan_dropped"] = torch.zeros(B, dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(data)
    loss = out["loss"]
    print(f"[fwd] loss = {loss.item():.4f}")
    assert torch.isfinite(loss), "loss not finite"
    loss.backward()
    g_res = sum(p.grad.norm().item() for p in model.plan_head.resampler.parameters() if p.grad is not None)
    g_ada = sum(p.grad.norm().item() for p in model.plan_head.adapter.parameters() if p.grad is not None)
    print(f"[bwd] grad resampler = {g_res:.3e}, adapter = {g_ada:.3e}")
    assert g_res > 0 and g_ada > 0, "no grad into plan_head"

    # ---- plan vs null effect (eval, fixed seed for noise/time)
    model.eval()
    def loss_for(dropped, seed):
        torch.manual_seed(seed)
        d = dict(data)
        d["plan_dropped"] = torch.full((B,), bool(dropped), device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(d)["loss"].item()
    l_plan = loss_for(False, 7)
    l_null = loss_for(True, 7)
    print(f"[effect] loss(plan)={l_plan:.4f}  loss(null)={l_null:.4f}  |diff|={abs(l_plan-l_null):.4f}")
    effect_ok = abs(l_plan - l_null) > 1e-4

    ok = (len(missing_non_plan) == 0 and len(unexpected) == 0 and g_res > 0 and g_ada > 0 and effect_ok)
    print("=" * 70)
    print(f"RESULT: load={'OK' if not missing_non_plan and not unexpected else 'BAD'} | "
          f"grad={'PASS' if g_res>0 and g_ada>0 else 'FAIL'} | "
          f"plan_effect={'PASS' if effect_ok else 'FAIL'}")
    print(f"OVERALL: {'*** INTEGRATION PASS ***' if ok else '!!! FAIL !!!'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
