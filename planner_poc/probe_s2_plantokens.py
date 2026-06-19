"""Diagnose the Stage-2 alignment gap (EXP-042 follow-up): own-plan == swapped-plan in
eval_stage2 => the DiT is blind to plan CONTENT. Two candidate mechanisms:
  (A) the 225 distinct semantic plans collapse to near-collinear PLAN TOKENS (the EXP-015
      universal-collinearity problem, now at the tactical-plan level), so the DiT literally
      cannot route on content; OR
  (B) the tokens are separable but the (LoRA) DiT learned to ignore content.
This probe measures the plan-token geometry: mean pairwise cosine of plan tokens across the
real Stage-2 plan library, vs token-level (per-plan-token) cosine, vs the raw VLM last-hidden
cosine. If plan-token cos ~1.0 the failure is representational (-> contrastive de-collinearize,
NOT capacity).
"""
import json
import sys
import numpy as np
import torch
REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO)
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache

device = "cuda"; K = 8
ck = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=f"{REPO}/ckpts/qwen35-0.8b")); pl.load()
cache = PlanHiddenCache(pl, device)
LOOKUP = json.load(open("/tmp/stage2_plan_lookup.json"))


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"
    lk = [k for k in sd if k.endswith(".lora_A")]
    if lk:
        mc.lora_dit_rank = int(sd[lk[0]].shape[0])
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()


def plan_tokens(m, text):
    h, kpm = cache.get(text if text else ".")
    d = {"plan_hidden": h.unsqueeze(0).to(device),
         "plan_key_padding_mask": kpm.unsqueeze(0).to(device),
         "plan_dropped": torch.tensor([False], device=device)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pt, _ = m.compute_plan_tokens(d)  # (1, K, dim)
    return pt[0].float().cpu()


def mean_pairwise_cos(mat):
    x = torch.nn.functional.normalize(mat, dim=-1)
    s = x @ x.t()
    n = s.shape[0]
    return (s.sum() - n) / (n * (n - 1))


def main(ckpt):
    m = load(ckpt)
    plans = [v["plan"] for v in list(LOOKUP.values())[:60]]
    # (1) raw VLM last-hidden mean (what the resampler reads from)
    raw = []
    for p in plans:
        h, kpm = cache.get(p)
        mask = (~kpm).float().unsqueeze(-1)
        raw.append((h * mask).sum(0) / mask.sum().clamp(min=1))
    raw = torch.stack(raw).float()
    # (2) resampled plan tokens: mean-pooled across K (what a content-blind DiT would see if
    #     it pooled) and (3) flattened K*dim (full token set the cross-attn actually sees)
    toks = torch.stack([plan_tokens(m, p) for p in plans])  # (N, K, dim)
    pooled = toks.mean(1)            # (N, dim)
    flat = toks.reshape(len(plans), -1)  # (N, K*dim)
    print(f"plans: {len(plans)} distinct tactical plans (Stage-2 library)\n")
    print(f"mean pairwise cosine (1.0 = collinear = content-blind):")
    print(f"  raw VLM last-hidden (mean-pool) : {mean_pairwise_cos(raw):.4f}")
    print(f"  plan tokens, mean-pooled over K : {mean_pairwise_cos(pooled):.4f}")
    print(f"  plan tokens, flattened K*dim    : {mean_pairwise_cos(flat):.4f}")
    # per-position: are individual plan-token slots content-discriminative?
    perpos = torch.stack([mean_pairwise_cos(toks[:, k, :]) for k in range(K)])
    print(f"  plan tokens, per-position (K={K}) : min {perpos.min():.4f} "
          f"mean {perpos.mean():.4f} max {perpos.max():.4f}")
    print("\nINTERPRETATION: cos near 1.0 => distinct plans map to near-identical tokens =>")
    print("the DiT cannot route on content (own==swapped). Fix = contrastive de-collinearize")
    print("(structure on plan tokens), NOT DiT capacity (LoRA didn't help: EXP-042).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_lora/plan_stage1_2500.pt")
