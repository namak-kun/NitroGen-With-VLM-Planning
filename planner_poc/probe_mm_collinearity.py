"""EXP-050 diagnostic: does FRAME-CONDITIONING de-collapse the plan tokens?

The EXP-042/049 wall was representational: text-only tactical plans collapse to near-collinear
plan tokens (mean pairwise cosine ~1.0), so the DiT can't route on content. This probe measures
the plan-token geometry for a FRAME-CONDITIONED student (tokens from the cached [before,after]+
text hidden, /tmp/stage2_mm_hidden.pt) and compares it to a TEXT-ONLY student (tokens from the
text plan hidden). Lower mean pairwise cosine = more de-collapsed = more content/scene-specific
plan tokens = the substrate for selective steering.

Usage: PYTHONPATH=. .venv/bin/python planner_poc/probe_mm_collinearity.py \
           <mm_ckpt> [text_ckpt]
"""
import json
import os
import sys

import numpy as np
import torch

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache

device = "cuda"; K = 8
MM_CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student_mm/plan_stage1_2500.pt"
TEXT_CKPT = sys.argv[2] if len(sys.argv) > 2 else "runs/stage2_student/plan_stage1_2500.pt"
LOOKUP = json.load(open("/tmp/stage2_plan_lookup_mm.json"))
MM_HIDDEN = torch.load("/tmp/stage2_mm_hidden.pt", map_location="cpu")
ck = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=f"{REPO}/ckpts/qwen35-0.8b")); pl.load()
text_cache = PlanHiddenCache(pl, device)


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


def tokens_from_hidden(m, h, kpm):
    d = {"plan_hidden": h.unsqueeze(0).to(device),
         "plan_key_padding_mask": kpm.unsqueeze(0).to(device),
         "plan_dropped": torch.tensor([False], device=device)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pt, _ = m.compute_plan_tokens(d)
    return pt[0].float().cpu()  # (K, d)


def collinearity(tok_list):
    """tok_list: list of (K,d). Returns mean pairwise cosine of MEAN-pooled tokens and
    mean per-token (position-wise) pairwise cosine."""
    P = torch.stack([t.mean(0) for t in tok_list])         # (N, d) pooled
    P = torch.nn.functional.normalize(P, dim=-1)
    C = P @ P.t()
    n = P.shape[0]
    off = (C.sum() - n) / (n * (n - 1))
    # per-token
    K_ = tok_list[0].shape[0]
    pertok = []
    for k in range(K_):
        Q = torch.stack([t[k] for t in tok_list])
        Q = torch.nn.functional.normalize(Q, dim=-1)
        Ck = Q @ Q.t()
        pertok.append(((Ck.sum() - n) / (n * (n - 1))).item())
    return off.item(), float(np.mean(pertok))


def main():
    uuids = [u for u in LOOKUP if LOOKUP[u].get("is_gameplay") and u in MM_HIDDEN]
    print(f"probing {len(uuids)} gameplay chunks\n")

    # MM student: frame-conditioned tokens
    mm = load(MM_CKPT)
    mm_tok = [tokens_from_hidden(mm, MM_HIDDEN[u]["h"].float(), MM_HIDDEN[u]["mask"]) for u in uuids]
    mm_pool, mm_pertok = collinearity(mm_tok)
    del mm; torch.cuda.empty_cache()

    # Text student: text-only tokens (same plan strings)
    out = {"mm": (mm_pool, mm_pertok)}
    if os.path.exists(TEXT_CKPT):
        tm = load(TEXT_CKPT)
        tx_tok = []
        for u in uuids:
            h, kpm = text_cache.get(LOOKUP[u]["plan"])
            tx_tok.append(tokens_from_hidden(tm, h, kpm))
        tx_pool, tx_pertok = collinearity(tx_tok)
        out["text"] = (tx_pool, tx_pertok)

    print("plan-token COLLINEARITY (mean pairwise cosine; lower = more de-collapsed/specific):")
    print(f"  {'model':22} {'pooled':>8} {'per-token':>10}")
    print(f"  {'MM (frame+text)':22} {out['mm'][0]:>8.3f} {out['mm'][1]:>10.3f}")
    if "text" in out:
        print(f"  {'TEXT-only':22} {out['text'][0]:>8.3f} {out['text'][1]:>10.3f}")
        d = out["text"][0] - out["mm"][0]
        print(f"\n  => frame-conditioning changes pooled collinearity by {-d:+.3f} "
              f"({'MORE specific' if d > 0 else 'LESS specific'})")


if __name__ == "__main__":
    main()
