"""Override-capability token separation: using the CF cache (frame_i + opposite-dir plan_j),
do the trained student's plan tokens separate by the INTENDED cf_dir? This is the correct
left/right capability test (the factual cache has no left/right info in its input, so probing
it measures the input, not the model). No gray frame, no flow sampling -> clean + fast.

Run: PYTHONPATH=. .venv/bin/python planner_poc/probe_cf_separation.py runs/<ckpt>.pt
"""
import os
import sys
from collections import Counter

import numpy as np
import torch

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig

CKPT = sys.argv[1]
CF = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stage2_mm_cf_2b.pt"
CURSOR = int(os.environ.get("CURSOR", "0"))
device = "cuda"; K = 8


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    ng = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
    CC = CkptConfig.model_validate(ng["ckpt_config"])
    mc = CC.model_cfg.model_copy(deep=True)
    mc.planner_cfg.enabled = True; mc.planner_cfg.num_plan_tokens = K
    mc.planner_cfg.null_mode = "masked"
    qk = "plan_head.resampler.queries"
    if qk in sd:
        mc.planner_cfg.backbone_hidden_size = int(sd[qk].shape[-1])
        mc.planner_cfg.num_chunks = int(sd[qk].shape[0]) // K
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(sd, strict=False)
    print(f"loaded {path} | bdim={mc.planner_cfg.backbone_hidden_size} A={mc.planner_cfg.num_chunks}")
    return m.to(device).eval()


def main():
    m = load(CKPT)
    cf = torch.load(CF, map_location="cpu", weights_only=False)
    X, y = [], []
    for u, e in cf.items():
        h = torch.as_tensor(np.asarray(e["h"])).float().unsqueeze(0).to(device)
        kpm = torch.as_tensor(np.asarray(e["mask"])).bool().unsqueeze(0).to(device)
        d = {"plan_hidden": h, "plan_key_padding_mask": kpm,
             "plan_dropped": torch.tensor([False], device=device),
             "plan_cursor": torch.tensor([CURSOR], device=device)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pt, _ = m.compute_plan_tokens(d)
        X.append(pt[0].float().mean(0).cpu().numpy()); y.append(e["cf_dir"])
    X = np.stack(X); y = np.array(y)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    print(f"{len(y)} cf-plan tokens; cf_dir dist:", dict(Counter(y)))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def probe(cls):
        mask = np.isin(y, cls)
        if min(Counter(y[mask]).values()) < 5:
            return None
        return cross_val_score(LogisticRegression(max_iter=3000), X[mask], y[mask], cv=5).mean(), int(mask.sum())

    print("\n=== CF-plan token separation by intended cf_dir (override capability) ===")
    for pair in [["left", "right"], ["up", "down"]]:
        r = probe(pair)
        print(f"  {pair[0]:>5} vs {pair[1]:<5}: " + (f"acc={r[0]:.3f} (n={r[1]})" if r else "n/a"))
    r = probe(["left", "right", "up", "down"])
    print(f"  4-way        : " + (f"acc={r[0]:.3f} (n={r[1]}, chance .25)" if r else "n/a"))
    print("\nThis is the CORRECT left/right capability test: input is a directional plan, so high\n"
          "acc => the model maps directional plans -> separated tokens (override substrate works).")


if __name__ == "__main__":
    main()
