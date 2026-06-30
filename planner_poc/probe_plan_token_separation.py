"""Probe: after training, do the resampler's OUTPUT plan tokens separate direction?

The raw-hidden probe (probe_hidden_separation.py) showed mean-pooled 2B hiddens do NOT separate
left/right (acc~0.51). The outcome-contrastive loss is supposed to fix that in the resampler
OUTPUT. This probe loads a trained checkpoint, runs the plan head over each uuid's cached mm-hidden
(block `cursor`, default 0 = act-now), mean-pools the K plan tokens, and measures left/right (and
4-way) linear separability + class-mean cosine. Compare to the 0.512 raw baseline.

Run: PYTHONPATH=. .venv/bin/python planner_poc/probe_plan_token_separation.py \
        runs/stage2_2b_a4/plan_stage1_1000.pt /tmp/stage2_mm_hidden_2b.pt /tmp/stage2_index_a4.pt
"""
import sys
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
HID = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stage2_mm_hidden_2b.pt"
INDEX = sys.argv[3] if len(sys.argv) > 3 else "/tmp/stage2_index_a4.pt"
CURSOR = int(sys.argv[4]) if len(sys.argv) > 4 else 0
device = "cuda"
K = 8


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    ng = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
    CC = CkptConfig.model_validate(ng["ckpt_config"])
    mc = CC.model_cfg.model_copy(deep=True)
    mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K
    mc.planner_cfg.null_mode = "masked"
    qk = "plan_head.resampler.queries"
    if qk in sd:
        mc.planner_cfg.backbone_hidden_size = int(sd[qk].shape[-1])
        mc.planner_cfg.num_chunks = int(sd[qk].shape[0]) // K
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(sd, strict=False)
    print(f"loaded {path} | backbone_dim={mc.planner_cfg.backbone_hidden_size} "
          f"num_chunks={mc.planner_cfg.num_chunks}")
    return m.to(device).eval()


def plan_tokens(m, h, mask):
    d = {"plan_hidden": h.unsqueeze(0).to(device).float(),
         "plan_key_padding_mask": mask.unsqueeze(0).to(device),
         "plan_dropped": torch.tensor([False], device=device),
         "plan_cursor": torch.tensor([CURSOR], device=device)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pt, _ = m.compute_plan_tokens(d)
    return pt[0].float().cpu()          # (K, dit_dim)


def main():
    m = load(CKPT)
    hid = torch.load(HID, map_location="cpu")
    idx = torch.load(INDEX, map_location="cpu", weights_only=False)
    Xm, Xf, y = [], [], []
    for u, e in hid.items():
        if u not in idx:
            continue
        pt = plan_tokens(m, e["h"], e["mask"])     # (K, d)
        Xm.append(pt.mean(0).numpy())              # mean-pool readout
        Xf.append(pt.reshape(-1).numpy())          # flatten readout (per-token info kept)
        y.append(idx[u]["dir"])
    Xm = np.stack(Xm); Xf = np.stack(Xf); y = np.array(y)
    from collections import Counter
    print(f"{len(y)} examples; K*d flat dim={Xf.shape[1]}; dir dist:", dict(Counter(y)))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def probe(X, classes):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        mask = np.isin(y, classes)
        if mask.sum() < 20 or min(Counter(y[mask]).values()) < 5:
            return None
        sc = cross_val_score(LogisticRegression(max_iter=3000), Xn[mask], y[mask], cv=5)
        return sc.mean(), sc.std(), int(mask.sum())

    for name, X in [("MEAN-POOL", Xm), ("FLATTEN K*d", Xf)]:
        print(f"\n=== TRAINED plan-token separation [{name}] (5-fold CV) ===")
        for pair in [["left", "right"], ["up", "down"]]:
            r = probe(X, pair)
            if r:
                print(f"  {pair[0]:>5} vs {pair[1]:<5}: acc={r[0]:.3f}±{r[1]:.3f} (n={r[2]})")
        r = probe(X, ["left", "right", "up", "down"])
        if r:
            print(f"  4-way dir   : acc={r[0]:.3f}±{r[1]:.3f} (n={r[2]}, chance=0.25)")
    print("\nBaseline (raw hidden mean): left/right acc 0.512. Want acc>>0.5.")


if __name__ == "__main__":
    main()
