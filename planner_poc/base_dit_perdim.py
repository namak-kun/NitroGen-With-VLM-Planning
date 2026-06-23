"""PER-DIMENSION base-DiT fidelity: for ALL 25 action dims, how close is the FROZEN DiT's
unconditioned (null, plan-dropped) prediction to the streamer's real action o, given the frame?

TRUE layout (mm_tokenizers.py:237, verified vs ng.pt):
  dims 0:21  = buttons (BUTTON_ACTION_TOKENS order), real {0,1}, model output continuous in [0,1]
  dims 21:23 = j_left  (x,y), 23:25 = j_right (x,y); real/[-1,1] -> eval-space (v+1)/2 in [0,1]

For each dim report over N chunks (chunk = 18-step window, averaged over H):
  buttons: real on-rate, pred mean, AUC (rank pred vs real-on), pred@on vs pred@off (separation)
  sticks : real mean, pred mean, MSE (in [-1,1] units), corr(real,pred), spread (null-vs-null noise)

Run: PYTHONPATH=. .venv/bin/python planner_poc/base_dit_perdim.py [N]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from PIL import Image
from nitrogen.shared import BUTTON_ACTION_TOKENS

LOOKUP = "/tmp/stage2_plan_lookup_mm_a4.json"
INDEX = "/tmp/stage2_index_a4.pt"
FRAMES_CC = "/tmp/frames_cc"
CKPT = "runs/stage2_2b_a1b48/plan_stage1_1000.pt"   # weights from ng.pt; null path = base DiT
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def pm1(v):
    return 2.0 * v - 1.0


def main():
    lookup = json.load(open(LOOKUP))
    idx = torch.load(INDEX, map_location="cpu", weights_only=False)
    uu = [u for u, e in idx.items()
          if u in lookup and lookup[u].get("frame_offsets")
          and os.path.exists(os.path.join(FRAMES_CC, f"{u}__{lookup[u]['frame_offsets'][0]}.png"))]
    import random; random.seed(0); random.shuffle(uu)
    uu = uu[:N]
    print(f"per-dim base-DiT fidelity on {len(uu)} chunks "
          f"({len(set(u.split('_chunk_')[0] for u in uu))} videos) | weights=ng.pt (unconditioned)")

    from eval_policy import NitroGenPolicy
    pol = NitroGenPolicy(CKPT, qwen="Qwen/Qwen3.5-2B", default_cfg=1.0)
    pol.mm_mode = False

    REAL = []   # (N, 25) real action (chunk-mean), eval-space [0,1]
    PRED = []   # (N, 25) base-DiT pred (chunk-mean)
    PRED2 = []  # second null sample (noise floor)
    for u in uu:
        e = idx[u]
        b = np.asarray(e["buttons"], dtype=np.float32)       # (H,21) in {0,1}
        jl = (np.asarray(e["j_left"], dtype=np.float32) + 1) / 2.0   # (H,2) -> [0,1]
        jr = (np.asarray(e["j_right"], dtype=np.float32) + 1) / 2.0
        real = np.concatenate([b, jl, jr], axis=-1).mean(0)  # (25,)
        f0 = lookup[u]["frame_offsets"][0]
        frame = np.asarray(Image.open(os.path.join(FRAMES_CC, f"{u}__{f0}.png")).convert("RGB"))
        torch.manual_seed(0); a1 = pol._sample_chunk(frame, "", 1.0, null=True).mean(0)
        torch.manual_seed(1); a2 = pol._sample_chunk(frame, "", 1.0, null=True).mean(0)
        REAL.append(real); PRED.append(a1); PRED2.append(a2)
    REAL = np.stack(REAL); PRED = np.stack(PRED); PRED2 = np.stack(PRED2)

    def auc(scores, labels):
        labels = labels.astype(bool)
        p, n = labels.sum(), (~labels).sum()
        if p == 0 or n == 0:
            return float("nan")
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
        return (ranks[labels].sum() - p * (p + 1) / 2) / (p * n)

    names = list(BUTTON_ACTION_TOKENS) + ["jL_x", "jL_y", "jR_x", "jR_y"]
    print("\n=== BUTTONS (dims 0:21) ===")
    print(f"  {'dim':>2} {'name':14} {'real_on%':>8} {'pred@on':>8} {'pred@off':>8} {'sep':>6} {'AUC':>6}")
    for j in range(21):
        on = REAL[:, j] > 0.05
        ron = 100.0 * on.mean()
        if on.sum() == 0:
            print(f"  {j:>2} {names[j]:14} {ron:7.1f}%   (never pressed)")
            continue
        pon = PRED[on, j].mean(); poff = PRED[~on, j].mean() if (~on).any() else float('nan')
        print(f"  {j:>2} {names[j]:14} {ron:7.1f}% {pon:8.3f} {poff:8.3f} {pon-poff:+6.3f} {auc(PRED[:,j],on):6.3f}")

    print("\n=== STICKS (dims 21:25, reported in [-1,1]) ===")
    print(f"  {'dim':>2} {'name':6} {'real_mean':>9} {'pred_mean':>9} {'MSE':>7} {'corr':>6} {'noise':>6}")
    for j in range(21, 25):
        r = pm1(REAL[:, j]); p = pm1(PRED[:, j]); p2 = pm1(PRED2[:, j])
        mse = float(np.mean((r - p) ** 2))
        corr = float(np.corrcoef(r, p)[0, 1]) if r.std() > 1e-6 and p.std() > 1e-6 else float('nan')
        noise = float(np.mean(np.abs(p - p2)))
        print(f"  {j:>2} {names[j]:6} {r.mean():+9.3f} {p.mean():+9.3f} {mse:7.3f} {corr:6.3f} {noise:6.3f}")

    print("\nREAD: button 'sep'/AUC>0.5 => DiT raises that button when the streamer did (fidelity).")
    print("      stick corr>0 + low MSE vs noise => DiT tracks the stick; corr~0 => hedges/ignores.")


if __name__ == "__main__":
    main()
