"""ANSWERS THE USER'S QUESTION: given frame f, does the FROZEN base DiT (no planner) reproduce
the streamer's canonical action o? => "how factual are the factual plans" — if base DiT already
does o from f alone, a plan that just describes o is redundant.

CORRECT action layout (mm_tokenizers.py, verified vs ng.pt): [buttons 0:21, j_left 21:23, j_right 23:25].
Sticks at 21:25 (NOT 0:4). Output in [0,1] (0.5=neutral) -> convert to [-1,1] via 2v-1.

Base DiT only: null=True (plan dropped/masked) -> pure unconditioned velocity. mm_mode off (no 2B
needed). Gated to NON-IDLE, clearly-directional chunks across diverse videos. Reports:
  * stick-x: real vs base-DiT pred (mean per left/right group) + sign-match rate
  * headroom: spread between two independent null samples (is p(a|f) multimodal?)
  * stick MSE(real, pred) in [-1,1] units, vs spread floor.

Run: PYTHONPATH=. .venv/bin/python planner_poc/base_dit_factual.py [N]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from PIL import Image

LOOKUP = "/tmp/stage2_plan_lookup_mm_a4.json"
INDEX = "/tmp/stage2_index_a4.pt"
FRAMES_CC = "/tmp/frames_cc"
CKPT = "runs/stage2_2b_a1b48/plan_stage1_1000.pt"   # any ckpt; null path = base DiT (ng.pt weights)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
JLX, JLY = 21, 22   # stick dims in the TRUE layout


def to_pm1(v):  # [0,1] -> [-1,1]
    return 2.0 * v - 1.0


def main():
    lookup = json.load(open(LOOKUP))
    idx = torch.load(INDEX, map_location="cpu", weights_only=False)

    # gather clearly-directional chunks, spread across videos
    by_vid = defaultdict(list)
    cand = []
    for u, e in idx.items():
        if e["dir"] not in ("left", "right", "up", "down"):
            continue
        if u not in lookup or not lookup[u].get("frame_offsets"):
            continue
        f0 = lookup[u]["frame_offsets"][0]
        if not os.path.exists(os.path.join(FRAMES_CC, f"{u}__{f0}.png")):
            continue
        x = float(np.asarray(e["j_left"])[:, 0].mean())
        if abs(x) < 0.25 and e["dir"] in ("left", "right"):
            continue
        cand.append(u)
    import random; random.seed(0); random.shuffle(cand)
    cand = cand[:N]
    print(f"base-DiT factual check on {len(cand)} directional chunks "
          f"across {len(set(u.split('_chunk_')[0] for u in cand))} videos | ckpt(weights)=ng.pt")

    from eval_policy import NitroGenPolicy
    pol = NitroGenPolicy(CKPT, qwen="Qwen/Qwen3.5-2B", default_cfg=1.0)
    pol.mm_mode = False   # base DiT needs only the frame; null masks the plan

    grp = defaultdict(lambda: {"real": [], "pred": [], "mse": [], "spread": []})
    sign_ok = sign_tot = 0
    for u in cand:
        e = idx[u]; d = e["dir"]
        f0 = lookup[u]["frame_offsets"][0]
        frame = np.asarray(Image.open(os.path.join(FRAMES_CC, f"{u}__{f0}.png")).convert("RGB"))
        torch.manual_seed(0)
        a = pol._sample_chunk(frame, "", 1.0, null=True)        # (H,25)
        torch.manual_seed(1)
        a2 = pol._sample_chunk(frame, "", 1.0, null=True)
        pred_x = to_pm1(float(a[:, JLX].mean()))
        spread_x = abs(to_pm1(float(a2[:, JLX].mean())) - pred_x)
        real_x = float(np.asarray(e["j_left"])[:, 0].mean())
        grp[d]["real"].append(real_x); grp[d]["pred"].append(pred_x)
        grp[d]["mse"].append((real_x - pred_x) ** 2); grp[d]["spread"].append(spread_x)
        if d in ("left", "right") and abs(real_x) > 0.25:
            sign_tot += 1
            sign_ok += int(np.sign(pred_x) == np.sign(real_x))

    print("\n  dir   | real_x  pred_x | stickMSE  spread(noise) | n")
    for d in ("left", "right", "up", "down"):
        g = grp[d]
        if not g["real"]:
            continue
        print(f"  {d:>5} | {np.mean(g['real']):+.3f}  {np.mean(g['pred']):+.3f} | "
              f"{np.mean(g['mse']):.3f}     {np.mean(g['spread']):.3f}        | {len(g['real'])}")
    print(f"\n  LEFT/RIGHT sign-match (base DiT pred vs real, |real|>0.25): {sign_ok}/{sign_tot} "
          f"= {sign_ok/max(sign_tot,1):.2f} (chance .5)")
    print("\n  READ: if pred_x tracks real_x (left negative, right positive) and stickMSE << real\n"
          "  variance, base DiT ALREADY does the action from the frame => factual plan is redundant\n"
          "  there. If pred_x ~0 for both and spread small => DiT hedges neutral, frame under-determines\n"
          "  direction => the plan has real work to do.")


if __name__ == "__main__":
    main()
