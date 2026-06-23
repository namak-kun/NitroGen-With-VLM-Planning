"""Does the env-free LEFT/RIGHT fix (clean-label retrain, token sep 0.688) translate to ACTION-level
steering authority? Token separation is necessary but not sufficient — the frozen DiT must actually
turn those tokens into a different stick_x.

For each model x frame, sample the chunk under opposing directional plans and measure
  sep = stick_x[right_plan] - stick_x[left_plan]   (positive => the plan steers the stick correctly).
Compare clean (the fix) vs override (env-free ceiling) vs dir (noisy-label baseline). Frames include
the real SuperTuxKart racing frame (the target steering env) + cached general-gameplay frames.

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. .venv/bin/python planner_poc/action_leftright_test.py
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from eval_policy import NitroGenPolicy

JLX = 21
QWEN = "Qwen/Qwen3.5-2B"
CKPTS = {
    "clean":    "runs/stage2_2b_clean/plan_stage1_2000.pt",
    "override": "runs/stage2_2b_override/plan_stage1_3000.pt",
    "dir":      "runs/stage2_2b_dir/plan_stage1_2500.pt",
}
WS = [4, 8, 16]
PLANS = [("move left", "move right"), ("go left", "go right")]


def load_frames():
    frames = {}
    for p in ["docs/env_candidates/stk_racing_frame.png",
              "docs/env_candidates/stk_env_controlled.png"]:
        if os.path.exists(p):
            frames[os.path.basename(p).replace(".png", "")] = np.asarray(Image.open(p).convert("RGB"))
    # a few cached general-gameplay frames the planner has seen
    extra = sorted(glob.glob("/tmp/frames_cc/*__250.png"))[:3] or sorted(glob.glob("/tmp/frames_pre/*.png"))[:3]
    for p in extra:
        frames["g_" + os.path.basename(p)[:8]] = np.asarray(Image.open(p).convert("RGB"))
    return frames


def stick_x(pol, frame, plan, w, seed=0):
    torch.manual_seed(seed)
    ch = pol._sample_chunk(frame, plan, float(w), plan_frames=[frame])
    return float(ch[:, JLX].mean())


def main():
    frames = load_frames()
    print(f"{len(frames)} frames: {list(frames)}\n")
    summary = {}
    for name, ck in CKPTS.items():
        if not os.path.exists(ck):
            print(f"[skip] {name}: missing {ck}"); continue
        pol = NitroGenPolicy(ck, qwen=QWEN, default_cfg=1.0)
        pol.mm_mode = True; pol.mm_text_only = True
        print(f"===== {name} ({ck}) =====")
        seps = []
        for fname, fr in frames.items():
            for w in WS:
                # average over the plan phrasings + a couple seeds to reduce sampler noise
                ls, rs = [], []
                for (pl, pr) in PLANS:
                    for s in (0, 1):
                        ls.append(stick_x(pol, fr, pl, w, s))
                        rs.append(stick_x(pol, fr, pr, w, s))
                xl, xr = float(np.mean(ls)), float(np.mean(rs))
                sep = xr - xl
                seps.append(sep)
                tag = "  <== steers" if sep > 0.15 else ("  L-obey" if xl < -0.1 else "")
                print(f"  {fname:18} w={w:>2}  L={xl:+.3f}  R={xr:+.3f}  sep={sep:+.3f}{tag}")
        seps = np.array(seps)
        summary[name] = (seps.mean(), (seps > 0.15).mean(), seps.max())
        print(f"  -> mean_sep={seps.mean():+.3f}  frac_steering={ (seps>0.15).mean():.2f}  max={seps.max():+.3f}\n")
        del pol; torch.cuda.empty_cache()
    print("===== SUMMARY (mean_sep, frac>0.15, max) =====")
    for n, (m, f, mx) in summary.items():
        print(f"  {n:10}: mean_sep={m:+.3f}  frac_steering={f:.2f}  max={mx:+.3f}")
    print("\n(positive mean_sep => directional plan controls stick_x; clean should beat dir if the\n"
          " label-cleaning fix gave the frozen DiT action-level left/right authority, not just tokens.)")


if __name__ == "__main__":
    main()
