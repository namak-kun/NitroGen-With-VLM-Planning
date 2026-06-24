"""Fast, env-free balance metric for a plan-conditioned checkpoint: on a set of saved frames, sample
the chunk under each directional plan and measure the stick separation per AXIS. Reports left/right and
up/down magnitudes + a BALANCE score (how symmetric the two directions of an axis are). No env boot, so
it's a quick sweep metric (~1-2 min/ckpt) that DOES capture the left-lean (unlike token-separation).

stick_x[JLX] for left/right; stick_y[JLY] for up/down. The model outputs [0,1], 0.5=neutral.
  lr_sep = mean(stick_x[right]) - mean(stick_x[left])      (should be > 0)
  lr_balance = min(|right-0.5|, |0.5-left|) / max(...)      in [0,1], 1=perfectly symmetric
similarly for up/down on JLY (up = value<0.5, down = value>0.5).

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B \
        .venv/bin/python planner_poc/eval_balance.py <ckpt> [<ckpt> ...] [--cfg 8]
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

JLX, JLY = 21, 22
QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")


def load_frames():
    pats = ["docs/env_candidates/stk_*.png", "docs/poc/stk_lora600_lr/poc_*.png"]
    files = [f for p in pats for f in sorted(glob.glob(p))][:6]
    return [np.asarray(Image.open(f).convert("RGB")) for f in files]


def axis_stats(pol, frames, dim, neg_plan, pos_plan, cfg):
    neg, pos = [], []
    for fr in frames:
        for seed in (0, 1):
            torch.manual_seed(seed)
            neg.append(float(pol._sample_chunk(fr, neg_plan, cfg, plan_frames=[fr])[:, dim].mean()))
            torch.manual_seed(seed)
            pos.append(float(pol._sample_chunk(fr, pos_plan, cfg, plan_frames=[fr])[:, dim].mean()))
    n, p = float(np.mean(neg)), float(np.mean(pos))
    sep = p - n                                   # >0 = correct ordering
    neg_mag, pos_mag = (0.5 - n), (p - 0.5)       # how far each side pushes from neutral
    bal = (min(neg_mag, pos_mag) / max(neg_mag, pos_mag)) if max(neg_mag, pos_mag) > 1e-6 else 0.0
    return n, p, sep, neg_mag, pos_mag, bal


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg = 8.0
    if "--cfg" in sys.argv:
        cfg = float(sys.argv[sys.argv.index("--cfg") + 1])
    frames = load_frames()
    print(f"{len(frames)} frames | cfg={cfg} | metric: per-axis sep + balance (1=symmetric)\n")
    print(f"{'ckpt':42} {'LR sep':>7} {'L':>5} {'R':>5} {'LRbal':>6}  {'UD sep':>7} {'U':>5} {'D':>5} {'UDbal':>6}")
    for ck in args:
        if not os.path.exists(ck):
            print(f"{ck}: MISSING"); continue
        pol = NitroGenPolicy(ck, qwen=QWEN, default_cfg=cfg)
        pol.mm_mode = True; pol.mm_text_only = True
        ln, lp, lsep, lnm, lpm, lbal = axis_stats(pol, frames, JLX, "go left", "go right", cfg)
        un, up, usep, unm, upm, ubal = axis_stats(pol, frames, JLY, "go up", "go down", cfg)
        tag = ck.split("/")[-2] + "/" + ck.split("/")[-1].replace("plan_stage1_", "s")
        print(f"{tag:42} {lsep:+7.3f} {ln:5.2f} {lp:5.2f} {lbal:6.2f}  "
              f"{usep:+7.3f} {un:5.2f} {up:5.2f} {ubal:6.2f}", flush=True)
        del pol; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
