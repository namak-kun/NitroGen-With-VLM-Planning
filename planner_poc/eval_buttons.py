"""Env-free BUTTON-steering metric for a plan-conditioned checkpoint: on saved frames, sample the
chunk under each terse BUTTON plan (jump/accelerate/attack/brake/dash) + null, and measure the mean
press probability of each controller button over the chunk. A button plan "works" if it raises its
OWN button's probability above null AND wins the argmax (selectivity) over the other button plans.

NitroGen action layout: buttons [0:21], j_left [21:23], j_right [23:25]. P(button b) = mean over the
18 chunk steps of action[:, b]. The model outputs [0,1]; null/unconditioned ~ the frame's prior.

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B \
        .venv/bin/python planner_poc/eval_buttons.py <ckpt> [<ckpt> ...] [--cfg 8]
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from eval_policy import NitroGenPolicy

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")
# target controller button index -> terse plan (matches cache_mm_button_cf.BUTTON_PLANS[*][0])
BUTTONS = {"accelerate": 16, "jump": 18, "attack": 20, "brake": 9, "dash": 5}


def load_frames():
    pats = ["docs/env_candidates/stk_*.png", "docs/poc/stk_lora600_lr/poc_*.png",
            "docs/cavestory_play/*.png"]
    files = [f for p in pats for f in sorted(glob.glob(p))][:6]
    return [np.asarray(Image.open(f).convert("RGB")) for f in files]


def button_prob(pol, frames, plan, dim, cfg):
    vals = []
    for fr in frames:
        for seed in (0, 1):
            torch.manual_seed(seed)
            ch = pol._sample_chunk(fr, plan, cfg, plan_frames=[fr])
            vals.append(float(ch[:, dim].mean()))
    return float(np.mean(vals))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg = 8.0
    if "--cfg" in sys.argv:
        cfg = float(sys.argv[sys.argv.index("--cfg") + 1])
    frames = load_frames()
    names = list(BUTTONS)
    print(f"{len(frames)} frames | cfg={cfg} | P(button) under each plan vs null (raise own + win argmax)\n")
    for ck in args:
        if not os.path.exists(ck):
            print(f"{ck}: MISSING"); continue
        pol = NitroGenPolicy(ck, qwen=QWEN, default_cfg=cfg)
        pol.mm_mode = True; pol.mm_text_only = True
        null = {n: button_prob(pol, frames, "", BUTTONS[n], cfg) for n in names}
        # matrix[plan][button] = P(button | plan)
        mat = {p: {b: button_prob(pol, frames, p, BUTTONS[b], cfg) for b in names} for p in names}
        print(f"=== {ck} ===")
        print("  null P(btn):  " + "  ".join(f"{n}={null[n]:.2f}" for n in names))
        hdr = "plan\\btn".ljust(12) + "".join(f"{b[:5]:>7}" for b in names)
        print("  " + hdr)
        own_win, own_raise = 0, 0
        for p in names:
            row = mat[p]
            best = max(names, key=lambda b: row[b] - null[b])   # which button this plan boosts most
            own_win += (best == p)
            own_raise += (row[p] > null[p] + 0.02)
            cells = "".join(f"{row[b]:>7.2f}" for b in names)
            mark = " <-own-wins" if best == p else f" (boosts {best})"
            print(f"  {p.ljust(10)}{cells}{mark}")
        print(f"  -> own-button raised above null: {own_raise}/{len(names)} | "
              f"own-button wins argmax-delta: {own_win}/{len(names)}\n", flush=True)
        del pol; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
