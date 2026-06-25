"""Demo: VLM objective-labeling on real YouTube gameplay frames (System-2 free-label proof).

Loads the FROZEN planner VLM and, on a window of consecutive farmed frames, generates a short grounded
objective -- the label the user proposed farming. Popular games being in the VLM's pretraining is the
ASSET here: the objectives come out correct for free.

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B \
     .venv/bin/python planner_poc/objective_label_demo.py --frames "docs/yt_farm/<id>/frames/*.png"
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from nitrogen.planner import PlanEncoder, PlannerConfig

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")

# the grounded-but-freeform objective prompt (won the earlier prompt sweep): one concrete sentence.
SYS = ("You are an expert game-playing assistant. You are shown a few recent frames of a video game, "
       "oldest to newest. Describe what the player should do next.")
INSTR = ("In one sentence, say what the player should do next and briefly why, naming the concrete "
         "direction or action (e.g. left, right, up, down, jump, shoot) rather than vague words like "
         "'forward'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="glob of farmed frames (oldest->newest by name)")
    ap.add_argument("--window", type=int, default=4, help="frames per objective")
    ap.add_argument("--stride", type=int, default=10, help="advance between objectives (in frames)")
    ap.add_argument("--n", type=int, default=6, help="how many objectives to emit")
    args = ap.parse_args()

    fs = sorted(glob.glob(args.frames))
    if not fs:
        raise SystemExit(f"no frames matched {args.frames}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load()
    if next(pl.backbone.parameters()).device != torch.device(dev):
        pl.backbone.to(dev)

    print(f"loaded {QWEN}; {len(fs)} frames; labeling {args.n} windows\n")
    emitted = 0
    i = 0
    while emitted < args.n and i + args.window <= len(fs):
        window = [np.asarray(Image.open(f).convert("RGB")) for f in fs[i:i + args.window]]
        plan = pl.generate_plan(window, dev, instruction=INSTR, system=SYS, max_new_tokens=40)
        # first sentence only
        plan = plan.replace("\n", " ").strip()
        if "." in plan:
            plan = plan[:plan.index(".") + 1]
        print(f"[{os.path.basename(fs[i])}..{os.path.basename(fs[i+args.window-1])}]  -> {plan}")
        emitted += 1
        i += args.stride


if __name__ == "__main__":
    main()
