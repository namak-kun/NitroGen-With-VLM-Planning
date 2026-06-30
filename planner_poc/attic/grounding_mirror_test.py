"""Mirror grounding test: is the System-2 VLM actually READING steering direction from the frame, or
just biased? Ask "which way to steer?" on each frame AND its horizontal mirror. A GROUNDED model must
flip its answer (right↔left) on the mirror; a BIASED model answers the same for both.

RESULT (2026-06-23, Qwen3.5-2B planner backbone): 0/8 frames flipped — the VLM answered "right" for
every frame and its mirror => the directional plans it emits are a BIAS, NOT grounded perception. This
confirms the documented "VLM can't read fine spatial control from pixels": the steering capability is
in System-1 + the plan TOKEN, and grounded System-2 plan GENERATION for steering is the open gap.

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B .venv/bin/python \
        planner_poc/grounding_mirror_test.py [glob ...]
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
from nitrogen.planner import PlanEncoder, PlannerConfig

Q = ("You are driving this kart. Which way should you steer to stay on the track? "
     "Answer 'left' or 'right'.")


def main():
    patterns = sys.argv[1:] or ["docs/env_candidates/stk_*.png", "docs/poc/stk_lora600_lr/*.png"]
    qwen = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=qwen)); pl.load()
    files = [f for p in patterns for f in sorted(glob.glob(p))]
    print("=== GROUNDING TEST: frame vs horizontal mirror (grounded => answers flip) ===")
    flips = n = 0
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB"))
        mir = img[:, ::-1, :].copy()
        a = pl.generate_plan([img], "cuda", instruction=Q, max_new_tokens=8).lower()
        b = pl.generate_plan([mir], "cuda", instruction=Q, max_new_tokens=8).lower()
        da = "left" if "left" in a else ("right" if "right" in a else "?")
        db = "left" if "left" in b else ("right" if "right" in b else "?")
        flipped = {da, db} == {"left", "right"}
        flips += flipped; n += 1
        print(f"  {os.path.basename(f)[:24]:24} orig={da:5} mirror={db:5} "
              f"{'FLIP(grounded)' if flipped else 'same(biased)'}")
    if n:
        verdict = "GROUNDED (reads the layout)" if flips >= n * 0.6 else "BIASED (not reading the frame)"
        print(f"\n  flipped {flips}/{n} -> {verdict}")


if __name__ == "__main__":
    main()
