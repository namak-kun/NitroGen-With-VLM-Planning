"""Build a GOLD plan-token library for env-free counterfactual control. For each direction
(left/right/up/down), pick the single TEACHER token (from a training chunk with that dominant
direction) that most strongly drives the corresponding stick output on a NEUTRAL Cave Story
frame under gold-token CFG. These tokens override NitroGen's strong rightward prior where the
Cave Story plan TEXT cannot (the text produces weak/washed plan tokens). Env-free: reuses the
existing teacher tokens, no training.

Out: /tmp/gold_tokens.pt = {dir: {"token": (1,K,d), "uuid": str, "stick_x":.., "stick_y":..}}
Run: PYTHONPATH=. python planner_poc/build_gold_tokens.py
"""
import sys
import numpy as np
import torch
import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
from PIL import Image
from eval_policy import NitroGenPolicy

JLX, JLY = 21, 22
FRAME = np.asarray(Image.open("/tmp/run_mm/f000.png").convert("RGB"))  # First Cave start (rightward prior)
CKPT = "runs/stage2_student_mm_txt_lora_cf/plan_stage1_2500.pt"
W = 16.0


def main():
    pol = NitroGenPolicy(CKPT, default_cfg=1.0); pol.mm_mode = True; pol.mm_text_only = True
    teach = torch.load("/tmp/stage2_teacher_tokens_mm.pt", map_location="cpu", weights_only=False)
    idx = torch.load("/tmp/stage2_index_mm.pt", map_location="cpu", weights_only=False)
    by_dir = {}
    for u, e in idx.items():
        if u in teach:
            by_dir.setdefault(e["dir"], []).append(u)
    # the stick component that each direction should maximize (sign): left=-x, right=+x, up=-y, down=+y
    target = {"left": (JLX, -1), "right": (JLX, +1), "up": (JLY, -1), "down": (JLY, +1)}
    gold = {}
    for d, (axis, sign) in target.items():
        best = None
        for u in by_dir.get(d, []):
            tt = torch.as_tensor(np.asarray(teach[u])).unsqueeze(0).float()
            # average a few seeds for a stable estimate
            vals = []
            for s in range(4):
                ch = pol.sample_chunk_token(FRAME, tt, W, seed=s)
                vals.append(float(ch[:, axis].mean()))
            score = sign * np.mean(vals)   # higher = stronger in the desired direction
            if best is None or score > best[0]:
                best = (score, u, tt, np.mean(vals))
        if best:
            sc, u, tt, axval = best
            gold[d] = {"token": tt, "uuid": u, "axis": axis, "sign": sign, "axval": float(axval)}
            print(f"GOLD {d:5}: uuid={u[:24]:24} axis={'x' if axis==JLX else 'y'} "
                  f"val={axval:+.3f} (sign-score {sc:+.3f})", flush=True)
    torch.save(gold, "/tmp/gold_tokens.pt")
    print(f"\nsaved {len(gold)} gold tokens -> /tmp/gold_tokens.pt")


if __name__ == "__main__":
    main()
