"""Is the left/right signal even PRESENT in the inputs we feed? (harness sanity, not model quality)

Three independent checks on the SAME chunks the probe used:
  (A) TEXT-level: do the FACTUAL plan texts for left vs right chunks differ at all? TF-IDF +
      logistic regression on plan strings. If ~chance, the plan TEXT carries no left/right -> the
      plan-token probe MUST fail on left/right (expected, not a bug).
  (B) FRAME-level (base DiT): run the FROZEN NitroGen UNCONDITIONED (null plan) on the context
      frame of strong-left vs strong-right chunks. Does predicted stick-x go the real way? Tests
      whether (a) the base DiT can produce left/right at all and (b) a single frame even determines
      direction. Reports raw + clamped stick-x so we also catch the earlier scaling artifact.
  (C) sanity: real stick-x per group (must be clearly negative for left, positive for right).

Run: PYTHONPATH=. .venv/bin/python planner_poc/verify_leftright_input.py
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
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir

LOOKUP = "/tmp/stage2_plan_lookup_mm_a4.json"
INDEX = "/tmp/stage2_index_a4.pt"
FRAMES_CC = "/tmp/frames_cc"
ROOTS = ["/tmp/stage1_big", "/tmp/stage1_more"]
H, STRIDE = 18, 2


def main():
    lookup = json.load(open(LOOKUP))
    idx = torch.load(INDEX, map_location="cpu", weights_only=False)

    # ---------- (A) TEXT-level separability of factual plans ----------
    texts, labels = [], []
    for u, e in idx.items():
        if e["dir"] in ("left", "right") and u in lookup and lookup[u].get("plan"):
            texts.append(lookup[u]["plan"]); labels.append(e["dir"])
    labels = np.array(labels)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    Xt = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(texts)
    acc = cross_val_score(LogisticRegression(max_iter=2000), Xt, labels, cv=5).mean()
    print(f"(A) TEXT-level left/right separability of FACTUAL plans: acc={acc:.3f} "
          f"(n={len(labels)}, chance .5)")
    # do ANY plans contain the literal words?
    nL = sum(1 for t, l in zip(texts, labels) if "left" in t.lower())
    nR = sum(1 for t, l in zip(texts, labels) if "right" in t.lower())
    print(f"    plans containing 'left': {nL}, 'right': {nR} (of {len(texts)})")
    print(f"    => if acc~chance and few literal mentions, the PLAN TEXT has no left/right signal.")

    # ---------- build chunk-dir lookup ----------
    dir_by_uuid = {}
    for r in ROOTS:
        for md in glob.glob(f"{r}/**/metadata.json", recursive=True):
            try:
                dir_by_uuid[json.load(open(md))["uuid"]] = os.path.dirname(md)
            except Exception:
                pass

    # ---------- pick STRONG left / right chunks ----------
    strong = {"left": [], "right": []}
    for u, e in idx.items():
        if e["dir"] not in ("left", "right") or u not in dir_by_uuid or u not in lookup:
            continue
        x = float(np.asarray(e["j_left"])[:, 0].mean())
        if abs(x) < 0.3:
            continue
        fo = lookup[u].get("frame_offsets")
        if not fo or not os.path.exists(os.path.join(FRAMES_CC, f"{u}__{fo[0]}.png")):
            continue
        strong[e["dir"]].append((u, x))
    for k in strong:
        strong[k] = strong[k][:25]
    print(f"\n(C) strong chunks: left={len(strong['left'])}, right={len(strong['right'])}")
    print(f"    real stick-x mean: left={np.mean([x for _,x in strong['left']]):+.3f} "
          f"right={np.mean([x for _,x in strong['right']]):+.3f}  (sanity: left<0<right)")

    # ---------- (B) base DiT unconditioned prediction ----------
    from eval_policy import NitroGenPolicy
    pol = NitroGenPolicy(CKPT_BASE, qwen="Qwen/Qwen3.5-2B", default_cfg=1.0)
    pol.mm_mode = True; pol.mm_text_only = True

    print("\n(B) base DiT UNCONDITIONED (null) predicted stick-x per group:")
    for d in ("left", "right"):
        raws, clamps = [], []
        for u, _ in strong[d]:
            e = lookup[u]
            fo = e["frame_offsets"]
            frame = np.asarray(Image.open(os.path.join(FRAMES_CC, f"{u}__{fo[0]}.png")).convert("RGB"))
            pf = [np.asarray(Image.open(os.path.join(FRAMES_CC, f"{u}__{o}.png")).convert("RGB"))
                  for o in fo if os.path.exists(os.path.join(FRAMES_CC, f"{u}__{o}.png"))]
            torch.manual_seed(0)
            a = pol._sample_chunk(frame, "", 1.0, plan_frames=pf, null=True)  # (H,25)
            sx = a[:, 0].mean()                       # decoded stick-x in [0,1] space (0.5=neutral)
            raws.append(float(sx)); clamps.append(float(np.clip(sx, 0, 1)))
        print(f"    {d:>5}: raw mean stick-x={np.mean(raws):+.3f}  clamped={np.mean(clamps):.3f}  "
              f"(0.5=neutral, <0.5=left, >0.5=right)")
    print("    => if left-group < 0.5 < right-group, base DiT ADHERES to the frame on left/right.")
    print("       if both ~same, a single frame does NOT determine left/right (expected; needs the plan).")


CKPT_BASE = "runs/stage2_2b_a1b48/plan_stage1_1000.pt"
if __name__ == "__main__":
    main()
