"""Build a FULL-A4-SPAN dominant-direction index for the directional-plan experiment.

The directional plan describes the whole ~1.8s A=4 span, so its left/right aligns with the
full-span action direction (~0.75), not the first 0.6s chunk (~0.53). For a fair token-separation
probe we label each uuid by the dominant stick direction over ALL A=4 chunks.

Out: {uuid: {buttons,j_left,j_right (chunk0, for compat), dir (FULL-SPAN dominant)}}
Run: PYTHONPATH=. .venv/bin/python planner_poc/build_fullspan_index.py \
        /tmp/stage2_plan_lookup_mm_dir.json /tmp/stage2_index_fullspan.pt
"""
import glob
import json
import os
import sys

import numpy as np
import torch

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO)
from nitrogen.training.actions import load_chunk_actions, assemble_chunk

LOOKUP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage2_plan_lookup_mm_dir.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stage2_index_fullspan.pt"
ROOTS = os.environ.get("ROOTS", "/tmp/stage1_big,/tmp/stage1_more").split(",")
H, STRIDE = 18, 2


def main():
    lookup = json.load(open(LOOKUP))
    dir_by_uuid = {}
    for r in ROOTS:
        for md in glob.glob(f"{r}/**/metadata.json", recursive=True):
            try:
                dir_by_uuid[json.load(open(md))["uuid"]] = os.path.dirname(md)
            except Exception:
                pass
    idx = {}
    for u, e in lookup.items():
        cs = e.get("chunk_starts")
        d = dir_by_uuid.get(u)
        if not cs or not d:
            continue
        pq = os.path.join(d, "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(d, "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        xs, ys, c0 = [], [], None
        for ci, c in enumerate(cs):
            rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], c, H, STRIDE)
            if rc is None:
                continue
            if ci == 0:
                c0 = rc
            xs.append(np.asarray(rc["j_left"])[:, 0])
            ys.append(np.asarray(rc["j_left"])[:, 1])
        if not xs or c0 is None:
            continue
        mx = float(np.concatenate(xs).mean()); my = float(np.concatenate(ys).mean())
        if max(abs(mx), abs(my)) < 0.12:
            direction = "idle"
        elif abs(mx) >= abs(my):
            direction = "right" if mx > 0 else "left"
        else:
            direction = "down" if my > 0 else "up"
        idx[u] = {"buttons": np.asarray(c0["buttons"], np.float32),
                  "j_left": np.asarray(c0["j_left"], np.float32),
                  "j_right": np.asarray(c0["j_right"], np.float32),
                  "dir": direction}
    torch.save(idx, OUT)
    from collections import Counter
    print(f"saved {len(idx)} full-span dirs -> {OUT}; dist:",
          dict(Counter(v["dir"] for v in idx.values())))


if __name__ == "__main__":
    main()
