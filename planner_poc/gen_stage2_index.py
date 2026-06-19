"""Precompute the Stage-2 chunk index for COUNTERFACTUAL training (EXP-047).

For each uuid: the real action chunk at frame 303 (the same window Stage-2 uses) + its
dominant direction. Training-time CFG pairs a frame_i with a real plan+action from a chunk j
of a DIFFERENT direction cluster (target follows the PLAN, not the frame) so the
plan-conditional learns to OVERRIDE the frame -> counterfactual override at LOW guidance.
Saved as {uuid: {buttons, j_left, j_right, dir}} -> /tmp/stage2_index.pt
"""
import glob, json, os, sys
import numpy as np
import torch
REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO)
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir

H = 18
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage2_index.pt"
LOOKUP = json.load(open("/tmp/stage2_plan_lookup.json"))


def main():
    idx = {}
    for md in sorted(glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True)):
        m_ = json.load(open(md)); uuid = m_["uuid"]
        if uuid not in LOOKUP:
            continue
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 303, H, 2) \
            or assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 3, H, 2)
        if rc is None:
            continue
        idx[uuid] = {
            "buttons": np.asarray(rc["buttons"], dtype=np.float32),
            "j_left": np.asarray(rc["j_left"], dtype=np.float32),
            "j_right": np.asarray(rc["j_right"], dtype=np.float32),
            "dir": chunk_dominant_dir(rc) or "idle",
        }
    torch.save(idx, OUT)
    from collections import Counter
    print(f"saved {len(idx)} chunks -> {OUT}; dir dist:", dict(Counter(v["dir"] for v in idx.values())))


if __name__ == "__main__":
    main()
