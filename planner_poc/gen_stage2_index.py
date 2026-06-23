"""Precompute the Stage-2 chunk index for COUNTERFACTUAL training (EXP-047).

For each uuid: the real action chunk at the Stage-2 window + its dominant direction. Training-
time CFG pairs a frame_i with a real plan+action from a chunk j of a DIFFERENT direction cluster
(target follows the PLAN, not the frame) so the plan-conditional learns to OVERRIDE the frame ->
counterfactual override at LOW guidance. Saved as {uuid: {buttons, j_left, j_right, dir}}.

The window is per-uuid: if the lookup carries `before_idx` (the mm lookup, window 250), the chunk
is taken at before_idx + action_shift so the transplant action matches the mm anchor. Else falls
back to the legacy frame 303.
"""
import glob, json, os, sys
import numpy as np
import torch
REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO)
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir

H = 18
STRIDE = 2
ACTION_SHIFT = 3
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage2_index.pt"
LOOKUP = json.load(open(os.environ.get("LOOKUP", "/tmp/stage2_plan_lookup.json")))
ROOTS = os.environ.get("ROOTS", "/tmp/stage1_big").split(",")


def main():
    idx = {}
    mds = []
    for root in ROOTS:
        mds += sorted(glob.glob(f"{root}/**/metadata.json", recursive=True))
    for md in mds:
        m_ = json.load(open(md)); uuid = m_["uuid"]
        if uuid not in LOOKUP:
            continue
        entry = LOOKUP[uuid]
        # EXP-050: skip non-gameplay chunks (consistent with training filter).
        if not entry.get("is_gameplay", True):
            continue
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        # window: mm anchor (before_idx + action_shift) if present, else legacy 303.
        bi = entry.get("before_idx")
        cs = (int(bi) + ACTION_SHIFT) if bi is not None else 303
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], cs, H, STRIDE) \
            or assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 3, H, STRIDE)
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
