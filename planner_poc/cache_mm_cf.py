"""EXP-051b: precompute COUNTERFACTUAL frame-conditioned plan-hidden + override targets.

The factual mm-hidden cache is keyed to each uuid's OWN plan: encode_multimodal(frame_i, plan_i).
Counterfactual override needs the CROSS pair encode_multimodal(frame_i, plan_j) where plan_j
comes from a DIFFERENT-direction chunk and the target is action_j -> teaches the plan to OVERRIDE
the frame's prior. We precompute ONE such pairing per gameplay uuid (seeded): pick a uuid j whose
dominant direction differs from i, encode [before_i, after_i] + plan_j with the frozen 0.8B, and
store the override action_j + its dir.

Out: torch.save({uuid_i: {h:(L,d)fp16, mask:(L,), buttons,j_left,j_right (action_j),
                          cf_dir, own_dir, cf_uuid}}) -> /tmp/stage2_mm_cf.pt
Run: PYTHONPATH=. .venv/bin/python planner_poc/cache_mm_cf.py
"""
import json, os, random, sys
import numpy as np
import torch
from PIL import Image

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.planner import PlanEncoder, PlannerConfig

DEVICE = "cuda"
LOOKUP = json.load(open(os.environ.get("LOOKUP", "/tmp/stage2_plan_lookup_mm.json")))
INDEX = torch.load(os.environ.get("INDEX", "/tmp/stage2_index_mm.pt"),
                   map_location="cpu", weights_only=False)
OUT = os.environ.get("OUT", "/tmp/stage2_mm_cf.pt")
FRAMES_CC = os.environ.get("FRAMES_CC", "/tmp/frames_cc")
QWEN = os.environ.get("QWEN", f"{REPO}/ckpts/qwen35-0.8b")
MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))
SEED = int(os.environ.get("SEED", "0"))
TEXT_ONLY = os.environ.get("TEXT_ONLY", "0") == "1"   # EXP-052: keep only text-token hiddens
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))


def _resize(im):
    w, h = im.size
    s = MAX_SIDE / max(w, h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else im


def main():
    rng = random.Random(SEED)
    # gameplay uuids that have a frame pair + plan + a directional (non-idle) index entry.
    uuids = [u for u, e in LOOKUP.items()
             if e.get("is_gameplay") and u in INDEX]
    # group by dominant dir for cross-direction pairing.
    by_dir = {}
    for u in uuids:
        by_dir.setdefault(INDEX[u]["dir"], []).append(u)
    print(f"{len(uuids)} gameplay uuids; dir dist: "
          f"{ {d: len(v) for d, v in by_dir.items()} }")

    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load()
    out, skip = {}, 0
    work = uuids[SHARD_INDEX::NUM_SHARDS] if NUM_SHARDS > 1 else uuids
    for i, u in enumerate(work):
        own_dir = INDEX[u]["dir"]
        e = LOOKUP[u]
        # N-frame author window of the OWN uuid (frame_i), paired with a DIFFERENT-direction
        # plan_j -> the cross-pair hidden. Counterfactual is single-chunk (act-now / cursor 0),
        # so it uses the same frames the cursor-0 factual hidden does.
        offs = e.get("frame_offsets") or [e.get("before_idx"), e.get("after_idx")]
        paths = [os.path.join(FRAMES_CC, f"{u}__{o}.png") for o in offs]
        # candidate cf uuids: a DIFFERENT, non-idle direction with a clear motor target.
        cand_dirs = [d for d in ("left", "right", "up", "down") if d != own_dir and by_dir.get(d)]
        if not (all(os.path.exists(p) for p in paths) and cand_dirs):
            skip += 1; continue
        cf_dir = rng.choice(cand_dirs)
        uj = rng.choice(by_dir[cf_dir])
        plan_j = LOOKUP[uj]["plan"]
        frames = [_resize(Image.open(p).convert("RGB")) for p in paths]
        h, kpm = pl.encode_multimodal(frames, plan_j, DEVICE, text_only=TEXT_ONLY)
        out[u] = {
            "h": h[0].half().cpu(), "mask": kpm[0].cpu(),
            "buttons": np.asarray(INDEX[uj]["buttons"], dtype=np.float32),
            "j_left": np.asarray(INDEX[uj]["j_left"], dtype=np.float32),
            "j_right": np.asarray(INDEX[uj]["j_right"], dtype=np.float32),
            "cf_dir": cf_dir, "own_dir": own_dir, "cf_uuid": uj,
        }
        if (i + 1) % 40 == 0:
            print(f"  [shard {SHARD_INDEX}/{NUM_SHARDS}] {i+1}/{len(work)} cached", flush=True)
    out_path = OUT if NUM_SHARDS == 1 else f"{OUT}.shard{SHARD_INDEX}"
    torch.save(out, out_path)
    from collections import Counter
    print(f"\nDONE: {len(out)} cf pairs -> {out_path}; skipped {skip}")
    print("  cf_dir dist:", dict(Counter(v["cf_dir"] for v in out.values())))
    print("  own->cf sample:", [(v["own_dir"], v["cf_dir"]) for v in list(out.values())[:6]])


if __name__ == "__main__":
    main()
