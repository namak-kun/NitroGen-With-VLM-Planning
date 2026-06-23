"""Pre-extract REAL per-chunk frames for cross-chunk R0 training. For each chunk dir,
at a few deterministic window starts s, extract the masked frame at chunk-a-start
(frame_idx = s + a*H*stride) for a in [0,A). Saves /tmp/frames_cc/<uuid>__<frame_idx>.png
so a frame_idx-aware dir provider can read real per-chunk frames without ffmpeg in the
training loop.
"""
import glob, os, json
import numpy as np
from PIL import Image
from nitrogen.training.video import VideoFrameFetcher, VideoFetchConfig

H, STRIDE = 18, 2
A = int(os.environ.get("A", "2"))   # frames per window: a in [0,A) -> offsets s + a*H*STRIDE (A=4 for cross-chunk training)
STARTS = [int(x) for x in os.environ.get("STARTS", "100,250,400").split(",")]  # window starts (frames)
OUT = os.environ.get("OUT", "/tmp/frames_cc")
ROOTS = os.environ.get("ROOTS", "/tmp/stage1_big,/tmp/stage1_more").split(",")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
os.makedirs(OUT, exist_ok=True)
f = VideoFrameFetcher(VideoFetchConfig(cache_dir="frame_cache"))
dirs = sorted({os.path.dirname(md) for root in ROOTS
               for md in glob.glob(f"{root}/**/metadata.json", recursive=True)})
if NUM_SHARDS > 1:
    dirs = dirs[SHARD_INDEX::NUM_SHARDS]

import polars as pl
saved = skipped = 0
for di, d in enumerate(dirs):
    if di % 200 == 0:
        print(f"[shard {SHARD_INDEX}/{NUM_SHARDS}] [{di}/{len(dirs)}] saved {saved} skipped {skipped}", flush=True)
    m = json.load(open(os.path.join(d, "metadata.json")))
    ov = m["original_video"]; uuid = m["uuid"]
    pq = os.path.join(d, "actions_processed.parquet")
    if not os.path.exists(pq):
        pq = os.path.join(d, "actions_raw.parquet")
    T = pl.read_parquet(pq).height
    try:
        path = f.download_slice(ov["url"], ov["video_id"], float(ov["start_time"]), float(ov["end_time"]))
    except Exception as e:
        print("slice ERR", uuid[:16], repr(e)[:60]); continue
    if not os.path.exists(path):
        continue
    for s in STARTS:
        last = s + (A - 1) * H * STRIDE
        if last + H * STRIDE + 3 >= T:   # need room for chunk actions after the frame
            continue
        for a in range(A):
            fidx = s + a * H * STRIDE
            outp = os.path.join(OUT, f"{uuid}__{fidx}.png")
            if os.path.exists(outp):
                skipped += 1; continue
            try:
                fr = f.extract_frame(path, fidx / 60.0, size=None)
                fr = f.mask_controller(fr, m["bbox_controller_overlay"], tuple(ov["resolution"]))
                Image.fromarray(fr).save(outp)
                saved += 1
            except Exception as e:
                print("frame ERR", uuid[:16], fidx, repr(e)[:50])
print(f"saved {saved}, skipped(existing) {skipped}, dirs {len(dirs)}")
