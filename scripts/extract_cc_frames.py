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

H, STRIDE, A = 18, 2, 2
STARTS = [100, 250, 400]           # window starts (frames); +0 and +36 for chunk0/chunk1
OUT = "/tmp/frames_cc"
os.makedirs(OUT, exist_ok=True)
f = VideoFrameFetcher(VideoFetchConfig(cache_dir="frame_cache"))
dirs = sorted(os.path.dirname(md) for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))

import polars as pl
saved = skipped = 0
for d in dirs:
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
