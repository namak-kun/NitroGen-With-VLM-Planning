"""Background: expand the training pool by extracting a context frame for chunks from
NEW videos (videos in SHARD_0000 we haven't fetched yet). Writes chunk dirs to
/tmp/stage1_more and a masked context frame per chunk to /tmp/frames_more/<uuid>.png
(mirrors /tmp/frames_pre so the existing dir frame provider can read it). Non-GPU;
safe to run in the background while training uses the GPU. Bounded by --max-chunks.
"""
import argparse, glob, json, os, shutil, sys, time
import numpy as np
from PIL import Image
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
from nitrogen.training.video import VideoFrameFetcher, VideoFetchConfig

SHARD = "/tmp/ds_real/SHARD_0000"
OUT_CHUNKS = "/tmp/stage1_more"
OUT_FRAMES = "/tmp/frames_more"
CTX_FRAME = 300  # fixed context-frame offset (~5s in); representative game frame


def have_videos():
    return {os.path.basename(d.rstrip("/")) for d in glob.glob("/tmp/stage1_big/*/")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=400)
    ap.add_argument("--max-videos", type=int, default=40)
    ap.add_argument("--chunks-per-video", type=int, default=12)
    ap.add_argument("--games", default=None,
                    help="comma-separated substrings to filter games (e.g. elden_ring,dark_souls); "
                         "prioritizes videos whose chunks match (planning matters more for RPG/action)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="partition the new-video list into N disjoint shards for parallel workers")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="which shard (0..num_shards-1) this worker handles (vids[shard_index::num_shards])")
    args = ap.parse_args()
    os.makedirs(OUT_CHUNKS, exist_ok=True)
    os.makedirs(OUT_FRAMES, exist_ok=True)
    f = VideoFrameFetcher(VideoFetchConfig(
        cache_dir=os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "frame_cache"),
        cookies_file=os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "cookies.txt"),
    ))

    have = have_videos()
    game_filter = [g.strip().lower() for g in args.games.split(",")] if args.games else None

    def video_matches(vdir):
        """True if any chunk in this video has a game matching the filter."""
        if game_filter is None:
            return True
        for c in glob.glob(os.path.join(vdir, "*", "metadata.json"))[:3]:
            try:
                g = str(json.load(open(c)).get("game", "")).lower()
            except Exception:
                continue
            if any(gf in g for gf in game_filter):
                return True
        return False

    vids = [d for d in sorted(glob.glob(f"{SHARD}/*/")) if os.path.basename(d.rstrip("/")) not in have]
    if game_filter:
        vids = [d for d in vids if video_matches(d)]
    if args.num_shards > 1:
        vids = vids[args.shard_index :: args.num_shards]
    print(f"[shard {args.shard_index}/{args.num_shards}] {len(vids)} new videos available "
          f"(game filter={game_filter}); "
          f"targeting <= {args.max_videos} videos / {args.max_chunks} chunks", flush=True)

    saved = vid_done = 0
    for vdir in vids:
        if vid_done >= args.max_videos or saved >= args.max_chunks:
            break
        vid = os.path.basename(vdir.rstrip("/"))
        chunk_dirs = sorted(glob.glob(os.path.join(vdir, "*", "metadata.json")))
        chunk_dirs = [os.path.dirname(c) for c in chunk_dirs][: args.chunks_per_video]
        if not chunk_dirs:
            continue
        v_saved = 0
        for cd in chunk_dirs:
            if saved >= args.max_chunks:
                break
            meta = json.load(open(os.path.join(cd, "metadata.json")))
            uuid = meta["uuid"]; ov = meta["original_video"]
            outp = os.path.join(OUT_FRAMES, uuid + ".png")
            if os.path.exists(outp):
                continue
            try:
                path = f.download_slice(ov["url"], ov["video_id"], float(ov["start_time"]), float(ov["end_time"]))
                if not os.path.exists(path):
                    continue
                fr = f.extract_frame(path, CTX_FRAME / 60.0, size=None)
                fr = f.mask_controller(fr, meta["bbox_controller_overlay"], tuple(ov["resolution"]))
                Image.fromarray(fr).save(outp)
                # copy chunk dir (metadata + parquets) into the training pool
                dst = os.path.join(OUT_CHUNKS, vid, os.path.basename(cd))
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copytree(cd, dst)
                saved += 1; v_saved += 1
            except Exception as e:
                print(f"  ERR {uuid[:18]} {repr(e)[:70]}", flush=True)
        if v_saved:
            vid_done += 1
            print(f"[{vid_done}] {vid}: +{v_saved} chunks (total {saved})", flush=True)
    print(f"DONE: {saved} chunks from {vid_done} new videos", flush=True)


if __name__ == "__main__":
    main()
