"""Cache a fixed, diverse set of Stage-2 windows to disk so we can iterate on PROMPTS and
MODELS cheaply (frame extraction via ffmpeg is the slow part). Saves per-window: the frames
(PNGs), the action summaries, the transcript window, and metadata -> /tmp/s2_windows/.
"""
import glob
import json
import os
import sys
import numpy as np
from PIL import Image
import os; sys.path.insert(0, os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "planner_poc"))
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
from nitrogen.training.actions import load_chunk_actions, assemble_chunk
from nitrogen.training.video import VideoFrameFetcher, VideoFetchConfig
from action_summary import summarize_chunk
from vtt_align import parse_vtt, window_text, best_vtt

H, STRIDE, FPS = 18, 2, 60.0
A = 4
N_FRAMES = 16
OUT = "/tmp/s2_windows"
TS_DIR = "/tmp/transcripts"
os.makedirs(OUT, exist_ok=True)
fetcher = VideoFrameFetcher(VideoFetchConfig(cache_dir=os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "frame_cache"),
                                             cookies_file=os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "cookies.txt")))


def main():
    mds = sorted(glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))
    seen, picks = set(), []
    for md in mds:
        m = json.load(open(md)); vid = m["original_video"]["video_id"]
        if vid in seen:
            continue
        seen.add(vid); picks.append((md, m))
        if len(picks) >= 6:
            break
    manifest = []
    for md, m in picks:
        vid = m["original_video"]["video_id"]; game = m.get("game", "?")
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            acts = load_chunk_actions(pq)
        except Exception:
            continue
        ov = m["original_video"]
        path = fetcher.download_slice(ov["url"], ov["video_id"], float(ov["start_time"]), float(ov["end_time"]))
        if not os.path.exists(path):
            continue
        vp = best_vtt(TS_DIR, vid); cues = parse_vtt(vp) if vp else []
        T = acts["buttons"].shape[0]; win_frames = A * H * STRIDE
        # take up to 3 windows per video, prefer ones with real action
        s = 100; taken = 0
        while s + win_frames + STRIDE * H < T and taken < 3:
            sums, ok, active = [], True, False
            for c in range(A):
                cs = s + c * H * STRIDE
                rc = assemble_chunk(acts["buttons"], acts["j_left"], acts["j_right"], cs, H, STRIDE)
                if rc is None:
                    ok = False; break
                sm = summarize_chunk(rc)
                sums.append(sm)
                if "neutral; no buttons" not in sm:
                    active = True
            if ok and active:  # only cache windows where something happened
                wid = f"{vid}_{s}"
                wdir = os.path.join(OUT, wid); os.makedirs(wdir, exist_ok=True)
                idxs = np.linspace(s, s + win_frames - 1, N_FRAMES).astype(int)
                for i, fi in enumerate(idxs):
                    fr = fetcher.extract_frame(path, int(fi) / FPS, size=None)
                    fr = fetcher.mask_controller(fr, ov["bbox_controller_overlay"], tuple(ov["resolution"])) \
                        if "bbox_controller_overlay" in ov else fetcher.mask_controller(fr, m["bbox_controller_overlay"], tuple(ov["resolution"]))
                    Image.fromarray(fr).convert("RGB").resize((448, 252)).save(os.path.join(wdir, f"f{i:02d}.png"))
                t0 = float(ov["start_time"]) + s / FPS; t1 = float(ov["start_time"]) + (s + win_frames) / FPS
                tw = window_text(cues, t0, t1) if cues else ""
                meta = {"wid": wid, "vid": vid, "game": game, "A": A, "n_frames": N_FRAMES,
                        "action_summaries": sums, "transcript": tw, "t0": t0, "t1": t1}
                json.dump(meta, open(os.path.join(wdir, "meta.json"), "w"), indent=2)
                manifest.append(wid)
                print(f"cached {wid} [{game}] active={active}", flush=True)
                taken += 1
            s += win_frames
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"))
    print(f"\ncached {len(manifest)} windows -> {OUT}")


if __name__ == "__main__":
    main()
