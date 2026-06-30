"""Stage-2 windowed plan generator (LOCKED spec, user-confirmed 2026-06-18).

Walk a video's chunks in order, in WINDOWS of A chunks (A in {4,8}). For each window the
VLM produces a grounded plan from:
  - N frames sampled across the window (sees motion; N sweepable, default 12)
  - an ACTION SUMMARY of what the player did over the window (grounding; the VLM can't read
    gamepad from pixels, EXP-036/037)
  - the last >=2 PRIOR PLANS (so the VLM sees plan -> reaction -> plan; continuity + feedback)
  - the TRANSCRIPT window (same A-chunk span; free weak signal, refined by actions+frames)
  - the game name
Plans are generated SEQUENTIALLY so plan(t) is conditioned on plan(t-1), plan(t-2), ...
forming a rolling history. Output: one terse grounded plan per window + JSON dump.

All cadence/context knobs are CLI flags so we can sweep A, n_frames, n_prior, model.
"""
import argparse
import glob
import json
import os
import re
import sys
import numpy as np
import torch
from PIL import Image
import os; sys.path.insert(0, os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "planner_poc"))
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
from transformers import AutoProcessor, AutoModelForImageTextToText
from nitrogen.training.actions import load_chunk_actions, assemble_chunk
from nitrogen.training.video import VideoFrameFetcher, VideoFetchConfig
from action_summary import summarize_chunk
from vtt_align import parse_vtt, window_text, best_vtt

H, STRIDE, FPS = 18, 2, 60.0
TS_DIR = "/tmp/transcripts"
fetcher = VideoFrameFetcher(VideoFetchConfig(cache_dir=os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "frame_cache"),
                                             cookies_file=os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "cookies.txt")))

SYS = (
    "You are a gameplay analyst producing a concise PLAN that explains the player's "
    "intent. You are given several frames spanning ~{secs:.1f}s of play, a precise summary "
    "of the gamepad inputs the player actually made (ground truth), the previous plans, and "
    "the streamer's spoken words (often off-topic chit-chat). The plan must describe what is "
    "happening in THIS window's frames and actions (use the previous plans only for "
    "continuity, do NOT just repeat them). Synthesize ONE short imperative plan (max 14 "
    "words) describing the player's immediate tactical goal RIGHT NOW, grounded in what they "
    "actually did. Be game-specific and concrete (e.g. 'cut left and shoot the ball into the "
    "open net', 'climb the ledge then drop onto the enemy'). Output ONLY the plan."
)


def load_vlm(model_name):
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16,
                                                        device_map="cuda").eval()
    return proc, model


def gen_plan(proc, model, frames, ctx_text, secs):
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": SYS.format(secs=secs) + "\n\n" + ctx_text})
    msgs = [{"role": "user", "content": content}]
    try:
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt", enable_thinking=False).to(model.device)
    except TypeError:
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=48, do_sample=False)
    txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return (lines[-1] if lines else txt).strip().strip('"')


def video_windows(meta, acts, A, n_frames):
    """Yield (window_start_frame, [per-chunk action summaries], frames, t0_sec, t1_sec)."""
    T = acts["buttons"].shape[0]
    # we can only use the local action slice (1 chunk dir == 1 mp4 slice); walk windows
    # of A chunks within this slice's action array.
    win_frames = A * H * STRIDE
    s = 100  # skip the very start
    ov = meta["original_video"]
    while s + win_frames + STRIDE * H < T:
        # action summaries for the A chunks
        summaries = []
        ok = True
        for c in range(A):
            cs = s + c * H * STRIDE
            rc = assemble_chunk(acts["buttons"], acts["j_left"], acts["j_right"], cs, H, STRIDE)
            if rc is None:
                ok = False; break
            summaries.append(summarize_chunk(rc))
        if not ok:
            break
        # frames across the window
        path = fetcher.download_slice(ov["url"], ov["video_id"], float(ov["start_time"]), float(ov["end_time"]))
        if not os.path.exists(path):
            return
        idxs = np.linspace(s, s + win_frames - 1, n_frames).astype(int)
        frames = []
        for fi in idxs:
            fr = fetcher.extract_frame(path, int(fi) / FPS, size=None)
            fr = fetcher.mask_controller(fr, meta["bbox_controller_overlay"], tuple(ov["resolution"]))
            frames.append(Image.fromarray(fr).convert("RGB").resize((448, 252)))
        t0 = float(ov["start_time"]) + s / FPS
        t1 = float(ov["start_time"]) + (s + win_frames) / FPS
        yield s, summaries, frames, t0, t1
        s += win_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=int, default=4, help="chunks per VLM call (cadence)")
    ap.add_argument("--n-frames", type=int, default=12, help="frames sampled per window")
    ap.add_argument("--n-prior", type=int, default=2, help="prior plans shown to the VLM")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-windows", type=int, default=8, help="windows per video (cap for the demo)")
    ap.add_argument("--videos", type=int, default=3, help="number of videos")
    ap.add_argument("--out", default="/tmp/stage2_plans.json")
    args = ap.parse_args()

    proc, model = load_vlm(args.model)
    secs = args.A * H * STRIDE / FPS
    # pick chunk dirs from distinct videos
    mds = sorted(glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))
    seen, picks = set(), []
    for md in mds:
        m = json.load(open(md)); vid = m["original_video"]["video_id"]
        if vid in seen:
            continue
        seen.add(vid); picks.append((md, m))
        if len(picks) >= args.videos:
            break

    results = []
    for md, m in picks:
        vid = m["original_video"]["video_id"]; game = m.get("game", "?")
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            acts = load_chunk_actions(pq)
        except Exception:
            continue
        vp = best_vtt(TS_DIR, vid); cues = parse_vtt(vp) if vp else []
        print(f"\n#### {game}  ({vid})  A={args.A} ({secs:.1f}s)  n_frames={args.n_frames}\n", flush=True)
        history = []  # rolling plan list
        for wi, (s, summaries, frames, t0, t1) in enumerate(video_windows(m, acts, args.A, args.n_frames)):
            if wi >= args.max_windows:
                break
            tw = window_text(cues, t0, t1) if cues else ""
            tw = (tw[:200] + "…") if len(tw) > 200 else tw
            # build context text
            ctx = [f"Game: {game}"]
            if history:
                prior = history[-args.n_prior:]
                ctx.append("Previous plans (oldest first): " + " | ".join(prior))
            ctx.append("Gamepad inputs the player actually made over this window (per "
                       f"{H*STRIDE/FPS:.1f}s chunk, in order):")
            for ci, sm in enumerate(summaries):
                ctx.append(f"  chunk {ci+1}: {sm}")
            ctx.append("Streamer said (may be irrelevant): " + (tw or "(silence/none)"))
            plan = gen_plan(proc, model, frames, "\n".join(ctx), secs)
            history.append(plan)
            print(f"  [win {wi}] {plan}")
            print(f"          actions: {summaries[0]}{' ...' if len(summaries)>1 else ''}")
            print(f"          said: {tw[:90] or '(none)'}\n", flush=True)
            results.append({"vid": vid, "game": game, "window": wi, "t0": t0, "t1": t1,
                            "A": args.A, "n_frames": args.n_frames, "plan": plan,
                            "action_summaries": summaries, "transcript": tw})
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {len(results)} plans -> {args.out}")


if __name__ == "__main__":
    main()
