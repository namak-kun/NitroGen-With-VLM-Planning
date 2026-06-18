"""Multi-frame VLM plan alignment. Same test as vlm_plan_alignment.py but feeds a SHORT
CLIP (N real frames spanning the chunk) so the VLM can see MOTION, then asks which way the
player is moving. Motion is what a single frame lacks. If multi-frame agreement beats the
single-frame chance result (EXP-036), video-grounded VLM plans are the Stage-2 path.
"""
import glob
import json
import os
import re
import sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/home/t-nagupta/NitroGen/planner_poc")
sys.path.insert(0, "/home/t-nagupta/NitroGen")
from transformers import AutoProcessor, AutoModelForImageTextToText
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir
from nitrogen.training.video import VideoFrameFetcher, VideoFetchConfig

MODEL = os.environ.get("VLM", "Qwen/Qwen3.5-4B")
H, STRIDE, SHIFT = 18, 2, 3
N_FRAMES = 5
WIN_START = 300  # frame index where we read the clip + the action window
DIR_WORDS = {"left": ["left"], "right": ["right"],
             "up": ["up", "upward", "forward", "north", "ahead", "climbing", "rising"],
             "down": ["down", "downward", "back", "backward", "south", "falling", "descending"]}
PROMPT = (
    "These frames are consecutive moments (in order) from a video game, ~1 second of "
    "gameplay. Watch how the player/character MOVES across the frames. In ONE short sentence "
    "(max 10 words), state which direction the player is moving and what they're doing, e.g. "
    "'moving left along the platform', 'driving right around the bend', 'climbing upward'. "
    "Base it on the MOTION between frames, not a guess. Output only the sentence."
)
fetcher = VideoFrameFetcher(VideoFetchConfig(cache_dir="/home/t-nagupta/NitroGen/frame_cache",
                                             cookies_file="/home/t-nagupta/NitroGen/cookies.txt"))


def load_vlm():
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def clip_frames(meta):
    ov = meta["original_video"]
    path = fetcher.download_slice(ov["url"], ov["video_id"], float(ov["start_time"]), float(ov["end_time"]))
    if not os.path.exists(path):
        return None
    idxs = np.linspace(WIN_START, WIN_START + (H - 1) * STRIDE, N_FRAMES).astype(int)
    imgs = []
    for fi in idxs:
        fr = fetcher.extract_frame(path, int(fi) / 60.0, size=None)
        fr = fetcher.mask_controller(fr, meta["bbox_controller_overlay"], tuple(ov["resolution"]))
        imgs.append(Image.fromarray(fr).convert("RGB").resize((384, 216)))
    return imgs


def plan(proc, model, imgs):
    content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": PROMPT}]
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
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip().lower()
    return txt.splitlines()[-1] if txt.splitlines() else txt


def plan_dir(text):
    for t in re.findall(r"[a-z]+", text):
        for d, syns in DIR_WORDS.items():
            if t in syns:
                return d
    return None


def main():
    proc, model = load_vlm()
    items = []
    for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True):
        m = json.load(open(md))
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], WIN_START + SHIFT, H, STRIDE)
        d = chunk_dominant_dir(rc) if rc is not None else None
        if d is not None:
            items.append((m, d, m.get("game", "?")))
    # diversify: cap per game so it isn't all one title
    per_game = {}
    picked = []
    for m, d, g in items:
        if per_game.get(g, 0) < 6:
            per_game[g] = per_game.get(g, 0) + 1
            picked.append((m, d, g))
        if len(picked) >= 40:
            break
    print(f"testing {len(picked)} chunks (multi-frame clips)\n")
    agree = have = axis_agree = 0
    for m, real_d, game in picked:
        try:
            imgs = clip_frames(m)
        except Exception as e:
            print("  clip err", repr(e)[:50]); continue
        if not imgs:
            continue
        p = plan(proc, model, imgs)
        pd = plan_dir(p)
        if pd is None:
            print(f"  real={real_d:5s} VLM=(no dir) [{game[:14]}] {p[:50]}"); continue
        have += 1; ok = (pd == real_d); agree += ok
        ax = lambda x: "x" if x in ("left", "right") else "y"
        axis_agree += (ax(pd) == ax(real_d))
        print(f"  real={real_d:5s} VLM={pd:5s} {'OK' if ok else '  '} [{game[:14]}] {p[:50]}")
    print(f"\n=== MULTI-FRAME alignment (vs single-frame EXP-036 = 25%/50% chance) ===")
    print(f"  chunks with a VLM direction: {have}")
    if have:
        print(f"  exact-direction agreement: {agree}/{have} = {100*agree/have:.0f}%  (chance 25%)")
        print(f"  axis (x/y) agreement:      {axis_agree}/{have} = {100*axis_agree/have:.0f}%  (chance 50%)")


if __name__ == "__main__":
    main()
