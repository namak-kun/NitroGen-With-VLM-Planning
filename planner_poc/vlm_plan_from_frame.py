"""VLM-on-frames plan generation: feed a real gameplay frame to a vision LM and ask for the
player's near-term intent, as a game-contextual PLAN (the "I need to go to Hyrule Castle"
idea). Compares against the transcript window for the same chunk. This tests whether a VLM
can produce grounded, action-relevant plans from pixels alone (no streamer narration).
"""
import glob
import json
import os
import sys
import time
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/home/t-nagupta/NitroGen/planner_poc")
from vtt_align import parse_vtt, window_text, best_vtt
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = os.environ.get("VLM", "Qwen/Qwen3.5-4B")
TS_DIR = "/tmp/transcripts"
FRAMES = "/tmp/frames_pre"

PROMPT = (
    "You are watching a single frame of a video game being played. In ONE short imperative "
    "sentence (max 12 words), state the player's most likely immediate goal or next action "
    "right now — a concrete plan they would think, like 'chase the ball and shoot' or "
    "'turn left into the corner' or 'climb toward the tower'. Only output the sentence."
)


def load_vlm():
    print(f"loading {MODEL} ...", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    return proc, model


def plan_for_frame(proc, model, img: Image.Image) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": PROMPT},
    ]}]
    # Qwen3.5 is a thinking model; disable thinking so we get the terse answer directly.
    try:
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt",
                                          enable_thinking=False).to(model.device)
    except TypeError:
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    txt = proc.decode(gen, skip_special_tokens=True).strip()
    # strip any leftover <think>...</think> and keep the last non-empty line
    import re as _re
    txt = _re.sub(r"<think>.*?</think>", "", txt, flags=_re.S).strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return (lines[-1] if lines else txt).replace("\n", " ")


def chunk_samples(n_per_game=2):
    """Pick a few chunks spread across games that have a pre-extracted frame + known slice."""
    by_game = {}
    for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True):
        m = json.load(open(md)); uuid = m["uuid"]
        png = os.path.join(FRAMES, uuid + ".png")
        if not os.path.exists(png):
            continue
        by_game.setdefault(m.get("game", "?"), []).append((md, m, png))
    out = []
    for g, items in by_game.items():
        out.extend(items[:n_per_game])
    return out


def main():
    proc, model = load_vlm()
    samples = chunk_samples(2)
    print(f"\n=== VLM plans vs transcript ({len(samples)} chunks) ===\n", flush=True)
    cue_cache = {}
    for md, m, png in samples:
        vid = m["original_video"]["video_id"]; game = m.get("game", "?")
        s = float(m["original_video"]["start_time"]); e = float(m["original_video"]["end_time"])
        img = Image.open(png).convert("RGB")
        vlm_plan = plan_for_frame(proc, model, img)
        # transcript window for the same slice
        if vid not in cue_cache:
            vp = best_vtt(TS_DIR, vid)
            cue_cache[vid] = parse_vtt(vp) if vp else []
        tw = window_text(cue_cache[vid], s, e)
        tw = (tw[:140] + "…") if len(tw) > 140 else (tw or "(no transcript)")
        print(f"[{game}]  {vid} @ {s:.0f}-{e:.0f}s")
        print(f"   VLM plan : {vlm_plan}")
        print(f"   transcript: {tw}\n", flush=True)


if __name__ == "__main__":
    main()
