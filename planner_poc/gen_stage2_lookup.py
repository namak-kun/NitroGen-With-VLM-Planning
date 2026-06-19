"""Scale: generate one tactical VLM plan per chunk (using the already-extracted context
frame in /tmp/frames_pre + the chunk's real action summary as grounding). Saves a lookup
{uuid: plan_text} -> /tmp/stage2_plan_lookup.json for Stage-2 training. Gemma-4-12B-it +
tactical prompt (no controller words). Fast: no ffmpeg (frames pre-extracted).
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
from nitrogen.training.actions import load_chunk_actions, assemble_chunk
from action_summary import summarize_chunk

MODEL = os.environ.get("VLM", "google/gemma-4-12B-it")
OUT = os.environ.get("OUT", "/tmp/stage2_plan_lookup.json")
# Scan one or more (frames_dir, metadata_root) pairs so we can label BOTH the original
# frames_pre/stage1_big set AND the already-downloaded frames_more/stage1_more set (scale-up
# with zero extra downloading). Override via SCAN="framesdir:metaroot,framesdir2:metaroot2".
_DEFAULT_SCAN = "/tmp/frames_pre:/tmp/stage1_big,/tmp/frames_more:/tmp/stage1_more"
SCAN = [tuple(p.split(":")) for p in os.environ.get("SCAN", _DEFAULT_SCAN).split(",")]
H, STRIDE = 18, 2

TACTICAL = (
    "You are a gameplay strategist. You see one gameplay frame and a summary of the player's "
    "actual inputs over the next ~1 second (for YOUR understanding only). Output ONE short "
    "tactical PLAN (max 12 words) stating the player's immediate GOAL in game terms. Describe "
    "INTENT, never controls — do NOT mention buttons, sticks, or directions like 'press B' or "
    "'move left'. Say what to ACHIEVE (e.g. 'intercept the ball before it reaches the net', "
    "'flank the enemy and break its guard', 'reach the next ledge'). Output ONLY the plan."
)


def main():
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    lookup = json.load(open(OUT)) if os.path.exists(OUT) else {}
    done = 0
    mds = []
    for frames_dir, meta_root in SCAN:
        for md in sorted(glob.glob(f"{meta_root}/**/metadata.json", recursive=True)):
            mds.append((frames_dir, md))
    print(f"scanning {len(mds)} chunks across {len(SCAN)} sources; {len(lookup)} already labeled", flush=True)
    for frames_dir, md in mds:
        m = json.load(open(md)); uuid = m["uuid"]; game = m.get("game", "?")
        if uuid in lookup:
            continue
        png = os.path.join(frames_dir, uuid + ".png")
        if not os.path.exists(png):
            continue
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 303, H, STRIDE)
        if rc is None:
            rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 3, H, STRIDE)
        if rc is None:
            continue
        summary = summarize_chunk(rc)
        img = Image.open(png).convert("RGB")
        ctx = f"Game: {game}\nPlayer's inputs over the next second (context only): {summary}"
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": TACTICAL + "\n\n" + ctx}]}]
        try:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt", enable_thinking=False).to(model.device)
        except TypeError:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False)
        txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
        plan = ([l.strip() for l in txt.splitlines() if l.strip()] or [txt])[-1].strip().strip('"')
        lookup[uuid] = {"plan": plan, "game": game, "action_summary": summary}
        done += 1
        if done % 10 == 0:
            json.dump(lookup, open(OUT, "w"), indent=2)
            print(f"  {done} done | [{game[:16]}] {plan}", flush=True)
    json.dump(lookup, open(OUT, "w"), indent=2)
    print(f"\nDONE: {len(lookup)} plans -> {OUT}")


if __name__ == "__main__":
    main()
