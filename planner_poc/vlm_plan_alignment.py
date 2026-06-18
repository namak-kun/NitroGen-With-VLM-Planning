"""Does a VLM-generated plan PREDICT the real action? Critical alignment test before
training on VLM plans. For chunks where the streamer clearly moved (dominant direction
known), generate a VLM plan from the frame, extract any direction word from the plan, and
measure agreement with the REAL chunk's dominant direction. Chance ~= 25% (4 cardinals);
well above chance => VLM plans are action-predictive and worth training on.
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

MODEL = os.environ.get("VLM", "Qwen/Qwen3.5-4B")
FRAMES = "/tmp/frames_pre"
H, STRIDE, SHIFT = 18, 2, 3

# direction synonyms the VLM might use
DIR_WORDS = {
    "left": ["left"], "right": ["right"],
    "up": ["up", "upward", "forward", "north", "ahead", "top"],
    "down": ["down", "downward", "back", "backward", "south", "retreat", "bottom"],
}
PROMPT = (
    "Look at this video game frame. In ONE short imperative sentence (max 10 words), say "
    "which way the player should move RIGHT NOW and what to do, e.g. 'move left to dodge', "
    "'go right toward the door', 'push up the field'. Output only the sentence."
)


def load_vlm():
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def plan(proc, model, img):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
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
    """Extract the first direction word mentioned."""
    toks = re.findall(r"[a-z]+", text)
    for t in toks:
        for d, syns in DIR_WORDS.items():
            if t in syns:
                return d
    return None


def main():
    proc, model = load_vlm()
    # gather chunks with a clear real dominant direction + a frame
    items = []
    for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True):
        m = json.load(open(md)); uuid = m["uuid"]
        png = os.path.join(FRAMES, uuid + ".png")
        if not os.path.exists(png):
            continue
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        # use the dominant dir over the whole first valid window
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 300 + SHIFT, H, STRIDE)
        if rc is None:
            rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], SHIFT, H, STRIDE)
        d = chunk_dominant_dir(rc) if rc is not None else None
        if d is not None:
            items.append((png, d, m.get("game", "?")))
    items = items[:60]
    print(f"testing {len(items)} chunks with a clear real direction\n")
    agree = 0; have = 0; axis_agree = 0
    conf = {}
    for png, real_d, game in items:
        p = plan(proc, model, Image.open(png).convert("RGB"))
        pd = plan_dir(p)
        if pd is None:
            print(f"  real={real_d:5s} VLM=(no dir)   [{game[:14]}] {p[:60]}")
            continue
        have += 1
        ok = (pd == real_d)
        agree += ok
        # axis agreement (x vs y) is a softer signal
        ax = lambda x: "x" if x in ("left", "right") else "y"
        axis_agree += (ax(pd) == ax(real_d))
        conf[(real_d, pd)] = conf.get((real_d, pd), 0) + 1
        print(f"  real={real_d:5s} VLM={pd:5s} {'OK' if ok else '  '} [{game[:14]}] {p[:55]}")
    print(f"\n=== alignment: VLM plan direction vs REAL dominant direction ===")
    print(f"  chunks with a VLM direction word: {have}/{len(items)}")
    if have:
        print(f"  exact-direction agreement: {agree}/{have} = {100*agree/have:.0f}%  (chance ~25%)")
        print(f"  axis (x/y) agreement:      {axis_agree}/{have} = {100*axis_agree/have:.0f}%  (chance ~50%)")
        print("  >> well above chance => VLM frame-plans are action-predictive (trainable).")
        print("  >> near chance => the VLM describes the scene, not the player's motion;")
        print("     need multi-frame/video plans or different prompting.")


if __name__ == "__main__":
    main()
