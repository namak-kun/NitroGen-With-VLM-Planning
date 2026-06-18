"""Reaction-aware Stage-2 plans. User insight: plans/transcripts should capture
plan -> reaction (intent, then how it turned out, then the adjustment). We make the rolling
context explicit: the VLM sees the PRIOR plan + a short note on what the player ACTUALLY did
next (from the next window's action summary) so the new plan can reflect an ADJUSTMENT, not
just continuity. Runs over cached windows of ONE video in order (cache_s2_windows.py).

This also yields better "transcript-like" supervision: each window gets (prior_plan,
what_happened, new_plan) -> a small narrative the planner can learn temporal intent from.
"""
import argparse
import glob
import json
import os
import re
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

WIN = "/tmp/s2_windows"

SYS = (
    "You are a gameplay strategist narrating a player's evolving tactics. For THIS ~{secs:.1f}s "
    "window you see: frames, a summary of the player's actual inputs (for your understanding "
    "only — never mention controls), your OWN previous plan, and what the player actually did "
    "afterward. Output ONE short tactical PLAN (max 12 words) for the player's immediate GOAL "
    "now. If the previous plan clearly failed or the situation changed, make the new plan an "
    "explicit ADJUSTMENT (e.g. 'that whiffed — fall back and reset for a clean approach'). "
    "Describe INTENT in game terms, never buttons/sticks/directions. Output ONLY the plan."
)


def load_vlm(name):
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def gen(proc, model, frames, ctx, secs):
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": SYS.format(secs=secs) + "\n\n" + ctx})
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
    ls = [l.strip() for l in txt.splitlines() if l.strip()]
    return (ls[-1] if ls else txt).strip().strip('"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--vid", default=None, help="restrict to one video id (else first found)")
    args = ap.parse_args()
    proc, model = load_vlm(args.model)
    # load windows for one video, in time order
    wins = []
    for wdir in sorted(glob.glob(os.path.join(WIN, "*"))):
        mp = os.path.join(wdir, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        frames = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(wdir, "f*.png")))]
        wins.append((meta, frames))
    vid = args.vid or wins[0][0]["vid"]
    wins = [w for w in wins if w[0]["vid"] == vid]
    wins.sort(key=lambda w: w[0]["t0"])
    print(f"reaction-aware plans for {vid} ({wins[0][0]['game']}), {len(wins)} windows, {args.model}\n")

    prior_plan = None
    for i, (meta, frames) in enumerate(wins):
        secs = meta["A"] * 18 * 2 / 60.0
        # "what happened" = a terse gloss of THIS window's first action (the outcome of the prior plan)
        outcome = meta["action_summaries"][0]
        ctx = [f"Game: {meta['game']}"]
        if prior_plan is not None:
            ctx.append(f"Your previous plan was: \"{prior_plan}\"")
            ctx.append(f"What the player then actually did (outcome): {outcome}")
        ctx.append("Player's inputs across THIS window (context only):")
        for ci, sm in enumerate(meta["action_summaries"]):
            ctx.append(f"  chunk {ci+1}: {sm}")
        tw = meta["transcript"][:160]
        ctx.append("Streamer said: " + (tw or "(none)"))
        plan = gen(proc, model, frames, "\n".join(ctx), secs)
        tag = "ADJUST" if prior_plan else "INIT  "
        print(f"  [{i}] prior: {prior_plan or '(none)'}")
        print(f"      did  : {outcome}")
        print(f"      PLAN : {plan}\n", flush=True)
        prior_plan = plan


if __name__ == "__main__":
    main()
