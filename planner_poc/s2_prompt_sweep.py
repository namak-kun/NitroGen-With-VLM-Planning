"""Fast Stage-2 plan exploration over CACHED windows (cache_s2_windows.py). Iterate on
PROMPT VARIANTS and MODELS without re-extracting frames. Scores each plan for the design
target — informative tactical intent, NOT per-action directive (user principle):
  - directiveness (LOW good): mentions of buttons/sticks/explicit-input words
  - specificity   (HIGH good): mentions of game nouns/objects (ball, net, enemy, ledge, ...)
  - length
Outputs plans side-by-side per window so quality is eyeball-comparable.
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

# words that signal LOW-LEVEL DIRECTIVE content (we want few of these in the PLAN)
DIRECTIVE = set("""button press a b x y rb lb rt lt l3 r3 dpad stick joystick trigger
left-stick hold-a tap accelerate brake jump-button up down left right northward southward""".split())
# game-object nouns that signal informative specificity (rough, game-agnostic)
OBJECTS = set("""ball net goal enemy boss door gate ledge wall platform ramp corner rail
tower bridge gap ring coin item health boost lane track road bend obstacle target gem orb
ball's opponent puck flag base tower checkpoint slope""".split())

PROMPTS = {
    # P0: current/baseline (tends directive)
    "baseline": (
        "You are a gameplay analyst. Given frames (~{secs:.1f}s), the gamepad inputs the "
        "player made, prior plans, and the streamer's words, output ONE short imperative "
        "plan (max 14 words) for the player's immediate goal, grounded in what they did. "
        "Output ONLY the plan."),
    # P1: explicitly goal-level, NO controller words (the design target)
    "tactical": (
        "You are a gameplay strategist. You see frames spanning ~{secs:.1f}s, a summary of "
        "what the player did (for YOUR understanding only), prior plans, and the streamer's "
        "words. Output ONE short tactical PLAN (max 12 words) stating the player's immediate "
        "GOAL in game terms. Describe INTENT, never controls — do NOT mention buttons, "
        "sticks, or directions like 'press B' or 'move left'. Say what to ACHIEVE (e.g. "
        "'intercept the ball before it reaches the net', 'flank the boss and break its "
        "guard'). Output ONLY the plan."),
    # P2: situation + intent, two clauses
    "situation": (
        "Watch ~{secs:.1f}s of gameplay (frames + the player's actual inputs for context + "
        "prior plans). In ONE sentence (max 14 words) describe the tactical SITUATION and the "
        "player's GOAL, in game terms, without naming any controller input. Example: 'ball "
        "loose at midfield — win possession and push toward goal'. Output ONLY the sentence."),
}


def load_vlm(name):
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def gen(proc, model, frames, sys_prompt, ctx):
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": sys_prompt + "\n\n" + ctx})
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


def score(plan):
    toks = re.findall(r"[a-z']+", plan.lower())
    direct = sum(t in DIRECTIVE for t in toks)
    spec = sum(t in OBJECTS for t in toks)
    return direct, spec, len(toks)


def load_windows():
    out = []
    for wdir in sorted(glob.glob(os.path.join(WIN, "*"))):
        mp = os.path.join(wdir, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        frames = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(wdir, "f*.png")))]
        out.append((meta, frames))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompts", default="baseline,tactical,situation")
    ap.add_argument("--n-windows", type=int, default=8)
    args = ap.parse_args()
    proc, model = load_vlm(args.model)
    wins = load_windows()[:args.n_windows]
    prompt_keys = args.prompts.split(",")
    agg = {k: [0, 0, 0] for k in prompt_keys}
    print(f"model={args.model}  prompts={prompt_keys}  windows={len(wins)}\n")
    for meta, frames in wins:
        secs = meta["A"] * 18 * 2 / 60.0
        ctx = [f"Game: {meta['game']}", "Player's actual inputs over this window (context only):"]
        for i, sm in enumerate(meta["action_summaries"]):
            ctx.append(f"  chunk {i+1}: {sm}")
        tw = meta["transcript"][:160]
        ctx.append("Streamer said: " + (tw or "(none)"))
        ctxs = "\n".join(ctx)
        print(f"#### {meta['game']}  {meta['wid']}")
        print(f"     actions: {meta['action_summaries'][0]}")
        for k in prompt_keys:
            plan = gen(proc, model, frames, PROMPTS[k].format(secs=secs), ctxs)
            d, s, n = score(plan)
            for j, v in enumerate((d, s, n)):
                agg[k][j] += v
            print(f"     [{k:9s}] (direct={d} spec={s}) {plan}")
        print(flush=True)
    print("=== aggregate (lower direct = better, higher spec = better) ===")
    n = len(wins)
    for k in prompt_keys:
        d, s, ln = agg[k]
        print(f"  {k:9s}: avg_directive={d/n:.2f}  avg_specificity={s/n:.2f}  avg_len={ln/n:.1f}")


if __name__ == "__main__":
    main()
