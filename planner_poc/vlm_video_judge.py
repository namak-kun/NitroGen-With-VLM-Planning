"""vlm_video_judge.py — INDEPENDENT, RAM-FREE verification of a rollout by having a VLM WATCH the frames.
Takes a rollout's evenly-sampled frames (the .frames.npz written by furthest_rollout.py) and asks a local VLM
to describe, purely from the visuals, how far the player progressed, whether they died/stalled/reached a goal,
and notable events. A COMPARE mode shows two runs' frames and asks which progressed further (the base-vs-model
A/B the user needs since they cannot watch the videos themselves). RAM can be sketchy; the video is ground truth.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=google/gemma-4-12B-it'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/vlm_video_judge.py describe \
     --frames docs/furthest/smw/pooled__state0.frames.npz --game smw
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/vlm_video_judge.py compare \
     --a docs/furthest/smw/base__state0.frames.npz --b docs/furthest/smw/pooled__state0.frames.npz --game smw
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
import torch
from nitrogen.planner import PlanEncoder, PlannerConfig

QWEN = os.environ.get("QWEN", "google/gemma-4-12B-it")

GAME_NAME = {"smw": "Super Mario World", "smbas": "Super Mario Bros (All-Stars)",
             "mmx": "Mega Man X", "sonic": "Sonic the Hedgehog 2", "metroid": "Super Metroid"}

SYS = ("You are a meticulous game-analysis judge. You are shown evenly-spaced frames (with timestamps) from ONE "
       "automated gameplay attempt, in time order. Judge ONLY from what you see — do not assume success.")

DESCRIBE = (
    "These frames are one attempt at playing {game}, oldest to newest, ~{span}s total.\n"
    "Answer concisely:\n"
    "1) PROGRESS: how far did the character get? Did the level/background visibly scroll/advance, or stay "
    "near the start / loop / get stuck? Name visual landmarks reached (pipes, hills, a castle, a goal post, "
    "new areas).\n"
    "2) OUTCOME: did the character DIE, get STUCK/idle, wander, or REACH a goal/level-end? At roughly which "
    "timestamp?\n"
    "3) RATING: rate overall progress 0-10 (0=did nothing/died instantly, 10=beat the level), one number.\n"
    "Format: 'PROGRESS: ... | OUTCOME: ... | RATING: N'."
)

COMPARE = (
    "You are shown TWO attempts at {game}: first RUN A frames, then RUN B frames (each in time order). "
    "Judge ONLY from visuals which attempt progressed FURTHER through the level (more level scrolled past / "
    "reached later areas / survived longer to advance). Cite the visual evidence (e.g. 'B reached a pipe/"
    "castle A never did', 'A's background barely changed'). Then give a verdict.\n"
    "Format: 'WINNER: A|B|tie | WHY: ... | A_progress: 0-10 | B_progress: 0-10'."
)


def _load(pl, dev):
    pl.load()
    if next(pl.backbone.parameters()).device != torch.device(dev):
        pl.backbone.to(dev)
    return pl


def _frames(npz):
    d = np.load(npz)
    return list(d["frames"]), list(d["t"])


@torch.no_grad()
def describe(pl, dev, frames, ts, game, max_new=220):
    items = []
    for f, t in zip(frames, ts):
        items.append(f)
        items.append(f"[t={t:.1f}s]")
    span = ts[-1] if ts else 0
    instr = DESCRIBE.format(game=GAME_NAME.get(game, game), span=f"{span:.0f}")
    return pl.generate_interleaved(items, dev, instruction=instr, system=SYS,
                                   max_new_tokens=max_new, enable_thinking=False)


@torch.no_grad()
def compare(pl, dev, fa, ta, fb, tb, game, max_new=240):
    items = ["=== RUN A ==="]
    for f, t in zip(fa, ta):
        items += [f, f"[A t={t:.1f}s]"]
    items.append("=== RUN B ===")
    for f, t in zip(fb, tb):
        items += [f, f"[B t={t:.1f}s]"]
    instr = COMPARE.format(game=GAME_NAME.get(game, game))
    return pl.generate_interleaved(items, dev, instruction=instr, system=SYS,
                                   max_new_tokens=max_new, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pd = sub.add_parser("describe"); pd.add_argument("--frames", required=True); pd.add_argument("--game", default="smw")
    pd.add_argument("--max-frames", type=int, default=16); pd.add_argument("--out", default=None)
    pc = sub.add_parser("compare"); pc.add_argument("--a", required=True); pc.add_argument("--b", required=True)
    pc.add_argument("--game", default="smw"); pc.add_argument("--max-frames", type=int, default=12); pc.add_argument("--out", default=None)
    for p in (pd, pc):
        p.add_argument("--qwen", default=QWEN)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pl = _load(PlanEncoder(PlannerConfig(backbone_name_or_path=args.qwen)), dev)

    def cap(frames, ts, n):
        if len(frames) <= n:
            return frames, ts
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in idx], [ts[i] for i in idx]

    if args.cmd == "describe":
        frames, ts = _frames(args.frames); frames, ts = cap(frames, ts, args.max_frames)
        verdict = describe(pl, dev, frames, ts, args.game)
        print(f"[judge:{os.path.basename(args.frames)}]\n{verdict}", flush=True)
        out = args.out or args.frames.replace(".frames.npz", ".judge.json")
        json.dump({"frames": args.frames, "qwen": args.qwen, "verdict": verdict}, open(out, "w"), indent=1)
        print(f"-> {out}", flush=True)
    else:
        fa, ta = _frames(args.a); fa, ta = cap(fa, ta, args.max_frames)
        fb, tb = _frames(args.b); fb, tb = cap(fb, tb, args.max_frames)
        verdict = compare(pl, dev, fa, ta, fb, tb, args.game)
        print(f"[compare A={os.path.basename(args.a)} vs B={os.path.basename(args.b)}]\n{verdict}", flush=True)
        out = args.out or os.path.join(os.path.dirname(args.a), "compare.judge.json")
        json.dump({"a": args.a, "b": args.b, "qwen": args.qwen, "verdict": verdict}, open(out, "w"), indent=1)
        print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
