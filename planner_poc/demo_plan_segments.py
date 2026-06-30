"""demo_plan_segments.py -- split human gold demos into A=2-chunk SEGMENTS and emit, per segment, the
interleaved frames + per-subchunk action summaries needed for a strong model (GPT-5.5) to AUTHOR a
grounded "what the player did/intended" plan (privileged: it sees the actions taken).

Output per game under docs/demo_plans/<game>/:
  frames/<seg_id>__f{c}_{s}.png         f0 + the frame AFTER each subchunk (interleaved obs)
  segments.jsonl                        one row per segment: {seg_id, demo, start, frames[], subchunks[
                                          {label, action_summary}], game_context, screen_x0 (if any)}
Then plan-authoring subagents read segments.jsonl + frames -> write a plan per seg_id (demo_train_stack
consumes them as gold plan labels).

NitroGen cadence: 1 step = `stride`(2) emu frames; chunk = H(18) steps = 36 emu; segment = A(2) chunks =
72 emu; subchunk = H//S (6) steps = 12 emu; S(3) subchunks/chunk -> 6 subchunk-frames + f0 per segment.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc'
  $RUN .venv/bin/python planner_poc/demo_plan_segments.py --game smw --max-seg-per-demo 24
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

# SNES 12-button order in the demos: B,Y,SELECT,START,UP,DOWN,LEFT,RIGHT,A,X,L,R
SNES12 = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
# GBA actual demo order (verified meta.json): B,-,SELECT,START,UP,DOWN,LEFT,RIGHT,A,-,L,R
GBA12 = ["B", "-", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "-", "L", "R"]

GAME = {
    "smw": dict(dir="SuperMarioWorld-Snes", buttons=SNES12,
                context="Super Mario World (SNES), a side platformer. Goal: advance RIGHT to the level goal. "
                        "B=jump, Y=run/dash, A=spin-jump."),
    "sonic": dict(dir="SonicTheHedgehog2-Genesis", buttons=["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"],
                  context="Sonic the Hedgehog 2 (Genesis), a fast side platformer. Goal: advance RIGHT. "
                          "A/B/C=jump; hold DOWN while moving=spin; DOWN+jump=spin dash."),
    "minish": dict(dir="LegendOfZeldaTheMinishCap-GbAdvance", buttons=GBA12,
                   context="Zelda: The Minish Cap (GBA), top-down. Goal: explore, move toward exits/doors in "
                           "UP/DOWN/LEFT/RIGHT. A=action/attack, B=item."),
    "fireemblem": dict(dir="FireEmblemTheSacredStones-GbAdvance", buttons=GBA12,
                       context="Fire Emblem: Sacred Stones (GBA), turn-based tactics. Cursor moves UP/DOWN/"
                               "LEFT/RIGHT, A=select/confirm, B=cancel."),
}


def summarize_subchunk(acts: np.ndarray, buttons: list[str]) -> str:
    """acts: (T,12) binary over the subchunk. Grounded summary: dominant direction + buttons held."""
    n = max(1, len(acts))
    held = {b: float(acts[:, i].mean()) for i, b in enumerate(buttons)}
    parts = []
    # direction from d-pad
    dirs = []
    for d in ("LEFT", "RIGHT", "UP", "DOWN"):
        if held.get(d, 0) > 0.25:
            dirs.append(f"{d}{'' if held[d] > 0.7 else ' briefly'}")
    parts.append("move " + " & ".join(dirs) if dirs else "no movement")
    # action buttons (exclude d-pad/select/start)
    btns = []
    for b in buttons:
        if b in ("LEFT", "RIGHT", "UP", "DOWN", "SELECT", "START", "MODE", "?"):
            continue
        if held.get(b, 0) > 0.15:
            when = "held" if held[b] > 0.6 else ("tapped" if held[b] < 0.35 else "pressed")
            btns.append(f"{b} {when}")
    parts.append("buttons: " + ", ".join(btns) if btns else "no buttons")
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="smw", choices=sorted(GAME))
    ap.add_argument("--demos-root", default="docs/demos/demos")
    ap.add_argument("--out-root", default="docs/demo_plans")
    ap.add_argument("--H", type=int, default=18)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--S", type=int, default=3, help="subchunks per chunk")
    ap.add_argument("--max-seg-per-demo", type=int, default=24)
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()

    g = GAME[args.game]
    seg_steps = args.A * args.H                 # NitroGen steps per segment
    seg_emu = seg_steps * args.stride           # emu frames per segment
    sub_steps = args.H // args.S                 # NitroGen steps per subchunk
    sub_emu = sub_steps * args.stride            # emu frames per subchunk
    n_sub = args.A * args.S                       # subchunk frames per segment

    out = os.path.join(args.out_root, args.game)
    fdir = os.path.join(out, "frames"); os.makedirs(fdir, exist_ok=True)
    rows = []
    demos = sorted(glob.glob(os.path.join(args.demos_root, g["dir"], "*")))
    for d in demos:
        npz = os.path.join(d, "demo.npz")
        if not os.path.exists(npz):
            continue
        z = np.load(npz, allow_pickle=True)
        obs, acts = z["observations"], z["actions"]   # (N+1,h,w,3), (N,12)
        demo_id = os.path.basename(d.rstrip("/"))
        nseg = 0
        start = 0
        while start + seg_emu <= len(acts) and nseg < args.max_seg_per_demo:
            seg_id = f"{demo_id}_s{start:05d}"
            frames_meta, subs = [], []

            def _save(emu_idx, tag):
                im = Image.fromarray(obs[min(emu_idx, len(obs) - 1)]).convert("RGB")
                if max(im.size) > args.res:
                    im.thumbnail((args.res, args.res))
                p = os.path.join(fdir, f"{seg_id}__{tag}.png"); im.save(p)
                frames_meta.append({"tag": tag, "path": os.path.relpath(p, out)})

            _save(start, "f0")
            for j in range(n_sub):
                ci, si = j // args.S + 1, j % args.S + 1
                lo = start + j * sub_emu; hi = lo + sub_emu
                subs.append({"label": f"s{ci}_{si}", "action_summary": summarize_subchunk(acts[lo:hi], g["buttons"])})
                _save(hi, f"f{ci}_{si}")
            rows.append({"seg_id": seg_id, "demo": demo_id, "game": args.game, "start_emu": int(start),
                         "seg_emu": int(seg_emu), "stride": args.stride, "H": args.H, "A": args.A, "S": args.S,
                         "game_context": g["context"], "frames": frames_meta, "subchunks": subs})
            start += seg_emu; nseg += 1
    with open(os.path.join(out, "segments.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[demo-segments] {args.game}: {len(rows)} segments from {len(demos)} demos -> {out}/segments.jsonl")
    print(f"  per segment: 1 + {n_sub} frames, {n_sub} subchunk summaries; seg={seg_steps} steps ({seg_emu} emu)")
    if rows:
        ex = rows[0]
        print(f"  example {ex['seg_id']}: " + " | ".join(s["label"] + ":" + s["action_summary"] for s in ex["subchunks"][:3]))


if __name__ == "__main__":
    main()
