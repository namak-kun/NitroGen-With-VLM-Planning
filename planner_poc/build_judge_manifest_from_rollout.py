"""build_judge_manifest_from_rollout.py -- turn an existing annotated rollout (docs/horizon_play*/<env>/
play.mp4 + actions.txt) into a bench judgment manifest, so the 3 judges can score REAL btn_s600 gameplay
WITHOUT re-running any game env (all proc envs are uninstalled on this box).

Crops the 120px annotation bar off the bottom of each frame (so judges can't read the plan/action off the
overlay) and parses actions.txt for per-cycle plan + executed stick/buttons + GROUND-TRUTH state. The GT
state (where present: thextech x/y/lives, solarus x/y/life/map) is stored AND reduced to an objective
progress anchor (gt) so we can validate each judge's progress rating against reality.

Run: .venv/bin/python planner_poc/build_judge_manifest_from_rollout.py \
        --rollout-dir docs/horizon_play_grounded/thextech_A2 --out docs/bench/thextech_A2
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

BAR_H = 120  # play_annotated_horizon.annotate() appends a fixed 120px overlay band at the bottom

JLX, JLY = 21, 22
_BTN = [(18, "A/jump"), (20, "X/shoot"), (5, "B"), (10, "Y"), (16, "RT/accel"), (9, "LT/brake"),
        (1, "down"), (2, "left"), (3, "right"), (4, "up")]

GENRE = {"thextech": "platformer", "sdlpop": "platformer", "solarus_zelda": "topdown",
         "stk": "racing", "blobwars": "platformer", "witchblast": "topdown",
         "chromium_bsu": "shmup"}

CONTROLS = {
    "racing": "Controls: steer LEFT or RIGHT, ACCELERATE, BRAKE.",
    "platformer": "Controls: move LEFT or RIGHT, look UP, crouch DOWN, JUMP, ATTACK.",
    "topdown": "Controls: move LEFT, RIGHT, UP or DOWN, ATTACK, ACTION to interact.",
    "shmup": "Controls: move LEFT, RIGHT, UP or DOWN, FIRE.",
}

CYCLE_RE = re.compile(r"^#\s*re-plan#(\d+)\s+plan='(.*)'\s*$")
EXEC_RE = re.compile(r"^\s*exec(\d+):\s*stick=\(([-+\d.]+),([-+\d.]+)\)\s*buttons=(\S+)\s*(?:state=(\{.*\}))?\s*$")


def parse_actions(path):
    header = {}
    cycles = []
    cur = None
    with open(path) as f:
        for line in f:
            if line.startswith("env="):
                m = re.search(r"env=(\S+).*A=(\d+)", line)
                if m:
                    header["env"] = m.group(1); header["A"] = int(m.group(2))
            if line.startswith("per_row=") and "objective=" in line:
                header["objective"] = line.split("objective=", 1)[1].strip()
            mc = CYCLE_RE.match(line)
            if mc:
                cur = {"i": int(mc.group(1)), "plan": mc.group(2), "rows": []}
                cycles.append(cur)
                continue
            me = EXEC_RE.match(line)
            if me and cur is not None:
                stick = (float(me.group(2)), float(me.group(3)))
                btns = [] if me.group(4) == "-" else me.group(4).split(",")
                st = {}
                if me.group(5):
                    try:
                        st = ast.literal_eval(me.group(5))
                    except Exception:
                        st = {}
                cur["rows"].append({"stick": stick, "buttons": btns, "state": st})
    return header, cycles


def summarize_rows(rows):
    arr = np.array([[r["stick"][0], r["stick"][1]] for r in rows], dtype=np.float32)
    mx, my = float(arr[:, 0].mean()), float(arr[:, 1].mean())
    th = 0.15
    xs = "RIGHT" if mx > 0.5 + th else ("LEFT" if mx < 0.5 - th else "x-neutral")
    ys = "DOWN" if my > 0.5 + th else ("UP" if my < 0.5 - th else "y-neutral")
    btn_ct = {}
    for r in rows:
        for b in r["buttons"]:
            btn_ct[b] = btn_ct.get(b, 0) + 1
    held = [b for b, c in btn_ct.items() if c >= max(1, len(rows) // 2)]
    return f"stick mostly {xs}/{ys}; buttons {','.join(held) if held else 'none'}", {"jlx": round(mx, 2), "jly": round(my, 2)}, held


def gt_anchor(env, sb, sa):
    """Reduce before/after GT state to an objective progress signal where available."""
    g = {}
    if "x" in sb and "x" in sa:
        g["dx"] = round(float(sa["x"]) - float(sb["x"]), 1)
        g["dpos"] = round(((float(sa.get("x", 0)) - float(sb.get("x", 0))) ** 2 +
                           (float(sa.get("y", 0)) - float(sb.get("y", 0))) ** 2) ** 0.5, 1)
    for k in ("life", "lives"):
        if k in sb and k in sa:
            g[f"d{k}"] = float(sa[k]) - float(sb[k])
    for k in ("dead", "in_menu", "won"):
        if k in sa:
            g[k] = int(bool(sa[k]))
    if "map" in sb and "map" in sa:
        g["map_changed"] = int(sb["map"] != sa["map"])
    # objective-aware scalar progress: platformers/racing advance RIGHT -> sign(dx); topdown -> moved
    if env in ("thextech", "sdlpop", "stk") and "dx" in g:
        g["progress_gt"] = 1 if g["dx"] > 5 else (-1 if g["dx"] < -5 else 0)
    elif "dpos" in g:
        g["progress_gt"] = 1 if g["dpos"] > 8 else 0
    if g.get("dead") or g.get("in_menu"):
        g["progress_gt"] = -1
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-cycles", type=int, default=24, help="cap cycles judged (cost control)")
    args = ap.parse_args()

    rd = args.rollout_dir.rstrip("/")
    env = os.path.basename(rd).split("_A")[0]
    out = args.out or os.path.join("docs/bench", os.path.basename(rd))
    fdir = os.path.join(out, "frames"); os.makedirs(fdir, exist_ok=True)
    raw = os.path.join(out, "_raw"); os.makedirs(raw, exist_ok=True)

    # 1) extract frames
    subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-i", os.path.join(rd, "play.mp4"),
                    os.path.join(raw, "f%04d.png")], check=True)
    frames = sorted(glob.glob(os.path.join(raw, "f*.png")))

    # 2) parse actions
    header, cycles = parse_actions(os.path.join(rd, "actions.txt"))
    A = header.get("A", 2)
    genre = GENRE.get(env, "other")
    objective = header.get("objective", "Make progress in the game.")
    controls = CONTROLS.get(genre, "Controls: move LEFT/RIGHT/UP/DOWN + action buttons.")

    def crop_save(src, dst):
        im = Image.open(src).convert("RGB")
        im.crop((0, 0, im.width, max(1, im.height - BAR_H))).save(dst)

    man = {"env": env, "genre": genre, "objective": objective, "controls": controls,
           "ckpt": "btn_s600", "A": A, "source": rd, "cycles": []}
    row_cursor = 0
    import json
    for cyc in cycles:
        n = len(cyc["rows"]) or A
        if not cyc["rows"]:
            row_cursor += n; continue
        bi, ai = row_cursor, min(row_cursor + n - 1, len(frames) - 1)
        if ai >= len(frames):
            break
        bp = os.path.join(fdir, f"c{cyc['i']:02d}_before.png")
        apath = os.path.join(fdir, f"c{cyc['i']:02d}_after.png")
        crop_save(frames[bi], bp); crop_save(frames[ai], apath)
        sb, sa = cyc["rows"][0]["state"], cyc["rows"][-1]["state"]
        summ, stick, held = summarize_rows(cyc["rows"])
        man["cycles"].append({
            "i": cyc["i"], "plan": cyc["plan"], "action_summary": summ,
            "stick": stick, "buttons_held": held,
            "before": os.path.relpath(bp, out), "after": os.path.relpath(apath, out),
            "state_before": sb, "state_after": sa, "gt": gt_anchor(env, sb, sa),
        })
        row_cursor += n
        if len(man["cycles"]) >= args.max_cycles:
            break

    # cleanup raw extraction
    for f in frames:
        os.remove(f)
    os.rmdir(raw)

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"{env:16s} genre={genre:10s} cycles={len(man['cycles'])} (A={A}) -> {out}/manifest.json")
    gtn = sum(1 for c in man["cycles"] if "progress_gt" in c["gt"])
    print(f"  ground-truth progress anchor available on {gtn}/{len(man['cycles'])} cycles")


if __name__ == "__main__":
    main()
