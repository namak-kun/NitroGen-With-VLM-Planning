"""Summarize a real NitroGen action chunk as natural-language CONTEXT for the VLM. The VLM
cannot infer the gamepad inputs from pixels (EXP-036/037), so we TELL it what the player
actually did and let it explain the intent in game terms — grounding the plan in real
actions by construction.

Action layout (per timestep): buttons[0:21], j_left[21:23] (x,y in [-1,1]; -x=left,
-y=up), j_right[23:25]. Chunk = 18 steps.
"""
import numpy as np
from nitrogen.training.actions import BUTTON_ORDER

# human-readable names for the buttons that matter for play (Xbox-ish gamepad)
BTN_LABEL = {
    "south": "A/jump", "east": "B", "west": "X/attack", "north": "Y",
    "left_shoulder": "LB", "right_shoulder": "RB",
    "left_trigger": "LT/brake", "right_trigger": "RT/accelerate",
    "left_thumb": "L3", "right_thumb": "R3", "start": "start", "back": "back",
    "dpad_up": "dpad-up", "dpad_down": "dpad-down", "dpad_left": "dpad-left", "dpad_right": "dpad-right",
}
_IDX = {n: i for i, n in enumerate(BUTTON_ORDER)}


def _stick_dir(xy, eps=0.25):
    x, y = float(xy[0]), float(xy[1])
    if max(abs(x), abs(y)) < eps:
        return "neutral"
    parts = []
    if y < -eps: parts.append("up")
    if y > eps: parts.append("down")
    if x < -eps: parts.append("left")
    if x > eps: parts.append("right")
    return "-".join(parts) if parts else "neutral"


def summarize_chunk(chunk: dict, n_seg: int = 3) -> str:
    """Return a terse description of the 18-step chunk: left-stick motion over n_seg
    sub-segments + any buttons pressed (with how long)."""
    H = chunk["buttons"].shape[0]
    jl = chunk["j_left"]; btn = chunk["buttons"]
    seg = max(1, H // n_seg)
    parts = []
    # left-stick (movement) per segment
    seg_descs = []
    for k in range(n_seg):
        lo, hi = k * seg, (H if k == n_seg - 1 else (k + 1) * seg)
        d = _stick_dir(jl[lo:hi].mean(0))
        seg_descs.append(d)
    # collapse consecutive identical
    collapsed = []
    for d in seg_descs:
        if not collapsed or collapsed[-1] != d:
            collapsed.append(d)
    if all(d == "neutral" for d in collapsed):
        parts.append("left stick: mostly neutral")
    else:
        parts.append("left stick: " + " then ".join(collapsed))
    # buttons: which were held and roughly how much
    held = []
    for name, idx in _IDX.items():
        frac = float((btn[:, idx] > 0.5).mean())
        if frac > 0.1:
            label = BTN_LABEL.get(name, name)
            when = "throughout" if frac > 0.7 else ("briefly" if frac < 0.35 else "for a while")
            held.append(f"{label} {when}")
    if held:
        parts.append("buttons: " + ", ".join(held))
    else:
        parts.append("no buttons")
    return "; ".join(parts)


if __name__ == "__main__":
    import sys, glob, os, json
    from nitrogen.training.actions import load_chunk_actions, assemble_chunk
    dirs = sorted(os.path.dirname(md) for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))
    for d in dirs[:8]:
        m = json.load(open(os.path.join(d, "metadata.json")))
        pq = os.path.join(d, "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(d, "actions_raw.parquet")
        a = load_chunk_actions(pq)
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 303, 18, 2)
        if rc:
            print(f"[{m.get('game','?')[:18]:18s}] {summarize_chunk(rc)}")
