"""Sweep several GROUNDING-INSTRUCTION phrasings for the frozen System-2 planner and score which gives
the cleanest grounded broad plan (short, names a concrete control direction, no frame narration). Runs
each candidate in the full context the live rollout will use: +controls (per genre) + the agent's own
recent ACTIONS (no privileged state). Pick the winner for the video rollouts.

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B \
        .venv/bin/python planner_poc/sweep_prompts.py
"""
import re
import sys

import torch

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import benchmark_prompts as B
from nitrogen.planner import PlanEncoder, PlannerConfig

# candidate grounding instructions (appended after the action history)
CANDS = {
    "A_broad_concrete": (
        "Give just your one-sentence recommendation for what the agent should broadly do next (no "
        "frame-by-frame description). Use concrete control or game-specific terms so the direction is "
        "clear -- e.g. 'move RIGHT', 'go UP', 'JUMP', 'ATTACK' -- not vague words like 'forward'."),
    "B_oneline_dirs": (
        "In one sentence, what should the agent broadly do next? Use the game's control directions "
        "(left, right, up, down, jump, attack) so the move is unambiguous; do not describe the frames."),
    "C_short_verb": (
        "Answer in one short sentence starting with the action. Say the move using control words "
        "(e.g. move right, go up, jump) -- not vague words like 'forward' or 'explore'."),
    "D_tell_player": (
        "Briefly, in one line, tell the player what to do next using concrete control terms (move "
        "left/right, go up/down, jump, attack). No frame description."),
    "E_intent_grounded": (
        "State the agent's next intent in one short clause, grounded in a concrete direction or button "
        "(e.g. 'head RIGHT to the door', 'JUMP the gap'), not vague words like 'forward'."),
    "F_minimal": (
        "One short sentence: what should the agent do next? Use a concrete direction (left/right/up/"
        "down) or action (jump/attack), not 'forward' or 'explore'."),
}

CTRL_WORD = re.compile(r"\b(LEFT|RIGHT|UP|DOWN|JUMP|ATTACK|SWORD|ACTION|ACCELERATE|BRAKE|head|climb)\b", re.I)
VAGUE = re.compile(r"\b(forward|explore|advance|proceed|move on)\b", re.I)
RAMBLE = re.compile(r"^\s*(based on|of course|here(?:'s| is)|let'?s|the analysis|sure|certainly|to "
                    r"determine|looking at|in the (?:image|frame))", re.I)


def score(plan):
    words = len(plan.split())
    grounded = bool(CTRL_WORD.search(plan))
    vague = bool(VAGUE.search(plan) and not grounded)
    rambly = bool(RAMBLE.search(plan)) or words > 18
    clean = grounded and not vague and not rambly
    return clean, grounded, vague, rambly, words


def main():
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=B.QWEN)); pl.load()
    dev = "cuda"
    if next(pl.backbone.parameters()).device != torch.device(dev):
        pl.backbone.to(dev)
    totals = {k: 0 for k in CANDS}
    for cand, instr_tail in CANDS.items():
        print(f"\n################ CANDIDATE {cand} ################")
        for name, genre, globs, history, typ in B.CASES:
            frames = B.imgs(globs)
            sys_p = B.BASE_SYS + " " + B.CONTROLS[genre]
            instr = f"{history}\n\n{instr_tail}"
            plan = pl.generate_plan(frames, dev, instruction=instr, system=sys_p, max_new_tokens=40)
            clean, g, v, r, w = score(plan)
            totals[cand] += clean
            tag = ("CLEAN" if clean else "    -") + (" G" if g else "  ") + (" v" if v else "  ") + \
                  (" R" if r else "  ")
            print(f"  [{tag}] {name:22} ({w:2d}w)  {plan[:90]!r}", flush=True)
    print("\n===== SCORE (clean plans out of 4 scenarios) =====")
    for cand, sc in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {cand:18} {sc}/4")
    best = max(totals, key=totals.get)
    print(f"\nBEST: {best}  ->  {CANDS[best]!r}")


if __name__ == "__main__":
    main()
