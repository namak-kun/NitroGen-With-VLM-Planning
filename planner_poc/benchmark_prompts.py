"""PROMPT bench for the (frozen) System-2 planner: most of the plan-quality gaps the rollouts exposed
are INPUT/PROMPT problems, not training problems -- so test them by PROMPTING alone, no GPU training.
We vary three orthogonal prompt knobs on real game frames and read the generated plans:

  +CONTROLS : tell the planner the control scheme (LEFT/RIGHT/UP/DOWN/JUMP/ATTACK/...) so it can ground
              the plan in actual controls instead of vague words ("move forward").
  +HISTORY  : dump the agent's OWN ACTIONS taken between the recent frames ("held RIGHT 16/18 steps")
              so that -- together with frames that show no progress -- it can notice it is stuck/looping.
              IMPORTANT: actions only; NO privileged game state (x/y/lives) -- the real rollout harness
              only feeds the planner frames + objective, so the bench must not leak privileged state.
  +GROUNDED : a light nudge to use concrete control/game-specific terms so the direction is unambiguous
              ("move RIGHT into the village" not "move forward to explore"). NOT a rigid output format.

Scenarios (real frames from the rollouts): a platformer progressing case, a platformer BLOCKED case, a
top-down BLOCKED case, and a racing STRAIGHTAWAY.

Metrics (keyword/regex, plus the raw plans to read):
  grounded     = names an explicit control direction (not just "forward/explore/advance").
  vague        = uses a vague word and no concrete control direction.
  blocked_aware= on BLOCKED cases: switches to a different control (JUMP/UP/turn) or names being stuck.

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B \
        .venv/bin/python planner_poc/benchmark_prompts.py
"""
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.planner import PlanEncoder, PlannerConfig

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")
BF = REPO + "/planner_poc/bench_frames"

CONTROLS = {
    "platformer": "Controls: move LEFT or RIGHT, look UP, crouch DOWN, JUMP, and ATTACK. To reach a "
                  "higher ledge you must JUMP (moving sideways alone will not climb).",
    "topdown": "Controls: move LEFT, RIGHT, UP or DOWN, SWORD to attack, ACTION to interact.",
    "racing": "Controls: steer LEFT or RIGHT, ACCELERATE, BRAKE.",
}
# NOTE: the action-history dump contains the AGENT'S OWN ACTIONS only (what it pressed) -- never
# privileged game state (x/y/lives/dead). The real rollout harness only feeds the planner frames +
# objective, so privileged state must not leak into the bench either.

# scenario = (name, genre, frame globs, action-history dump (AGENT'S OWN ACTIONS ONLY -- no privileged
# game state like x/y/lives), scenario type)
CASES = [
    ("platformer_progress", "platformer", ["tx_prog_1.png", "tx_prog_2.png", "tx_prog_3.png"],
     "Actions taken between these frames: moved RIGHT for 12 of the last 18 steps, JUMP 6 times.",
     "normal"),
    # REAL stuck moment from the thextech_A2 rollout (frames 22-24): ground-truth state showed the
    # character jammed at the base of the central pillar -- held RIGHT (stick ~+1.2) but x frozen at
    # 424.0, velocity ~0. The right move is to JUMP onto the platform, not keep pushing RIGHT.
    ("platformer_BLOCKED", "platformer", ["tx_blocked_1.png", "tx_blocked_2.png", "tx_blocked_3.png"],
     "Actions taken between these frames: held RIGHT for most of the last 18 steps. The frames look the "
     "same -- the character has not moved.", "blocked"),
    ("topdown_BLOCKED", "topdown", ["sol_1.png", "sol_2.png", "sol_3.png"],
     "Actions taken between these frames: held DOWN for 15 of the last 18 steps.", "blocked"),
    ("racing_STRAIGHT", "racing", ["stk_clean_1.png", "stk_clean_2.png"],
     "Actions taken between these frames: ACCELERATE every step, steering near-centre.", "normal"),
]

BASE_SYS = ("You are the high-level planner for an agent playing a 2D video game. You see the most "
            "recent frames (oldest first). Decide the single best next move.")
# Lightweight GROUNDING nudge (not a rigid output format): keep the plan a normal short imperative,
# but make the DIRECTION unambiguous by using concrete control / game-specific terms instead of vague
# words. This is what turns "move forward to explore" into "move RIGHT into the village".
GROUND_INSTR = ("Give just your one-sentence recommendation for what the agent should broadly do next "
                "(no frame-by-frame description). Use concrete control or game-specific terms so the "
                "direction is clear -- e.g. 'move RIGHT', 'go UP', 'JUMP', 'ATTACK' -- not vague words "
                "like 'forward' or 'explore'.")

CONFIGS = ["C0_base", "C1_controls", "C2_history", "C3_grounded", "C4_all"]
GROUND_RE = re.compile(r"\b(LEFT|RIGHT|UP|DOWN|JUMP|ATTACK|SWORD|ACTION|ACCELERATE|BRAKE)\b", re.I)
VAGUE_RE = re.compile(r"\b(forward|explore|advance|proceed|continue exploring|move on|the area|the image)\b", re.I)
BLOCK_RE = re.compile(r"\b(stuck|block|wall|different|instead|another|can't|cannot|turn around|back|try)\b", re.I)


def imgs(globs):
    out = []
    for g in globs:
        p = os.path.join(BF, g)
        if os.path.exists(p):
            out.append(np.asarray(Image.open(p).convert("RGB")))
    return out


def build_prompt(cfg, genre, history):
    sys_p = BASE_SYS
    instr = "What should the agent do next?"
    if cfg in ("C1_controls", "C4_all"):
        sys_p += " " + CONTROLS[genre]
    if cfg in ("C2_history", "C4_all"):
        instr = f"{history}\n\nGiven that, what should the agent do next?"
    if cfg in ("C3_grounded", "C4_all"):
        instr = instr + " " + GROUND_INSTR
    return sys_p, instr


def main():
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load()
    dev = "cuda"
    if next(pl.backbone.parameters()).device != torch.device(dev):
        pl.backbone.to(dev)
    tallies = {c: {"grounded": 0, "vague": 0, "blocked_aware": 0, "n": 0} for c in CONFIGS}
    for name, genre, globs, history, typ in CASES:
        frames = imgs(globs)
        print(f"\n===== {name}  ({genre}, type={typ}, {len(frames)} frames) =====")
        for cfg in CONFIGS:
            sys_p, instr = build_prompt(cfg, genre, history)
            plan = pl.generate_plan(frames, dev, instruction=instr, system=sys_p,
                                    max_new_tokens=40)
            g = bool(GROUND_RE.search(plan)); v = bool(VAGUE_RE.search(plan) and not g)
            # BLOCKED recovery: switches to a productive DIFFERENT control (JUMP/UP/turn) or names being
            # stuck -- i.e. does NOT just keep pushing the stuck direction (RIGHT/DOWN).
            switched = bool(re.search(r"\b(JUMP|UP|LEFT|ATTACK|SWORD|ACTION|turn|back)\b", plan, re.I))
            ba = bool(BLOCK_RE.search(plan)) or (g and switched
                  and not re.search(r"\b(RIGHT|DOWN)\b.{0,12}$", plan, re.I))
            t = tallies[cfg]; t["n"] += 1
            t["grounded"] += g; t["vague"] += v
            if typ == "blocked":
                t["blocked_aware"] += ba
            flags = ("G" if g else "-") + ("v" if v else "-") + \
                    ("B" if (typ == "blocked" and ba) else "-")
            print(f"  {cfg:12} [{flags}] {plan[:96]!r}", flush=True)
    print("\n===== SUMMARY (per config, across cases) =====")
    print(f"{'config':12} {'grounded':>9} {'vague':>6} {'blk-aware(/2)':>13}")
    for c in CONFIGS:
        t = tallies[c]
        print(f"{c:12} {t['grounded']}/{t['n']:>7} {t['vague']:>6} {t['blocked_aware']:>11}")
    print("\nRead the plans: does +controls/+format make the plan GROUNDED in a concrete direction "
          "(G, fewer vague 'forward'/'explore' = v)? does +history make BLOCKED cases switch to a "
          "different control (B)? This tells us how far pure prompting/input-format gets us (the action "
          "dump is the agent's OWN actions -- no privileged game state like x/y/lives).")


if __name__ == "__main__":
    main()
