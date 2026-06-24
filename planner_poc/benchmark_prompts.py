"""PROMPT-FORMAT bench for the (frozen) System-2 planner: most of the plan-quality gaps the rollouts
exposed are INPUT/PROMPT problems, not training problems -- so test them by PROMPTING alone, no GPU
training. We vary three orthogonal prompt knobs on real game frames and read the generated plans:

  +CONTROLS : tell the planner the control scheme (LEFT/RIGHT/UP/DOWN/JUMP/ATTACK/...) so it grounds
              the plan in actual controls instead of vague words ("move forward").
  +HISTORY  : dump the actions taken between the recent frames + the resulting state delta
              ("held RIGHT 14/18 steps; x unchanged -> BLOCKED") so it can notice it is stuck/looping.
  +FORMAT   : instruct a structured output `PLAN: [CONTROL] <reason>` with an allowed token set that
              INCLUDES `DEFER` (let System-1 drive) -> tests grounding AND the deferral decision.

Scenarios (real frames from the rollouts): a platformer progressing case, a platformer BLOCKED case, a
top-down BLOCKED case, and a racing STRAIGHTAWAY (where the right answer is DEFER to System-1).

Metrics (keyword/regex, plus the raw plans to read):
  grounded     = names an explicit control token (not just "forward/explore/advance").
  blocked_aware= on BLOCKED cases: names stuck/blocked/try-different OR switches away from the stuck dir.
  defers       = on the DEFER case: outputs DEFER/continue/maintain/hold.

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
# allowed structured-output tokens per genre (DEFER everywhere)
TOKENS = {
    "platformer": "LEFT/RIGHT/UP/DOWN/JUMP/ATTACK/DEFER",
    "topdown": "LEFT/RIGHT/UP/DOWN/SWORD/ACTION/DEFER",
    "racing": "LEFT/RIGHT/ACCELERATE/BRAKE/DEFER",
}

# scenario = (name, genre, frame globs, action-history dump, scenario type)
CASES = [
    ("platformer_progress", "platformer", ["tx_prog_1.png", "tx_prog_2.png", "tx_prog_3.png"],
     "Last 18 control steps: RIGHT x12, JUMP x6. Position moved x 226 -> 375 (advancing right), y rose "
     "(jumped onto a ledge).", "normal"),
    ("platformer_BLOCKED", "platformer", ["tx_prog_3.png", "tx_prog_3.png", "tx_prog_3.png"],
     "Last 18 control steps: RIGHT x16 (held right), JUMP x2. Position x 472 -> 472 and y 450 -> 450: "
     "NO movement at all despite holding RIGHT -- you are stuck against something.", "blocked"),
    ("topdown_BLOCKED", "topdown", ["sol_1.png", "sol_2.png", "sol_3.png"],
     "Last 18 control steps: DOWN x15 (held down). Position barely changed: you are walking into a wall "
     "or obstacle and not making progress.", "blocked"),
    ("racing_STRAIGHT", "racing", ["stk_clean_1.png", "stk_clean_2.png"],
     "Last 18 control steps: ACCELERATE x18, steering near-centre. The kart is centred on the track on a "
     "straight section, making good progress.", "defer"),
]

BASE_SYS = ("You are the high-level planner for an agent playing a 2D video game. You see the most "
            "recent frames (oldest first). Decide the single best next move.")
FORMAT_INSTR = ("Reply with ONLY one line in this exact form: PLAN: [TOKEN] reason, where TOKEN is one "
                "or more of {toks} joined with '+', and reason is at most 8 words. Example: "
                "'PLAN: JUMP+RIGHT clear the gap'. Use DEFER ONLY if the agent is clearly making good "
                "progress; if it is stuck or not advancing, pick a DIFFERENT control (do NOT defer).")

CONFIGS = ["C0_base", "C1_controls", "C2_history", "C3_format", "C4_all"]
GROUND_RE = re.compile(r"\b(LEFT|RIGHT|UP|DOWN|JUMP|ATTACK|SWORD|ACTION|ACCELERATE|BRAKE|DEFER)\b", re.I)
VAGUE_RE = re.compile(r"\b(forward|explore|advance|proceed|continue exploring|move on|the area|the image)\b", re.I)
BLOCK_RE = re.compile(r"\b(stuck|block|wall|different|instead|another|can't|cannot|turn around|back|try)\b", re.I)
DEFER_RE = re.compile(r"\b(defer|continue|maintain|hold|keep going|stay)\b", re.I)


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
    if cfg in ("C3_format", "C4_all"):
        if cfg == "C3_format":  # format alone still needs the token list
            instr = "What should the agent do next? " + FORMAT_INSTR.format(toks=TOKENS[genre])
        else:
            instr = instr + " " + FORMAT_INSTR.format(toks=TOKENS[genre])
    return sys_p, instr


def main():
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load()
    dev = "cuda"
    if next(pl.backbone.parameters()).device != torch.device(dev):
        pl.backbone.to(dev)
    tallies = {c: {"grounded": 0, "vague": 0, "blocked_aware": 0, "defers": 0, "n": 0} for c in CONFIGS}
    for name, genre, globs, history, typ in CASES:
        frames = imgs(globs)
        print(f"\n===== {name}  ({genre}, type={typ}, {len(frames)} frames) =====")
        for cfg in CONFIGS:
            sys_p, instr = build_prompt(cfg, genre, history)
            plan = pl.generate_plan(frames, dev, instruction=instr, system=sys_p,
                                    max_new_tokens=40)
            g = bool(GROUND_RE.search(plan)); v = bool(VAGUE_RE.search(plan) and not g)
            defers_here = bool(DEFER_RE.search(plan))
            # BLOCKED recovery is correct only if it does NOT defer AND switches to a productive,
            # different control (JUMP/UP/turn) or explicitly names being stuck. Deferring while stuck
            # = FAILURE, not awareness.
            switched = bool(re.search(r"\b(JUMP|UP|LEFT|ATTACK|SWORD|ACTION|turn|back)\b", plan, re.I))
            ba = (not defers_here) and (bool(BLOCK_RE.search(plan)) or (g and switched
                  and not re.search(r"\b(RIGHT|DOWN)\b.{0,12}$", plan, re.I)))
            df = defers_here
            t = tallies[cfg]; t["n"] += 1
            t["grounded"] += g; t["vague"] += v
            if typ == "blocked":
                t["blocked_aware"] += ba
            if typ == "defer":
                t["defers"] += df
            flags = ("G" if g else "-") + ("v" if v else "-") + \
                    ("B" if (typ == "blocked" and ba) else "-") + ("D" if (typ == "defer" and df) else "-")
            print(f"  {cfg:12} [{flags}] {plan[:96]!r}", flush=True)
    print("\n===== SUMMARY (per config, across cases) =====")
    print(f"{'config':12} {'grounded':>9} {'vague':>6} {'blk-aware(/2)':>13} {'defers(/1)':>11}")
    for c in CONFIGS:
        t = tallies[c]
        print(f"{c:12} {t['grounded']}/{t['n']:>7} {t['vague']:>6} {t['blocked_aware']:>11} "
              f"{t['defers']:>11}")
    print("\nRead the plans: does +controls/+format ground the directive (G, fewer v)? does +history "
          "make BLOCKED cases switch action (B)? does the racing straight DEFER (D)? This tells us how "
          "far pure prompting/input-format gets us before any training.")


if __name__ == "__main__":
    main()
