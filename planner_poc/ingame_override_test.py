"""In-game OVERRIDE test: spawn Quote directly at the stuck right-corner (DRS_START_* patch) and
give a 'move left' plan. Does the model actually move Quote LEFT (against its rightward prior)?
Measures net dx over the run for each (variant, plan). The decisive in-game version of the CFG
override eval.

CONFOUND (observed 2026-06-21): keeping the new_game STORY SCRIPT (required for map rendering)
overrides the DRS_START_X/Y position (its <TRA> teleports Quote to the scripted First Cave start,
x=32) AND disables control during the intro cutscene -> Quote stays frozen (net dx 0 for all
plans). This in-game test is therefore inconclusive; use the OFFLINE eval_cfg_override.py (no
cutscene) as the clean override measurement. A proper in-game override test needs a doukutsu
patch that spawns into a stage WITHOUT the story script while still running map init/<TRA>.

Run: PYTHONPATH=. python planner_poc/ingame_override_test.py
"""
import os, sys
import numpy as np
import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
from nitrogen.eval import Scenario, JLX, JLY
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, _NAME2IDX, B_NORTH
from eval_policy import NitroGenPolicy

# start_new_game spawns directly; a few confirms clear the title/new-save prompts.
BOOT = [("wait", 1.5)] + sum(([("btn", MENU_OK), ("wait", 1.0)] for _ in range(6)), []) + [("wait", 1.5)]
GAMEPLAY_MASK = tuple(MENU_BUTTONS) + (B_NORTH,)
# spawn near the right side of First Cave (the corner Quote always drifts to)
START_STAGE, START_X, START_Y = 13, 14, 8   # tiles; x=14 is well to the right in First Cave
NSTEPS = 14

VARIANTS = {
    "text_student": ("runs/stage2_student/plan_stage1_2500.pt", False),
    "mm_lora_cf":   ("runs/stage2_student_mm_lora_cf/plan_stage1_2500.pt", True),
}
PLANS = ["move left", "move right"]


def run(ckpt, mm, plan, cfg=4.0):
    pol = NitroGenPolicy(ckpt, default_cfg=cfg); pol.mm_mode = mm
    env = CaveStoryEnv(boot_wait=9.0, freeze_during_inference=True,
                       start_stage=START_STAGE, start_pos=(START_X, START_Y))
    try:
        sc = Scenario("ov", plan=plan, objective=plan, success_spec={"reset_macro": BOOT},
                      max_steps=NSTEPS, cfg_scale=cfg)
        obs = env.reset(sc)
        x0 = obs.state.get("tile_x")
        xs = [x0]
        for t in range(NSTEPS):
            ch = pol._sample_chunk(obs.frame, plan, cfg, plan_frames=[obs.frame] if mm else None)
            for b in GAMEPLAY_MASK:
                ch[:, b] = 0.0
            obs = env.step(ch)
            xs.append(obs.state.get("tile_x"))
        xs = [x for x in xs if x is not None]
        return x0, xs[-1], (xs[-1] - x0) if (x0 is not None and xs) else None
    finally:
        env.close()
        import torch; torch.cuda.empty_cache()


def main():
    print(f"in-game override | spawn First Cave x={START_X} | {NSTEPS} steps | plan-CFG=4.0\n")
    print(f"{'variant':14} {'plan':12} {'x0':>6} {'xT':>6} {'net dx':>8}  (dx<0 = moved LEFT, obeying)")
    for vname, (ckpt, mm) in VARIANTS.items():
        if not os.path.exists(ckpt):
            print(f"{vname}: no ckpt"); continue
        for plan in PLANS:
            x0, xt, dx = run(ckpt, mm, plan)
            tag = "  <-- LEFT" if (dx is not None and dx < -0.5) else ""
            print(f"{vname:14} {plan:12} {x0!s:>6} {xt!s:>6} "
                  f"{(f'{dx:+.1f}' if dx is not None else 'NA'):>8}{tag}", flush=True)


if __name__ == "__main__":
    main()
