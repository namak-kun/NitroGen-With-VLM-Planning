"""Jump-guidance test: can a PLAN guide NitroGen to jump in Cave Story (despite freeware-vs-CS+
asset mismatch)? Measures, per plan, (1) the model's jump-button (south) output rate and (2)
whether Quote ACTUALLY leaves the ground, via the doukutsu-rs player-y export (y increases
downward; a jump = y decreases). Each plan runs from a fresh First Cave start (env restart).

Run: PYTHONPATH=. python planner_poc/cavestory_jump_test.py [ckpt] [cfg] [nsteps]
"""
import os, sys, time
import numpy as np
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
import os; sys.path.insert(0, os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "planner_poc"))

from nitrogen.eval import Scenario
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, _NAME2IDX
from eval_policy import NitroGenPolicy

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
I_SOUTH = _NAME2IDX["south"]   # jump
PLANS = ["jump up as high as you can to reach the ledge above",
         "walk to the right along the ground, stay low"]


def run_plan(pol, plan, cfg, nsteps):
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    try:
        sc = Scenario("jump", plan=plan, objective="jump",
                      success_spec={"reset_macro": BOOT}, max_steps=nsteps, cfg_scale=cfg)
        obs = env.reset(sc)
        st = obs.state
        if "y" not in st:
            print("  WARNING: no state export (is the DRS_STATE patch built?)")
        rest_y = st.get("tile_y", 0.0)
        south_rates, ys = [], [rest_y]
        for t in range(nsteps):
            chunk = pol._sample_chunk(obs.frame, plan, cfg)
            for b in MENU_BUTTONS:
                chunk[:, b] = 0.0
            south_rates.append(float((chunk[:, I_SOUTH] > 0.5).mean()))
            obs = env.step(chunk)
            ys.append(obs.state.get("tile_y", rest_y))
        ys = np.array(ys)
        # y increases downward -> jump = y goes BELOW rest (smaller). jump height in tiles:
        jump_tiles = float(rest_y - ys.min())
        return {"plan": plan, "south_rate": float(np.mean(south_rates)),
                "rest_y": rest_y, "min_y": float(ys.min()), "jump_tiles": jump_tiles,
                "y_traj": [round(float(v), 2) for v in ys]}
    finally:
        env.close()


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student/plan_stage1_2500.pt"
    cfg = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    print(f"jump-guidance test | ckpt={ckpt} cfg={cfg} nsteps={nsteps}\n")
    pol = NitroGenPolicy(ckpt, default_cfg=cfg)
    results = [run_plan(pol, p, cfg, nsteps) for p in PLANS]
    print("\n=== RESULTS ===")
    for r in results:
        print(f"\nplan: '{r['plan']}'")
        print(f"  jump-button output rate : {r['south_rate']:.2f}  (fraction of steps jumping)")
        print(f"  actual jump height      : {r['jump_tiles']:+.2f} tiles "
              f"(rest_y {r['rest_y']:.1f} -> min_y {r['min_y']:.1f})")
        print(f"  y trajectory (tiles, down=+): {r['y_traj']}")
    if len(results) == 2:
        a, b = results
        print(f"\n=== JUMP-PLAN vs WALK-PLAN ===")
        print(f"  jump-button rate : {a['south_rate']:.2f} (jump) vs {b['south_rate']:.2f} (walk)")
        print(f"  actual jump      : {a['jump_tiles']:+.2f} (jump) vs {b['jump_tiles']:+.2f} (walk) tiles")
        print(f"  => plan guides jumping if jump-plan > walk-plan on both")


if __name__ == "__main__":
    main()
