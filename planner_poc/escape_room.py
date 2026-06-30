"""Escape the First Cave save room into the enemy area by climbing LEFT+jump toward the top-left
door and entering it (press DOWN on the door). Detects success via the exported stage id change
(door -> new stage). Captures frames; once in the new room we have an IN-DOMAIN enemy frame.

Run: PYTHONPATH=. python planner_poc/escape_room.py
"""
import os, sys
import numpy as np
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
from nitrogen.eval import Scenario
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import _NAME2IDX, JLX, JLY

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
H = 18
I_SOUTH = _NAME2IDX["south"]
OUT = "/tmp/escape_room"


def ch(dx=0.0, dy=0.0, jump=False, down=False):
    c = np.zeros((H, 25), np.float32)
    c[:, JLX] = dx
    c[:, JLY] = dy
    if jump:
        c[:9, I_SOUTH] = 1.0
    if down:
        c[:, JLY] = 1.0   # hold down (enter door)
    return c


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=False)
    fi = 0
    try:
        sc = Scenario("escape", plan="", objective="escape", success_spec={"reset_macro": BOOT}, max_steps=120)
        obs = env.reset(sc)
        start_stage = obs.state.get("stage")
        env.save_frame(f"{OUT}/f{fi:03d}.png"); fi += 1
        print(f"start stage={start_stage} x={obs.state.get('tile_x')} y={obs.state.get('tile_y')}", flush=True)

        # Phase 1: climb up-left toward the door (hold left, jump every chunk).
        for t in range(16):
            obs = env.step(ch(dx=-1.0, jump=True))
            env.save_frame(f"{OUT}/f{fi:03d}.png"); fi += 1
            print(f"climb {t}: x={obs.state.get('tile_x')} y={obs.state.get('tile_y')} stage={obs.state.get('stage')}", flush=True)
            if obs.state.get("stage") != start_stage:
                print("STAGE CHANGED during climb!"); return
        # Phase 2: at upper area, sweep left/right pressing DOWN to find+enter the door.
        for t in range(20):
            dx = -1.0 if t % 4 < 2 else 1.0
            obs = env.step(ch(dx=dx, down=(t % 2 == 1)))
            env.save_frame(f"{OUT}/f{fi:03d}.png"); fi += 1
            stg = obs.state.get("stage")
            print(f"door-hunt {t}: x={obs.state.get('tile_x')} y={obs.state.get('tile_y')} stage={stg}", flush=True)
            if stg != start_stage:
                print(f"STAGE CHANGED -> {stg}! entered a new room.")
                # capture a few more frames in the new room
                for k in range(4):
                    obs = env.step(ch()); env.save_frame(f"{OUT}/f{fi:03d}.png"); fi += 1
                return
        print("did not change stage; inspect frames")
    finally:
        env.close()
        print(f"saved {fi} frames to {OUT}/")


if __name__ == "__main__":
    main()
