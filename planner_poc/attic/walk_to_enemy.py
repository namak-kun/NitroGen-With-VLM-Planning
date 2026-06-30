"""Scripted navigation: boot First Cave normally (renders correctly), then drive Quote RIGHT
(with periodic jumps to clear gaps/steps) via DIRECT gamepad control, capturing every frame.
Goal: reach the first enemies (Critters/Bats past the save point) so we have an IN-DOMAIN frame
with an enemy on screen, for the 'which button does the model shoot with' offline test.

Run: PYTHONPATH=. python planner_poc/walk_to_enemy.py [nsteps]
Saves /tmp/walk_enemy/f###.png + prints player x/y per step.
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
I_SOUTH = _NAME2IDX["south"]  # jump
OUT = "/tmp/walk_enemy"


def walk_chunk(jump=False):
    c = np.zeros((H, 25), np.float32)
    c[:, JLX] = 1.0          # stick full right
    if jump:
        c[:9, I_SOUTH] = 1.0  # tap jump in the first half of the chunk
    return c


def main():
    nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=False)
    try:
        sc = Scenario("walk", plan="", objective="walk", success_spec={"reset_macro": BOOT}, max_steps=nsteps)
        obs = env.reset(sc)
        env.save_frame(f"{OUT}/f000.png")
        last = (obs.state.get("tile_x"), obs.state.get("tile_y"))
        for t in range(nsteps):
            # jump every 3rd chunk to clear small steps / the water gap
            obs = env.step(walk_chunk(jump=(t % 3 == 2)))
            env.save_frame(f"{OUT}/f{t+1:03d}.png")
            cur = (obs.state.get("tile_x"), obs.state.get("tile_y"))
            print(f"step {t:2d}: x={cur[0]} y={cur[1]}", flush=True)
            last = cur
    finally:
        env.close()
    print(f"saved {nsteps+1} frames to {OUT}/")


if __name__ == "__main__":
    main()
