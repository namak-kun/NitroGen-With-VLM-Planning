"""Clean shoot-only verification: spawn First Cave with the Polar Star granted, do NOT jump
(so Quote doesn't fall in the water and drown), then hold each face button and check whether
bullets spawn (DRS_STATE bullets field). Confirms which NitroGen button index = SHOOT in the
remapped doukutsu build (expected: West=X).

Run: PYTHONPATH=. python planner_poc/shoot_only_test.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/t-nagupta/NitroGen")
from nitrogen.eval import Scenario
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import _NAME2IDX

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
H = 18
FACE = ["south", "east", "west", "north"]
NAMES = {"south": "A", "east": "B", "west": "X", "north": "Y"}


def btn(idx):
    c = np.zeros((H, 25), np.float32)
    c[:, idx] = 1.0
    return c


def main():
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=False, give_weapon=True)
    try:
        sc = Scenario("shoot", plan="", objective="shoot", success_spec={"reset_macro": BOOT}, max_steps=60)
        obs = env.reset(sc)
        print(f"booted. state={obs.state}\n")
        # idle a moment (no jump) to confirm Quote is grounded + alive
        for _ in range(2):
            obs = env.step(np.zeros((H, 25), np.float32))
        print(f"after idle: life={obs.state.get('life')} bullets={obs.state.get('bullets')} "
              f"y={obs.state.get('tile_y')}\n")
        print("shoot test (no jumping; max bullets while holding each face button):")
        for name in FACE:
            idx = _NAME2IDX[name]
            # let bullets despawn between tests
            for _ in range(3):
                obs = env.step(np.zeros((H, 25), np.float32))
            maxb = 0
            for _ in range(4):
                obs = env.step(btn(idx))
                maxb = max(maxb, obs.state.get("bullets", 0))
            env.save_frame(f"/tmp/shoot_{name}.png")
            print(f"  {NAMES[name]}({name:5}) idx {idx:2}: max bullets={maxb} "
                  f"life={obs.state.get('life')} {'<-- SHOOT' if maxb > 0 else ''}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
