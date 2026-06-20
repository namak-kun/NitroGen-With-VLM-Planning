"""Validate CaveStoryEnv (the Python class, not manual commands): boot the headless game, run a
menu macro to reach gameplay, then drive it with a scripted policy and save frames. Proves the
env class works end-to-end. Run: python planner_poc/cavestory_demo.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/t-nagupta/NitroGen")

from nitrogen.eval import Scenario, ACTION_DIM, JLX
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK

# Menu + intro macro (GAMEPAD) from the title -> FREE-ROAM gameplay (First Cave "Start Point").
# Start Game -> New Save -> Normal -> Single Player, then spam confirm to clear the opening
# cutscene ("Connecting to network... Logged on...") until Quote is controllable.
BOOT_TO_GAME = (
    [("wait", 1.2), ("btn", MENU_OK)] * 4          # 4 menu confirms
    + [("wait", 4.0)]
    + [("wait", 1.0), ("btn", MENU_OK)] * 18       # clear the opening cutscene
)


def main():
    env = CaveStoryEnv(width=640, height=480, boot_wait=11.0, use_gamepad=True)
    try:
        sc = Scenario("demo", plan="explore right", objective="move right",
                      success_spec={"reset_macro": BOOT_TO_GAME}, max_steps=6)
        obs = env.reset(sc)
        env.save_frame("/tmp/cs_env_reset.png")
        print(f"reset ok; gamepad={'yes' if env._pad else 'no'}; frame {obs.frame.shape}")
        # analog: full left-stick RIGHT (JLX=+1) faithfully via the virtual pad
        right = np.zeros((18, ACTION_DIM), np.float32); right[:, JLX] = 1.0
        left = np.zeros((18, ACTION_DIM), np.float32); left[:, JLX] = -1.0
        prev = obs.frame.astype(np.int16)
        for t in range(5):
            obs = env.step(right if t < 3 else left)
            env.save_frame(f"/tmp/cs_env_step{t}.png")
            diff = float(np.abs(obs.frame.astype(np.int16) - prev).mean())
            print(f"  step {t} ({'RIGHT' if t<3 else 'LEFT'}): frame-delta vs prev = {diff:.2f}")
            prev = obs.frame.astype(np.int16)
        print("DONE — frames saved to /tmp/cs_env_*.png (nonzero frame-delta => player/world moved)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
