"""Validate CaveStoryEnv (the Python class, not manual commands): boot the headless game, run a
menu macro to reach gameplay, then drive it with a scripted policy and save frames. Proves the
env class works end-to-end. Run: python planner_poc/cavestory_demo.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/t-nagupta/NitroGen")

from nitrogen.eval import Scenario, ACTION_DIM, JLX
from nitrogen.eval.envs.cavestory import CaveStoryEnv

# Menu macro from the title screen -> in-game: Start Game -> New Save -> Normal -> Single Player.
BOOT_TO_GAME = [
    ("wait", 1.0), ("key", "z"),   # Start Game
    ("wait", 1.0), ("key", "z"),   # New Save (slot 1)
    ("wait", 1.0), ("key", "z"),   # Normal difficulty
    ("wait", 1.0), ("key", "z"),   # Single Player
    ("wait", 5.0),                  # intro stage loads
    ("key", "z"), ("wait", 0.5), ("key", "z"), ("wait", 0.5),  # advance intro text
]


def main():
    env = CaveStoryEnv(width=640, height=480, boot_wait=11.0)
    try:
        sc = Scenario("demo", plan="explore right", objective="move right",
                      success_spec={"reset_macro": BOOT_TO_GAME}, max_steps=6)
        obs = env.reset(sc)
        env.save_frame("/tmp/cs_env_reset.png")
        print(f"reset ok; frame {obs.frame.shape}, mean px {obs.frame.mean():.1f}")
        # scripted "move right" chunk
        right = np.zeros((18, ACTION_DIM), np.float32); right[:, JLX] = 1.0
        for t in range(5):
            obs = env.step(right)
            env.save_frame(f"/tmp/cs_env_step{t}.png")
            print(f"  step {t}: frame mean px {obs.frame.mean():.1f}")
        print("DONE — frames saved to /tmp/cs_env_*.png")
    finally:
        env.close()


if __name__ == "__main__":
    main()
