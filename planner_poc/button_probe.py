"""Button-semantics probe: drive the virtual gamepad DIRECTLY (no model) to discover which
NitroGen button index maps to JUMP in the current doukutsu-rs build. Boots First Cave once,
then for each candidate button holds it for a few control steps and records whether Quote
leaves the ground (tile_y decreases; y increases downward). Also tests stick-up. This isolates
button SEMANTICS from model behavior so we can map the game's jump to whatever the model emits.

Run: PYTHONPATH=. python planner_poc/button_probe.py
"""
import sys
import numpy as np
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
import os; sys.path.insert(0, os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "planner_poc"))

from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import _NAME2IDX, JLX, JLY

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
H = 18
CANDIDATES = ["south", "east", "west", "north"]


def chunk_button(idx):
    c = np.zeros((H, 25), np.float32)
    c[:, idx] = 1.0
    return c


def chunk_stick(x, y):
    c = np.zeros((H, 25), np.float32)
    c[:, JLX] = x; c[:, JLY] = y
    return c


def settle(env, steps=3):
    """Let Quote rest (no input) and return resting tile_y."""
    ys = []
    for _ in range(steps):
        obs = env.step(np.zeros((H, 25), np.float32))
        ys.append(obs.state.get("tile_y", 0.0))
    return ys[-1]


def main():
    from nitrogen.eval import Scenario
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True, give_weapon=True)
    try:
        sc = Scenario("probe", plan="", objective="probe",
                      success_spec={"reset_macro": BOOT}, max_steps=200)
        obs = env.reset(sc)
        print("booted. initial state:", {k: obs.state.get(k) for k in ("tile_x", "tile_y", "stage", "life")})
        for name in CANDIDATES:
            rest = settle(env, 4)
            idx = _NAME2IDX[name]
            ys = [rest]
            for _ in range(4):
                obs = env.step(chunk_button(idx))
                ys.append(obs.state.get("tile_y", rest))
            dy = rest - min(ys)  # positive => went UP => jumped
            print(f"  button {name:6} (idx {idx:2}): rest_y={rest:.1f} min_y={min(ys):.1f} "
                  f"-> up {dy:+.2f} tiles {'<-- JUMP' if dy > 1 else ''}")
        # stick up
        rest = settle(env, 4)
        ys = [rest]
        for _ in range(4):
            obs = env.step(chunk_stick(0.0, -1.0))
            ys.append(obs.state.get("tile_y", rest))
        dy = rest - min(ys)
        print(f"  stick UP        : rest_y={rest:.1f} min_y={min(ys):.1f} -> up {dy:+.2f} tiles "
              f"{'<-- JUMP' if dy > 1 else ''}")
        # stick right (sanity: should change tile_x)
        rest_x = env._read_state().get("tile_x", 0.0)
        for _ in range(4):
            obs = env.step(chunk_stick(1.0, 0.0))
        dx = obs.state.get("tile_x", rest_x) - rest_x
        print(f"  stick RIGHT     : dx={dx:+.2f} tiles (sanity: movement works)")
        # SHOOT test: the starting Polar Star fires bullets; pressing the shoot button should
        # spawn bullets (DRS_STATE bullets field). Tests each face button for bullet spawning.
        print("\nshoot test (bullets spawned while holding each button):")
        for name in CANDIDATES:
            idx = _NAME2IDX[name]
            settle(env, 3)
            maxb = 0
            for _ in range(4):
                obs = env.step(chunk_button(idx))
                maxb = max(maxb, obs.state.get("bullets", 0))
            print(f"  button {name:6} (idx {idx:2}): max bullets={maxb} "
                  f"{'<-- SHOOT' if maxb > 0 else ''}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
