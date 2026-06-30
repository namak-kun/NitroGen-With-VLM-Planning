"""Spawn-sweep: boot doukutsu-rs directly into a range of stage IDs (DRS_START_STAGE patch) and
grab one screenshot each, so we can pick frames that actually contain enemies / a boss for the
'which button does the model shoot with' test (and reproducible combat scenarios for the eval).

Run: PYTHONPATH=. python planner_poc/stage_sweep.py [id_start id_end]
Saves /tmp/stage_sweep/stage_<id>.png
"""
import os, sys, time
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
from nitrogen.eval import Scenario
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK

# start_new_game skips the long intro cutscene, so boot is just: confirm through the title /
# new-save / difficulty prompts, then we're in the stage. A few spaced confirms cover it.
BOOT = [("wait", 1.5)] + sum(([("btn", MENU_OK), ("wait", 1.0)] for _ in range(6)), []) + [("wait", 1.5)]
OUT = "/tmp/stage_sweep"


def main():
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    os.makedirs(OUT, exist_ok=True)
    for sid in range(a, b + 1):
        env = CaveStoryEnv(boot_wait=9.0, freeze_during_inference=False, start_stage=sid)
        try:
            sc = Scenario("sweep", plan="", objective="sweep", success_spec={"reset_macro": BOOT}, max_steps=1)
            obs = env.reset(sc)
            st = obs.state
            env.save_frame(f"{OUT}/stage_{sid:02d}.png")
            print(f"stage {sid:2d}: saved | state={{x:{st.get('tile_x')}, y:{st.get('tile_y')}, "
                  f"stage:{st.get('stage')}, life:{st.get('life')}}}", flush=True)
        except Exception as e:
            print(f"stage {sid:2d}: ERROR {e}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
