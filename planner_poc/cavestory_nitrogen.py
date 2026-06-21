"""First NitroGen-in-the-loop run: drive the real Cave Story env with NitroGenPolicy, with the
game FROZEN during inference (speedhack). Sanity check — does NitroGen's System-1 produce
sensible actions / move Quote on this OOD 2D platformer? Reports per-step action stats + frame
movement, saves frames + a video.

Run: PYTHONPATH=. python planner_poc/cavestory_nitrogen.py [ckpt] [plan] [cfg] [nsteps]
"""
import os, sys, time, subprocess
import numpy as np
sys.path.insert(0, "/home/t-nagupta/NitroGen")
sys.path.insert(0, "/home/t-nagupta/NitroGen/planner_poc")

from nitrogen.eval import Scenario, ACTION_DIM, JLX, JLY, N_BUTTONS
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, _NAME2IDX
from eval_policy import NitroGenPolicy

# title -> free-roam: 4 menu confirms, then HOLD skip (X) to fast-forward the opening cutscene,
# with confirms interspersed to clear text boxes.
BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
OUT = "/tmp/nitrogen_run"
# names of the buttons we report (in canonical order), for readable logs
_RPT = ["south", "east", "west", "north", "left_shoulder", "right_shoulder",
        "left_trigger", "right_trigger", "dpad_left", "dpad_right", "dpad_up", "dpad_down"]


def mask_menu(chunk):
    """Zero START/BACK/GUIDE so the model can't open menus (NitroGen play.py NO_MENU)."""
    for b in MENU_BUTTONS:
        chunk[:, b] = 0.0
    return chunk


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student/plan_stage1_2500.pt"
    plan = sys.argv[2] if len(sys.argv) > 2 else "explore the cave and move to the right"
    cfg = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    print(f"loading NitroGenPolicy: {ckpt} | plan='{plan}' | cfg={cfg}")
    pol = NitroGenPolicy(ckpt, default_cfg=cfg)
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    try:
        sc = Scenario("nitro", plan=plan, objective="play",
                      success_spec={"reset_macro": BOOT}, max_steps=nsteps, cfg_scale=cfg)
        obs = env.reset(sc)
        env.save_frame(f"{OUT}/f000.png")
        prev = obs.frame.astype(int)
        print(f"\nrunning {nsteps} steps (game frozen during each inference):\n")
        moved = 0.0
        for t in range(nsteps):
            chunk = pol._sample_chunk(obs.frame, plan, cfg)  # (18,25)
            chunk = mask_menu(chunk)
            sx, sy = float(chunk[:, JLX].mean()), float(chunk[:, JLY].mean())
            pressed = [n for n in _RPT if (chunk[:, _NAME2IDX[n]] > 0.5).mean() > 0.25]
            obs = env.step(chunk)
            d = float(np.abs(obs.frame.astype(int) - prev).mean()); prev = obs.frame.astype(int)
            moved += d
            env.save_frame(f"{OUT}/f{t+1:03d}.png")
            print(f"  step {t:2d}: stick=({sx:+.2f},{sy:+.2f}) buttons={pressed or '-'} "
                  f"| frame-delta={d:.2f}")
        print(f"\n total frame movement over {nsteps} steps: {moved:.1f} "
              f"(higher => NitroGen is actively changing the game)")
        # make a video
        subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", "4",
                        "-i", f"{OUT}/f%03d.png", "-pix_fmt", "yuv420p", f"{OUT}/run.mp4"])
        print(f" frames + video saved to {OUT}/ (run.mp4)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
