"""First NitroGen-in-the-loop run: drive the real Cave Story env with NitroGenPolicy, with the
game FROZEN during inference (speedhack). Sanity check — does NitroGen's System-1 produce
sensible actions / move Quote on this OOD 2D platformer? Reports per-step action stats + frame
movement, saves frames + a video.

Run: PYTHONPATH=. python planner_poc/cavestory_nitrogen.py [ckpt] [plan] [cfg] [nsteps]
"""
import os, sys, time, subprocess
import numpy as np
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
import os; sys.path.insert(0, os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "planner_poc"))

from nitrogen.eval import Scenario, ACTION_DIM, JLX, JLY, N_BUTTONS
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, _NAME2IDX, B_NORTH
from eval_policy import NitroGenPolicy

# title -> free-roam: 4 menu confirms, then HOLD skip (X) to fast-forward the opening cutscene,
# with confirms interspersed to clear text boxes.
BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
OUT = "/tmp/nitrogen_run"
# names of the buttons we report (in canonical order), for readable logs
_RPT = ["south", "east", "west", "north", "left_shoulder", "right_shoulder",
        "left_trigger", "right_trigger", "dpad_left", "dpad_right", "dpad_up", "dpad_down"]
# In doukutsu-rs (patched to CS+ default): South=jump, West=shoot. North=skip/inventory/map ->
# mask it (plus START/BACK/GUIDE) so the model can only jump/shoot/move and never opens a menu
# or skips. West (shoot) is intentionally NOT masked.
GAMEPLAY_MASK = tuple(MENU_BUTTONS) + (B_NORTH,)


def mask_menu(chunk):
    """Zero menu/inventory/map/skip buttons so the model can only jump (South), shoot (East),
    and move. Mirrors NitroGen play.py NO_MENU, extended for doukutsu-rs's button map."""
    for b in GAMEPLAY_MASK:
        chunk[:, b] = 0.0
    return chunk


def _chunk_to_text(t, chunk):
    """Render one (18,25) action chunk as exact per-row values: left/right stick + any button
    pressed (>0.5) on that row. Returns a list of text lines."""
    lines = [f"# chunk {t:02d}  (18 rows x 25 dims)"]
    for r in range(chunk.shape[0]):
        lx, ly = chunk[r, JLX], chunk[r, JLY]
        rx, ry = chunk[r, 23], chunk[r, 24]
        btns = [n for n in _RPT if chunk[r, _NAME2IDX[n]] > 0.5]
        lines.append(f"  r{r:02d}: Lstick=({lx:+.3f},{ly:+.3f}) Rstick=({rx:+.3f},{ry:+.3f}) "
                     f"buttons={','.join(btns) if btns else '-'}")
    return lines


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student/plan_stage1_2500.pt"
    plan = sys.argv[2] if len(sys.argv) > 2 else "explore the cave and move to the right"
    cfg = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    print(f"loading NitroGenPolicy: {ckpt} | plan='{plan}' | cfg={cfg}")
    mm = os.environ.get("MM", "0") == "1"
    null = os.environ.get("NULL", "0") == "1"  # no-plan baseline (unconditioned frozen DiT)
    pol = NitroGenPolicy(ckpt, default_cfg=cfg)
    pol.mm_mode = mm  # EXP-050: frame-conditioned plan tokens (encode_multimodal on the live frame)
    print(f" policy mode: {'NO-PLAN (null)' if null else ('FRAME-CONDITIONED (mm)' if mm else 'text-only')}")
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    try:
        sc = Scenario("nitro", plan=plan, objective="play",
                      success_spec={"reset_macro": BOOT}, max_steps=nsteps, cfg_scale=cfg)
        obs = env.reset(sc)
        env.save_frame(f"{OUT}/f000.png")
        prev = obs.frame.astype(int)
        print(f"\nrunning {nsteps} steps (game frozen during each inference):\n")
        moved = 0.0
        all_chunks = []
        txt_lines = []
        traj = [(obs.state.get("tile_x"), obs.state.get("tile_y"), obs.state.get("stage"))]
        for t in range(nsteps):
            # mm: condition the plan tokens on the live frame (single-frame history here).
            chunk = pol._sample_chunk(obs.frame, plan, cfg,
                                      plan_frames=[obs.frame] if mm else None, null=null)  # (18,25)
            chunk = mask_menu(chunk)
            all_chunks.append(chunk.copy())
            txt_lines += _chunk_to_text(t, chunk)
            sx, sy = float(chunk[:, JLX].mean()), float(chunk[:, JLY].mean())
            pressed = [n for n in _RPT if (chunk[:, _NAME2IDX[n]] > 0.5).mean() > 0.25]
            obs = env.step(chunk)
            tx, ty, stg = obs.state.get("tile_x"), obs.state.get("tile_y"), obs.state.get("stage")
            traj.append((tx, ty, stg))
            d = float(np.abs(obs.frame.astype(int) - prev).mean()); prev = obs.frame.astype(int)
            moved += d
            env.save_frame(f"{OUT}/f{t+1:03d}.png")
            print(f"  step {t:2d}: stick=({sx:+.2f},{sy:+.2f}) buttons={pressed or '-'} "
                  f"| pos=({tx},{ty}) | frame-delta={d:.2f}")
        print(f"\n total frame movement over {nsteps} steps: {moved:.1f} "
              f"(higher => NitroGen is actively changing the game)")
        # trajectory summary: where did Quote actually go / get stuck?
        xs = [p[0] for p in traj if p[0] is not None]
        ys = [p[1] for p in traj if p[1] is not None]
        if xs:
            print(f" trajectory: x {xs[0]:.0f}->{xs[-1]:.0f} (range {min(xs):.0f}..{max(xs):.0f}), "
                  f"y {ys[0]:.0f}->{ys[-1]:.0f} (min {min(ys):.0f}=highest), "
                  f"net dx={xs[-1]-xs[0]:+.0f} dy={ys[-1]-ys[0]:+.0f} (dy<0 = climbed up)")
        # dump the EXACT actions: full (nsteps,18,25) array + a human-readable per-row text file
        np.save(f"{OUT}/actions.npy", np.stack(all_chunks))
        with open(f"{OUT}/actions.txt", "w") as fh:
            fh.write("\n".join(txt_lines) + "\n")
        print(f" exact actions saved: {OUT}/actions.npy (shape {np.stack(all_chunks).shape}) "
              f"+ {OUT}/actions.txt")
        # make a video
        subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", "4",
                        "-i", f"{OUT}/f%03d.png", "-pix_fmt", "yuv420p", f"{OUT}/run.mp4"])
        print(f" frames + video saved to {OUT}/ (run.mp4)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
