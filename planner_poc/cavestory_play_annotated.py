"""Fine-grained annotated playthrough: run NitroGen in Cave Story and capture a frame after
EVERY action row (18 per chunk), with the exact action (left stick + pressed buttons) drawn ON
each frame. Assembles an annotated MP4 so you can see, action-by-action, what the model did and
how the game responded.

Run: PYTHONPATH=. python planner_poc/cavestory_play_annotated.py [ckpt] [plan] [cfg] [nchunks]
Env: MM=1 frame-conditioned student | NULL=1 no-plan baseline.
Out: docs/cavestory_play/annotated_<tag>.mp4
"""
import os, sys, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/home/t-nagupta/NitroGen")
sys.path.insert(0, "/home/t-nagupta/NitroGen/planner_poc")

from nitrogen.eval import Scenario, JLX, JLY
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, _NAME2IDX, B_NORTH
from eval_policy import NitroGenPolicy

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
GAMEPLAY_MASK = tuple(MENU_BUTTONS) + (B_NORTH,)
# the meaningful buttons in this game (after remap): South=jump, West=shoot.
_RPT = [("south", "JUMP(A)"), ("west", "SHOOT(X)"), ("east", "B"), ("north", "Y"),
        ("left_shoulder", "L"), ("right_shoulder", "R"),
        ("dpad_left", "DL"), ("dpad_right", "DR"), ("dpad_up", "DU"), ("dpad_down", "DD")]
OUT = "/tmp/anno_frames"


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F = font(16); Fbig = font(20)


def annotate(frame, row, chunk_i, row_i, plan, mode, pos):
    """Draw the action (stick + buttons) + context onto a copy of the frame (scaled 2x)."""
    im = Image.fromarray(frame).convert("RGB")
    im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
    bar_h = 96
    canvas = Image.new("RGB", (im.width, im.height + bar_h), (12, 12, 12))
    canvas.paste(im, (0, 0))
    dr = ImageDraw.Draw(canvas)
    lx, ly = float(row[JLX]), float(row[JLY])
    btns = [lbl for name, lbl in _RPT if row[_NAME2IDX[name]] > 0.5]
    y0 = im.height + 4
    dr.text((8, y0), f"chunk {chunk_i:2d}  action {row_i:2d}/18   [{mode}]", fill=(255, 255, 0), font=Fbig)
    # stick: arrow text (x+=right, y+=DOWN)
    hx = "RIGHT" if lx > 0.3 else ("LEFT" if lx < -0.3 else "  -  ")
    vy = "DOWN" if ly > 0.3 else ("UP" if ly < -0.3 else " - ")
    dr.text((8, y0 + 26), f"stick: x={lx:+.2f}[{hx}]  y={ly:+.2f}[{vy}]", fill=(120, 220, 255), font=F)
    bcol = (255, 120, 120) if btns else (140, 140, 140)
    dr.text((8, y0 + 46), "buttons: " + (" ".join(btns) if btns else "(none)"), fill=bcol, font=F)
    px = pos[0] if pos[0] is not None else -1
    py = pos[1] if pos[1] is not None else -1
    dr.text((8, y0 + 66), f"plan: '{plan[:46]}'   Quote@({px:.0f},{py:.0f})", fill=(180, 255, 180), font=F)
    # mini stick gauge (top-right)
    cx, cy, r = im.width - 60, 56, 40
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(90, 90, 90))
    dr.line([cx, cy, cx + int(lx * r), cy + int(ly * r)], fill=(120, 220, 255), width=4)
    return np.asarray(canvas)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student_mm/plan_stage1_2500.pt"
    plan = sys.argv[2] if len(sys.argv) > 2 else "jump up to reach the door above"
    cfg = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    nchunks = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    mm = os.environ.get("MM", "0") == "1"
    null = os.environ.get("NULL", "0") == "1"
    mode = "NO-PLAN" if null else ("MM" if mm else "TEXT")
    tag = os.environ.get("TAG", mode.lower())
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    print(f"annotated play | ckpt={ckpt} plan='{plan}' cfg={cfg} mode={mode}")
    pol = NitroGenPolicy(ckpt, default_cfg=cfg)
    pol.mm_mode = mm
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    fi = 0
    try:
        sc = Scenario("anno", plan=plan, objective="play",
                      success_spec={"reset_macro": BOOT}, max_steps=nchunks, cfg_scale=cfg)
        obs = env.reset(sc)
        cur_frame = obs.frame
        action_log = [f"plan: {plan}", f"mode: {mode}  cfg: {cfg}", ""]
        for c in range(nchunks):
            chunk = pol._sample_chunk(cur_frame, plan, cfg,
                                      plan_frames=[cur_frame] if mm else None, null=null)
            for b in GAMEPLAY_MASK:
                chunk[:, b] = 0.0
            rows = env.apply_chunk_capture(chunk)        # 18 (row, frame) pairs
            st = env._read_state()
            pos = (st.get("tile_x"), st.get("tile_y"))
            action_log.append(f"# chunk {c:2d}  Quote@({pos[0]},{pos[1]})")
            for ri, (row, frame) in enumerate(rows):
                img = annotate(frame, row, c, ri, plan, mode, pos)
                Image.fromarray(img).save(f"{OUT}/f{fi:04d}.png"); fi += 1
                lx, ly = float(row[JLX]), float(row[JLY])
                btns = [lbl for name, lbl in _RPT if row[_NAME2IDX[name]] > 0.5]
                action_log.append(f"  a{ri:02d}: stick=({lx:+.2f},{ly:+.2f}) "
                                  f"buttons={','.join(btns) if btns else '-'}")
            cur_frame = rows[-1][1]        # last captured frame = next chunk's input
            print(f"  chunk {c:2d}: Quote@({pos[0]},{pos[1]})  ({fi} frames)", flush=True)
    finally:
        env.close()
    outdir = "/home/t-nagupta/NitroGen/docs/cavestory_play"
    os.makedirs(outdir, exist_ok=True)
    outmp4 = f"{outdir}/annotated_{tag}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", "6",
                    "-i", f"{OUT}/f%04d.png", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", outmp4])
    with open(f"{outdir}/actions_{tag}.txt", "w") as fh:
        fh.write("\n".join(action_log) + "\n")
    print(f"\nsaved {fi} annotated frames -> {outmp4}")
    print(f" per-action log + plan -> {outdir}/actions_{tag}.txt")


if __name__ == "__main__":
    main()
