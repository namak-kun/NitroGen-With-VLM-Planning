"""Cave Story COUNTERFACTUAL POC: drive Quote LEFT + UP to the First Cave door and ENTER it,
against NitroGen's rightward bias, using GOLD (hand-written) plans + the best override student.

First Cave start: Quote center; DOOR top-LEFT (counterfactual: model drifts RIGHT to the save
refiller). Success = entering the door -> the exported stage id changes from 13.

Position-driven gold-plan schedule (System-2 intent; System-1 handles the jumps):
  - far right of door : "go left toward the door on the upper left"
  - left & low        : "jump up onto the ledge to reach the door"
  - at the door       : "enter the door"  (+ hold DOWN, the Cave Story door action)

Run: PYTHONPATH=. python planner_poc/cavestory_poc_door.py [ckpt] [cfg] [nchunks] [MM=1 TXT=1]
Out: docs/cavestory_play/poc_door.mp4 + poc_door.txt (trajectory + plan log)
"""
import os, sys, subprocess
import numpy as np
from PIL import Image
import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
from nitrogen.eval import Scenario, JLX, JLY
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, _NAME2IDX, B_NORTH, B_DDOWN
from cavestory_play_annotated import annotate, _RPT
from eval_policy import NitroGenPolicy

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
GAMEPLAY_MASK = tuple(MENU_BUTTONS) + (B_NORTH,)
OUT = "/tmp/poc_door"
I_SOUTH = _NAME2IDX["south"]

# First Cave reference tiles (from the start frame): Quote starts ~x=10,y=8; door ~x=6,y=4.
DOOR_X = 6.0


def gold_plan(x, y):
    """Position-driven gold plan. x/y in tiles (y increases downward)."""
    if x is None:
        return "go to the door on the left", False
    if x > DOOR_X + 1.0:
        return "go to the door on the left", False
    if y > 5.0:
        return "jump up onto the ledge where the door is", False
    return "enter the door", True   # at the door: interact (hold down)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student_mm_txt_lora_cf/plan_stage1_2500.pt"
    cfg = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    nchunks = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    mm = os.environ.get("MM", "1") == "1"
    txt = os.environ.get("TXT", "1") == "1"
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    print(f"POC door | ckpt={ckpt} cfg={cfg} mm={mm} text_only={txt}")
    pol = NitroGenPolicy(ckpt, default_cfg=cfg); pol.mm_mode = mm; pol.mm_text_only = txt
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    fi = 0
    log = []
    try:
        sc = Scenario("poc", plan="reach the door", objective="reach and enter the door",
                      success_spec={"reset_macro": BOOT}, max_steps=nchunks, cfg_scale=cfg)
        obs = env.reset(sc)
        st = obs.state; x0, y0, stage0 = st.get("tile_x"), st.get("tile_y"), st.get("stage")
        cur = obs.frame
        log.append(f"start: x={x0} y={y0} stage={stage0} door_x~{DOOR_X}")
        print(f" start: x={x0} y={y0} stage={stage0}")
        entered = False
        for c in range(nchunks):
            st = env._read_state(); x, y = st.get("tile_x"), st.get("tile_y")
            plan, at_door = gold_plan(x, y)
            ch = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur] if mm else None)
            for b in GAMEPLAY_MASK:
                ch[:, b] = 0.0
            if at_door:
                ch[:, B_DDOWN] = 1.0          # interact: hold DOWN at the door
            rows = env.apply_chunk_capture(ch)
            for ri, (row, frame) in enumerate(rows):
                img = annotate(frame, row, c, ri, plan, "POC-DOOR", (x, y))
                Image.fromarray(img).save(f"{OUT}/f{fi:04d}.png"); fi += 1
            cur = rows[-1][1]
            st2 = env._read_state(); nx, ny, nstage = st2.get("tile_x"), st2.get("tile_y"), st2.get("stage")
            log.append(f"chunk {c:2d}: x={x}->{nx} y={y}->{ny} stage={nstage} plan={plan!r}")
            print(f"  c{c:2d}: x={x}->{nx} y={ny} stage={nstage} | {plan}", flush=True)
            if nstage is not None and stage0 is not None and nstage != stage0:
                entered = True
                log.append(f"*** DOOR ENTERED: stage {stage0} -> {nstage} at chunk {c} ***")
                print(f"  *** DOOR ENTERED! stage {stage0}->{nstage} ***")
                break
        log.append(f"RESULT: {'SUCCESS (entered door)' if entered else 'did not enter door'}; "
                   f"net dx={None if x0 is None else (nx - x0):+.1f} (negative=moved LEFT, counterfactual)")
    finally:
        env.close()
    outdir = os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "docs/cavestory_play")
    subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", "6",
                    "-i", f"{OUT}/f%04d.png", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", f"{outdir}/poc_door.mp4"])
    with open(f"{outdir}/poc_door.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")
    print(f"\nsaved {fi} frames -> {outdir}/poc_door.mp4 ; log -> {outdir}/poc_door.txt")
    print("\n".join(log[-3:]))


if __name__ == "__main__":
    main()
