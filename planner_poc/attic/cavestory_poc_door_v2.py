"""Cave Story door POC v2 — STAGED RECOVERY route (user's plan): the v1 gold-left walk made Quote
step off the platform into the water (can't jump high in water). This version reads Quote's state
each chunk and follows a coordinated, staged route with COMBINED jump+direction primitives:

  Phase WATER   (y deep / low):     gold-RIGHT + JUMP  -> climb out of the water back onto ground
  Phase APPROACH(on ground, right of door): gold-LEFT + JUMP at takeoff -> JUMP-LEFT onto the
                                    ledge below the upper-left door (clears the gap, no fall)
  Phase DOOR    (near door x, up):  interact (hold DOWN) to enter -> stage id change = SUCCESS

Coordinated primitive = inject the gold direction token (stick) AND force the SOUTH (jump) button
in the first half of the chunk (takeoff), so Quote jumps WHILE moving in the chosen direction
(the v1 bug was independent, mistimed jumps). Multi-seed: pick the chunk most extreme in the
desired stick direction.

Run: PYTHONPATH=. python planner_poc/cavestory_poc_door_v2.py [cfg_w] [nchunks] [seeds]
Out: docs/cavestory_play/poc_door_v2.mp4 + poc_door_v2.txt
"""
import os, sys, subprocess
import numpy as np
import torch
from PIL import Image
import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
from nitrogen.eval import Scenario, JLX, JLY
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, B_NORTH, B_DDOWN, B_SOUTH
from cavestory_play_annotated import annotate
from eval_policy import NitroGenPolicy

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
GAMEPLAY_MASK = tuple(MENU_BUTTONS) + (B_NORTH,)
OUT = "/tmp/poc_door_v2"
CKPT = "runs/stage2_student_mm_txt_lora_cf/plan_stage1_2500.pt"
# First Cave tiles (from frames): start x=160 y=128; water sits ~y>=165; door is up-LEFT near
# x<=150 and the ledge below it ~y<=132. (tile = subpixel/512.)
WATER_Y = 165.0
DOOR_X = 150.0
DOOR_LEDGE_Y = 134.0


def pick(pol, frame, token, w, seeds, axis, sign):
    """multi-seed: pick the chunk whose stick `axis` is most extreme in `sign` direction."""
    best = None
    for s in range(seeds):
        ch = pol.sample_chunk_token(frame, token, w, seed=s)
        v = sign * float(ch[:, axis].mean())
        if best is None or v > best[0]:
            best = (v, ch, float(ch[:, JLX].mean()), float(ch[:, JLY].mean()))
    return best[1], best[2], best[3]


def main():
    w = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
    nchunks = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    gold = torch.load("/tmp/gold_tokens.pt", map_location="cpu", weights_only=False)
    gL, gR = gold["left"]["token"], gold["right"]["token"]
    pol = NitroGenPolicy(CKPT, default_cfg=1.0); pol.mm_mode = True; pol.mm_text_only = True
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    fi = 0; log = []
    print(f"POC v2 staged | w={w} nchunks={nchunks} seeds={seeds}")
    try:
        sc = Scenario("pocv2", plan="reach the door", objective="reach and enter the door",
                      success_spec={"reset_macro": BOOT}, max_steps=nchunks, cfg_scale=w)
        obs = env.reset(sc)
        st = obs.state; x0, y0, stage0 = st.get("tile_x"), st.get("tile_y"), st.get("stage")
        cur = obs.frame; entered = False
        log.append(f"start x={x0} y={y0} stage={stage0}; WATER_Y={WATER_Y} DOOR_X={DOOR_X}")
        print(f" start x={x0} y={y0} stage={stage0}")
        for c in range(nchunks):
            st = env._read_state(); x, y = st.get("tile_x"), st.get("tile_y")
            in_water = (y is not None and y >= WATER_Y)
            at_door = (x is not None and x <= DOOR_X and y is not None and y <= DOOR_LEDGE_Y + 2)
            if in_water:
                phase = "WATER->right+jump"; ch, sx, sy = pick(pol, cur, gR, w, seeds, JLX, +1)
                ch[:12, B_SOUTH] = 1.0                       # sustained jump to climb out
            elif at_door:
                phase = "DOOR->interact"; ch, sx, sy = pick(pol, cur, gL, w, seeds, JLX, -1)
                ch[:, B_DDOWN] = 1.0
            else:
                phase = "APPROACH->jump-left"; ch, sx, sy = pick(pol, cur, gL, w, seeds, JLX, -1)
                ch[:9, B_SOUTH] = 1.0                        # takeoff jump WHILE moving left
            for b in GAMEPLAY_MASK:
                ch[:, b] = 0.0
            rows = env.apply_chunk_capture(ch)
            for ri, (row, frame) in enumerate(rows):
                img = annotate(frame, row, c, ri, f"{phase} (sx={sx:+.2f})", "POC-DOOR-v2", (x, y))
                Image.fromarray(img).save(f"{OUT}/f{fi:04d}.png"); fi += 1
            cur = rows[-1][1]
            st2 = env._read_state(); nx, ny, nstage = st2.get("tile_x"), st2.get("tile_y"), st2.get("stage")
            log.append(f"c{c:2d}: x={x}->{nx} y={y}->{ny} stage={nstage} phase={phase} sx={sx:+.2f}")
            print(f"  c{c:2d}: x={x}->{nx} y={y}->{ny} [{phase}] sx={sx:+.2f}", flush=True)
            if nstage is not None and stage0 is not None and nstage != stage0:
                entered = True
                log.append(f"*** DOOR ENTERED stage {stage0}->{nstage} at chunk {c} ***")
                print(f"  *** DOOR ENTERED! {stage0}->{nstage} ***"); break
        net = (nx - x0) if (x0 is not None and nx is not None) else None
        log.append(f"RESULT: entered_door={entered} net_dx={net:+.1f} final=({nx},{ny})")
    finally:
        env.close()
    outdir = os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "docs/cavestory_play")
    subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", "6",
                    "-i", f"{OUT}/f%04d.png", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", f"{outdir}/poc_door_v2.mp4"])
    with open(f"{outdir}/poc_door_v2.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")
    print(f"\nsaved {fi} frames -> {outdir}/poc_door_v2.mp4")
    print("\n".join(log[-2:]))


if __name__ == "__main__":
    main()
