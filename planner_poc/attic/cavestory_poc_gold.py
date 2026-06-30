"""Cave Story COUNTERFACTUAL POC (gold-token version): drive Quote LEFT + UP to the First Cave
door against NitroGen's rightward prior, using GOLD plan TOKENS (env-free; build_gold_tokens.py)
instead of plan text. The Cave Story plan text produces weak tokens that can't override the
rightward bias; a training-derived 'left' teacher token CAN (gold-token CFG, w=16, multi-seed).

Schedule (position-driven, gold tokens):
  - while right of the door : inject GOLD-LEFT token (multi-seed: pick the most-left chunk)
                              + periodic JUMP (south) to climb ledges toward the upper-left door
  - at the door            : interact (hold DOWN) to enter -> stage id changes = SUCCESS

Success = stage id changes (entered the door) OR net dx < 0 by a clear margin (moved LEFT,
the counterfactual win). Saves annotated video + trajectory.

Run: PYTHONPATH=. python planner_poc/cavestory_poc_gold.py [cfg_w] [nchunks] [seeds]
Out: docs/cavestory_play/poc_gold.mp4 + poc_gold.txt
"""
import os, sys, subprocess
import numpy as np
import torch
from PIL import Image
import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
from nitrogen.eval import Scenario, JLX, JLY
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, _NAME2IDX, B_NORTH, B_DDOWN, B_SOUTH
from cavestory_play_annotated import annotate
from eval_policy import NitroGenPolicy

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
GAMEPLAY_MASK = tuple(MENU_BUTTONS) + (B_NORTH,)
OUT = "/tmp/poc_gold"
CKPT = "runs/stage2_student_mm_txt_lora_cf/plan_stage1_2500.pt"
# First Cave: Quote starts ~tile_x=160; door is a few tiles LEFT + UP. We declare success when
# Quote has moved a clear margin left (counterfactual) and/or the door is entered (stage change).
LEFT_MARGIN = 3.0   # tiles left of start to call the counterfactual demonstrated


def pick_left_chunk(pol, frame, gold_left, w, seeds):
    """Multi-seed: sample `seeds` gold-left chunks, return the one with the most-LEFT stick."""
    best = None
    for s in range(seeds):
        ch = pol.sample_chunk_token(frame, gold_left, w, seed=s)
        sx = float(ch[:, JLX].mean())
        if best is None or sx < best[0]:
            best = (sx, ch)
    return best[1], best[0]


def main():
    w = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
    nchunks = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    gold = torch.load("/tmp/gold_tokens.pt", map_location="cpu", weights_only=False)
    gold_left = gold["left"]["token"]
    print(f"POC gold | w={w} nchunks={nchunks} seeds={seeds} | gold-left uuid={gold['left']['uuid'][:20]}")

    pol = NitroGenPolicy(CKPT, default_cfg=1.0); pol.mm_mode = True; pol.mm_text_only = True
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    fi = 0; log = []
    try:
        sc = Scenario("pocg", plan="go left to the door", objective="reach and enter the door",
                      success_spec={"reset_macro": BOOT}, max_steps=nchunks, cfg_scale=w)
        obs = env.reset(sc)
        st = obs.state; x0, y0, stage0 = st.get("tile_x"), st.get("tile_y"), st.get("stage")
        cur = obs.frame; entered = False; minx = x0
        log.append(f"start x={x0} y={y0} stage={stage0}; gold-left uuid={gold['left']['uuid']}")
        print(f" start x={x0} y={y0} stage={stage0}")
        for c in range(nchunks):
            st = env._read_state(); x, y = st.get("tile_x"), st.get("tile_y")
            moved_left = (x0 is not None and x is not None and (x0 - x) >= LEFT_MARGIN)
            ch, sx = pick_left_chunk(pol, cur, gold_left, w, seeds)
            for b in GAMEPLAY_MASK:
                ch[:, b] = 0.0
            # jump-assist to climb toward the upper-left door (gold tokens are stick-only)
            if c % 2 == 1:
                ch[:9, B_SOUTH] = 1.0
            if moved_left:
                ch[:, B_DDOWN] = 1.0   # at/near the door area: interact (enter)
            rows = env.apply_chunk_capture(ch)
            for ri, (row, frame) in enumerate(rows):
                img = annotate(frame, row, c, ri, f"GOLD-LEFT (stick_x={sx:+.2f})", "POC-GOLD", (x, y))
                Image.fromarray(img).save(f"{OUT}/f{fi:04d}.png"); fi += 1
            cur = rows[-1][1]
            st2 = env._read_state(); nx, ny, nstage = st2.get("tile_x"), st2.get("tile_y"), st2.get("stage")
            if nx is not None:
                minx = min(minx, nx)
            log.append(f"c{c:2d}: x={x}->{nx} y={ny} stage={nstage} chunk_stick_x={sx:+.2f}")
            print(f"  c{c:2d}: x={x}->{nx} y={ny} stage={nstage} stick_x={sx:+.2f}", flush=True)
            if nstage is not None and stage0 is not None and nstage != stage0:
                entered = True
                log.append(f"*** DOOR ENTERED stage {stage0}->{nstage} at chunk {c} ***")
                print(f"  *** DOOR ENTERED! {stage0}->{nstage} ***"); break
        net = (nx - x0) if (x0 is not None and nx is not None) else None
        log.append(f"RESULT: entered_door={entered} | net dx={net:+.1f} | min_x={minx:.1f} "
                   f"(start {x0:.1f}) -> moved LEFT {x0-minx:+.1f} tiles (counterfactual override)")
    finally:
        env.close()
    outdir = os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "docs/cavestory_play")
    subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", "6",
                    "-i", f"{OUT}/f%04d.png", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", f"{outdir}/poc_gold.mp4"])
    with open(f"{outdir}/poc_gold.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")
    print(f"\nsaved {fi} frames -> {outdir}/poc_gold.mp4 ; log -> {outdir}/poc_gold.txt")
    print("\n".join(log[-2:]))


if __name__ == "__main__":
    main()
