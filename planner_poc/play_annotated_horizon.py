"""Annotated CLOSED-LOOP playthrough with a configurable ACTION/EXECUTION HORIZON A, to surface
failure cases. The NitroGen DiT predicts a chunk of H=18 action rows; at inference we execute only
the first A of them open-loop, capture a frame after EVERY executed row (with the action + the
System-2 plan that drove it drawn on the frame), then re-observe + re-plan. This is the standard
action-chunking knob (GR00T/NitroGen): small A = reactive (re-plan often), large A = more open-loop
commitment to possibly-stale actions -> MORE failures. Comparing A=2 vs A=4 makes that visible.

Like cavestory_play_annotated.py but generic over any ProcGameEnv (uses apply_chunk_capture) and with
the A knob + per-cycle System-2 plan generation.

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B \
       .venv/bin/python planner_poc/play_annotated_horizon.py --env stk \
       --ckpt runs/stage2_2b_btn/plan_stage1_600.pt --A 4 --nrows 48 --cfg 8
  --plan "go right"   : fix the plan (else a fresh System-2 plan is generated each re-infer cycle).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.eval import Scenario
from nitrogen.eval.core import JLX, JLY
from run_poc import make_env_factory
from eval_policy import NitroGenPolicy

# controller button index -> short label drawn on the frame (NitroGen 25-dim action: buttons 0:21).
_BTN = [(18, "A/jump"), (20, "X/shoot"), (5, "B"), (10, "Y"), (16, "RT/accel"), (9, "LT/brake"),
        (7, "LB"), (14, "RB"), (1, "Dn"), (4, "Up"), (2, "Dl"), (3, "Dr"), (19, "Start")]
STEER_THRESH = 0.2   # match the envs: stick is 0.5-neutral; right = JLX>0.5+t, left = <0.5-t

# per-env System-2 objective (steers generate_plan) + boot wait
OBJECTIVES = {
    "stk": "You are driving a kart. What should the agent do next to stay on the track and race well?",
    "trigger_rally": "You are driving a rally car. What should the agent do next to stay on the road?",
    "xmoto": "You are riding a motorbike over terrain. What should the agent do next?",
    "sdlpop": "You are a platformer hero. What should the agent do next to advance through the level?",
    "thextech": "You are a platformer hero. What should the agent do next to advance and avoid hazards?",
    "castlevania_godot": "You are a platformer hero with a whip. What should the agent do next?",
    "solarus_zelda": "You are a top-down adventurer. What should the agent do next to explore?",
}

# buttons to ZERO before applying, per env, so a spurious press can't strand the agent on a
# menu (e.g. TheXTech maps START->Enter = pause). Mirrors the Cave Story harness's GAMEPLAY_MASK:
# this keeps the agent IN GAMEPLAY so we observe real platforming failures (deaths) not pause-strands.
MENU_MASK = {"thextech": (19,), "sdlpop": (19,), "castlevania_godot": (19,),
             "solarus_zelda": (16, 19)}  # 19 = START; solarus also maps RTRIG(16) -> pause


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F = font(15); Fbig = font(19)

# game-agnostic System-2 prompt so generated plans aren't biased by the platformer DEFAULT_SYS.
PLAN_SYS = ("You are the high-level planner for an agent playing a 2D video game. You see the most "
            "recent frames (oldest first). Output ONE short imperative directive (max 8 words) for "
            "what the agent should do next. Output only the directive, no explanation.")


def annotate(frame, row, cycle, row_i, A, plan, env_name, state):
    """Draw the executed action (stick + buttons) + the plan + ground-truth state onto the frame."""
    im = Image.fromarray(np.asarray(frame)).convert("RGB")
    scale = 2 if im.width < 480 else 1
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    bar_h = 120
    canvas = Image.new("RGB", (max(im.width, 360), im.height + bar_h), (12, 12, 12))
    canvas.paste(im, (0, 0))
    dr = ImageDraw.Draw(canvas)
    lx, ly = float(row[JLX]), float(row[JLY])
    btns = [lbl for idx, lbl in _BTN if float(row[idx]) > 0.5]
    y0 = im.height + 4
    dr.text((8, y0), f"A={A}  re-plan#{cycle:02d}  exec-row {row_i + 1}/{A}   [{env_name}]",
            fill=(255, 255, 0), font=Fbig)
    hx = "RIGHT" if lx > 0.5 + STEER_THRESH else ("LEFT" if lx < 0.5 - STEER_THRESH else "  -  ")
    vy = "DOWN" if ly > 0.5 + STEER_THRESH else ("UP" if ly < 0.5 - STEER_THRESH else " - ")
    dr.text((8, y0 + 24), f"stick x={lx:+.2f}[{hx}]  y={ly:+.2f}[{vy}]", fill=(120, 220, 255), font=F)
    bcol = (255, 120, 120) if btns else (140, 140, 140)
    dr.text((8, y0 + 44), "buttons: " + (" ".join(btns) if btns else "(none)"), fill=bcol, font=F)
    dr.text((8, y0 + 64), f"plan: '{plan[:52]}'", fill=(180, 255, 180), font=F)
    # ground-truth game state (from the source-patched export): position, lives, death, menu.
    if state:
        dead = bool(state.get("dead")); menu = bool(state.get("in_menu"))
        scol = (255, 90, 90) if (dead or menu) else (200, 200, 200)
        flags = (" DEAD" if dead else "") + (" IN-MENU" if menu else "")
        pos = (f"pos=({state.get('x','?')},{state.get('y','?')}) v=({state.get('vx','?')},"
               f"{state.get('vy','?')})  lives={state.get('lives','?')}")
        dr.text((8, y0 + 84), f"state: {pos}{flags}", fill=scol, font=F)
    # mini stick gauge (top-right), centered on 0.5-neutral
    cx, cy, r = canvas.width - 52, 50, 38
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(90, 90, 90))
    dr.line([cx, cy, cx + int((lx - 0.5) * 2 * r), cy + int((ly - 0.5) * 2 * r)],
            fill=(120, 220, 255), width=4)
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="stk")
    ap.add_argument("--ckpt", default="runs/stage2_2b_btn/plan_stage1_600.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--A", type=int, default=4, help="execution horizon: rows executed open-loop per re-infer")
    ap.add_argument("--nrows", type=int, default=48, help="total action rows to execute")
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--plan", default=None, help="fixed plan text; if omitted, generate a fresh plan each cycle")
    ap.add_argument("--null", action="store_true",
                    help="BASE ablation: run the unconditioned base DiT (no System-2 plan / null plan) "
                         "to compare against the plan-conditioned A=2/A=4 runs.")
    ap.add_argument("--boot-wait", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kw = {}
    if args.boot_wait is not None:
        kw["boot_wait"] = args.boot_wait
    env = make_env_factory(args.env, **kw)()
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    H = pol.m.config.action_horizon
    A = max(1, min(args.A, H))
    per_row = env.chunk_seconds / H
    objective = OBJECTIVES.get(args.env, "What should the agent do next?")
    tag = "base" if args.null else f"A{A}"
    tag = f"{args.env}_{tag}"
    out_root = args.out or f"docs/horizon_play/{tag}"
    os.makedirs(out_root, exist_ok=True)
    frame_dir = f"/tmp/anno_{tag}"
    os.makedirs(frame_dir, exist_ok=True)
    for f in os.listdir(frame_dir):
        os.remove(os.path.join(frame_dir, f))

    mode_str = "NULL/base (no plan)" if args.null else ("FIXED: " + args.plan if args.plan
                                                        else "GENERATED per re-infer")
    print(f"annotated horizon play | env={args.env} A={A} (H={H}) nrows={args.nrows} cfg={args.cfg} "
          f"ckpt={os.path.basename(args.ckpt)} mode={mode_str}")
    log = [f"env={args.env}  ckpt={args.ckpt}  A={A}  H={H}  cfg={args.cfg}",
           f"plan_mode={mode_str}",
           f"per_row={per_row:.3f}s  objective={objective}", ""]
    fi = 0
    cycle = 0
    executed = 0
    died = False
    entered_menu = False
    try:
        sc = Scenario(tag, plan=args.plan or "", objective=objective,
                      max_steps=args.nrows, cfg_scale=args.cfg)
        obs = env.reset(sc)
        cur = obs.frame
        while executed < args.nrows:
            # System-2: the plan that will drive the next A actions (skipped in --null base mode).
            if args.null:
                plan = "(null / no plan — base DiT)"
            elif args.plan:
                plan = args.plan
            else:
                plan = pol.pl.generate_plan([cur], pol.device, instruction=objective,
                                            system=PLAN_SYS) or "advance"
            chunk = pol._sample_chunk(cur, "" if args.null else plan, args.cfg,
                                      plan_frames=[cur], null=args.null)  # (H, 25)
            for b in MENU_MASK.get(args.env, ()):     # keep the agent in gameplay (no pause-strand)
                chunk[:, b] = 0.0
            rows = env.apply_chunk_capture(chunk[:A], per_row)                 # A (row, frame, state)
            log.append(f"# re-plan#{cycle:02d}  plan='{plan}'")
            for ri, (row, frame, st) in enumerate(rows):
                st = {k: v for k, v in (st or {}).items() if v is not None}
                img = annotate(frame, row, cycle, ri, A, plan, args.env, st)
                Image.fromarray(img).save(f"{frame_dir}/f{fi:04d}.png"); fi += 1
                lx, ly = float(row[JLX]), float(row[JLY])
                btns = [lbl for idx, lbl in _BTN if float(row[idx]) > 0.5]
                ev = ""
                if st.get("dead") and not died:
                    ev = "   <<< DEATH"; died = True
                elif st.get("in_menu") and not entered_menu:
                    ev = "   <<< ENTERED MENU (stranded)"; entered_menu = True
                log.append(f"  exec{ri}: stick=({lx:+.2f},{ly:+.2f}) buttons={','.join(btns) or '-'}"
                           f"  state={st or '{}'}{ev}")
            cur = rows[-1][1]
            executed += A
            cycle += 1
            print(f"  re-plan#{cycle:02d}: '{plan[:42]}' -> {executed}/{args.nrows} rows  ({fi} frames)",
                  flush=True)
    finally:
        env.close()

    mp4 = f"{out_root}/play.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", "5",
                    "-i", f"{frame_dir}/f%04d.png", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4])
    with open(f"{out_root}/actions.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")
    # also drop a contact-sheet of every Nth executed frame for quick eyeballing
    print(f"\nsaved {fi} annotated frames -> {mp4}\n  per-action log -> {out_root}/actions.txt")


if __name__ == "__main__":
    main()
