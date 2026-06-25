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
    "chromium_bsu": "You are piloting a ship in a vertical shoot-em-up. What should the agent do next "
                    "to dodge enemies and shoot?",
    "castlevania_godot": "You are a whip-wielding hero in a side-scrolling action game. What next?",
    "blobwars": "You are a run-and-gun hero. What should the agent do next to advance and shoot enemies?",
    "witchblast": "You are in a top-down dungeon shooter. What should the agent do next to fight and "
                  "move through the room?",
}

# buttons to ZERO before applying, per env, so a spurious press can't strand the agent on a
# menu (e.g. TheXTech maps START->Enter = pause). Mirrors the Cave Story harness's GAMEPLAY_MASK:
# this keeps the agent IN GAMEPLAY so we observe real platforming failures (deaths) not pause-strands.
MENU_MASK = {"thextech": (19,), "sdlpop": (19,), "castlevania_godot": (19,),
             "solarus_zelda": (16, 19)}  # 19 = START; solarus also maps RTRIG(16) -> pause

# ---- GROUNDED System-2 prompt (best from planner_poc/sweep_prompts.py: A_broad_concrete) -----------
# The planner gets: control scheme (per genre) + the agent's OWN recent actions (no privileged state)
# + a light grounding nudge to use concrete control terms ("move RIGHT") not vague words ("forward").
GENRE_OF = {
    "stk": "racing", "trigger_rally": "racing", "xmoto": "racing", "dustracing": "racing",
    "thextech": "platformer", "sdlpop": "platformer", "castlevania_godot": "platformer",
    "blobwars": "platformer", "pekka_kana_2": "platformer", "mighty_retro_zero": "platformer",
    "solarus_zelda": "topdown", "witchblast": "topdown", "freedink": "topdown", "flare_arpg": "topdown",
    "chromium_bsu": "shmup", "opentyrian": "shmup", "starfighter": "shmup",
}
CONTROLS = {
    "racing": "Controls: steer LEFT or RIGHT, ACCELERATE, BRAKE.",
    "platformer": "Controls: move LEFT or RIGHT, look UP, crouch DOWN, JUMP, ATTACK. To reach a higher "
                  "ledge you must JUMP (moving sideways alone will not climb).",
    "topdown": "Controls: move LEFT, RIGHT, UP or DOWN, ATTACK, ACTION to interact.",
    "shmup": "Controls: move LEFT, RIGHT, UP or DOWN, FIRE.",
}
GROUND_INSTR = ("In one sentence, say what the agent should do next and briefly why, naming the "
                "concrete direction to move (e.g. left, right, up, down, jump) rather than vague words "
                "like 'forward', 'explore' or 'the image'.")


def _first_sentence(plan: str) -> str:
    """Keep the plan to ONE clean sentence: drop a leading 'Based on .../Looking at ...' preamble and
    cut after the first sentence so freeform plans stay concise (and never truncate mid-thought)."""
    p = plan.strip().strip('"').strip()
    p = p.replace("**", "").replace("*", "")        # strip stray markdown emphasis
    # strip a description-preamble clause up to the first comma if it starts with a stock opener.
    low = p.lower()
    for opener in ("based on", "looking at", "from the", "in the provided", "the analysis"):
        if low.startswith(opener) and "," in p:
            p = p.split(",", 1)[1].strip()
            p = p[:1].upper() + p[1:]
            break
    # first sentence only
    for sep in (". ", "! ", "? "):
        if sep in p:
            p = p.split(sep, 1)[0] + sep.strip()
            break
    return p.strip()
# direction/button decode for the action-history summary (the agent's OWN recent actions).
_HIST_BTN = [(18, "JUMP"), (20, "ATTACK"), (16, "ACCELERATE/FIRE"), (9, "BRAKE")]


def summarize_actions(rows, thresh=0.2):
    """Summarize the agent's recent executed action rows into a short, non-privileged history string
    (its own inputs only -- NO game state). E.g. 'Recent actions: moved RIGHT 12, JUMP 6 of last 18.'"""
    if not rows:
        return ""
    n = len(rows)
    cnt = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0}
    btn = {lbl: 0 for _, lbl in _HIST_BTN}
    for row in rows:
        lx, ly = float(row[JLX]), float(row[JLY])
        if lx < 0.5 - thresh:
            cnt["LEFT"] += 1
        elif lx > 0.5 + thresh:
            cnt["RIGHT"] += 1
        if ly < 0.5 - thresh:
            cnt["UP"] += 1
        elif ly > 0.5 + thresh:
            cnt["DOWN"] += 1
        for idx, lbl in _HIST_BTN:
            if float(row[idx]) > 0.5:
                btn[lbl] += 1
    parts = [f"{k} {v}" for k, v in cnt.items() if v]
    parts += [f"{k} {v}" for k, v in btn.items() if v]
    if not parts:
        return f"Recent actions (last {n} steps): mostly idle (no strong direction)."
    return f"Recent actions (last {n} steps): " + ", ".join(parts) + "."


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F = font(15); Fbig = font(19)

# game-agnostic System-2 prompt so generated plans aren't biased by the platformer DEFAULT_SYS.
PLAN_SYS = ("You are the high-level planner for an agent playing a 2D video game. You see the most "
            "recent frames (oldest first). Decide the single best next move.")


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
    ap.add_argument("--fps", type=int, default=5, help="output video framerate (lower = slower playback)")
    ap.add_argument("--nframes", type=int, default=4,
                    help="number of recent frames the TEXT planner sees (multi-frame -> perceives "
                         "dynamics like a breaking tile). DiT conditioning stays single-frame.")
    ap.add_argument("--frame-stride", type=int, default=3,
                    help="stride between the planner's frames (in captured-frame units), so the window "
                         "spans more time without redundant near-identical frames.")
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--plan", default=None, help="fixed plan text; if omitted, generate a fresh plan each cycle")
    ap.add_argument("--null", action="store_true",
                    help="BASE ablation: run the unconditioned base DiT (no System-2 plan / null plan) "
                         "to compare against the plan-conditioned A=2/A=4 runs.")
    ap.add_argument("--boot-wait", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None, help="seed torch/numpy for a distinct sample")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.seed is not None:
        import torch as _t
        _t.manual_seed(args.seed); np.random.seed(args.seed)

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
        sys_prompt = PLAN_SYS          # role only -- injecting a control SCHEME makes plans terse
        recent_rows = []          # the agent's own recent executed action rows (for grounded history)
        frame_hist = [cur]        # rolling window of recently captured frames (for multi-frame planning)
        while executed < args.nrows:
            # System-2: freeform-but-GROUNDED plan over a multi-frame WINDOW (so it can perceive
            # dynamics like a breaking tile) + the agent's OWN recent actions (no privileged state).
            # DiT conditioning stays single-frame (it was trained that way); only the TEXT planner gets
            # the window. Keep the objective (elicits a descriptive plan); the grounding nudge only
            # anchors the DIRECTION word (left/right/up/down) -- it does NOT compress the sentence.
            if args.null:
                plan = "(null / no plan — base DiT)"
            elif args.plan:
                plan = args.plan
            else:
                win = frame_hist[::-1][::args.frame_stride][:args.nframes][::-1]  # oldest->newest
                # feed the frame WINDOW (perception of dynamics) but do NOT announce it verbosely --
                # a "you see N frames" preamble makes the 2B ramble "Based on the provided frames...".
                instr = f"{objective} {GROUND_INSTR}"
                plan = pol.pl.generate_plan(win, pol.device, instruction=instr,
                                            system=sys_prompt, max_new_tokens=40) or "advance"
                plan = _first_sentence(plan)
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
                frame_hist.append(frame)
            recent_rows.extend(r for r, _, _ in rows)
            recent_rows = recent_rows[-H:]
            frame_hist = frame_hist[-(args.nframes * args.frame_stride + 2):]
            cur = rows[-1][1]
            executed += A
            cycle += 1
            print(f"  re-plan#{cycle:02d}: '{plan[:42]}' -> {executed}/{args.nrows} rows  ({fi} frames)",
                  flush=True)
    finally:
        env.close()

    mp4 = f"{out_root}/play.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", str(args.fps),
                    "-i", f"{frame_dir}/f%04d.png", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4])
    with open(f"{out_root}/actions.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")
    # also drop a contact-sheet of every Nth executed frame for quick eyeballing
    print(f"\nsaved {fi} annotated frames -> {mp4}\n  per-action log -> {out_root}/actions.txt")


if __name__ == "__main__":
    main()
