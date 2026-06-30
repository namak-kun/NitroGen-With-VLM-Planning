"""vlm_narrate_demo.py — generate VLM "System-2" narration/objectives over a recorded demo, using the SAME
planner stack (PlanEncoder.generate_plan) but optionally a BIGGER same-family VLM (QWEN env var, e.g.
Qwen/Qwen3.5-9B). One-off proof + a way to vet VLM plan quality vs the owner's hand narration.

Two modes:
  --at-human-notes : narrate at the SAME frames as the owner's narration.json, printing HUMAN vs VLM side by
                     side (quality vetting on the 2 narrated SMW demos).
  --every N        : narrate every N seconds across the whole demo (for un-narrated demos / other games).

The VLM sees a short window of recent frames (oldest->newest) sampled from demo.npz observations, exactly as
deployment does. Output -> docs/demos/demos/<Game>/<ts>/vlm_narration.json (distinct from the human narration.json).

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-9B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/vlm_narrate_demo.py \
      --demo docs/demos/demos/SuperMarioWorld-Snes/20260627-105939 --at-human-notes
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.planner import PlanEncoder, PlannerConfig

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-9B")

# Per-game CONTROL SCHEMA: telling the VLM what each button does gives it the MANEUVER VOCABULARY (down=duck,
# A=spin-jump) so it can NAME situational moves (duck under a bullet, spin-jump a tough enemy) instead of
# defaulting to generic 'jump right'. Keyed by game id. (Owner request: "give it a control schema of what each
# key press does".)
CONTROL_SCHEMA = {
    "smw": (
        "Controls (Super Mario World):\n"
        "- Left / Right: walk; hold Run to dash. At max running speed Mario runs UP triangular blocks and "
        "up adjacent pipes/walls.\n"
        "- Down: DUCK/crouch — slip under high projectiles (e.g. a Bullet Bill) and slide down slopes; on top "
        "of a pipe, press Down to ENTER it.\n"
        "- Up: look up; enter a DOOR (stand in front, press Up); grab a ROPE (jump and press Up, then "
        "Up/Down to climb); enter a suspended pipe (jump and hold Up).\n"
        "- B: NORMAL JUMP — higher; bounce off most enemies.\n"
        "- A: SPIN JUMP — lower spinning jump that defeats/kills most enemies in one hit, bounces safely off "
        "spiked hazards a normal stomp can't, and breaks rotating blocks below; also dismounts Yoshi.\n"
        "- Y/X: RUN/accelerate; hold to GRAB and carry a shell/item (look Up + release to throw it up); as "
        "Fire Mario, Y/X SHOOTS a fireball; as Caped Mario, Y is a cape SPIN attack.\n"
        "- FENCE/NET: regular-jump into a fence and press Up or Down to GRAB and climb it; press Y to PUNCH "
        "enemies through the fence or flip around a revolving door.\n"
        "- CAPE: run at full speed then jump to FLY; spin-jump to break blocks.\n"
        "- SWIM (in water): press a jump button (A/B) to rise, hold Up to rise fast, hold Down to swim level.\n"
        "- L/R shoulder: scroll the view to see ahead."
    ),
    "smbas": (
        "Controls (Super Mario Bros, All-Stars / SMB1 — standard SNES scheme):\n"
        "- Left / Right: walk; hold Run to dash.\n"
        "- Down: DUCK/crouch; standing on a pipe, press Down to ENTER it.\n"
        "- A / B: JUMP — stomp an enemy from above; longer hold = higher; in water, press to SWIM up.\n"
        "- X / Y: RUN/accelerate; as Fire Mario, press to SHOOT a fireball.\n"
        "- (SMB1 has NO spin-jump and NO cape.)"
    ),
    "mmx": (
        "Controls (Mega Man X — standard SNES scheme):\n"
        "- Left / Right: move X. Up/Down: aim up/down; Up climbs ladders.\n"
        "- Y: JUMP (hold for higher jumps; jump into a wall to slide, then jump+away to WALL-JUMP).\n"
        "- B: SHOOT the X-Buster (hold to charge: bigger charged shot on release).\n"
        "- A: DASH (a fast ground dash; dash then jump = a long dash-jump across gaps).\n"
        "- X: fire the equipped SPECIAL WEAPON. L/R: cycle special weapons."
    ),
    "sonic": (
        "Controls (Sonic the Hedgehog 2 — Genesis):\n"
        "- Left / Right: walk and build up running speed.\n"
        "- Down: look down; while RUNNING, hold Down to ROLL into a ball through enemies.\n"
        "- A / B / C (any): JUMP (a spinning jump that damages enemies on contact).\n"
        "- Down + jump (A/B/C) while standing still: SPIN DASH — charge in place, release to dash off at "
        "high speed (used to climb slopes/loops and break through).\n"
        "- The goal is forward momentum: build speed, jump over hazards, spin-dash up slopes and loops."
    ),
    "metroid": (
        "Controls (Super Metroid, Samus — standard SNES scheme):\n"
        "- Left / Right: run. Up: aim the weapon diagonally/straight up. Down: kneel; press again to morph "
        "into a ball (rolls through low gaps); Down in the air aims the weapon down.\n"
        "- A: JUMP (hold to spin-jump).\n"
        "- X: FIRE the beam/missiles (hold for rapid/charged shots); in morph-ball, lays a bomb.\n"
        "- B: DASH (hold to run fast / build speed for a speed-booster).\n"
        "- Y: cancel/switch special item. L: aim down-angle. R: aim up-angle (L+R kneeling = aim straight up).\n"
        "- In a boss fight (e.g. Ridley): keep distance, dodge swoops/fireballs by running and jumping, and "
        "repeatedly FIRE. In an escape sequence: dash for the exit, do not stop."
    ),
}

# Grounded System-2 prompt: abstracted strategy (NOT a frame-by-frame button log). The owner wants plans that
# are LESS terse than the BATTERY plans but LESS granular/noisy than the raw human narration -- a clean,
# higher-level objective naming the concrete next action + why.
SYS_BASE = ("You are the high-level strategist for an agent playing a 2D action platformer. You see the most "
            "recent game frames in time order (oldest first, newest last). Judge the situation and give the "
            "next short objective.")
INSTR = ("In ONE short imperative sentence (max ~12 words), state the player's next objective: the concrete "
         "action and direction (e.g. 'jump right over the gap', 'duck under the bullet', 'spin-jump the Rex', "
         "'climb the net'). Use the right maneuver for the threat — if a tall/armored enemy is ahead consider a "
         "spin-jump, if a projectile is coming consider ducking. No step-by-step button presses, no commentary.")


def _window(obs, frame, w, fps_stride):
    """A list of `w` frames ending at `frame` (oldest->newest), strided by fps_stride (downsample 60fps)."""
    idxs = [max(0, frame - (w - 1 - k) * fps_stride) for k in range(w)]
    return [np.asarray(obs[i]) for i in idxs]


# Per-game demo.npz button ORDER (the npz stores raw console buttons; ORDER DIFFERS by console).
# SNES games: B,Y,SELECT,START,UP,DOWN,LEFT,RIGHT,A,X,L,R. Genesis (Sonic): B,A,MODE,START,UP,DOWN,LEFT,RIGHT,C,Y,X,Z.
_BTN_SNES = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
_BTN_GENESIS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
_BTN = _BTN_SNES  # back-compat default


def _btn_order(game):
    return _BTN_GENESIS if game == "sonic" else _BTN_SNES


def render_action(a12, game="smw"):
    """a12: (12,) binary. -> e.g. 'Right + Spin-jump(A)' / 'Down=DUCK' / 'idle'. Uses the per-game button order."""
    on = {b for b, v in zip(_btn_order(game), a12) if v}
    parts = []
    if "LEFT" in on: parts.append("Left")
    if "RIGHT" in on: parts.append("Right")
    if "UP" in on: parts.append("Up")
    if "DOWN" in on: parts.append("Down=DUCK" if game in ("smw", "smbas") else "Down")
    if game == "smw":
        if "B" in on or "A" in on: parts.append("Jump")
        if "A" in on: parts.append("Spin-jump(A)")
        if "Y" in on or "X" in on: parts.append("Run/Grab")
        # contextual advanced hints (the VLM resolves which from the frame): Up while jumping = grab a
        # fence/rope/door/suspended-pipe; holding Up+Run with no jump = look up / carry+throw an item up.
        if "UP" in on and ("B" in on or "A" in on): parts.append("Up=grab-fence/rope/door")
    elif game == "smbas":
        if "A" in on or "B" in on: parts.append("Jump")        # also Swim in water
        if "Y" in on or "X" in on: parts.append("Run/Fireball")
    elif game == "metroid":
        if "A" in on: parts.append("Jump")
        if "X" in on: parts.append("Fire")
        if "B" in on: parts.append("Dash")
        if "DOWN" in on: parts.append("Kneel/Morph")
        if "L" in on or "R" in on or "UP" in on: parts.append("Aim")
    elif game == "mmx":
        if "Y" in on: parts.append("Jump")
        if "B" in on: parts.append("Shoot")
        if "A" in on: parts.append("Dash")
        if "X" in on: parts.append("Special-weapon")
        if "UP" in on: parts.append("Up/climb")
    elif game == "sonic":
        jump = ("A" in on) or ("B" in on) or ("C" in on)
        if jump and "DOWN" in on: parts.append("Spin-dash")
        elif jump: parts.append("Jump")
        elif "DOWN" in on: parts.append("Roll/crouch")
    else:
        if "A" in on or "B" in on: parts.append("Jump")
        if "Y" in on or "X" in on: parts.append("Shoot")
    return " + ".join(parts) if parts else "idle"


def _interleaved_items(obs, acts, start, end, n_frames, game):
    """Build [frame, 'action: ...', frame, 'action: ...', ...] over [start,end), sampling n_frames evenly and
    summarizing the gold actions held between consecutive sampled frames (so the VLM sees frame->action->frame)."""
    end = min(end, len(obs) - 1)
    if end <= start:
        return []
    idxs = np.linspace(start, end, n_frames).astype(int)
    items = []
    for k, fi in enumerate(idxs):
        items.append(np.asarray(obs[fi]))
        if k < len(idxs) - 1:
            seg = acts[fi:idxs[k + 1]]                      # actions until the next sampled frame
            # most-common non-idle maneuver in this segment (else idle)
            rendered = [render_action(a, game) for a in seg]
            nonidle = [r for r in rendered if r != "idle"]
            lab = max(set(nonidle), key=nonidle.count) if nonidle else "idle"
            items.append(f"[action held next: {lab}]")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="demo dir (has demo.npz, optionally narration.json)")
    ap.add_argument("--qwen", default=QWEN)
    ap.add_argument("--window", type=int, default=4, help="frames per objective")
    ap.add_argument("--frame-stride", type=int, default=8, help="emu-frame gap between the window's frames")
    ap.add_argument("--at-human-notes", action="store_true", help="narrate at the human narration.json frames")
    ap.add_argument("--every", type=float, default=0.0, help="else: narrate every N seconds across the demo")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--think", action="store_true", help="allow reasoning <think> (default: disabled for direct answers)")
    ap.add_argument("--game", default="smw", help="control-schema key (smw/smbas/mmx); '' = no schema")
    ap.add_argument("--no-schema", action="store_true", help="omit the control schema (A/B baseline)")
    ap.add_argument("--interleave-actions", action="store_true",
                    help="feed the VLM a SEGMENT of (frame, gold-action) pairs ending at each point, so it sees "
                         "what was actually pressed (grounds the narration, esp. for duck/spin-jump)")
    ap.add_argument("--seg-frames", type=int, default=8, help="interleaved: # frames sampled per segment")
    ap.add_argument("--seg-seconds", type=float, default=2.0, help="interleaved: segment length (s) before each point")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    schema = "" if (args.no_schema or not CONTROL_SCHEMA.get(args.game)) else CONTROL_SCHEMA[args.game]
    SYS = SYS_BASE + ("\n\n" + schema if schema else "")
    print(f"[schema] {'ON ('+args.game+')' if schema else 'OFF'}", flush=True)

    z = np.load(os.path.join(args.demo, "demo.npz"), allow_pickle=True)
    obs = z["observations"]; acts = z["actions"]; N = len(obs); fps = 60.0
    INSTR_IL = ("You are shown a short clip as alternating game frames and the gold action the expert HELD "
                "right after each frame (using the control scheme above, e.g. 'Jump', 'Fire', 'Dash', "
                "'Kneel/Morph'). In ONE short imperative sentence (max ~12 words), state the expert's "
                "intent/objective for this clip, naming the maneuver the actions imply (per the controls above). "
                "No button list, no commentary.")
    human = None
    hp = os.path.join(args.demo, "narration.json")
    if os.path.exists(hp):
        human = {e["frame"]: e["text"] for e in json.load(open(hp)).get("entries", [])}

    # choose the frames to narrate at
    if args.at_human_notes and human:
        frames = sorted(human.keys())
    else:
        step = int((args.every or 2.0) * fps)
        frames = list(range(0, N - 1, max(1, step)))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.qwen} ...", flush=True)
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=args.qwen)); pl.load()
    if next(pl.backbone.parameters()).device != torch.device(dev):
        pl.backbone.to(dev)
    nparams = sum(p.numel() for p in pl.backbone.parameters()) / 1e9
    print(f"loaded {args.qwen} ({nparams:.1f}B params); narrating {len(frames)} points on {os.path.basename(args.demo)}\n", flush=True)

    entries = []
    out = args.out or os.path.join(args.demo, "vlm_narration.json")
    for fr in frames:
        if args.interleave_actions:
            start = max(0, fr - int(args.seg_seconds * fps))
            items = _interleaved_items(obs, acts, start, fr + 1, args.seg_frames, args.game)
            plan = pl.generate_interleaved(items, dev, instruction=INSTR_IL, system=SYS,
                                           max_new_tokens=args.max_new,
                                           enable_thinking=(None if args.think else False)) or ""
        else:
            win = _window(obs, fr, args.window, args.frame_stride)
            plan = pl.generate_plan(win, dev, instruction=INSTR, system=SYS, max_new_tokens=args.max_new,
                                    enable_thinking=(True if args.think else False)) or ""
        plan = plan.replace("\n", " ").strip()
        if "." in plan:
            plan = plan[:plan.index(".") + 1]
        e = {"frame": int(fr), "t": round(fr / fps, 2), "vlm": plan}
        if human is not None and fr in human:
            e["human"] = human[fr]
        entries.append(e)
        if human is not None and fr in human:
            print(f"  t={e['t']:>5}s  HUMAN: {human[fr][:72]}")
            print(f"            VLM  : {plan[:72]}\n", flush=True)
        else:
            print(f"  t={e['t']:>5}s  VLM: {plan}", flush=True)
        # write INCREMENTALLY so a mid-run crash never loses the narration
        json.dump({"demo": os.path.basename(args.demo), "qwen": args.qwen, "n": len(entries),
                   "interleave_actions": args.interleave_actions, "schema": bool(schema),
                   "system": SYS, "instruction": INSTR_IL if args.interleave_actions else INSTR,
                   "entries": entries}, open(out, "w"), indent=1)

    print(f"\n[vlm-narrate] -> {out}  ({len(entries)} points)", flush=True)


if __name__ == "__main__":
    main()
