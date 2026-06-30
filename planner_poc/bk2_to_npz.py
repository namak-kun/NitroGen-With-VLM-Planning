"""bk2_to_npz.py — regenerate a demo.npz from a stable-retro movie (demo.bk2) by replaying the input log
through the in-process SNES emulator from the demo's initial.state. Used to recover demos whose live capture
dropped the npz (only mp4+bk2+states survived).

The bk2 is a zip {Input Log.txt, Header.txt, Core.bin}. Header 'MovieVersion Retro' (stable-retro). Input Log
header names the P1 button columns; each frame line '|<sys>|<12 buttons>|' marks pressed buttons by their
letter (else '.'). Replaying is 1:1 per emulator frame (demo#1: bk2 frames == npz actions exactly).

Writes npz matching demo_bc's expected format: observations (N+1,H,W,3 uint8), actions (N,12 uint8) in the
META button order [B,Y,SELECT,START,UP,DOWN,LEFT,RIGHT,A,X,L,R], rewards/dones (zeros), buttons (12,).

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc'
  # validate the method against a demo that HAS a ground-truth npz:
  $RUN .venv/bin/python planner_poc/bk2_to_npz.py --demo docs/demos/demos/SuperMetroid-Snes/20260629-132934 \
       --rom "Game data/Super Metroid.sfc" --validate
  # regenerate the missing one:
  $RUN .venv/bin/python planner_poc/bk2_to_npz.py --demo docs/demos/demos/SuperMetroid-Snes/20260629-133403 \
       --rom "Game data/Super Metroid.sfc" --out docs/demos/demos/SuperMetroid-Snes/20260629-133403/demo.npz
"""
from __future__ import annotations
import argparse, gzip, io, os, sys, zipfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))

# META button order expected by demo_bc.map_action (SNES 12-button)
META = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
META_I = {b: i for i, b in enumerate(META)}


def parse_bk2(path):
    """Return (frames list of set[str] button-NAMEs uppercased in SNES naming, header_button_order)."""
    with zipfile.ZipFile(path) as z:
        log = z.read("Input Log.txt").decode("utf-8", "replace").splitlines()
    # header line like: P1 R|P1 L|P1 X|P1 A|P1 Right|...|P1 B|
    cols = None
    frames = []
    for ln in log:
        s = ln.strip()
        if not s or s in ("[Input]", "[/Input]"):
            continue
        if s.startswith("P1 ") or (cols is None and "P1" in s):
            cols = [c.strip().replace("P1 ", "").upper() for c in s.split("|") if c.strip()]
            continue
        if s.startswith("|"):
            # split sections by '|'; the section whose length == len(cols) is the P1 button section
            secs = s.split("|")[1:-1]  # drop leading/trailing empties
            btn_sec = None
            for sec in secs:
                if cols is not None and len(sec) == len(cols):
                    btn_sec = sec
                    break
            if btn_sec is None:               # fallback: last (widest) section
                btn_sec = max(secs, key=len)
            pressed = {cols[i] for i, ch in enumerate(btn_sec) if ch not in (".", " ")}
            frames.append(pressed)
    return frames, cols


def buttons_to_action(pressed):
    """set of SNES button NAMEs -> (12,) uint8 in META order."""
    a = np.zeros(12, np.uint8)
    for b in pressed:
        if b in META_I:
            a[META_I[b]] = 1
    return a


def replay(rom, init_state_path, frames):
    from nitrogen.eval.envs.snes_env import SnesEnv
    env = SnesEnv(rom_path=rom)
    raw = open(init_state_path, "rb").read()
    state = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    env.load_state(state)
    obs = [env.frame().copy()]
    acts = []
    for pressed in frames:
        names = {b for b in pressed}
        env._emu_set_buttons(names)
        env._emu_step(1)
        obs.append(env.frame().copy())
        acts.append(buttons_to_action(pressed))
    env.close()
    obs = np.stack(obs)
    if len(obs) > 1 and float(obs[0].mean()) < 1.0:   # core hasn't rendered pre-step -> obs[0] blank; the
        obs[0] = obs[1]                               # initial.state stores no frame. 1-frame delta is negligible.
    return obs, np.stack(acts) if acts else np.zeros((0, 12), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="demo dir (has demo.bk2 + initial.state)")
    ap.add_argument("--rom", required=True)
    ap.add_argument("--out", default=None, help="output npz (default <demo>/demo.npz)")
    ap.add_argument("--validate", action="store_true", help="compare to existing demo.npz, do not overwrite")
    args = ap.parse_args()

    bk2 = os.path.join(args.demo, "demo.bk2")
    init = os.path.join(args.demo, "initial.state")
    frames, cols = parse_bk2(bk2)
    print(f"[bk2] {len(frames)} input frames; columns={cols}", flush=True)
    npress = sum(len(f) for f in frames)
    print(f"[bk2] total button-presses={npress}", flush=True)

    obs, acts = replay(args.rom, init, frames)
    print(f"[replay] observations={obs.shape} actions={acts.shape}  per-button sums={acts.sum(0).tolist()}  (META {META})", flush=True)

    if args.validate:
        gt = np.load(os.path.join(args.demo, "demo.npz"), allow_pickle=True)
        ga, go = gt["actions"], gt["observations"]
        print(f"[validate] GT actions={ga.shape} obs={go.shape}")
        n = min(len(acts), len(ga))
        amatch = float((acts[:n] == ga[:n]).all(axis=1).mean()) if n else 0.0
        print(f"[validate] ACTION exact-row match over {n}: {amatch:.4f}  (per-button GT sums={ga.sum(0).tolist()})")
        m = min(len(obs), len(go))
        fdiff = float(np.abs(obs[:m].astype(np.int16) - go[:m].astype(np.int16)).mean())
        f0 = float(np.abs(obs[0].astype(np.int16) - go[0].astype(np.int16)).mean())
        print(f"[validate] OBS mean|diff| over {m} frames: {fdiff:.3f}  (frame0 diff {f0:.3f}); identical={fdiff < 1e-6}")
        return

    out = args.out or os.path.join(args.demo, "demo.npz")
    np.savez_compressed(out, observations=obs, actions=acts,
                        rewards=np.zeros(len(acts), np.float32), dones=np.zeros(len(acts), bool),
                        buttons=np.array(META, dtype="<U6"))
    print(f"[bk2->npz] wrote {out}  ({len(acts)} actions, {len(obs)} frames)", flush=True)


if __name__ == "__main__":
    main()
