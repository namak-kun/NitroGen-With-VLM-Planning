"""idm_gen_emulator_data.py -- generate GROUND-TRUTH (frame, action) trajectories from an emulator for
training an Inverse Dynamics Model (IDM / VPT-style). WE drive the emulator, so every action is exact.

Key design for IDM learnability: actions must CAUSE visible motion. So we sample SEGMENT-based macros
(hold a direction/button combo for a run of frames, like a human) rather than per-frame noise -- this
produces sustained, clearly-attributable frame changes.

Output: one .npz per trajectory in <out>/:
  frames   (T, h, w, 3) uint8     -- the frame AFTER applying actions[t]
  actions  (T, 25)      float32   -- the NitroGen action row applied at step t
  buttons  (T, B)       uint8     -- compact multi-label target for the IDM (see BTN_VOCAB)
  meta.json: rom, system, btn_vocab, frame size, frames_per_row

Usage:
  python idm_gen_emulator_data.py --rom "Game data/Super Mario World.sfc" --system snes \
     --steps 4000 --warmup 240 --out docs/idm_data/smw --res 128
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]

# Compact IDM target vocabulary (order matters; index = bit position in the `buttons` array).
BTN_VOCAB = ["LEFT", "RIGHT", "UP", "DOWN", "B", "A", "Y", "X", "L", "R", "START"]

# NitroGen 25-dim action indices we set. dpad 1-4, SOUTH=18(B), EAST=5(A), WEST=20(Y), NORTH=10(X),
# LEFT_SHOULDER=7, RIGHT_SHOULDER=14, START=19. (JLX/JLY 21/22 left at neutral 0.5; we use the dpad.)
IDX = {"DOWN": 1, "LEFT": 2, "RIGHT": 3, "UP": 4, "A": 5, "L": 7, "X": 10,
       "R": 14, "B": 18, "START": 19, "Y": 20}


def _row(buttons: set[str]) -> np.ndarray:
    a = np.zeros((25,), np.float32)
    a[21] = a[22] = 0.5                      # neutral left stick
    for b in buttons:
        if b in IDX:
            a[IDX[b]] = 1.0
    return a


def _buttons_to_multilabel(buttons: set[str]) -> np.ndarray:
    return np.asarray([1 if v in buttons else 0 for v in BTN_VOCAB], np.uint8)


# Segment macros for a side-scrolling platformer (the dominant genre in our ROM set). Each is a button
# SET held for a run of frames. Weighted to emphasize movement + jumps (the actions we most want to
# teach), with idle/neutral for contrast.
MACROS = [
    (("RIGHT",), 6),
    (("RIGHT", "B"), 6),                     # run right (B = dash/run in SMW)
    (("RIGHT", "A"), 5),                     # jump-right (A = jump in SMW)
    (("RIGHT", "B", "A"), 5),                # run-jump right
    (("LEFT",), 3),
    (("LEFT", "A"), 2),
    (("A",), 3),                             # stationary jump
    (("DOWN",), 1),                          # crouch / pipe
    (("UP",), 1),
    ((), 2),                                 # idle (neutral) -- contrast for the IDM
]


def sample_action_sequence(n: int, rng: np.random.Generator) -> list[set[str]]:
    """Produce n per-frame button-sets via random segments (hold a macro 6-30 frames, then switch)."""
    macros = [m for m, _ in MACROS]
    weights = np.asarray([w for _, w in MACROS], float); weights /= weights.sum()
    seq: list[set[str]] = []
    while len(seq) < n:
        mi = rng.choice(len(macros), p=weights)
        hold = int(rng.integers(6, 31))
        bset = set(macros[mi])
        seq.extend([set(bset) for _ in range(hold)])
    return seq[:n]


def make_env(rom: str, system: str):
    if system == "snes":
        from nitrogen.eval.envs.snes_env import SnesEnv
        return SnesEnv(rom_path=rom, freeze_during_inference=False)
    if system == "gba":
        from nitrogen.eval.envs.mgba_env import MgbaEnv
        return MgbaEnv(rom_path=rom, freeze_during_inference=False)
    raise SystemExit(f"unknown system {system!r} (use snes|gba)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", required=True)
    ap.add_argument("--system", choices=["snes", "gba"], required=True)
    ap.add_argument("--steps", type=int, default=4000, help="frames of gameplay to record")
    ap.add_argument("--warmup", type=int, default=240, help="START/B mashing frames to clear title")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=128, help="square frame size stored (downscaled)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rom = args.rom if os.path.isabs(args.rom) else str(REPO / args.rom)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = make_env(rom, args.system)

    # warmup: mash START then B to get past title/file-select into gameplay
    warm = ([{"START"}] * (args.warmup // 2)) + ([{"B"}] * (args.warmup - args.warmup // 2))
    seq = warm + sample_action_sequence(args.steps, rng)

    frames, actions, buttons = [], [], []
    chunk = np.stack([_row(b) for b in seq], 0)              # (N,25)
    rows = env.apply_chunk_capture(chunk, per_row=None)      # [(row, frame, state)] aligned
    for i, (row, frame, _state) in enumerate(rows):
        if i < args.warmup:                                  # don't train the IDM on the title screen
            continue
        f = np.asarray(frame, np.uint8)
        im = Image.fromarray(f).resize((args.res, args.res), Image.BILINEAR)
        frames.append(np.asarray(im, np.uint8))
        actions.append(row.astype(np.float32))
        buttons.append(_buttons_to_multilabel(seq[i]))
    env.close()

    frames = np.stack(frames); actions = np.stack(actions); buttons = np.stack(buttons)
    np.savez_compressed(out / "traj.npz", frames=frames, actions=actions, buttons=buttons)
    (out / "meta.json").write_text(json.dumps({
        "rom": os.path.basename(rom), "system": args.system, "btn_vocab": BTN_VOCAB,
        "res": args.res, "n": int(frames.shape[0])}, indent=2))
    # quick diagnostics: per-button positive rate + mean frame-to-frame motion
    motion = float(np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)).mean())
    rates = {v: round(float(buttons[:, j].mean()), 3) for j, v in enumerate(BTN_VOCAB)}
    print(f"[idm_gen] {frames.shape[0]} frames {frames.shape[1:]} -> {out}/traj.npz")
    print(f"[idm_gen] mean frame-motion {motion:.2f} (want >2 = actions cause visible change)")
    print(f"[idm_gen] button positive-rates: {rates}")


if __name__ == "__main__":
    main()
