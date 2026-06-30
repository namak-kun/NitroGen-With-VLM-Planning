"""ingest_external_state.py -- make an EXTERNALLY-authored emulator savestate usable by our RL envs.

@namak-kun authors save states directly in external emulators (level start/end, dungeon starts). A
stable-retro RL env loads states via `env.load_state(bytes)` = the libretro core's raw serialize blob.
This tool takes an external `.state` file, NORMALIZES it (strips a RetroArch RASTATE wrapper if present),
verifies it actually LOADS into the matching core, reports the resulting reward-var value, and copies the
normalized raw blob to tmp/states/<game>/<label>.state for RL/eval to use as a start state.

CORE/FORMAT COMPATIBILITY (verified on this box):
  Genesis : core = Genesis Plus GX 1.7.5  -> state is a binary blob starting `GENPLUS-GX 1.7.5`.
            Loadable ONLY from the SAME core version (use RetroArch's Genesis Plus GX 1.7.5 core;
            standalone Kega/BlastEm/Gens use different formats -> will NOT load).
  SNES    : core = snes9x (libretro)      -> state starts `#!s9xsnp:0009` (snes9x native snapshot v9).
            snes9x's snapshot format is shared by standalone Snes9x AND the libretro core; a state from
            either should load IF its snapshot version == 0009 (older/newer Snes9x may bump the version).
  RetroArch wraps savestates in a `RASTATE` container (the raw core block is inside a `MEM ` chunk); this
  tool detects + unwraps that automatically.

The robust rule: author ONE test state, run this tool, confirm `LOADS = True` before authoring many.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc'
  $RUN CUDA_VISIBLE_DEVICES="" .venv/bin/python planner_poc/ingest_external_state.py \
      --state /path/to/external.state --game SonicTheHedgehog2-Genesis-v0 --system Genesis \
      --rom "Game data/Sonic The Hedgehog 2.md" --label sonic_zone2_start
  # SNES example:
  ... --state smw_1-1_start.state --rom "Game data/Super Mario World.sfc" --system Snes --label smw_1-1_start
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import os as _os; _R = _os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))


def unwrap_rastate(blob: bytes) -> tuple[bytes, str]:
    """If `blob` is a RetroArch RASTATE container, return the inner core serialize block (the `MEM `
    chunk); else return the blob unchanged. RASTATE layout: b'RASTATE' magic + 1 version byte, then a
    sequence of chunks each = 4-byte ASCII id + int32-LE size + data. The core state is the `MEM ` id."""
    if blob[:7] != b"RASTATE":
        return blob, "raw"
    import struct
    off = 8  # 'RASTATE' (7) + 1 version byte
    while off + 8 <= len(blob):
        cid = blob[off:off + 4]
        (size,) = struct.unpack_from("<i", blob, off + 4)
        data_off = off + 8
        if cid == b"MEM ":
            return blob[data_off:data_off + size], "rastate:MEM"
        if cid == b"END " or size < 0 or data_off + size > len(blob):
            break
        off = data_off + size
    return blob, "rastate:unparsed"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="external savestate file to ingest")
    ap.add_argument("--rom", default="Game data/Sonic The Hedgehog 2.md")
    ap.add_argument("--game", default="SonicTheHedgehog2-Genesis-v0",
                    help="stable-retro integration name (for reward-var readout); optional for raw SNES")
    ap.add_argument("--system", default="Genesis")
    ap.add_argument("--reward-var", default="screen_x")
    ap.add_argument("--label", default=None, help="output label -> tmp/states/<game>/<label>.state")
    ap.add_argument("--out-root", default="tmp/states")
    args = ap.parse_args()

    raw = open(args.state, "rb").read()
    blob, kind = unwrap_rastate(raw)
    print(f"[ingest] {args.state}: {len(raw)} bytes ({kind}); core block {len(blob)} bytes")
    print(f"[ingest] core header: {blob[:24]!r}")

    # Load via the matching env/core and verify.
    from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
    loaded = False
    rv = None
    try:
        env = RetroRLEnv(rom_path=args.rom, game=args.game, system=args.system, reward_var=args.reward_var)
        env.reset()
        env.load_state(blob)
        loaded = True
        try:
            rv = env._var(args.reward_var)
        except Exception:
            rv = None
        # prove it's a live, steppable state: one RIGHT chunk should not error
        row = np.full((25,), 0.5, np.float32); row[21] = 1.0
        env.step(np.tile(row, (18, 1)))
        env.close()
    except Exception as e:
        print(f"[ingest] LOADS = False  -- {type(e).__name__}: {e}")
        print("[ingest] -> the state is NOT from a matching core/version. See the compatibility note in "
              "this file's docstring (use RetroArch with the bundled core version, or a snes9x snapshot v9).")
        return 1

    print(f"[ingest] LOADS = True   reward_var({args.reward_var}) = {rv}")
    if args.label:
        gdir = os.path.join(args.out_root, args.game)
        os.makedirs(gdir, exist_ok=True)
        outp = os.path.join(gdir, f"{args.label}.state")
        open(outp, "wb").write(blob)
        import json, time
        json.dump({"game": args.game, "system": args.system, "reward_var": args.reward_var,
                   "reward_value": rv, "source": os.path.abspath(args.state), "kind": kind,
                   "ts": time.time()}, open(outp.replace(".state", ".json"), "w"), indent=2)
        print(f"[ingest] normalized raw state -> {outp} (+ .json). Use it as an RL/eval start state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
