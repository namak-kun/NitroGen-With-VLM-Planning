#!/usr/bin/env python3
"""Characterize TheXTech levels with a simple scripted right+jump runner."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

REPO = "/home/t-nagupta/NitroGen-With-VLM-Planning"
os.environ.setdefault("NITROGEN_REPO", REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "planner_poc"))

from nitrogen.eval.envs.proc_rl_env import ProcRLEnv  # noqa: E402

JLX = 21
JLY = 22
I_SOUTH = 18

DEFAULT_LEVELS = ["intro"] + [f"level{i}" for i in range(1, 9)] + ["bonus1", "level20"]


@dataclass
class Result:
    name: str
    boots: bool
    start_x: float | None
    max_x: float | None
    total_dx: float | None
    outcome: str
    classification: str
    chunks: int


def level_path(name: str) -> str:
    if name.endswith(".lvlx") or "/" in name:
        return name
    return f"worlds/the first adventure/{name}.lvlx"


def make_chunk(A: int, jump_every: int, jump_len: int, hold_run: bool) -> np.ndarray:
    chunk = np.zeros((A, 25), dtype=np.float32)
    chunk[:, 21:25] = 0.5    # neutral sticks; button channels are 0 unless explicitly pulsed
    chunk[:, JLX] = 1.0      # canonical stick: +x = right (0..1 normalized path)
    chunk[:, JLY] = 0.5      # neutral vertical stick
    if hold_run:
        chunk[:, 16] = 1.0   # optional run/hold-item (not used by default)
    for i in range(A):
        if (i % jump_every) < jump_len:
            chunk[i, I_SOUTH] = 1.0
    return chunk


def state_booted(st: dict) -> bool:
    return bool(st and not st.get("in_menu") and st.get("x") is not None and float(st.get("lives", 0)) > 0)


def classify(boots: bool, start_x: float | None, max_x: float | None, outcome: str, chunks: int) -> str:
    if not boots:
        return "UNPLAYABLE"
    dx = (max_x or 0.0) - (start_x or 0.0)
    if "won/beat_code" in outcome and chunks <= 1:
        return "SHORT/CEILING"
    if dx >= 450:
        return "HEADROOM"
    if dx >= 300 and chunks >= 10:
        return "HEADROOM"
    return "SHORT/CEILING"


def run_one(name: str, chunks: int, A: int, boot_wait: float, jump_every: int, jump_len: int, hold_run: bool) -> Result:
    os.environ["THEXTECH_LEVEL"] = level_path(name)
    env = None
    try:
        env = ProcRLEnv("thextech", A=A, boot_wait=boot_wait)
        env.A = A
        env.H = A
        env.per_row = env.env.chunk_seconds / A
        env.reset()
        st0 = dict(env._prev or {})
        boots = state_booted(st0)
        if not boots:
            return Result(name, False, None, None, None, f"no playable state: {st0}", "UNPLAYABLE", 0)
        start_x = float(st0["x"])
        max_x = start_x
        last = st0
        outcome = "ok"
        chunk = make_chunk(A, jump_every, jump_len, hold_run)
        ran = 0
        for t in range(chunks):
            _, _, done, info = env.step(chunk)
            ran = t + 1
            st = dict(info.get("state") or {})
            if st.get("x") is not None:
                max_x = max(max_x, float(st["x"]))
            last = st
            if done or st.get("dead") or st.get("won") or st.get("in_menu"):
                flags = []
                if st.get("dead"):
                    flags.append("dead")
                beat_code = st.get("beat_code")
                if st.get("won") or (beat_code not in (None, 0, 0.0) and float(beat_code) > 0):
                    flags.append(f"won/beat_code={st.get('beat_code')}")
                if st.get("in_menu"):
                    flags.append("menu")
                outcome = "+".join(flags) or "done"
                break
        if outcome == "ok":
            lx = last.get("x")
            lvx = last.get("vx")
            outcome = f"alive; last_x={lx}; vx={lvx}"
        total_dx = max_x - start_x
        return Result(name, True, start_x, max_x, total_dx, outcome, classify(True, start_x, max_x, outcome, ran), ran)
    except Exception as e:
        return Result(name, False, None, None, None, f"error: {type(e).__name__}: {e}", "UNPLAYABLE", 0)
    finally:
        if env is not None:
            env.close()
        os.environ.pop("THEXTECH_LEVEL", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("levels", nargs="*", default=DEFAULT_LEVELS)
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--A", type=int, default=18)
    ap.add_argument("--boot-wait", type=float, default=8.0)
    ap.add_argument("--jump-every", type=int, default=9)
    ap.add_argument("--jump-len", type=int, default=5)
    ap.add_argument("--hold-run", action="store_true")
    args = ap.parse_args()

    run_text = "+run" if args.hold_run else ""
    print(f"Using ProcRLEnv('thextech') with env.A={args.A}, {args.chunks} chunks, right{run_text}, jump pulses {args.jump_len}/{args.jump_every} rows")
    print("level\tboots\tstart_x\tmax_x\ttotal_dx\tchunks\toutcome\tclassification", flush=True)
    results = []
    for name in args.levels:
        res = run_one(name, args.chunks, args.A, args.boot_wait, args.jump_every, args.jump_len, args.hold_run)
        results.append(res)
        def fmt(v): return "" if v is None else f"{v:.1f}"
        print(f"{res.name}\t{res.boots}\t{fmt(res.start_x)}\t{fmt(res.max_x)}\t{fmt(res.total_dx)}\t{res.chunks}\t{res.outcome}\t{res.classification}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
