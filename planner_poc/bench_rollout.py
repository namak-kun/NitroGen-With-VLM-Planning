"""bench_rollout.py -- run a plan-conditioned btn_s600 rollout and emit a STRUCTURED judgment manifest
(per re-plan cycle: before frame, the generated plan, executed-action summary, after frame, GT state).

This is the benchmark substrate: the manifest + frames feed three judges (the frozen Qwen VLM-judge and
the GPT-5.5 / Gemini-3.5-Flash strong-model judges) which rate plan sensibility + progress and attribute
failures to PLANNER / ACTOR(DiT) / BOTH. Reuses the exact closed-loop inference path of
play_annotated_horizon.py (generate a grounded plan each cycle, A chunks open-loop, re-observe).

Run (one GPU):
  CUDA_VISIBLE_DEVICES=3 env -u VIRTUAL_ENV -u PYTHONPATH \
    PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B \
    .venv/bin/python planner_poc/bench_rollout.py --env thextech --A 2 --cycles 8 \
      --ckpt ckpts/btn_s600_full.pt --out docs/bench/thextech
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from run_poc import make_env_factory
from eval_policy import NitroGenPolicy
from play_annotated_horizon import (
    CONTROLS, GENRE_OF, GROUND_INSTR, MENU_MASK, OBJECTIVES, PLAN_SYS,
    _first_sentence, summarize_actions,
)

JLX, JLY = 21, 22
_BTN = [(18, "A/jump"), (20, "X/shoot"), (5, "B"), (10, "Y"), (16, "RT/accel"), (9, "LT/brake"),
        (1, "down"), (2, "left"), (3, "right"), (4, "up")]


def _save(frame, path):
    Image.fromarray(np.asarray(frame).astype(np.uint8)).save(path)


def _clean_state(st):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in (st or {}).items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--A", type=int, default=2, help="execution horizon (chunks open-loop per re-plan)")
    ap.add_argument("--cycles", type=int, default=8, help="number of re-plan cycles")
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--nframes", type=int, default=4, help="recent frames the planner sees")
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--plan", default=None, help="fixed plan (skip generation); default generate each cycle")
    ap.add_argument("--null", action="store_true", help="base ablation: null plan (no System-2)")
    ap.add_argument("--boot-wait", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    genre = GENRE_OF.get(args.env, "other")
    objective = OBJECTIVES.get(args.env, "Make progress in the game.")
    controls = CONTROLS.get(genre, "Controls: move LEFT/RIGHT/UP/DOWN, action buttons (JUMP/ATTACK).")
    out = args.out or f"docs/bench/{args.env}"
    fdir = os.path.join(out, "frames")
    os.makedirs(fdir, exist_ok=True)

    kw = {}
    if args.boot_wait is not None:
        kw["boot_wait"] = args.boot_wait
    env = make_env_factory(args.env, **kw)()
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    H = pol.m.config.action_horizon
    A = max(1, min(args.A, H))
    per_row = env.chunk_seconds / H

    manifest = {"env": args.env, "genre": genre, "objective": objective, "controls": controls,
                "ckpt": os.path.basename(args.ckpt), "cfg": args.cfg, "A": A, "H": H,
                "mode": "null" if args.null else ("fixed" if args.plan else "generated"),
                "cycles": []}
    t0 = time.time()
    print(f"bench_rollout env={args.env} genre={genre} A={A} cfg={args.cfg} cycles={args.cycles} "
          f"ckpt={os.path.basename(args.ckpt)}", flush=True)
    try:
        sc = Scenario(args.env, plan=args.plan or "", objective=objective,
                      max_steps=args.cycles * A, cfg_scale=args.cfg)
        obs = env.reset(sc)
        cur = obs.frame
        frame_hist = [cur]
        for c in range(args.cycles):
            state_before = _clean_state(obs.state if c == 0 else state_after_last)
            before_p = os.path.join(fdir, f"c{c:02d}_before.png"); _save(cur, before_p)
            # ---- System-2: generate a grounded plan (same path as play_annotated_horizon) ----
            if args.null:
                plan = "(null / base DiT, no plan)"
            elif args.plan:
                plan = args.plan
            else:
                win = frame_hist[::-1][::args.frame_stride][:args.nframes][::-1]
                instr = f"{objective} {GROUND_INSTR}"
                plan = _first_sentence(
                    pol.pl.generate_plan(win, pol.device, instruction=instr, system=PLAN_SYS,
                                         max_new_tokens=40) or "advance")
            # ---- System-1: A chunks open-loop, capture frames + state ----
            chunk = pol._sample_chunk(cur, "" if args.null else plan, args.cfg,
                                      plan_frames=[cur], null=args.null)  # (H,25)
            for b in MENU_MASK.get(args.env, ()):
                chunk[:, b] = 0.0
            rows = env.apply_chunk_capture(chunk[:A], per_row)  # A x (row, frame, state)
            exec_rows = [r for r, _, _ in rows]
            mids = []
            for ri, (row, frame, st) in enumerate(rows):
                mp = os.path.join(fdir, f"c{c:02d}_step{ri}.png"); _save(frame, mp)
                mids.append(os.path.relpath(mp, out))
            after = rows[-1][1]
            after_p = os.path.join(fdir, f"c{c:02d}_after.png"); _save(after, after_p)
            state_after = _clean_state(rows[-1][2])
            state_after_last = state_after
            act_summary = summarize_actions(exec_rows)
            # per-chunk dominant stick/buttons for the action-grounded view
            arr = np.asarray(exec_rows, dtype=np.float32)
            stick = {"jlx": round(float(arr[:, JLX].mean()), 3), "jly": round(float(arr[:, JLY].mean()), 3)}
            btns = [lbl for idx, lbl in _BTN if float((arr[:, idx] > 0.5).mean()) > 0.3]
            manifest["cycles"].append({
                "i": c, "plan": plan, "action_summary": act_summary,
                "stick": stick, "buttons_held": btns,
                "before": os.path.relpath(before_p, out), "after": os.path.relpath(after_p, out),
                "steps": mids, "state_before": state_before, "state_after": state_after,
            })
            print(f"  cycle{c:02d} plan='{plan[:48]}' stick={stick} btns={btns} "
                  f"state_after={state_after}", flush=True)
            frame_hist.append(after); frame_hist = frame_hist[-(args.nframes * args.frame_stride + 2):]
            cur = after
    finally:
        env.close()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    manifest["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {out}/manifest.json  ({len(manifest['cycles'])} cycles, {manifest['seconds']}s)")


if __name__ == "__main__":
    main()
