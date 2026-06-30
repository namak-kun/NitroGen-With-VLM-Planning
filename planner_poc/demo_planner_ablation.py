"""demo_planner_ablation.py -- QUANTITATIVE ablation of the planner conditioning, on GT reward, from the
demo start states. Compares, deterministically (matched per-(state,chunk) latent seeds):
  base  : null DiT (no plan)
  plan  : per-game CORRECT planner prompt, plan only (no memory)
  learn : per-game planner + carried-forward <learnings> context

Measures mean reward-var advance over N chunks from each demo start state. Answers: (a) does the corrected
per-game prompt help vs base/old? (b) does carrying LEARNINGS (context handling) beat stateless planning?

Run (smw/sonic have screen_x; minish has movement reward):
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/demo_planner_ablation.py --game smw --chunks 16
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys

import numpy as np
import torch

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from demo_train_stack import GAME_META
from game_planner import ClosedLoopPlanner, applied_action_desc

ENVMAP = {"fireemblem": "gba_fire_emblem_sacred_stones"}


@torch.no_grad()
def run_mode(pol, env, states, game, mode, chunks, A, cfg, seed):
    # smw/sonic expose a monotone reward_var (screen_x): measure its delta. GBA top-down envs (minish)
    # have no single monotone var -> accumulate the env's native per-step reward instead.
    use_var = hasattr(env, "reward_var")
    deltas = []
    for si, st in enumerate(states):
        env.reset(); env.load_state(st)
        x0 = env._var(env.reward_var) if use_var else 0.0
        acc = 0.0
        clp = ClosedLoopPlanner(pol, game, mode=("learn" if mode == "learn" else "plan")) if mode != "base" else None
        hist = [env.frame()]; plan = ""
        for t in range(chunks):
            if mode != "base" and t % A == 0:
                plan = clp.plan(hist[-4:])
            f = env.frame()
            ch = pol._sample_chunk(f, "" if mode == "base" else plan, cfg, plan_frames=[f],
                                   null=(mode == "base"), noise_seed=seed * 100003 + si * 101 + t)
            out = env.step(ch[:A]); hist.append(env.frame())
            if mode == "learn":     # feed the executed play back so the next plan can review it
                clp.observe(applied_action_desc(env, game, ch[0]), env.frame())
            if not use_var and isinstance(out, tuple) and len(out) >= 2:
                acc += float(out[1])
        deltas.append(float(env._var(env.reward_var) - x0) if use_var else acc)
    return float(np.mean(deltas)), deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--modes", nargs="+", default=["base", "plan", "learn"])
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=4)
    args = ap.parse_args()
    device = "cuda"

    meta = GAME_META[args.game]; envname = meta["env"] or ENVMAP.get(args.game)
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("plan_ablation", plan="", objective="progress", cfg_scale=args.cfg))

    states = [gzip.decompress(open(p, "rb").read())
              for p in sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"], "*", "initial.state")))]
    states = states[: args.max_states]
    env = make_env(envname)
    try:
        print(f"\n=== PLANNER ABLATION ({args.game}, {len(states)} states, {args.seeds} seeds, "
              f"{args.chunks} chunks, reward={getattr(env, 'reward_var', 'step-reward (top-down)')}) ===", flush=True)
        res = {}
        for mode in args.modes:
            rs = [run_mode(pol, env, states, args.game, mode, args.chunks, args.A, args.cfg, s)[0]
                  for s in range(args.seeds)]
            res[mode] = float(np.mean(rs))
            print(f"  {mode:6s} advance = {res[mode]:+7.1f}  (per-seed {[round(x,1) for x in rs]})", flush=True)
        if "base" in res:
            for m in args.modes:
                if m != "base":
                    print(f"   Δ {m} vs base = {res[m]-res['base']:+.1f}")
        if "plan" in res and "learn" in res:
            print(f"   Δ learn vs plan (value of carried LEARNINGS) = {res['learn']-res['plan']:+.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
