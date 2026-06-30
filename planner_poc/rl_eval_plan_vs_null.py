"""rl_eval_plan_vs_null.py -- closed-loop NitroGen policy on the RetroRLEnv RL substrate, comparing
PLAN-conditioned vs NULL (base) from the IDENTICAL start state on the GROUND-TRUTH reward (screen_x
forward progress). This is the rigorous, controlled version of the base-vs-plan test (real reward, not
a VLM judge), and it validates the full RL data path: policy -> action -> env -> reward.

Run (one GPU):
  CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc \
    QWEN=Qwen/Qwen3.5-2B .venv/bin/python planner_poc/rl_eval_plan_vs_null.py \
      --ckpt ckpts/btn_s600_full.pt --chunks 16 --A 2 --cfg 8 --seeds 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from play_annotated_horizon import _first_sentence

SONIC_SYS = ("You are the high-level planner for an agent playing Sonic the Hedgehog, a fast 2D "
             "side-scrolling platformer. The goal is to move RIGHT through the level quickly, jump over "
             "obstacles and gaps, and avoid enemies. Output ONE short imperative plan (max 10 words).")
SONIC_INSTR = ("In one sentence say what to do next, using concrete directions (right, up, down, jump) "
               "not vague words like 'forward'.")


def run(pol, env, chunks, A, cfg, mode, replan=True):
    """Run one episode from the env's fixed start state; return cumulative reward + screen_x trajectory."""
    start = env._ensure_start_state()
    obs = env.reset(start)
    cur = obs
    frame_hist = [cur]
    total = 0.0
    xs = [env._var(env.reward_var)]
    plan = ""
    for c in range(chunks):
        if mode == "null":
            plan = ""
            chunk = pol._sample_chunk(cur, "", cfg, plan_frames=[cur], null=True)
        else:
            if replan and c % A == 0:
                win = frame_hist[-4:]
                plan = _first_sentence(pol.pl.generate_plan(
                    win, pol.device, instruction=SONIC_INSTR, system=SONIC_SYS,
                    max_new_tokens=32) or "move right")
            chunk = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False)
        obs, reward, done, info = env.step(chunk[:A])
        total += reward
        xs.append(info.get(env.reward_var, 0.0))
        cur = obs
        frame_hist.append(cur)
        if done:
            break
    return total, xs, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--rom", default="Game data/Sonic The Hedgehog 2.md")
    ap.add_argument("--game", default="SonicTheHedgehog2-Genesis-v0")
    ap.add_argument("--system", default="Genesis")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    import torch
    env = RetroRLEnv(rom_path=args.rom, game=args.game, system=args.system)
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("sonic", plan="", objective="move right through the level", cfg_scale=args.cfg))

    print(f"RL plan-vs-null on {args.game} | ckpt={os.path.basename(args.ckpt)} chunks={args.chunks} "
          f"A={args.A} cfg={args.cfg} seeds={args.seeds}\n")
    res = {"null": [], "plan": []}
    for mode in ("null", "plan"):
        for s in range(args.seeds):
            torch.manual_seed(s); np.random.seed(s)
            total, xs, plan = run(pol, env, args.chunks, args.A, args.cfg, mode)
            res[mode].append(total)
            print(f"  {mode:4s} seed{s}: reward={total:+7.2f}  screen_x {xs[0]:.0f}->{xs[-1]:.0f} "
                  f"(Δ={xs[-1]-xs[0]:+.0f})  last_plan='{plan[:40]}'")
    import statistics as st
    nm, pm = st.mean(res["null"]), st.mean(res["plan"])
    print(f"\n  NULL mean reward  = {nm:+7.2f}  (n={len(res['null'])})")
    print(f"  PLAN mean reward  = {pm:+7.2f}  (n={len(res['plan'])})")
    print(f"  PLAN - NULL       = {pm - nm:+7.2f}  -> plan {'HELPS' if pm>nm else 'HURTS/NEUTRAL'} "
          f"on {args.game} (GROUND-TRUTH reward, same start)")
    env.close()


if __name__ == "__main__":
    main()
