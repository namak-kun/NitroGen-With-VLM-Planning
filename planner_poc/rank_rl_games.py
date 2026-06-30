"""rank_rl_games.py -- rank the RL-reward envs by how much GROUND-TRUTH reward the plan-conditioned
btn_s600 policy already earns (a proxy for ACTOR LEARNABILITY: the GPT-5.5 rubber-duck said pick the
first RL game where the base plan-conditioned DiT is already coherent, NOT an OOD game like Sonic).
Higher reward + lower stuck/death = better first RL target.

Handles both env backends: retro_rl_env (Sonic/Genesis, frame-exact) and proc_rl_env (TheXTech/Solarus).

Run (one GPU):
  CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc \
    QWEN=Qwen/Qwen3.5-2B .venv/bin/python planner_poc/rank_rl_games.py --chunks 12 --A 2 --cfg 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from play_annotated_horizon import _first_sentence

PLANS_SYS = {
    "sonic": "You are the planner for Sonic, a fast 2D platformer. Goal: move RIGHT, jump gaps/enemies. "
             "Output ONE short imperative plan (max 10 words).",
    "thextech": "You are the planner for a Mario-style platformer. Goal: advance RIGHT, jump platforms, "
                "avoid hazards. Output ONE short imperative plan (max 10 words).",
    "solarus": "You are the planner for a top-down Zelda-like. Goal: explore, move to new areas, fight. "
               "Output ONE short imperative plan (max 10 words).",
    "gba_minish_cap": "You are the planner for The Minish Cap, a top-down Zelda game. Goal: explore rooms "
                     "and move to new areas. Output ONE short imperative plan (max 10 words).",
    "gba_pokemon_emerald": "You are the planner for Pokemon Emerald. Goal: explore the map. Output ONE "
                          "short imperative plan (max 10 words).",
    "gba_fire_emblem_sacred_stones": "You are the planner for Fire Emblem. Goal: move the cursor and make "
                                    "tactical progress. Output ONE short imperative plan (max 10 words).",
}
INSTR = ("In one sentence say what to do next using concrete directions (left,right,up,down,jump) not "
         "vague words like 'forward'.")


def make_env(kind):
    if kind == "sonic":
        from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
        return RetroRLEnv(), "retro"
    if kind.startswith("gba_"):
        from nitrogen.eval.envs.gba_env import make_gba_env
        return make_gba_env(kind), "retro"
    from nitrogen.eval.envs.proc_rl_env import ProcRLEnv
    return ProcRLEnv(kind), "proc"


def run_game(pol, kind, chunks, A, cfg):
    env, backend = make_env(kind)
    sysp = PLANS_SYS.get(kind, PLANS_SYS["thextech"])
    obs = env.reset()
    cur = obs
    hist = [cur]
    total = 0.0
    stuck = 0
    died = False
    plan = "move right"
    for c in range(chunks):
        if c % A == 0:
            plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                                        system=sysp, max_new_tokens=32) or "move right")
        chunk = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False)
        obs, r, done, info = env.step(chunk[:A] if backend == "proc" else chunk[:A])
        total += r
        if abs(r) < 1e-3:
            stuck += 1
        cur = obs
        hist.append(cur)
        if done:
            died = True
            break
    env.close()
    return {"reward": round(total, 2), "stuck_frac": round(stuck / max(1, chunks), 2),
            "ended_early": died, "last_plan": plan[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--games", nargs="+", default=["thextech", "sonic", "solarus_zelda", "gba_minish_cap"])
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("rank", plan="", objective="make progress", cfg_scale=args.cfg))

    print(f"RL game ranking (actor learnability) | ckpt={os.path.basename(args.ckpt)} "
          f"chunks={args.chunks} A={args.A} cfg={args.cfg}\n")
    results = {}
    for g in args.games:
        key = "sonic" if g == "sonic" else g
        try:
            results[g] = run_game(pol, key, args.chunks, args.A, args.cfg)
            print(f"  {g:16s} {results[g]}")
        except Exception as e:
            print(f"  {g:16s} ERROR {type(e).__name__}: {str(e)[:80]}")
    # rank by reward (desc), tie-break lower stuck
    ranked = sorted(results.items(), key=lambda kv: (-kv[1]["reward"], kv[1]["stuck_frac"]))
    print("\n  RANK (best first RL target = highest reward, lowest stuck):")
    for i, (g, r) in enumerate(ranked, 1):
        print(f"    {i}. {g:16s} reward={r['reward']:+.2f} stuck={r['stuck_frac']}")


if __name__ == "__main__":
    main()
