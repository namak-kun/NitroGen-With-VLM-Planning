"""demo_eval_clean.py -- DETERMINISTIC base-vs-bootstrap eval (fixes the noisy stochastic eval in
demo_train_stack). From each demo start state, run N chunks with a FIXED per-(state,chunk) latent seed
(noise_sigma=0 -> deterministic flow integration), measure screen_x advance. Compare the BASE policy vs a
saved bootstrap delta under IDENTICAL conditions + multiple global seeds for error bars.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/demo_eval_clean.py \
     --eval-game smw --deltas ckpts/demo_stack_platformers.pt ckpts/demo_stack_full.pt
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
from demo_train_stack import GAME_META, demo_states, EVAL_PLAN


@torch.no_grad()
def advance(pol, env, states, plan, chunks, A, cfg, base_seed):
    """Deterministic per-(state,chunk) seeded rollout -> mean screen_x advance over states."""
    deltas = []
    for si, st in enumerate(states):
        env.reset(); env.load_state(st)
        x0 = env._var(env.reward_var)
        for t in range(chunks):
            f = env.frame()
            ch = pol._sample_chunk(f, plan, cfg, plan_frames=[f], null=False,
                                   noise_seed=base_seed * 100003 + si * 101 + t)
            env.step(ch[:A])
        deltas.append(float(env._var(env.reward_var) - x0))
    return float(np.mean(deltas)), deltas


def load_delta(m, path):
    d = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
    sd = m.state_dict()
    applied = 0
    for k, v in d.items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].dtype)); applied += 1
    print(f"   applied {applied}/{len(d)} delta tensors from {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--deltas", nargs="*", default=[], help="bootstrap delta ckpts to compare vs base")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda"

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("demo_eval", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m
    base_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}  # to restore between deltas

    env = make_env(GAME_META[args.eval_game]["env"])
    try:
        states = demo_states(args.eval_game)
        print(f"[clean-eval] {args.eval_game}: {len(states)} demo start states, {args.seeds} seeds, "
              f"deterministic\n", flush=True)

        def measure(tag):
            rs = [advance(pol, env, states, EVAL_PLAN, args.chunks, args.A, args.cfg, s)[0]
                  for s in range(args.seeds)]
            print(f"  {tag:28s} advance = {np.mean(rs):+7.1f}  (per-seed {[round(x,1) for x in rs]})", flush=True)
            return float(np.mean(rs))

        base = measure("BASE (btn_s600)")
        for dp in args.deltas:
            m.load_state_dict(base_sd)  # restore base
            load_delta(m, dp)
            m.eval()
            post = measure(os.path.basename(dp).replace(".pt", ""))
            print(f"     -> Δ vs base = {post - base:+.1f}\n")
        m.load_state_dict(base_sd)
    finally:
        env.close()


if __name__ == "__main__":
    main()
