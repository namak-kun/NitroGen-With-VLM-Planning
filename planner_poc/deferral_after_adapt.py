"""deferral_after_adapt.py -- the deferral-gate test on an RWBC-ADAPTED (more competent) actor.

The plain deferral_gate on the BASE Sonic actor gave DEGENERATE labels (plan~=null everywhere -> defer-rate
~100%) because the OOD base actor barely moves. This first RWBC-adapts the actor (LoRA-only, Sonic +1.5->
more competent), THEN runs the per-situation plan-vs-null gate collection on the SAME adapted model + env.
A more competent actor should make plan-vs-null VARY -> non-degenerate labels -> a real learnability test.

One env for the whole run (emulator-segfault fix). Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=1 .venv/bin/python planner_poc/deferral_after_adapt.py --warmup-steps 300 --episodes 14
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env, collect as rwbc_collect, build_batch, eval_reward
from joint_alternating import set_trainable, bc_phase, collect_filter
from deferral_gate import collect as gate_collect, train_eval_gate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--warmup-lr", type=float, default=2e-5)
    ap.add_argument("--collect-eps", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=14, help="gate-data episodes")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--top-frac", type=float, default=0.4)
    ap.add_argument("--eps-margin", type=float, default=0.02)
    args = ap.parse_args()
    device = "cuda"

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("defer_adapt", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m

    env = make_env(args.env)
    try:
        base_r, _ = eval_reward(pol, env, args.env, 4, args.chunks, args.A, args.cfg)
        print(f"\n[defer-after-adapt on {args.env}] baseline reward = {base_r:+.3f}", flush=True)

        # ---- RWBC warmup (LoRA-only) to make the actor more competent ----
        params = set_trainable(m, "lora")
        keep, mk = collect_filter(pol, env, args.env, args.collect_eps, args.chunks, args.A,
                                  args.cfg, args.top_frac)
        print(f"  warmup: collected+kept {len(keep)} chunks @{mk:+.2f}; {args.warmup_steps} BC steps...", flush=True)
        bc_phase(pol, env, m, "warmup", params, args.warmup_lr, args.warmup_steps, keep, device)
        adapt_r, _ = eval_reward(pol, env, args.env, 4, args.chunks, args.A, args.cfg)
        print(f"  adapted reward = {adapt_r:+.3f} (Δ={adapt_r-base_r:+.3f})\n", flush=True)

        # ---- deferral gate on the ADAPTED actor ----
        for p in m.parameters():
            p.requires_grad_(False)
        X, y, pr, nr, ep = gate_collect(pol, env, args.env, args.episodes, args.chunks, args.A,
                                        args.cfg, device, args.eps_margin)
    finally:
        env.close()
    print(f"[defer-after-adapt] gate on the ADAPTED actor (base {base_r:+.2f} -> adapted {adapt_r:+.2f}):")
    train_eval_gate(X, y, pr, nr, ep, device, args.eps_margin)


if __name__ == "__main__":
    main()
