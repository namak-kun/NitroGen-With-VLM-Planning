"""opsd_diag.py -- the OPSD/planner-bottleneck DIAGNOSTIC (per the GPT-5.5 duck): does a BETTER plan raise
the actor's return? Compare GT reward on a game under (a) Qwen-GENERATED plans (current planner) vs (b) a
FIXED hand-written "good" plan vs (c) NULL (no plan). If fixed-good ≫ Qwen → the planner is a limiting
factor → plan-improvement (OPSD / GPT-5.5-plan distillation) is worth it. If fixed-good ≈ Qwen → the
planner isn't the bottleneck (the actor is) → OPSD is moot.

Run (one GPU):
  CUDA_VISIBLE_DEVICES=1 env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc \
    QWEN=Qwen/Qwen3.5-2B .venv/bin/python planner_poc/opsd_diag.py --env sonic --seeds 5
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
from rwbc_actor_adapt import make_env, SYS, INSTR

FIXED_GOOD = {
    "sonic": "run right fast, jump over enemies and gaps, keep moving right",
    "thextech": "move right, jump over pits and onto platforms, avoid enemies",
}


def run(pol, name, chunks, A, cfg, mode, fixed_plan, seed):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    sk = "sonic" if name == "sonic" else "thextech"
    env = make_env(name); cur = env.reset(); hist = [cur]; plan = fixed_plan or "move right"; tot = 0.0
    for t in range(chunks):
        if mode == "null":
            chunk = pol._sample_chunk(cur, "", cfg, plan_frames=[cur], null=True)
        else:
            if mode == "generated" and t % A == 0:
                plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                       system=SYS[sk], max_new_tokens=32) or "move right")
            elif mode == "fixed":
                plan = fixed_plan
            chunk = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False)
        obs, r, done, info = env.step(chunk[:A]); tot += r; cur = obs; hist.append(cur)
        if done:
            break
    env.close()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--env", default="sonic")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("opsd", plan="", objective="progress", cfg_scale=args.cfg))
    fixed = FIXED_GOOD.get(args.env, FIXED_GOOD["sonic"])
    print(f"OPSD diagnostic on {args.env} | fixed plan: '{fixed}'\n")
    res = {"null": [], "generated": [], "fixed": []}
    for mode in res:
        for s in range(args.seeds):
            res[mode].append(run(pol, args.env, args.chunks, args.A, args.cfg, mode, fixed, 200 + s))
        print(f"  {mode:10s} mean reward {np.mean(res[mode]):+7.3f}  {[round(x,2) for x in res[mode]]}")
    import statistics as st
    g, f = st.mean(res["generated"]), st.mean(res["fixed"])
    print(f"\n  FIXED-GOOD vs GENERATED(Qwen): {f:+.2f} vs {g:+.2f}  (Δ={f-g:+.2f})")
    print(f"  => {'PLANNER is a limiting factor (better plan helps) -> OPSD worth it' if f > g + 0.3 else 'planner NOT the bottleneck (actor is) -> OPSD moot; better plans dont raise return'}")


if __name__ == "__main__":
    main()
