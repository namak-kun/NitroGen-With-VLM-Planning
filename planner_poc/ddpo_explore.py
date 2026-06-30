"""ddpo_explore.py -- the DDPO PREMISE test: does stochastic diffusion sampling EXPLORE to find better
actions than greedy? (foundation check before building the full diffusion-PG loop; DDPO_DESIGN.md §6.1-2)

Self-imitation BC only works in a competence BAND (it sharpens existing behavior). DDPO needs the sampler
to EXPLORE — produce diverse chunks whose BEST beats greedy. This adds late-step Gaussian noise to the
flow-matching sampler (eval_policy._sample_chunk noise_sigma) and, from frame-exact save-states, measures:
  - greedy reward (sigma=0),
  - K stochastic-sample rewards from the IDENTICAL state -> reward STD (exploration spread) + best-of-K.
If best-of-K > greedy by a useful margin across states/sigmas, diffusion-PG has signal to optimize.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/ddpo_explore.py --env sonic --states 8 --K 8
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
from play_annotated_horizon import _first_sentence
from rwbc_actor_adapt import make_env, eval_reward, SYS, INSTR, _syskey
from joint_alternating import set_trainable, bc_phase, collect_filter


@torch.no_grad()
def reward_of(env, chunk, A):
    obs, r, d, _ = env.step(chunk[:A])
    return float(r), bool(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--states", type=int, default=8, help="distinct save-states to probe")
    ap.add_argument("--K", type=int, default=8, help="stochastic samples per state")
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    ap.add_argument("--noise-last-n", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=0, help=">0: RWBC-adapt the actor (LoRA-only) first so it's competent")
    ap.add_argument("--warmup-lr", type=float, default=2e-5)
    ap.add_argument("--collect-eps", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--top-frac", type=float, default=0.4)
    args = ap.parse_args()
    device = "cuda"

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("ddpo_explore", plan="", objective="progress", cfg_scale=args.cfg))
    sk = _syskey(args.env)

    env = make_env(args.env)
    try:
        if args.warmup_steps > 0:
            m = pol.m
            base_r, _ = eval_reward(pol, env, args.env, 4, args.chunks, args.A, args.cfg)
            params = set_trainable(m, "lora")
            keep, mk = collect_filter(pol, env, args.env, args.collect_eps, args.chunks, args.A, args.cfg, args.top_frac)
            print(f"[warmup] base {base_r:+.3f}; {len(keep)} chunks @{mk:+.2f}; {args.warmup_steps} BC steps...", flush=True)
            bc_phase(pol, env, m, "warmup", params, args.warmup_lr, args.warmup_steps, keep, device)
            for p in m.parameters():
                p.requires_grad_(False)
            adapt_r, _ = eval_reward(pol, env, args.env, 4, args.chunks, args.A, args.cfg)
            print(f"[warmup] adapted reward {adapt_r:+.3f} (Δ={adapt_r-base_r:+.3f})", flush=True)
        # gather distinct save-states by advancing a greedy rollout
        cur = env.reset(); hist = [cur]; plan = "move right"; states = []
        for t in range(args.states * 2):
            if t % args.A == 0:
                plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                       system=SYS[sk], max_new_tokens=32) or "move right")
            if t % 2 == 0:
                states.append((env.save_state(), np.asarray(cur).copy(), plan))
            ch = pol._sample_chunk(cur, plan, args.cfg, plan_frames=[cur], null=False)
            obs, _, d, _ = env.step(ch[:args.A]); cur = obs; hist.append(cur)
            if d:
                cur = env.reset(); hist = [cur]
            if len(states) >= args.states:
                break

        print(f"\n=== DDPO premise on {args.env}: {len(states)} states, K={args.K} ===", flush=True)
        for sigma in args.sigmas:
            g_list, bestK_list, std_list = [], [], []
            for (st, frame, plan) in states:
                env.load_state(st)
                gch = pol._sample_chunk(frame, plan, args.cfg, plan_frames=[frame], null=False)  # greedy
                gr, _ = reward_of(env, gch, args.A)
                rs = []
                for k in range(args.K):
                    env.load_state(st)
                    sch = pol._sample_chunk(frame, plan, args.cfg, plan_frames=[frame], null=False,
                                            noise_sigma=sigma, noise_last_n=args.noise_last_n, noise_seed=1000+k)
                    r, _ = reward_of(env, sch, args.A); rs.append(r)
                rs = np.array(rs)
                g_list.append(gr); bestK_list.append(float(rs.max())); std_list.append(float(rs.std()))
            g = float(np.mean(g_list)); bk = float(np.mean(bestK_list)); sd = float(np.mean(std_list))
            print(f"  sigma={sigma:.2f}: greedy={g:+.3f}  best-of-{args.K}={bk:+.3f}  (Δ={bk-g:+.3f})  "
                  f"sample-std={sd:.3f}  -> {'EXPLORES + finds better' if bk>g+0.02 and sd>0.01 else ('explores (variance) but no better' if sd>0.01 else 'no exploration')}", flush=True)
        print("\n  => if best-of-K > greedy with reward variance, diffusion-PG (DDPO) has signal to optimize "
              "(reweight toward the better samples). If variance~0, the late-step noise knob needs raising.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
