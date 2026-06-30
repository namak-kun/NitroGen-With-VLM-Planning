"""deferral_horizon.py -- deferral learnability with the CORRECTED (horizon-level) signal.

deferral_gate/deferral_after_adapt showed per-chunk plan-vs-null is degenerate AND misattributes the
plan's value (plan helps EPISODE-level, not per isolated chunk). This uses a HORIZON window: from each
save-state, roll the SAME plan for H chunks -> plan_R(window); restore; roll NULL for H chunks ->
null_R(window). Label = defer if plan_R <= null_R + margin (plan doesn't help over the horizon). Feature =
the frozen plan hidden at the window start. First RWBC-adapts the actor so plan-vs-null actually varies.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/deferral_horizon.py --warmup-steps 300 --windows 60 --horizon 5
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
from deferral_gate import train_eval_gate


@torch.no_grad()
def roll_window(pol, env, cur, plan, horizon, A, cfg, null):
    """Roll `horizon` chunks from the CURRENT emu state with a fixed plan (or null). Return summed reward.
    Caller must save/load around this to compare from the identical state."""
    tot = 0.0
    f = cur
    for _ in range(horizon):
        ch = pol._sample_chunk(f, "" if null else plan, cfg, plan_frames=[f], null=null)
        obs, r, d, _ = env.step(ch[:A]); tot += float(r); f = obs
        if d:
            break
    return tot


@torch.no_grad()
def collect_horizon(pol, env, env_name, windows, horizon, A, cfg, device, margin=0.05):
    sk = _syskey(env_name)
    feats, labels, prR, nuR, epi = [], [], [], [], []
    ep = 0
    cur = env.reset(); hist = [cur]
    for w in range(windows):
        plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                               system=SYS[sk], max_new_tokens=32) or "move right")
        st = env.save_state()
        pr = roll_window(pol, env, cur, plan, horizon, A, cfg, null=False)
        env.load_state(st)
        nr = roll_window(pol, env, cur, plan, horizon, A, cfg, null=True)
        env.load_state(st)
        # feature at window start
        h, kpm = pol.pl.encode_multimodal([cur], plan or ".", device, text_only=True)
        valid = (~kpm[0]).float().unsqueeze(-1)
        feats.append(((h[0]*valid).sum(0)/valid.sum().clamp(min=1)).float().cpu().numpy())
        prR.append(pr); nuR.append(nr); labels.append(1 if pr <= nr + margin else 0); epi.append(ep)
        # advance one chunk along the PLAN branch to the next window start (continue trajectory)
        ch = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False)
        obs, _, d, _ = env.step(ch[:A]); cur = obs; hist.append(cur)
        if d:
            cur = env.reset(); hist = [cur]; ep += 1
    return (np.asarray(feats, np.float32), np.asarray(labels, np.int64),
            np.asarray(prR, np.float32), np.asarray(nuR, np.float32), np.asarray(epi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--warmup-lr", type=float, default=2e-5)
    ap.add_argument("--collect-eps", type=int, default=10)
    ap.add_argument("--windows", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=5, help="chunks per plan-vs-null window")
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--top-frac", type=float, default=0.4)
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()
    device = "cuda"

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("defer_horizon", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m

    env = make_env(args.env)
    try:
        base_r, _ = eval_reward(pol, env, args.env, 4, args.chunks, args.A, args.cfg)
        print(f"\n[defer-horizon on {args.env}] baseline reward = {base_r:+.3f}", flush=True)
        params = set_trainable(m, "lora")
        keep, mk = collect_filter(pol, env, args.env, args.collect_eps, args.chunks, args.A, args.cfg, args.top_frac)
        print(f"  warmup: {len(keep)} chunks @{mk:+.2f}; {args.warmup_steps} BC steps...", flush=True)
        bc_phase(pol, env, m, "warmup", params, args.warmup_lr, args.warmup_steps, keep, device)
        adapt_r, _ = eval_reward(pol, env, args.env, 4, args.chunks, args.A, args.cfg)
        print(f"  adapted reward = {adapt_r:+.3f} (Δ={adapt_r-base_r:+.3f})\n", flush=True)
        for p in m.parameters():
            p.requires_grad_(False)
        X, y, pr, nr, ep = collect_horizon(pol, env, args.env, args.windows, args.horizon, args.A,
                                           args.cfg, device, args.margin)
    finally:
        env.close()
    print(f"[defer-horizon] H={args.horizon}-chunk windows on adapted actor (base {base_r:+.2f} -> {adapt_r:+.2f}):")
    train_eval_gate(X, y, pr, nr, ep, device, args.margin)


if __name__ == "__main__":
    main()
