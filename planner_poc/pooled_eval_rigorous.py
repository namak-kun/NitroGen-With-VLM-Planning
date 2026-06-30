"""pooled_eval_rigorous.py — tighten the pooled-generalist headline: eval the 3 pooled plan-head delta seeds
vs base across MORE fixed demo start states, per game (SMW, MMX), with paired bootstrap CIs on Δ_plan.
Low-variance (no training, no env reward) — just inference from fixed save-states.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from plan_graded_test import BATTERY
from demo_bc import demo_start_states, GAME_CFG

F = "/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files"


@torch.no_grad()
def dplan_per_state(pol, game, states, chunks=16, A=2, cfg=8.0):
    """Per-state Δ_plan = plan-advance − null-advance (paired same start)."""
    env = make_env(GAME_CFG[game]["env"]); plan = BATTERY[game]["correct"]
    out = []
    try:
        for st in states:
            env.reset(); env.load_state(st)
            if env.frame().mean() < 1.0:
                env._emu_step([], 1)
            base_st = env.save_state()
            # null
            env.load_state(base_st)
            if env.frame().mean() < 1.0:
                env._emu_step([], 1)
            x0 = env._var(env.reward_var)
            for _ in range(chunks):
                ch = pol._sample_chunk(env.frame(), "", cfg, plan_frames=[env.frame()], null=True)
                env.step(ch[:A])
            nadv = env._var(env.reward_var) - x0
            # plan
            env.load_state(base_st)
            if env.frame().mean() < 1.0:
                env._emu_step([], 1)
            x0 = env._var(env.reward_var)
            for _ in range(chunks):
                ch = pol._sample_chunk(env.frame(), plan, cfg, plan_frames=[env.frame()], null=False)
                env.step(ch[:A])
            padv = env._var(env.reward_var) - x0
            out.append(padv - nadv)
    finally:
        env.close()
    return np.array(out, float)


def fresh(ckpt, qwen):
    pol = NitroGenPolicy(ckpt, qwen=qwen, default_cfg=8.0)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("pe", plan="", objective="progress", cfg_scale=8.0))
    return pol


def apply_planhead(pol, path):
    d = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
    msd = pol.m.state_dict()
    for k, v in d.items():
        if k in msd:
            msd[k] = v.to(msd[k].device, msd[k].dtype)
    pol.m.load_state_dict(msd, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--games", default="smw,mmx")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-starts", type=int, default=8)
    args = ap.parse_args()

    # base
    polb = fresh(args.ckpt, args.qwen)
    states = {g: demo_start_states(GAME_CFG[g]["demo_glob"])[:args.n_starts] for g in args.games.split(",")}
    base = {g: dplan_per_state(polb, g, states[g]) for g in states}
    for g in base:
        print(f"  base {g}: Δ_plan mean {base[g].mean():+.1f} (n={len(base[g])})", flush=True)
    del polb; torch.cuda.empty_cache()

    # each pooled seed
    rows = {g: [] for g in states}
    for s in args.seeds.split(","):
        pol = fresh(args.ckpt, args.qwen)
        apply_planhead(pol, os.path.join(F, f"pooled_planfit_s{s}.pt"))
        for g in states:
            dp = dplan_per_state(pol, g, states[g])
            rows[g].append(dp)
            print(f"  pooled_s{s} {g}: Δ_plan mean {dp.mean():+.1f}", flush=True)
        del pol; torch.cuda.empty_cache()

    print("\n===== POOLED GENERALIST (rigorous) =====")
    rng = np.random.default_rng(0)
    for g in states:
        b = base[g]
        post = np.stack(rows[g])                 # (seeds, n_starts)
        # paired per-state change averaged over seeds
        change = post.mean(0) - b                # (n_starts,)
        bs = [rng.choice(change, len(change), replace=True).mean() for _ in range(10000)]
        lo, hi = np.quantile(bs, [.025, .975])
        print(f"  {g}: base {b.mean():+.1f} -> pooled {post.mean():+.1f}  "
              f"change {change.mean():+.1f}  95%CI [{lo:+.1f},{hi:+.1f}]  "
              f"{'SIG>0' if lo > 0 else 'ns'}")


if __name__ == "__main__":
    main()
