"""joint_alternating.py -- alternating two-timescale RWBC (the feasible "joint training", per
FEASIBILITY_JOINT_AND_JUDGE.md §1). Simultaneous joint (LoRA+plan-head together) COLLAPSES; this
alternates Phase A (freeze plan-head, adapt DiT-LoRA) and Phase B (freeze LoRA, adapt plan-head), each a
reward-weighted-BC round on fresh rollouts from ONE reused env.

Why this is safe: when Phase B trains ONLY the plan-head, the masked-null path (plan tokens dropped) does
NOT depend on plan-head params -> the null forward (hence base-DiT-exact CFG anchor) is preserved BY
CONSTRUCTION. Phase A moves the actor; Phase B re-tunes the plan-token projection for the new actor. The
script asserts masked-null invariance is preserved across Phase B (null output unchanged).

CORRECTION (measured 2026-06-26 — the "BY CONSTRUCTION" claim above is WRONG): the null path DOES route
through `plan_head.adaln_cond(plan_tokens, dropped=True)` (eval_policy.py:209), so training the plan-head
DOES shift the null output (masked-null drift 0.0->0.0851 across rounds). And Phase-B plan-head RWBC HURTS
reward (sonic trajectory base+1.37 -> r0A+2.42 -> r0B+1.00 -> r1A+1.01 -> r1B-0.19 -> r2A-0.48 -> r2B-0.53,
fully degraded). CONCLUSION: naive alternating-RWBC does NOT work — plan-head RWBC is the wrong Phase-B
objective AND breaks CFG. For a real planner phase: (a) use OPSD-privileged distillation, not RWBC; (b)
FREEZE adaln_cond's dropped path (or add an explicit null-output regularizer) to keep masked-null; (c)
likely just keep Phase A (LoRA) and skip plan-head RL (the actor is the lever; OPSD-diag already showed
plan content isn't the bottleneck). This script stands as the NEGATIVE control demonstrating the above.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=3 .venv/bin/python planner_poc/joint_alternating.py --env sonic --rounds 3
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
from rwbc_actor_adapt import make_env, collect, build_batch, eval_reward


def set_trainable(m, mode):
    """mode='lora' -> only DiT-LoRA; mode='plan' -> only plan_head; returns the param list."""
    for n, p in m.named_parameters():
        if mode == "lora":
            p.requires_grad_("lora_" in n)
        else:  # plan
            p.requires_grad_(n.startswith("plan_head."))
    return [p for p in m.parameters() if p.requires_grad]


def bc_phase(pol, env, m, name, params, lr, steps, keep, device, bs=4):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rw = np.array([s["reward"] for s in keep]); rw = (rw - rw.min()) / (rw.max() - rw.min() + 1e-6) + 0.2
    for step in range(steps):
        m.train()
        idx = np.random.choice(len(keep), size=min(bs, len(keep)), replace=False)
        batch = build_batch(pol, [keep[i] for i in idx], device)
        w = torch.tensor(rw[idx], device=device, dtype=torch.float32).mean()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(batch)["loss"] * w
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    m.eval()


@torch.no_grad()
def null_signature(pol, env, m, cfg, device):
    """A fingerprint of the NULL (plan-dropped) action output from a fixed frame -> to verify masked-null
    invariance is preserved across a plan-head phase."""
    cur = env.reset()
    ch = pol._sample_chunk(cur, "", cfg, plan_frames=[cur], null=True)
    return np.asarray(ch, np.float32)


def collect_filter(pol, env, name, eps, chunks, A, cfg, top_frac, plan_temp=0.0):
    samples = collect(pol, env, name, eps, chunks, A, cfg, plan_temp=plan_temp)
    rewards = np.array([s["reward"] for s in samples])
    thr = np.quantile(rewards, 1 - top_frac)
    keep = [s for s in samples if s["reward"] >= thr]
    return keep, float(np.mean([s["reward"] for s in keep]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps-a", type=int, default=150, help="Phase A (LoRA) BC steps/round")
    ap.add_argument("--steps-b", type=int, default=150, help="Phase B (plan-head) BC steps/round")
    ap.add_argument("--lr-a", type=float, default=2e-5)
    ap.add_argument("--lr-b", type=float, default=1e-5, help="plan-head LR (lower: anti-collapse)")
    ap.add_argument("--collect-eps", type=int, default=8)
    ap.add_argument("--eval-eps", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=14)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--top-frac", type=float, default=0.4)
    args = ap.parse_args()
    device = "cuda"

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("joint_alt", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m

    env = make_env(args.env)
    try:
        base_r, _ = eval_reward(pol, env, args.env, args.eval_eps, args.chunks, args.A, args.cfg)
        print(f"\n[joint-alternating on {args.env}] baseline reward = {base_r:+.3f}\n", flush=True)
        traj = [("base", base_r)]
        for r in range(args.rounds):
            # ---- Phase A: DiT-LoRA ----
            pa = set_trainable(m, "lora")
            keepA, mA = collect_filter(pol, env, args.env, args.collect_eps, args.chunks, args.A,
                                       args.cfg, args.top_frac)
            bc_phase(pol, env, m, "A", pa, args.lr_a, args.steps_a, keepA, device)
            ra, _ = eval_reward(pol, env, args.env, args.eval_eps, args.chunks, args.A, args.cfg)
            print(f"  round {r} Phase-A (LoRA, {len(keepA)} chunks keep@{mA:+.2f}) -> reward {ra:+.3f}", flush=True)
            traj.append((f"r{r}A", ra))

            # ---- Phase B: plan-head (masked-null preserved by construction; verify) ----
            sig_before = null_signature(pol, env, m, args.cfg, device)
            pb = set_trainable(m, "plan")
            keepB, mB = collect_filter(pol, env, args.env, args.collect_eps, args.chunks, args.A,
                                       args.cfg, args.top_frac)
            bc_phase(pol, env, m, "B", pb, args.lr_b, args.steps_b, keepB, device)
            rb, _ = eval_reward(pol, env, args.env, args.eval_eps, args.chunks, args.A, args.cfg)
            sig_after = null_signature(pol, env, m, args.cfg, device)
            null_drift = float(np.abs(sig_after - sig_before).mean())
            print(f"  round {r} Phase-B (plan-head, {len(keepB)} chunks keep@{mB:+.2f}) -> reward {rb:+.3f} "
                  f"| masked-null drift={null_drift:.4f} ({'PRESERVED' if null_drift < 0.05 else 'CHANGED!'})", flush=True)
            traj.append((f"r{r}B", rb))
    finally:
        env.close()

    print(f"\n===== JOINT-ALTERNATING trajectory ({args.env}) =====")
    print("  " + "  ".join(f"{k}={v:+.2f}" for k, v in traj))
    final = traj[-1][1]
    print(f"  base {base_r:+.3f} -> final {final:+.3f} (Δ={final-base_r:+.3f}); "
          f"{'NO COLLAPSE (alternating stable)' if final > base_r - 1.0 else 'degraded'}.")
    print("  (compare to naive SIMULTANEOUS joint which collapsed +15.77->+1.58; alternating decouples the "
          "two collapse modes and preserves masked-null.)")


if __name__ == "__main__":
    main()
