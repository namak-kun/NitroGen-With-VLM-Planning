"""rwbc_heldout.py -- rigor check on the actor-adaptation result (the GPT-5.5 duck's anti-overfit
concern): does the reward-weighted-BC gain TRANSFER to HELD-OUT start states, or only the one trained on?

Sonic's frame-exact save/load lets us make multiple start states (drive right different amounts + save).
We collect + train RWBC from the BASE start state only, then eval reward from base + several HELD-OUT
states (different level positions). If the gain transfers -> not memorization. If only base improves ->
overfit (treat as failure, per the duck).

Run (one GPU):
  CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc \
    QWEN=Qwen/Qwen3.5-2B .venv/bin/python planner_poc/rwbc_heldout.py --collect-eps 12 --steps 200
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from play_annotated_horizon import _first_sentence
from rwbc_actor_adapt import collect, build_batch, SYS, INSTR
from nitrogen.eval.envs.retro_rl_env import RetroRLEnv


def make_states(env, n, drive_each=120):
    """Make n+1 start states: base, then progressively driven-right snapshots (different level positions)."""
    base = env._ensure_start_state()
    states = [base]
    env.reset(base)
    right = np.full((18, 25), 0.5, np.float32); right[:, 21] = 1.0; right[:, 18] = 1.0
    for _ in range(n):
        for _ in range(drive_each // (env.frames_per_row * 2)):
            env.step(right[:2])
        states.append(env.save_state())
    return states


def eval_from(pol, env, state, chunks, A, cfg):
    sk = "sonic"
    env.reset(state); cur = env.frame(); hist = [cur]; plan = "move right"; r = 0.0
    for t in range(chunks):
        if t % A == 0:
            plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                   system=SYS[sk], max_new_tokens=32) or "move right")
        chunk = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False)
        obs, rr, done, info = env.step(chunk[:A]); r += rr; cur = obs; hist.append(cur)
        if done:
            break
    return r


def eval_states(pol, env, states, chunks, A, cfg, seeds=3):
    out = []
    for si, s in enumerate(states):
        rs = []
        for sd in range(seeds):
            torch.manual_seed(1000 + si * 10 + sd); np.random.seed(1000 + si * 10 + sd)
            rs.append(eval_from(pol, env, s, chunks, A, cfg))
        out.append(float(np.mean(rs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--n-heldout", type=int, default=3)
    ap.add_argument("--collect-eps", type=int, default=12)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--top-frac", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=6)
    args = ap.parse_args()
    device = "cuda"

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("ho", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m
    for n, p in m.named_parameters():
        p.requires_grad_("lora_" in n)   # LoRA-only (the stable config)
    train_params = [p for p in m.parameters() if p.requires_grad]

    env = RetroRLEnv()
    states = make_states(env, args.n_heldout)
    print(f"made {len(states)} start states (state0=train, state1..{args.n_heldout}=HELD-OUT)\n")

    print("[1/3] baseline reward per start state...", flush=True)
    base = eval_states(pol, env, states, args.chunks, args.A, args.cfg)
    print("   baseline:", [round(x, 2) for x in base], flush=True)
    env.close()   # stable-retro allows ONE emulator/process; collect() makes its own envs

    print(f"\n[2/3] collect from BASE state only + RWBC ({args.steps} steps)...", flush=True)
    # collect uses env.reset() = base state (its own env instances, sequential)
    samples = collect(pol, "sonic", args.collect_eps, args.chunks, args.A, args.cfg)
    rewards = np.array([s["reward"] for s in samples])
    thr = np.quantile(rewards, 1 - args.top_frac)
    keep = [s for s in samples if s["reward"] >= thr]
    print(f"   {len(samples)} chunks -> keep {len(keep)} (reward>={thr:.3f})", flush=True)
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    rw = np.array([s["reward"] for s in keep]); rw = (rw - rw.min()) / (rw.max() - rw.min() + 1e-6) + 0.2
    m.train()
    for step in range(args.steps):
        idx = np.random.choice(len(keep), size=min(args.bs, len(keep)), replace=False)
        batch = build_batch(pol, [keep[i] for i in idx], device)
        w = torch.tensor(rw[idx], device=device).float().mean()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(batch)["loss"] * w
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        if step % 50 == 0:
            print(f"   step {step} loss {float(loss):.4f}", flush=True)
    m.eval()

    print("\n[3/3] post reward per start state...", flush=True)
    env = RetroRLEnv()   # reopen for eval (collect closed its envs)
    post = eval_states(pol, env, states, args.chunks, args.A, args.cfg)
    print("   post:    ", [round(x, 2) for x in post])
    print("\n===== HELD-OUT TRANSFER =====")
    print(f"   {'state':<10}{'baseline':>10}{'post':>10}{'Δ':>9}")
    for i, (b, p) in enumerate(zip(base, post)):
        tag = "train" if i == 0 else f"heldout{i}"
        print(f"   {tag:<10}{b:>10.2f}{p:>10.2f}{p-b:>+9.2f}")
    ho_b = np.mean(base[1:]); ho_p = np.mean(post[1:])
    print(f"\n   held-out mean: {ho_b:+.2f} -> {ho_p:+.2f} (Δ={ho_p-ho_b:+.2f}) -> "
          f"{'GAIN TRANSFERS (not overfit)' if ho_p > ho_b + 0.1 else 'NO transfer (overfit to train state)'}")
    env.close()


if __name__ == "__main__":
    main()
