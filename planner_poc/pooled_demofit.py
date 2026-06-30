"""pooled_demofit.py — the "ONE supervised objective over ALL plan-OOD games" generalist test (R8).

Fits the PLAN-HEAD (LoRA+base frozen) on the POOLED demo chunks of multiple plan-OOD games (SMW + MMX + SMB1),
each chunk conditioned on ITS game's correct plan (BATTERY[game]['correct']). Then evals inference-time Δ_plan
(plan-advance − null-advance from fixed demo save-states) on each eval game. Tests whether one pooled fit
generalizes across obstacle platformers (vs per-game fits), and naturally no-ops where the frame determines the
action. Reuses demo_bc primitives (console->25dim, trim, build_batch, eval_from_states). Recoverable, no commits.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env, build_batch
from plan_graded_test import BATTERY
from demo_bc import load_demo_chunks, demo_start_states, eval_from_states, GAME_CFG

F = "/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-games", default="smw,mmx,smbas", help="pool demos from these games")
    ap.add_argument("--eval-games", default="smw,mmx", help="eval Δ_plan on these (need an env+start states)")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--eval-chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--max-per-game", type=int, default=2500)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--save-delta", default=os.path.join(F, "pooled_planfit.pt"))
    args = ap.parse_args()
    np.random.seed(args.seed_offset); torch.manual_seed(args.seed_offset)
    device = "cuda"

    # pool demos, each tagged with its game's correct plan
    pool = []
    for g in args.train_games.split(","):
        cfg = GAME_CFG[g]
        plan = BATTERY[g]["correct"]
        chunks = load_demo_chunks(cfg["demo_glob"], chunk_stride=18, plan=plan)
        if len(chunks) > args.max_per_game:
            idx = np.random.choice(len(chunks), args.max_per_game, replace=False)
            chunks = [chunks[i] for i in idx]
        pool.extend(chunks)
        print(f"[pool] {g}: +{len(chunks)} chunks (plan={plan[:40]!r})", flush=True)
    print(f"[pool] total {len(pool)} chunks", flush=True)

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("pooled", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith("plan_head."))     # PLAN-HEAD ONLY (LoRA+base frozen)
    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable {sum(p.numel() for p in train_params)/1e6:.2f}M (plan_head only)", flush=True)

    # PRE eval (frozen)
    def eval_all(tag):
        out = {}
        for g in args.eval_games.split(","):
            cfg = GAME_CFG[g]; env = make_env(cfg["env"])
            try:
                states = demo_start_states(cfg["demo_glob"])[:6]
                torch.manual_seed(0)
                nu, _ = eval_from_states(pol, env, states, "", args.eval_chunks, args.A, args.cfg, null=True)
                torch.manual_seed(0)
                pl, _ = eval_from_states(pol, env, states, BATTERY[g]["correct"], args.eval_chunks, args.A, args.cfg)
            finally:
                env.close()
            out[g] = (round(nu, 1), round(pl, 1), round(pl - nu, 1))
            print(f"   [{tag}] {g}: null={nu:+.1f} plan={pl:+.1f} Δ_plan={pl-nu:+.1f}", flush=True)
        return out

    print("[pre] frozen Δ_plan:", flush=True); pre = eval_all("pre")

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    for step in range(args.steps):
        m.train()
        idx = np.random.choice(len(pool), size=min(args.bs, len(pool)), replace=False)
        batch = build_batch(pol, [pool[i] for i in idx], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(batch)["loss"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"   step {step:3d} loss {float(loss.detach()):.4f}", flush=True)
    m.eval()

    if args.save_delta:
        sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
        torch.save({"trainable": sd, "train_games": args.train_games}, args.save_delta)
        print(f"  saved pooled delta ({len(sd)} tensors) -> {args.save_delta}", flush=True)

    print("[post] pooled Δ_plan:", flush=True); post = eval_all("post")
    print("\n===== POOLED PLAN-OOD DEMO-FIT =====")
    for g in args.eval_games.split(","):
        print(f"  {g}: Δ_plan {pre[g][2]:+.1f} -> {post[g][2]:+.1f}  (change {post[g][2]-pre[g][2]:+.1f})")


if __name__ == "__main__":
    main()
