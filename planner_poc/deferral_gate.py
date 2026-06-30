"""deferral_gate.py -- is "when to defer" LEARNABLE? (Path B from FEASIBILITY_JOINT_AND_JUDGE.md §3)

The frozen 2B never defers zero-shot (0/5 probe). The user wants deferral via output, but agreed it needs
training. This tests the prerequisite: is the plan-vs-null BENEFIT predictable from the situation? If a
cheap head on the frozen plan hidden can predict "plan does NOT beat null here" (defer-correct), then a
learned defer GATE is viable; if not, deferral is hard even with training.

DATA (frame-exact save/load, Sonic): at each step, save_state; roll a NULL chunk -> null_r; load_state;
roll a PLAN chunk -> plan_r; record (plan_hidden_feature, null_r, plan_r). Continue from the PLAN branch
(on-policy-ish). Defer label = 1 if plan_r <= null_r + eps (planning doesn't help) else 0.

GATE: a 1-layer head on the frozen plan-hidden mean-pool -> P(defer), trained supervised (episode-split).
Reports val accuracy vs majority, and reward-impact of an ORACLE gate (how much GT reward you'd keep by
deferring exactly the no-benefit situations) as the upside ceiling.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=3 .venv/bin/python planner_poc/deferral_gate.py --episodes 10 --chunks 16
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
from rwbc_actor_adapt import make_env, SYS, INSTR, _syskey


@torch.no_grad()
def collect(pol, env, env_name, episodes, chunks, A, cfg, device, eps_margin=0.02):
    """Per-situation plan-vs-null via save/load on a REUSED env (caller owns construct/close).
    Returns (feats Nx d, labels N, plan_r, null_r, eps)."""
    sk = _syskey(env_name)
    feats, labels, prs, nrs, epi = [], [], [], [], []
    for ep in range(episodes):
            torch.manual_seed(ep); np.random.seed(ep)
            cur = env.reset(); hist = [cur]; plan = "move right"
            for t in range(chunks):
                if t % A == 0:
                    plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                           system=SYS[sk], max_new_tokens=32) or "move right")
                st = env.save_state()
                # NULL branch
                cn = pol._sample_chunk(cur, "", cfg, plan_frames=[cur], null=True)
                _, nr, nd, _ = env.step(cn[:A]); 
                # restore, PLAN branch
                env.load_state(st)
                cp = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False)
                obs, pr, pd, _ = env.step(cp[:A])
                # feature = frozen plan hidden (mean-pooled text tokens)
                h, kpm = pol.pl.encode_multimodal([cur], plan or ".", device, text_only=True)
                valid = (~kpm[0]).float().unsqueeze(-1)
                feat = (h[0] * valid).sum(0) / valid.sum().clamp(min=1)
                feats.append(feat.float().cpu().numpy())
                prs.append(float(pr)); nrs.append(float(nr))
                labels.append(1 if pr <= nr + eps_margin else 0)   # 1 = defer (plan doesn't help)
                epi.append(ep)
                cur = obs; hist.append(cur)
                if pd:
                    break
    return (np.asarray(feats, np.float32), np.asarray(labels, np.int64),
            np.asarray(prs, np.float32), np.asarray(nrs, np.float32), np.asarray(epi))


def train_eval_gate(X, y, pr, nr, ep, device, eps_margin):
    """Train a 1-layer defer gate (episode-split) + report acc/oracle. Returns nothing (prints)."""
    n = len(y)
    print(f"\ncollected {n} situations | defer-rate(plan<=null+{eps_margin}) = {y.mean():.2f}")
    print(f"  mean plan_r={pr.mean():+.3f}  null_r={nr.mean():+.3f}  (plan-null={np.mean(pr-nr):+.3f})")
    if n < 20 or y.sum() < 3 or (1 - y).sum() < 3:
        print("  too few / imbalanced situations to train a gate -> increase --episodes/--chunks.")
        return
    uniq = sorted(set(ep.tolist())); rng = np.random.default_rng(0); rng.shuffle(uniq)
    nval = max(1, len(uniq) // 3); val = set(uniq[:nval])
    tr = np.array([e not in val for e in ep]); va = ~tr
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr = torch.tensor((X[tr]-mu)/sd, device=device); Xva = torch.tensor((X[va]-mu)/sd, device=device)
    ytr = torch.tensor(y[tr], device=device)
    gate = torch.nn.Linear(X.shape[1], 2).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(300):
        opt.zero_grad(); loss = lossf(gate(Xtr), ytr); loss.backward(); opt.step()
    with torch.no_grad():
        pred = gate(Xva).argmax(1).cpu().numpy()
    yva = y[va]
    acc = float((pred == yva).mean()); maj = float(max(yva.mean(), 1-yva.mean()))
    tp = int(((pred==1)&(yva==1)).sum()); fp = int(((pred==1)&(yva==0)).sum())
    pos = int((yva==1).sum()); neg = int((yva==0).sum())
    print(f"\n=== DEFER GATE (predict 'plan does NOT beat null'), val n={len(yva)} ===")
    print(f"  gate acc = {acc:.3f}  (majority {maj:.3f})  defer-recall={tp}/{pos}  false-defer={fp}/{neg}")
    base = float(pr.mean())
    oracle = float(np.where(y==1, np.maximum(pr, nr), pr).mean())
    print(f"  per-situation mean reward: as-is(plan always)={base:+.3f}  oracle-defer={oracle:+.3f} "
          f"(Δ={oracle-base:+.3f})")
    print(f"  => deferral is {'LEARNABLE (gate beats majority)' if acc>maj+0.05 else 'NOT clearly learnable from the plan hidden'}; "
          f"oracle upside {'material' if oracle-base>0.05 else 'small'}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--eps-margin", type=float, default=0.02)
    args = ap.parse_args()
    device = "cuda"

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("defer_gate", plan="", objective="progress", cfg_scale=args.cfg))

    env = make_env(args.env)
    try:
        X, y, pr, nr, ep = collect(pol, env, args.env, args.episodes, args.chunks, args.A, args.cfg,
                                   device, args.eps_margin)
    finally:
        env.close()
    train_eval_gate(X, y, pr, nr, ep, device, args.eps_margin)


if __name__ == "__main__":
    main()
