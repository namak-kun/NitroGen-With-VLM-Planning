"""eval_prevplans_ablation.py -- ablate the prev-plans knob (@namak-kun: "put the last two plans in the
prompt to incentivize exploration ... maybe ablate?").

Runs matched rollouts from the SAME start/seed with include_prev_plans OFF vs ON, using the interleaved
RL planner observation (rl_planner_prompt.build_rl_messages) in think mode, on a live env. Reports per
condition: cumulative GT reward, plan FLIP-rate (fraction of replans where the new plan != previous plan
-> higher = less repetition / more exploration), and unique-plan fraction. The question: does showing the
last 2 plans change behavior (more diverse plans? better reward? less stuck-looping)?

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/eval_prevplans_ablation.py --env sonic --seeds 3 --cycles 6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from play_annotated_horizon import summarize_actions, _first_sentence
from rl_planner_prompt import build_rl_messages, generate_rl_plan, RLPlannerConfig
from rl_rollout_demo import make_env, GAME_INFO


def _norm(p: str) -> str:
    return " ".join((p or "").lower().split())


def run_condition(pol, env_name, include_prev, seed, cycles, cfg, device, rlcfg,
                  think_budget=180, plan_budget=36):
    ginfo, genre = GAME_INFO[env_name]
    A, S, sub_len = rlcfg.num_chunks, rlcfg.intra_chunk_rate, rlcfg.subchunk_len
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(env_name)
    if hasattr(env, "A"):
        try: env.A = sub_len
        except Exception: pass
    cur = env.reset()
    plan = "move right"
    plan_hist, plans, cumr = [], [], 0.0
    done = False
    for cyc in range(cycles):
        f0 = np.asarray(cur).copy(); recs = []
        for ci in range(A):
            chunk = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False)
            subs = []
            for si in range(S):
                sub = chunk[si*sub_len:(si+1)*sub_len]
                if sub.shape[0] == 0: break
                obs, r, d, info = env.step(sub); cumr += float(r)
                subs.append((summarize_actions([row for row in sub], 0.2), np.asarray(obs).copy()))
                cur = obs; done = done or bool(d)
                if done: break
            recs.append({"subs": subs})
            if done: break
        sysd, content, images, _ = build_rl_messages(
            f0, recs, genre=genre, game_info=ginfo, cfg=rlcfg,
            prev_plans=plan_hist, include_prev_plans=include_prev)
        res = generate_rl_plan(pol.pl, sysd, content, images, device, enable_thinking=True,
                               think_budget=think_budget, plan_budget=plan_budget)
        new_plan = _first_sentence(res["plan"]) or plan
        plans.append(new_plan)
        plan_hist = (plan_hist + [plan])[-2:]
        plan = new_plan
        if done: break
    env.close()
    # metrics
    flips = sum(1 for i in range(1, len(plans)) if _norm(plans[i]) != _norm(plans[i-1]))
    flip_rate = flips / max(1, len(plans)-1)
    uniq = len({_norm(p) for p in plans}) / max(1, len(plans))
    return {"cumr": cumr, "flip_rate": flip_rate, "uniq": uniq, "n": len(plans), "plans": plans}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sonic", choices=sorted(GAME_INFO))
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--cfg", type=float, default=8.0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rlcfg = RLPlannerConfig()
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.reset(Scenario("prevplans_ablation", plan="", objective="progress", cfg_scale=args.cfg))

    agg = {False: [], True: []}
    for seed in range(args.seeds):
        for include_prev in (False, True):
            m = run_condition(pol, args.env, include_prev, 1000+seed, args.cycles, args.cfg, device, rlcfg)
            agg[include_prev].append(m)
            tag = "PREV-ON " if include_prev else "PREV-OFF"
            print(f"[seed {seed}] {tag}: cumR={m['cumr']:+.2f} flip={m['flip_rate']:.2f} "
                  f"uniq={m['uniq']:.2f} n={m['n']}")

    def mean(cond, k): return float(np.mean([x[k] for x in agg[cond]]))
    print(f"\n===== PREV-PLANS ABLATION ({args.env}, {args.seeds} seeds x {args.cycles} cycles) =====")
    for cond, lbl in ((False, "OFF"), (True, "ON ")):
        print(f"  prev-plans {lbl}: cumR={mean(cond,'cumr'):+.2f}  flip_rate={mean(cond,'flip_rate'):.2f}  "
              f"uniq={mean(cond,'uniq'):.2f}")
    dR = mean(True,'cumr')-mean(False,'cumr'); dF = mean(True,'flip_rate')-mean(False,'flip_rate')
    print(f"  Δ(ON-OFF): reward {dR:+.2f}, flip_rate {dF:+.2f}")
    print(f"  -> prev-plans {'INCREASE plan diversity' if dF>0.05 else ('DECREASE diversity' if dF<-0.05 else 'no clear diversity effect')}; "
          f"reward {'up' if dR>0.3 else ('down' if dR<-0.3 else 'flat')}.")


if __name__ == "__main__":
    main()
