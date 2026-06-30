"""collect_samples.py -- collect RWBC samples for ONE game (fixed correct plan + exploration noise) and save
to disk. Run once PER GAME in its OWN process so emulator-core constructions are segfault-isolated (multiple
stable-retro constructions in one long-lived CUDA process crash). The merged samples feed train_multi.py
(the multi-genre one-LoRA capacity test -- the user's "per-genre actors won't scale / won't generalize"
question, now with the FIXED button maps + corrected plans).

Run:  RUN=... CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/collect_samples.py --env sonic \
        --collect-eps 8 --chunks 12 --explore-sigma 0.3 --out docs/multigenre/sonic.pkl
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env, collect
from plan_graded_test import BATTERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--collect-eps", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--explore-sigma", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    key = {"smw": "smw", "sonic": "sonic", "minish": "minish"}.get(args.env, "smw")
    fixed_plan = BATTERY[key]["correct"]
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("collect", plan="", objective="progress", cfg_scale=args.cfg))

    env = make_env(args.env)
    try:
        samples = collect(pol, env, args.env, args.collect_eps, args.chunks, args.A, args.cfg,
                          explore_sigma=args.explore_sigma, fixed_plan=fixed_plan)
    finally:
        env.close()
    rewards = np.array([s["reward"] for s in samples])
    # tag each sample with its game so train_multi can balance/inspect
    for s in samples:
        s["game"] = args.env
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"env": args.env, "fixed_plan": fixed_plan, "samples": samples}, f)
    print(f"[collect] {args.env}: {len(samples)} chunks, reward mean {rewards.mean():+.3f} "
          f"(min {rewards.min():+.2f} max {rewards.max():+.2f}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
