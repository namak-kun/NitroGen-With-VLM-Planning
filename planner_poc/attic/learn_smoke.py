"""Smoke-test the REDESIGNED closed-loop learn mode: roll a demo, replan every A chunks, and print the
prior plan -> executed trace (what System 1 did) -> updated <learnings> -> new <plan>, so we can see the
learnings are now BEHAVIOURAL (grounded on the prior plan + execution) rather than control restatements."""
from __future__ import annotations
import argparse, glob, gzip, os, sys
import numpy as np

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from demo_train_stack import GAME_META
from game_planner import ClosedLoopPlanner, applied_action_desc

ENVMAP = {"fireemblem": "gba_fire_emblem_sacred_stones"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    args = ap.parse_args()
    device = "cuda"

    meta = GAME_META[args.game]; envname = meta["env"] or ENVMAP.get(args.game)
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("smoke", plan="", objective="progress", cfg_scale=args.cfg))

    st = gzip.decompress(open(sorted(glob.glob(os.path.join(
        _R, "docs/demos/demos", meta["dir"], "*", "initial.state")))[0], "rb").read())
    env = make_env(envname)
    try:
        env.reset(); env.load_state(st)
        clp = ClosedLoopPlanner(pol, args.game, mode="learn")
        hist = [env.frame()]
        print(f"\n############ {args.game.upper()} closed-loop LEARN smoke ############", flush=True)
        for t in range(args.chunks):
            if t % args.A == 0:
                # show what the planner is about to consume
                print(f"\n--- replan @t={t} ---")
                print(f"  prior_plan: {clp.last_plan!r}")
                print(f"  trace_in  : {[d for d,_ in clp.trace]}")
                plan = clp.plan(hist[-4:])
                print(f"  LEARNINGS : {clp.learnings}")
                print(f"  NEW PLAN  : {plan}", flush=True)
            f = env.frame()
            ch = pol._sample_chunk(f, plan, args.cfg, plan_frames=[f], null=False,
                                   noise_seed=7 * t + 1)
            env.step(ch[:args.A]); hist.append(env.frame())
            clp.observe(applied_action_desc(env, args.game, ch[0]), env.frame())
    finally:
        env.close()


if __name__ == "__main__":
    main()
