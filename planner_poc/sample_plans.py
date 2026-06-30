"""Quick: print the CORRECTED per-game plan (game_planner.ClosedLoopPlanner, plan mode) on each demo's
first frame, so we can eyeball that the fixed prompts are sensible (Minish no-jump, FE cursor/menu, etc.)."""
from __future__ import annotations
import glob, gzip, os, sys
import numpy as np

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from demo_train_stack import GAME_META
from game_planner import ClosedLoopPlanner

ENVMAP = {"fireemblem": "gba_fire_emblem_sacred_stones"}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["smw", "sonic", "minish", "fireemblem"])
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--states", type=int, default=2)
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=8.0)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("sample", plan="", objective="progress", cfg_scale=8.0))

    for game in args.games:
        meta = GAME_META[game]; envname = meta["env"] or ENVMAP.get(game)
        states = [gzip.decompress(open(p, "rb").read())
                  for p in sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"], "*", "initial.state")))][: args.states]
        env = make_env(envname)
        print(f"\n========== {game.upper()} ({envname}) ==========", flush=True)
        try:
            for si, st in enumerate(states):
                env.reset(); env.load_state(st)
                f = env.frame()
                clp = ClosedLoopPlanner(pol, game, mode="plan")
                plan = clp.plan([f])
                # also show learnings (learn mode) for one state
                clp2 = ClosedLoopPlanner(pol, game, mode="learn")
                _ = clp2.plan([f])
                print(f"  [state{si}] PLAN: {plan}", flush=True)
                if si == 0 and getattr(clp2, "learnings", ""):
                    print(f"  [state{si}] LEARNINGS: {clp2.learnings}", flush=True)
        finally:
            env.close()

if __name__ == "__main__":
    main()
