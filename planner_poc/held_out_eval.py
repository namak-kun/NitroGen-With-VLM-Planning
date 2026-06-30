"""held_out_eval.py -- does the RWBC actor-adaptation GENERALIZE to held-out levels? RWBC trained from the
env's DEFAULT start state; here we eval BASE vs ADAPTED (loaded delta) on the human-DEMO start states
(different sections of the SAME game), with matched seeds + the fixed correct plan. Tests cross-section
transfer (the user's "won't generalize" worry).

Run:  RUN=... CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/held_out_eval.py \
        --game sonic --delta ckpts/rwbc_sonic_correct_scaled.pt
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, sys
import numpy as np
import torch

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from demo_train_stack import GAME_META
from plan_graded_test import BATTERY

ENVMAP = {"fireemblem": "gba_fire_emblem_sacred_stones"}


@torch.no_grad()
def eval_states(pol, env, states, plan, w, chunks, A, seeds):
    use_var = hasattr(env, "reward_var")
    out, shaped = [], []      # out = progress (screen_x advance); shaped = death/stuck-penalized env reward
    for base in seeds:
        for si, st in enumerate(states):
            env.reset(); env.load_state(st)
            x0 = env._var(env.reward_var) if use_var else 0.0
            acc = 0.0; sh = 0.0
            for t in range(chunks):
                f = env.frame()
                ch = pol._sample_chunk(f, plan, w, plan_frames=[f], null=False, noise_seed=base * 1009 + t)
                o = env.step(ch[:A])
                if isinstance(o, tuple) and len(o) >= 2:
                    sh += float(o[1])
                    if not use_var:
                        acc += float(o[1])
            out.append(float(env._var(env.reward_var) - x0) if use_var else acc)
            shaped.append(sh)
    return float(np.mean(out)), out, float(np.mean(shaped))


def load_delta(pol, path, device):
    d = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
    sd = dict(pol.m.named_parameters())
    n = 0
    for k, t in d.items():
        if k in sd:
            sd[k].data.copy_(t.to(device)); n += 1
    return n, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--delta", required=True)
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--w", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=4)
    args = ap.parse_args()
    device = "cuda"

    meta = GAME_META[args.game]; envname = meta["env"] or ENVMAP.get(args.game)
    plan = BATTERY[args.game]["correct"]
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.w)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("heldout", plan="", objective="progress", cfg_scale=args.w))

    states = [gzip.decompress(open(p, "rb").read())
              for p in sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"],
                                                      "*", "initial.state")))][: args.max_states]
    seeds = list(range(args.seeds))
    env = make_env(envname)
    try:
        print(f"\n===== HELD-OUT GEN EVAL ({args.game}, {len(states)} demo states, delta={os.path.basename(args.delta)}) =====", flush=True)
        base_m, base_l, base_sh = eval_states(pol, env, states, plan, args.w, args.chunks, args.A, seeds)
        print(f"  BASE    progress={base_m:+.2f}  shaped={base_sh:+.2f}   {[round(x,1) for x in base_l]}", flush=True)
        nset, ntot = load_delta(pol, args.delta, device)
        print(f"  loaded delta: {nset}/{ntot} tensors", flush=True)
        adpt_m, adpt_l, adpt_sh = eval_states(pol, env, states, plan, args.w, args.chunks, args.A, seeds)
        print(f"  ADAPTED progress={adpt_m:+.2f}  shaped={adpt_sh:+.2f}   {[round(x,1) for x in adpt_l]}", flush=True)
        print(f"  >>> Δprogress = {adpt_m-base_m:+.2f} | Δshaped = {adpt_sh-base_sh:+.2f}  "
              f"({'GENERALIZES (both up)' if adpt_m>base_m and adpt_sh>base_sh else 'screen_x up but shaped flat/down -> CHECK exploit' if adpt_m>base_m else 'no transfer'})",
              flush=True)
        out = os.path.join(_R, "docs", "graded", f"heldout_{args.game}.json")
        json.dump({"game": args.game, "delta": args.delta, "base": base_m, "adapted": adpt_m,
                   "base_shaped": base_sh, "adapted_shaped": adpt_sh,
                   "base_list": base_l, "adapted_list": adpt_l}, open(out, "w"), indent=2)
        print(f"  wrote {out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
