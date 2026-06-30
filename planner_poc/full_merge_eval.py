"""full_merge_eval.py — the COMPLETE one-generalist: pooled plan-OOD plan-head + Sonic actor-OOD lora, merged
onto one btn_s600, evaluated across SMW + MMX (plan-OOD) and Sonic (actor-OOD). Tests that the full routed-+-
merged generalist keeps every game's gain with no interference.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from plan_graded_test import BATTERY
from demo_bc import demo_start_states, eval_from_states, GAME_CFG

F = "/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files"


def load(path, prefixes, substr=False):
    d = torch.load(path, map_location="cpu", weights_only=False)
    t = d.get("trainable", d.get("trainable_ema", {}))
    if substr:
        return {k: v for k, v in t.items() if any(p in k for p in prefixes)}
    return {k: v for k, v in t.items() if any(k.startswith(p) for p in prefixes)}


def fresh(ckpt, qwen):
    pol = NitroGenPolicy(ckpt, qwen=qwen, default_cfg=8.0)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("fm", plan="", objective="progress", cfg_scale=8.0))
    return pol


def apply(pol, deltas):
    msd = pol.m.state_dict(); n = 0
    for t in deltas:
        for k, v in t.items():
            if k in msd:
                msd[k] = v.to(msd[k].device, msd[k].dtype); n += 1
    pol.m.load_state_dict(msd, strict=False); return n


@torch.no_grad()
def ev(pol, game, n_starts=6):
    cfg = GAME_CFG[game]; env = make_env(cfg["env"])
    try:
        states = demo_start_states(cfg["demo_glob"])[:n_starts]
        torch.manual_seed(0); nu, _ = eval_from_states(pol, env, states, "", 16, 2, 8.0, null=True)
        torch.manual_seed(0); pl, _ = eval_from_states(pol, env, states, BATTERY[game]["correct"], 16, 2, 8.0)
    finally:
        env.close()
    return {"null": round(nu, 1), "plan": round(pl, 1), "dplan": round(pl - nu, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--pooled", default=os.path.join(F, "pooled_planfit_s0.pt"))
    ap.add_argument("--sonic-lora", default=os.path.join(F, "r7local_sonic_s0.pt"))
    ap.add_argument("--games", default="smw,mmx,sonic")
    ap.add_argument("--out", default=os.path.join(F, "full_merge_eval.json"))
    args = ap.parse_args()

    ph = load(args.pooled, ["plan_head."])
    lora = load(args.sonic_lora, ["lora"], substr=True)
    print(f"pooled plan-head: {len(ph)} tensors | sonic lora: {len(lora)} tensors", flush=True)
    games = args.games.split(",")

    cells = {"base": [], "pooled_planhead": [ph], "sonic_lora": [lora], "FULL_MERGE": [ph, lora]}
    rep = {}
    for name, deltas in cells.items():
        pol = fresh(args.ckpt, args.qwen); napp = apply(pol, deltas)
        rep[name] = {g: ev(pol, g) for g in games}
        line = "  ".join(f"{g}:Δp={rep[name][g]['dplan']:+.0f}(pl{rep[name][g]['plan']:+.0f})" for g in games)
        print(f"  [{name:16s}] {line}", flush=True)
        del pol; torch.cuda.empty_cache()

    print("\n===== FULL GENERALIST INTERFERENCE CHECK =====")
    ok = True
    for g in games:
        b = rep["base"][g]["dplan"]
        solo = rep["pooled_planhead" if g != "sonic" else "sonic_lora"][g]
        sk = "plan" if g != "sonic" else "plan"
        solo_v = solo["dplan"] if g != "sonic" else rep["sonic_lora"][g]["plan"]
        full = rep["FULL_MERGE"][g]["dplan"] if g != "sonic" else rep["FULL_MERGE"][g]["plan"]
        keep = full >= solo_v - (10 if g != "sonic" else 40)
        ok = ok and keep
        print(f"  {g:6s}: base {b:+.0f} -> solo {solo_v:+.0f} -> FULL {full:+.0f}  keeps={keep}")
    print(f"  => {'FULL ONE-GENERALIST OK (all games retained)' if ok else 'INTERFERENCE on some game'}")
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
