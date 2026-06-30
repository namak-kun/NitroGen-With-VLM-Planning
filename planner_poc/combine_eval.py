"""combine_eval.py — the "ONE generalist, two training modes" test.

SMW (plan-OOD) mode contributes its PLAN-HEAD demo-fit delta; Sonic (actor-OOD) mode contributes its LoRA
RWBC delta. Merge BOTH onto btn_s600 (plan_head from SMW, lora_ from Sonic) and eval on BOTH games to check
for cross-interference vs each single-mode delta and the base. Eval = mean screen_x advance from FIXED demo
start states (low-variance): SMW reports Δ_plan (plan − null); Sonic reports plan-advance.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from plan_graded_test import BATTERY
from demo_bc import demo_start_states, eval_from_states, GAME_CFG

F = "/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files"


def load_delta(path, only_prefix=None):
    d = torch.load(path, map_location="cpu", weights_only=False)
    t = d.get("trainable", d.get("trainable_ema", {}))
    if only_prefix:
        t = {k: v for k, v in t.items() if any(k.startswith(p) or p in k for p in only_prefix)}
    return t


def apply_deltas(pol, deltas):
    msd = pol.m.state_dict()
    n = 0
    for t in deltas:
        for k, v in t.items():
            if k in msd:
                msd[k] = v.to(msd[k].device, msd[k].dtype); n += 1
    pol.m.load_state_dict(msd, strict=False)
    return n


@torch.no_grad()
def eval_game(pol, game, n_starts=6):
    cfg = GAME_CFG[game]
    plan = BATTERY[game]["correct"]
    env = make_env(cfg["env"])
    try:
        states = demo_start_states(cfg["demo_glob"])[:n_starts]
        torch.manual_seed(0)
        null, _ = eval_from_states(pol, env, states, "", 16, 2, 8.0, null=True)
        torch.manual_seed(0)
        pl, _ = eval_from_states(pol, env, states, plan, 16, 2, 8.0)
    finally:
        env.close()
    return {"null": round(null, 1), "plan": round(pl, 1), "dplan": round(pl - null, 1)}


def fresh_pol(ckpt, qwen):
    pol = NitroGenPolicy(ckpt, qwen=qwen, default_cfg=8.0)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("combine", plan="", objective="progress", cfg_scale=8.0))
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--smw-planhead", default=os.path.join(F, "r7demo_smw_plain_s2.pt"))
    ap.add_argument("--sonic-lora", default=os.path.join(F, "r7local_sonic_s0.pt"))
    ap.add_argument("--out", default=os.path.join(F, "combine_eval.json"))
    args = ap.parse_args()

    smw_ph = load_delta(args.smw_planhead, only_prefix=["plan_head."])
    son_lora = load_delta(args.sonic_lora, only_prefix=["lora_"])
    print(f"SMW plan-head delta: {len(smw_ph)} tensors | Sonic lora delta: {len(son_lora)} tensors", flush=True)

    cells = {
        "base":      [],
        "smw_only":  [smw_ph],
        "sonic_only":[son_lora],
        "combined":  [smw_ph, son_lora],
    }
    report = {}
    for name, deltas in cells.items():
        pol = fresh_pol(args.ckpt, args.qwen)   # fresh load each cell (clean, avoids stacking)
        nap = apply_deltas(pol, deltas)
        smw = eval_game(pol, "smw")
        sonic = eval_game(pol, "sonic")
        report[name] = {"applied_tensors": nap, "smw": smw, "sonic": sonic}
        print(f"  [{name:11s}] SMW Δ_plan={smw['dplan']:+.1f} (plan {smw['plan']:+.1f}/null {smw['null']:+.1f}) | "
              f"Sonic plan={sonic['plan']:+.1f}", flush=True)
        del pol; torch.cuda.empty_cache()

    # interference verdict
    b, s, so, c = report["base"], report["smw_only"], report["sonic_only"], report["combined"]
    print("\n===== INTERFERENCE CHECK =====")
    print(f"  SMW Δ_plan:  base {b['smw']['dplan']:+.1f} -> smw_only {s['smw']['dplan']:+.1f} -> combined {c['smw']['dplan']:+.1f}")
    print(f"  Sonic plan:  base {b['sonic']['plan']:+.1f} -> sonic_only {so['sonic']['plan']:+.1f} -> combined {c['sonic']['plan']:+.1f}")
    smw_keep = c['smw']['dplan'] >= s['smw']['dplan'] - 10
    son_keep = c['sonic']['plan'] >= so['sonic']['plan'] - 30
    print(f"  => combined keeps SMW gain: {smw_keep} ; keeps Sonic gain: {son_keep}")
    print(f"  => {'ONE GENERALIST OK (both modes coexist)' if smw_keep and son_keep else 'INTERFERENCE detected'}")
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
