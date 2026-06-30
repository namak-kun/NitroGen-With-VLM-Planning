"""expressiveness_battery.py — R9 Q4: does a demo-fit preserve the BASE bridge's plan-conditioned expressiveness
across the WHOLE action space, not just advance? Scores 5 plan->action contrasts (duck/retreat/jump/wait/up) the
base bridge already obeys, for base + any list of deltas, from fixed demo start states with matched seeds.

effect_i = rate(probe_plan) - rate(neutral_plan)  for the contrast's action dim.
Pass (per delta): sign-match >=4/5 vs base, responsiveness R = mean_i clip(d_i/b_i,0,1) >= 0.5, duck mandatory.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/expressiveness_battery.py --game smw \
      --deltas r9_smw_situ_s0.pt r9_smw_ctrl_s0.pt r9_smw_kl_s0.pt --tag r9
"""
from __future__ import annotations
import argparse, gzip, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from demo_bc import GAME_CFG, demo_start_states
from plan_graded_test import BATTERY

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
_F = "/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files"

# (name, probe_plan, neutral_plan_or_None=advance, dim_fn) ; dim_fn(allrows)->rate the BASE should raise (or drop)
NEUTRAL = None  # -> the game's advance plan


def rate_down(a):       return float((a[:, 1] > 0.5).mean())         # DUCK (dpad_down dim1 = the canonical channel;
#   the base's stick-y dim22 rests high/noisy so a combined stick|dpad metric mis-reads base duck as negative)
def rate_left(a):       return float((a[:, 21] < 0.4).mean())        # RETREAT (JLX left)
def rate_jump(a):       return float((a[:, 18] > 0.5).mean())        # JUMP (south)
def rate_right(a):      return float((a[:, 21] > 0.6).mean())        # WAIT measured as DROP in RIGHT
def rate_up(a):         return float((a[:, 22] < 0.4).mean())        # CLIMB/UP (stick up)

CONTRASTS = [
    ("duck",    "press down to duck under it",            NEUTRAL, rate_down,      +1),
    ("retreat", "turn around and move left, go back",     NEUTRAL, rate_left,      +1),
    ("jump",    "jump now to clear the gap ahead",        "walk forward slowly, do not jump", rate_jump, +1),
    ("wait",    "stop and wait, hold still",              NEUTRAL, rate_right,     -1),   # expect RIGHT to drop
    ("up",      "climb up the ladder or vine",            NEUTRAL, rate_up,        +1),
]


@torch.no_grad()
def rollout_rate(pol, env, states, plan, dim_fn, n_chunks=12):
    rates = []
    for st in states:
        env.reset(); env.load_state(st)
        if env.frame().mean() < 1.0:
            env._emu_step([], 1)
        rows = []
        for _ in range(n_chunks):
            ch = np.asarray(pol._sample_chunk(env.frame(), plan, 8.0, plan_frames=[env.frame()]), np.float32)
            rows.append(ch); env.step(ch[:2])
        rates.append(dim_fn(np.concatenate(rows, 0)))
    return float(np.mean(rates))


def score(pol, env, states, advance):
    out = {}
    for name, probe, neutral, fn, sign in CONTRASTS:
        npl = advance if neutral is NEUTRAL else neutral
        r_probe = rollout_rate(pol, env, states, probe, fn)
        r_neutral = rollout_rate(pol, env, states, npl, fn)
        out[name] = {"effect": (r_probe - r_neutral) * sign, "probe": r_probe, "neutral": r_neutral, "sign": sign}
    return out


def load_delta(pol, path):
    p = path if os.path.isabs(path) else os.path.join(_F, path)
    d = torch.load(p, map_location="cpu", weights_only=False)["trainable"]
    sd = pol.m.state_dict()
    for k, v in d.items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].dtype))
    return os.path.basename(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--deltas", nargs="*", default=[])
    ap.add_argument("--eval-starts", type=int, default=8)
    ap.add_argument("--tag", default="r9")
    args = ap.parse_args()

    cfg = GAME_CFG[args.game]; advance = BATTERY[args.game]["correct"]
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=8.0)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("battery", plan="", objective="progress", cfg_scale=8.0))
    base_sd = {k: v.detach().clone() for k, v in pol.m.state_dict().items()}
    env = make_env(cfg["env"])
    states = demo_start_states(cfg["demo_glob"])[:args.eval_starts]

    results = {}
    pol.m.eval()
    print("=== BASE ===", flush=True)
    base = score(pol, env, states, advance)
    for n, v in base.items():
        print(f"  {n:8s} effect={v['effect']:+.3f}  (probe {v['probe']:.3f} vs neutral {v['neutral']:.3f})")
    results["base"] = base
    b = {n: base[n]["effect"] for n in base}

    for dpath in args.deltas:
        pol.m.load_state_dict(base_sd, strict=False)            # reset to base
        name = load_delta(pol, dpath); pol.m.eval()
        print(f"\n=== {name} ===", flush=True)
        d = score(pol, env, states, advance)
        sign_ok = 0; ratios = []
        for n, v in d.items():
            bi = b[n]; di = v["effect"]
            smatch = (np.sign(di) == np.sign(bi)) or abs(bi) < 1e-3
            sign_ok += int(smatch)
            ratios.append(np.clip(di / bi, 0, 1) if abs(bi) > 1e-3 else 1.0)
            print(f"  {n:8s} effect={di:+.3f}  base={bi:+.3f}  sign={'ok' if smatch else 'FLIP'}  "
                  f"ratio={ratios[-1]:.2f}")
        R = float(np.mean(ratios))
        duck_ok = d["duck"]["probe"] * 100 >= 5 or d["duck"]["effect"] >= b["duck"] * 0.5
        verdict = "PASS" if (sign_ok >= 4 and R >= 0.5 and duck_ok) else "FAIL"
        print(f"  => sign-match {sign_ok}/5  R={R:.2f}  duck_ok={duck_ok}  ==> {verdict}", flush=True)
        results[name] = {"contrasts": d, "sign_match": sign_ok, "R": R, "duck_ok": bool(duck_ok), "verdict": verdict}

    env.close()
    outp = os.path.join(_F, f"expr_battery_{args.tag}_{args.game}.json")
    json.dump(results, open(outp, "w"), indent=1)
    print(f"\n[battery] -> {outp}", flush=True)


if __name__ == "__main__":
    main()
