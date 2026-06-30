"""maneuver_router.py — R10/R11 EVOCATION-vs-ADDITION router on the FROZEN DiT. Answers the owner's literal
questions: "does the DiT know spin-jump kills a Rex?" and (when annotated) "can it grab the mesh?" — by
reconstructing frame-exact save-states AT each maneuver (from the gold-narration frame indices) and measuring,
with NO training:
  - reflex_rate : base-NULL DiT maneuver-dim press-rate (is the maneuver the DiT's DEFAULT reflex?)
  - evoc_ratio  : maneuver-dim rate under an EXPLICIT plan / null rate (does the plan SUMMON it?)
  - outcome_Δ   : EMULATOR counterfactual P(success|plan) - P(success|null) [decisive for OUTCOME maneuvers like
                  spin-jump-kills-Rex, where a button-press != the outcome]. success = score-up AND survived.
2-stage gate (Opus R10/R11): reflex (reflex_rate high) -> no-op; else evoke (evoc_ratio>=2 OR outcome_Δ>=+0.3)
-> short path (KL-anchored demo-fit); else ADD (DiT-LoRA). null-AUC alone misroutes (duck is evocable but low
default-rate) so we use evoc_ratio + the emulator outcome, not just the reflex rate.

Run (frozen, ~1 GPU):
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=3 .venv/bin/python planner_poc/maneuver_router.py --out <file>.json
"""
from __future__ import annotations
import argparse, gzip, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from demo_bc import map_action, GAME_CFG

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
DEMODIR = "docs/demos/demos/SuperMarioWorld-Snes"

# maneuver-dim press-rate fns over a chunk (H,25): stick-down=22>0.6, dpad-down=1, jump/south=18
def r_down(ch):  a = np.asarray(ch); return float(((a[:, 22] > 0.6) | (a[:, 1] > 0.5)).mean())
def r_jump(ch):  return float((np.asarray(ch)[:, 18] > 0.5).mean())

# (demo, narrated_frame) lists from narration.json; reconstruct ~PRE_OFFSET frames BEFORE the action so the
# maneuver is imminent but not yet executed (tests whether the DiT CAN be steered into it from that state).
PRE_OFFSET = 30
DUCK_FRAMES = [("20260627-105913", 480), ("20260627-105939", 840), ("20260627-105939", 5340)]
REX_FRAMES = [("20260627-105939", f) for f in
              (960, 1260, 1320, 1440, 1740, 3960, 4320, 4980, 6120, 6480)]

MANEUVERS = {
    "duck": dict(frames=DUCK_FRAMES, dim=r_down, outcome=False,
                 plan="press down to duck under the bullet bill"),
    "spinjump_rex": dict(frames=REX_FRAMES, dim=r_jump, outcome=True,
                         plan="spin jump on the rex to kill it and keep moving right"),
}


def reconstruct(env, demo, frame):
    """Replay the demo's own actions to (frame - PRE_OFFSET) from initial.state; return a save-state there."""
    d = os.path.join(_R, DEMODIR, demo)
    acts = np.load(os.path.join(d, "demo.npz"), allow_pickle=True)["actions"]
    env.reset(); env.load_state(gzip.decompress(open(os.path.join(d, "initial.state"), "rb").read()))
    if env.frame().mean() < 1.0:
        env._emu_step([], 1)
    target = max(0, frame - PRE_OFFSET)
    for i in range(min(target, len(acts))):
        env._emu_step(env.action_row_to_buttons(map_action(acts[i])), 1)
    return env.save_state()


@torch.no_grad()
def behavioral_rate(pol, env, state, plan, dim_fn, null, cfg=8.0, k=4):
    """Mean maneuver-dim press-rate over k stochastic chunks sampled from the state under (plan or null)."""
    rates = []
    for s in range(k):
        env.reset(); env.load_state(state)
        if env.frame().mean() < 1.0:
            env._emu_step([], 1)
        ch = pol._sample_chunk(env.frame(), "" if null else plan, cfg, plan_frames=[env.frame()],
                               null=null, noise_sigma=0.6, noise_seed=s + 1)
        rates.append(dim_fn(ch))
    return float(np.mean(rates))


@torch.no_grad()
def emulator_outcome(pol, env, state, plan, null, cfg=8.0, k=6, chunks=4, A=2, advance_px=25.0):
    """P(handled) from the state under (plan or null): success = ADVANCED past the threat (screen_x rose by
    >= advance_px) AND no life lost (survived contact) within `chunks`. NOTE: this is 'handled the rex'
    (kill OR precise dodge), not strictly 'killed' -- the SMW 'score' var does NOT register enemy kills
    (verified: human's spin-jump-kill left score flat), and distinguishing kill-vs-dodge needs sprite-status
    RAM (follow-up). Survive+advance is the robust, honest signal."""
    succ = 0
    for s in range(k):
        env.reset(); env.load_state(state)
        if env.frame().mean() < 1.0:
            env._emu_step([], 1)
        x0 = env._var(env.reward_var); lv0 = env._var("lives")
        advanced = False; died = False
        for c in range(chunks):
            ch = pol._sample_chunk(env.frame(), "" if null else plan, cfg, plan_frames=[env.frame()],
                                   null=null, noise_sigma=0.6, noise_seed=10 * s + c + 1)
            env.step(ch[:A])
            if env._var(env.reward_var) - x0 >= advance_px:
                advanced = True
            lv = env._var("lives")
            if 0 < lv0 - lv <= 3:
                died = True; break
        succ += int(advanced and not died)
    return succ / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=8.0)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("router", plan="", objective="progress", cfg_scale=8.0)); pol.m.eval()
    env = make_env(GAME_CFG[args.game]["env"])
    results = {}
    try:
        for name, cfg in MANEUVERS.items():
            print(f"\n=== maneuver: {name}  (plan={cfg['plan']!r}) ===", flush=True)
            states = []
            for demo, fr in cfg["frames"]:
                try:
                    states.append(reconstruct(env, demo, fr))
                except Exception as e:
                    print(f"  reconstruct {demo} f{fr} FAILED: {e}", flush=True)
            print(f"  reconstructed {len(states)} save-states", flush=True)
            null_r = np.mean([behavioral_rate(pol, env, st, cfg["plan"], cfg["dim"], True, k=args.k) for st in states])
            plan_r = np.mean([behavioral_rate(pol, env, st, cfg["plan"], cfg["dim"], False, k=args.k) for st in states])
            evoc_ratio = plan_r / max(null_r, 1e-3)
            row = {"n_states": len(states), "reflex_rate": float(null_r), "evoc_rate": float(plan_r),
                   "evoc_ratio": float(evoc_ratio)}
            print(f"  reflex(null) {null_r:.3f} | evoke(plan) {plan_r:.3f} | evoc_ratio {evoc_ratio:.2f}", flush=True)
            if cfg["outcome"]:
                on = np.mean([emulator_outcome(pol, env, st, cfg["plan"], True, k=args.k + 2) for st in states])
                op = np.mean([emulator_outcome(pol, env, st, cfg["plan"], False, k=args.k + 2) for st in states])
                row.update(outcome_null=float(on), outcome_plan=float(op), outcome_delta=float(op - on))
                print(f"  EMULATOR P(kill&survive): null {on:.2f} -> plan {op:.2f}  (Δ {op-on:+.2f})", flush=True)
            # classify (2-stage gate)
            reflexive = null_r >= 0.5
            evocable = evoc_ratio >= 2.0 or row.get("outcome_delta", 0) >= 0.3
            verdict = "REFLEX" if reflexive else ("EVOKE" if evocable else "ADD")
            row["verdict"] = verdict
            print(f"  => VERDICT: {verdict}", flush=True)
            results[name] = row
    finally:
        env.close()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"\n[router] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
