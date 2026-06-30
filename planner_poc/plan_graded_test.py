"""plan_graded_test.py -- THE GATE experiment (rubber-duck critique #1): is plan value GRADED within plan
space (better plan -> more return) or merely ACTIVATION (any non-null plan > null, all alike)?

We bypass the planner and feed FIXED plan strings through the SAME bridge (eval_policy._sample_chunk with
explicit plan_text), from the human-demo start states, with matched per-(state,chunk) noise seeds. The
battery (per the duck) is a plan-QUALITY ladder, so we can analyze WITHIN the non-null set:
  null        : base DiT, no plan tokens (the inert/activation floor)
  dummy       : off-topic nonsense text, similar length (pure ACTIVATION control: if dummy ~= correct -> activation)
  wrong_game  : another game's correct plan (semantically wrong, same register)
  generic     : "make progress and avoid obstacles" (valid but content-free)
  bad         : direction-OPPOSITE / "turn back" (actively wrong)
  correct     : concise genre+direction-correct plan
  hand_good   : detailed expert-opening plan (the practical ceiling)
Decision: correct/hand_good >> bad/wrong_game ~= dummy > null  => GRADED (planner optimization worth it).
          dummy ~= correct                                      => ACTIVATION only (invest in actor/bridge).

ROBUSTNESS: stable-retro cores SILENTLY SEGFAULT in long-lived CUDA processes after many steps. So we write
each condition's result to a JSON sidecar AS IT COMPLETES (ordered by decisiveness) -> a mid-run crash still
leaves the decisive comparisons on disk.

Run:  RUN=... CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u planner_poc/plan_graded_test.py --game smw
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

ENVMAP = {"fireemblem": "gba_fire_emblem_sacred_stones"}

DUMMY = ("The quarterly logistics report notes a modest seasonal uptick in regional warehouse throughput "
         "across the northern distribution corridor.")
GENERIC = "Make progress and avoid obstacles."
BATTERY = {
    "smw": dict(
        correct="Move right, run, and jump over pits and enemies to advance through the level.",
        hand_good="Hold run and move right, hopping over the first enemies and clearing the gaps with "
                  "well-timed jumps to keep advancing rightward toward the level exit.",
        bad="Turn around and move left, going back toward the start of the level."),
    "sonic": dict(
        correct="Run right at full speed, jumping over gaps and enemies to reach the end of the act.",
        hand_good="Build speed and barrel right across the slopes, jumping the gaps and bopping enemies, "
                  "keeping momentum to blaze toward the end of the act.",
        bad="Turn around and run left, back the way you came."),
    "minish": dict(
        correct="Move right toward the next area and attack anything blocking the path.",
        hand_good="Head right across the room toward the doorway, slashing bushes or enemies in the way, "
                  "and step through to the next screen.",
        bad="Move left, away from the exit, back into the corner."),
    "fireemblem": dict(
        correct="Move the cursor to select a unit and advance it toward the enemies.",
        hand_good="Open the menu, pick a strong unit near the front, and move it toward the enemy line to "
                  "begin engaging.",
        bad="Cancel and move the cursor away into an empty corner of the map."),
    "smbas": dict(
        correct="Move right, run, and jump over pits and enemies to advance through the level.",
        hand_good="Hold run and move right, hopping over the Goombas and Koopas and clearing the pits with "
                  "well-timed jumps.",
        bad="Turn around and move left, going back toward the start of the level."),
    "mmx": dict(
        correct="Move right through the stage, jumping over gaps and shooting or dodging enemies to advance.",
        hand_good="Advance right across the highway, dashing and jumping over the gaps, shooting enemies and "
                  "climbing past obstacles.",
        bad="Turn around and go left, back toward the start of the stage."),
}
ORDER = ["null", "dummy", "wrong_game", "generic", "bad", "correct", "hand_good"]


@torch.no_grad()
def run_cond(pol, env, states, plan_text, null, w, chunks, A, seeds):
    use_var = hasattr(env, "reward_var")
    per_seed = []
    for seed in range(seeds):
        deltas = []
        for si, st in enumerate(states):
            env.reset(); env.load_state(st)
            x0 = env._var(env.reward_var) if use_var else 0.0
            acc = 0.0
            for t in range(chunks):
                f = env.frame()
                ch = pol._sample_chunk(f, "" if null else plan_text, w, plan_frames=[f], null=null,
                                       noise_seed=seed * 100003 + si * 101 + t)
                out = env.step(ch[:A])
                if not use_var and isinstance(out, tuple) and len(out) >= 2:
                    acc += float(out[1])
            deltas.append(float(env._var(env.reward_var) - x0) if use_var else acc)
        per_seed.append(float(np.mean(deltas)))
    return float(np.mean(per_seed)), per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--w", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = GAME_META[args.game]; envname = meta["env"] or ENVMAP.get(args.game)
    bat = dict(BATTERY[args.game])
    bat["dummy"] = DUMMY
    bat["generic"] = GENERIC
    others = [g for g in BATTERY if g != args.game]
    bat["wrong_game"] = BATTERY[others[0]]["correct"]

    out = args.out or os.path.join(_R, "docs", "graded", f"{args.game}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = {"game": args.game, "w": args.w, "chunks": args.chunks, "seeds": args.seeds,
           "envname": envname, "conditions": {}}

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.w)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("graded", plan="", objective="progress", cfg_scale=args.w))

    states = [gzip.decompress(open(p, "rb").read())
              for p in sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"],
                                                      "*", "initial.state")))][: args.max_states]
    rec["n_states"] = len(states)
    env = make_env(envname)
    print(f"\n===== GRADED-PLAN LADDER ({args.game}, {len(states)} states, {args.seeds} seeds, "
          f"{args.chunks} chunks, w={args.w}) =====", flush=True)
    try:
        for name in ORDER:
            is_null = (name == "null")
            mean, per_seed = run_cond(pol, env, states, "" if is_null else bat[name], is_null,
                                      args.w, args.chunks, args.A, args.seeds)
            rec["conditions"][name] = {"mean": mean, "per_seed": per_seed,
                                       "text": ("" if is_null else bat[name])}
            json.dump(rec, open(out, "w"), indent=2)        # persist after EACH condition (crash-safe)
            print(f"  {name:10s} = {mean:+8.2f}   (seeds {[round(x,1) for x in per_seed]})", flush=True)
    finally:
        env.close()

    c = rec["conditions"]
    def g(n): return c[n]["mean"] if n in c else float("nan")
    print("\n  --- ladder summary ---", flush=True)
    print(f"  correct-null       = {g('correct')-g('null'):+.2f}   (activation+semantics)")
    print(f"  correct-dummy      = {g('correct')-g('dummy'):+.2f}   (SEMANTICS over pure activation)")
    print(f"  correct-wrong_game = {g('correct')-g('wrong_game'):+.2f}   (genre semantics)")
    print(f"  correct-bad        = {g('correct')-g('bad'):+.2f}   (direction semantics)")
    print(f"  hand_good-correct  = {g('hand_good')-g('correct'):+.2f}   (headroom above concise)")
    graded = (g('correct') - g('dummy') > 0.15 * max(1.0, abs(g('correct')))) and (g('correct') > g('bad'))
    print(f"  VERDICT: {'GRADED (planner optimization worth it)' if graded else 'mostly ACTIVATION (invest in actor/bridge)'}", flush=True)
    json.dump(rec, open(out, "w"), indent=2)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
