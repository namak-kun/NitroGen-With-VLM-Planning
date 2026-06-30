"""best_of_k.py -- exploit the GATE result (plan value is GRADED for side-scrollers): sample K plans from the
frozen planner (temperature), roll each from the demo save-states, and SELECT the best by return. Tests the
duck's stage-2: does best-of-K plan selection beat the greedy plan? And produces the data generator for
behavioral distillation (the winning plan's high-return rollouts).

Winner's-curse-proof: plans are SELECTED on TRAIN seeds, then the selected plan is re-scored on disjoint VAL
seeds; we report VAL returns. We also report the spread of the K plans' returns (if plan choice matters, the
spread is wide -> consistent with graded plan value).

Run:  RUN=... CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u planner_poc/best_of_k.py --game smw --K 6
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
from game_planner import GAMES, system_prompt, PLAN_ONLY_INSTR
from plan_graded_test import BATTERY

ENVMAP = {"fireemblem": "gba_fire_emblem_sacred_stones"}


@torch.no_grad()
def sample_plans(pol, game, frame, K, temperature=0.9):
    """Sample K candidate plans from the frozen planner on this frame (per-game correct prompt)."""
    from PIL import Image
    pl = pol.pl; pl.load()
    if next(pl.backbone.parameters()).device != torch.device("cuda"):
        pl.backbone.to("cuda")
    img = Image.fromarray(np.asarray(frame)).convert("RGB")
    sys_p = system_prompt(game, "plan")
    content = [{"type": "image"}, {"type": "text", "text": PLAN_ONLY_INSTR}]
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": content}]
    text = pl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = pl.processor(text=[text], images=[img], return_tensors="pt").to("cuda")
    plans = []
    for _ in range(K):
        out = pl.backbone.generate(**inp, max_new_tokens=48, do_sample=True, temperature=temperature, top_p=0.95)
        gen = pl.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        plans.append(gen.strip().strip('"').split("\n")[0].strip())
    return plans


@torch.no_grad()
def rollout(pol, env, states, plan, null, w, H, A, seeds):
    use_var = hasattr(env, "reward_var")
    vals = []
    for base in seeds:
        for st in states:
            env.reset(); env.load_state(st)
            x0 = env._var(env.reward_var) if use_var else 0.0
            acc = 0.0
            for t in range(H):
                f = env.frame()
                ch = pol._sample_chunk(f, "" if null else plan, w, plan_frames=[f], null=null,
                                       noise_seed=base * 1009 + t)
                out = env.step(ch[:A])
                if not use_var and isinstance(out, tuple) and len(out) >= 2:
                    acc += float(out[1])
            vals.append(float(env._var(env.reward_var) - x0) if use_var else acc)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--w", type=float, default=8.0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = GAME_META[args.game]; envname = meta["env"] or ENVMAP.get(args.game)
    correct = BATTERY[args.game]["correct"]
    out = args.out or os.path.join(_R, "docs", "graded", f"bestofk_{args.game}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.w)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("bestofk", plan="", objective="progress", cfg_scale=args.w))

    state_paths = sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"], "*", "initial.state")))
    states = [gzip.decompress(open(p, "rb").read()) for p in state_paths][: args.max_states]
    train_seeds = list(range(args.seeds)); val_seeds = list(range(100, 100 + args.seeds))

    env = make_env(envname)
    rec = {"game": args.game, "K": args.K, "H": args.H, "w": args.w, "n_states": len(states),
           "train_seeds": train_seeds, "val_seeds": val_seeds, "plans": []}
    try:
        env.reset(); env.load_state(states[0]); f0 = env.frame()
        plans = sample_plans(pol, args.game, f0, args.K, args.temperature)
        print(f"\n===== BEST-OF-K ({args.game}, K={args.K}, {len(states)} states, train={train_seeds} "
              f"val={val_seeds}) =====", flush=True)
        for i, p in enumerate(plans):
            print(f"  plan[{i}]: {p}", flush=True)

        null_v = rollout(pol, env, states, "", True, args.w, args.H, args.A, val_seeds)
        greedy = correct  # the hand-correct concise plan is our 'greedy/reference' plan
        greedy_v = rollout(pol, env, states, greedy, False, args.w, args.H, args.A, val_seeds)

        # score each sampled plan on TRAIN, then re-score the winner on VAL (no winner's curse)
        train_scores = []
        for i, p in enumerate(plans):
            r = rollout(pol, env, states, p, False, args.w, args.H, args.A, train_seeds)
            train_scores.append(r)
            rec["plans"].append({"plan": p, "train_score": r})
            json.dump(rec, open(out, "w"), indent=2)
            print(f"  [train] plan[{i}] = {r:+.2f}", flush=True)
        best_i = int(np.argmax(train_scores))
        best_plan = plans[best_i]
        best_v = rollout(pol, env, states, best_plan, False, args.w, args.H, args.A, val_seeds)
        # also val-score every plan to estimate the achievable ceiling + selection regret
        val_scores = [rollout(pol, env, states, p, False, args.w, args.H, args.A, val_seeds) for p in plans]
        rec.update(dict(null_val=null_v, greedy_val=greedy_v, best_train_i=best_i, best_plan=best_plan,
                        best_of_k_val=best_v, val_scores=val_scores,
                        oracle_val=float(np.max(val_scores)), mean_val=float(np.mean(val_scores))))
    finally:
        env.close()

    json.dump(rec, open(out, "w"), indent=2)
    print("\n  --- summary (val seeds) ---", flush=True)
    print(f"  null            = {rec['null_val']:+.2f}")
    print(f"  greedy(correct) = {rec['greedy_val']:+.2f}")
    print(f"  mean-of-K       = {rec['mean_val']:+.2f}")
    print(f"  best-of-K (sel) = {rec['best_of_k_val']:+.2f}   (selected on train, scored on val)")
    print(f"  oracle-of-K     = {rec['oracle_val']:+.2f}   (max on val; upper bound)")
    print(f"  best_of_k - greedy = {rec['best_of_k_val']-rec['greedy_val']:+.2f}  (value of SELECTION)")
    print(f"  spread (max-min val) = {max(rec['val_scores'])-min(rec['val_scores']):+.2f}  (does plan choice matter?)")
    json.dump(rec, open(out, "w"), indent=2)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
