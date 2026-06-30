"""latent_plan_search.py -- the duck's actor-vs-bridge-vs-text DISENTANGLER. From a demo save-state, directly
optimize the CONTINUOUS plan tokens z (K x d, injected via eval_policy.sample_chunk_token, bypassing
text/Qwen entirely) to MAXIMIZE emulator return, via CEM. Compare:
  null         : base DiT (no plan)
  text_correct : the correct plan TEXT -> bridge -> tokens (the normal path)
  pt_inject    : inject the text plan's tokens directly (sanity; should ~= text_correct)
  best_z       : the CEM-optimized latent plan tokens

Decision (per the rubber-duck):
  best_z >> text_correct (and >> null)  => the K-token interface CAN steer the DiT far better than the text
        path delivers => the TEXT->TOKEN path (planner/bridge-input) is the bottleneck => planner
        optimization / latent-RL / distillation is worth it.
  best_z ~= text_correct >> null        => text already saturates the token interface; plan matters but is
        maxed -> focus elsewhere.
  best_z ~= null                        => the K-token interface cannot steer this actor => the ACTOR (DiT)
        is the wall => invest in DiT-LoRA RL.

All rollouts share a FIXED action-noise seed so only z differs (matched). Crash-safe incremental JSON.

Run:  RUN=... CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u planner_poc/latent_plan_search.py --game smw
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
def rollout_z(pol, env, states, z, w, H, A, seeds):
    """Mean return of latent plan-tokens z over states x seeds. Noise VARIES per chunk (else the rollout
    degenerates). seeds is a list of base seeds; per-chunk seed = base*1009 + t."""
    use_var = hasattr(env, "reward_var")
    vals = []
    for base in seeds:
        for st in states:
            env.reset(); env.load_state(st)
            x0 = env._var(env.reward_var) if use_var else 0.0
            acc = 0.0
            for t in range(H):
                f = env.frame()
                ch = pol.sample_chunk_token(f, z, w=w, seed=base * 1009 + t)
                out = env.step(ch[:A])
                if not use_var and isinstance(out, tuple) and len(out) >= 2:
                    acc += float(out[1])
            vals.append(float(env._var(env.reward_var) - x0) if use_var else acc)
    return float(np.mean(vals))


@torch.no_grad()
def rollout_text(pol, env, states, plan, null, w, H, A, seeds):
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
    ap.add_argument("--H", type=int, default=8)          # chunks per rollout
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--w", type=float, default=8.0)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--elites", type=int, default=4)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--sigma0", type=float, default=0.6) # init CEM std (plan-token per-elem std ~1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)       # # train seeds (val seeds disjoint)
    ap.add_argument("--max-states", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = GAME_META[args.game]; envname = meta["env"] or ENVMAP.get(args.game)
    correct = BATTERY[args.game]["correct"]
    out = args.out or os.path.join(_R, "docs", "graded", f"latent_{args.game}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.w)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("latent", plan="", objective="progress", cfg_scale=args.w))

    state_paths = sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"], "*", "initial.state")))
    states = [gzip.decompress(open(p, "rb").read()) for p in state_paths][: args.max_states]
    env = make_env(envname)
    # train seeds drive CEM selection; VAL seeds (disjoint) score the final best_z -> NO winner's curse.
    train_seeds = list(range(args.seeds))
    val_seeds = list(range(100, 100 + args.seeds))
    rec = {"game": args.game, "w": args.w, "H": args.H, "pop": args.pop, "iters": args.iters,
           "n_states": len(states), "train_seeds": train_seeds, "val_seeds": val_seeds,
           "envname": envname, "baselines": {}, "cem": []}
    try:
        # reference plan tokens from the correct text (on the first state's frame)
        env.reset(); env.load_state(states[0]); f0 = env.frame()
        d = pol._prep(f0, correct, plan_frames=[f0]); d["plan_dropped"] = torch.tensor([False], device=device)
        pt_correct, _ = pol.m.compute_plan_tokens(d)
        pt_correct = pt_correct.detach().float()            # (1,K,dd)
        K, dd = pt_correct.shape[1], pt_correct.shape[2]
        print(f"\n===== LATENT PLAN SEARCH ({args.game}, {len(states)} states, K={K} d={dd}, H={args.H}, "
              f"w={args.w}, pop={args.pop} iters={args.iters}, train={train_seeds} val={val_seeds}) =====",
              flush=True)

        # baselines on VAL seeds (held-out, fair comparison to best_z on val)
        null_v = rollout_text(pol, env, states, "", True, args.w, args.H, args.A, val_seeds)
        text_v = rollout_text(pol, env, states, correct, False, args.w, args.H, args.A, val_seeds)
        ptinj_v = rollout_z(pol, env, states, pt_correct, args.w, args.H, args.A, val_seeds)
        rec["baselines"] = {"null": null_v, "text_correct": text_v, "pt_inject": ptinj_v}
        json.dump(rec, open(out, "w"), indent=2)
        print(f"  [val] null={null_v:+.2f}  text_correct={text_v:+.2f}  pt_inject={ptinj_v:+.2f}", flush=True)

        # CEM over z around pt_correct, SELECTED on train seeds
        mean = pt_correct.clone().view(-1)
        std = torch.full_like(mean, args.sigma0)
        best_train = {"r": -1e9, "z": pt_correct.clone()}
        for it in range(args.iters):
            pops = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(args.pop, mean.numel(), device=device)
            scores = []
            for p in range(args.pop):
                z = pops[p].view(1, K, dd)
                r = rollout_z(pol, env, states, z, args.w, args.H, args.A, train_seeds)
                scores.append(r)
                if r > best_train["r"]:
                    best_train = {"r": r, "z": z.detach().clone()}
            scores = torch.tensor(scores)
            elite_idx = torch.topk(scores, args.elites).indices
            elites = pops[elite_idx]
            mean = elites.mean(0); std = elites.std(0).clamp_min(0.05)
            rec["cem"].append({"iter": it, "train_mean": float(scores.mean()),
                               "train_max": float(scores.max()), "best_train": best_train["r"]})
            json.dump(rec, open(out, "w"), indent=2)
            print(f"  CEM it{it}: train mean={float(scores.mean()):+.2f} max={float(scores.max()):+.2f} "
                  f"| best_train={best_train['r']:+.2f}", flush=True)
        # FINAL: score the train-selected best_z and the CEM mean on HELD-OUT val seeds
        best_z_val = rollout_z(pol, env, states, best_train["z"], args.w, args.H, args.A, val_seeds)
        mean_z_val = rollout_z(pol, env, states, mean.view(1, K, dd), args.w, args.H, args.A, val_seeds)
        best_z_val = max(best_z_val, mean_z_val)
        rec["best_z_val"] = best_z_val; rec["best_z_train"] = best_train["r"]
    finally:
        env.close()

    json.dump(rec, open(out, "w"), indent=2)
    b = rec["baselines"]
    print("\n  --- summary (all on HELD-OUT val seeds) ---", flush=True)
    print(f"  null={b['null']:+.2f}  text_correct={b['text_correct']:+.2f}  best_z(val)={best_z_val:+.2f}  "
          f"(best_z train={best_train['r']:+.2f})", flush=True)
    print(f"  best_z - text_correct = {best_z_val-b['text_correct']:+.2f}  (token-space headroom over text)")
    print(f"  best_z - null         = {best_z_val-b['null']:+.2f}  (steerability of the K-token interface)")
    if best_z_val - b['null'] < 0.15 * max(1.0, abs(b['null'])):
        verdict = "ACTOR is the wall (K-token interface barely steers, even optimized)"
    elif best_z_val - b['text_correct'] > 0.2 * max(1.0, abs(b['text_correct'])):
        verdict = "TEXT->TOKEN path is the bottleneck (token space has headroom text misses)"
    else:
        verdict = "TEXT ~saturates the token interface"
    print(f"  VERDICT: {verdict}", flush=True)
    json.dump(rec, open(out, "w"), indent=2)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
