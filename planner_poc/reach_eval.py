"""reach_eval.py -- the REUSABLE generalist eval harness (built on the ordinal_sanity falsifier, which proved
the metric must be SURVIVAL-WEIGHTED: reach × survived, not peak-progress-before-death which rewards
suicide-sprint). Compares any set of policies (base + named deltas) across games and HORIZONS, from many
expert-mined start states, and rolls up to ONE generalist score.

METRIC per (game, start, horizon):
  P = running-max of camera screen_x (Sonic loops make player-x non-monotone), gated by ALIVE (lives).
  reach   = clip( (P_max_alive - sx0) / (P_expert - sx0), 0, 1 )    # expert-normalized progress GAINED
  surv    = survived the whole horizon budget (no death)
  HEADLINE score = reach * surv                                     # survival-weighted (anti suicide-sprint)
  diagnostics: reach, survived_frac, death_after_peak, raw P_max, time-to-best.
HORIZONS (budget in emu frames) test LOCAL competence (short) vs LONG-horizon reach (long) from each start.
ROLLUP: GeneralistScore = mean over games (EQUAL weight) of mean-over-(start,horizon,seed) headline; report
per-game table + worst-game. Baselines: EXPERT replay (sanity ~1.0), no-op (idle ~0), random.

Usage (compare base vs two deltas on sonic+smw):
  RUN=... CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/reach_eval.py \
     --games sonic smw --policies base rwbc:ckpts/rwbc_sonic_correct_scaled.pt --horizons 450 900 \
     --n-starts 8 --seeds 2
(`rwbc:<path>` loads that delta; per-game deltas can be given as rwbc:sonic=...,smw=... )
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
from ordinal_sanity import (idle_chunk, random_chunk, right_jump_chunk, alive, rollout,
                            expert_denom, mine_starts, PROGRESS, PROGRESS_NOTE)


def parse_policies(specs):
    """spec forms: 'base' | 'idle' | 'random' | 'right_jump' | 'NAME:path' (one delta for all games) |
    'NAME:game1=path1,game2=path2' (per-game delta)."""
    pols = []
    for s in specs:
        if ":" not in s:
            pols.append({"name": s, "kind": s, "delta": None}); continue
        name, rhs = s.split(":", 1)
        if "=" in rhs:
            per = {}
            for kv in rhs.split(","):
                g, p = kv.split("="); per[g] = p
            pols.append({"name": name, "kind": "dit", "delta": per})
        else:
            pols.append({"name": name, "kind": "dit", "delta": rhs})
    return pols


def delta_for(pol, game):
    d = pol["delta"]
    if d is None: return None
    return d.get(game) if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["sonic", "smw"])
    ap.add_argument("--policies", nargs="+", default=["idle", "random", "right_jump", "base"],
                    help="idle|random|right_jump|base|NAME:path|NAME:g1=p1,g2=p2")
    ap.add_argument("--horizons", type=int, nargs="+", default=[450, 900])
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join(_R, "docs/graded/reach_eval.json"))
    args = ap.parse_args()
    device = "cuda"

    pols = parse_policies(args.policies)
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("reach", plan="", objective="progress", cfg_scale=args.cfg))
    init = {n: p.detach().clone() for n, p in pol.m.named_parameters() if p.requires_grad or "lora_" in n}

    def restore_base():
        sd = dict(pol.m.named_parameters())
        for k, t in init.items():
            if k in sd: sd[k].data.copy_(t.to(device))

    def load_delta(path):
        d = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
        sd = dict(pol.m.named_parameters()); n = 0
        for k, t in d.items():
            if k in sd: sd[k].data.copy_(t.to(device)); n += 1
        return n

    HANDCODED = {"idle": idle_chunk, "random": random_chunk, "right_jump": right_jump_chunk}
    rec = {"games": args.games, "horizons": args.horizons, "policies": [p["name"] for p in pols],
           "n_starts": args.n_starts, "seeds": args.seeds, "per_game": {}}

    # per_game[game][policy_name][horizon] = {headline, reach, survived, death_after_peak}
    for game in args.games:
        meta = GAME_META[game]
        fixed_plan = BATTERY[game]["correct"]
        spec = PROGRESS.get(game, dict(prog="screen_x", life="lives"))
        prog_var, life_var = spec["prog"], spec["life"]
        env = make_env(game); fpr = env.frames_per_row
        cands = [d for d in sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"], "*")))
                 if os.path.exists(os.path.join(d, "demo.npz"))]
        # pick the LONGEST demo (most frames) -> best quality start-state source (avoids junk short demos)
        demo_dir = max(cands, key=lambda d: np.load(os.path.join(d, "demo.npz"))["actions"].shape[0])
        print(f"\n===== {game.upper()} (fpr={fpr}, prog={prog_var}, demo={os.path.basename(demo_dir)}) =====", flush=True)
        snaps, acts, bmap = mine_starts(env, demo_dir, args.n_starts, prog_var=prog_var)

        try:
            # NORMALIZATION (fixed): the EXPERT's short-horizon forward gain is a BROKEN denominator -- Sonic
            # (and any game with loops/backtracking) has near-zero or negative expert forward-gain at many
            # mid-level starts, so a base policy that legitimately advances scores >1 or garbage. Instead use
            # a ROBUST, ALWAYS-FORWARD reference: scripted RIGHT+jump as the CEILING and idle as the FLOOR.
            # skill = clip((gain - idle_gain) / (rightjump_gain - idle_gain), 0, 1). Expert gain is logged as
            # a REFERENCE line only (not the denominator). A start is usable if the ceiling clears the floor.
            ceil_gain = {H: {} for H in args.horizons}; idle_floor = {H: {} for H in args.horizons}
            expert_ref = {H: {} for H in args.horizons}
            for H in args.horizons:
                for i, (st, idx, sx0) in enumerate(snaps):
                    ri = rollout(env, st, (lambda t, rng: idle_chunk(t, rng)), H, fpr,
                                 seed=7 + i, prog_var=prog_var, life_var=life_var)
                    rj = rollout(env, st, (lambda t, rng: right_jump_chunk(t, rng)), H, fpr,
                                 seed=5 + i, prog_var=prog_var, life_var=life_var)
                    idle_floor[H][i] = ri["p_max_alive"] - sx0
                    ceil_gain[H][i] = rj["p_max_alive"] - sx0
                    expert_ref[H][i] = expert_denom(env, st, acts, idx, bmap, H, prog_var, life_var) - sx0
            # usable: scripted ceiling must beat the floor by a margin (else the segment can't discriminate).
            usable = sorted([i for i in range(len(snaps))
                             if all(ceil_gain[H][i] - idle_floor[H][i] >= 50 for H in args.horizons)])
            print(f"  mined {len(snaps)} starts; usable {len(usable)} (scripted-RIGHT ceiling clears floor); "
                  f"expert_ref/ceil @H{args.horizons[0]}: "
                  f"{[(round(expert_ref[args.horizons[0]][i]), round(ceil_gain[args.horizons[0]][i])) for i in usable]}",
                  flush=True)
            denom = ceil_gain  # downstream uses denom[H][i] as the ceiling
            rec["per_game"][game] = {}
            for P in pols:
                name = P["name"]
                if P["kind"] in HANDCODED:
                    step_fn = lambda t, rng, f=HANDCODED[P["kind"]]: f(t, rng)
                    label = name
                else:
                    restore_base()
                    if P["kind"] == "base":
                        plan, null = "", True; label = "base(null)"
                    else:
                        dp = delta_for(P, game)
                        if dp and os.path.exists(dp):
                            ns = load_delta(dp); plan, null = fixed_plan, False
                            label = f"{name}({ns}t)"
                        else:
                            plan, null = fixed_plan, False; label = f"{name}(NODELTA)"
                    def step_fn(t, rng, _plan=plan, _null=null):
                        f = env.frame()
                        return pol._sample_chunk(f, _plan, args.cfg, plan_frames=[f], null=_null,
                                                 noise_seed=rng.randint(1 << 30))
                rec["per_game"][game][name] = {}
                for H in args.horizons:
                    heads, reaches, survs, daps = [], [], [], []
                    for i in usable:
                        if i not in denom[H]:
                            continue
                        st, idx, sx0 = snaps[i]
                        floor = idle_floor[H].get(i, 0.0)
                        ceil = denom[H][i]          # already a GAIN (scripted-RIGHT ceiling), not absolute
                        for s in range(args.seeds):
                            r = rollout(env, st, step_fn, H, fpr, seed=1000 * i + s + hash(H) % 97,
                                        prog_var=prog_var, life_var=life_var)
                            gain = r["p_max_alive"] - sx0
                            # floor=idle, ceil=scripted-RIGHT: idle->0, hold-right->1; clip [0,1].
                            reach = float(np.clip((gain - floor) / max(1.0, ceil - floor), 0, 1))
                            surv = 1.0 if r["survived"] else 0.0
                            heads.append(reach * surv); reaches.append(reach); survs.append(surv)
                            daps.append(1.0 if r["death_after_peak"] else 0.0)
                    rec["per_game"][game][name][str(H)] = dict(
                        headline=float(np.mean(heads)) if heads else float("nan"),
                        reach=float(np.mean(reaches)) if reaches else float("nan"),
                        survived=float(np.mean(survs)) if survs else float("nan"),
                        death_after_peak=float(np.mean(daps)) if daps else float("nan"))
                    json.dump(rec, open(args.out, "w"), indent=2)
                hl = {H: rec["per_game"][game][name][str(H)]["headline"] for H in args.horizons}
                print(f"  {label:16s} headline/horizon " +
                      " ".join(f"{H}f={hl[H]:.3f}" for H in args.horizons), flush=True)
        finally:
            env.close()

    # rollup: GeneralistScore = equal-weight over games of mean-over-(policy? no: per policy)
    print("\n===== GENERALIST ROLLUP (headline = reach × survived, equal-weight games) =====", flush=True)
    rollup = {}
    for P in pols:
        name = P["name"]
        per_game_scores = []
        for game in args.games:
            vals = [rec["per_game"][game][name][str(H)]["headline"] for H in args.horizons
                    if not np.isnan(rec["per_game"][game][name][str(H)]["headline"])]
            if vals:
                per_game_scores.append((game, float(np.mean(vals))))
        gen = float(np.mean([s for _, s in per_game_scores])) if per_game_scores else float("nan")
        worst = min(per_game_scores, key=lambda x: x[1]) if per_game_scores else ("-", float("nan"))
        rollup[name] = {"generalist": gen, "per_game": dict(per_game_scores), "worst_game": worst[0],
                        "worst_score": worst[1]}
        pg = "  ".join(f"{g}={s:.3f}" for g, s in per_game_scores)
        print(f"  {name:12s} GENERALIST={gen:.3f}  (worst {worst[0]}={worst[1]:.3f}) | {pg}", flush=True)
    rec["rollup"] = rollup
    json.dump(rec, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
