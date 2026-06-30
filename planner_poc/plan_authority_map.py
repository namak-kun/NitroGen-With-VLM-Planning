"""plan_authority_map.py — the R4 decisive measurement.

Red-team claim (GPT-5.5 R4): the "near-zero plan gradient" (base DiT reproduces the action from the frame
alone) was measured ON near-expert/on-trajectory states. The DRIFT regime is OFF-trajectory, where the plan
MAY re-acquire authority. This script tests that directly.

Method: drive a rollout with the NULL/base actor (no plan) so it drifts via its OWN compounding errors
(actor-OOD by construction). At each visited state measure how much the CORRECT plan WOULD change the
emitted action vs the null chunk:
    plan_authority = || chunk_correct - chunk_null ||   (does the plan have a say here?)
    dir_separation = || chunk_correct - chunk_wrong ||  (does it distinguish good vs bad objective?)
Stratify by DRIFT DEPTH proxies: chunk index (later = more compounded), actor disagreement D_t (K-sample
spread = epistemic uncertainty / OOD), and progress-stall. If authority RISES with drift depth, the plan is
load-bearing off-manifold (red-team vindicated). If flat/falling, the consensus holds (plan can't rescue a
drifted actor).

Deploy-relevant CFG (w=8). Behaviorally-meaningful dims: sticks (JLX=21,JLY=22) + jump (18,5); also full-25.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from plan_graded_test import BATTERY

JLX, JLY = 21, 22
DIRJUMP = [JLX, JLY, 18, 5]   # directional + jump dims (the behaviorally meaningful subset)


def _chunk(pol, frame, plan, w, null=False, noise_sigma=0.0, seed=None):
    return np.asarray(pol._sample_chunk(frame, plan, w, plan_frames=[frame], null=null,
                                        noise_sigma=noise_sigma, noise_seed=seed), np.float32)  # (H,25)


def _l2(a, b, dims=None):
    if dims is not None:
        a, b = a[:, dims], b[:, dims]
    return float(np.linalg.norm(a - b, axis=-1).mean())   # mean over H of per-step L2


def disagreement(pol, frame, plan, w, null, K=4, sigma=0.08):
    cs = [_chunk(pol, frame, plan, w, null=null, noise_sigma=sigma, seed=1000 + i) for i in range(K)]
    ds = [np.linalg.norm((cs[i] - cs[j])[:, DIRJUMP], axis=-1).mean()
          for i in range(K) for j in range(i + 1, K)]
    return float(np.mean(ds))


def run_game(pol, game, w, n_chunks, A):
    env = make_env(game)
    correct = BATTERY[game]["correct"]; wrong = BATTERY[game]["bad"]
    cur = env.reset()
    rows = []
    for t in range(n_chunks):
        ch_null = _chunk(pol, cur, "", w, null=True)
        ch_corr = _chunk(pol, cur, correct, w, null=False)
        ch_wrng = _chunk(pol, cur, wrong, w, null=False)
        Dt = disagreement(pol, cur, "", w, null=True)            # actor's OWN uncertainty (null path)
        rows.append(dict(
            t=t,
            D_t=Dt,
            auth_full=_l2(ch_corr, ch_null),                     # plan authority (all dims)
            auth_dir=_l2(ch_corr, ch_null, DIRJUMP),             # plan authority (dir+jump)
            dirsep_dir=_l2(ch_corr, ch_wrng, DIRJUMP),           # correct-vs-wrong separation
        ))
        # DRIVE with the null/base actor so it drifts on its own (actor-OOD by construction)
        obs, reward, done, info = env.step(ch_null[:A])
        rows[-1]["reward"] = float(reward)
        cur = obs
        if done:
            cur = env.reset()
    try:
        env.close()
    except Exception:
        pass
    return rows


def stratify(rows, key, q=(0.33, 0.66)):
    vals = np.array([r[key] for r in rows])
    lo, hi = np.quantile(vals, q)
    buckets = {"low": [], "mid": [], "high": []}
    for r in rows:
        v = r[key]
        buckets["low" if v <= lo else ("high" if v > hi else "mid")].append(r)
    return buckets


def summarize(rows):
    out = {"n": len(rows)}
    arr = lambda k: np.array([r[k] for r in rows])
    # correlations of plan authority with drift-depth proxies
    for depth in ("t", "D_t"):
        for auth in ("auth_dir", "auth_full"):
            d, a = arr(depth), arr(auth)
            if d.std() > 1e-9 and a.std() > 1e-9:
                out[f"r({depth},{auth})"] = round(float(np.corrcoef(d, a)[0, 1]), 3)
    # stratified means by actor-disagreement (the agreed OOD detector)
    by_D = stratify(rows, "D_t")
    out["by_D_t"] = {b: {"n": len(rs),
                         "auth_dir": round(float(np.mean([r["auth_dir"] for r in rs])), 4),
                         "auth_full": round(float(np.mean([r["auth_full"] for r in rs])), 4),
                         "dirsep_dir": round(float(np.mean([r["dirsep_dir"] for r in rs])), 4)}
                     for b, rs in by_D.items() if rs}
    # stratified by chunk-index third (early=on-manifold-ish, late=compounded drift)
    by_t = stratify(rows, "t")
    out["by_t"] = {b: {"n": len(rs),
                       "auth_dir": round(float(np.mean([r["auth_dir"] for r in rs])), 4),
                       "mean_reward": round(float(np.mean([r["reward"] for r in rs])), 3)}
                   for b, rs in by_t.items() if rs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="smw,sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--n-chunks", type=int, default=40)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("authmap", plan="", objective="progress", cfg_scale=args.cfg))

    report = {}
    for game in args.games.split(","):
        print(f"\n===== plan-authority map: {game} (null-driven drift, n={args.n_chunks}, cfg={args.cfg}) =====", flush=True)
        rows = run_game(pol, game, args.cfg, args.n_chunks, args.A)
        s = summarize(rows)
        report[game] = {"summary": s, "rows": rows}
        print(json.dumps(s, indent=2), flush=True)
        ad = s.get("by_D_t", {})
        if "low" in ad and "high" in ad:
            lo, hi = ad["low"]["auth_dir"], ad["high"]["auth_dir"]
            verdict = ("PLAN AUTHORITY RISES with actor-OOD (red-team)" if hi > lo * 1.15 else
                       "PLAN AUTHORITY FLAT/FALLS with actor-OOD (consensus)" if hi < lo * 0.87 else
                       "PLAN AUTHORITY ~FLAT across actor-OOD (ambiguous)")
            print(f"  >>> {game}: auth_dir low-D_t={lo} high-D_t={hi}  => {verdict}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
