"""narration_residual2.py — full-action frame-counterfactual residual (fixes the weak v1 direction-only proxy).

Per frame, two cheap signals the war-room named, computed for SMW (plan-OOD) vs Sonic (actor-OOD):
  (1) SEMANTIC residual  rho_sem = mean_k |human_rate_k - ditnull_rate_k|, k in {LEFT,RIGHT,UP,DOWN,JUMP}
      = "does the base DiT (frame, null plan) reproduce the human's FULL action?" High => frame-underdetermined.
  (2) PLAN AUTHORITY      auth     = ||a_correct - a_null|| over {dir(jlx), jump} = "does the CORRECT plan move
      the action here?" (action-space proxy for ||v_c-v_u||). High => plan has a say.

Goal: find whether ANY cheap frame-level proxy separates the plan-OOD game (SMW) from the actor-OOD game
(Sonic). If neither does, the residual-MASK auto-allocation idea is weak and the recipe must rely on the
REWARD-level Δ_plan signal instead. Stream-level (timing-noise-immune), no narration needed.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from plan_graded_test import BATTERY
from nitrogen.eval.envs.retro_rl_env import JLX

GAME_DIR = {"smw": "SuperMarioWorld-Snes", "sonic": "SonicTheHedgehog2-Genesis"}
KEYS = ["LEFT", "RIGHT", "UP", "DOWN", "JUMP"]


def _demo_for(game):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs/demos/demos", GAME_DIR[game])
    cands = [d for d in sorted(glob.glob(os.path.join(root, "*"))) if os.path.exists(os.path.join(d, "demo.npz"))]
    return max(cands, key=lambda d: np.load(os.path.join(d, "demo.npz"))["actions"].shape[0])


def _human_vec(actions, bidx, f, n):
    """Semantic rate vector over demo frames [f,f+n): {LEFT,RIGHT,UP,DOWN,JUMP} each in [0,1]."""
    sl = actions[f:f + n].astype(np.float32)
    if len(sl) == 0:
        return np.zeros(len(KEYS))
    jump_keys = [k for k in ("A", "B", "C") if k in bidx]
    v = {
        "LEFT": sl[:, bidx["LEFT"]].mean() if "LEFT" in bidx else 0.0,
        "RIGHT": sl[:, bidx["RIGHT"]].mean() if "RIGHT" in bidx else 0.0,
        "UP": sl[:, bidx["UP"]].mean() if "UP" in bidx else 0.0,
        "DOWN": sl[:, bidx["DOWN"]].mean() if "DOWN" in bidx else 0.0,
        "JUMP": float(np.clip(sum(sl[:, bidx[k]] for k in jump_keys), 0, 1).mean()) if jump_keys else 0.0,
    }
    return np.array([v[k] for k in KEYS])


def _chunk_vec(env, chunk):
    """Semantic rate vector over a DiT chunk (H,25) via action_row_to_buttons; JUMP=jump_button present."""
    cnt = {k: 0.0 for k in KEYS}
    for r in chunk:
        bs = env.action_row_to_buttons(r)
        for name in ("LEFT", "RIGHT", "UP", "DOWN"):
            if name in bs:
                cnt[name] += 1.0
        if env.jump_button in bs:
            cnt["JUMP"] += 1.0
    n = max(len(chunk), 1)
    return np.array([cnt[k] / n for k in KEYS])


def _dirjump(chunk):
    return float(np.clip(np.mean((chunk[:, JLX] - 0.5) * 2.0), -1, 1))


def measure(pol, game, cfg, H, fpr, stride):
    d = _demo_for(game)
    z = np.load(os.path.join(d, "demo.npz"))
    obs, acts = z["observations"], z["actions"]
    meta = json.load(open(os.path.join(d, "meta.json")))
    bidx = {b: i for i, b in enumerate(meta["buttons"])}
    env = make_env(game)
    correct = BATTERY[game]["correct"]
    win = H * fpr
    sem, auth = [], []
    for f in range(0, len(acts) - win, stride):
        frame = obs[f]
        a_null = np.asarray(pol._sample_chunk(frame, "", cfg, plan_frames=[frame], null=True), np.float32)
        a_corr = np.asarray(pol._sample_chunk(frame, correct, cfg, plan_frames=[frame], null=False), np.float32)
        hv = _human_vec(acts, bidx, f, win)
        dv = _chunk_vec(env, a_null)
        sem.append(float(np.mean(np.abs(hv - dv))))                              # full-action residual
        auth.append(float(abs(_dirjump(a_corr) - _dirjump(a_null))))             # plan authority (dir)
    env.close()
    sem, auth = np.array(sem), np.array(auth)
    return {"game": game, "demo": os.path.basename(d), "n": len(sem),
            "sem_mean": round(float(sem.mean()), 4), "sem_p75": round(float(np.quantile(sem, .75)), 4),
            "auth_mean": round(float(auth.mean()), 4), "auth_p75": round(float(np.quantile(auth, .75)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--H", type=int, default=18)
    ap.add_argument("--fpr", type=int, default=4)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("res2", plan="", objective="progress", cfg_scale=args.cfg))

    rep = {}
    for g in ("smw", "sonic"):
        s = measure(pol, g, args.cfg, args.H, args.fpr, args.stride)
        rep[g] = s
        print(f"  {g:6s}: sem_mean={s['sem_mean']:.4f} sem_p75={s['sem_p75']:.4f} | "
              f"auth_mean={s['auth_mean']:.4f} auth_p75={s['auth_p75']:.4f} (n={s['n']})", flush=True)
    if "smw" in rep and "sonic" in rep:
        for sig in ("sem_mean", "auth_mean"):
            r = rep["smw"][sig] / max(rep["sonic"][sig], 1e-6)
            verd = "SEPARATES (SMW>=1.3x)" if r >= 1.3 else ("REFUTE (<=1.05x)" if r <= 1.05 else "WEAK")
            print(f"  >>> {sig}: SMW/Sonic = {r:.2f} => {verd}")
        print("  (anchor: SMW Δ_plan=+0.20 plan-OOD ; Sonic Δ_plan=-0.18 actor-OOD)")
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
