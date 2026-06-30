"""narration_residual.py — the R5 frame-counterfactual experiment on the owner's REAL narration.

Tests the war-room consensus: a narrated action is System-1 reflex (type-B) iff the FROZEN base DiT already
reproduces it FROM THE FRAME ALONE; System-2 (type-A) is the residual the frame UNDERdetermines. We measure
this on ACTIONS (Nisbett-Wilson: don't trust the words), so it's robust to the owner's post-hoc, temporally
noisy narration.

For a frame f we compare, in a continuous DIRECTION (+jump) space:
  DiT_dir  = mean over the base-DiT null chunk of the left-stick X mapped to [-1(left),+1(right)]
  human_dir= mean over demo frames [f, f+H*fpr) of (RIGHT_press - LEFT_press)  in [-1,+1]
  rho_dir  = |DiT_dir - human_dir| / 2            (0 = base DiT matches the human => frame-determined/type-B;
                                                   high => frame-underdetermined => type-A)
  rho      = 0.7*rho_dir + 0.3*|DiT_jumprate - human_jumprate|

Outputs:
  (1) per-narration-line rho ranking for SMW (validates the labeler against the owner's actual words);
  (2) stream-level f_A = mean(rho_dir > tau) per game (SMW vs Sonic) — TIMING-NOISE-IMMUNE — to test the
      unifying hypothesis that type-A fraction predicts plan benefit (SMW plan-OOD > Sonic actor-OOD).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from nitrogen.eval.envs.retro_rl_env import JLX

GAME_DIR = {"smw": "SuperMarioWorld-Snes", "sonic": "SonicTheHedgehog2-Genesis"}


def _demo_for(game):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs/demos/demos", GAME_DIR[game])
    cands = [d for d in sorted(glob.glob(os.path.join(root, "*"))) if os.path.exists(os.path.join(d, "demo.npz"))]
    return max(cands, key=lambda d: np.load(os.path.join(d, "demo.npz"))["actions"].shape[0])  # longest


def _human_dir_jump(actions, bidx, f, n):
    """Continuous human direction in [-1,1] and jump-rate over demo frames [f,f+n)."""
    sl = actions[f:f + n]
    if len(sl) == 0:
        return 0.0, 0.0
    R = sl[:, bidx["RIGHT"]].mean() if "RIGHT" in bidx else 0.0
    L = sl[:, bidx["LEFT"]].mean() if "LEFT" in bidx else 0.0
    jump_keys = [k for k in ("A", "B", "C") if k in bidx]
    jr = float(np.clip(sum(sl[:, bidx[k]] for k in jump_keys), 0, 1).mean()) if jump_keys else 0.0
    return float(R - L), jr


def _dit_dir_jump(pol, env, frame, cfg):
    """Continuous base-DiT (null plan) direction in [-1,1] and jump-rate over its chunk."""
    chunk = np.asarray(pol._sample_chunk(frame, "", cfg, plan_frames=[frame], null=True), np.float32)  # (H,25)
    dirv = float(np.mean((chunk[:, JLX] - 0.5) * 2.0))                      # JLX>0.5 => right(+)
    jr = float(np.mean([1.0 if env.jump_button in env.action_row_to_buttons(r) else 0.0 for r in chunk]))
    return float(np.clip(dirv, -1, 1)), jr


def _rho(dit_dir, dit_jr, hum_dir, hum_jr):
    return 0.7 * abs(dit_dir - hum_dir) / 2.0 + 0.3 * abs(dit_jr - hum_jr)


def per_line(pol, game, cfg, H, fpr):
    d = _demo_for(game)
    z = np.load(os.path.join(d, "demo.npz"))
    obs, acts = z["observations"], z["actions"]
    meta = json.load(open(os.path.join(d, "meta.json")))
    bidx = {b: i for i, b in enumerate(meta["buttons"])}
    env = make_env(game)
    njson = os.path.join(d, "narration.json")
    if not os.path.exists(njson):
        print(f"  [skip per-line] no narration.json in {os.path.basename(d)} (run demo_narration.py ingest)")
        env.close(); return None
    entries = json.load(open(njson))["entries"]
    win = H * fpr
    rows = []
    for e in entries:
        f = int(np.clip(e["frame"], 0, len(obs) - 2))
        hd, hj = _human_dir_jump(acts, bidx, f, win)
        dd, dj = _dit_dir_jump(pol, env, obs[f], cfg)
        rows.append({"t": e["t"], "frame": f, "rho": round(_rho(dd, dj, hd, hj), 3),
                     "dit_dir": round(dd, 2), "hum_dir": round(hd, 2), "text": e["text"]})
    env.close()
    rows.sort(key=lambda r: -r["rho"])
    return rows


def stream_fa(pol, game, cfg, H, fpr, stride, tau):
    """Timing-noise-immune: sample frames across the demo, fraction where base DiT direction disagrees."""
    d = _demo_for(game)
    z = np.load(os.path.join(d, "demo.npz"))
    obs, acts = z["observations"], z["actions"]
    meta = json.load(open(os.path.join(d, "meta.json")))
    bidx = {b: i for i, b in enumerate(meta["buttons"])}
    env = make_env(game)
    win = H * fpr
    rhos, dis = [], []
    for f in range(0, len(acts) - win, stride):
        hd, hj = _human_dir_jump(acts, bidx, f, win)
        dd, dj = _dit_dir_jump(pol, env, obs[f], cfg)
        rd = abs(dd - hd) / 2.0
        rhos.append(rd); dis.append(1.0 if rd > tau else 0.0)
    env.close()
    return {"game": game, "demo": os.path.basename(d), "n": len(rhos),
            "f_A": round(float(np.mean(dis)), 3), "mean_rho_dir": round(float(np.mean(rhos)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--H", type=int, default=18)
    ap.add_argument("--fpr", type=int, default=4, help="demo frames per DiT row (env frames_per_row)")
    ap.add_argument("--stride", type=int, default=60, help="stream-fa frame stride")
    ap.add_argument("--tau", type=float, default=0.35, help="rho_dir threshold for type-A")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("authres", plan="", objective="progress", cfg_scale=args.cfg))

    report = {}
    print("\n===== PER-LINE residual on SMW narration (base DiT vs your real actions) =====", flush=True)
    rows = per_line(pol, "smw", args.cfg, args.H, args.fpr)
    if rows:
        report["smw_per_line"] = rows
        print("  --- TOP 12 (predicted TYPE-A: frame-underdetermined; base DiT disagrees with you) ---")
        for r in rows[:12]:
            print(f"   rho={r['rho']:.2f} dit_dir={r['dit_dir']:+.2f} you={r['hum_dir']:+.2f} | t={r['t']}s :: {r['text'][:64]}")
        print("  --- BOTTOM 12 (predicted TYPE-B: frame-determined reflex; base DiT matches you) ---")
        for r in rows[-12:]:
            print(f"   rho={r['rho']:.2f} dit_dir={r['dit_dir']:+.2f} you={r['hum_dir']:+.2f} | t={r['t']}s :: {r['text'][:64]}")

    print("\n===== STREAM-LEVEL f_A per game (timing-noise-immune unifying hypothesis) =====", flush=True)
    for g in ("smw", "sonic"):
        s = stream_fa(pol, g, args.cfg, args.H, args.fpr, args.stride, args.tau)
        report[f"{g}_stream"] = s
        print(f"  {g:6s}: f_A={s['f_A']:.3f}  mean_rho_dir={s['mean_rho_dir']:.3f}  (n={s['n']}, demo={s['demo']})")
    if "smw_stream" in report and "sonic_stream" in report:
        fa_s, fa_so = report["smw_stream"]["f_A"], report["sonic_stream"]["f_A"]
        ratio = fa_s / max(fa_so, 1e-6)
        verdict = ("CONFIRM (SMW f_A >= 1.3x Sonic => type-A fraction tracks plan benefit)" if ratio >= 1.3 else
                   "REFUTE (SMW f_A not > Sonic => hypothesis fails)" if ratio <= 1.05 else
                   "WEAK (1.05-1.3x)")
        print(f"\n  >>> SMW/Sonic f_A ratio = {ratio:.2f}  => {verdict}")
        print(f"      (anchor: SMW Δ_plan=+0.20 plan-OOD ; Sonic Δ_plan=-0.18 actor-OOD)")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
