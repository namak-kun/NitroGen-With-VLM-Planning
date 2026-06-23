"""In-env STK steering: from the SAME race start, drive the clean plan-conditioned model under
null / 'go left' / 'go right' for N chunks and measure whether the kart actually steers differently.
Steering proxy = net horizontal scene pan (cv2 phase correlation between consecutive frames; the
world sweeps opposite to the turn). Also logs per-chunk stick_x + the keys the env actually pressed
(reveals the keyboard-threshold effect: a soft 'go left' may not cross STEER_THRESH).

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. .venv/bin/python planner_poc/stk_steering_demo.py
"""
import os
import sys

import cv2
import numpy as np

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.eval.envs.supertuxkart import SuperTuxKartEnv, STEER_THRESH
from nitrogen.eval.core import JLX
from eval_policy import NitroGenPolicy

CKPT = os.environ.get("CKPT", "runs/stage2_2b_clean/plan_stage1_2000.pt")
NCHUNKS = int(os.environ.get("NCHUNKS", "8"))
CFG = float(os.environ.get("CFG", "8"))
TRACK = os.environ.get("TRACK", "hacienda")


def pan(a, b):
    """Net horizontal shift (px) of b relative to a via phase correlation; +ve = world moved right."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, _), _ = cv2.phaseCorrelate(ga * win, gb * win)
    return dx


def run(pol, plan, null):
    env = SuperTuxKartEnv(track=TRACK, ai=1, laps=1, boot_wait=20.0, freeze_during_inference=True)
    try:
        obs = env.reset()
        cur = obs.frame
        frames = [cur]
        sxs, keylog, pans = [], [], []
        for _ in range(NCHUNKS):
            ch = pol._sample_chunk(cur, plan, CFG, plan_frames=[cur], null=null)
            sx = float(ch[:, JLX].mean())
            keys = env.action_to_keys(ch)
            steerk = "Left" if "Left" in keys else ("Right" if "Right" in keys else "·")
            obs = env.step(ch)
            pans.append(pan(cur, obs.frame))
            cur = obs.frame; frames.append(cur)
            sxs.append(sx); keylog.append(steerk)
        return dict(stick=sxs, keys=keylog, pans=pans, net_pan=float(np.sum(pans)), frames=frames)
    finally:
        env.close()
        import torch; torch.cuda.empty_cache()


def main():
    print(f"ckpt={CKPT} track={TRACK} cfg={CFG} nchunks={NCHUNKS} steer_thresh=±{STEER_THRESH}")
    pol = NitroGenPolicy(CKPT, qwen="Qwen/Qwen3.5-2B", default_cfg=1.0)
    pol.mm_mode = True; pol.mm_text_only = True
    conds = [("null", "", True), ("go_left", "go left", False), ("go_right", "go right", False)]
    out = {}
    for name, plan, null in conds:
        r = run(pol, plan, null)
        out[name] = r
        print(f"\n=== {name} (plan={plan!r}) ===")
        print("  stick_x:", " ".join(f"{x:+.2f}" for x in r["stick"]))
        print("  keys   :", " ".join(f"{k:>4}" for k in r["keys"]))
        print("  pan/chk:", " ".join(f"{p:+.0f}" for p in r["pans"]))
        print(f"  NET horizontal scene pan = {r['net_pan']:+.0f} px  "
              f"(world pans {'LEFT->kart turned RIGHT' if r['net_pan']<-20 else 'RIGHT->kart turned LEFT' if r['net_pan']>20 else 'straight'})")
    # save a trajectory strip for each condition
    os.makedirs("docs/env_candidates", exist_ok=True)
    for name, r in out.items():
        strip = np.concatenate([cv2.resize(f, (160, 120)) for f in r["frames"][::2]], axis=1)
        cv2.imwrite(f"docs/env_candidates/stk_steer_{name}.png", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print("\n=== STEERING SEPARATION (net pan) ===")
    print(f"  go_left net_pan  = {out['go_left']['net_pan']:+.0f}")
    print(f"  null    net_pan  = {out['null']['net_pan']:+.0f}")
    print(f"  go_right net_pan = {out['go_right']['net_pan']:+.0f}")
    print("  saved docs/env_candidates/stk_steer_{null,go_left,go_right}.png")


if __name__ == "__main__":
    main()
