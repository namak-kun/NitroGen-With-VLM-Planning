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

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.eval.envs.supertuxkart import SuperTuxKartEnv, STEER_THRESH
from nitrogen.eval.core import JLX
from eval_policy import NitroGenPolicy

CKPT = os.environ.get("CKPT", "runs/stage2_2b_clean/plan_stage1_2000.pt")
NCHUNKS = int(os.environ.get("NCHUNKS", "8"))
CFG = float(os.environ.get("CFG", "8"))
TRACK = os.environ.get("TRACK", "hacienda")


def yaw_flow(a, b, scale=0.25):
    """Mean horizontal optical flow (Farneback) from a->b. Forward motion = divergent flow that
    CANCELS in the mean; a yaw/turn sweeps the whole scene one way so the mean is signed:
      world flows LEFT (mean u < 0) => camera yawed RIGHT (kart turned RIGHT);
      world flows RIGHT (mean u > 0) => kart turned LEFT.
    Robust to the large 0.6s-apart displacements that break phase correlation."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    h, w = ga.shape
    ga = cv2.resize(ga, (int(w * scale), int(h * scale)))
    gb = cv2.resize(gb, (int(w * scale), int(h * scale)))
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    return float(np.median(flow[..., 0]))   # median u: robust signed yaw proxy (px, downscaled)


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
            pans.append(yaw_flow(cur, obs.frame))
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
        print("  yawflow:", " ".join(f"{p:+.2f}" for p in r["pans"]))
        print(f"  MEAN yaw flow = {np.mean(r['pans']):+.3f}  "
              f"({'turned RIGHT' if np.mean(r['pans'])<-0.05 else 'turned LEFT' if np.mean(r['pans'])>0.05 else 'straight'})")
    # save a trajectory strip + raw frames (so measurement can be redone offline w/o re-running the model)
    os.makedirs("docs/env_candidates", exist_ok=True)
    np.savez_compressed("/tmp/stk_steer_frames.npz",
                        **{n: np.stack(r["frames"]) for n, r in out.items()})
    for name, r in out.items():
        strip = np.concatenate([cv2.resize(f, (160, 120)) for f in r["frames"][::2]], axis=1)
        cv2.imwrite(f"docs/env_candidates/stk_steer_{name}.png", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print("\n=== IN-ENV STEERING SEPARATION (mean yaw flow; <0 right, >0 left) ===")
    ml, mn, mr = (np.mean(out[k]["pans"]) for k in ("go_left", "null", "go_right"))
    print(f"  go_left  mean_yaw = {ml:+.3f}")
    print(f"  null     mean_yaw = {mn:+.3f}")
    print(f"  go_right mean_yaw = {mr:+.3f}")
    print(f"  separation (left - right) = {ml - mr:+.3f}  "
          f"(positive => go_left yaws more LEFT than go_right => plan steers the kart in-env)")
    print("  saved docs/env_candidates/stk_steer_{null,go_left,go_right}.png + /tmp/stk_steer_frames.npz")


if __name__ == "__main__":
    main()
