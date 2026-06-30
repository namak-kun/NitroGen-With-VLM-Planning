"""collect_rl_trajectories.py -- roll out btn_s600 on a GROUND-TRUTH-reward env (TheXTech / Sonic) and
save per-chunk trajectory data. Shared data engine for (a) JUDGE CALIBRATION (before/after frame + GT
progress label) and (b) ACTOR-ADAPTATION reward-weighted BC (frame + plan + action_chunk + reward).

Diversity: multiple seeds + optional plan temperature so the data isn't a single deterministic path
(the rubber-duck: need exploration + train/val splits, not one trajectory).

Saves an .npz per episode under <out>/ and a manifest.jsonl with per-chunk records:
  {ep, t, plan, reward, gt_progress(sign), state_before, state_after, before_png, after_png, act_idx}
plus actions.npy (N,H,25) and frames as PNGs (downscaled). before/after frames are the CROP-free game frames.

Run (one GPU):
  CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc \
    QWEN=Qwen/Qwen3.5-2B .venv/bin/python planner_poc/collect_rl_trajectories.py \
      --env thextech --episodes 6 --chunks 16 --A 2 --out docs/rl_data/thextech
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from play_annotated_horizon import _first_sentence

SYS = {
    "thextech": "You are the planner for a Mario-style platformer. Goal: advance RIGHT, jump platforms, "
                "avoid hazards. Output ONE short imperative plan (max 10 words).",
    "sonic": "You are the planner for Sonic, a fast 2D platformer. Goal: move RIGHT, jump gaps/enemies. "
             "Output ONE short imperative plan (max 10 words).",
}
INSTR = ("In one sentence say what to do next using concrete directions (left,right,up,down,jump).")


def make_env(name):
    if name == "sonic":
        from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
        return RetroRLEnv()
    if name == "smw":
        from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
        return RetroRLEnv(rom_path="Game data/Super Mario World.sfc",
                          game="SuperMarioWorld-Snes-v0", system="Snes", reward_var="screen_x")
    if name in ("minish", "minish_cap"):
        from nitrogen.eval.envs.gba_env import GbaRLEnv
        return GbaRLEnv(game="minish_cap")
    if name.startswith("gba_"):
        from nitrogen.eval.envs.gba_env import make_gba_env
        return make_gba_env(name)
    from nitrogen.eval.envs.proc_rl_env import ProcRLEnv
    return ProcRLEnv(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="thextech")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--plan-temp", type=float, default=0.0, help=">0 -> sample plans (diversity)")
    ap.add_argument("--res", type=int, default=256, help="saved frame max side")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    out = args.out or f"docs/rl_data/{args.env}"
    fdir = os.path.join(out, "frames"); os.makedirs(fdir, exist_ok=True)
    syskey = "sonic" if args.env == "sonic" else "thextech"

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("collect", plan="", objective="make progress", cfg_scale=args.cfg))

    def save_png(frame, name):
        im = Image.fromarray(np.asarray(frame).astype(np.uint8))
        s = args.res / max(im.size)
        if s < 1:
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        p = os.path.join(fdir, name); im.save(p)
        return os.path.relpath(p, out)

    actions = []
    records = []
    print(f"collecting {args.episodes} eps x {args.chunks} chunks on {args.env} (A={args.A})", flush=True)
    env = make_env(args.env)             # ONE env reused across episodes (emulator-segfault fix)
    try:
        for ep in range(args.episodes):
            torch.manual_seed(ep); np.random.seed(ep)
            cur = env.reset()
            hist = [cur]
            plan = "move right"
            ep_reward = 0.0
            for t in range(args.chunks):
                before = cur
                if t % args.A == 0:
                    plan = _first_sentence(pol.pl.generate_plan(
                        hist[-4:], pol.device, instruction=INSTR, system=SYS[syskey],
                        max_new_tokens=32) or "move right")
                chunk = pol._sample_chunk(cur, plan, args.cfg, plan_frames=[cur], null=False)  # (H,25)
                obs, reward, done, info = env.step(chunk[:args.A])
                ep_reward += reward
                bname = save_png(before, f"ep{ep:02d}_t{t:02d}_b.png")
                aname = save_png(obs, f"ep{ep:02d}_t{t:02d}_a.png")
                records.append({
                    "ep": ep, "t": t, "plan": plan, "reward": round(float(reward), 3),
                    "gt_progress": int(np.sign(reward)) if abs(reward) > 1e-3 else 0,
                    "state_after": info.get("state", {k: info.get(k) for k in ("screen_x", "score", "lives")}),
                    "before_png": bname, "after_png": aname, "act_idx": len(actions), "done": bool(done),
                })
                actions.append(np.asarray(chunk[:args.A], dtype=np.float32))
                cur = obs; hist.append(cur)
                if done:
                    break
            print(f"  ep{ep}: reward={ep_reward:+.2f} chunks={t+1}", flush=True)
    finally:
        env.close()

    # pad action chunks to uniform (A,25) stack
    A = args.A
    acts = np.zeros((len(actions), A, 25), np.float32)
    for i, a in enumerate(actions):
        acts[i, :a.shape[0]] = a
    np.save(os.path.join(out, "actions.npy"), acts)
    with open(os.path.join(out, "manifest.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    rewards = [r["reward"] for r in records]
    print(f"\nwrote {out}: {len(records)} chunks, reward mean={np.mean(rewards):+.3f} "
          f"max={np.max(rewards):+.3f} min={np.min(rewards):+.3f}")
    pos = sum(1 for r in records if r["gt_progress"] > 0)
    print(f"  gt_progress: +{pos} / 0:{sum(1 for r in records if r['gt_progress']==0)} / "
          f"-{sum(1 for r in records if r['gt_progress']<0)}  (for judge calibration labels)")


if __name__ == "__main__":
    main()
