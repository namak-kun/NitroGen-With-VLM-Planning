"""demo_video.py -- record ~30s videos from the DEMO start states, comparing the BASE DiT (null/no-plan)
vs the PLANNER-conditioned policy. Qualitative diagnostic: watch what planning actually adds (or doesn't).

From each demo initial.state we roll the policy in receding-horizon closed loop, capturing every emulator
frame, and write an mp4 per (state, mode). The planner mode regenerates the plan every A=2 chunks via the
frozen VLM; base mode runs null (plan-dropped = base DiT exact). A label + live plan + reward var is drawn
on each frame. mp4s -> docs/demo_videos/<game>/<state>__{base,planner}.mp4 (user scp's them).

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/demo_video.py --game smw --seconds 30 --states 2
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import subprocess
import sys

import numpy as np

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from play_annotated_horizon import _first_sentence
from rwbc_actor_adapt import make_env, SYS, INSTR, _syskey
from demo_train_stack import GAME_META, EVAL_PLAN
from game_planner import ClosedLoopPlanner, applied_action_desc
from PIL import Image, ImageDraw, ImageFont


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def overlay(frame, mode, plan, rv, scale=3, learn_tag=""):
    im = Image.fromarray(frame).convert("RGB")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im); f = _font(16); fs = _font(13)
    h = 64 if mode == "learn" else 46
    dr.rectangle([0, 0, im.width, h], fill=(0, 0, 0))
    col = {"learn": (150, 255, 150), "plan": (120, 220, 255), "base": (255, 180, 120)}[mode]
    dr.text((6, 3), f"{mode.upper()}", fill=col, font=f)
    dr.text((110, 5), f"{rv}", fill=(200, 200, 200), font=fs)
    dr.text((6, 26), (plan or "(null / base DiT)")[:64], fill=(255, 255, 255), font=fs)
    if mode == "learn" and learn_tag:
        dr.text((6, 46), "learned: " + learn_tag, fill=(150, 255, 150), font=fs)
    return np.asarray(im)


def ffmpeg_writer(path, w, h, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(int(fps)),
           "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def record(pol, env, state, mode, seconds, A, cfg, sk, game, exec_rows=6, scale=3):
    """Roll the policy from `state` for ~seconds, capture every emu frame, return (frames, cap_fps).
    mode: 'base' (null DiT), 'plan' (per-game planner, no memory), 'learn' (planner + carried learnings)."""
    env.reset(); env.load_state(state)
    fps = getattr(env, "fps", 60.0); fpr = env.frames_per_row
    cap_fps = fps / fpr                    # we capture ONE frame per row (= fpr emu frames) -> real rate
    total_emu = int(seconds * fps)
    captured = []
    plan = ""; hist = [env.frame()]; chunk_i = 0; emu = 0
    clp = ClosedLoopPlanner(pol, game, mode=("learn" if mode == "learn" else "plan")) if mode != "base" else None
    while emu < total_emu:
        if mode != "base" and chunk_i % A == 0:
            plan = clp.plan(hist[-4:])
        cur = env.frame()
        ch = pol._sample_chunk(cur, "" if mode == "base" else plan, cfg,
                               plan_frames=[cur], null=(mode == "base"))
        rv = f"{env.reward_var}={env._var(env.reward_var):.0f}" if hasattr(env, "reward_var") else ""
        learn_tag = clp.learnings.splitlines()[0][:60] if (clp and clp.learnings) else ""
        for r in range(min(exec_rows, len(ch))):
            env._emu_step(env.action_row_to_buttons(ch[r]), fpr)
            captured.append(overlay(env.frame(), mode, plan, rv, scale, learn_tag))
            emu += fpr
            if emu >= total_emu:
                break
        hist.append(env.frame()); chunk_i += 1
        if mode == "learn":      # feed the executed play back so the next plan reviews the controller
            clp.observe(applied_action_desc(env, game, ch[0]), env.frame())
    return captured, cap_fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="smw", choices=["smw", "sonic", "minish", "fireemblem"])
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--delta", default=None, help="optional bootstrap delta to load (else base btn_s600)")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--states", type=int, default=0, help="how many demo start states (0 = ALL)")
    ap.add_argument("--modes", nargs="+", default=["base", "plan", "learn"],
                    help="base (null DiT) / plan (per-game planner) / learn (planner + carried learnings)")
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--exec-rows", type=int, default=6, help="rows executed per chunk (receding horizon)")
    ap.add_argument("--out-root", default="docs/demo_videos")
    args = ap.parse_args()
    device = "cuda"

    meta = GAME_META[args.game]
    envname = meta["env"] or {"fireemblem": "gba_fire_emblem_sacred_stones"}.get(args.game)
    sk = _syskey(envname)
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("demo_video", plan="", objective="progress", cfg_scale=args.cfg))
    if args.delta:
        d = __import__("torch").load(args.delta, map_location="cpu", weights_only=False)["trainable"]
        sd = pol.m.state_dict()
        for k, v in d.items():
            if k in sd: sd[k].copy_(v.to(sd[k].dtype))
        print(f"[video] loaded delta {os.path.basename(args.delta)}")
    pol.m.eval()

    states = [gzip.decompress(open(p, "rb").read())
              for p in sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"], "*", "initial.state")))]
    if args.states > 0:
        states = states[: args.states]
    env = make_env(envname)
    outdir = os.path.join(_R, args.out_root, args.game)
    try:
        for si, st in enumerate(states):
            for mode in args.modes:
                frames, fps = record(pol, env, st, mode, args.seconds, args.A, args.cfg, sk, args.game, args.exec_rows)
                h, w = frames[0].shape[:2]
                path = os.path.join(outdir, f"state{si}__{mode}.mp4")
                w_ = ffmpeg_writer(path, w, h, fps)
                for fr in frames:
                    w_.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
                w_.stdin.close(); w_.wait()
                print(f"[video] {args.game} state{si} {mode}: {len(frames)} frames -> {path}", flush=True)
    finally:
        env.close()
    print(f"\n[video] done -> {outdir}/  ({len(states)} states x {len(args.modes)} modes)")


if __name__ == "__main__":
    main()
