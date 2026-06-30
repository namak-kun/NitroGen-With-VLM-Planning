"""record_all_videos.py — render closed-loop videos for ALL games + ALL demo start states, base vs a delta,
under base(null) and live-VLM-plan modes. Game-general (make_env + generate_plan + BATTERY), so it works for
smw/mmx/smbas/sonic without per-game planner prompts. Overlays mode + live plan text + progress per frame.
mp4 -> <out_root>/<game>/state{N}__{mode}.mp4 (user scp's them).

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/record_all_videos.py --games smw,mmx \
      --modes base plan --seconds 20 --delta <delta.pt> --out-root docs/demo_videos_pooled
"""
from __future__ import annotations
import argparse, glob, gzip, os, subprocess, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env, SYS, INSTR
from plan_graded_test import BATTERY
from demo_bc import GAME_CFG
from play_annotated_horizon import _first_sentence
from eval_common import _lives
from PIL import Image, ImageDraw, ImageFont

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
SYS = dict(SYS)


def syskey(game):
    return "sonic" if game == "sonic" else "thextech"


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def overlay(frame, mode, plan, rv, scale=3):
    im = Image.fromarray(frame).convert("RGB").resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im); f = _font(16); fs = _font(13)
    dr.rectangle([0, 0, im.width, 46], fill=(0, 0, 0))
    col = (120, 220, 255) if mode == "plan" else (255, 180, 120)
    dr.text((6, 3), mode.upper(), fill=col, font=f)
    dr.text((120, 5), rv, fill=(200, 200, 200), font=fs)
    dr.text((6, 26), (plan or "(null / base DiT)")[:64], fill=(255, 255, 255), font=fs)
    return np.asarray(im)


def writer(path, w, h, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(int(fps)),
           "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@torch.no_grad()
def record(pol, env, st, mode, game, seconds, A, cfg, exec_rows=6, reset_drop=40.0):
    env.reset(); env.load_state(st)
    if env.frame().mean() < 1.0:
        env._emu_step([], 1)
    fps = getattr(env, "fps", 60.0); fpr = env.frames_per_row
    total = int(seconds * fps); sk = syskey(game)
    plan, null = "", (mode == "base"); hist = [env.frame()]; ci = 0; emu = 0; caps = []
    prev_rv = env._var(env.reward_var); lives_seen = _lives(env); died = False
    while emu < total and not died:
        if mode == "plan" and ci % A == 0:
            plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                   system=SYS[sk], max_new_tokens=32) or "move right")
        cur = env.frame()
        ch = np.asarray(pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=null), np.float32)
        rv = f"{env.reward_var}={env._var(env.reward_var):.0f}" if hasattr(env, "reward_var") else ""
        for r in range(min(exec_rows, len(ch))):
            env._emu_step(env.action_row_to_buttons(ch[r]), fpr)
            cur_rv = env._var(env.reward_var); lv = _lives(env)
            lost = lives_seen is not None and lv is not None and 0 < (lives_seen - lv) <= 3
            if lost or cur_rv < prev_rv - reset_drop:        # death / level reset -> end episode (no menu motion)
                died = True
                for _ in range(int(fps)):                    # ~1s "DIED" tail so the cutoff is visible
                    caps.append(overlay(env.frame(), mode, "DIED - episode end", rv))
                break
            caps.append(overlay(env.frame(), mode, plan, rv)); emu += fpr
            prev_rv = cur_rv
            if lv is not None and lives_seen is not None:
                lives_seen = max(lives_seen, lv)
            if emu >= total:
                break
        hist.append(env.frame()); ci += 1
    return caps, fps / fpr


def load_delta(pol, path):
    d = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
    sd = pol.m.state_dict()
    for k, v in d.items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="smw,mmx,smbas,sonic")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--delta", default=None)
    ap.add_argument("--modes", nargs="+", default=["base", "plan"])
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--states", type=int, default=0, help="0 = ALL")
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--out-root", default="docs/demo_videos")
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("vid", plan="", objective="progress", cfg_scale=args.cfg))
    if args.delta:
        load_delta(pol, args.delta); print(f"[video] loaded delta {os.path.basename(args.delta)}", flush=True)
    pol.m.eval()

    for game in args.games.split(","):
        glob_dir = GAME_CFG[game]["demo_glob"]
        sps = sorted(glob.glob(os.path.join(_R, glob_dir, "initial.state")) +
                     glob.glob(os.path.join(_R, glob_dir, "*", "initial.state")))
        states = []
        for p in sps:
            raw = open(p, "rb").read()
            states.append(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        if args.states > 0:
            states = states[: args.states]
        env = make_env(GAME_CFG[game]["env"]); outdir = os.path.join(_R, args.out_root, game)
        try:
            for si, st in enumerate(states):
                for mode in args.modes:
                    frames, fps = record(pol, env, st, mode, game, args.seconds, args.A, args.cfg)
                    h, w = frames[0].shape[:2]
                    path = os.path.join(outdir, f"state{si}__{mode}.mp4")
                    wr = writer(path, w, h, fps)
                    for fr in frames:
                        wr.stdin.write(np.ascontiguousarray(fr, np.uint8).tobytes())
                    wr.stdin.close(); wr.wait()
                    print(f"[video] {game} state{si} {mode}: {len(frames)}f -> {path}", flush=True)
        finally:
            env.close()
    print("[video] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
