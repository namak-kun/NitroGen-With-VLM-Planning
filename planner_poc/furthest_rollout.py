"""furthest_rollout.py — run the LONGEST closed-loop rollout of a model (base or a loaded delta) from a start
state, capturing every frame + an mp4, until death/level-reset or a generous time cap. Logs the RAM progress
trace (screen_x) and survival, but the MP4/frames are meant for INDEPENDENT VLM video-judging (RAM can be
sketchy; the video is ground truth a human/VLM can watch). Reuses the death-aware cutoff (eval_common).

Output: <out_dir>/<tag>__state<N>.mp4  +  <out_dir>/<tag>__state<N>.frames.npz (downsampled frames for the
judge) + a per-rollout json (RAM screen_x reached, survived frames, died, where).

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/furthest_rollout.py --game smw \
     --tag base --seconds 90 --state 0 --out docs/furthest/smw
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/furthest_rollout.py --game smw \
     --tag pooled --delta <pooled.pt> --mode plan --seconds 90 --state 0 --out docs/furthest/smw
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, subprocess, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env, SYS, INSTR
from demo_bc import GAME_CFG, demo_start_states
from play_annotated_horizon import _first_sentence
from eval_common import _lives, _step_done_info
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


def overlay(frame, tag, rv, scale=3, plan=None):
    im = Image.fromarray(frame).convert("RGB").resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im); f = _font(15)
    dr.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    dr.text((4, 3), f"{tag}  {rv}", fill=(120, 220, 255), font=f)
    if plan:
        pf = _font(13)
        dr.rectangle([0, im.height - 20, im.width, im.height], fill=(0, 0, 0))
        dr.text((4, im.height - 17), ("PLAN: " + plan)[:80], fill=(255, 220, 120), font=pf)
    return np.asarray(im)


def ffmpeg_writer(path, w, h, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(int(fps)),
           "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@torch.no_grad()
def run(pol, env, state, mode, game, seconds, A=2, cfg=8.0, exec_rows=6, reset_drop=40.0, judge_frames=24,
        log_plans=False):
    """Closed-loop rollout until death/level-reset or `seconds`. Returns dict with frames(for video), a small
    evenly-sampled set of RAW frames (for the VLM judge), RAM screen_x trace, survived, died."""
    env.reset(); env.load_state(state)
    if env.frame().mean() < 1.0:
        env._emu_step([], 1)
    fps = getattr(env, "fps", 60.0); fpr = env.frames_per_row
    total = int(seconds * fps); sk = syskey(game)
    plan, null = "", (mode == "base"); hist = [env.frame()]; ci = 0; emu = 0
    x0 = env._var(env.reward_var); best = 0.0; prev = x0; lives_seen = _lives(env)
    vid = []; raw = [env.frame().copy()]; xtrace = [0.0]; died = False
    plan_log = []  # [(t_sec, progress, plan_text)] -- what System-2 said, when (for the attribution diagnostic)
    while emu < total:
        if mode == "plan" and ci % A == 0:
            plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                   system=SYS[sk], max_new_tokens=32) or "move right")
            plan_log.append((round(emu / fps, 1), round(env._var(env.reward_var) - x0, 1), plan))
        cur = env.frame()
        ch = np.asarray(pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=null), np.float32)
        for r in range(min(exec_rows, len(ch))):
            done, info = _step_done_info(env, ch[r:r + 1]) if False else (False, {})
            env._emu_step(env.action_row_to_buttons(ch[r]), fpr)
            rv = env._var(env.reward_var); lv = _lives(env)
            lost = lives_seen is not None and lv is not None and 0 < (lives_seen - lv) <= 3
            if lost or rv < prev - reset_drop:
                died = True; break
            best = max(best, rv - x0); prev = rv
            if lv is not None and lives_seen is not None:
                lives_seen = max(lives_seen, lv)
            vid.append(overlay(env.frame(), mode.upper(), f"{env.reward_var}={rv:.0f}",
                               plan=plan if log_plans else None))
            raw.append(env.frame().copy()); xtrace.append(rv - x0)
            emu += fpr
            if emu >= total:
                break
        if died:
            for _ in range(int(fps)):
                vid.append(overlay(env.frame(), mode.upper() + " DIED", f"{env.reward_var}={env._var(env.reward_var):.0f}",
                                   plan=plan if log_plans else None))
            break
        hist.append(env.frame()); ci += 1
    # evenly sample judge frames
    idx = np.linspace(0, len(raw) - 1, min(judge_frames, len(raw))).astype(int)
    judge = [raw[i] for i in idx]
    return {"video": vid, "judge_frames": judge, "judge_t": [round(i / (fps / fpr), 1) for i in idx],
            "xtrace": xtrace, "screen_x_reached": float(best), "survived_rows": len(xtrace) - 1,
            "survived_sec": round((len(xtrace) - 1) * fpr / fps, 1), "died": died, "cap_fps": fps / fpr,
            "plan_log": plan_log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--delta", default=None)
    ap.add_argument("--mode", default=None, help="base|plan (default: base if no delta else plan)")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--state", type=int, default=0, help="which demo start state (0=first)")
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--out", default="docs/furthest/smw")
    ap.add_argument("--log-plans", action="store_true",
                    help="overlay the live System-2 plan on the video + write <tag>__state<N>.plans.json "
                         "[(t_sec, progress, plan_text)] for the plan-quality-vs-DiT-adherence diagnostic")
    args = ap.parse_args()
    mode = args.mode or ("base" if not args.delta else "plan")

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("furthest", plan="", objective="progress", cfg_scale=args.cfg))
    if args.delta:
        d = torch.load(args.delta, map_location="cpu", weights_only=False)["trainable"]
        sd = pol.m.state_dict()
        for k, v in d.items():
            if k in sd: sd[k].copy_(v.to(sd[k].dtype))
        print(f"[furthest] loaded delta {os.path.basename(args.delta)}", flush=True)
    pol.m.eval()

    states = demo_start_states(GAME_CFG[args.game]["demo_glob"])
    st = states[args.state]
    env = make_env(GAME_CFG[args.game]["env"])
    try:
        res = run(pol, env, st, mode, args.game, args.seconds, A=args.A, cfg=args.cfg, log_plans=args.log_plans)
    finally:
        env.close()

    os.makedirs(os.path.join(_R, args.out), exist_ok=True)
    base = os.path.join(_R, args.out, f"{args.tag}__state{args.state}")
    # mp4
    fr = res["video"]
    if fr:
        h, w = fr[0].shape[:2]
        wr = ffmpeg_writer(base + ".mp4", w, h, res["cap_fps"])
        for f in fr:
            wr.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
        wr.stdin.close(); wr.wait()
    # judge frames
    np.savez_compressed(base + ".frames.npz", frames=np.stack(res["judge_frames"]),
                        t=np.array(res["judge_t"], np.float32))
    meta = {k: res[k] for k in ("screen_x_reached", "survived_rows", "survived_sec", "died", "judge_t")}
    meta.update(tag=args.tag, mode=mode, state=args.state, game=args.game, seconds=args.seconds)
    json.dump(meta, open(base + ".json", "w"), indent=1)
    if args.log_plans and res.get("plan_log"):
        json.dump([{"t": t, "progress": p, "plan": txt} for t, p, txt in res["plan_log"]],
                  open(base + ".plans.json", "w"), indent=1)
        print(f"[furthest] logged {len(res['plan_log'])} plans -> {base}.plans.json", flush=True)
    print(f"[furthest] {args.tag} state{args.state}: screen_x +{res['screen_x_reached']:.0f}, "
          f"survived {res['survived_sec']}s, died={res['died']} -> {base}.mp4", flush=True)


if __name__ == "__main__":
    main()
