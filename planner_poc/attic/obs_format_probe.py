"""obs_format_probe.py -- compare PLANNER OBSERVATION FORMATS in THINK MODE on a real demo rollout.

The user wants to know, with RL's think-mode planner: (a) does feeding the recent play as a VIDEO of the
two action chunks work, vs interleaving actions, and (b) what do the prompt outputs look like. We roll a
demo forward A=2 NitroGen chunks with the BASE DiT (recording, per sub-chunk, the EXECUTED action summary +
the resulting frame -- a real cause/effect trace), then generate a think-mode plan under each format:

  frames4          : the last 4 raw frames, no action text          (what demo_video.py currently uses)
  interleave_text  : f0 then (subchunk action text, resulting frame) ...   (rl_planner_prompt.py format)
  video_then_acts  : f0 + ONE video of all sub-frames, THEN the per-chunk action summaries as text
  video_interleave : f0 + per-chunk (short VIDEO clip of that chunk, then that chunk's action text)  <- direct way

Qwen3.5 is natively multimodal (Qwen3VLProcessor): it accepts interleaved {image|video|text} blocks, so
the "direct way" (multiple short video clips interleaved with action text) tokenizes natively -- verified.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/obs_format_probe.py --game smw --states 1
"""
from __future__ import annotations
import argparse, glob, gzip, os, sys
import numpy as np
from PIL import Image

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from nitrogen.training.actions import summarize_chunk
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from demo_train_stack import GAME_META
from game_planner import GAMES

ENVMAP = {"fireemblem": "gba_fire_emblem_sacred_stones"}
A_CHUNKS = 2      # planner spans A NitroGen chunks
S_SUB = 3         # sub-chunks (sampled frames) per chunk
H_CHUNK = 18      # NitroGen steps per chunk
SUB = H_CHUNK // S_SUB

SYS_TRACE = ("You are the high-level planner (System 2) for an agent playing {title}. A fast low-level "
             "controller (System 1) turns your plan into inputs. You are invoked periodically and shown the "
             "RECENT PLAY so far {how}. Use this cause-and-effect trace (you CANNOT read the controller from "
             "a still frame alone) to judge whether the agent is progressing or stuck, then decide what to do "
             "next. {controls} {objective}")
SYS_FRAMES = ("You are the high-level planner (System 2) for an agent playing {title}. You are shown the most "
              "recent game frames (oldest first). {controls} {objective}")
OUT_INSTR = ("Output ONE short plan sentence for what to do NEXT, grounded ONLY in the listed controls "
             "(use concrete terms like LEFT/RIGHT/UP/DOWN/JUMP/ATTACK; never 'forward'/'explore'). Plan only.")


def _pil(x):
    return x if isinstance(x, Image.Image) else Image.fromarray(np.asarray(x)).convert("RGB")


def roll_trace(pol, env, state, cfg):
    """Base-DiT rollout over A chunks; return f0 and a list of (ci, si, action_summary, frame_after)."""
    env.reset(); env.load_state(state)
    f0 = env.frame()
    trace = []
    for ci in range(A_CHUNKS):
        f = env.frame()
        ch = pol._sample_chunk(f, "", cfg, plan_frames=[f], null=True,
                               noise_seed=1234 + ci * 7)  # (18,25)
        for si in range(S_SUB):
            lo, hi = si * SUB, (H_CHUNK if si == S_SUB - 1 else (si + 1) * SUB)
            env.step(ch[lo:hi])
            summ = summarize_chunk({"buttons": ch[lo:hi, 0:21], "j_left": ch[lo:hi, 21:23]}, n_seg=1)
            trace.append((ci, si, summ, env.frame()))
    return f0, trace


def build_obs(fmt, game, f0, trace, recent_frames):
    """Return (system_text, content, images, videos) for the chosen observation format."""
    g = GAMES[game]
    fill = dict(title=g["title"], controls=g["controls"], objective=g["objective"])
    content, images, videos = [], [], []

    def txt(t): content.append({"type": "text", "text": t})
    def img(im): content.append({"type": "image"}); images.append(_pil(im))
    def vid(frames): content.append({"type": "video"}); videos.append([_pil(x) for x in frames])

    if fmt == "frames4":
        sysmsg = SYS_FRAMES.format(**fill)
        txt("The most recent game frames (oldest first):")
        for fr in recent_frames:
            img(fr)
        txt(OUT_INSTR)
        return sysmsg, content, images, videos

    if fmt == "interleave_text":
        sysmsg = SYS_TRACE.format(how="as an interleaved trace: a frame, then the inputs taken over a short "
                                  "sub-interval, then the resulting frame, and so on", **fill)
        txt("Current situation (frame f0, before this interval's inputs):"); img(f0)
        for (ci, si, summ, fr) in trace:
            txt(f"Chunk {ci+1}, inputs s{ci+1}_{si+1}: {summ}")
            tag = " (chunk end)" if si == S_SUB - 1 else ""
            txt(f"Resulting frame f{ci+1}_{si+1}{tag}:"); img(fr)
        txt(OUT_INSTR)
        return sysmsg, content, images, videos

    if fmt == "video_then_acts":
        sysmsg = SYS_TRACE.format(how="as a short VIDEO of the recent play, followed by a list of the "
                                  "controller inputs taken during it", **fill)
        txt("Frame before this interval (f0):"); img(f0)
        txt("Video of the recent play (the two most recent action chunks, in order):")
        vid([f0] + [fr for (_c, _s, _su, fr) in trace])
        txt("The controller inputs taken during that video, in order:")
        for (ci, si, summ, _fr) in trace:
            txt(f"  chunk {ci+1} sub {si+1}: {summ}")
        txt(OUT_INSTR)
        return sysmsg, content, images, videos

    if fmt == "video_interleave":
        sysmsg = SYS_TRACE.format(how="as an interleaved trace: a short VIDEO clip of one action chunk, then "
                                  "the inputs the controller took during that clip, and so on", **fill)
        txt("Frame before this interval (f0):"); img(f0)
        for ci in range(A_CHUNKS):
            subs = [t for t in trace if t[0] == ci]
            clip_frames = ([f0] if ci == 0 else []) + [fr for (_c, _s, _su, fr) in subs]
            txt(f"Video of chunk {ci+1}:"); vid(clip_frames)
            ins = "; ".join(f"s{ci+1}_{si+1}: {su}" for (_c, si, su, _fr) in subs)
            txt(f"Inputs during chunk {ci+1}: {ins}")
        txt(OUT_INSTR)
        return sysmsg, content, images, videos

    raise ValueError(fmt)


def generate_think_mm(pl, system_text, content, images, videos, device,
                      think_budget=256, plan_budget=48):
    """Think-mode generate that carries BOTH image and video tensors through the force-close phase."""
    import torch
    msgs = [{"role": "system", "content": system_text}, {"role": "user", "content": content}]
    try:
        text = pl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=True)
    except TypeError:
        text = pl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pk = {"text": [text], "return_tensors": "pt"}
    if images: pk["images"] = images
    if videos: pk["videos"] = videos
    inp = pl.processor(**pk).to(device)
    inlen = inp["input_ids"].shape[1]
    mm_kw = {k: inp[k] for k in ("pixel_values", "image_grid_thw",
                                 "pixel_values_videos", "video_grid_thw") if k in inp}
    with torch.no_grad():
        out1 = pl.backbone.generate(**inp, max_new_tokens=think_budget, do_sample=False)
    gen1 = pl.processor.batch_decode(out1[:, inlen:], skip_special_tokens=False)[0]
    if "</think>" in gen1:
        think, _, after = gen1.partition("</think>")
        think = think.split("<think>")[-1].strip()
        if after.strip():
            plan = after.replace("</think>", "").strip().strip('"').split("\n")[0].strip()
            return think, plan, False
    else:
        think = gen1.split("<think>")[-1].strip()
    tok = getattr(pl.processor, "tokenizer", pl.processor)
    close_ids = tok("</think>\n\nPlan:", add_special_tokens=False,
                    return_tensors="pt")["input_ids"].to(device)
    full_ids = torch.cat([out1, close_ids], dim=1)
    p2 = dict(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
              max_new_tokens=plan_budget, do_sample=False, **mm_kw)
    if "mm_token_type_ids" in inp:
        pad = torch.zeros((1, full_ids.shape[1] - inp["mm_token_type_ids"].shape[1]),
                          dtype=inp["mm_token_type_ids"].dtype, device=device)
        p2["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], pad], dim=1)
    with torch.no_grad():
        out2 = pl.backbone.generate(**p2)
    plan = pl.processor.batch_decode(out2[:, full_ids.shape[1]:], skip_special_tokens=True)[0]
    plan = plan.strip().strip('"').strip()
    if plan.lower().startswith("plan:"):
        plan = plan[5:].strip()
    return think, plan.split("\n")[0].strip(), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--states", type=int, default=1)
    ap.add_argument("--formats", nargs="+",
                    default=["frames4", "interleave_text", "video_then_acts", "video_interleave"])
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--think-budget", type=int, default=256)
    args = ap.parse_args()
    device = "cuda"

    meta = GAME_META[args.game]; envname = meta["env"] or ENVMAP.get(args.game)
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("obs_probe", plan="", objective="progress", cfg_scale=args.cfg))
    pl = pol.pl; pl.load()
    import torch
    if next(pl.backbone.parameters()).device != torch.device(device):
        pl.backbone.to(device)

    states = [gzip.decompress(open(p, "rb").read())
              for p in sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"],
                                                      "*", "initial.state")))][: args.states]
    env = make_env(envname)
    try:
        for si, st in enumerate(states):
            f0, trace = roll_trace(pol, env, st, args.cfg)
            recent = [f0] + [fr for (_c, _s, _su, fr) in trace]
            recent4 = recent[-4:]
            print(f"\n################ {args.game.upper()} state{si} ################", flush=True)
            print("  EXECUTED sub-chunk action trace (base DiT):")
            for (ci, sj, summ, _fr) in trace:
                print(f"    chunk{ci+1} sub{sj+1}: {summ}")
            for fmt in args.formats:
                sysmsg, content, images, videos = build_obs(fmt, args.game, f0, trace, recent4)
                think, plan, forced = generate_think_mm(pl, sysmsg, content, images, videos, device,
                                                        think_budget=args.think_budget)
                nv = len(videos); ni = len(images)
                print(f"\n  ===== FORMAT: {fmt}  ({ni} imgs, {nv} videos, think_forced={forced}) =====")
                print(f"    THINK: {(think[:600] + '...') if len(think) > 600 else think}")
                print(f"    PLAN : {plan}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
