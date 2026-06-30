"""demo_train_stack.py -- bootstrap the FULL stack (System-2 plan-head + System-1 DiT) on human gold
demos with strong-model-authored gold plans.

Per @namak-kun's framing: the planner (frozen VLM + plan-head) is the cross-game GENERALIZER; the DiT is a
per-game EXECUTOR (low capacity, won't generalize broadly). So we train the plan-head JOINTLY across games
(distill the gold plans -> K tokens) AND clone the DiT on the expert action chunks conditioned on those
tokens. Gold plans come from planner_poc/build_plan_jobs.py + subagent authoring -> docs/demo_plans/<game>/plans.jsonl.

Data: docs/demo_plans/<game>/segments.jsonl (segment metadata) + plans.jsonl (gold plan per seg_id) +
docs/demos/demos/<dir>/<demo>/demo.npz (exact obs + 12-button actions). Each segment -> A chunk examples
{frame@chunk-start, gold plan, expert 18-step action chunk}.

Trainable (default): plan_head (generalizer) + DiT LoRA (executor routing). --tune-action-head also opens
the diffusion action head (more capacity for clean expert supervision; safe because this is dense BC, not
noisy RL). NOTE: training plan_head re-routes the masked-null path (adaln_cond) -> CFG anchor must be
re-established after; acceptable for a bootstrap.

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/demo_train_stack.py --games smw sonic --steps 800
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import torch

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env, build_batch

JLX, JLY = 21, 22
# FUNCTION-BASED map: the same GAME FUNCTION -> the same NitroGen dim across consoles, so SOUTH(18) is
# ALWAYS "primary action / jump / confirm" regardless of the console's button letter (SNES B == GBA A ==
# Genesis jump). Keyed by the demo npz button INDEX (verified per-game order). NitroGen dims: south18
# east5 west20 north10 lshoulder7 rshoulder14 back0 start19.  (d-pad handled via the stick, below.)
# npz orders (verified from meta.json):
#   SNES   : B0 Y1 SELECT2 START3 UP4 DOWN5 LEFT6 RIGHT7 A8 X9 L10 R11
#   Genesis: B0 A1 MODE2 START3 UP4 DOWN5 LEFT6 RIGHT7 C8 Y9 X10 Z11
#   GBA    : B0 -1 SELECT2 START3 UP4 DOWN5 LEFT6 RIGHT7 A8 -9 L10 R11
BTN_MAPS = {
    # SMW: B=jump(primary)->south, A=spin->east, Y=run/dash->west, X->north, L/R shoulders, SELECT->back
    "snes": {0: 18, 8: 5, 1: 20, 9: 10, 10: 7, 11: 14, 2: 0, 3: 19},
    # Sonic: A/B/C ALL jump -> all to south (one consistent "jump" signal); Y/X/Z unused->north/shoulders
    "genesis": {0: 18, 1: 18, 8: 18, 9: 10, 10: 14, 11: 7, 2: 0, 3: 19},
    # GBA: A=primary(confirm/sword)->south, B=secondary(item/cancel)->west, L/R shoulders, SELECT->back
    "gba": {8: 18, 0: 20, 10: 7, 11: 14, 2: 0, 3: 19},
}
# d-pad indices are the SAME across all three demo layouts: (LEFT, RIGHT, UP, DOWN) = (6, 7, 4, 5)
DPAD = {"snes": (6, 7, 4, 5), "genesis": (6, 7, 4, 5), "gba": (6, 7, 4, 5)}

GAME_META = {
    "smw":        dict(dir="SuperMarioWorld-Snes", plat="snes", env="smw"),
    "sonic":      dict(dir="SonicTheHedgehog2-Genesis", plat="genesis", env="sonic"),
    "minish":     dict(dir="LegendOfZeldaTheMinishCap-GbAdvance", plat="gba", env="gba_minish_cap"),
    "fireemblem": dict(dir="FireEmblemTheSacredStones-GbAdvance", plat="gba", env=None),
}
EVAL_PLAN = "move right, run and jump over obstacles to advance"


def map_row(b12: np.ndarray, plat: str) -> np.ndarray:
    a = np.zeros(25, np.float32); a[JLX] = a[JLY] = 0.5; a[23] = a[24] = 0.5
    for s_idx, n_dim in BTN_MAPS[plat].items():
        if s_idx < len(b12) and b12[s_idx]:
            a[n_dim] = 1.0
    L, R, U, D = DPAD[plat]
    if b12[L]: a[JLX] = 0.0
    if b12[R]: a[JLX] = 1.0
    if b12[U]: a[JLY] = 0.0
    if b12[D]: a[JLY] = 1.0
    return a


def load_examples(game, demos_root="docs/demos/demos", plans_root="docs/demo_plans",
                  min_frames=1500, min_active=0.3):
    """Per-chunk examples {frame, plan, action(18,25), reward, game}. Uses segments.jsonl (metadata) +
    plans.jsonl (gold plans) + the demo npz (exact obs/actions). DROPS junk demos: < min_frames long OR
    < min_active fraction of frames with any button held (short menu/death fragments — user-flagged)."""
    meta = GAME_META[game]; plat = meta["plat"]
    gdir = os.path.join(_R, plans_root, game)
    plans = {json.loads(l)["seg_id"]: json.loads(l)["plan"] for l in open(os.path.join(gdir, "plans.jsonl"))}
    npz_cache = {}
    good_demo = {}
    out = []
    for line in open(os.path.join(gdir, "segments.jsonl")):
        s = json.loads(line)
        plan = plans.get(s["seg_id"])
        if not plan:
            continue
        if "no action needed" in plan.lower():
            continue  # skip idle segments for actor cloning (they still informed the planner)
        demo = s["demo"]
        if demo not in npz_cache:
            npz = os.path.join(_R, demos_root, meta["dir"], demo, "demo.npz")
            z = np.load(npz, allow_pickle=True); npz_cache[demo] = (z["observations"], z["actions"])
            acts_all = z["actions"]
            good_demo[demo] = (len(acts_all) >= min_frames and float((acts_all.sum(1) > 0).mean()) >= min_active)
        if not good_demo[demo]:
            continue  # junk demo (too short / mostly idle) -> excluded
        obs, acts = npz_cache[demo]
        H, A, stride, start = s["H"], s["A"], s["stride"], s["start_emu"]
        chunk_emu = H * stride
        for c in range(A):
            base = start + c * chunk_emu
            if base + chunk_emu > len(acts):
                break
            rows = np.stack([map_row(acts[base + k * stride], plat) for k in range(H)])
            out.append({"frame": np.asarray(obs[base]), "plan": plan,
                        "action": rows.astype(np.float32), "reward": 1.0, "game": game})
    dropped = [d for d, ok in good_demo.items() if not ok]
    if dropped:
        print(f"   [{game}] dropped {len(dropped)} junk demos (<{min_frames}f or <{min_active:.0%} active): "
              + ", ".join(d[-13:] for d in dropped))
    return out


def demo_states(game, demos_root="docs/demos/demos"):
    meta = GAME_META[game]
    return [gzip.decompress(open(p, "rb").read())
            for p in sorted(glob.glob(os.path.join(_R, demos_root, meta["dir"], "*", "initial.state")))]


@torch.no_grad()
def eval_advance(pol, env, states, plan, chunks, A, cfg):
    deltas = []
    for st in states:
        env.reset(); env.load_state(st)
        x0 = env._var(env.reward_var)
        for _ in range(chunks):
            f = env.frame()
            ch = pol._sample_chunk(f, plan, cfg, plan_frames=[f], null=False)
            env.step(ch[:A])
        deltas.append(float(env._var(env.reward_var) - x0))
    return float(np.mean(deltas)), deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", default=["smw", "sonic"])
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--tune-action-head", action="store_true", help="also train the DiT diffusion action head (more capacity)")
    ap.add_argument("--eval-game", default="smw")
    ap.add_argument("--eval-chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--save", default=None, help="path to save the trainable delta (slim)")
    args = ap.parse_args()
    device = "cuda"

    samples = []
    for g in args.games:
        ex = load_examples(g)
        print(f"[demo-stack] {g}: {len(ex)} chunk examples", flush=True)
        samples += ex
    print(f"[demo-stack] total {len(samples)} examples across {len(args.games)} games", flush=True)
    if not samples:
        print("no examples (plans.jsonl missing?) -- run the plan-author subagents first."); return

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("demo_stack", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m
    for n, p in m.named_parameters():
        train = ("lora_" in n) or n.startswith("plan_head.")
        if args.tune_action_head and (".action_decoder" in n or ".action_encoder" in n):
            train = True
        p.requires_grad_(train)
    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"[demo-stack] trainable {sum(p.numel() for p in train_params)/1e6:.2f}M "
          f"(plan_head + LoRA{' + action_head' if args.tune_action_head else ''})", flush=True)

    env = None
    if GAME_META[args.eval_game]["env"]:
        env = make_env(GAME_META[args.eval_game]["env"])
    try:
        states = demo_states(args.eval_game)
        if env is not None and states:
            pre, _ = eval_advance(pol, env, states, EVAL_PLAN, args.eval_chunks, args.A, args.cfg)
            print(f"[demo-stack] PRE: {args.eval_game} screen_x advance from {len(states)} demo starts = {pre:+.1f}", flush=True)
        else:
            pre = None

        opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
        for step in range(args.steps):
            m.train()
            idx = np.random.choice(len(samples), size=min(args.bs, len(samples)), replace=False)
            batch = build_batch(pol, [samples[i] for i in idx], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(batch)["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
            if step % 50 == 0 or step == args.steps - 1:
                print(f"   step {step:3d} loss {float(loss.detach()):.4f}", flush=True)
        m.eval()

        if env is not None and states:
            post, dl = eval_advance(pol, env, states, EVAL_PLAN, args.eval_chunks, args.A, args.cfg)
            print(f"\n===== DEMO-STACK RESULT (eval {args.eval_game}) =====")
            print(f"  screen_x advance from demo starts: PRE {pre:+.1f} -> POST {post:+.1f}  (Δ={post-pre:+.1f})")
            print(f"  => bootstrap {'LIFTS the actor' if post > pre + 20 else 'no clear actor lift'} (planner+DiT on gold demos).")
    finally:
        if env is not None:
            env.close()

    if args.save:
        delta = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
        torch.save({"trainable": delta, "games": args.games}, args.save)
        print(f"  saved trainable delta -> {args.save}")


if __name__ == "__main__":
    main()
