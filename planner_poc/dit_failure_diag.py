"""dit_failure_diag.py -- DIAGNOSE where the DiT diverges from the human EXPERT and what it should have
done. On good expert demo chunks (frame + gold plan + expert 18-step action), run the DiT (base or
bootstrapped) conditioned on the SAME frame + gold plan, and compare the produced action to the expert's,
per dimension. Answers: is the DiT's failure "WHAT to do" (direction/selection) or "HOW to execute" (motor
precision)? -> decides whether to lean on the VLM (richer plans / action tokens) or a better actor.

Per chunk we compare (NitroGen 25-dim: j_left x=21,y=22 [0.5=neutral]; jump=south18; run=west20; spin=east5):
  - dir_x: LEFT/RIGHT/NEUTRAL (deadzone .08) agreement (expert vs DiT)
  - dir_y: UP/DOWN/NEUTRAL agreement
  - jump: did DiT jump (rate>.2) when expert jumped? -> recall + precision
  - per-dim mean-abs-error; a per-chunk divergence score -> rank worst chunks
Outputs an aggregate report + the top-divergence chunks (frame path, gold plan, expert vs DiT action
summary) to docs/demo_plans/<game>/dit_failures.json for GPT-5.5 to explain "what it should have done".

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/dit_failure_diag.py --game smw \
     --delta ckpts/demo_stack_plat_filtered.pt --top 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from demo_train_stack import load_examples, GAME_META

JLX, JLY, JUMP, RUN, SPIN = 21, 22, 18, 20, 5
DEAD = 0.08


def dir3(v, lo, hi):
    if v < 0.5 - DEAD: return lo
    if v > 0.5 + DEAD: return hi
    return "·"


def summ(chunk):
    x, y = chunk[:, JLX].mean(), chunk[:, JLY].mean()
    return (f"x={dir3(x,'L','R')}({x:.2f}) y={dir3(y,'U','D')}({y:.2f}) "
            f"jump={(chunk[:,JUMP]>0.5).mean():.2f} run={(chunk[:,RUN]>0.5).mean():.2f}")


def load_delta(m, path):
    d = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
    sd = m.state_dict()
    for k, v in d.items():
        if k in sd: sd[k].copy_(v.to(sd[k].dtype))


@torch.no_grad()
def diagnose(pol, samples, cfg, nsample=2):
    rows = []
    for i, s in enumerate(samples):
        exp = s["action"]
        # average a couple DiT samples (greedy-ish via fixed seeds) for a stable estimate
        dits = [pol._sample_chunk(s["frame"], s["plan"], cfg, plan_frames=[s["frame"]], null=False,
                                  noise_seed=1000 * i + j) for j in range(nsample)]
        dit = np.mean(dits, 0)
        ex_x, ex_y = exp[:, JLX].mean(), exp[:, JLY].mean()
        di_x, di_y = dit[:, JLX].mean(), dit[:, JLY].mean()
        ex_j, di_j = (exp[:, JUMP] > 0.5).mean(), (dit[:, JUMP] > 0.5).mean()
        rows.append({
            "i": i, "plan": s["plan"], "frame": s.get("_frame_path"),
            "dx_match": dir3(ex_x, "L", "R") == dir3(di_x, "L", "R"),
            "dy_match": dir3(ex_y, "U", "D") == dir3(di_y, "U", "D"),
            "exp_jump": ex_j > 0.2, "dit_jump": di_j > 0.2,
            "div": abs(ex_x - di_x) + abs(ex_y - di_y) + abs(ex_j - di_j),
            "exp_summ": summ(exp), "dit_summ": summ(dit),
        })
    return rows


def report(tag, rows):
    n = len(rows)
    dx = np.mean([r["dx_match"] for r in rows])
    dy = np.mean([r["dy_match"] for r in rows])
    ej = np.array([r["exp_jump"] for r in rows]); dj = np.array([r["dit_jump"] for r in rows])
    jrecall = (ej & dj).sum() / max(1, ej.sum())
    jprec = (ej & dj).sum() / max(1, dj.sum())
    print(f"\n=== {tag} (n={n}) ===")
    print(f"  direction LEFT/RIGHT agreement = {dx:.2f}   (the WHAT axis -- selection)")
    print(f"  direction UP/DOWN   agreement = {dy:.2f}")
    print(f"  JUMP recall (DiT jumps when expert does) = {jrecall:.2f}  precision = {jprec:.2f}")
    print(f"  expert-jump chunks = {int(ej.sum())}/{n};  DiT-jump chunks = {int(dj.sum())}/{n}")
    return {"n": n, "dx_match": float(dx), "dy_match": float(dy),
            "jump_recall": float(jrecall), "jump_precision": float(jprec)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--delta", default=None, help="bootstrap delta to ALSO diagnose (compare vs base)")
    ap.add_argument("--max-examples", type=int, default=80)
    ap.add_argument("--top", type=int, default=16, help="worst-divergence chunks to dump for GPT-5.5")
    ap.add_argument("--cfg", type=float, default=8.0)
    args = ap.parse_args()
    device = "cuda"

    samples = load_examples(args.game)[: args.max_examples]
    # attach a frame path for the dump (re-extract the obs to a png lazily)
    fdir = os.path.join(_R, "docs/demo_plans", args.game, "diag_frames"); os.makedirs(fdir, exist_ok=True)
    from PIL import Image
    for i, s in enumerate(samples):
        p = os.path.join(fdir, f"chunk_{i:03d}.png"); Image.fromarray(s["frame"]).convert("RGB").save(p)
        s["_frame_path"] = p
    print(f"[dit-diag] {args.game}: {len(samples)} good expert chunks", flush=True)

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("dit_diag", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m
    base_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}

    base_rows = diagnose(pol, samples, args.cfg)
    base_agg = report("BASE DiT", base_rows)

    boot_rows = None
    if args.delta:
        m.load_state_dict(base_sd); load_delta(m, args.delta); m.eval()
        boot_rows = diagnose(pol, samples, args.cfg)
        report(f"BOOTSTRAPPED ({os.path.basename(args.delta)})", boot_rows)
        m.load_state_dict(base_sd)

    # dump worst-divergence chunks (base) for GPT-5.5 to explain "what it should have done"
    worst = sorted(base_rows, key=lambda r: -r["div"])[: args.top]
    out = os.path.join(_R, "docs/demo_plans", args.game, "dit_failures.json")
    json.dump([{"frame": r["frame"], "gold_plan": r["plan"], "expert_action": r["exp_summ"],
                "dit_action": r["dit_summ"], "divergence": round(r["div"], 3),
                "dir_x_match": r["dx_match"], "jump_match": r["exp_jump"] == r["dit_jump"]}
               for r in worst], open(out, "w"), indent=2)
    print(f"\n[dit-diag] top-{args.top} DiT failures (frame+expert vs DiT) -> {out}")
    print("  worst 5 (expert -> DiT):")
    for r in worst[:5]:
        print(f"   div={r['div']:.2f} | plan: {r['plan'][:46]}")
        print(f"       expert: {r['exp_summ']}\n       DiT   : {r['dit_summ']}")


if __name__ == "__main__":
    main()
