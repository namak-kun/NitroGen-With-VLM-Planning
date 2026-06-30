"""train_multi.py -- the MULTI-GENRE one-LoRA capacity test (the user's "per-genre actors won't scale /
won't generalize to all games" concern, now with FIXED button maps + corrected plans). Loads RWBC samples
collected per-game (collect_samples.py), keeps each game's top-return chunks, and trains ONE LoRA-only delta
on the MERGED set via reward-weighted BC. Eval is done separately per game (held_out_eval.py with this
delta) so emulator constructions stay segfault-isolated.

Question: does one shared LoRA improve ALL genres, or does multi-genre training COLLAPSE (the prior
'4-genre collapse' was confounded by the GBA button-map bug — now fixed)?

Run:  RUN=... CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/train_multi.py \
        --pkls docs/multigenre/sonic.pkl docs/multigenre/smw.pkl docs/multigenre/minish.pkl \
        --top-frac 0.4 --steps 150 --lr 5e-5 --anchor 0.002 --save ckpts/rwbc_multigenre.pt
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np
import torch

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import build_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls", nargs="+", required=True)
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--top-frac", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--anchor", type=float, default=0.002)
    ap.add_argument("--balance", action="store_true", help="downsample each game to the smallest game's kept count")
    ap.add_argument("--per-game-rwnorm", action="store_true", help="normalize reward-weights WITHIN each game so no genre dominates the gradient")
    ap.add_argument("--save", required=True)
    args = ap.parse_args()
    device = "cuda"

    # load + per-game top-frac filtering (each game contributes its BEST chunks)
    keep = []
    per_game = {}
    for p in args.pkls:
        d = pickle.load(open(p, "rb"))
        s = d["samples"]; r = np.array([x["reward"] for x in s])
        thr = np.quantile(r, 1 - args.top_frac)
        k = [x for x in s if x["reward"] >= thr]
        per_game[d["env"]] = k
        print(f"  {d['env']}: {len(s)} -> keep {len(k)} (reward>={thr:+.3f}, kept-mean {np.mean([x['reward'] for x in k]):+.3f})", flush=True)
    if args.balance:
        m = min(len(v) for v in per_game.values())
        for g in per_game:
            idx = np.random.choice(len(per_game[g]), m, replace=False)
            per_game[g] = [per_game[g][i] for i in idx]
        print(f"  balanced each game to {m} chunks", flush=True)
    # reward-weights: either global min-max, or normalized WITHIN each game (no genre dominates gradient)
    if args.per_game_rwnorm:
        for g, v in per_game.items():
            r = np.array([s["reward"] for s in v])
            rwn = (r - r.min()) / (r.max() - r.min() + 1e-6) + 0.2
            for s, wv in zip(v, rwn):
                s["_rw"] = float(wv)
        print("  reward-weights normalized WITHIN each game", flush=True)
    for g, v in per_game.items():
        keep.extend(v)
    print(f"  MERGED kept chunks: {len(keep)} across {len(per_game)} games", flush=True)

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=8.0)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("multi", plan="", objective="progress", cfg_scale=8.0))
    m = pol.m
    for n, p in m.named_parameters():
        p.requires_grad_("lora_" in n)               # LoRA-only (anti-collapse)
    train_params = [p for n, p in m.named_parameters() if p.requires_grad]
    init = [p.detach().clone() for p in train_params]
    print(f"  trainable {sum(p.numel() for p in train_params)/1e6:.2f}M (LoRA-only) lr={args.lr} anchor={args.anchor}", flush=True)

    rw = np.array([s.get("_rw", s["reward"]) for s in keep])
    if not args.per_game_rwnorm:
        rw = (rw - rw.min()) / (rw.max() - rw.min() + 1e-6) + 0.2
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    for step in range(args.steps):
        m.train()
        idx = np.random.choice(len(keep), size=min(args.bs, len(keep)), replace=False)
        batch = build_batch(pol, [keep[i] for i in idx], device)
        w = torch.tensor(rw[idx], device=device, dtype=torch.float32).mean()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(batch); loss = out["loss"] * w
        if args.anchor > 0:
            loss = loss + args.anchor * sum(((p - p0) ** 2).sum() for p, p0 in zip(train_params, init))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"   step {step:3d} loss {float(out['loss'].detach()):.4f}", flush=True)
    m.eval()
    sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
    torch.save({"trainable": sd, "games": list(per_game.keys()), "lora_only": True}, args.save)
    print(f"  saved multi-genre delta ({len(sd)} tensors) -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
