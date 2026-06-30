"""judge_calibrate.py -- test whether a CHEAP TRAINED judge (a linear probe on frozen SigLIP features of
the before/after frame pair) predicts GROUND-TRUTH progress better than the raw Qwen-2B VLM judge.

The night's bench showed the raw 2B judge is not RL-reward-grade (over-credits; 0.56 vs GT). This asks:
can we cheaply CALIBRATE a judge against GT-labeled trajectory data (from collect_rl_trajectories.py)?
If a frozen-feature linear probe beats majority + the raw judge, a trained judge is the reward path for
games without RAM state.

Uses SigLIP2-large (NitroGen's vision encoder) pooled features: x = [f_before, f_after, f_after-f_before]
-> linear probe -> P(progress>0). Train/val split BY EPISODE (no leakage). Reports val accuracy vs the
majority-class baseline (and the raw-judge GT-match from the bench for reference).

Run (one GPU):
  CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc \
    .venv/bin/python planner_poc/judge_calibrate.py --data-dirs docs/rl_data/thextech docs/rl_data/sonic
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R)


def load_records(dirs):
    recs = []
    for d in dirs:
        mf = os.path.join(d, "manifest.jsonl")
        if not os.path.exists(mf):
            continue
        for line in open(mf):
            r = json.loads(line)
            r["_dir"] = d
            r["_ep"] = f"{os.path.basename(d)}:{r['ep']}"
            recs.append(r)
    return recs


@torch.no_grad()
def siglip_features(recs, device):
    from transformers import AutoModel, AutoImageProcessor
    name = "google/siglip2-large-patch16-256"
    ip = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    vis = model.vision_model if hasattr(model, "vision_model") else model
    cache = {}

    def feat(path):
        if path in cache:
            return cache[path]
        im = Image.open(path).convert("RGB")
        pv = ip(images=[im], return_tensors="pt")["pixel_values"].to(device)
        out = vis(pixel_values=pv)
        f = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state.mean(1)
        v = f[0].float().cpu().numpy()
        cache[path] = v
        return v

    X, y, eps = [], [], []
    for r in recs:
        bp = os.path.join(r["_dir"], r["before_png"]); ap = os.path.join(r["_dir"], r["after_png"])
        if not (os.path.exists(bp) and os.path.exists(ap)):
            continue
        fb, fa = feat(bp), feat(ap)
        X.append(np.concatenate([fb, fa, fa - fb]))
        y.append(1 if r["gt_progress"] > 0 else 0)   # binary: real progress vs not
        eps.append(r["_ep"])
    return np.array(X, np.float32), np.array(y, np.int64), np.array(eps)


def train_probe(Xtr, ytr, Xva, yva, device, epochs=300, lr=1e-3, wd=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    xv = torch.tensor(Xva, device=device)
    clf = torch.nn.Linear(Xtr.shape[1], 2).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(clf(xt), yt); loss.backward(); opt.step()
    with torch.no_grad():
        pred = clf(xv).argmax(1).cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--val-frac", type=float, default=0.3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recs = load_records(args.data_dirs)
    print(f"loaded {len(recs)} chunks from {len(args.data_dirs)} dirs")
    X, y, eps = siglip_features(recs, device)
    print(f"features {X.shape}, positives {y.mean():.2f}")

    # split BY EPISODE (no frame leakage)
    uniq = sorted(set(eps)); rng = np.random.default_rng(0); rng.shuffle(uniq)
    nval = max(1, int(len(uniq) * args.val_frac))
    val_eps = set(uniq[:nval])
    tr = np.array([e not in val_eps for e in eps]); va = ~tr
    if va.sum() == 0 or tr.sum() == 0:
        print("not enough episodes to split"); return
    pred = train_probe(X[tr], y[tr], X[va], y[va], device)
    yva = y[va]
    acc = float((pred == yva).mean())
    maj = float(max(yva.mean(), 1 - yva.mean()))   # majority-class baseline
    # progress-detection: among true-progress, how many caught; among non-progress, false-positive rate
    tp = int(((pred == 1) & (yva == 1)).sum()); fp = int(((pred == 1) & (yva == 0)).sum())
    pos = int((yva == 1).sum()); neg = int((yva == 0).sum())
    print(f"\n=== TRAINED SigLIP-probe judge (binary progress) ===")
    print(f"  val n={len(yva)} (pos {pos}/neg {neg})")
    print(f"  probe accuracy   = {acc:.3f}")
    print(f"  majority baseline= {maj:.3f}")
    print(f"  recall(progress) = {tp}/{pos} = {tp/max(1,pos):.2f}  false-pos = {fp}/{neg} = {fp/max(1,neg):.2f}")
    print(f"  (reference: raw Qwen-2B judge GT-progress exact was 0.56 / false-pos 0.22 in the bench)")
    print(f"\n  => trained-probe {'BEATS' if acc>maj else 'does NOT beat'} majority; "
          f"{'feasible' if acc>maj+0.05 else 'inconclusive (need more/varied data)'} cheap trained judge.")


if __name__ == "__main__":
    main()
