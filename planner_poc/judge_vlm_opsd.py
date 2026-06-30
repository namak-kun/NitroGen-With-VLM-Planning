"""judge_vlm_opsd.py -- VLM-as-judge (discriminative) + OPSD privileged-info self-distillation.

@namak-kun's judge direction: NOT a SigLIP probe, NOT SFT on bigger-model thinking traces. Instead train
the FROZEN Qwen planner-VLM's own multimodal features into a progress judge, and improve it via OPSD-style
self-distillation with GT (privileged) info: a PRIVILEGED student that sees the GT hint distills into the
UNPRIVILEGED (pixels-only) student. No traces; generation/think untouched (a separate scalar head on the
frozen backbone).

Three judges trained + compared on the SAME episode-split + binary-progress metric as judge_calibrate.py:
  1. SigLIP probe (baseline to beat) -- reuses judge_calibrate.siglip_features.
  2. VLM-UNPRIV: frozen-Qwen features over [before,after]+objective -> head. (the "VLM-as-judge" core.)
  3. VLM-OPSD : a fresh unpriv head distilled from a PRIVILEGED teacher (same backbone, GT hint in prompt).
Tests: (a) do VLM features beat SigLIP? (b) does privileged distillation beat plain-supervised VLM-unpriv?

Run (GPU; ~few min for 253 pairs):
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=1 .venv/bin/python planner_poc/judge_vlm_opsd.py \
      --data-dirs docs/rl_data/sonic docs/rl_data/thextech docs/rl_data/solarus_zelda
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from judge_calibrate import load_records
from PIL import Image

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")

# objective per game-dir (the judge's grounding); dir basename -> objective text.
OBJ = {
    "sonic": "advance RIGHT through the level as far as possible",
    "thextech": "advance RIGHT and reach the level exit",
    "solarus_zelda": "explore and move toward new areas/exits",
}


def _obj_for(d):
    return OBJ.get(os.path.basename(d.rstrip("/")), "make progress in the game")


@torch.no_grad()
def vlm_features(recs, device, privileged: bool, cache_path: str, priv_mode: str = "hint", future_k: int = 2):
    """Frozen-Qwen pooled hidden over [before,after] frames + a judge prompt. privileged=True adds a
    PRIVILEGED signal (teacher): priv_mode='hint' appends the GT progress-sign hint (trivial); priv_mode=
    'future' instead shows a FUTURE frame (k chunks ahead, same episode) that reveals the outcome WITHOUT
    stating the label (non-trivial privileged info). Caches to cache_path."""
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d["X"], d["y"], d["eps"]
    from nitrogen.planner import PlanEncoder, PlannerConfig
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load(); pl.backbone.to(device)
    # future-frame index for priv_mode='future': (dir, ep, t) -> after_png path
    fut = {(r["_dir"], r.get("ep"), r.get("t")): os.path.join(r["_dir"], r["after_png"]) for r in recs}
    X, y, eps = [], [], []
    for r in recs:
        bp = os.path.join(r["_dir"], r["before_png"]); ap = os.path.join(r["_dir"], r["after_png"])
        if not (os.path.exists(bp) and os.path.exists(ap)):
            continue
        obj = _obj_for(r["_dir"])
        prompt = (f"Objective: {obj}. The first image is BEFORE a short action, the second is AFTER. "
                  f"Did the player make progress toward the objective?")
        frames = [Image.open(bp).convert("RGB"), Image.open(ap).convert("RGB")]
        if privileged and priv_mode == "hint":
            prompt += f" Privileged hint: the progress metric changed by {int(r['gt_progress']):+d} " \
                      f"(positive means progress)."
        elif privileged and priv_mode == "future":
            fp = fut.get((r["_dir"], r.get("ep"), (r.get("t") or 0) + future_k))
            if fp and os.path.exists(fp):
                frames.append(Image.open(fp).convert("RGB"))   # privileged: a few chunks into the future
                prompt += (" The third image is the situation a few moments LATER; use it as privileged "
                           "hindsight to judge whether real progress was being made.")
        h, kpm = pl.encode_multimodal(frames, prompt, device, text_only=True)  # (1,L,d), text tokens
        valid = (~kpm[0]).float().unsqueeze(-1)                                 # (L,1)
        feat = (h[0] * valid).sum(0) / valid.sum().clamp(min=1)                 # mean-pool text tokens
        X.append(feat.float().cpu().numpy())
        y.append(1 if r["gt_progress"] > 0 else 0)
        eps.append(r["_ep"])
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64); eps = np.asarray(eps)
    np.savez(cache_path, X=X, y=y, eps=eps)
    del pl
    torch.cuda.empty_cache()
    return X, y, eps


def _split(eps, val_frac=0.3):
    uniq = sorted(set(eps.tolist())); rng = np.random.default_rng(0); rng.shuffle(uniq)
    nval = max(1, int(len(uniq) * val_frac)); val = set(uniq[:nval])
    tr = np.array([e not in val for e in eps]); return tr, ~tr


def _norm(Xtr, Xva):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xva - mu) / sd


def train_head(Xtr, ytr, Xva, device, epochs=300, lr=1e-3, wd=1e-2, hidden=0,
               teacher_logits=None, distill_T=2.0, distill_w=1.0):
    """Linear (hidden=0) or 1-hidden-MLP head. If teacher_logits given (on TRAIN rows), add a KL
    distillation term (OPSD): student matches the privileged teacher's soft distribution + GT CE."""
    xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    xv = torch.tensor(Xva, device=device)
    if hidden > 0:
        clf = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(),
                                  torch.nn.Linear(hidden, 2)).to(device)
    else:
        clf = torch.nn.Linear(Xtr.shape[1], 2).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    ce = torch.nn.CrossEntropyLoss()
    tl = torch.tensor(teacher_logits, device=device) if teacher_logits is not None else None
    for _ in range(epochs):
        opt.zero_grad()
        logit = clf(xt)
        loss = ce(logit, yt)
        if tl is not None:
            ls = torch.log_softmax(logit / distill_T, dim=1)
            pt = torch.softmax(tl / distill_T, dim=1)
            loss = loss + distill_w * (distill_T ** 2) * torch.nn.functional.kl_div(ls, pt, reduction="batchmean")
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = clf(xv).argmax(1).cpu().numpy()
        train_logits = clf(xt).detach().cpu().numpy()
    return pred, train_logits


def report(name, pred, yva, extra=""):
    acc = float((pred == yva).mean()); maj = float(max(yva.mean(), 1 - yva.mean()))
    pos = int((yva == 1).sum()); neg = int((yva == 0).sum())
    tp = int(((pred == 1) & (yva == 1)).sum()); fp = int(((pred == 1) & (yva == 0)).sum())
    print(f"  {name:16s} acc={acc:.3f}  (maj {maj:.3f})  recall={tp}/{pos}={tp/max(1,pos):.2f}  "
          f"fp={fp}/{neg}={fp/max(1,neg):.2f}  {extra}")
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--cache-dir", default="tmp/judge_feats")
    ap.add_argument("--hidden", type=int, default=0, help="MLP hidden dim (0=linear)")
    ap.add_argument("--siglip", action="store_true", help="also run the SigLIP baseline")
    ap.add_argument("--priv-mode", default="hint", choices=["hint", "future"],
                    help="privileged teacher signal: 'hint'=GT-sign text (trivial); 'future'=a future frame (non-trivial)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.cache_dir, exist_ok=True)
    key = hashlib.md5(("|".join(sorted(args.data_dirs)) + "|" + args.priv_mode).encode()).hexdigest()[:8]

    recs = load_records(args.data_dirs)
    print(f"loaded {len(recs)} records from {len(args.data_dirs)} dirs (priv-mode={args.priv_mode})")

    Xu, yu, epu = vlm_features(recs, device, False, os.path.join(args.cache_dir, f"unpriv_{key}.npz"))
    Xp, yp, epp = vlm_features(recs, device, True, os.path.join(args.cache_dir, f"priv_{key}.npz"),
                               priv_mode=args.priv_mode)
    print(f"VLM features: unpriv {Xu.shape}, priv {Xp.shape}, positives {yu.mean():.2f}")

    tr, va = _split(epu); yva = yu[va]
    Xtr_u, Xva_u = _norm(Xu[tr], Xu[va])
    Xtr_p, Xva_p = _norm(Xp[tr], Xp[va])
    print(f"\n=== JUDGES (val n={int(va.sum())}, pos {int(yva.sum())}/neg {int((1-yva).sum())}) ===")
    print("  (reference: raw Qwen-2B generative judge exact=0.56; SigLIP probe 0.78 platformer / 0.656 all-3)")

    if args.siglip:
        from judge_calibrate import siglip_features, train_probe
        Xs, ys, eps = siglip_features(recs, device)
        trs, vas = _split(eps)
        ps = train_probe(Xs[trs], ys[trs], Xs[vas], ys[vas], device)
        report("SigLIP-probe", ps, ys[vas])

    # 2. VLM unprivileged (the VLM-as-judge core)
    pred_u, logit_u_tr = train_head(Xtr_u, yu[tr], Xva_u, device, hidden=args.hidden)
    report("VLM-unpriv", pred_u, yva)

    # teacher: privileged head; produce its TRAIN-row soft logits to distill
    pred_p, logit_p_tr = train_head(Xtr_p, yp[tr], Xva_p, device, hidden=args.hidden)
    report("VLM-priv(teacher)", pred_p, yva, "<- sees GT hint (upper bound)")

    # 3. OPSD: fresh UNPRIV head distilled from the privileged teacher's soft logits
    pred_d, _ = train_head(Xtr_u, yu[tr], Xva_u, device, hidden=args.hidden,
                           teacher_logits=logit_p_tr, distill_T=2.0, distill_w=1.0)
    report("VLM-OPSD", pred_d, yva, "<- unpriv distilled from priv teacher")

    print("\n  => compare VLM-unpriv vs SigLIP (is the VLM a better judge?) and VLM-OPSD vs VLM-unpriv "
          "(does privileged self-distillation help?). Small data -> treat as directional.")


if __name__ == "__main__":
    main()
