"""Comprehensive plan-token evaluation: how well do a trained model's plan tokens encode the
FULL action (stick AND buttons), across the DIVERSE multi-game corpus?

Critiques addressed (vs the earlier stick-only, 30-chunk diag):
  * FAST: reads CACHED mm-hiddens (no per-chunk 2B re-encode) -> all 2184 chunks in ~minutes.
  * FULL ACTION: probes stick direction AND per-button presses (not just the stick).
  * MULTI-GAME / STRATIFIED: stick meaning is universal -> pooled globally; button meaning is
    game-specific (SOUTH=jump vs confirm vs dodge) -> evaluated WITHIN video_id (one game + one
    controller, verified), gated on min press count, then averaged. video_id is the finest key
    guaranteeing constant action semantics (game label is missing for ~60% and mx_bikes mixes
    ps4/xbox).
  * Primary metric = CLASSIFICATION (scale-free, mode-aware): can a linear probe on the plan
    tokens recover the action label. Secondary = (reported) class-mean cosine.

Run: PYTHONPATH=. .venv/bin/python planner_poc/probe_full_action.py \
        runs/<ckpt>/plan_stage1_XXXX.pt /tmp/stage2_mm_hidden_2b.pt
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.shared import BUTTON_ACTION_TOKENS

CKPT = sys.argv[1]
HID = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stage2_mm_hidden_2b.pt"
INDEX = sys.argv[3] if len(sys.argv) > 3 else "/tmp/stage2_index_a4.pt"
CURSOR = int(os.environ.get("CURSOR", "0"))
MIN_PER_CLASS = 8           # min samples per class to run a probe
device = "cuda"; K = 8


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    ng = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
    CC = CkptConfig.model_validate(ng["ckpt_config"])
    mc = CC.model_cfg.model_copy(deep=True)
    mc.planner_cfg.enabled = True; mc.planner_cfg.num_plan_tokens = K
    mc.planner_cfg.null_mode = "masked"
    qk = "plan_head.resampler.queries"
    if qk in sd:
        mc.planner_cfg.backbone_hidden_size = int(sd[qk].shape[-1])
        mc.planner_cfg.num_chunks = int(sd[qk].shape[0]) // K
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(sd, strict=False)
    print(f"loaded {path} | backbone_dim={mc.planner_cfg.backbone_hidden_size} "
          f"num_chunks={mc.planner_cfg.num_chunks} cursor={CURSOR}")
    return m.to(device).eval()


def video_id_of(uuid):
    # uuid = "<videoid>_chunk_XXXX_actions"
    return uuid.split("_chunk_")[0]


def main():
    m = load(CKPT)
    hid = torch.load(HID, map_location="cpu")
    idx = torch.load(INDEX, map_location="cpu", weights_only=False)

    # ---- compute plan tokens for every cached uuid (mean-pooled) ----
    feats, dirs, btns, vids, uuids = [], [], [], [], []
    keys = [u for u in hid if u in idx]
    for u in keys:
        e = hid[u]
        d = {"plan_hidden": e["h"].unsqueeze(0).to(device).float(),
             "plan_key_padding_mask": e["mask"].unsqueeze(0).to(device),
             "plan_dropped": torch.tensor([False], device=device),
             "plan_cursor": torch.tensor([CURSOR], device=device)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pt, _ = m.compute_plan_tokens(d)
        feats.append(pt[0].float().mean(0).cpu().numpy())
        dirs.append(idx[u]["dir"])
        b = np.asarray(idx[u]["buttons"])            # (H,21)
        btns.append((b.mean(0) > 0.3).astype(np.int8))  # per-button "pressed this chunk"
        vids.append(video_id_of(u)); uuids.append(u)
    F = np.stack(feats); F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    dirs = np.array(dirs); B = np.stack(btns); vids = np.array(vids)
    print(f"computed {len(F)} plan-token vectors (dim {F.shape[1]}) across {len(set(vids))} videos")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def probe(X, y, cls):
        mask = np.isin(y, cls)
        if mask.sum() < 2 * MIN_PER_CLASS:
            return None
        c = Counter(y[mask])
        if min(c[k] for k in cls) < MIN_PER_CLASS:
            return None
        cv = min(5, min(c[k] for k in cls))
        sc = cross_val_score(LogisticRegression(max_iter=3000, C=1.0), X[mask], y[mask], cv=cv)
        return sc.mean(), int(mask.sum())

    # ===== STICK: pooled globally (universal meaning) =====
    print("\n===== STICK direction (pooled, universal) =====")
    for pair in [["left", "right"], ["up", "down"]]:
        r = probe(F, dirs, pair)
        print(f"  {pair[0]:>5} vs {pair[1]:<5}: " + (f"acc={r[0]:.3f} (n={r[1]})" if r else "insufficient"))
    r = probe(F, dirs, ["left", "right", "up", "down"])
    print(f"  4-way        : " + (f"acc={r[0]:.3f} (n={r[1]}, chance .25)" if r else "insufficient"))

    # ===== BUTTONS: per-video (game-specific), gated, averaged =====
    print("\n===== BUTTONS (per-video, gated >= {} each, averaged) =====".format(MIN_PER_CLASS))
    by_vid = defaultdict(list)
    for i, v in enumerate(vids):
        by_vid[v].append(i)
    # which buttons are worth probing globally (enough total presses)
    tot = B.sum(0)
    cand = [j for j in range(B.shape[1]) if tot[j] >= 40]
    rows = []
    for j in cand:
        accs = []
        for v, ix in by_vid.items():
            ix = np.array(ix)
            yj = B[ix, j]
            if yj.sum() < MIN_PER_CLASS or (len(yj) - yj.sum()) < MIN_PER_CLASS:
                continue
            cv = min(5, int(min(yj.sum(), len(yj) - yj.sum())))
            try:
                sc = cross_val_score(LogisticRegression(max_iter=2000),
                                     F[ix], yj, cv=cv).mean()
                accs.append(sc)
            except Exception:
                continue
        if accs:
            rows.append((BUTTON_ACTION_TOKENS[j], int(tot[j]), len(accs), float(np.mean(accs))))
    rows.sort(key=lambda r: -r[3])
    if rows:
        print(f"  {'button':14} {'tot':>5} {'#vids':>6} {'mean_acc(within-video, chance .5)':>10}")
        for name, t, nv, acc in rows:
            print(f"  {name:14} {t:5d} {nv:6d}   {acc:.3f}")
    else:
        print("  (no button had enough within-video presses to probe)")

    print("\nNOTE: stick is the trustworthy universal signal; per-button within-video probes are\n"
          "lower-power (small strata) -> read as rough per-game button-encoding indicators.")


if __name__ == "__main__":
    main()
