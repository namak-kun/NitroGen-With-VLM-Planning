"""Foundational check: does the Stage-1 grounding model separate left/right from EXPLICIT
directional text plans ("keep going left" vs "keep going right")?

This tests the PREMISE of the whole pipeline. The factual/mm probes showed contrastive-on-base
can't separate left/right. But the proven recipe FIRST grounds the base direction<->token mapping
with synthetic directional plans. If even explicit "move left"/"move right" text fails to produce
separated tokens at 2B, the premise is broken. If it succeeds, Stage-1 grounding is the right
foundation and we build teacher/distill on top.

For each direction we feed many phrasings (hold/tap), encode_text -> plan_head -> K tokens,
mean-pool, and measure left/right (+ 4-way) linear separability.

Run: PYTHONPATH=. .venv/bin/python planner_poc/probe_syntext_separation.py \
        runs/stage1_2b_syntest/plan_stage1_800.pt Qwen/Qwen3.5-2B
"""
import sys
import numpy as np
import torch

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/stage1_2b_syntest/plan_stage1_800.pt"
QWEN = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-2B"
device = "cuda"; K = 8

HOLD = ["keep going {d}", "hold {d} the whole time", "move {d} continuously",
        "go {d}", "head {d}", "keep moving {d}", "continue {d}", "press {d}"]
TAP = ["tap {d} then keep playing", "briefly go {d}", "nudge {d} then continue", "quick {d}"]
DIRS = ["left", "right", "up", "down"]


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
          f"num_chunks={mc.planner_cfg.num_chunks}")
    return m.to(device).eval()


def main():
    m = load(CKPT)
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load()
    cache = PlanHiddenCache(pl, device)

    def toks(text):
        h, kpm = cache.get(text)
        d = {"plan_hidden": h.unsqueeze(0).to(device).float(),
             "plan_key_padding_mask": kpm.unsqueeze(0).to(device),
             "plan_dropped": torch.tensor([False], device=device),
             "plan_cursor": torch.tensor([0], device=device)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pt, _ = m.compute_plan_tokens(d)
        return pt[0].float().mean(0).cpu().numpy()

    X, y = [], []
    for d in DIRS:
        for tmpl in HOLD + TAP:
            X.append(toks(tmpl.format(d=d))); y.append(d)
    X = np.stack(X); y = np.array(y)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    def cmean_cos(a, b):
        ma = Xn[y == a].mean(0); mb = Xn[y == b].mean(0)
        ma /= np.linalg.norm(ma) + 1e-8; mb /= np.linalg.norm(mb) + 1e-8
        return float(ma @ mb)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def probe(classes):
        mask = np.isin(y, classes)
        sc = cross_val_score(LogisticRegression(max_iter=3000), Xn[mask], y[mask],
                             cv=min(5, (y[mask] == classes[0]).sum()))
        return sc.mean(), sc.std(), int(mask.sum())

    print(f"\n=== SYNTHETIC directional text -> token separation ({len(y)} samples) ===")
    for pair in [["left", "right"], ["up", "down"]]:
        r = probe(pair)
        print(f"  {pair[0]:>5} vs {pair[1]:<5}: acc={r[0]:.3f}±{r[1]:.3f} (n={r[2]}) "
              f"| class-mean cos={cmean_cos(*pair):+.3f}")
    r = probe(DIRS)
    print(f"  4-way dir   : acc={r[0]:.3f}±{r[1]:.3f} (n={r[2]}, chance=0.25)")
    print("\nPREMISE CHECK: if left/right acc ~1.0 and cos low -> Stage-1 grounding WORKS at 2B,\n"
          "the base direction<->token mapping is learnable; build teacher/distill on top.")


if __name__ == "__main__":
    main()
