"""Quick order-separation probe: cos(seq_left_right, seq_right_left) and cos(UD,DU) of the
final plan tokens (flattened). Low (~0.1-0.2) = orderings separated (good); high (~0.9-1.0)
= collapsed (the EXP-032 regression). Usage: python probe_order_sep.py <ckpt> [which] [self_attn]
"""
import os, sys, torch, numpy as np
sys.path.insert(0, "/home/t-nagupta/NitroGen")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache

dev = "cuda"; K = 8
ck = torch.load("/home/t-nagupta/NitroGen/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
pl = PlanEncoder(PlannerConfig(backbone_name_or_path="/home/t-nagupta/NitroGen/ckpts/qwen35-0.8b")); pl.load()
cache = PlanHiddenCache(pl, dev)


def toks(path, which="model_ema", sa=False):
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"
    mc.planner_cfg.resampler_query_self_attn = sa
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)[which], strict=False)
    m = m.to(dev).eval(); o = {}
    for t in ["go left then right", "go right then left", "go up then down", "go down then up"]:
        h, kpm = cache.get(t)
        d = {"plan_hidden": h.unsqueeze(0).to(dev), "plan_key_padding_mask": kpm.unsqueeze(0).to(dev),
             "plan_dropped": torch.tensor([False], device=dev)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pt, _ = m.compute_plan_tokens(d)
        o[t] = pt[0].float().cpu().numpy().flatten()
    del m; torch.cuda.empty_cache(); return o


def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


if __name__ == "__main__":
    path = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else "model_ema"
    sa = (len(sys.argv) > 3 and sys.argv[3] == "1")
    o = toks(path, which, sa)
    lr = cos(o["go left then right"], o["go right then left"])
    ud = cos(o["go up then down"], o["go down then up"])
    print(f"{os.path.basename(os.path.dirname(path))}: cos(LR,RL)={lr:.3f} cos(UD,DU)={ud:.3f} | {'SEPARATED' if max(lr,ud)<0.5 else 'COLLAPSED'}")
