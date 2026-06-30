"""merge_slim_to_full.py -- merge a slim handoff checkpoint (plan_head + LoRA trainable_ema delta)
onto the frozen base ng.pt and write a FULL checkpoint that NitroGenPolicy/eval scripts can load
directly (key "model").

The slim zips in ckpts/handoff_zips/ only carry the ~70 trainable tensors (EMA, fp16). The dev box
loaded them by overlaying on a freshly-built model; on a fresh box without runs/ full checkpoints, the
eval scripts (which do torch.load(path)["model"]) need a full state dict. This reconstructs it once.

Run:
  env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B \
    .venv/bin/python planner_poc/merge_slim_to_full.py ckpts/handoff_zips/btn_s600.pt ckpts/btn_s600_full.pt
"""
from __future__ import annotations

import os
import sys

import torch

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)

from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig


def main():
    slim_path = sys.argv[1] if len(sys.argv) > 1 else "ckpts/handoff_zips/btn_s600.pt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ckpts/btn_s600_full.pt"
    K = int(os.environ.get("K", "8"))
    ng_path = os.path.join(_R, "ckpts/nitrogen/ng.pt")

    base = torch.load(ng_path, map_location="cpu", weights_only=False)
    slim = torch.load(slim_path, map_location="cpu", weights_only=False)
    delta = slim["trainable_ema"]
    CC = CkptConfig.model_validate(base["ckpt_config"])

    mc = CC.model_cfg.model_copy(deep=True)
    mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K
    mc.planner_cfg.null_mode = "masked"
    qk = "plan_head.resampler.queries"
    if qk in delta:
        mc.planner_cfg.backbone_hidden_size = int(delta[qk].shape[-1])
        nq = int(delta[qk].shape[0])
        if nq % K == 0:
            mc.planner_cfg.num_chunks = nq // K
    lk = [k for k in delta if k.endswith(".lora_A")]
    if lk:
        mc.lora_dit_rank = int(delta[lk[0]].shape[0])
    if any(k.endswith("plan_head.adaln_proj.weight") for k in delta):
        mc.planner_cfg.plan_adaln = True

    print(f"building NitroGen: backbone_hidden={mc.planner_cfg.backbone_hidden_size} "
          f"num_chunks={mc.planner_cfg.num_chunks} lora_rank={mc.lora_dit_rank} K={K}")
    m = NitroGen(config=mc, game_mapping=None)

    # 1) base weights (vision encoder also loads from pretrained inside __init__; ng.pt fills the rest).
    #    CRITICAL: when LoRA wraps a Linear, `to_q.weight` becomes `to_q.base.weight`. Remap the base
    #    state-dict keys so the 16 wrapped cross-attn projections keep their PRETRAINED weights
    #    (otherwise they load as random — same remap the trainer uses, train_planner.remap_for_lora).
    base_sd = base["model"] if "model" in base else base
    model_sd = m.state_dict()
    base_sd = dict(base_sd)
    for mk in model_sd:
        if ".base." in mk:
            src = mk.replace(".base.", ".")
            if src in base_sd:
                base_sd[mk] = base_sd.pop(src)
    miss_b, unexp_b = m.load_state_dict(base_sd, strict=False)
    # 2) overlay the trainable delta (fp16 EMA -> cast to model dtype)
    delta_cast = {k: v.float() for k, v in delta.items()}
    miss_a, unexp_a = m.load_state_dict(delta_cast, strict=False)

    # sanity: every delta key must have been consumed (not 'unexpected')
    unexpected_delta = [k for k in unexp_a if k in delta]
    print(f"base load (post-remap): missing={len(miss_b)} unexpected={len(unexp_b)}")
    # after remap, the only acceptable 'missing' are the trainable delta keys (filled in step 2)
    missing_nontrainable = [k for k in miss_b if k not in delta]
    print(f"  missing-but-not-in-delta={len(missing_nontrainable)} (should be 0)")
    if missing_nontrainable:
        print("  !! base weights left random:", missing_nontrainable[:8])
    if unexp_b:
        print("  !! base unexpected (unconsumed):", list(unexp_b)[:8])
    print(f"delta overlay: unexpected-delta-keys={len(unexpected_delta)} (should be 0)")
    if unexpected_delta:
        print("  !! unexpected delta keys:", unexpected_delta[:5])

    full_sd = {k: v.float().cpu() for k, v in m.state_dict().items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"model": full_sd, "ckpt_config": base["ckpt_config"],
                "merged_from": [os.path.basename(ng_path), os.path.basename(slim_path)],
                "K": K}, out_path)
    print(f"wrote FULL checkpoint -> {out_path} ({len(full_sd)} tensors)")


if __name__ == "__main__":
    main()
