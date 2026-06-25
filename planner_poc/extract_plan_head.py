"""extract_plan_head.py -- slim a full NitroGen stage-2 checkpoint down to just its TRAINABLE weights
(the plan head + any LoRA keys) so the research checkpoints are tiny + portable for handoff.

A full stage-2 .pt bundles the frozen 601M NitroGen model (vision + DiT) twice (model + model_ema) =>
~2.8 GB. Only `plan_head.*` (and `*lora*` keys if LoRA was used) are trained. The frozen base is
reproducible: `hf download nvidia/NitroGen ng.pt`. So we keep only the trainable delta.

To RELOAD for eval: build the NitroGen model, load the base ng.pt, then
    model.load_state_dict(slim["trainable_ema"], strict=False)   # (or trainable_model)
(use the EMA weights for eval, matching how stage-2 evaluated).

Usage:
  python extract_plan_head.py runs/stage2_2b_btn/plan_stage1_600.pt out/ckpt_slim/btn_s600.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def _trainable(sd: dict) -> dict:
    return {k: v for k, v in sd.items() if k.startswith("plan_head") or "lora" in k.lower()}


def main():
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Usage: extract_plan_head.py <full_checkpoint.pt> <slim_out.pt>")
        sys.exit(0 if (len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help")) else 2)
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    model = ck.get("model", {})
    ema = ck.get("model_ema", model)
    # EMA weights are what stage-2 evaluated with. Keep ONLY those, cast to fp16, for a compact +
    # portable handoff artifact. (Full-precision / non-EMA weights for resuming training remain in
    # the original runs/<exp>/plan_stage1_*.pt on the training machine.)
    tr_ema = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in _trainable(ema).items()}
    slim = {
        "trainable_ema": tr_ema,
        "trainable_keys": sorted(tr_ema.keys()),
        "ckpt_config": ck.get("ckpt_config"),
        "step": ck.get("step"),
        "source": src,
        "dtype": "fp16",
        "note": "EMA plan_head (+lora) only, fp16. Reload: build the NitroGen model, load base "
                "nvidia/NitroGen ng.pt, then model.load_state_dict(slim['trainable_ema'], strict=False). "
                "For full-precision or resume-training weights use the original runs/ checkpoint.",
    }
    out = Path(dst); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)
    nbytes = out.stat().st_size
    nparams = sum(v.numel() for v in tr_ema.values() if hasattr(v, "numel"))
    print(f"{src} -> {dst}  ({len(tr_ema)} tensors, {nparams/1e6:.2f}M params, "
          f"{nbytes/1e6:.1f} MB, step={ck.get('step')})")


if __name__ == "__main__":
    main()
