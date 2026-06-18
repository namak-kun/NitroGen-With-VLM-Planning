"""End-to-end data-path test (stub frames; network-independent).

Exercises the full Stage-1 pipeline against REAL modules, substituting only the
network frame fetch with a random-RGB stub (YouTube is bot-blocked from this
host). Proves: chunk discovery -> frame processing -> action assembly ->
contrastive plan targets -> tokenization -> frozen-Qwen plan-hidden cache ->
collation -> NitroGen.forward + backward.

Run:
  cd /home/t-nagupta/NitroGen && source .venv/bin/activate && \
  python3 planner_poc/test_datapath.py
"""
import os, sys
import numpy as np
import torch

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO)

import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda self: self)

from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import (
    NitrogenPlanDataset, PlanDatasetConfig, PlanHiddenCache, make_collate_fn,
)

NG_CKPT = os.path.join(REPO, "ckpts/nitrogen/ng.pt")
SAMPLE = os.path.join(REPO, "planner_poc")  # contains sample_chunk/


def stub_frame_provider(meta, frame_idx):
    h, w = meta["original_video"]["resolution"]
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def main():
    device = "cuda"
    K = 8
    print("=" * 70)
    print("Data-path test (stub frames)")
    print("=" * 70)

    # Build a tiny dataset from the local sample chunk (duplicated to fake >1 example)
    import shutil, tempfile
    tmp = tempfile.mkdtemp()
    for i in range(4):
        dst = os.path.join(tmp, f"vid{i}", f"vid{i}_chunk_0")
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(os.path.join(SAMPLE, "sample_chunk")):
            shutil.copy(os.path.join(SAMPLE, "sample_chunk", f), os.path.join(dst, f))

    img_proc = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
    ds_cfg = PlanDatasetConfig(shard_roots=[tmp], num_plan_tokens=K,
                               plan_ratio=0.5, idle_only_for_plan=False, seed=1)
    ds = NitrogenPlanDataset(ds_cfg, stub_frame_provider, img_proc)
    print(f"[dataset] {len(ds)} chunks discovered")

    ex = ds[0]
    print(f"[example] vl_token_ids {np.asarray(ex['vl_token_ids']).shape}, "
          f"actions {np.asarray(ex['actions']).shape}, "
          f"plan={ex['plan_name']!r} dropped={ex['plan_dropped']} idle={ex['is_idle']}")

    # frozen Qwen plan-hidden cache
    planner = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b")))
    planner.load()
    cache = PlanHiddenCache(planner, device)
    collate = make_collate_fn(cache, plan_dim=1024)

    batch = collate([ds[i] for i in range(4)])
    print(f"[collate] images {tuple(batch['images'].shape)}, "
          f"plan_hidden {tuple(batch['plan_hidden'].shape)}, "
          f"plan_dropped {batch['plan_dropped'].tolist()}")

    # real model with planner enabled
    ckpt = torch.load(NG_CKPT, map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    cc.model_cfg.planner_cfg.enabled = True
    cc.model_cfg.planner_cfg.num_plan_tokens = K
    model = NitroGen(config=cc.model_cfg, game_mapping=None)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).train()

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    batch["images"] = batch["images"].float()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(batch)
    loss = out["loss"]
    print(f"[forward] loss = {loss.item():.4f}")
    assert torch.isfinite(loss)
    loss.backward()
    g = sum(p.grad.norm().item() for p in model.plan_head.parameters() if p.grad is not None)
    print(f"[backward] plan_head grad norm = {g:.3e}")

    shutil.rmtree(tmp)
    ok = torch.isfinite(loss) and g > 0
    print("=" * 70)
    print(f"OVERALL: {'*** DATA-PATH PASS ***' if ok else '!!! FAIL !!!'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
