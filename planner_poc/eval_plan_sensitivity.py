"""Plan-sensitivity eval: does the Stage-1-trained plan channel steer actions?

For a set of real frames, compare the model's sampled action chunk under a
"go left" plan vs the null plan. The left stick x-axis (index: buttons(17)+0)
should become MORE negative (left) under the plan if alignment worked. We compare
the trained checkpoint against the untrained baseline (random-init plan head).
"""
import os, sys, glob, json
import numpy as np
import torch

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO)
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda self: self)

from transformers import AutoImageProcessor
from PIL import Image
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache
from nitrogen.training.actions import BUTTON_ORDER

K = 8
NB = len(BUTTON_ORDER)  # 17 -> j_left x is index NB+0
device = "cuda"


def build_model(planner_enabled=True, load_path=None, which="model"):
    ckpt = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    cc.model_cfg.planner_cfg.enabled = planner_enabled
    cc.model_cfg.planner_cfg.num_plan_tokens = K
    model = NitroGen(config=cc.model_cfg, game_mapping=None)
    if load_path:
        sd = torch.load(load_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd[which], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device).eval(), cc


def main():
    img_proc = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
    planner = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b")))
    planner.load()
    cache = PlanHiddenCache(planner, device)

    # a few real frames
    pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[:6]
    tok = NitrogenTokenizer(NitrogenTokenizerConfig(
        training=False, num_plan_tokens=K, action_horizon=18, max_sequence_length=256 + K))

    def make_data(png):
        frame = np.asarray(Image.open(png).convert("RGB"))
        pv = img_proc([frame], return_tensors="pt")["pixel_values"][0].numpy()
        ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
        d = {}
        for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]:
            d[k] = torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
        d["images"] = d["images"].float()
        d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
        d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
        return d

    h, kpm = cache.get("go left, move to the left side immediately")
    plan_hidden = h.unsqueeze(0).to(device)
    plan_kpm = kpm.unsqueeze(0).to(device)

    def left_push(model, png, dropped):
        d = make_data(png)
        d["plan_hidden"] = plan_hidden
        d["plan_key_padding_mask"] = plan_kpm
        d["plan_dropped"] = torch.tensor([dropped], dtype=torch.bool, device=device)
        torch.manual_seed(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.get_action(d)
        act = out["action_tensor"][0].float().cpu().numpy()  # (H, 25)
        return act

    def report(tag, load_path, which):
        model, _ = build_model(True, load_path, which=which)
        jl_deltas, l2 = [], []
        for png in pngs:
            a_plan = left_push(model, png, dropped=False)
            a_null = left_push(model, png, dropped=True)
            jl_deltas.append(a_plan[:6, NB + 0].mean() - a_null[:6, NB + 0].mean())
            l2.append(np.linalg.norm(a_plan - a_null))
        jl_deltas, l2 = np.array(jl_deltas), np.array(l2)
        print(f"{tag}:")
        print(f"    j_left_x[plan]-[null] mean = {jl_deltas.mean():+.4f}  (negative => steers LEFT; left=0.0,center=0.5)")
        print(f"    ||action_plan - action_null|| mean = {l2.mean():.4f}  (overall plan influence)")
        del model
        torch.cuda.empty_cache()

    report("BASELINE (untrained plan head)", None, "model")
    ckpt = os.path.join(REPO, "runs/stage1_real/plan_stage1_200.pt")
    report("TRAINED raw weights (stage1_200)", ckpt, "model")
    report("TRAINED EMA weights (stage1_200)", ckpt, "model_ema")


if __name__ == "__main__":
    main()
