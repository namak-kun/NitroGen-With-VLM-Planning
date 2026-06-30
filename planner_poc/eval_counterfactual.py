"""Counterfactual evaluation for plan-conditioned NitroGen (Stage 1).

Reports the two signals @namak-kun requires:
  (1) STEERING: for each directional plan, does conditioning move the policy in the
      plan's direction? Measured on the CORRECT dims (j_left x/y at 21/22, and the
      relevant d-pad button dims), as Δ(plan − null).
  (2) NULL-INVARIANCE: under the null plan, is the policy ≈ the pretrained base
      NitroGen? Measured as ‖action(null) − action(base)‖ (should be small).

Compares the trained checkpoint (raw + EMA) against the untrained baseline.
"""
import os, sys, glob, json
import numpy as np
import torch

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO)
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda self: self)

from PIL import Image
from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache
from nitrogen.training.actions import DATASET_COL_TO_MODEL_DIM

device = "cuda"
K = 8
# action dim indices
JLX, JLY, JRX, JRY = 21, 22, 23, 24
DP = {d: DATASET_COL_TO_MODEL_DIM[f"dpad_{d}"] for d in ["left", "right", "up", "down"]}

# Directional plans -> (expected stick delta sign on (x,y), dpad dims that should rise)
# stick: negative x = left, negative y = up (raw); in packed [0,1], center 0.5.
DIR_PLANS = {
    "go left":  {"text": "go left, move to the left side", "stick": (-1, 0), "dpad": ["left"]},
    "go right": {"text": "go right, move to the right side", "stick": (1, 0), "dpad": ["right"]},
    "go up":    {"text": "go up, move forward", "stick": (0, -1), "dpad": ["up"]},
    "go down":  {"text": "go down, move back", "stick": (0, 1), "dpad": ["down"]},
}


def build_model(load_path=None, which="model"):
    ckpt = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    cc.model_cfg.planner_cfg.enabled = True
    cc.model_cfg.planner_cfg.num_plan_tokens = K
    model = NitroGen(config=cc.model_cfg, game_mapping=None)
    if load_path:
        sd = torch.load(load_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd[which], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device).eval(), cc


def base_model():
    ckpt = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    cc.model_cfg.planner_cfg.enabled = False
    m = NitroGen(config=cc.model_cfg, game_mapping=None)
    m.load_state_dict(ckpt["model"], strict=False)
    return m.to(device).eval()


def main():
    img_proc = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
    planner = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b")))
    planner.load()
    cache = PlanHiddenCache(planner, device)
    tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K,
                                                    action_horizon=18, max_sequence_length=256 + K))
    tok_base = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=0,
                                                         action_horizon=18, max_sequence_length=256))
    pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[:10]

    def data_for(png, with_plan_slots):
        frame = np.asarray(Image.open(png).convert("RGB"))
        pv = img_proc([frame], return_tensors="pt")["pixel_values"][0].numpy()
        t = tok if with_plan_slots else tok_base
        ex = t.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
        d = {}
        for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]:
            d[k] = torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
        d["images"] = d["images"].float()
        d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
        d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
        return d

    def act(model, png, plan_text=None, dropped=False, seed=0):
        d = data_for(png, with_plan_slots=(plan_text is not None or model.planner_cfg.enabled))
        if model.planner_cfg.enabled:
            txt = plan_text if plan_text is not None else "."
            h, kpm = cache.get(txt)
            d["plan_hidden"] = h.unsqueeze(0).to(device)
            d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device)
            d["plan_dropped"] = torch.tensor([dropped], dtype=torch.bool, device=device)
        torch.manual_seed(seed)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model.get_action(d)["action_tensor"][0].float().cpu().numpy()  # (18,25)

    base = base_model()

    def evaluate(tag, load_path, which):
        model, _ = build_model(load_path, which)
        # ---- null-invariance: null-plan vs pretrained base, same frame/seed ----
        inv = []
        for png in pngs:
            a_null = act(model, png, plan_text="", dropped=True, seed=1)
            a_base = act(base, png, seed=1)
            inv.append(np.linalg.norm(a_null - a_base))
        inv = float(np.mean(inv))
        # ---- steering: per plan, Δ(plan − null) on the plan's expected dims ----
        print(f"\n=== {tag} ===")
        print(f"  null-invariance ‖null − base‖ = {inv:.3f}  (lower = base policy preserved)")
        for name, spec in DIR_PLANS.items():
            d_stick = np.zeros(2); d_dpad = 0.0
            for png in pngs:
                a_plan = act(model, png, plan_text=spec["text"], dropped=False, seed=2)
                a_null = act(model, png, plan_text="", dropped=True, seed=2)
                # stick delta over first 6 steps (packed; left=0,center=.5,right=1)
                d_stick += (a_plan[:6, [JLX, JLY]] - a_null[:6, [JLX, JLY]]).mean(0)
                # dpad delta over first 6 steps
                for dd in spec["dpad"]:
                    d_dpad += (a_plan[:6, DP[dd]] - a_null[:6, DP[dd]]).mean()
            d_stick /= len(pngs); d_dpad /= len(pngs)
            sx, sy = spec["stick"]
            # expected: stick moves in sign(sx,sy); dpad rises (>0)
            stick_proj = d_stick[0] * sx + d_stick[1] * sy  # >0 means moved the right way
            print(f"  {name:9s}: stick Δ=({d_stick[0]:+.3f},{d_stick[1]:+.3f}) "
                  f"proj={stick_proj:+.3f}  dpadΔ={d_dpad:+.3f}  "
                  f"{'OK' if stick_proj > 0.01 or d_dpad > 0.01 else '~'}")
        del model
        torch.cuda.empty_cache()

    evaluate("BASELINE (untrained plan head)", None, "model")
    ck = os.path.join(REPO, "runs/stage1_v2/plan_stage1_400.pt")
    if os.path.exists(ck):
        evaluate("TRAINED raw (stage1_v2 400)", ck, "model")
        evaluate("TRAINED EMA (stage1_v2 400)", ck, "model_ema")
    else:
        print("\n(no trained checkpoint yet at", ck, ")")


if __name__ == "__main__":
    main()
