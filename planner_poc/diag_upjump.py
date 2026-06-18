"""Diagnostic for the up,jump holdout: measure FIRST-HALF direction steering for every
cardinal paired with each button (seqhet_{dir}_{btn}) and for pure holds, on het_lora_bal.
Prediction (from the data coupling button->down): UP is suppressed with BOTH buttons,
while left/right/down are fine. If up,attack also fails, it's a data-prior conflict, not
an up+jump-specific or capacity bug.
"""
import os, sys, glob
import numpy as np
import torch
REPO = "/home/t-nagupta/NitroGen"
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

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[-8:]
N_SEED = 4
AX = {"left": (JLX, -1), "right": (JLX, +1), "up": (JLY, -1), "down": (JLY, +1)}
CKPT, WHICH, LORA = "runs/het_lora_bal/plan_stage1_1500.pt", "model_ema", 16


def load():
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"; mc.lora_dit_rank = LORA
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location="cpu", weights_only=False)[WHICH], strict=False)
    return m.to(device).eval()


def sample(m, png, text, dr, seed):
    fr = np.asarray(Image.open(png).convert("RGB")); pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device) for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device); d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else "."); d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device); d["plan_dropped"] = torch.tensor([dr], dtype=torch.bool, device=device)
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return m.get_action(d)["action_tensor"][0].float().cpu().numpy()


def first_half_dir(m, text, d):
    axis, want = AX[d]
    acc = 0.0; n = 0
    for png in pngs:
        for s in range(N_SEED):
            dv = sample(m, png, text, False, s) - sample(m, png, "", True, s)
            acc += dv[:9, axis].mean(); n += 1
    v = acc / n
    return v, "OK" if np.sign(v) == want else "x"


m = load()
print("=== het_lora_bal: FIRST-HALF direction steering by (dir, button) ===")
print(f"  {'dir':6s} {'+jump':>14s} {'+attack':>14s} {'pure hold':>14s}")
for d in ["left", "right", "up", "down"]:
    vj, oj = first_half_dir(m, f"go {d}, then jump", d)
    va, oa = first_half_dir(m, f"go {d}, then attack", d)
    vh, oh = first_half_dir(m, f"keep going {d}", d)
    print(f"  {d:6s} {vj:+8.3f} {oj:>4s} {va:+8.3f} {oa:>4s} {vh:+8.3f} {oh:>4s}")
