"""Evaluate within-chunk TEMPORAL structure of SEQ plans.

HOLD plans steer the whole chunk one way (verified, EXP-009/010). SEQ plans
("go left then right") should steer the FIRST half one way and the SECOND half the
other. This tests whether the model learned *temporal* (intra-chunk) plan structure
— the "complex action" question — using the DiT's positional embeddings on the 18
action tokens.

Metric per SEQ plan: compare (v_plan - v_null) in the first-half vs second-half on
the relevant stick axis. A working SEQ shows opposite-signed deltas in the two
halves (e.g. left_right: 1st half -x, 2nd half +x).
"""
import os, sys, glob
import numpy as np
import torch
from numpy.linalg import norm

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

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22; TS = [200, 500, 800]
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[-12:]
fn = torch.tensor(np.random.RandomState(1).randn(1, H, 25), dtype=torch.float32, device=device)

# SEQ probes: text, axis (0=x/JLX, 1=y/JLY), expected (first_sign, second_sign)
SEQ = {
    "left->right": ("go left then right", "x", (-1, +1)),
    "right->left": ("go right then left", "x", (+1, -1)),
    "up->down":    ("go up then down", "y", (-1, +1)),
    "down->up":    ("go down then up", "y", (+1, -1)),
}


def load(path, which, null_mode="masked"):
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = null_mode
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)[which] if path else ck["model"], strict=False)
    return m.to(device).eval()


def vel(m, png, text, dr, t):
    fr = np.asarray(Image.open(png).convert("RGB")); pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device) for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device); d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else "."); d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device); d["plan_dropped"] = torch.tensor([dr], dtype=torch.bool, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vis = m.encode_images(d["images"]); pt, pdp = m.compute_plan_tokens(d)
        vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, m.action_encoder(fn, torch.tensor([t], device=device), d["embodiment_id"]), d["dropped_images"], game_ids=d["game_ids"], plan_tokens=pt)
        vlm = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp); sm = m._additive_key_mask(vlm, vl.dtype)
        vl = m.vl_self_attention_model(vl, attention_mask=sm)
        mo = m.model(hidden_states=sa, encoder_hidden_states=vl, encoder_attention_mask=vlm, timestep=torch.tensor([t], device=device))
        return m.action_decoder(mo, d["embodiment_id"])[0, -H:].float().cpu().numpy()


def evaluate(tag, path, which, null_mode="masked"):
    m = load(path, which, null_mode)
    print(f"\n=== {tag} ===")
    half = H // 2
    for name, (text, axis, (s1, s2)) in SEQ.items():
        ax = JLX if axis == "x" else JLY
        d1 = d2 = 0.0
        for png in pngs:
            for t in TS:
                dv = vel(m, png, text, False, t) - vel(m, png, "", True, t)
                d1 += dv[:half, ax].mean(); d2 += dv[half:, ax].mean()
        n = len(pngs) * len(TS); d1 /= n; d2 /= n
        # success if 1st half matches s1 sign and 2nd half matches s2 sign AND they differ
        ok = (np.sign(d1) == s1) and (np.sign(d2) == s2) and ((d2 - d1) * (s2 - s1) > 0)
        print(f"  {name:12s} ({axis}): 1st={d1:+.3f}(want {'+' if s1>0 else '-'}) "
              f"2nd={d2:+.3f}(want {'+' if s2>0 else '-'})  split={d2-d1:+.3f}  {'OK' if ok else '~'}")
    del m; torch.cuda.empty_cache()


runs = [
    ("SEQ-flatten frozen (order-aware con)", "runs/seq_flatten/plan_stage1_1500.pt", "model_ema", "masked"),
    ("SEQ-pertoken frozen (order-aware con)", "runs/seq_pertoken/plan_stage1_1500.pt", "model_ema", "masked"),
    ("SEQ-B frozen+SEQ-heavy (mean, ref)", "runs/seq_frozen/plan_stage1_1500.pt", "model_ema", "masked"),
    ("best_large (hold-heavy, ref)", "runs/best_large/plan_stage1_1500.pt", "model_ema", "masked"),
]
for tag, ckp, which, nm in runs:
    full = os.path.join(REPO, ckp)
    if os.path.exists(full):
        evaluate(tag, full, which, nm)
    else:
        print(f"\n(missing {ckp})")
