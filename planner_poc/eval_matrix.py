"""Compare overnight experiment-matrix checkpoints on the EXP-007 metrics.

For each run: (1) left/right & up/down plan-token cosine (collinearity — lower is
better, <1 means separated); (2) stick steering direction-specificity (does Δv flip
sign between opposite plans?); (3) null-invariance vs base. Deterministic (fixed
noise+timestep).
"""
import os, sys, glob
import numpy as np
import torch
from numpy.linalg import norm

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

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22
TS = [200, 500, 800]
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])

ipx = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[:10]
fixed_noise = torch.tensor(np.random.RandomState(0).randn(1, H, 25), dtype=torch.float32, device=device)
DIRS = {"left": ("keep going left", (-1, 0)), "right": ("keep going right", (1, 0)),
        "up": ("keep going up", (0, -1)), "down": ("keep going down", (0, 1))}


def load(path, which, null_mode="masked"):
    mc = CC.model_cfg.model_copy(deep=True)
    mc.planner_cfg.enabled = True; mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = null_mode
    m = NitroGen(config=mc, game_mapping=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)[which] if path else ck["model"]
    m.load_state_dict(sd, strict=False); return m.to(device).eval()


def plan_tok(m, text):
    h, kpm = cache.get(text)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pt, _ = m.plan_head(h.unsqueeze(0).to(device), key_padding_mask=kpm.unsqueeze(0).to(device), dropped=None)
    return pt.float().mean(1).cpu().numpy().ravel()


def velocity(m, png, text, dropped, t):
    fr = np.asarray(Image.open(png).convert("RGB")); pv = ipx([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device) for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device); d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else "."); d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device)
    d["plan_dropped"] = torch.tensor([dropped], dtype=torch.bool, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vis = m.encode_images(d["images"]); pt, pdp = m.compute_plan_tokens(d)
        vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, m.action_encoder(fixed_noise, torch.tensor([t], device=device), d["embodiment_id"]), d["dropped_images"], game_ids=d["game_ids"], plan_tokens=pt)
        vlm = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp); sm = m._additive_key_mask(vlm, vl.dtype)
        vl = m.vl_self_attention_model(vl, attention_mask=sm)
        mo = m.model(hidden_states=sa, encoder_hidden_states=vl, encoder_attention_mask=vlm, timestep=torch.tensor([t], device=device))
        return m.action_decoder(mo, d["embodiment_id"])[0, -H:].float().cpu().numpy()


def base_vel(png, t):
    return velocity(base, png, "", True, t) if base.planner_cfg.enabled else None


def cos(a, b): return float(a @ b / (norm(a) * norm(b)))


def evaluate(tag, path, which, null_mode):
    m = load(path, which, null_mode)
    # separability
    tl, tr = plan_tok(m, DIRS["left"][0]), plan_tok(m, DIRS["right"][0])
    tu, td = plan_tok(m, DIRS["up"][0]), plan_tok(m, DIRS["down"][0])
    clr, cud = cos(tl, tr), cos(tu, td)
    # steering: per-direction raw stick Δv (plan − null), averaged
    dv = {}
    for name, (text, (dx, dy)) in DIRS.items():
        ax = ay = 0
        for png in pngs:
            for t in TS:
                d = velocity(m, png, text, False, t) - velocity(m, png, "", True, t)
                ax += d[:, JLX].mean(); ay += d[:, JLY].mean()
        n = len(pngs) * len(TS); dv[name] = (ax / n, ay / n)
    # direction-specificity: does x flip sign left vs right, y flip up vs down?
    x_split = dv["right"][0] - dv["left"][0]   # want > 0 (right more +x than left)
    y_split = dv["down"][1] - dv["up"][1]       # want > 0
    # null-invariance vs base (masked → should be ~weight drift)
    inv = np.mean([norm(velocity(m, p, "", True, t) - base_vel(p, t)) for p in pngs for t in TS])
    print(f"\n=== {tag} ===")
    print(f"  cos(L,R)={clr:.3f} cos(U,D)={cud:.3f}  (lower=better separated)")
    print(f"  stick Δv: L=({dv['left'][0]:+.3f},{dv['left'][1]:+.3f}) R=({dv['right'][0]:+.3f},{dv['right'][1]:+.3f}) "
          f"U=({dv['up'][0]:+.3f},{dv['up'][1]:+.3f}) D=({dv['down'][0]:+.3f},{dv['down'][1]:+.3f})")
    print(f"  DIRECTION-SPECIFICITY: x_split(R−L)={x_split:+.3f} y_split(D−U)={y_split:+.3f}  (>0 = direction-specific!)")
    print(f"  null-invariance vs base = {inv:.3f}")
    del m; torch.cuda.empty_cache()


base = load(None, "model", "masked"); base.planner_cfg.enabled = True  # masked base for null ref
runs = [
    ("SEQ-flatten frozen",    "runs/seq_flatten/plan_stage1_1500.pt", "masked"),
    ("SEQ-pertoken frozen",   "runs/seq_pertoken/plan_stage1_1500.pt", "masked"),
    ("best_large (ref)",      "runs/best_large/plan_stage1_1500.pt", "masked"),
]
# untrained reference separability
mu = load(None, "model", "masked")
tl, tr = plan_tok(mu, DIRS["left"][0]), plan_tok(mu, DIRS["right"][0])
print(f"UNTRAINED cos(L,R)={cos(tl,tr):.3f}")
del mu; torch.cuda.empty_cache()
for tag, ckp, nm in runs:
    full = os.path.join(REPO, ckp)
    if os.path.exists(full):
        evaluate(f"{tag} EMA", full, "model_ema", nm)
    else:
        print(f"\n(missing {ckp})")
