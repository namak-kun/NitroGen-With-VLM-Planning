"""Evaluate 4-segment (SEQ4) within-chunk ordering. Extends eval_seq3_temporal to
quarters of the 18-action chunk (~4-5 steps each). For a plan "go a, then b, then c,
then d", each quarter should steer toward a/b/c/d on the relevant stick axis.

Primary purpose: test COMPOSITIONAL GENERALIZATION. The current best checkpoints were
trained on SEQ2 (halves) + SEQ3 (thirds) only. SEQ4 plans are OUT OF DISTRIBUTION both
in text ("...then d") and in segment count. If routing generalizes zero-shot, the
position-embedding mechanism is compositional; if not, we need to add SEQ4 to training.

Quarters with H=18: round(0.25*18)=4, round(0.5*18)=9, round(0.75*18)=14 (banker's)
-> segments (0,4),(4,9),(9,14),(14,18) = lengths 4,5,5,4.
"""
import os, sys, glob
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

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22; TS = [200, 500, 800]
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[-12:]
fn = torch.tensor(np.random.RandomState(1).randn(1, H, 25), dtype=torch.float32, device=device)

# axis/sign per cardinal: (axis_index, want_sign)
AX = {"left": (JLX, -1), "right": (JLX, +1), "up": (JLY, -1), "down": (JLY, +1)}
# SEQ4 plans: rotate through all 4 cardinals so consecutive segments alternate axes.
SEQ4 = {
    "left,up,right,down": ("go left, then up, then right, then down", ["left", "up", "right", "down"]),
    "up,right,down,left": ("go up, then right, then down, then left", ["up", "right", "down", "left"]),
    "down,left,up,right": ("go down, then left, then up, then right", ["down", "left", "up", "right"]),
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
    quarters = [(0, 4), (4, 9), (9, 14), (14, 18)]
    for name, (text, dirs) in SEQ4.items():
        seg = np.zeros(4)
        for png in pngs:
            for t in TS:
                dv = vel(m, png, text, False, t) - vel(m, png, "", True, t)
                for i, (lo, hi) in enumerate(quarters):
                    ax, _ = AX[dirs[i]]
                    seg[i] += dv[lo:hi, ax].mean()
        seg /= (len(pngs) * len(TS))
        oks = [int(np.sign(seg[i]) == AX[dirs[i]][1]) for i in range(4)]
        msg = "  ".join(f"{dirs[i]}:{seg[i]:+.3f}({'+' if AX[dirs[i]][1]>0 else '-'}){'OK' if oks[i] else 'x'}" for i in range(4))
        print(f"  {name:20s} {msg}   [{sum(oks)}/4]")
    del m; torch.cuda.empty_cache()


runs = [
    ("SEQ-pertoken frozen", "runs/seq_pertoken/plan_stage1_1500.pt", "model_ema"),
    ("SEQ-flatten frozen", "runs/seq_flatten/plan_stage1_1500.pt", "model_ema"),
    ("best_large (ref)", "runs/best_large/plan_stage1_1500.pt", "model_ema"),
]
for tag, ckp, which in runs:
    full = os.path.join(REPO, ckp)
    if os.path.exists(full):
        evaluate(tag, full, which)
    else:
        print(f"\n(missing {ckp})")
