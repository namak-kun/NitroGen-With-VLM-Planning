"""Evaluate HETEROGENEOUS 2-segment plans that mix MODALITIES in sequence, e.g.
"go left then jump": first half steers the stick (continuous dims 21-24), second half
presses a button (discrete dims 0-20). This is the hardest routing test so far: the
two segments live in DIFFERENT parts of the 25-dim action vector, so the model can't
reuse a single "stick axis" — it must route plan-segment-i to the right region AND the
right modality.

Zero-shot context: the best checkpoints trained on (a) dir-only SEQ (halves/thirds) and
(b) standalone button plans where the press was always at the chunk START (0-0.15) then
free. They NEVER saw a button in the 2nd half, nor dir+button in sequence. So this evals
compositional generalization ACROSS modality, not just more same-type segments.

Action layout (25 dims): buttons[0:21], j_left[21:23], j_right[23:25].
Buttons of interest: south(jump)=18, west(attack)=20. Stick: x=21, y=22.
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
SOUTH, WEST = 18, 20
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[-12:]
fn = torch.tensor(np.random.RandomState(1).randn(1, H, 25), dtype=torch.float32, device=device)

# A "probe" describes how to score one segment: (kind, dim, want_sign).
#   dir   : check sign(mean dv over dim) == want_sign  (stick axis 21/22)
#   button: check mean dv over button dim > 0          (pressed more than null)
DIR = {"left": ("dir", JLX, -1), "right": ("dir", JLX, +1),
       "up": ("dir", JLY, -1), "down": ("dir", JLY, +1)}
BTN = {"jump": ("button", SOUTH, +1), "attack": ("button", WEST, +1)}
PROBE = {**DIR, **BTN}

# Heterogeneous plans: one stick segment + one button segment, both orders.
HET = {
    "left,jump":   ("go left, then jump",   ["left", "jump"]),
    "jump,left":   ("jump, then go left",   ["jump", "left"]),
    "right,attack":("go right, then attack",["right", "attack"]),
    "attack,right":("attack, then go right",["attack", "right"]),
    "up,jump":     ("go up, then jump",     ["up", "jump"]),
    "jump,down":   ("jump, then go down",   ["jump", "down"]),
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
    halves = [(0, 9), (9, 18)]
    for name, (text, segs) in HET.items():
        val = np.zeros(2)
        for png in pngs:
            for t in TS:
                dv = vel(m, png, text, False, t) - vel(m, png, "", True, t)
                for i, (lo, hi) in enumerate(halves):
                    _, dim, _ = PROBE[segs[i]]
                    val[i] += dv[lo:hi, dim].mean()
        val /= (len(pngs) * len(TS))
        oks = []
        parts = []
        for i in range(2):
            kind, dim, want = PROBE[segs[i]]
            ok = int(np.sign(val[i]) == want)
            oks.append(ok)
            sgn = "+" if want > 0 else "-"
            parts.append(f"{segs[i]}:{val[i]:+.3f}({sgn}){'OK' if ok else 'x'}")
        print(f"  {name:14s} {'  '.join(parts)}   [{sum(oks)}/2]")
    del m; torch.cuda.empty_cache()


runs = [
    ("HET-flatten (trained)", "runs/het_flatten/plan_stage1_1500.pt", "model_ema"),
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
