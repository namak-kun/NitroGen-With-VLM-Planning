"""Evaluate UNEVEN-duration plans (seqdur_*). Same two directions, but the plan TEXT
implies how long the first segment lasts ("briefly left then right" vs "left for a long
time then right"), and we measure WHERE the transition actually happens in the sampled
action chunk. Tests whether routing keys off plan CONTENT (duration words) or has a fixed
even 50/50 split prior.

Method: sample the 18-step chunk under the plan and under null; take the per-step stick-
axis delta dv (plan-null). For an a->b plan, dv goes from the a-sign early to the b-sign
late; the TRANSITION STEP is the zero-crossing. If the model reads duration, the
transition tracks the intended split (4/9/14 for 25/50/75%); if it has a fixed prior, all
three land near 9.

Run zero-shot first on seq_pertoken (trained ONLY on even 50/50 SEQ) to see if "briefly"/
"for a long time" are understood with no duration training.
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
# (a, b, axis): a-sign early, b-sign late. expected transitions at 25/50/75% = 4/9/14.
PAIRS = [("left", "right"), ("right", "left"), ("up", "down"), ("down", "up")]
SPLITS = [25, 50, 75]
EXPECT = {25: 4, 50: 9, 75: 14}


def load(path, which, null_mode="masked", lora_rank=0):
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = null_mode
    mc.lora_dit_rank = lora_rank
    m = NitroGen(config=mc, game_mapping=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)[which] if path else ck["model"]
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()


def sample(m, png, text, dr, seed):
    fr = np.asarray(Image.open(png).convert("RGB")); pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device) for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device); d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else "."); d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device); d["plan_dropped"] = torch.tensor([dr], dtype=torch.bool, device=device)
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m.get_action(d)["action_tensor"]
    return out[0].float().cpu().numpy()


def transition_step(profile, a_sign):
    """profile: (18,) per-step axis delta. a_sign = sign wanted in the FIRST segment.
    Find the boundary t in [1,17] that best splits a_sign-early from (-a_sign)-late, by
    maximizing (a_sign*mean(profile[:t]) - a_sign*mean(profile[t:])). Returns t."""
    best_t, best_s = 9, -1e9
    for t in range(1, H):
        early = a_sign * profile[:t].mean()
        late = -a_sign * profile[t:].mean()
        s = early + late
        if s > best_s:
            best_s, best_t = s, t
    return best_t


def evaluate(tag, path, which, lora_rank=0):
    m = load(path, which, lora_rank=lora_rank)
    print(f"\n=== {tag} ===")
    print(f"  {'plan':18s} {'25%':>10s} {'50%':>10s} {'75%':>10s}   monotonic?")
    from nitrogen.training.plans import PLAN_BY_NAME
    for a, b in PAIRS:
        axis, a_sign = AX[a]
        trans = {}
        for pct in SPLITS:
            name = f"seqdur_{a}_{b}_{pct}"
            text = PLAN_BY_NAME[name].phrasings[0]
            acc = np.zeros(H); n = 0
            for png in pngs:
                for s in range(N_SEED):
                    dv = sample(m, png, text, False, s) - sample(m, png, "", True, s)
                    acc += dv[:, axis]; n += 1
            trans[pct] = transition_step(acc / n, a_sign)
        mono = trans[25] <= trans[50] <= trans[75] and trans[25] < trans[75]
        cells = "  ".join(f"t={trans[p]:2d}(e{EXPECT[p]:2d})" for p in SPLITS)
        print(f"  {a+'->'+b:18s} {cells}   {'YES' if mono else 'no'}")
    del m; torch.cuda.empty_cache()


runs = [
    ("dur_lora (LoRA)", "runs/dur_lora/plan_stage1_1500.pt", "model_ema", 16),
    ("dur_pertoken (frozen)", "runs/dur_pertoken/plan_stage1_1500.pt", "model_ema", 0),
]
for tag, ckp, which, lr in runs:
    full = os.path.join(REPO, ckp)
    if os.path.exists(full):
        evaluate(tag, full, which, lora_rank=lr)
    else:
        print(f"\n(missing {ckp})")
