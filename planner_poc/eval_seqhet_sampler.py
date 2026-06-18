"""Heterogeneous (dir+button) eval via the ACTUAL flow-matching sampler (get_action),
not the single-timestep velocity-delta proxy (which is unreliable for binary buttons).

For each plan we sample the full 18-step action chunk under the plan and under null, and
compare. Action layout (25 dims): buttons[0:21] (0/1), j_left[21:23], j_right[23:25],
where pack_actions normalizes sticks to [0,1] (0.5=neutral, 0=left/up, 1=right/down).

We score TEMPORAL LOCALIZATION, not just presence:
  - dir segment  : stick-axis delta (plan-null) in the segment's half has the right sign
  - button segment: the button-dim delta in the segment's half is positive AND larger
    than in the other half (the press is localized to the right half).
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
SOUTH, WEST = 18, 20
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[-8:]
N_SEED = 4  # average over sampler noise seeds for stability

# dir: (kind, dim, want_sign of delta);  button: (kind, dim, +1)
DIR = {"left": ("dir", JLX, -1), "right": ("dir", JLX, +1),
       "up": ("dir", JLY, -1), "down": ("dir", JLY, +1)}
BTN = {"jump": ("button", SOUTH, +1), "attack": ("button", WEST, +1)}
PROBE = {**DIR, **BTN}
HET = {
    "left,jump":    ("go left, then jump",    ["left", "jump"]),
    "jump,left":    ("jump, then go left",    ["jump", "left"]),
    "right,attack": ("go right, then attack", ["right", "attack"]),
    "attack,right": ("attack, then go right", ["attack", "right"]),
    "up,jump":      ("go up, then jump",      ["up", "jump"]),
    "jump,down":    ("jump, then go down",    ["jump", "down"]),
    "down,attack":  ("go down, then attack",  ["down", "attack"]),
    "up,attack":    ("go up, then attack",    ["up", "attack"]),
}


def load(path, which, null_mode="masked", lora_rank=0):
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = null_mode
    mc.lora_dit_rank = lora_rank
    m = NitroGen(config=mc, game_mapping=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)[which] if path else ck["model"]
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()


def base_actions(png, seeds=range(N_SEED)):
    """Base NitroGen (no planner) sampled actions, for null-invariance reference."""
    mc = CC.model_cfg.model_copy(deep=True)
    m = NitroGen(config=mc, game_mapping=None); m.load_state_dict(ck["model"], strict=False)
    m = m.to(device).eval()
    fr = np.asarray(Image.open(png).convert("RGB")); pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device) for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device); d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    outs = []
    for s in seeds:
        torch.manual_seed(s)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outs.append(m.get_action(d)["action_tensor"][0].float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return outs


def sample(m, png, text, dr, seed):
    fr = np.asarray(Image.open(png).convert("RGB")); pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device) for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device); d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else "."); d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device); d["plan_dropped"] = torch.tensor([dr], dtype=torch.bool, device=device)
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m.get_action(d)["action_tensor"]
    return out[0].float().cpu().numpy()  # (18, 25)


def evaluate(tag, path, which, null_mode="masked", lora_rank=0):
    m = load(path, which, null_mode, lora_rank)
    print(f"\n=== {tag} ===")
    # null-invariance: null-plan action vs base-model action (should be ~0 for frozen
    # DiT + masked null; LoRA perturbs the DiT for ALL examples so expect > 0).
    ni = 0.0; nin = 0
    for png in pngs[:4]:
        bases = base_actions(png)
        for s in range(N_SEED):
            nullact = sample(m, png, "", True, s)
            ni += float(np.abs(nullact - bases[s]).mean()); nin += 1
    print(f"  null-invariance |null - base| = {ni/nin:.4f}")
    halves = [(0, 9), (9, 18)]
    for name, (text, segs) in HET.items():
        # mean action delta (plan - null) over frames x seeds
        acc = np.zeros((H, 25)); n = 0
        for png in pngs:
            for s in range(N_SEED):
                acc += sample(m, png, text, False, s) - sample(m, png, "", True, s); n += 1
        dv = acc / n
        oks, parts = [], []
        for i in range(2):
            kind, dim, want = PROBE[segs[i]]
            lo, hi = halves[i]; olo, ohi = halves[1 - i]
            here = dv[lo:hi, dim].mean()
            if kind == "dir":
                ok = int(np.sign(here) == want)
                parts.append(f"{segs[i]}:{here:+.3f}({'+' if want>0 else '-'}){'OK' if ok else 'x'}")
            else:  # button: positive here AND more pressed here than other half (localized)
                there = dv[olo:ohi, dim].mean()
                ok = int(here > 0 and here > there)
                parts.append(f"{segs[i]}:{here:+.3f}>oth{there:+.3f}{'OK' if ok else 'x'}")
            oks.append(ok)
        print(f"  {name:14s} {'  '.join(parts)}   [{sum(oks)}/2]")
    del m; torch.cuda.empty_cache()


runs = [
    ("HET-LoRA r32 (2500)", "runs/het_lora_r32/plan_stage1_2500.pt", "model_ema", 32),
    ("HET-LoRA-bal r16", "runs/het_lora_bal/plan_stage1_1500.pt", "model_ema", 16),
]
for tag, ckp, which, lr in runs:
    full = os.path.join(REPO, ckp)
    if os.path.exists(full):
        evaluate(tag, full, which, lora_rank=lr)
    else:
        print(f"\n(missing {ckp})")
