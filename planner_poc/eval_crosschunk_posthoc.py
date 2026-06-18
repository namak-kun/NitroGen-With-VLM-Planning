"""R0 post-hoc eval. On REAL episodes where chunk0 and chunk1 have DISTINCT dominant
directions (d0 != d1), give the post-hoc plan "go d0, then d1" and a FIXED frame
(chunk0's real frame). Measure the sampled output direction under cursor0 vs cursor1:
  - cursor0 should lean d0 (agrees with the frame),
  - cursor1 should lean d1 (the plan's 2nd chunk) DESPITE the chunk0 frame.
If cursor1 produces d1 != d0 on the same frame, the cursor+plan is routing (not the
frame) -> the post-hoc-trained model follows the plan causally on real data.

Also reports null (no plan) direction as the frame-only baseline.
"""
import os, sys, glob, json
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
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22; A = 2; STRIDE = 2; SHIFT = 3
CC_STARTS = [100, 250, 400]
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
N_SEED = 4
# (axis, +sign meaning) -> for measuring which cardinal the output leans
AXVAL = {"left": (JLX, -1), "right": (JLX, +1), "up": (JLY, -1), "down": (JLY, +1)}


def out_dir(vec25):
    """Classify a sampled chunk's mean stick into a cardinal (matches chunk_dominant_dir
    axes; sampler output stick is in [0,1] with 0.5 neutral, so center it)."""
    x = vec25[:, JLX].mean() - 0.5; y = vec25[:, JLY].mean() - 0.5
    if max(abs(x), abs(y)) < 0.02:
        return "none", x, y
    if abs(x) >= abs(y):
        return ("right" if x > 0 else "left"), x, y
    return ("down" if y > 0 else "up"), x, y


def find_episodes(n=16):
    dirs = sorted(os.path.dirname(md) for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))
    eps = []
    for d in dirs:
        meta = json.load(open(os.path.join(d, "metadata.json"))); uuid = meta["uuid"]
        pq = os.path.join(d, "actions_processed.parquet")
        if not os.path.exists(pq):
            continue
        a = load_chunk_actions(pq); T = a["buttons"].shape[0]
        for s in CC_STARTS:
            f0 = f"/tmp/frames_cc/{uuid}__{s}.png"
            if s + H * STRIDE + H * STRIDE + SHIFT >= T or not os.path.exists(f0):
                continue
            d0 = chunk_dominant_dir(assemble_chunk(a["buttons"], a["j_left"], a["j_right"], s + SHIFT, H, STRIDE) or {})
            rc1 = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], s + H * STRIDE + SHIFT, H, STRIDE)
            d1 = chunk_dominant_dir(rc1) if rc1 is not None else None
            if d0 and d1 and d0 != d1:
                eps.append((f0, d0, d1))
            if len(eps) >= n:
                return eps
    return eps


def load(path, which):
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"; mc.planner_cfg.num_chunks = A
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)[which], strict=False)
    return m.to(device).eval()


def sample(m, png, text, dr, cursor, seed):
    fr = np.asarray(Image.open(png).convert("RGB")); pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device) for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device); d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else "."); d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device)
    d["plan_dropped"] = torch.tensor([dr], dtype=torch.bool, device=device)
    d["plan_cursor"] = torch.tensor([cursor], dtype=torch.long, device=device)
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return m.get_action(d)["action_tensor"][0].float().cpu().numpy()


def mean_out(m, png, text, dr, cursor):
    acc = np.zeros((H, 25))
    for s in range(N_SEED):
        acc += sample(m, png, text, dr, cursor, s)
    return acc / N_SEED


def evaluate(tag, path, which):
    m = load(path, which)
    eps = find_episodes(16)
    print(f"\n=== {tag}  ({len(eps)} distinct-dir episodes, FIXED chunk0 frame) ===")
    c0_ok = c1_ok = both = 0
    for (png, d0, d1) in eps:
        text = f"go {d0}, then {d1}"
        o0 = out_dir(mean_out(m, png, text, False, 0))[0]
        o1 = out_dir(mean_out(m, png, text, False, 1))[0]
        on = out_dir(mean_out(m, png, "", True, 0))[0]
        g0 = (o0 == d0); g1 = (o1 == d1)
        c0_ok += g0; c1_ok += g1; both += (g0 and g1)
        print(f"  plan[{d0:5s}->{d1:5s}]  null:{on:5s}  c0:{o0:5s}{'OK' if g0 else 'x'}  c1:{o1:5s}{'OK' if g1 else 'x'}")
    n = len(eps)
    print(f"  cursor0->d0: {c0_ok}/{n} | cursor1->d1: {c1_ok}/{n} | both: {both}/{n}")
    del m; torch.cuda.empty_cache()


runs = [
    ("cc_posthoc", "runs/cc_posthoc/plan_stage1_2000.pt", "model_ema"),
]
for tag, ckp, which in runs:
    full = os.path.join(REPO, ckp)
    if os.path.exists(full):
        evaluate(tag, full, which)
    else:
        print(f"(missing {ckp})")
