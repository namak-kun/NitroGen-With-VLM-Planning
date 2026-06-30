"""Eval A=4 cross-chunk cursor scaling: one plan spans 4 chunks (rotating 4 cardinals);
cursor 0..3 must select 4 distinct whole-chunk directions. Tests whether the resampler's
K*A=32 blocks specialize and the cursor scales beyond A=2. Shared-frame (synthetic forcing
is frame-independent).
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
from nitrogen.training.plans import CROSS_CHUNK_PLANS_A4

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22; A = 4
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[-8:]
N_SEED = 3
AX = {"left": (JLX, -1), "right": (JLX, +1), "up": (JLY, -1), "down": (JLY, +1)}


def load(path, which, self_attn=False):
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"; mc.planner_cfg.num_chunks = A
    mc.planner_cfg.resampler_query_self_attn = self_attn
    m = NitroGen(config=mc, game_mapping=None)
    miss, unexp = m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)[which], strict=False)
    sa_bad = [k for k in (list(miss) + list(unexp)) if "self_attn" in k or "ln_sa" in k]
    if sa_bad:
        print(f"  WARN self-attn mismatch: {sa_bad[:2]}")
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


def evaluate(tag, path, which, self_attn=False):
    m = load(path, which, self_attn)
    print(f"\n=== {tag} (A=4) ===")
    total = okn = 0; allsplit = []
    for plan in CROSS_CHUNK_PLANS_A4:
        text = plan.phrasings[0]; parts = []
        for a in range(plan.num_chunks):
            cdir = plan.chunk_labels[a][4:]; axis, want = AX[cdir]
            acc = 0.0; n = 0
            for png in pngs:
                for s in range(N_SEED):
                    acc += (sample(m, png, text, False, a, s) - sample(m, png, "", True, a, s))[:, axis].mean(); n += 1
            v = acc / n; good = int(np.sign(v) == want); okn += good; total += 1; allsplit.append(abs(v))
            parts.append(f"c{a}:{cdir}{v:+.2f}{'OK' if good else 'x'}")
        print(f"  {plan.name:26s} {'  '.join(parts)}")
    print(f"  TOTAL: {okn}/{total} | mean|delta|={np.mean(allsplit):.3f}")
    del m; torch.cuda.empty_cache()


evaluate("CC4 attn-OFF", os.path.join(REPO, "runs/attn_cc4_off/plan_stage1_2000.pt"), "model_ema", False)
evaluate("CC4 attn-ON ", os.path.join(REPO, "runs/attn_cc4_on/plan_stage1_2000.pt"), "model_ema", True)
