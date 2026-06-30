"""Evaluate the CROSS-CHUNK cursor mechanism (K*A plan tokens + per-chunk cursor).

A cross-chunk plan like "go left, then right" must produce DIFFERENT whole-chunk
behavior depending on the per-chunk CURSOR: cursor 0 -> hold left, cursor 1 -> hold
right, from the SAME plan text and SAME frame. This proves the resampler's A blocks
specialize and the cursor selects the right one. (Within-chunk SEQ routed via action-
position embeddings; this routes via the cursor across chunks.)

Method: build the model with num_chunks=A, sample the chunk under the plan with each
cursor and under null, and check the WHOLE-CHUNK stick-axis delta (plan-null) matches
that cursor's intended direction.
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
from nitrogen.training.plans import CROSS_CHUNK_PLANS, CC_PLAN_BY_NAME

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22; A = 2
STRIDE = 2; CC_STARTS = [100, 250, 400]
ck = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b"))); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[-8:]
# Real per-chunk frame episodes: (uuid_prefix_path, start) that have both chunk frames.
import re
_cc = glob.glob("/tmp/frames_cc/*.png")
_byuuid = {}
for p in _cc:
    mobj = re.match(r"(.+)__(\d+)\.png$", os.path.basename(p))
    if mobj:
        _byuuid.setdefault(mobj.group(1), set()).add(int(mobj.group(2)))
REAL_EPISODES = []  # (uuid, start) with frames at start + a*H*stride for all a
for uuid, offs in _byuuid.items():
    for s in CC_STARTS:
        if all((s + a * H * STRIDE) in offs for a in range(A)):
            REAL_EPISODES.append((uuid, s))
REAL_EPISODES = sorted(REAL_EPISODES)[:8]
N_SEED = 3
AX = {"left": (JLX, -1), "right": (JLX, +1), "up": (JLY, -1), "down": (JLY, +1)}


def _real_frame(uuid, start, cursor):
    fidx = start + cursor * H * STRIDE
    return f"/tmp/frames_cc/{uuid}__{fidx}.png"


def load(path, which, lora_rank=0, num_chunks=A):
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"
    mc.planner_cfg.num_chunks = num_chunks; mc.lora_dit_rank = lora_rank
    m = NitroGen(config=mc, game_mapping=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)[which] if path else ck["model"]
    m.load_state_dict(sd, strict=False)
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


def evaluate(tag, path, which, lora_rank=0, pool="hold", real_frames=False):
    from nitrogen.training.plans import CROSS_CHUNK_PLANS, NESTED_CHUNK_PLANS
    plans = NESTED_CHUNK_PLANS if pool == "nested" else CROSS_CHUNK_PLANS
    m = load(path, which, lora_rank)
    src = "REAL per-chunk frames" if real_frames else "shared frame"
    print(f"\n=== {tag}  [{pool}, {src}] ===")
    total = ok_n = 0
    for plan in plans:
        text = plan.phrasings[0]
        parts = []
        for a in range(plan.num_chunks):
            label = plan.chunk_labels[a]
            # frame source(s) for this cursor
            if real_frames:
                frames = [(_real_frame(u, s, a)) for (u, s) in REAL_EPISODES]
            else:
                frames = pngs
            # accumulate per-step delta over frames x seeds
            acc = np.zeros((H, 25)); n = 0
            for fp in frames:
                for sd in range(N_SEED):
                    acc += sample(m, fp, text, False, a, sd) - sample(m, fp, "", True, a, sd); n += 1
            dv = acc / n
            if label.startswith("dir_"):  # whole-chunk hold
                cdir = label[4:]; axis, want = AX[cdir]
                v = dv[:, axis].mean(); good = int(np.sign(v) == want)
                ok_n += good; total += 1
                parts.append(f"c{a}:{cdir}:{v:+.2f}{'OK' if good else 'x'}")
            else:  # seq_X_Y within-chunk
                _, x, y = label.split("_")
                ax1, w1 = AX[x]; ax2, w2 = AX[y]
                v1 = dv[:9, ax1].mean(); v2 = dv[9:, ax2].mean()
                g1 = int(np.sign(v1) == w1); g2 = int(np.sign(v2) == w2)
                ok_n += g1 + g2; total += 2
                parts.append(f"c{a}:{x}{v1:+.2f}{'OK' if g1 else 'x'}>{y}{v2:+.2f}{'OK' if g2 else 'x'}")
        print(f"  {plan.name:26s} {'  '.join(parts)}")
    print(f"  TOTAL: {ok_n}/{total}")
    del m; torch.cuda.empty_cache()


runs = [
    ("cc_posthoc [synth retained]", "runs/cc_posthoc/plan_stage1_2000.pt", "model_ema", 0, "hold", True),
]
for tag, ckp, which, lr, pool, real in runs:
    full = os.path.join(REPO, ckp)
    if os.path.exists(full):
        evaluate(tag, full, which, lora_rank=lr, pool=pool, real_frames=real)
    else:
        print(f"\n(missing {ckp})")
