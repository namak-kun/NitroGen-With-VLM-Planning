"""Stage-2 DIRECTION-SPECIFICITY eval (EXP-043): the velocity-MSE eval (eval_stage2.py)
averages over all 25 action dims x 18 steps, so direction steering (just the 2 left-stick
dims) is drowned out -> own==swapped even when the plan DOES steer. This eval measures the
axis we actually trained: does conditioning on a REAL VLM plan whose action-outcome is
direction d steer the SAMPLED left-stick toward d, ON A FIXED FRAME (so any steering is
plan-content-driven, not frame-driven)? If a left-cluster plan -> left stick and a
right-cluster plan -> right stick on the SAME frame, the plan content causally overrides the
frame => counterfactual/OOD planning is feasible without env training.

Clusters = the 225 Stage-2 plans grouped by their real chunk's dominant direction
(chunk_dominant_dir) -- the same labels EXP-043 used for outcome-contrastive.
"""
import glob
import json
import os
import sys
import numpy as np
import torch
REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from PIL import Image
from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22
AX = {"left": (JLX, -1), "right": (JLX, +1), "up": (JLY, -1), "down": (JLY, +1)}
ck = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=f"{REPO}/ckpts/qwen35-0.8b")); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
LOOKUP = json.load(open("/tmp/stage2_plan_lookup.json"))


def load(path, which):
    sd = torch.load(path, map_location="cpu", weights_only=False)[which]
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"
    lk = [k for k in sd if k.endswith(".lora_A")]
    if lk:
        mc.lora_dit_rank = int(sd[lk[0]].shape[0])
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()


def sample(m, png, text, dropped, seed):
    fr = np.asarray(Image.open(png).convert("RGB"))
    pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
         for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
    d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else ".")
    d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device)
    d["plan_dropped"] = torch.tensor([dropped], dtype=torch.bool, device=device)
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m.get_action(d)["action_tensor"][0].float().cpu().numpy()


def build_clusters():
    """{dir: [plan_text,...]} from the Stage-2 lookup, grouped by real-chunk dominant dir."""
    clusters = {"left": [], "right": [], "up": [], "down": []}
    frames = {}
    mds = sorted(glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))
    for md in mds:
        m_ = json.load(open(md)); uuid = m_["uuid"]
        png = f"/tmp/frames_pre/{uuid}.png"
        if uuid not in LOOKUP or not os.path.exists(png):
            continue
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 303, H, 2) \
            or assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 3, H, 2)
        if rc is None:
            continue
        dd = chunk_dominant_dir(rc)
        if dd in clusters:
            clusters[dd].append(LOOKUP[uuid]["plan"])
            frames[uuid] = png
    return clusters, list(frames.values())


def main(ckpt, which="model"):
    m = load(ckpt, which)
    clusters, all_frames = build_clusters()
    print("cluster sizes:", {k: len(v) for k, v in clusters.items()})
    # Fixed reference frames (steering must be plan-driven, not frame-driven). Use a handful
    # of distinct frames; average the (plan - null) stick delta over frames x plans x seeds.
    ref_frames = all_frames[:16]
    n_plan = 6; n_seed = 3
    rng = np.random.RandomState(0)
    print(f"\nref frames: {len(ref_frames)}  plans/dir: {n_plan}  seeds: {n_seed}")
    print("does a direction-d cluster plan steer the sampled left-stick toward d? (same frames)\n")
    correct = total = 0
    for d, (axis, want) in AX.items():
        pool = clusters[d]
        if not pool:
            continue
        plans = [pool[i] for i in rng.choice(len(pool), min(n_plan, len(pool)), replace=False)]
        deltas = []
        for fp in ref_frames:
            for pt in plans:
                for sd in range(n_seed):
                    dv = sample(m, fp, pt, False, sd) - sample(m, fp, "", True, sd)
                    deltas.append(dv[:, axis].mean())
        deltas = np.array(deltas)
        v = float(deltas.mean()); sem = float(deltas.std() / np.sqrt(len(deltas)))
        sign_acc = float(np.mean(np.sign(deltas) == want))
        good = int(np.sign(v) == want)
        sig = abs(v) > 2 * sem  # mean is >2 SEM from zero (rough p<0.05)
        correct += good; total += 1
        print(f"  dir={d:5s}: delta {v:+.4f} +/-{sem:.4f}  sign-acc {sign_acc:.0%}  "
              f"want {want:+d} -> {'OK' if good else 'MISS'}{' *sig' if sig and good else ''}")
    print(f"\n  direction-specific steering: {correct}/{total} dirs correct")
    print("  (4/4 => real VLM plans causally steer by CONTENT on a fixed frame => the plan")
    print("   overrides the frame => counterfactual/OOD planning feasible w/o env training.)")
    # Cross-direction contrast on ONE frame: left-cluster vs right-cluster stick-x must FLIP.
    fp = ref_frames[0]
    def cluster_mean_x(d):
        ps = clusters[d][:n_plan]
        vals = [(sample(m, fp, p, False, s) - sample(m, fp, "", True, s))[:, JLX].mean()
                for p in ps for s in range(n_seed)]
        return float(np.mean(vals))
    lx, rx = cluster_mean_x("left"), cluster_mean_x("right")
    print(f"\n  single-frame flip test (frame={os.path.basename(fp)}):")
    print(f"    left-cluster stick-x  = {lx:+.4f}   right-cluster stick-x = {rx:+.4f}")
    print(f"    -> {'FLIPS (left<right): content overrides the frame' if lx < rx else 'NO FLIP'}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_con/plan_stage1_2500.pt"
    which = sys.argv[2] if len(sys.argv) > 2 else "model"
    main(ckpt, which)
