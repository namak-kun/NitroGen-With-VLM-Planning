"""Precompute TEACHER plan tokens for Stage-2 distillation (EXP-045).

The EXP-044 teacher (trained on the action-augmented prompt P+ = base plan + action summary)
steers the DiT 2-3x stronger than a base-plan student. Since the teacher is frozen and the
Stage-2 window is fixed per uuid (frame 303), its plan tokens are a deterministic function of
uuid -> precompute {uuid: (K, d)} once. The student (base plan P only) is then distilled
toward these targets (nitrogen.training: teacher_token_lookup + --distill-weight), recovering
the teacher's steering without seeing the actions at test time.
"""
import glob, json, os, sys
import numpy as np
import torch
REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, summarize_chunk

device = "cuda"; K = 8; H = 18
TEACHER = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_teacher/plan_stage1_2500.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stage2_teacher_tokens.pt"
ck = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=f"{REPO}/ckpts/qwen35-0.8b")); pl.load()
cache = PlanHiddenCache(pl, device)
LOOKUP = json.load(open(os.environ.get("LOOKUP", "/tmp/stage2_plan_lookup.json")))
ROOTS = os.environ.get("ROOTS", "/tmp/stage1_big").split(",")


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"
    lk = [k for k in sd if k.endswith(".lora_A")]
    if lk:
        mc.lora_dit_rank = int(sd[lk[0]].shape[0])
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()


def plan_tokens(m, text):
    h, kpm = cache.get(text if text else ".")
    d = {"plan_hidden": h.unsqueeze(0).to(device),
         "plan_key_padding_mask": kpm.unsqueeze(0).to(device),
         "plan_dropped": torch.tensor([False], device=device)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pt, _ = m.compute_plan_tokens(d)
    return pt[0].float().cpu()


def main():
    m = load(TEACHER)
    out = {}
    mds = []
    for root in ROOTS:
        mds += sorted(glob.glob(f"{root}/**/metadata.json", recursive=True))
    for md in mds:
        m_ = json.load(open(md)); uuid = m_["uuid"]
        if uuid not in LOOKUP:
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
        pplus = LOOKUP[uuid]["plan"] + " To do this I take the following actions: " \
            + summarize_chunk(rc) + "."
        out[uuid] = plan_tokens(m, pplus)
    torch.save(out, OUT)
    print(f"saved {len(out)} teacher token sets (K={K}, d={next(iter(out.values())).shape[-1]}) -> {OUT}")


if __name__ == "__main__":
    main()
