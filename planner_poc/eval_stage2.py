"""Stage-2 KEY EVAL (user's alignment/redundancy concern): does conditioning on the
VLM tactical plan predict the streamer's REAL action better than the NULL plan (frame
alone)? If the plan is redundant with the frame (the streamer already reacts to what they
see, and the plan is derived from that), the plan adds nothing -> equal MSE. If the plan
carries non-redundant intent, plan-conditioned MSE < null MSE.

Metric: flow-matching velocity MSE to the REAL action chunk, averaged over flow timesteps
and held-out chunks, for (a) plan-conditioned vs (b) null. Lower = better action
prediction. Also reports the per-chunk win rate (plan better than null).
"""
import glob
import json
import os
import sys
import numpy as np
import torch
REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/planner_poc")
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
from nitrogen.training.actions import load_chunk_actions, assemble_chunk

device = "cuda"; K = 8; H = 18
ck = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=f"{REPO}/ckpts/qwen35-0.8b")); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
LOOKUP = json.load(open("/tmp/stage2_plan_lookup.json"))
TS = [100, 300, 500, 700, 900]  # flow timesteps (buckets) to average MSE over
SEED0 = 0


def load(path, which):
    sd = torch.load(path, map_location="cpu", weights_only=False)[which]
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"
    # Auto-detect LoRA rank from the checkpoint so wrapped Linears (to_q.base.weight +
    # to_q.lora_A/lora_B) are rebuilt; otherwise lora weights are silently dropped and the
    # eval would test the frozen DiT WITHOUT the trained LoRA delta.
    lora_keys = [k for k in sd if k.endswith(".lora_A")]
    if lora_keys:
        mc.lora_dit_rank = int(sd[lora_keys[0]].shape[0])
    m = NitroGen(config=mc, game_mapping=None)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, unexp[:5]
    return m.to(device).eval()


def real_target(real_chunk):
    a = np.concatenate([(real_chunk["j_left"] + 1) / 2, (real_chunk["j_right"] + 1) / 2,
                        real_chunk["buttons"]], axis=-1).astype(np.float32)  # (H,25)
    return torch.from_numpy(a).to(device)


def vel_mse(m, png, plan_text, dropped, target, t_bucket, seed):
    """Velocity-prediction MSE to the REAL action at one flow timestep (teacher-forced):
    noisy = (1-t)eps + t*a; predict velocity; compare to (a - eps)."""
    fr = np.asarray(Image.open(png).convert("RGB"))
    pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
         for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
    d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(plan_text if plan_text else ".")
    d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device)
    d["plan_dropped"] = torch.tensor([dropped], dtype=torch.bool, device=device)
    a = target.unsqueeze(0)  # (1,H,25)
    g = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(a.shape, generator=g, device=device)
    tval = t_bucket / 1000.0
    noisy = (1 - tval) * eps + tval * a
    vel = a - eps
    tb = torch.tensor([t_bucket], device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vis = m.encode_images(d["images"]); pt, pdp = m.compute_plan_tokens(d)
        af = m.action_encoder(noisy, tb, d["embodiment_id"])
        vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, af,
                                      d["dropped_images"], game_ids=d["game_ids"], plan_tokens=pt)
        vlm = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp)
        vl = m.vl_self_attention_model(vl, attention_mask=m._additive_key_mask(vlm, vl.dtype))
        mo = m.model(hidden_states=sa, encoder_hidden_states=vl, encoder_attention_mask=vlm, timestep=tb)
        pred = m.action_decoder(mo, d["embodiment_id"])[:, -H:].float()
    return float(((pred - vel) ** 2).mean().cpu())


def main(ckpt, which="model_ema"):
    m = load(ckpt, which)
    mds = sorted(glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))
    items = []
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
        items.append((uuid, png, LOOKUP[uuid]["plan"], rc, m_.get("game", "?")))
    items = items[-60:]
    # swapped plan: each chunk gets ANOTHER chunk's plan (deterministic shift) — a WRONG
    # but valid tactical plan. If own_plan fits the real action better than swapped, the
    # plan content shapes the action (specificity) -> OOD/counterfactual planning feasible.
    swapped = [items[(i + 17) % len(items)][2] for i in range(len(items))]
    print(f"eval on {len(items)} chunks (null / own-plan / swapped-plan; lower MSE=better)\n")
    plan_mse = null_mse = swap_mse = 0.0; n = 0; win_null = win_swap = 0
    for i, (uuid, png, plan_text, rc, game) in enumerate(items):
        tgt = real_target(rc)
        pm = nm = sm = 0.0
        for t in TS:
            pm += vel_mse(m, png, plan_text, False, tgt, t, SEED0)
            nm += vel_mse(m, png, "", True, tgt, t, SEED0)
            sm += vel_mse(m, png, swapped[i], False, tgt, t, SEED0)
        pm /= len(TS); nm /= len(TS); sm /= len(TS)
        plan_mse += pm; null_mse += nm; swap_mse += sm; n += 1
        win_null += (pm < nm); win_swap += (pm < sm)
    print(f"=== Stage-2: velocity MSE to the REAL action (lower=better) ===")
    print(f"  null (frame alone)   : {null_mse/n:.4f}")
    print(f"  OWN VLM plan         : {plan_mse/n:.4f}")
    print(f"  SWAPPED (wrong) plan : {swap_mse/n:.4f}")
    print(f"\n  own vs null : {100*(null_mse-plan_mse)/null_mse:+.1f}%  (own better on {win_null}/{n})  [steering/informativeness]")
    print(f"  own vs swap : {100*(swap_mse-plan_mse)/swap_mse:+.1f}%  (own better on {win_swap}/{n})  [SPECIFICITY -> OOD planning]")
    print("\n  KEY: own << swapped => the plan CONTENT shapes the action; a different plan")
    print("       would give different actions => counterfactual/OOD planning is feasible")
    print("       WITHOUT env training. (own ~= null is FINE — redundant-but-informative.)")
    del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_vlm/plan_stage1_2000.pt"
    which = sys.argv[2] if len(sys.argv) > 2 else "model_ema"
    main(ckpt, which)
