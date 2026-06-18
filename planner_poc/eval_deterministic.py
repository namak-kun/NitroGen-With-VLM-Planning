"""Deterministic counterfactual eval for plan-conditioned NitroGen (Stage 1).

Unlike sampling-based eval, this fixes the flow-matching noise AND timestep and
reads the model's predicted VELOCITY field directly, so plan vs null vs base are
compared at the *same* point with no sampling noise. Two signals:

  (1) STEERING: for plan P, predicted velocity should push the action toward P.
      We integrate one Euler step from a fixed start and read the resulting
      structure on the correct dims (j_left 21/22; relevant d-pad dims).
  (2) NULL-INVARIANCE: null-plan velocity ≈ pretrained-base velocity (same noise,
      same timestep). Reported as mean ‖v_null − v_base‖ over dims & frames.

We evaluate at several fixed timesteps and average. Reports schedule-aware
steering: for HOLD we check the whole chunk; for SEQ we check first-half vs
second-half move in the right directions.
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
from nitrogen.training.actions import DATASET_COL_TO_MODEL_DIM

device = "cuda"
K = 8
H = 18
JLX, JLY = 21, 22
DP = {d: DATASET_COL_TO_MODEL_DIM[f"dpad_{d}"] for d in ["left", "right", "up", "down"]}
TIMESTEPS = [100, 300, 500, 700, 900]  # fixed discrete timesteps to average over

# plan text -> expected direction vector (dx,dy) and dpad name, per half if seq
PROBES = {
    "hold_left":  {"text": "keep going left", "whole": ((-1, 0), "left")},
    "hold_right": {"text": "keep going right", "whole": ((1, 0), "right")},
    "hold_up":    {"text": "keep going up", "whole": ((0, -1), "up")},
    "hold_down":  {"text": "keep going down", "whole": ((0, 1), "down")},
    "seq_left_right": {"text": "go left then right",
                       "first": ((-1, 0), "left"), "second": ((1, 0), "right")},
    "seq_up_down":    {"text": "go up then down",
                       "first": ((0, -1), "up"), "second": ((0, 1), "down")},
}


def load(planner_enabled, path=None, which="model", null_mode="learned"):
    ckpt = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    cc.model_cfg.planner_cfg.enabled = planner_enabled
    cc.model_cfg.planner_cfg.num_plan_tokens = K
    cc.model_cfg.planner_cfg.null_mode = null_mode
    m = NitroGen(config=cc.model_cfg, game_mapping=None)
    sd = torch.load(path, map_location="cpu", weights_only=False)[which] if path else ckpt["model"]
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()


def main():
    img_proc = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
    planner = PlanEncoder(PlannerConfig(backbone_name_or_path=os.path.join(REPO, "ckpts/qwen35-0.8b")))
    planner.load()
    cache = PlanHiddenCache(planner, device)
    tokK = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K,
                                                     action_horizon=H, max_sequence_length=256 + K))
    tok0 = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=0,
                                                     action_horizon=H, max_sequence_length=256))
    pngs = sorted(glob.glob("/tmp/frames_pre/*.png"))[:12]

    # Precompute a FIXED noisy action + per-frame visual context.
    rng = np.random.RandomState(0)
    fixed_noise = torch.tensor(rng.randn(1, H, 25), dtype=torch.float32, device=device)

    def encode(model, png, with_slots):
        frame = np.asarray(Image.open(png).convert("RGB"))
        pv = img_proc([frame], return_tensors="pt")["pixel_values"][0].numpy()
        tk = tokK if with_slots else tok0
        ex = tk.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
        d = {}
        for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]:
            d[k] = torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
        d["images"] = d["images"].float()
        d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
        d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
        return d

    @torch.no_grad()
    def velocity(model, png, plan_text, dropped, tdisc):
        """Predicted velocity (H,25) at fixed noise & timestep tdisc."""
        d = encode(model, png, with_slots=model.planner_cfg.enabled)
        if model.planner_cfg.enabled:
            h, kpm = cache.get(plan_text if plan_text else ".")
            d["plan_hidden"] = h.unsqueeze(0).to(device)
            d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device)
            d["plan_dropped"] = torch.tensor([dropped], dtype=torch.bool, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vis = model.encode_images(d["images"])  # (B, num_frames, 256, hidden)
            if model.planner_cfg.enabled:
                plan_tokens, plan_dropped = model.compute_plan_tokens(d)
            else:
                plan_tokens, plan_dropped = None, None
            vl, sa = model.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis,
                                              model.action_encoder(
                                                  fixed_noise,
                                                  torch.tensor([tdisc], device=device),
                                                  d["embodiment_id"]),
                                              d["dropped_images"],
                                              game_ids=d["game_ids"], plan_tokens=plan_tokens)
            vlm = model.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], plan_dropped) \
                if model.planner_cfg.enabled else d["vl_attn_mask"]
            vl_self_mask = model._additive_key_mask(vlm, vl.dtype)
            vl = model.vl_self_attention_model(vl, attention_mask=vl_self_mask)
            mo = model.model(hidden_states=sa, encoder_hidden_states=vl,
                             encoder_attention_mask=vlm,
                             timestep=torch.tensor([tdisc], device=device))
            v = model.action_decoder(mo, d["embodiment_id"])[:, -H:]
        return v[0].float().cpu().numpy()

    base = load(False)

    def steer_metric(model, png, plan_text, region, expect):
        """Mean over timesteps of (v_plan - v_null) projected onto expected dir."""
        (dx, dy), dname = expect
        lo, hi = region
        proj_s = 0.0; dpad = 0.0
        for t in TIMESTEPS:
            vp = velocity(model, png, plan_text, False, t)
            vn = velocity(model, png, "", True, t)
            dv = vp - vn
            proj_s += (dv[lo:hi, JLX] * dx + dv[lo:hi, JLY] * dy).mean()
            dpad += dv[lo:hi, DP[dname]].mean()
        return proj_s / len(TIMESTEPS), dpad / len(TIMESTEPS)

    def null_invariance(model, ref):
        """Mean ‖v_null(model) − v_null(ref)‖. ref=base measures absolute drift;
        ref=untrained (same arch) isolates *training* drift (fair metric)."""
        tot = 0.0; n = 0
        for png in pngs:
            for t in TIMESTEPS:
                vn = velocity(model, png, "", True, t)
                vr = (velocity(ref, png, None, False, t) if not ref.planner_cfg.enabled
                      else velocity(ref, png, "", True, t))
                tot += np.linalg.norm(vn - vr); n += 1
        return tot / n

    def evaluate(tag, path, which, null_mode, untrained_ref):
        model = load(True, path, which, null_mode=null_mode)
        inv_base = null_invariance(model, base)
        inv_fair = null_invariance(model, untrained_ref)
        print(f"\n=== {tag} ===")
        print(f"  null-invariance vs base       ‖v_null − v_base‖     = {inv_base:.4f}")
        print(f"  null-invariance vs untrained  ‖v_null − v_untrained‖ = {inv_fair:.4f}  (isolates training drift)")
        for name, spec in PROBES.items():
            if "whole" in spec:
                s, dpad = np.mean([steer_metric(model, p, spec["text"], (0, H), spec["whole"]) for p in pngs], 0)
                ok = "OK" if (s > 0.002 or dpad > 0.01) else "~"
                print(f"  {name:16s}: whole  stick_proj={s:+.4f} dpadΔ={dpad:+.4f}  {ok}")
            else:
                s1, d1 = np.mean([steer_metric(model, p, spec["text"], (0, H//2), spec["first"]) for p in pngs], 0)
                s2, d2 = np.mean([steer_metric(model, p, spec["text"], (H//2, H), spec["second"]) for p in pngs], 0)
                ok1 = "OK" if (s1 > 0.002 or d1 > 0.01) else "~"
                ok2 = "OK" if (s2 > 0.002 or d2 > 0.01) else "~"
                print(f"  {name:16s}: 1st proj={s1:+.4f} dpad={d1:+.4f} {ok1} | "
                      f"2nd proj={s2:+.4f} dpad={d2:+.4f} {ok2}")
        del model; torch.cuda.empty_cache()

    # A/B comparison of null modes (each vs its own untrained reference)
    runs = [("learned", "runs/ab_learned/plan_stage1_600.pt"),
            ("masked",  "runs/ab_masked/plan_stage1_600.pt")]
    for nm, ckp in runs:
        full = os.path.join(REPO, ckp)
        if not os.path.exists(full):
            print(f"\n(missing {ckp})"); continue
        untrained = load(True, None, "model", null_mode=nm)
        evaluate(f"NULL_MODE={nm}  ({os.path.basename(ckp)} EMA)", full, "model_ema", nm, untrained)
        del untrained; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
