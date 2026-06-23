"""COUNTERFACTUAL-FLIP scorecard (EXP-046's 95%-flip metric) at 2B.

The env-free override ceiling: given frame_i but the plan of a DIFFERENT-direction chunk j
(cf_dir != own_dir), does the trained student's action FLIP toward cf_dir, against the frame's
natural own_dir? This is the ONE capability that isn't redundant with the base DiT.

Uses /tmp/stage2_mm_cf_2b.pt (frame_i + plan_j hidden, cf_dir, own_dir). For each cf pair we run
CFG: v = v_uncond + w*(v_cond - v_uncond) at several w, where cond = the cf plan tokens, uncond =
masked-null. We then read the stick (dims 21:22 jL_x, and jL_y) and check whether it moved toward
cf_dir. Flip-rate = fraction of pairs whose dominant stick direction == cf_dir.

CORRECT layout: jL_x=21, jL_y=22 (in [0,1], 0.5 neutral). left=x<0.5, right=x>0.5, up=y<0.5, down=y>0.5.

Run: PYTHONPATH=. .venv/bin/python planner_poc/cf_flip_scorecard.py runs/<ckpt>.pt [N]
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 240
CF = os.environ.get("CF", "/tmp/stage2_mm_cf_2b.pt")
WEIGHTS = [float(x) for x in os.environ.get("W", "1,4,8,12").split(",")]
device = "cuda"; K = 8; H = 18
JLX, JLY = 21, 22


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    ng = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
    CC = CkptConfig.model_validate(ng["ckpt_config"])
    mc = CC.model_cfg.model_copy(deep=True)
    mc.planner_cfg.enabled = True; mc.planner_cfg.num_plan_tokens = K
    mc.planner_cfg.null_mode = "masked"
    qk = "plan_head.resampler.queries"
    if qk in sd:
        mc.planner_cfg.backbone_hidden_size = int(sd[qk].shape[-1])
        mc.planner_cfg.num_chunks = int(sd[qk].shape[0]) // K
    lk = [k for k in sd if k.endswith(".lora_A")]
    if lk:
        mc.lora_dit_rank = int(sd[lk[0]].shape[0])
    m = NitroGen(config=mc, game_mapping=None)
    m.load_state_dict(sd, strict=False)
    print(f"loaded {path} | bdim={mc.planner_cfg.backbone_hidden_size} A={mc.planner_cfg.num_chunks}")
    return m.to(device).eval()


def dir_of(jlx, jly):
    # jlx,jly in [0,1], 0.5 neutral
    dx, dy = jlx - 0.5, jly - 0.5
    if max(abs(dx), abs(dy)) < 0.03:
        return "idle"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def main():
    m = load(CKPT)
    cf = torch.load(CF, map_location="cpu", weights_only=False)
    tok = NitrogenTokenizer(NitrogenTokenizerConfig(
        training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
    items = [(u, e) for u, e in cf.items()]
    import random; random.seed(0); random.shuffle(items)
    items = items[:N]
    print(f"cf-flip scorecard on {len(items)} cross-pairs | weights={WEIGHTS}")

    # build a dummy frame token layout once (we need vl/sa token ids + a blank image)
    # We bypass the image: cf hidden already encodes the frame via the VLM; the DiT still needs a
    # context frame though. Use a gray frame -> the OVERRIDE must come from the plan tokens (the
    # honest test of token authority, frame-independent). This matches the gold-token POC setup.
    gray = np.full((256, 256, 3), 127, dtype=np.uint8)
    from transformers import AutoImageProcessor
    ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
    px = ip([gray], return_tensors="pt")["pixel_values"][0].numpy()

    flips = {w: defaultdict(lambda: [0, 0]) for w in WEIGHTS}  # w -> cf_dir -> [flipped, total]
    for u, e in items:
        h = torch.as_tensor(np.asarray(e["h"])).float()
        kpm = torch.as_tensor(np.asarray(e["mask"])).bool()
        cf_dir = e["cf_dir"]
        ex = tok.encode({"frames": px[None], "dropped_frames": np.zeros((1,), bool),
                         "buttons": np.zeros((1, H, 21), np.float32),
                         "j_left": np.zeros((1, H, 2), np.float32),
                         "j_right": np.zeros((1, H, 2), np.float32)})
        d = {k: torch.as_tensor(np.asarray(v))[None].to(device) for k, v in ex.items()
             if k in ("images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask")}
        d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
        d["game_ids"] = None
        d["plan_hidden"] = h[None].to(device)
        d["plan_key_padding_mask"] = kpm[None].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vis = m.encode_images(d["images"])
            dc = dict(d); dc["plan_dropped"] = torch.tensor([False], device=device)
            du = dict(d); du["plan_dropped"] = torch.tensor([True], device=device)
            pt_c, pdp_c = m.compute_plan_tokens(dc)
            pt_u, pdp_u = m.compute_plan_tokens(du)
            vlm_c = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp_c)
            vlm_u = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp_u)

            def vel(acts, tb, pt, vlm):
                af = m.action_encoder(acts.to(vis.dtype), tb, d["embodiment_id"])
                vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, af,
                                              d["dropped_images"], game_ids=d["game_ids"], plan_tokens=pt)
                vl = m.vl_self_attention_model(vl, attention_mask=m._additive_key_mask(vlm, vl.dtype))
                mo = m.model(hidden_states=sa, encoder_hidden_states=vl,
                             encoder_attention_mask=vlm, timestep=tb)
                return m.action_decoder(mo, d["embodiment_id"])[:, -H:].float()

            ns = m.num_inference_timesteps; dt = 1.0 / ns
            for w in WEIGHTS:
                torch.manual_seed(0)
                a = torch.randn(1, H, m.config.action_dim, device=device)
                for i in range(ns):
                    tb = torch.tensor([int((i / ns) * m.num_timestep_buckets)], device=device)
                    if w == 1.0:
                        a = a + dt * vel(a, tb, pt_c, vlm_c)
                    else:
                        vc = vel(a, tb, pt_c, vlm_c); vu = vel(a, tb, pt_u, vlm_u)
                        a = a + dt * (vu + w * (vc - vu))
                ag = a[0].float().cpu().numpy().mean(0)
                got = dir_of(ag[JLX], ag[JLY])
                flips[w][cf_dir][1] += 1
                flips[w][cf_dir][0] += int(got == cf_dir)

    print("\n=== COUNTERFACTUAL FLIP-RATE (pred stick dir == cf_dir) ===")
    print(f"  {'w':>3} | {'left':>10} {'right':>10} {'up':>10} {'down':>10} | {'ALL':>8}")
    for w in WEIGHTS:
        cells = []
        tot_f = tot_n = 0
        for d_ in ("left", "right", "up", "down"):
            f, n = flips[w][d_]
            tot_f += f; tot_n += n
            cells.append(f"{(f/n if n else 0):.2f}({n})")
        print(f"  {w:>3} | " + " ".join(f"{c:>10}" for c in cells) +
              f" | {tot_f/max(tot_n,1):.3f}")
    print("\nBaseline ref: EXP-046 0.8B distilled = 0%@w1 -> 85%@w8 -> 95%@w12 (in-dist).")


if __name__ == "__main__":
    main()
