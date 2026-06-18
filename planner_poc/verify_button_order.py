"""Empirically recover the button-column -> model-output-index permutation.

The dataset parquet names its 17 button columns; NitroGen's DiT outputs 17 button
dims in an INTERNAL order that is NOT documented. If our assumed
`actions.BUTTON_ORDER` is wrong, training null-plan examples teaches a permuted
action space and corrupts the policy.

Method (no new downloads; uses already-cached video slices + shard parquets):
  1. Match each cached slice (.mp4 in frame_cache) to its chunk's parquet via the
     same hash key VideoFrameFetcher uses.
  2. For each chunk, extract the frame at t and read the GROUND-TRUTH action a few
     frames later (action_shift), giving per-frame true button states (by name).
  3. Run the pretrained NitroGen (null plan / no planner) -> predicted 25-d action;
     buttons are dims [0:17] in NitroGen's internal order.
  4. Accumulate correlation between predicted_button_dim[k] and true_button[name].
     A near-diagonal matrix under our BUTTON_ORDER confirms it; otherwise the
     argmax per row reveals the true permutation.
"""
import os, sys, glob, json, hashlib
import numpy as np
import torch

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO)
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda self: self)

import polars as pl
from PIL import Image
from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig

# The 17 dataset button columns (alphabetical-ish, as in the parquet/README list).
DATASET_BUTTONS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
    "right_thumb", "right_trigger", "south", "start", "west",
]
NB = 17
device = "cuda"
CACHE = os.path.join(REPO, "frame_cache")
SHARD = "/tmp/ds_real/SHARD_0000"


def cache_path(vid, start, end):
    key = f"{vid}_{start:.2f}_{end:.2f}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(CACHE, f"{vid}_{h}.mp4")


def extract_frame(path, t):
    import subprocess, io
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path,
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.asarray(Image.open(io.BytesIO(p.stdout)).convert("RGB"))


def main():
    # Match cached slices to parquets
    matches = []  # (parquet_path, metadata, cache_mp4)
    for md_path in glob.glob(os.path.join(SHARD, "**", "metadata.json"), recursive=True):
        md = json.load(open(md_path))
        ov = md["original_video"]
        cp = cache_path(ov["video_id"], float(ov["start_time"]), float(ov["end_time"]))
        if os.path.exists(cp) and os.path.getsize(cp) > 1000:
            pq = os.path.join(os.path.dirname(md_path), "actions_processed.parquet")
            if os.path.exists(pq):
                matches.append((pq, md, cp))
    print(f"matched {len(matches)} cached chunks to parquets")
    if len(matches) < 5:
        print("not enough cached chunks; run a fetch first."); return 1

    # Load pretrained NitroGen (no planner, original behavior)
    ckpt = torch.load(os.path.join(REPO, "ckpts/nitrogen/ng.pt"), map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    model = NitroGen(config=cc.model_cfg, game_mapping=None)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    H = cc.model_cfg.action_horizon
    img_proc = AutoImageProcessor.from_pretrained(cc.model_cfg.vision_encoder_name)
    tok = NitrogenTokenizer(NitrogenTokenizerConfig(
        training=False, action_horizon=H, max_sequence_length=256))

    # Accumulate sums for correlation between pred dim k and true button name j
    # Use multiple sample times per slice to get button variety.
    sample_times = [1.0, 4.0, 7.0, 10.0, 13.0, 16.0]
    pred_all, true_all = [], []
    pred_joy, true_joy = [], []
    n_used = 0
    for pq, md, cp in matches[:60]:
        df = pl.read_parquet(pq)
        true_btn = {b: (df[b].to_numpy() if b in df.columns else np.zeros(df.height)) for b in DATASET_BUTTONS}
        jl = np.asarray(df["j_left"].to_list(), dtype=np.float32)
        jr = np.asarray(df["j_right"].to_list(), dtype=np.float32)
        for t in sample_times:
            frame = extract_frame(cp, t)
            if frame is None:
                continue
            fidx = int(round(t * 60))  # 60 fps within the slice (slice starts at chunk start)
            fidx = min(fidx + 3, df.height - 1)  # action_shift~3
            true_vec = np.array([true_btn[b][fidx] for b in DATASET_BUTTONS], dtype=np.float32)
            true_j = np.array([jl[fidx, 0], jl[fidx, 1], jr[fidx, 0], jr[fidx, 1]], dtype=np.float32)
            pv = img_proc([frame], return_tensors="pt")["pixel_values"][0].numpy()
            ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
            d = {}
            for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]:
                d[k] = torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
            d["images"] = d["images"].float()
            d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
            d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                act = model.get_action(d)["action_tensor"][0].float().cpu().numpy()  # (H,25)
            pred_btn = act[:3, :NB].mean(0)  # first 3 steps, button dims (NitroGen order)
            pred_all.append(pred_btn)
            true_all.append(true_vec)
            pred_joy.append(act[:3, NB:NB + 4].mean(0))  # packed j_left xy, j_right xy
            true_joy.append(true_j)
            n_used += 1
    pred_all = np.array(pred_all)  # (N, 17) NitroGen order
    true_all = np.array(true_all)  # (N, 17) dataset order
    print(f"samples used: {n_used}")
    print(f"true button activation rate (dataset order): "
          f"{dict(zip(DATASET_BUTTONS, np.round(true_all.mean(0),2)))}")

    # ---- CRITICAL DIAGNOSTIC: joystick correlation ----
    # Validates frame->action time alignment AND the [buttons(17), j_left(2),
    # j_right(2)] packed layout. Directional plans depend on this being correct.
    pj = np.array(pred_joy)   # (N, 4) model dims 17..20: j_left x,y, j_right x,y (packed [0,1])
    tj = np.array(true_joy)   # (N, 4) ground-truth joysticks (raw [-1,1])
    # model joystick is normalized to [0,1] via (j+1)/2; map true to same space
    tj_norm = (tj + 1.0) / 2.0
    names = ["j_left_x", "j_left_y", "j_right_x", "j_right_y"]
    print("\nJOYSTICK diagnostic (model packed dim 17..20 vs ground truth):")
    for i, nm in enumerate(names):
        a, b = pj[:, i], tj_norm[:, i]
        c = np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
        print(f"  {nm:10s}: corr={c:+.3f}  pred_mean={a.mean():.2f} true_mean={b.mean():.2f} true_std={b.std():.2f}")
    jc = np.nanmean([np.corrcoef(pj[:, i], tj_norm[:, i])[0, 1] for i in range(4)
                     if pj[:, i].std() > 1e-6 and tj_norm[:, i].std() > 1e-6])
    print(f"  mean joystick corr = {jc:+.3f}  "
          f"({'ALIGNMENT+LAYOUT OK' if jc > 0.2 else 'WEAK -> alignment/layout suspect'})")

    # Correlation matrix: corr[k, j] = corr(pred_dim_k, true_button_j)
    corr = np.zeros((NB, NB))
    for k in range(NB):
        for j in range(NB):
            pk, tj = pred_all[:, k], true_all[:, j]
            if pk.std() < 1e-6 or tj.std() < 1e-6:
                corr[k, j] = 0.0
            else:
                corr[k, j] = np.corrcoef(pk, tj)[0, 1]
    # For each NitroGen output dim k, best-matching dataset button
    print("\nNitroGen_out_dim -> best dataset button (by corr):")
    used_j = {}
    for k in range(NB):
        j = int(np.nanargmax(corr[k]))
        used_j.setdefault(j, []).append(k)
        print(f"  dim {k:2d} -> {DATASET_BUTTONS[j]:15s} (corr {corr[k,j]:+.2f})")
    # Recovered order = for each model dim k, the dataset button name
    recovered = [DATASET_BUTTONS[int(np.nanargmax(corr[k]))] for k in range(NB)]
    print("\nRECOVERED BUTTON_ORDER (model dim 0..16):")
    print(recovered)
    # Sanity: how many dims have a confident (>0.2) match
    conf = sum(1 for k in range(NB) if np.nanmax(corr[k]) > 0.2)
    print(f"confident matches (corr>0.2): {conf}/{NB}")
    np.save("/tmp/button_corr.npy", corr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
