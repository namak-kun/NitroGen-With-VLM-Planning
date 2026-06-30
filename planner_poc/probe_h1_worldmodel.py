"""Probe H1 — is a 1-step world model latent in NitroGen's DiT?

Hypothesis (MULTICHUNK_DESIGN.md §5): NitroGen emits a coherent 18-action chunk, so its
DiT must implicitly model how the state evolves over those steps. If we can decode the
NEXT frame's features from the DiT's internal hidden states (conditioned on frame_t + the
real action chunk), NitroGen already "contains" a free 1-step world model -> enables R2.

Setup (uses /tmp/frames_cc real consecutive frames):
- frame_t  = <uuid>__<s>.png         (start of chunk 0)
- frame_t1 = <uuid>__<s+H*stride>.png (start of chunk 1 == result of executing chunk 0)
- TARGET   = SigLIP last_hidden_state of frame_t1, mean-pooled -> (1024,)
- INPUT    = pooled DiT internals: run get_action-style forward conditioned on frame_t,
             feed the REAL chunk-0 actions as the (denoised) action tokens at small t,
             grab DiT hidden states + final action-token features, mean-pool -> features.
- PROBE    = small MLP/ridge trained INPUT->TARGET. Compare R^2 vs baselines:
             (a) predict frame_t features from frame_t (identity/persistence baseline),
             (b) predict mean target (trivial).
If the DiT-internal probe beats persistence, NitroGen encodes predictive next-frame
structure. Env-free; targets are real future frames we already have.
"""
import glob, os, re, sys
import numpy as np
import torch
import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO)
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from PIL import Image
from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.training.actions import load_chunk_actions, assemble_chunk
import json

device = "cuda"; H = 18; STRIDE = 2; SHIFT = 3
ck = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, action_horizon=H, max_sequence_length=256))

mc = CC.model_cfg.model_copy(deep=True)
m = NitroGen(config=mc, game_mapping=None)
m.load_state_dict(ck["model"], strict=False)
m = m.to(device).eval()


def siglip_feat(png):
    """Mean-pooled SigLIP last_hidden_state of a frame -> (1024,) numpy. This is the
    world-model TARGET (next frame) and the persistence-baseline INPUT."""
    fr = np.asarray(Image.open(png).convert("RGB"))
    pv = ip([fr], return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = m.vision_encoder(pv)["last_hidden_state"]  # (1, 256, d)
    return h[0].float().mean(0).cpu().numpy()


def dit_internal(png, real_chunk, t_bucket=50):
    """Run the conditioned DiT forward with the REAL action chunk as the action tokens at
    a small flow-time t (near-clean), grab DiT internal hidden states + action features,
    mean-pool -> INPUT feature vector."""
    pv = ip([np.asarray(Image.open(png).convert("RGB"))], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
         for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float()
    d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
    d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    # pack real chunk -> (1, H, 25): j_left, j_right normalized to [0,1] as in pack_actions
    a = np.concatenate([(real_chunk["j_left"] + 1) / 2, (real_chunk["j_right"] + 1) / 2,
                        real_chunk["buttons"]], axis=-1).astype(np.float32)  # (H,25)
    actions = torch.from_numpy(a).unsqueeze(0).to(device)
    tb = torch.tensor([t_bucket], device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vis = m.encode_images(d["images"])
        af = m.action_encoder(actions, tb, d["embodiment_id"])
        vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, af,
                                      d["dropped_images"], game_ids=d["game_ids"], plan_tokens=None)
        vl = m.vl_self_attention_model(vl, attention_mask=m._additive_key_mask(d["vl_attn_mask"], vl.dtype))
        mo, allh = m.model(hidden_states=sa, encoder_hidden_states=vl,
                           encoder_attention_mask=d["vl_attn_mask"], timestep=tb,
                           return_all_hidden_states=True)
    # pool: final action-token features (mo) + mean over the DiT layer stack
    feats = [mo[0].float().mean(0).cpu().numpy()]
    if isinstance(allh, (list, tuple)) and len(allh):
        feats.append(torch.stack([h[0].float().mean(0) for h in allh]).mean(0).cpu().numpy())
    return np.concatenate(feats)


def build_dataset(max_n=120):
    cc = glob.glob("/tmp/frames_cc/*.png")
    byu = {}
    for p in cc:
        mo = re.match(r"(.+)__(\d+)\.png$", os.path.basename(p))
        if mo:
            byu.setdefault(mo.group(1), set()).add(int(mo.group(2)))
    # map uuid -> chunk dir (for real actions)
    dirmap = {}
    for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True):
        u = json.load(open(md)).get("uuid")
        if u:
            dirmap[u] = os.path.dirname(md)
    X, Yt, Yp, persist = [], [], [], []
    starts = [100, 250, 400]
    for uuid, offs in byu.items():
        if uuid not in dirmap:
            continue
        pq = os.path.join(dirmap[uuid], "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(dirmap[uuid], "actions_raw.parquet")
        try:
            acts = load_chunk_actions(pq)
        except Exception:
            continue
        for s in starts:
            s1 = s + H * STRIDE
            if s not in offs or s1 not in offs:
                continue
            f0 = f"/tmp/frames_cc/{uuid}__{s}.png"; f1 = f"/tmp/frames_cc/{uuid}__{s1}.png"
            rc = assemble_chunk(acts["buttons"], acts["j_left"], acts["j_right"], s + SHIFT, H, STRIDE)
            if rc is None:
                continue
            try:
                X.append(dit_internal(f0, rc))
                Yt.append(siglip_feat(f1))          # next-frame target
                persist.append(siglip_feat(f0))     # current-frame (persistence baseline)
            except Exception as e:
                print("skip", uuid, s, repr(e)[:60]); continue
            if len(X) >= max_n:
                return map(np.array, (X, Yt, persist))
    return map(np.array, (X, Yt, persist))


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=10.0, pca_dim=None):
    """Closed-form ridge; return mean per-dim R^2 on held-out. Optional PCA on X to avoid
    the high-dim/few-sample degeneracy (fit PCA on train only)."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    if pca_dim is not None and pca_dim < Xtr.shape[1]:
        # PCA via SVD on train
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        V = Vt[:pca_dim].T
        Xtr = Xtr @ V; Xte = Xte @ V
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ W
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss_res / ss_tot).mean())


def main():
    print("building probe dataset (DiT internals -> next-frame SigLIP)...")
    X, Yt, P = build_dataset(max_n=675)
    print(f"n={len(X)}  X dim={X.shape[1]}  target dim={Yt.shape[1]}")
    if len(X) < 30:
        print("too few samples; need more /tmp/frames_cc pairs"); return
    n = len(X); idx = np.random.RandomState(0).permutation(n); tr, te = idx[:int(.8*n)], idx[int(.8*n):]
    # PCA dim chosen < n_train to keep the probe well-posed
    pd = min(64, int(0.5 * len(tr)))
    r2_persist = ridge_r2(P[tr], Yt[tr], P[te], Yt[te], pca_dim=pd)
    r2_dit = ridge_r2(X[tr], Yt[tr], X[te], Yt[te], pca_dim=pd)
    XP = np.concatenate([X, P], 1)
    r2_both = ridge_r2(XP[tr], Yt[tr], XP[te], Yt[te], pca_dim=pd)
    print(f"\n=== Probe H1: predict NEXT-frame SigLIP features (held-out R^2, PCA={pd}) ===")
    print(f"  persistence (frame_t feats -> frame_t1):   R^2 = {r2_persist:+.3f}")
    print(f"  DiT internals only          -> frame_t1:   R^2 = {r2_dit:+.3f}")
    print(f"  DiT internals + persistence -> frame_t1:   R^2 = {r2_both:+.3f}")
    gain = r2_both - r2_persist
    print(f"\n  DiT adds over persistence: {gain:+.3f}")
    if gain > 0.03:
        print("  -> DiT internals carry NEXT-FRAME info beyond the current frame:")
        print("     a (partial) world model is latent in NitroGen. Worth a real decoder.")
    else:
        print("  -> DiT internals add little over just seeing the current frame:")
        print("     no strong evidence of a latent world model from this (linear, pooled) probe.")


if __name__ == "__main__":
    main()
