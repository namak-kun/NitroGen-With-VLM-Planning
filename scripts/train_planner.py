"""Stage-1 alignment training for plan-conditioned NitroGen.

Loads the released ng.pt as initialization, enables plan conditioning, freezes the
VLM planner (hidden states cached), and trains the plan head (high LR) + DiT /
VL-mixing (low LR) with the flow-matching objective. The vision tower is frozen.
An EMA of trainable weights is kept (paper uses EMA for all results).

Frames: by default uses real yt-dlp ingestion (needs cookies in datacenter envs;
see --cookies-file). Use --stub-frames to dry-run the loop without network.

Example (dry run, no network):
  python scripts/train_planner.py --ng-ckpt ckpts/nitrogen/ng.pt \
      --qwen ckpts/qwen35-0.8b --shard-root <dir-with-chunks> \
      --stub-frames --steps 5 --batch-size 2

Example (real frames):
  python scripts/train_planner.py --ng-ckpt ckpts/nitrogen/ng.pt \
      --qwen ckpts/qwen35-0.8b --shard-root <dir> \
      --cookies-file cookies.txt --steps 1000
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# transformers>=5 SigLIP shim (also handled in-package; harmless belt-and-suspenders)
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda self: self)

from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import (
    NitrogenPlanDataset, PlanDatasetConfig, PlanHiddenCache, make_collate_fn,
    make_video_frame_provider,
)
from nitrogen.training.video import VideoFetchConfig


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        # Decay warmup so short runs aren't dominated by the random init: the
        # effective decay ramps from ~0 up to `decay` (standard EMA warmup).
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    def state_dict(self):
        return self.shadow


def stub_frame_provider(meta, frame_idx):
    h, w = meta["original_video"]["resolution"]
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def remap_for_lora(base_sd, model_sd):
    """Map base-checkpoint keys onto LoRA-wrapped module keys: for each model key
    containing '.base.' whose ckpt source key (without '.base') exists, copy it and
    drop the original (so it isn't reported as 'unexpected')."""
    out = dict(base_sd)
    for mk in model_sd:
        if ".base." in mk:
            src = mk.replace(".base.", ".")
            if src in out:
                out[mk] = out.pop(src)
    return out


def build_param_groups(model, lr_plan, lr_dit):
    plan_params, dit_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("plan_head."):
            plan_params.append(p)
        else:
            dit_params.append(p)
    groups = [{"params": plan_params, "lr": lr_plan, "name": "plan_head"}]
    if dit_params:
        groups.append({"params": dit_params, "lr": lr_dit, "name": "dit_vlmix"})
    return groups


def wsd_lr(step, warmup, total, decay_frac=0.1):
    """Warmup-stable-decay multiplier in [0,1]."""
    if step < warmup:
        return step / max(1, warmup)
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return 1.0
    # linear decay to 0 over the last decay_frac of training
    return max(0.0, (total - step) / max(1, total - decay_start))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ng-ckpt", required=True)
    ap.add_argument("--qwen", required=True, help="Qwen3.5 planner path or HF id")
    ap.add_argument("--shard-root", action="append", required=True,
                    help="dir containing chunk subdirs (repeatable)")
    ap.add_argument("--out-dir", default="runs/stage1")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--num-plan-tokens", type=int, default=8)
    ap.add_argument("--plan-ratio", type=float, default=0.3)
    ap.add_argument("--lr-plan", type=float, default=1e-4)
    ap.add_argument("--lr-dit", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--plan-dropout", type=float, default=0.15)
    ap.add_argument("--null-mode", default="learned", choices=["learned", "masked"],
                    help="null plan realization: learned embedding vs masked-out plan tokens")
    ap.add_argument("--freeze-dit", action="store_true",
                    help="freeze DiT + vl-mixing + action enc/dec; train only the plan head")
    ap.add_argument("--contrastive-weight", type=float, default=0.0,
                    help="weight of SupCon plan-token loss (push apart opposite-direction plans)")
    ap.add_argument("--contrastive-mode", default="mean", choices=["mean", "flatten", "pertoken"],
                    help="SupCon representation: mean (order-blind), flatten (order-aware, concat K tokens), pertoken (per-position SupCon)")
    ap.add_argument("--resampler-self-attn", action="store_true",
                    help="let the K resampler queries self-attend to each other each layer (Q-former style) so plan tokens coordinate; default off = Perceiver cross-attn-only")
    ap.add_argument("--direction-heavy", action="store_true",
                    help="oversample directional (hold/tap) plans so opposite dirs co-occur in batches")
    ap.add_argument("--seq-heavy", action="store_true",
                    help="oversample SEQ/SEQ3 plans for dense within-chunk temporal targets")
    ap.add_argument("--het-heavy", action="store_true",
                    help="include button + heterogeneous (dir+button sequence) plans and oversample them; sets grounded_only=False so buttons are supervised (EXP-019)")
    ap.add_argument("--dur-heavy", action="store_true",
                    help="oversample uneven-duration seqdur plans (briefly/long-time splits) to train content-driven transition timing (EXP-023)")
    ap.add_argument("--num-chunks", type=int, default=1,
                    help="A: cross-chunk plan spans A action-chunks; resampler emits K*A tokens, a per-chunk cursor selects block a (MULTICHUNK_DESIGN R0)")
    ap.add_argument("--cross-chunk", action="store_true",
                    help="use cross-chunk plans (one plan -> per-cursor chunk direction); requires --num-chunks A>1")
    ap.add_argument("--cc-real-frames", action="store_true",
                    help="cross-chunk R0: use the REAL frame at each chunk's start (reads <uuid>__<frame_idx>.png from --frames-dir; see scripts/extract_cc_frames.py)")
    ap.add_argument("--cc-pool", default="hold", choices=["hold", "nested", "both", "hold4"],
                    help="cross-chunk plan family: hold (per-chunk hold), nested (each chunk is a within-chunk SEQ), both, or hold4 (A=4 horizon)")
    ap.add_argument("--cc-posthoc", action="store_true",
                    help="R0 post-hoc: target = REAL chunk actions, plan = post-hoc dominant-dir description (mixed with synthetic to keep plan causal); requires --cc-real-frames")
    ap.add_argument("--cc-posthoc-ratio", type=float, default=0.5,
                    help="fraction of cross-chunk plan examples that are post-hoc (rest synthetic forced)")
    ap.add_argument("--vlm-plan-lookup", default=None,
                    help="Stage-2: path to {uuid: {plan}} VLM-generated tactical plans; plan_text comes from here, target = real chunk (gen_stage2_lookup.py)")
    ap.add_argument("--s2-outcome-contrastive", action="store_true",
                    help="Stage-2: label each VLM plan by its real chunk's dominant direction so contrastive de-collinearizes plan tokens along the action axis (EXP-043; pair with --contrastive-weight>0)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start plan head (and DiT) from a prior Stage-1 checkpoint")
    ap.add_argument("--lora-dit", type=int, default=0,
                    help="LoRA rank on DiT cross-attention (0=off). Use with --freeze-dit for capacity without full FT")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--stub-frames", action="store_true", help="dry-run without network")
    ap.add_argument("--frames-dir", default=None, help="use pre-extracted PNG frames (<uuid>.png); fully local")
    ap.add_argument("--cookies-file", default=None)
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--frame-cache", default="frame_cache")
    args = ap.parse_args()

    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- model
    ckpt = torch.load(args.ng_ckpt, map_location="cpu", weights_only=False)
    cc = CkptConfig.model_validate(ckpt["ckpt_config"])
    mc = cc.model_cfg
    mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = args.num_plan_tokens
    mc.planner_cfg.plan_dropout = args.plan_dropout
    mc.planner_cfg.null_mode = args.null_mode
    mc.planner_cfg.contrastive_weight = args.contrastive_weight
    mc.planner_cfg.contrastive_mode = args.contrastive_mode
    mc.planner_cfg.resampler_query_self_attn = args.resampler_self_attn
    mc.planner_cfg.num_chunks = args.num_chunks
    mc.lora_dit_rank = args.lora_dit
    # Stage-1 freeze policy: vision frozen; DiT/vl-mix low LR; plan head high LR.
    mc.tune_vision_tower = False
    mc.tune_diffusion_model = not args.freeze_dit
    mc.tune_vl_mixing = not args.freeze_dit
    mc.tune_multi_projector = not args.freeze_dit
    mc.tune_plan_head = True
    model = NitroGen(config=mc, game_mapping=None)
    base_sd = ckpt["model"]
    if args.lora_dit:
        # LoRA renames wrapped Linears (to_q.weight -> to_q.base.weight). Remap base
        # checkpoint keys so the (frozen) base weights still load.
        base_sd = remap_for_lora(base_sd, model.state_dict())
    miss, unexp = model.load_state_dict(base_sd, strict=False)
    miss_non_plan = [k for k in miss if not k.startswith("plan_head.") and ".lora_" not in k]
    assert not miss_non_plan and not unexp, (miss_non_plan[:5], unexp[:5])
    if args.init_from:
        # Warm-start from a prior Stage-1 checkpoint (e.g. the aligned plan head),
        # then continue (e.g. with SEQ-heavy + unfrozen DiT for temporal influence).
        # Skip params whose shape doesn't match the current model (e.g. a wider K*A
        # cross-chunk resampler.queries, or a different LoRA rank) so warm-start is
        # robust across architecture tweaks.
        prior = torch.load(args.init_from, map_location="cpu", weights_only=False)
        psd = prior.get("model_ema", prior["model"])
        cur = model.state_dict()
        kept = {k: v for k, v in psd.items() if k in cur and cur[k].shape == v.shape}
        skipped = [k for k in psd if k not in kept]
        m2, u2 = model.load_state_dict(kept, strict=False)
        if skipped:
            print(f"warm-start skipped {len(skipped)} shape-mismatched params, e.g. {skipped[:3]}")
        print(f"warm-started from {args.init_from}: {len(m2)} missing, {len(u2)} unexpected")
    model.to(device)
    model.set_trainable_parameters(
        tune_multi_projector=mc.tune_multi_projector,
        tune_diffusion_model=mc.tune_diffusion_model,
        tune_vision_tower=mc.tune_vision_tower,
        tune_mm_projector=mc.tune_mm_projector,
        tune_vl_mixing=mc.tune_vl_mixing,
        tune_plan_head=mc.tune_plan_head,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M")

    # ---- data
    img_proc = AutoImageProcessor.from_pretrained(mc.vision_encoder_name)
    if args.stub_frames:
        frame_provider = stub_frame_provider
    elif args.cc_real_frames:
        from nitrogen.training.dataset import make_dir_frame_provider_multi
        frame_provider = make_dir_frame_provider_multi(args.frames_dir)
    elif args.frames_dir:
        from nitrogen.training.dataset import make_dir_frame_provider
        frame_provider = make_dir_frame_provider(args.frames_dir)
    else:
        vf_cfg = VideoFetchConfig(
            cache_dir=args.frame_cache, cookies_file=args.cookies_file,
            cookies_from_browser=args.cookies_from_browser, proxy=args.proxy,
        )
        frame_provider = make_video_frame_provider(vf_cfg)

    group_weights = None
    if args.direction_heavy:
        # Make directional plans (hold/tap) dense so opposite directions co-occur in
        # batches (needed for the contrastive loss; see EXPERIMENTS EXP-008).
        group_weights = {"hold": 4.0, "tap": 2.0, "seq": 1.0, "seq3": 0.5, "idle": 0.5}
    if args.seq_heavy:
        # Emphasize SEQ/SEQ3 (dense within-chunk temporal targets) while keeping HOLD
        # dense enough for the contrastive direction-separation (EXP-011/012). Zero out
        # seqdur/seqhet so seq-heavy stays the CLEAN 50/50-split SEQ recipe (seqdur is
        # grounded and would otherwise pollute the ordering signal; EXP-032).
        group_weights = {"hold": 2.0, "tap": 1.0, "seq": 4.0, "seq3": 2.0, "idle": 0.3,
                         "seqdur": 0.0, "seqhet": 0.0}
    if args.het_heavy:
        # Emphasize heterogeneous (dir+button) sequence plans + standalone buttons so the
        # model learns to PRESS a button under a plan (never trained before, EXP-019) and
        # to ROUTE across modalities. Keep HOLD/SEQ dense for direction/order contrastive.
        group_weights = {"hold": 1.5, "tap": 0.5, "seq": 1.5, "seq3": 0.5,
                         "idle": 0.3, "button": 2.0, "seqhet": 4.0}
    if args.dur_heavy:
        # Emphasize uneven-duration seqdur plans so transition TIMING becomes content-
        # driven (vs the fixed even-split prior, EXP-023). Keep HOLD/SEQ for direction +
        # order contrastive. seqdur is grounded (stick-only) so grounded_only stays True.
        group_weights = {"hold": 1.5, "tap": 0.5, "seq": 1.0, "seq3": 0.5,
                         "seqdur": 4.0, "idle": 0.3}
    ds_cfg = PlanDatasetConfig(
        shard_roots=args.shard_root, action_horizon=mc.action_horizon,
        num_plan_tokens=args.num_plan_tokens, plan_ratio=args.plan_ratio,
        max_sequence_length=256 + args.num_plan_tokens,
        grounded_only=not args.het_heavy,
        num_chunks=args.num_chunks, cross_chunk=args.cross_chunk,
        cc_real_frames=args.cc_real_frames, cc_pool=args.cc_pool,
        cc_posthoc=args.cc_posthoc, cc_posthoc_ratio=args.cc_posthoc_ratio,
        vlm_plan_lookup=args.vlm_plan_lookup,
        s2_outcome_contrastive=args.s2_outcome_contrastive,
        group_weights=group_weights,
    )
    ds = NitrogenPlanDataset(ds_cfg, frame_provider, img_proc)

    planner = PlanEncoder(PlannerConfig(backbone_name_or_path=args.qwen))
    planner.load()
    plan_cache = PlanHiddenCache(planner, device)
    collate = make_collate_fn(plan_cache, plan_dim=mc.planner_cfg.plan_hidden_size)
    # Note: plan_cache uses CUDA (frozen Qwen) -> keep workers=0 unless cache is
    # precomputed; DataLoader workers can't share the CUDA encoder.
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)

    # ---- optim
    groups = build_param_groups(model, args.lr_plan, args.lr_dit)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]
    ema = EMA(model, decay=args.ema_decay)

    # ---- loop
    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    running = 0.0
    running_con = 0.0
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            batch["images"] = batch["images"].float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_dict = model(batch)
                loss = out_dict["loss"] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            if "contrastive_loss" in out_dict:
                running_con += float(out_dict["contrastive_loss"])

        # LR schedule
        mult = wsd_lr(step, args.warmup, args.steps)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * mult
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        ema.update(model)
        running += accum_loss
        step += 1

        if step % args.log_every == 0:
            avg = running / args.log_every
            con = running_con / args.log_every
            running = 0.0; running_con = 0.0
            sps = step / (time.time() - t0)
            print(f"step {step}/{args.steps} loss {avg:.4f} con {con:.4f} lr_mult {mult:.3f} "
                  f"{sps:.2f} it/s")

        if step % args.save_every == 0 or step == args.steps:
            path = os.path.join(args.out_dir, f"plan_stage1_{step}.pt")
            save_ckpt(path, model, ema, cc, step)
            print(f"saved {path}")

    print(f"done in {time.time()-t0:.1f}s")


def save_ckpt(path, model, ema, ckpt_config, step):
    """Save a checkpoint compatible with inference_session.load_model.

    Stores full model weights, EMA shadow (trainable subset), ckpt_config and step.
    """
    full = model.state_dict()
    ema_full = dict(full)  # start from current, overwrite trainable with EMA
    for n, v in ema.state_dict().items():
        ema_full[n] = v
    torch.save({
        "model": full,
        "model_ema": ema_full,
        "ckpt_config": ckpt_config.model_dump(),
        "step": step,
    }, path)


if __name__ == "__main__":
    sys.exit(main())
