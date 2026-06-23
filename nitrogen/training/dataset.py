"""Stage-1 plan-conditioned dataset + collator for NitroGen.

Ties together: chunk discovery -> frame fetch (pluggable) + controller masking +
image processing -> real action chunk assembly -> contrastive plan target
(null=real action, synthetic plan=counterfactual) -> tokenization. A collator
batches examples and resolves plan hidden-states from a cache backed by the
frozen VLM planner (so the VLM stays out of the training hot loop in Stage 1).
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from ..mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from .actions import load_chunk_actions, is_idle_window, assemble_chunk, chunk_dominant_dir, summarize_chunk
from .plans import SyntheticPlanSampler
from .video import VideoFrameFetcher, VideoFetchConfig

# 60 fps dataset; control freq decimation (stride 2 -> 30 Hz).
DATASET_FPS = 60.0


# Stable integer label per plan INTENT for the contrastive loss. Design:
#  - HOLD/TAP of the same direction share a label (hold_left, tap_left -> "dir_left"),
#    so the loss separates DIRECTIONS (the EXP-009 win we must keep).
#  - Each SEQ / SEQ3 VARIANT gets its own label (seq_left_right != seq_right_left), so
#    the loss separates opposite TEMPORAL orderings (EXP-012: lumping all seq_*
#    together made up->down == down->up).
def _plan_intent(plan_name: str) -> str:
    for key in ("left", "right", "up", "down"):
        if plan_name in (f"hold_{key}", f"tap_{key}"):
            return f"dir_{key}"
    return plan_name  # seq_*/seq3_*/idle/jump/attack each distinct


def _build_plan_labels():
    from .plans import PLANS
    intents = sorted({_plan_intent(p.name) for p in PLANS}) + ["null"]
    return intents, {n: i for i, n in enumerate(intents)}


_PLAN_LABELS, _PLAN_LABEL_TO_ID = _build_plan_labels()


def plan_label_id(plan_name: str) -> int:
    if plan_name == "null":
        return _PLAN_LABEL_TO_ID["null"]
    return _PLAN_LABEL_TO_ID.get(_plan_intent(plan_name), _PLAN_LABEL_TO_ID.get("idle", 0))


def discover_chunks(shard_root: str) -> list[str]:
    """Find all chunk directories (containing metadata.json) under a shard root."""
    out = []
    for md in glob.glob(os.path.join(shard_root, "**", "metadata.json"), recursive=True):
        out.append(os.path.dirname(md))
    return sorted(out)


# A frame provider returns an RGB uint8 (H,W,3) array given the spec. The default
# uses yt-dlp+ffmpeg; tests can inject a stub that returns random frames.
FrameProvider = Callable[[dict, float], np.ndarray]


def make_video_frame_provider(fetch_cfg: VideoFetchConfig) -> FrameProvider:
    fetcher = VideoFrameFetcher(fetch_cfg)

    def provide(meta: dict, frame_idx: int) -> np.ndarray:
        ov = meta["original_video"]
        start, end = float(ov["start_time"]), float(ov["end_time"])
        path = fetcher.download_slice(ov["url"], ov["video_id"], start, end)
        t_in_slice = frame_idx / DATASET_FPS
        frame = fetcher.extract_frame(path, t_in_slice, size=None)  # native res
        frame = fetcher.mask_controller(
            frame, meta["bbox_controller_overlay"], tuple(ov["resolution"])
        )
        return frame

    return provide


def make_dir_frame_provider(frames_dir: str) -> FrameProvider:
    """Frame provider that reads pre-extracted (already controller-masked) PNGs
    named '<uuid>.png'. Fully local — no network — for fast/robust training once
    frames have been pre-extracted from cached video slices. `frames_dir` may be a
    comma-separated list of dirs (searched in order) so frames split across pools
    (e.g. frames_pre + frames_more) resolve without copying/symlinking.
    """
    import os
    from PIL import Image

    dirs = [d for d in frames_dir.split(",") if d]

    def provide(meta: dict, frame_idx: int) -> np.ndarray:
        for d in dirs:
            path = os.path.join(d, meta["uuid"] + ".png")
            if os.path.exists(path):
                return np.asarray(Image.open(path).convert("RGB"))
        raise FileNotFoundError(f"{meta['uuid']}.png not in {dirs}")

    return provide


def make_dir_frame_provider_multi(frames_dir: str) -> FrameProvider:
    """Frame-idx-aware provider reading pre-extracted masked PNGs named
    '<uuid>__<frame_idx>.png' (see scripts/extract_cc_frames.py). Lets cross-chunk R0
    fetch the REAL frame at each chunk's start (frame_idx = s + a*H*stride). Falls back
    to '<uuid>.png' if the offset-specific file is absent.
    """
    import os
    from PIL import Image

    def provide(meta: dict, frame_idx: int) -> np.ndarray:
        p = os.path.join(frames_dir, f"{meta['uuid']}__{int(frame_idx)}.png")
        if not os.path.exists(p):
            p = os.path.join(frames_dir, meta["uuid"] + ".png")
        return np.asarray(Image.open(p).convert("RGB"))

    return provide


@dataclass
class PlanDatasetConfig:
    shard_roots: list[str]
    action_horizon: int = 18
    frame_stride: int = 2            # 60 fps -> 30 Hz
    action_shift: int = 3           # frames between context frame and action chunk
    plan_ratio: float = 0.5          # fraction of examples that get a synthetic plan
    idle_only_for_plan: bool = False  # attach plans to any window (idle was too restrictive)
    num_plan_tokens: int = 8
    max_sequence_length: int = 264   # 256 image tokens + K plan tokens
    joystick_only_loss: bool = False  # buttons now correctly mapped -> supervise all dims (needed for d-pad direction plans)
    grounded_only: bool = True       # only use grounded (joystick/d-pad) synthetic plans
    modality: str = "auto"           # per-chunk stick-vs-dpad detection for directional plans
    group_weights: dict | None = None  # optional balance over plan groups (hold/tap/seq/seq3/idle)
    num_chunks: int = 1              # A: cross-chunk plans span A chunks (cursor-selected block)
    cross_chunk: bool = False        # use cross-chunk plan sampler (one plan -> per-cursor chunk dir)
    cc_pool: str = "hold"            # cross-chunk plan family: 'hold' | 'nested' | 'both'
    cc_real_frames: bool = False     # cross-chunk R0: use the REAL frame at chunk-a-start per cursor (vs one shared frame)
    cc_posthoc: bool = False         # R0 post-hoc: target = REAL chunk a actions, plan = post-hoc dominant-dir description
    cc_posthoc_ratio: float = 0.5    # fraction of cross-chunk plan examples that are post-hoc (rest synthetic, to keep plan causal)
    cc_starts: tuple = (100, 250, 400)  # candidate window starts (must match scripts/extract_cc_frames.py)
    vlm_plan_lookup: str | None = None  # Stage-2: path to {uuid: {plan: text}} VLM-generated tactical plans; when set, plan_text comes from here (target = real chunk at frame 303) and plan_label is a single 'vlm' class
    s2_outcome_contrastive: bool = False  # Stage-2: label each VLM plan by its real chunk's dominant direction so contrastive de-collinearizes plan tokens along the action axis (EXP-043)
    s2_augment_plan: bool = False    # Stage-2 TEACHER: append the real chunk's action summary to the plan text (privileged-info P+ = P + "...take these actions <seq>") (EXP-044)
    teacher_token_lookup: str | None = None  # Stage-2 STUDENT (EXP-045): path to a torch-saved {uuid: (K,d) teacher plan tokens} (gen_teacher_tokens.py); attaches teacher_tokens for distillation
    mm_plan_hidden_lookup: str | None = None  # Stage-2 FRAME-CONDITIONED (EXP-050): path to torch-saved {uuid: {h:(L,d), mask:(L,)}} precomputed frozen-VLM hidden states over [before,after] frames + plan text (cache_mm_hidden.py). When set, the plan_hidden fed to the resampler comes from HERE (frame-grounded) instead of the text-only PlanHiddenCache -> plan tokens become frame-specific. Null/dropped rows reuse the same hidden (masked out), forming a clean CFG pair.
    gameplay_only: bool = False  # Stage-2 data hygiene (EXP-050): drop chunks whose vlm_plan_lookup entry has is_gameplay=False (tag_gameplay.py) -> exclude stream intro/menu/logo/loading frames from training.
    mm_cf_lookup: str | None = None  # Stage-2 COUNTERFACTUAL + frame-conditioned (EXP-051b): path to torch-saved {uuid: {h,mask (=encode_multimodal(frame_i, plan_j)), buttons,j_left,j_right (=action_j), cf_dir}} (cache_mm_cf.py). When set + s2_cf_ratio>0, cf examples use THIS frame-conditioned cross-pair hidden + override target instead of the s2_index text path. Teacher-distillation is disabled on cf examples (the factual teacher token doesn't match the override target).
    s2_index: str | None = None      # Stage-2 COUNTERFACTUAL (EXP-047): path to {uuid: {buttons,j_left,j_right,dir}} (gen_stage2_index.py); enables transplanting a real (plan,action) from a different direction cluster
    s2_cf_ratio: float = 0.0         # Stage-2: fraction of PLAN examples that are counterfactual (frame_i + plan_j/action_j from a DIFFERENT dir cluster; target follows the PLAN). 0 disables. Teaches the plan to OVERRIDE the frame -> low-guidance counterfactual control.
    seed: int = 0


class NitrogenPlanDataset(torch.utils.data.Dataset):
    def __init__(self, config: PlanDatasetConfig, frame_provider: FrameProvider,
                 image_processor, tokenizer: Optional[NitrogenTokenizer] = None):
        self.cfg = config
        self.frame_provider = frame_provider
        self.img_proc = image_processor
        self.rng = random.Random(config.seed)
        self.plan_sampler = SyntheticPlanSampler(
            action_horizon=config.action_horizon, grounded_only=config.grounded_only,
            modality=config.modality, group_weights=config.group_weights, seed=config.seed)
        self.cc_sampler = None
        if config.cross_chunk:
            from .plans import CrossChunkPlanSampler
            self.cc_sampler = CrossChunkPlanSampler(
                action_horizon=config.action_horizon, modality=config.modality,
                seed=config.seed, pool=config.cc_pool)
        # Stage-2: load VLM-generated tactical plans keyed by uuid (gen_stage2_lookup.py).
        self.vlm_plans = None
        if config.vlm_plan_lookup and os.path.exists(config.vlm_plan_lookup):
            self.vlm_plans = json.load(open(config.vlm_plan_lookup))
        # Stage-2 STUDENT: precomputed teacher plan tokens keyed by uuid (gen_teacher_tokens.py)
        # for privileged-info distillation (EXP-045).
        self.teacher_tokens = None
        if config.teacher_token_lookup and os.path.exists(config.teacher_token_lookup):
            self.teacher_tokens = torch.load(config.teacher_token_lookup, map_location="cpu")
        # Stage-2 FRAME-CONDITIONED (EXP-050): precomputed frozen-VLM hidden states over
        # [before,after] frames + plan text, keyed by uuid (cache_mm_hidden.py). When present,
        # these replace the text-only plan_hidden so the resampler reads frame+text -> the K
        # plan tokens become frame-specific.
        self.mm_plan_hidden = None
        if config.mm_plan_hidden_lookup and os.path.exists(config.mm_plan_hidden_lookup):
            self.mm_plan_hidden = torch.load(config.mm_plan_hidden_lookup, map_location="cpu")
        # Stage-2 COUNTERFACTUAL frame-conditioned (EXP-051b): precomputed cross-pair hidden
        # encode_multimodal(frame_i, plan_j) + override action_j, keyed by uuid (cache_mm_cf.py).
        self.mm_cf = None
        if config.mm_cf_lookup and os.path.exists(config.mm_cf_lookup):
            self.mm_cf = torch.load(config.mm_cf_lookup, map_location="cpu", weights_only=False)
        # Stage-2 COUNTERFACTUAL (EXP-047): chunk index {uuid: {buttons,j_left,j_right,dir}} +
        # per-direction uuid pools, for transplanting a real (plan, action) from a DIFFERENT
        # direction cluster onto a frame (target follows the PLAN -> override training).
        self.s2_index = None
        self.s2_dir_pools = None
        if config.s2_index and os.path.exists(config.s2_index):
            self.s2_index = torch.load(config.s2_index, map_location="cpu", weights_only=False)
            pools: dict[str, list] = {}
            for uu, v in self.s2_index.items():
                pools.setdefault(v["dir"], []).append(uu)
            self.s2_dir_pools = pools

        if tokenizer is None:
            tok_cfg = NitrogenTokenizerConfig(
                training=True, action_horizon=config.action_horizon,
                num_plan_tokens=config.num_plan_tokens,
                joystick_only_loss=config.joystick_only_loss,
                max_sequence_length=config.max_sequence_length)
            tokenizer = NitrogenTokenizer(tok_cfg)
        self.tokenizer = tokenizer

        self.chunks: list[str] = []
        for root in config.shard_roots:
            self.chunks.extend(discover_chunks(root))
        if not self.chunks:
            raise RuntimeError(f"No chunks found under {config.shard_roots}")
        # Stage-2: restrict to chunks that actually have a VLM plan (the only valid training
        # examples), so a moving/partial download pool (extra chunks without plans/frames)
        # cannot break training. Keyed by uuid == metadata["uuid"].
        if self.vlm_plans is not None:
            keep = []
            n_nongame = 0
            for cd in self.chunks:
                try:
                    uu = json.load(open(os.path.join(cd, "metadata.json")))["uuid"]
                except Exception:
                    continue
                if uu not in self.vlm_plans:
                    continue
                # EXP-050 data hygiene: drop chunks tagged non-gameplay (stream intro/menu/logo).
                if config.gameplay_only and not self.vlm_plans[uu].get("is_gameplay", True):
                    n_nongame += 1
                    continue
                keep.append(cd)
            self.chunks = keep
            msg = (f"[dataset] Stage-2: {len(self.chunks)} chunks with plans "
                   f"(of {sum(1 for _ in self.vlm_plans)} in lookup)")
            if config.gameplay_only:
                msg += f"; dropped {n_nongame} non-gameplay"
            print(msg)
            if not self.chunks:
                raise RuntimeError("No Stage-2 chunks with plans found")

    def __len__(self):
        return len(self.chunks)

    def _pick_window(self, acts: dict) -> int:
        """Pick a start frame index that leaves room for the context frame + chunk."""
        T = acts["buttons"].shape[0]
        span = self.cfg.action_horizon * self.cfg.frame_stride + self.cfg.action_shift
        hi = T - span - 1
        if hi <= 0:
            return 0
        return self.rng.randint(0, hi)

    def __getitem__(self, idx: int) -> dict:
        chunk_dir = self.chunks[idx]
        meta = json.load(open(os.path.join(chunk_dir, "metadata.json")))
        uuid = meta.get("uuid")
        pq = os.path.join(chunk_dir, "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(chunk_dir, "actions_raw.parquet")
        acts = load_chunk_actions(pq)

        cc_real = self.cfg.cross_chunk and self.cfg.cc_real_frames
        stage2 = self.vlm_plans is not None
        # In cross-chunk R0 (real frames) we sample the cursor FIRST so the context frame
        # can be the REAL frame at chunk-a-start (frame_idx = start + a*H*stride). The
        # window start is drawn from cc_starts (the offsets pre-extracted to disk).
        cc_pre = None
        posthoc_data = None
        cursor = 0
        if stage2:
            # Stage-2 FRAME-CONDITIONED (EXP-050): when mm-hidden is active, the plan + frames +
            # teacher were all generated at a per-uuid window (before_idx, default 250). The DiT
            # context frame MUST be that SAME 'before' frame so the planner and System-1 agree on
            # the scene; the target chunk starts at before_idx + action_shift. (Legacy text-only
            # Stage-2 used a fixed frame 303 with the single-frame provider.)
            entry0 = self.vlm_plans.get(meta.get("uuid"), {}) or {}
            # A=4 CROSS-CHUNK FACTUAL: one frame-grounded plan (the mm-hidden) spans A chunks. The
            # resampler emits K*A tokens; a sampled cursor 'a' picks block a -> supervise it against
            # the REAL chunk a (chunk_starts[a]) with the REAL frame at chunk-a-start (frame_offsets[a]).
            # Block 0 stays the 'act-now' head (also gets counterfactual + distill below); blocks 1..A-1
            # are factual look-ahead only. Requires the a4 lookup (chunk_starts/frame_offsets present).
            cs_list = entry0.get("chunk_starts")
            fo_list = entry0.get("frame_offsets")
            if (self.cfg.num_chunks > 1 and cs_list and fo_list
                    and len(cs_list) >= self.cfg.num_chunks and len(fo_list) >= self.cfg.num_chunks):
                cursor = self.rng.randrange(self.cfg.num_chunks)
                ctx_frame_idx = int(fo_list[cursor])
                chunk_start = int(cs_list[cursor])
            else:
                win = entry0.get("before_idx") if self.mm_plan_hidden is not None else None
                if win is not None:
                    ctx_frame_idx = int(win)
                    chunk_start = int(win) + self.cfg.action_shift
                else:
                    ctx_frame_idx = 303
                    chunk_start = 303
                    if 303 + self.cfg.action_horizon * self.cfg.frame_stride >= acts["buttons"].shape[0]:
                        chunk_start = self.cfg.action_shift
                    ctx_frame_idx = chunk_start
        elif cc_real and self.cc_sampler is not None:
            T = acts["buttons"].shape[0]
            span = (self.cfg.num_chunks - 1) * self.cfg.action_horizon * self.cfg.frame_stride
            span += self.cfg.action_horizon * self.cfg.frame_stride + self.cfg.action_shift
            valid = [s for s in self.cfg.cc_starts if s + span < T]
            start = self.rng.choice(valid) if valid else 0
            cursor = self.rng.randrange(self.cfg.num_chunks)
            ctx_frame_idx = start + cursor * self.cfg.action_horizon * self.cfg.frame_stride
            chunk_start = ctx_frame_idx + self.cfg.action_shift
            # R0 post-hoc: describe what the streamer ACTUALLY did per chunk and target the
            # REAL chunk a. Only when every chunk has a clear dominant direction (else fall
            # back to a synthetic forced plan, which also keeps the plan causal).
            if self.cfg.cc_posthoc and self.rng.random() < self.cfg.cc_posthoc_ratio:
                dirs, rcs = [], []
                for c in range(self.cfg.num_chunks):
                    cs = start + c * self.cfg.action_horizon * self.cfg.frame_stride + self.cfg.action_shift
                    rc = assemble_chunk(acts["buttons"], acts["j_left"], acts["j_right"],
                                        cs, self.cfg.action_horizon, self.cfg.frame_stride)
                    rcs.append(rc)
                    dirs.append(chunk_dominant_dir(rc) if rc is not None else None)
                if all(dd is not None for dd in dirs):
                    text = "go " + ", then ".join(dirs)
                    posthoc_data = (text, rcs[cursor], f"dir_{dirs[cursor]}")
            if posthoc_data is None:
                cc_pre = self.cc_sampler.sample(real_chunk=None, cursor=cursor)
        else:
            start = self._pick_window(acts)
            ctx_frame_idx = start
            chunk_start = start + self.cfg.action_shift
        real_chunk = assemble_chunk(
            acts["buttons"], acts["j_left"], acts["j_right"], chunk_start,
            self.cfg.action_horizon, self.cfg.frame_stride)
        if real_chunk is None:
            # fall back to the very first valid window
            chunk_start = self.cfg.action_shift
            real_chunk = assemble_chunk(
                acts["buttons"], acts["j_left"], acts["j_right"], chunk_start,
                self.cfg.action_horizon, self.cfg.frame_stride)

        idle = is_idle_window(acts["buttons"], acts["j_left"], acts["j_right"],
                              chunk_start, self.cfg.action_horizon, self.cfg.frame_stride)

        # Decide plan vs null
        use_plan = (self.rng.random() < self.cfg.plan_ratio) and \
                   (idle or not self.cfg.idle_only_for_plan)
        plan_cursor = cursor   # A=4 cross-chunk: PlanHead gathers resampler block `cursor`
        plan_label_override = None
        cf_hidden = None        # EXP-051b: set to the cf cross-pair mm-hidden for cf examples
        if stage2:
            # Stage-2: VLM tactical plan (real text) vs null. Target is ALWAYS the real
            # chunk (the plan describes what the streamer did). plan-dropout -> null = base.
            entry = self.vlm_plans.get(meta.get("uuid"))
            if use_plan and entry and entry.get("plan"):
                # COUNTERFACTUAL (EXP-047, single-chunk motor override): with prob s2_cf_ratio,
                # transplant a real (plan_j, action_j) from a DIFFERENT direction cluster onto
                # THIS frame -> target follows the PLAN, not the frame. The null case (below)
                # keeps the factual frame-following target, so plan vs null form a CFG pair that
                # teaches the plan to OVERRIDE the frame at low guidance. Only single-chunk:
                # transplanting a multichunk trajectory would need game dynamics (no env).
                # EXP-051b: when mm_cf is loaded, use the PRECOMPUTED frame-conditioned cross-pair
                # (encode_multimodal(frame_i, plan_j)) + its override action -> keeps cf consistent
                # with the frame-conditioned plan tokens (the text s2_index path would mismatch).
                cf_hidden = None
                # Counterfactual transplant is single-chunk motor override -> only the act-now
                # block (cursor 0) carries it; look-ahead blocks (a>0) stay purely factual.
                cf_allowed = (cursor == 0)
                mm_cf_ok = (cf_allowed and self.mm_cf is not None and uuid in self.mm_cf
                            and self.cfg.s2_cf_ratio > 0 and self.rng.random() < self.cfg.s2_cf_ratio)
                cf = mm_cf_ok or (cf_allowed and self.mm_cf is None and self.s2_index is not None
                                  and self.s2_dir_pools is not None and self.cfg.s2_cf_ratio > 0
                                  and self.rng.random() < self.cfg.s2_cf_ratio)
                cf_dir = None
                if mm_cf_ok:
                    ce = self.mm_cf[uuid]
                    cf_dir = ce.get("cf_dir")
                    plan_name = "vlm_cf_mm"
                    plan_text = self.vlm_plans.get(ce.get("cf_uuid"), {}).get("plan", entry["plan"])
                    target = {"buttons": np.asarray(ce["buttons"]),
                              "j_left": np.asarray(ce["j_left"]),
                              "j_right": np.asarray(ce["j_right"])}
                    cf_hidden = ce
                elif cf:
                    own_dir = chunk_dominant_dir(real_chunk) if real_chunk is not None else None
                    other_dirs = [d for d in ("left", "right", "up", "down")
                                  if d != own_dir and self.s2_dir_pools.get(d)]
                    if other_dirs:
                        cf_dir = self.rng.choice(other_dirs)
                        uj = self.rng.choice(self.s2_dir_pools[cf_dir])
                        ej = self.vlm_plans.get(uj)
                        ij = self.s2_index.get(uj)
                        if ej and ej.get("plan") and ij is not None:
                            plan_name = "vlm_cf"; plan_text = ej["plan"]
                            target = {"buttons": np.asarray(ij["buttons"]),
                                      "j_left": np.asarray(ij["j_left"]),
                                      "j_right": np.asarray(ij["j_right"])}
                        else:
                            cf_dir = None
                    if cf_dir is None:
                        cf = False
                if not cf:
                    plan_name = "vlm"; plan_text = entry["plan"]; target = real_chunk
                plan_dropped = False
                # TEACHER (EXP-044): append the real action sequence to the plan -> privileged
                # P+ = P + "...take these actions <seq>". The student (base P) is later distilled
                # to match this teacher (contrastive), transferring finer-than-direction action
                # grounding without seeing the actions at test time. (Augment the chunk the plan
                # actually describes -> for counterfactual that is the transplanted target.)
                if self.cfg.s2_augment_plan and target is not None:
                    plan_text = plan_text + " To do this I take the following actions: " \
                        + summarize_chunk(target) + "."
                # Action-outcome contrastive label: group plans by the dominant direction of the
                # chunk the plan describes (the TARGET) so contrastive de-collinearizes plan
                # tokens ALONG the action axis (EXP-042). For counterfactual, that is cf_dir.
                if self.cfg.s2_outcome_contrastive:
                    if cf and cf_dir is not None:
                        dd = cf_dir
                    else:
                        dd = chunk_dominant_dir(real_chunk) if real_chunk is not None else None
                    key = f"dir_{dd}" if dd is not None else "idle"
                    plan_label_override = _PLAN_LABEL_TO_ID.get(key, _PLAN_LABEL_TO_ID.get("idle", 0))
                else:
                    plan_label_override = _PLAN_LABEL_TO_ID.get("idle", 0)  # single class; contrastive off
            else:
                plan_name, plan_text, target = "null", "", real_chunk
                plan_dropped = True
        elif use_plan and self.cc_sampler is not None and cc_real and posthoc_data is not None:
            # R0 post-hoc: plan describes the streamer's real per-chunk dirs; target = REAL chunk.
            plan_text, target, label_key = posthoc_data
            plan_name = "posthoc"; plan_cursor = cursor
            plan_dropped = False
            plan_label_override = _PLAN_LABEL_TO_ID.get(label_key)
        elif use_plan and self.cc_sampler is not None and cc_real:
            # Cross-chunk R0 (synthetic forced): reuse pre-sampled plan; REAL frame at chunk-a.
            plan_name, plan_text, plan_cursor, target, label_key = cc_pre
            plan_dropped = False
            plan_label_override = _PLAN_LABEL_TO_ID.get(label_key)
        elif use_plan and self.cc_sampler is not None:
            # Cross-chunk MVP: one plan -> per-cursor chunk behavior, one shared frame
            # (synthetic forcing is frame-independent).
            plan_name, plan_text, plan_cursor, target, label_key = self.cc_sampler.sample(
                real_chunk=real_chunk)
            plan_dropped = False
            plan_label_override = _PLAN_LABEL_TO_ID.get(label_key)
        elif use_plan:
            plan_name, plan_text, target = self.plan_sampler.sample(
                real_chunk=real_chunk, exclude="idle" if idle else None)
            plan_dropped = False
        else:
            plan_name, plan_text, target = "null", "", real_chunk
            plan_dropped = True

        # Fetch + process the context frame. Single-frame providers cache by uuid; the
        # cross-chunk R0 multi-frame provider needs caching by (uuid, frame_idx) since a
        # uuid has multiple real frames (one per chunk offset).
        pix_cache = getattr(self, "_pix_cache", None)
        if pix_cache is None:
            pix_cache = {}
            self._pix_cache = pix_cache
        uuid = meta.get("uuid")
        # Key by (uuid, frame_idx) whenever the context frame can vary within a uuid: cross-chunk
        # R0 (cc_real) AND A=4 stage-2 cross-chunk (cursor-dependent frame_offsets[a]).
        per_frame = cc_real or (stage2 and self.cfg.num_chunks > 1)
        cache_key = (uuid, ctx_frame_idx) if per_frame else uuid
        pixel_values = pix_cache.get(cache_key)
        if pixel_values is None:
            frame_rgb = self.frame_provider(meta, ctx_frame_idx)
            pixel_values = self.img_proc([frame_rgb], return_tensors="pt")["pixel_values"][0].numpy()
            if uuid is not None:
                pix_cache[cache_key] = pixel_values

        # Tokenize (build vl/sa token ids, action target, masks)
        ex_in = {
            "frames": pixel_values[None],                 # (1, C, H, W)
            "dropped_frames": np.zeros((1,), dtype=bool),
            "buttons": target["buttons"][None],            # (1, H, 17)
            "j_left": target["j_left"][None],
            "j_right": target["j_right"][None],
        }
        ex = self.tokenizer.encode(ex_in)
        ex["plan_text"] = plan_text
        ex["plan_dropped"] = bool(plan_dropped)
        ex["plan_name"] = plan_name
        ex["plan_label"] = plan_label_override if plan_label_override is not None else plan_label_id(plan_name)
        ex["plan_cursor"] = int(plan_cursor)
        ex["is_idle"] = bool(idle)
        # Stage-2 FRAME-CONDITIONED (EXP-050): attach the precomputed frame+text hidden for this
        # uuid so the resampler reads it instead of the text-only PlanHiddenCache. Attached for
        # BOTH plan and null rows (null masks the plan-token positions out -> clean CFG pair).
        # EXP-051b: for a counterfactual example, attach the cf CROSS-PAIR hidden (frame_i+plan_j)
        # so the plan tokens that must produce the override action are the frame-conditioned cf
        # tokens; null rows still use the factual hidden (the CFG-pair's factual side).
        if cf_hidden is not None and not plan_dropped:
            ex["plan_hidden_pre"] = np.asarray(cf_hidden["h"], dtype=np.float32)
            ex["plan_mask_pre"] = np.asarray(cf_hidden["mask"], dtype=bool)
        elif self.mm_plan_hidden is not None and uuid in self.mm_plan_hidden:
            mm = self.mm_plan_hidden[uuid]
            ex["plan_hidden_pre"] = np.asarray(mm["h"], dtype=np.float32)
            ex["plan_mask_pre"] = np.asarray(mm["mask"], dtype=bool)
        # Stage-2 STUDENT distillation (EXP-045): attach the precomputed teacher tokens for
        # this uuid; only valid when a (non-dropped, NON-counterfactual) plan is used and a teacher
        # exists. The factual teacher (plan_i + action_i) does NOT match a cf override target, so
        # distillation is disabled on cf examples (they learn override via the action loss).
        K = self.cfg.num_plan_tokens
        d = getattr(self, "_teacher_dim", None)
        if (self.teacher_tokens is not None and not plan_dropped and cf_hidden is None
                and cursor == 0 and uuid in self.teacher_tokens):
            tt = np.asarray(self.teacher_tokens[uuid], dtype=np.float32)
            self._teacher_dim = tt.shape[-1]
            ex["teacher_tokens"] = tt
            ex["has_teacher"] = True
        elif self.teacher_tokens is not None:
            dim = d if d is not None else int(np.asarray(next(iter(self.teacher_tokens.values()))).shape[-1])
            self._teacher_dim = dim
            ex["teacher_tokens"] = np.zeros((K, dim), dtype=np.float32)
            ex["has_teacher"] = False
        return ex


class PlanHiddenCache:
    """Caches frozen-VLM last-layer hidden states keyed by plan text.

    In Stage 1 the planner is frozen and the synthetic-plan vocabulary is small,
    so we compute each plan text's hidden states once. The empty string "" maps to
    the null plan (dropped), for which any zero placeholder works since the model
    substitutes the learned null embedding when plan_dropped is True.
    """

    def __init__(self, plan_encoder, device):
        self.enc = plan_encoder
        self.device = device
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if text not in self._cache:
            h, kpm = self.enc.encode_text([text if text else "."], self.device)
            self._cache[text] = (h[0].detach().cpu(), kpm[0].detach().cpu())
        return self._cache[text]


def make_collate_fn(plan_cache: Optional[PlanHiddenCache] = None, plan_dim: int = 1024):
    """Collate examples into the batch dict NitroGen.forward expects.

    If `plan_cache` is provided, plan_hidden / plan_key_padding_mask are resolved
    and padded to the batch's max plan length. Otherwise those keys are omitted
    (e.g. when hidden states are precomputed elsewhere).
    """
    array_keys = ["images", "dropped_images", "vl_token_ids", "sa_token_ids",
                  "vl_attn_mask", "actions", "actions_mask"]

    def collate(batch: list[dict]) -> dict:
        out = {}
        for k in array_keys:
            out[k] = torch.stack([torch.as_tensor(np.asarray(b[k])) for b in batch], 0)
        out["embodiment_id"] = torch.zeros(len(batch), dtype=torch.long)
        out["has_real_action"] = torch.ones(len(batch))
        out["plan_dropped"] = torch.tensor([b["plan_dropped"] for b in batch], dtype=torch.bool)
        out["plan_label"] = torch.tensor([b.get("plan_label", -1) for b in batch], dtype=torch.long)
        out["plan_cursor"] = torch.tensor([b.get("plan_cursor", 0) for b in batch], dtype=torch.long)
        if "teacher_tokens" in batch[0]:
            out["teacher_tokens"] = torch.stack(
                [torch.as_tensor(np.asarray(b["teacher_tokens"])) for b in batch], 0)
            out["has_teacher"] = torch.tensor([b.get("has_teacher", False) for b in batch], dtype=torch.bool)

        # Stage-2 FRAME-CONDITIONED (EXP-050): if examples carry precomputed frame+text hidden
        # (plan_hidden_pre), pad THOSE to batch-max L. Takes precedence over the text-only cache.
        if "plan_hidden_pre" in batch[0]:
            hs = [torch.as_tensor(np.asarray(b["plan_hidden_pre"])) for b in batch]
            masks = [torch.as_tensor(np.asarray(b["plan_mask_pre"])) for b in batch]
            L = max(h.shape[0] for h in hs)
            B = len(batch)
            # Precomputed frame+text hiddens are in the VLM BACKBONE dim (0.8B=1024, 2B=2048,
            # 9B=4096), not the DiT dim -> size the buffer from the hidden itself, not plan_dim.
            hdim = hs[0].shape[-1]
            plan_hidden = torch.zeros(B, L, hdim)
            plan_kpm = torch.ones(B, L, dtype=torch.bool)  # True == pad
            for i, (h, m) in enumerate(zip(hs, masks)):
                li = h.shape[0]
                plan_hidden[i, :li] = h.float()
                plan_kpm[i, :li] = m
            out["plan_hidden"] = plan_hidden
            out["plan_key_padding_mask"] = plan_kpm
        elif plan_cache is not None:
            hs, masks = [], []
            for b in batch:
                h, kpm = plan_cache.get(b["plan_text"])
                hs.append(h); masks.append(kpm)
            L = max(h.shape[0] for h in hs)
            B = len(batch)
            hdim = hs[0].shape[-1]   # backbone dim (2B=2048), not the DiT dim
            plan_hidden = torch.zeros(B, L, hdim)
            plan_kpm = torch.ones(B, L, dtype=torch.bool)  # True == pad
            for i, (h, m) in enumerate(zip(hs, masks)):
                li = h.shape[0]
                plan_hidden[i, :li] = h
                plan_kpm[i, :li] = m
            out["plan_hidden"] = plan_hidden
            out["plan_key_padding_mask"] = plan_kpm
        return out

    return collate
