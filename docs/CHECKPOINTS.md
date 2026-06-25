# Checkpoints — detailed training & results

Extreme-detail record of the five released plan-conditioning checkpoints (the Stage-2, 2B-backbone
era, 2026-06-22 → 06-24). For each: lineage, exact training recipe, the data it was trained on, what
happened, and eval numbers. Slim copies (EMA plan_head + LoRA, fp16) are in `ckpts/handoff_zips/`; full
checkpoints are `runs/<exp>/plan_stage1_*.pt` on the dev box.

> **TL;DR ranking.** Default = **`btn_s600`** (direction + 5/5 buttons). Pure balanced direction =
> **`clean_lora_s1200`**. Left/right-recovery research = **`clean_s2000`**. Counterfactual override =
> **`override_s2000`**. **`dir_s2500` is a NEGATIVE control** (collapsed left/right) — keep only for
> comparison.

---

## 0. Shared setup (all five)

- **Base (frozen):** NVIDIA NitroGen DiT (`ckpts/nitrogen/ng.pt`). Config from each ckpt's `ckpt_config`:
  8-layer flow-matching DiT, 16 heads × 64 dim, output_dim 1024, ada_norm, dropout 0.2; action_horizon
  **18**, max_action_dim **25**, 256 visual tokens/frame, frame_per_sample 1, action_shift 3.
- **Planner backbone (frozen):** Qwen3.5-**2B** VL (`--qwen Qwen/Qwen3.5-2B`). (0.8B also supported; the
  2B redesign moved the resampler to the backbone's native hidden dim and the linear adapter to *after*
  the resampler — `nitrogen/planner.py`, `PlannerConfig.backbone_hidden_size`.)
- **Trainable parts:** the **PlanHead** = resampler (8 queries) + adapter, and (btn / clean_lora only)
  **LoRA** on the DiT cross-attention. Everything else frozen. `--num-plan-tokens 8`, `--null-mode
  masked` (null plan == base model exactly, so CFG works), `--freeze-dit`.
- **Stage-2 objective:** `--s2-outcome-contrastive --contrastive-weight 1.0 --contrastive-mode
  pertoken` — label each VLM plan by its real chunk's dominant direction so the contrastive loss
  de-collinearizes plan tokens along the action axis (the EXP-043 fix). `--plan-ratio 0.6`.
- **Data substrate:** Stage-2 multimodal (mm) caches built ZERO-new-download from the existing
  chunk pool (`/tmp/stage1_more` shards, `/tmp/frames_cc` frames):
  - `gen_stage2_lookup_mm.py` → `{uuid: {plan, game, action_summary, window, before_idx, after_idx}}`.
    The planner sees the **two boundary frames** (decision point + ~1s later) so it can perceive motion;
    thinking DISABLED (train/inference plan distributions must match).
  - `cache_mm_hidden.py` → frame-conditioned plan-encoder hidden states (resampler cross-attends over
    BOTH frames and plan text → frame-specific K tokens; the fix for text-only collinearity).
  - `cache_mm_cf.py` / `cache_mm_button_cf.py` → counterfactual override pairs (see per-ckpt below).

---

## 1. `clean_s2000` — clean-label run that RECOVERS env-free left/right  🔬✅

**Run:** `runs/stage2_2b_clean/plan_stage1_2000.pt` (step 2000). plan_head only (38 tensors, no LoRA).

**Why it exists.** Directional plans were failing to teach env-free left/right (token separation stuck
~0.50, "collapsed"). Root-cause analysis (see `dir_s2500` below) found the failure was **contrastive
LABEL NOISE**, not a fundamental wall: **31%** of left/right directional plans had the plan-*stated*
direction ≠ the actual *action* direction (reference-frame / window mismatch). So the contrastive loss
was being pulled apart by mislabeled pairs.

**What was different in training.** The Stage-2 plan lookup + hidden + cf caches were **filtered to
direction-consistent chunks only** (plan-stated dir == action dir): the `*_clean` data
(`/tmp/stage2_plan_lookup_mm_clean.json`, `/tmp/stage2_mm_hidden_2b_clean.pt`) — **1357**
direction-consistent chunks. Same recipe as §0, plan_head-only, frozen DiT.

**What happened (result).** Cleaning the labels **recovered** left/right:
- DiT-space left/right token separation: resampler-out **0.734 → adapter-out 0.688** (vs the noisy
  `dir` run: 0.639 → **0.501 collapse**). The adapter no longer destroys the direction.
- Up/down was never the problem (whole-scene correlated): ~0.86 → 0.88 preserved.
- **Diagnosis localized the loss to the PlanAdapter** (the 2048→1024 projection after the resampler),
  NOT the VLM or resampler — an env-free representation/training bug in ONE module, fixable with clean
  contrastive labels + the projection-after-resampler redesign.

**Use it for:** studying the left/right-recovery result; as the warm-start for `clean_lora` and the
override sweeps.

---

## 2. `dir_s2500` — directional baseline that COLLAPSED left/right  ❌ (negative control)

**Run:** `runs/stage2_2b_dir/plan_stage1_2500.pt` (step 2500). plan_head only.

**What it is.** The directional Stage-2 run trained on the **noisy** (unfiltered) directional plans —
the same recipe as `clean` but BEFORE the 31% label-noise was removed. It is the **contrast** that
proved label noise (not a capability wall) was the cause.

**What happened (result).** Left/right **collapsed**: DiT-space token separation 0.639 (resampler) →
**0.501** (adapter) — i.e. the adapter washed out the direction back to chance (~0.50). Up/down stayed
fine (0.864 → 0.878). Text-only probes confirmed the mechanism: a single direction word in a long
tactical plan is diluted by scene-description words (mean-pool + the trained resampler don't extract it;
TF-IDF catches it at 0.64, embeddings only 0.56; short "move left" separates at 1.0).

**Use it for:** the A/B against `clean_s2000` ONLY. Do not deploy — left/right is at chance.

---

## 3. `btn_s600` — MAIN checkpoint: direction + 5/5 BUTTON steering  ✅ (default)

**Run:** `runs/stage2_2b_btn/plan_stage1_600.pt` (step **600**). **plan_head + LoRA** (70 tensors).

**Why it exists.** To broaden "Job-1" steering BEYOND stick directions to **buttons/actions**
(jump/accelerate/attack/brake/dash) — the motor primitives a plan should be able to command.

**Data — the button counterfactual cache** (`cache_mm_button_cf.py` →
`/tmp/stage2_mm_button_cf_2b.pt`, ~1357 pairs): for each gameplay frame_i, pair it with a chunk_j whose
**dominant button** (pressed in ≥0.5 of steps) is a target button B; use a **terse single-concept
synthetic plan** for B; store the override action_j (which presses B). Terse button plans are clean
(unlike scene-diluted directional plans) and buttons are NOT scene-symmetric the way left/right is, so
button override is at least as learnable as direction override. **Button map:**
| token | button | plans |
|---|---|---|
| 16 RIGHT_TRIGGER | accelerate (racing throttle) | "accelerate" / "speed up" / "go fast" |
| 18 SOUTH (A) | jump / confirm | "jump" / "jump now" / "leap up" |
| 20 WEST (X) | attack / shoot | "attack" / "shoot" / "fire" |
| 9 LEFT_TRIGGER | brake | "brake" / "slow down" / "stop" |
| 5 EAST (B) | dash / secondary | "dash" / "dodge" / "use action" |

**Training recipe.** Warm-started on the clean plan-head, frozen DiT + **LoRA on DiT cross-attn**, the
button-cf objective mixed into the plan examples:
`--mm-button-cf-lookup /tmp/stage2_mm_button_cf_2b.pt --s2-button-cf-ratio <f>` plus the standard
`--s2-outcome-contrastive --contrastive-weight 1.0 --contrastive-mode pertoken --num-plan-tokens 8
--null-mode masked --freeze-dit --plan-ratio 0.6 --lr-plan 1e-4 --batch-size 32`. Frequent checkpoints
(300/600/900/1200).

**What happened (result).** Baseline (pre-button) was **2/5** buttons selective; **`s600` = 5/5**
(jump/accelerate/attack/brake/dash all selective, measured env-free by `eval_buttons.py` =
P(button|plan)) **while retaining** the left/right + up/down directions. Step 600 was the sweet spot
(later steps drift). This is the most capable single checkpoint and the default eval ckpt.

**Use it for:** the default — combined direction + button steering.

---

## 4. `clean_lora_s1200` — best-balanced 4-way direction  ✅

**Run:** `runs/stage2_2b_clean_lora/plan_stage1_1200.pt` (step 1200). **plan_head + LoRA** (70 tensors).

**What it is.** The LoRA balance/robustness sweep on top of the clean plan-head (`scripts/lora_sweep.sh`):
warm-start `runs/stage2_2b_clean/plan_stage1_2000.pt`, add **LoRA on the DiT cross-attn**, add the
**counterfactual override** objective (`--mm-cf-lookup /tmp/stage2_mm_cf_2b.pt --s2-cf-ratio`), vary
DiT learning rate / cf-ratio / LoRA rank to control the 600-balanced vs 1200-left-lean drift. Exact
common args (from `scripts/lora_sweep.sh`):
```
--ng-ckpt ckpts/nitrogen/ng.pt --qwen Qwen/Qwen3.5-2B
--shard-root /tmp/stage1_more --frames-dir /tmp/frames_cc
--init-from runs/stage2_2b_clean/plan_stage1_2000.pt
--vlm-plan-lookup /tmp/stage2_plan_lookup_mm_clean.json
--mm-plan-hidden-lookup /tmp/stage2_mm_hidden_2b_clean.pt
--mm-cf-lookup /tmp/stage2_mm_cf_2b.pt
--gameplay-only --s2-outcome-contrastive --contrastive-weight 1.0 --contrastive-mode pertoken
--num-plan-tokens 8 --num-chunks 1 --null-mode masked --freeze-dit
--plan-ratio 0.6 --lr-plan 1e-4 --batch-size 32 --save-every 250
```

**What happened (result).** `clean_lora/s1200` was the **best-balanced 4-way** checkpoint:
**LR-balance 0.96, UD-balance 0.93** (`eval_balance.py` across the sweep), and **selectivity 1.00**
in-env on SuperTuxKart. The sweep configs (lowlr / lowcf / rank32) did NOT beat it.

**Use it for:** the cleanest balanced 4-direction steering (no button steering).

---

## 5. `override_s2000` — env-free counterfactual override (gold-token)  🔬

**Run:** `runs/stage2_2b_override/plan_stage1_2000.pt` (step 2000; also 1000/3000 on disk). plan_head only.

**What it is.** The counterfactual-override experiments: making a plan **flip the frame's action prior**
env-free. Trained with the directional counterfactual cross-pair cache
(`cache_mm_cf.py` → `/tmp/stage2_mm_cf_2b.pt`) via `--mm-cf-lookup … --s2-cf-ratio>0`: a fraction of
examples transplant a real (plan, action) from a DIFFERENT direction cluster, teaching the plan token to
override the frame.

**What happened (result).** Key finding: **plan-TEXT tokens are too weak to flip a strong frame prior**;
env-free override works via **GOLD-TOKEN injection** (a strong training-teacher token + CFG w≈16 +
multi-seed), not plan text. Verified in-game (Cave Story): a gold-left teacher token drove Quote x
160→125 (left) while plan-text gave 160→211 (right); gold left-vs-right stick_x +0.285 vs +0.944 (sep
0.66). In the TARGET racing env (STK), "steer left" vs "steer right" gave only **+0.12** jL_x separation
at w=1, breaking down at higher CFG — concrete in-env evidence that **left/right steering needs RL/envs**
(up/down steering works, 0.998). This checkpoint is the substrate for those override probes.

**Use it for:** counterfactual-override / gold-token research (`eval_cfg_override.py`, the cavestory
gold POCs).

---

## How to reload (all)

```python
import torch
slim = torch.load("ckpts/handoff_zips/btn_s600.pt", map_location="cpu", weights_only=False)
# build the NitroGen model + PlanHead, load base ng.pt as usual, then overlay the trainable delta:
missing, unexpected = model.load_state_dict(slim["trainable_ema"], strict=False)
# missing = the frozen base keys (expected); unexpected should be empty.
```
`slim["trainable_keys"]` lists exact tensors (`plan_head.*` and, for btn/clean_lora,
`model.transformer_blocks.*.lora_A/B`). These are EMA/fp16 (eval). For full-precision or to RESUME
training use the original `runs/<exp>/plan_stage1_*.pt`. Re-slim any checkpoint with
`planner_poc/extract_plan_head.py <full.pt> <slim.pt>`.

## Provenance note
EXPERIMENTS.md (EXP-000..049b) predates this 2B/clean/dir/btn/override era (it ends 2026-06-19); the
authoritative record for these five checkpoints is THIS file + the stored project memories. The data
caches (`/tmp/stage2_*`) live on the dev box and are reproducible from the chunk pool via the
`gen_stage2_lookup_mm.py` / `cache_mm_*.py` scripts (zero new downloads).
