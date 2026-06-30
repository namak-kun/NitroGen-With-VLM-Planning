# NitroGen + Planner: Hierarchical Plan-Conditioned VLA — Design Doc

> Status: design + PoC + **implemented training stack (Phase A/B/C built & verified)**.
> Authored for @namak-kun.
> Everything marked **[verified]** was checked against the real `ng.pt` checkpoint,
> the real `Qwen/Qwen3.5-0.8B` config, the paper LaTeX source, and a real sample
> of the HF dataset. Things marked **[assumption]** still need your sign-off.

---

## IMPLEMENTATION STATUS (what's built and tested)

**Phase A — model integration (DONE, verified):**
- `nitrogen/planner.py`: `PlannerConfig`, `PlanResampler` (K queries), `PlanAdapter`
  (1024→1024), `PlanHead` (resampler+adapter+null, plan-dropout), `PlanEncoder`
  (frozen Qwen3.5 wrapper, cacheable hidden states).
- `nitrogen/flow_matching_transformer/nitrogen.py`: `_PLAN_TOKEN=7`; `planner_cfg`
  + `tune_planner`/`tune_plan_head` in config; `PlanHead` built in `__init__`;
  plan-token injection in `prepare_input_embs`; `compute_plan_tokens` + plan-dropout
  in `forward`; plan wired into `get_action`; SigLIP transformers-5 shim.
- `nitrogen/mm_tokenizers.py`: `_PLAN_TOKEN`, `num_plan_tokens`, K plan placeholders
  prepended to the VL stream.
- Verified (`planner_poc/test_integration.py`): ng.pt loads with **0 non-plan
  missing / 0 unexpected**; forward+backward; grads into plan head; plan≠null effect.
  Backward-compat: planner-disabled load is still 0/0.

**Phase B — data pipeline (DONE, verified offline; live fetch blocked):**
- `nitrogen/training/actions.py`: parquet→action chunks, button order, idle/density.
- `nitrogen/training/plans.py`: 7 atomic synthetic plans + counterfactual targets.
- `nitrogen/training/video.py`: yt-dlp slice download + ffmpeg frame extract +
  controller-bbox masking (cookie/proxy support).
- `nitrogen/training/dataset.py`: `NitrogenPlanDataset`, `PlanHiddenCache`,
  `make_collate_fn`, pluggable frame provider.
- Verified (`planner_poc/test_datapath.py`): full path with STUB frames →
  dataset→img_proc→tokenizer→frozen-Qwen cache→collate→forward+backward. PASS.

**Phase C — training loop (DONE, verified dry-run):**
- `scripts/train_planner.py`: loads ng.pt init, enables planner, split-LR AdamW
  (plan head 1e-4 / DiT+vl-mix 1e-5), vision frozen, WSD schedule, EMA 0.9999,
  plan-dropout, grad-clip, ckpt save (model + EMA + config).
- Verified: 6-step dry-run trains (207M trainable), LR schedule + EMA + save OK.

### ⚠️ Two things that block a REAL training run

1. **YouTube access — SOLVED.** Unauthenticated yt-dlp is bot-blocked, *and*
   YouTube now requires a PO token + JS `n`-challenge. Working recipe (see
   `docs/SETUP_YOUTUBE.md`): cookies.txt + bgutil PO-token server (:4416) + Deno +
   `remote_components=ejs:github` + ffmpeg. Verified: sampled YouTube videos
   **16/16 live**; a 20s slice is ~3–10 MB (download-sections). Twitch VODs ~1/21
   (mostly expired — skip `source=="twitch"`).

2. **Button order (OPEN QUESTION, accuracy risk).** The exact 17-button column
   order NVIDIA trained with is not in the released code. We default to the dataset
   README's documented "standard gamepad layout" (`actions.py:BUTTON_ORDER`). If the
   pretrained policy's null-plan predictions don't match streamers' real actions,
   this ordering is the prime suspect.

**Phase D — real Stage-1 run (DONE, verified):**
- Fetched frames for 100 chunks across 5 YouTube videos (pre-extracted +
  controller-masked PNGs), ran `train_planner.py` 200 steps, batch 8, on real
  frames (2.6 it/s, 78 s, A100).
- **Plan-sensitivity eval** (`planner_poc/eval_plan_sensitivity.py`): a "go left"
  plan vs null plan, raw trained weights:
  - `j_left_x[plan]−[null]` = **−0.0519** (baseline −0.0001) → steers the left
    stick left (left=0.0, center=0.5 in packed space).
  - overall ‖action_plan − action_null‖ = **4.68** (baseline 1.57) → ~3× more plan
    influence than the untrained head.
  - The plan→action channel **demonstrably learned on real data**. Signal is modest
    (small dataset, 200 steps, strong pretrained prior); a production run (thousands
    of examples, more steps) should strengthen it.
- **EMA caveat fixed:** decay 0.9999 over 200 steps left EMA ≈ random init (≈
  baseline in the eval). Added an EMA decay-warmup in `train_planner.py` so short
  runs aren't dominated by the init.

---

## 0. TL;DR

We bolt a small **frozen VLM planner (Qwen3.5-0.8B)** onto the existing **frozen-ish
NitroGen DiT**. The planner reads past frames (+ a plan text), and produces a small
set of **K "plan tokens"** via a learned **Perceiver-style resampler + adapter**.
These K tokens are injected into NitroGen's vision-language (VL) cross-attention
stream alongside the 256 SigLIP image tokens. The DiT then flow-matches an
18-step action chunk that is *steered by the plan*.

Training is two-stage:

- **Stage 1 (alignment / warm-start):** plan text is **synthetic** ("go left").
  VLM frozen. We teach the adapter + DiT that **null plan ⇒ reproduce the
  streamer's real action**, **plan present ⇒ snap toward the plan**. This is a
  contrastive, CFG-style objective. Goal: carve out a working plan→action channel
  and a good adapter init.
- **Stage 2 (real plans / joint):** plan text comes from **stream transcripts**
  (cleaned). Unfreeze the VLM (or LoRA it) and optimize the planner + adapter + DiT
  jointly, so the VLM learns to *generate* useful plans, not just consume them.

No game rollouts are needed at any point — we do **offline CFG on streamer data**,
exactly like the NitroGen paper trains its policy offline.

---

## 1. What NitroGen actually is (verified from `ng.pt`)

**[verified]** Pulled the real `ckpt_config` out of the 1.97 GB checkpoint
(range-fetched only the 83 KB pickle, no full download). The released model:

| Thing | Value | Notes |
|---|---|---|
| Vision encoder | `google/siglip2-large-patch16-256` | 256 image tokens/frame, `vision_hidden_size = 1024` |
| `hidden_size` (DiT width) | **1024** | == `vision_hidden_size` → no projector needed anywhere |
| DiT | **8 layers**, 16 heads × 64, `interleave_self_attention = True` | alternating self-attn / cross-attn blocks |
| DiT `cross_attention_dim` | **null** | cross-attn runs at `hidden_size`=1024, so VL stream is already 1024-d |
| VL self-attention | 4 layers, 16 × 64 | refines VL tokens *before* the DiT |
| `action_dim` | **25** | (padded; real packed dim is 21 — see §4) |
| `action_horizon` | **18** | 18 actions per chunk (paper text says 16; trust the ckpt) |
| flow-matching noise | Beta(α=1.6, β=0.8), s=0.999 | shifted-beta, prioritizes small t |
| `num_inference_timesteps` | 16 | Euler steps |
| `add_pos_embed` | **True** | learned positional embedding on the 18 action tokens |
| `game_mapping` | **None** | the released ckpt has **no game-ID conditioning** — that code path is dormant |
| `max_sequence_length` | 256 | VL budget = exactly the 256 image tokens of 1 frame |
| `action_shift` | 3 | action chunk starts 3 frames after the context frame |

**[verified]** State dict top-level modules (595 tensors): `vision_encoder` (400),
`vl_self_attention_model` (64), `model` = DiT (120), `action_encoder` (6),
`action_decoder` (4), `position_embedding` (1). **No** `game_embedding`,
**no** `mm_projector`, **no** `vis_sep_embedding`. Checkpoint also carries
`step`, `epoch`, `ckpt_config`.

### Forward path (today)
```
current frame ─► SigLIP2 ─► 256 tok @1024 ─► vl_self_attn(4L) ─┐
                                                               ├─► DiT(8L, self/cross alt) ─► action_decoder ─► velocity
noisy actions (B,18,25) ─► action_encoder ─► sa_embs ──────────┘            (flow-matching MSE vs a-ε)
```

**Design fact from the paper that matters:** they found **no benefit from >1 past
frame**, even with temporal gaps — "the initial state already provides sufficient
context." So the DiT stays a single-frame, reactive *system-1*. Our planner is a
**separate semantic channel**, not "more frames for the DiT." This keeps the design
clean: DiT still sees 1 frame; the *planner* is what gets temporal context.

### Why this is the intended extension, not a hack
Paper's Limitations: *"NitroGen cannot plan over long horizons or follow language
instructions… we aim for it to serve as a foundation for future generalist agent
development, where post-training for language-following… can be applied to enhance
planning."* We are literally building their stated future work.

---

## 2. The planner backbone (verified)

**[verified]** `Qwen/Qwen3.5-0.8B` is a **native VLM** (`Qwen3_5ForConditionalGeneration`,
`model_type: qwen3_5`, Feb 2026), not a text model. Real config:

- **text `hidden_size = 1024`** — *exactly* NitroGen's width. The adapter is
  fundamentally a **1024 → 1024** map (plus the K-query resampler). Zero dimension
  juggling anywhere in the system.
- Native **vision + video** input (`image_token_id`, `video_token_id`, a 12-layer
  ViT, patch 16, spatial-merge 2, **temporal-patch 2** → it natively ingests
  *sequences of frames*, perfect for "past frames at a sample rate").
- Hybrid attention (Gated-DeltaNet linear layers + full attention every 4 layers),
  head_dim 256, 24 layers, 262K context.
- `transformers==5.12.1` in our venv **has full `qwen3_5` support** [verified:
  `Qwen3_5ForConditionalGeneration`, `Qwen3_5TextModel`, `Qwen3_5VisionModel`,
  `AutoConfig` all import and resolve].

**Consequence — the "frames into the planner?" question is settled: YES, and it's
free.** Because Qwen3.5 is a real VLM, the *same backbone* serves Stage 1 and
Stage 2 (true warm-start, no swap), and it sees frames/video natively without any
SigLIP-glue. The earlier "text-only now, swap later" idea is **dropped** — it would
have broken the warm-start (adapter trained on the wrong hidden-state manifold).

---

## 3. Proposed architecture

```
 past frames @ sample-rate           current frame (single)
 (e.g. 4–8 frames over the              │
  preceding ~2–10 s)                    ▼
        │                        ┌──────────────┐
        ▼                        │  SigLIP2     │  (NitroGen, frozen)
 ┌───────────────┐               └──────────────┘
 │ Qwen3.5-0.8B  │  + plan text         │ 256 tok @1024
 │  VLM (frozen  │  (Stage1: synthetic  │
 │  in Stage 1)  │   Stage2: transcript)│
 └───────────────┘                      │
        │ last-layer hidden states      │
        │   H ∈ (B, L, 1024)            │
        ▼                               │
 ┌──────────────────────────┐          │
 │ PlanResampler            │          │
 │  K learned queries        │          │
 │  cross-attend to H        │          │
 │  (1–2 layers)             │          │
 └──────────────────────────┘          │
        │ (B, K, 1024)                  │
        ▼                               │
 ┌──────────────────────────┐          │
 │ PlanAdapter              │          │
 │  LN→Linear→GELU→Linear→LN│          │
 │  (1024 → 1024)            │          │
 └──────────────────────────┘          │
        │ plan tokens (B, K, 1024)      │
        ▼                               ▼
  VL stream:  [PLAN × K]  +  [IMG × 256]  ─► vl_self_attn(4L) ─► DiT cross-attn ─► actions
```

### Key design choices

1. **K resampled tokens, not "last token".** A single 1024-d vector is a hard
   bottleneck (esp. for Stage-2 long transcripts). A Perceiver/Q-former-style
   resampler with **K learned queries** (start **K=8**) cross-attending to the
   VLM hidden states gives a fixed-width plan representation **decoupled from plan
   text length**. Same module works for short synthetic phrases and long
   transcripts. (BLIP-2 Q-former / Flamingo Perceiver Resampler pattern.)

2. **A new, independent `_PLAN_TOKEN` slot.** We do *not* piggyback on the dormant
   game-ID code. New token id `_PLAN_TOKEN = 7`, prepended to `vl_token_ids`.
   `max_sequence_length` 256 → **256 + K**. The tokenizer already left-pads variable
   VL length, so this is a tiny change.

3. **Plan-dropout → reuse CFG.** With probability `p_drop` (~0.1–0.2) we replace the
   plan tokens with a single learned **null-plan embedding** (broadcast to K). This
   gives one model that supports both "no plan" and "plan", and lets us crank plan
   influence at inference via the **already-existing `get_action_with_cfg`**
   (cond = real plan, uncond = null plan). No new sampling code.

4. **Adapter/resampler output is 1024-d** to match `vision_hidden_size == hidden_size`
   and DiT `cross_attention_dim=null`. Nothing to reconcile.

### New modules (file: `nitrogen/planner.py`)
- `PlanEncoder`: wraps Qwen3.5; `encode(frames, plan_text) -> H (B,L,1024)`.
  In Stage 1, run in **encode/teacher-forced** mode (no generation). Frozen ⇒
  hidden states are **cacheable** so the VLM never runs in the training loop.
- `PlanResampler`: K learned queries, 1–2 cross-attn layers over H.
- `PlanAdapter`: LN→Linear→GELU→Linear→LN, 1024→1024.
- `null_plan`: a single learned (1024,) parameter, broadcast to (K,1024).

### NitroGen changes (`flow_matching_transformer/nitrogen.py`)
- `NitroGen_Config`: add `planner_cfg` (backbone path, K, resampler depth,
  adapter hidden, `p_drop`, freeze flags).
- `__init__`: build planner + resampler + adapter + null_plan; honor freezes.
- `prepare_input_embs`: place plan tokens at `_PLAN_TOKEN` positions (same
  `masked_scatter` pattern as the game-id branch). Handle dropped/null plans.
- `forward` / `get_action` / `get_action_with_cfg`: compute plan tokens, apply
  plan-dropout, inject. CFG uncond branch = null plan.
- `set_trainable_parameters`: add `tune_planner`, `tune_plan_adapter`,
  `tune_plan_resampler`.

### Tokenizer changes (`mm_tokenizers.py`)
- Add `_PLAN_TOKEN = 7`, `n_plan_tokens = K`.
- Prepend `[_PLAN_TOKEN] * K` in `_build_token_ids`.
- Carry `plan_text` / tokenized planner inputs / `plan_dropped` through `encode`.

---

## 4. Action representation (verified from dataset + ckpt)

**[verified]** Dataset parquet schema (per frame, 60 fps):
- **17 boolean button columns**: `back, dpad_down, dpad_left, dpad_right, dpad_up,
  east, guide, left_shoulder, left_thumb, left_trigger, north, right_shoulder,
  right_thumb, right_trigger, south, start, west`.
- **2 joystick columns** `j_left`, `j_right`, each a `[x, y]` list in `[-1, 1]`.
  `(-1,-1)` is **top-left**.

So packed action = `[17 buttons] + [j_left xy] + [j_right xy]` = **21 dims**, then
zero-padded to `max_action_dim = 25`. The tokenizer's `pack_actions`
(new layout) is `concat([buttons, j_left, j_right])` with joysticks renormalized to
`[0,1]`. **Note** the existing `mm_tokenizers.pack_actions` assumes the caller
supplies `buttons` already grouped — our dataset loader must assemble the 17-wide
button vector in the column order above.

**[verified]** Chunking: each dataset chunk = **20 s = 1200 frames @ 60 fps**. The
model consumes **18-action chunks** with `frame_spacing=18`, `action_shift=3`. So a
training example is: 1 context frame, then the next 18 control steps (after some
downsample from 60 fps — see open question Q3).

### The on-screen controller must be masked
**[verified]** `metadata.json` gives `bbox_controller_overlay = [xtl, ytl, w, h]`
in pixel space (and optional `bbox_game_area`, `bbox_others`). The paper masks the
overlay so the policy can't cheat by reading the gamepad off-screen. **Any frame we
feed to SigLIP *or* the planner must have this bbox masked**, or we are off the
training distribution.

---

## 5. THE BIG DATASET FINDING — there are no pixels

**[verified]** `nvidia/NitroGen` (the HF *dataset*) ships **action labels only**.
102 files = 100 × `SHARD_xxxx.tar.gz` (~1.5 GB each, **~150 GB total**) + README.
Each shard expands to:
```
SHARD_0000/<video_id>/<video_id>_chunk_0030/
    ├── actions_raw.parquet        # per-frame raw gamepad state (1200 rows)
    ├── actions_processed.parquet  # quality-filtered/remapped version
    └── metadata.json              # url, game, controller, bbox_controller_overlay, timestamps
```
**The frames are NOT in the dataset.** `metadata.json.original_video.url` points to
the source (mostly YouTube), with `start_time`/`end_time`/`start_frame`/`end_frame`.
README is explicit: *"This dataset only includes the gamepad action labels."* and
*"reproducing results requires additional filtering, such as IDLE frame filtering"*
(idle filtering is **not** pre-applied).

### Implications (important — read before building the data pipeline)
1. **You must build a video-ingestion pipeline**: for each chunk, download the
   source video (yt-dlp), seek to `start_time`, extract frames at the needed rate,
   mask `bbox_controller_overlay`, resize to 256×256 for SigLIP, and a
   (possibly different) res for the Qwen ViT. This is *the* real infra cost — but
   it's plain data engineering, **not game rollouts**. Your instinct to avoid
   running games is correct and fully supported by this data design.
2. **Transcripts for Stage 2 are not in the dataset either** — but the **video URL
   is**, so YouTube auto-captions / Whisper on the audio gives you the streamer
   transcript aligned to `start_time`. That is your real-plan source. The dataset
   hands you the time alignment for free (`start_time`, `end_time`).
3. **Storage discipline:** at 150 GB for *actions alone* and far more for frames,
   do **not** download everything. Stream shards, extract frames on the fly, cache
   only decoded 256×256 frames (or even just SigLIP/Qwen features) for the chunks
   you actually use. For PoC/early training, a few shards + a few hundred videos is
   plenty.

### Sample statistics (1 partial shard, ~1,870 valid chunks) **[verified]**
- 60 fps, 1200 frames/chunk.
- **Action density per chunk**: mean 0.66, median 0.73, p10 0.16, p90 0.97.
- **~27 % of chunks are < 0.5 density** → these are your **idle-ish** chunks (the
  paper *discards* the low-density ~45 %; you *want* them for synthetic-plan Stage 1).
- This shard: `game = "other"` for all sampled chunks (game labels are sparse /
  this video is unlabeled); controllers seen: ps4, switch, xboxone.

### Idle-frame extraction (answers your Stage-1 concern)
You don't need a fancy detector. **Idle = low ground-truth action density**, which
is directly computable from the parquet (you already have the labels). Concretely:
take chunks/sub-windows where button+joystick activity is low → the streamer was
coasting → "multiple actions are reasonable" → ideal substrate for a synthetic
counterfactual plan. Optional refinement: **flow-prior entropy** — run NitroGen's
`get_action` with N different noise seeds and measure spread of denoised chunks;
high spread ≈ multimodal ≈ genuinely ambiguous state. But density filtering alone is
enough to start.

---

## 6. Two-stage training recipe

### Stage 1 — Alignment / warm-start (VLM frozen)

**Data construction (per example):**
- Sample a frame `o` (prefer idle/low-density windows) from a chunk; mask the
  controller bbox; build the streamer's **real** 18-action target `a_real` (the
  action chunk that actually followed `o`).
- Build a small set of **synthetic plan templates** mapped to **synthetic action
  targets** `a_plan` that *start with* the plan's intent:
  - "go left" → left d-pad / left-stick deflection for the first few steps
  - "go right", "jump" (south/A), "brake", "attack", "do nothing", etc.
  - Keep them *atomic* and *1-chunk-horizon* (18 steps ≈ a fraction of a second to a
    couple seconds depending on downsample), since a chunk can't express long plans.
- Emit **two example types**, mixed:
  - `(o, null plan) → a_real`   (preserve original NitroGen behavior)  — **majority**
  - `(o, synthetic plan) → a_plan`  (teach plan-following)  — **minority**
  Suggested mix ~70/30 null/plan. (Plan-dropout implements the null branch.)

**Trainable:** adapter + resampler (full). DiT + `vl_self_attn`: **LoRA**
(safer than full-FT on biased synthetic data; the `set_trainable_parameters`
plumbing already exists, LoRA is a small add). VLM: **frozen** ⇒ cache its hidden
states. Maintain **EMA (0.9999)** — the paper's results are all EMA, the released
ckpt is presumably EMA, and EMA protects the policy.

**Loss:** unchanged flow-matching MSE on velocity `(a - ε)`. The null-vs-plan
behavior is enforced entirely by the **data + plan-dropout**, *not* by a new loss
term. (Optional later: an explicit contrastive/divergence term that pushes
`v(plan) ≠ v(null)` on idle frames, but start without it.)

**Why this is a real warm-start, not throwaway:** it initializes the adapter +
resampler on **image-grounded Qwen hidden states** (same manifold as Stage 2),
and teaches the DiT a working plan→action pathway, so Stage 2 doesn't start from a
random projection that the VLM gradients would have to fight.

### Stage 2 — Real plans / joint

**Data:** transcripts (Whisper/auto-caption) aligned to chunk `start_time`, cleaned
into plan-like spans ("now I'll go for the boss", "let's grab that item"). Pair each
plan span with the frame(s) and the streamer's **real** action chunk. Now the plan
is *consistent* with what the streamer did (not a counterfactual), so the model
learns plan→action grounding from real behavior.

**Trainable:** unfreeze VLM (or **LoRA** it to save memory + reduce forgetting) +
adapter + resampler + DiT. Now the VLM *generates* plans; supervise the text head
with the transcript spans (LM loss) **jointly** with the action flow-matching loss:
`L = L_flow + λ · L_LM`. Because generation is discrete (non-differentiable through
sampling), use teacher-forced LM loss for the text and feed the *generated/forced*
hidden states into the resampler — the resampler→adapter→DiT path is identical to
Stage 1, which is exactly why Stage 1 warm-starts it.

**Hyperparams (start from paper):** AdamW, wd 0.001, WSD schedule, constant LR 1e-4
for new params; **1e-5** for the pretrained DiT (or LoRA); even lower / LoRA for the
VLM. EMA 0.9999. Flow t ~ Beta(1.6, 0.8). 16 Euler steps at inference.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| DiT forgets pretrained policy on biased synthetic data | LoRA on DiT, high null-plan ratio, small LR, EMA, keep real-action batches |
| Plan token ignored (model conditions only on frame) | plan-dropout + CFG amplification; monitor `‖v(plan) − v(null)‖` on idle frames as a *plan-sensitivity* metric |
| Single-chunk horizon can't express long plans | Stage 1 uses atomic plans; for long-horizon, persist plan tokens across consecutive chunks (plan changes slower than actions) |
| Synthetic action targets are crude / unrealistic | keep them simple & directional (only supervise first few steps' intent), rely on Stage 2 real data for realism |
| Video ingestion is the real cost | stream shards, cache only used frames/features, start with a few shards |
| Off-distribution frames | always mask `bbox_controller_overlay`; match SigLIP 256×256 preprocessing |
| `action_dim` mismatch (21 vs 25 vs paper's 24) | pack 21, pad to 25 — matches ckpt; do **not** trust paper's "16/24" |

---

## 8. Open questions for you (Q-list)

- **Q1 — Past-frame sampling for the planner.** How many frames & over what window?
  (e.g., 4–8 frames over the preceding 2–10 s.) Drives Qwen input cost.
- **Q2 — K (number of plan tokens).** Proposing **8**. OK?
- **Q3 — Control-frequency / downsample.** Dataset is 60 fps; model takes 18 actions
  with `frame_spacing=18`, `action_shift=3`. Need to confirm the exact frame→action
  decimation NitroGen trained with (likely ~30→? Hz). I can probe more, but the
  *training* recipe isn't in the repo, so we may have to pick a sensible value and
  validate by behavior.
- **Q4 — DiT: LoRA vs light full-FT** in Stage 1. Proposing LoRA.
- **Q5 — Synthetic plan vocabulary** for Stage 1 (which atomic intents, and the
  hand-written action target for each). I'll draft a starter set; you curate.
- **Q6 — Stage 2 transcript pipeline:** Whisper-large-v3 on audio vs YouTube
  auto-captions? (Captions are cheaper & time-aligned; Whisper is higher quality.)

---

## 9. What exists after tonight

- `ckpts/nitrogen/ng.pt` — downloaded **[verified, 1.97 GB]**.
- `ckpts/qwen35-0.8b/` — downloaded **[verified, 1.7 GB]**.
- `.venv` (uv, py3.12) with torch 2.11+cu130, vllm 0.23, transformers 5.12.1,
  polars, pyarrow — **[verified GPU: A100 80GB]**.
- `files/sample_chunk/` — one real dataset chunk (parquets + metadata) for reference.
- `scripts/poc_planner.py` — standalone PoC wiring planner→resampler→adapter→DiT,
  run end-to-end with random data; verifies shapes, grad flow, and that
  **null vs plan produce different velocities**. See PoC notes at the bottom.

Nothing in the shipped `nitrogen/` package was modified yet — the PoC imports the
real modules but adds the new pieces standalone, so your repo is untouched and the
design is de-risked before we edit core files.

---

## 10. PoC results — **PASS** ✅

Ran `files/poc_planner.py` on the A100 against the **real** `ng.pt` + **real**
`Qwen3.5-0.8B`. Full log in `files/poc_results.txt`. Highlights:

```
[1] NitroGen load: hidden 1024, action 25/horizon 18, DiT 8L interleave=True
    state_dict load: 0 real-missing, 0 unexpected   <-- clean load after the shim
[2] Qwen3.5-0.8B: 0.85B params (frozen)
[3] PlanHead (resampler+adapter+null): K=8, 29.4M NEW trainable params
[5] VLM hidden states (B,L,1024); dim == NitroGen 1024 -> NO projector needed
[6] forward+backward: flow loss 2.04
    grad norm resampler = 2.15   adapter = 1.13   -> GRAD FLOW: PASS
[7] ||v_plan - v_null|| = 0.54 (rel 1.4%)          -> PLAN CHANNEL HAS EFFECT: PASS
[8] Qwen3.5 image+text hidden (1,81,1024)          -> VLM ingests frames: PASS
OVERALL: *** PoC PASS ***  (8.7s)
```

**What this proves:**
- The architecture wires end-to-end against the shipped checkpoint with **zero
  missing/unexpected** weights (after the one-line SigLIP shim, §11).
- Qwen3.5's hidden size is **exactly** NitroGen's 1024 — the adapter is a clean
  1024→1024, the warm-start claim holds, **no projector anywhere**.
- Gradients flow into the **new** modules (resampler 2.15, adapter 1.13).
- The plan channel **changes the predicted velocity** (null vs plan differ by 1.4%
  even at random init) — i.e. the conditioning pathway carries signal into the DiT.
- Qwen3.5 **natively ingests a frame** (image+text → hidden states), confirming
  "frames into the planner" needs no SigLIP-glue.

**Notes / caveats (honest):**
- `null_plan.grad == 0` in step [6] is **correct**, not a bug: that step does not
  drop the plan, so `null_plan` isn't in the graph. It receives gradient only on
  plan-dropped examples (Stage-1 null branch).
- The "effect" magnitude (1.4%) is with **random-init** plan modules; after Stage-1
  training on the contrastive null-vs-plan objective, this should grow substantially
  on idle frames (and we'll track `‖v(plan)−v(null)‖` as the plan-sensitivity metric).
- Perf nit: Qwen3.5 uses Gated-DeltaNet; installing `causal-conv1d` + `flash-linear-
  attention` enables its fast path (the run logged a fallback-to-torch warning). Not
  needed for correctness; worth it for Stage-2 throughput.
- The PoC calls NitroGen submodules directly (encode_images → vl_self_attn → DiT →
  decoder) and prepends the K plan tokens to the VL stream. The real implementation
  will instead route this through `prepare_input_embs` via the new `_PLAN_TOKEN`
  (§3), but the PoC proves the math/shapes/grads are right first.

---

## 11. The one required env fix (transformers 5.12.1)

`transformers>=5.x` changed `SiglipVisionModel`: it no longer exposes
`.vision_model` (the class **is** the vision model; children:
`embeddings/encoder/post_layernorm/head`). NitroGen's `__init__` does
`self.vision_encoder = model.vision_model` → `AttributeError`.

We **cannot** downgrade transformers (Qwen3.5 needs 5.12.x). Fix options:
- **PoC (used):** monkeypatch `SiglipVisionModel.vision_model = property(lambda s: s)`.
- **Repo (later):** change `nitrogen.py:186` to set `self.vision_encoder = model`
  (the SiglipVisionModel itself) when the `.vision_model` attr is absent. State-dict
  keys (`vision_encoder.embeddings.*` etc.) are **unchanged** because child module
  names are identical — verified by the 0-missing/0-unexpected load.

This is the only environment incompatibility found; everything else in the shipped
`nitrogen/` package runs as-is under the new stack.
