# Experiment Log — Plan-Conditioned NitroGen

Chronological, exact-config record of every experiment. Newest at bottom.
Conventions: all runs on 1×A100-80GB, torch 2.11+cu130, transformers 5.12.1,
NitroGen `ng.pt` (EMA, action_dim=25, horizon=18), planner Qwen3.5-0.8B (frozen).

> ⚠️ Evaluation philosophy (per @namak-kun): the success signal is **counterfactual**:
> (1) **steering** — does conditioning on plan P change the policy *in P's direction*?
> (2) **null-invariance** — under the null plan, does the policy stay ≈ the
> pretrained NitroGen (we must not damage the base policy)?
> Always report BOTH.

---

## Paper ↔ released-checkpoint discrepancies (reference)

| quantity | paper | released `ng.pt` | notes |
|---|---|---|---|
| actions per chunk (`action_horizon`) | 16 | **18** | ckpt model_cfg/tokenizer_cfg/modality_cfg all 18 |
| action_dim | 24 | **25** | 21 buttons + 4 joystick |
| denoising steps (k) | 16 | 16 | agree; `num_inference_timesteps`. Likely source of "16 actions" confusion |
| button count | 16 (eval) / 20 (draft) | **21** | `BUTTON_ACTION_TOKENS` |
| joystick dims | last 4 | **dims 21–24** | confirmed by per-dim stats + corr +0.45 |

`inference_viz.py:19` docstring still says "16x2" (stale) — corroborating the drift.
Always trust the checkpoint config, not the paper.

---



**Per-example construction** (`NitrogenPlanDataset.__getitem__`):
1. Pick a chunk; pick a context-frame index; the action target spans the next
   `action_horizon=18` control steps, decimated `frame_stride=2` (60→30 Hz),
   starting `action_shift=3` frames after the context frame.
2. Branch:
   - **NULL** (prob `1-plan_ratio`): `plan_dropped=True` (plan tokens → learned
     null embedding). Target = streamer's **real** 18-step chunk. Purpose: keep
     base policy (null-invariance).
   - **PLAN** (prob `plan_ratio`, optionally only on idle windows): sample a
     synthetic plan (e.g. "go left"); frozen Qwen encodes the text → resampler →
     K plan tokens. Target = **counterfactual** chunk (see "intent window").
3. Loss = flow-matching MSE on velocity, masked by `actions_mask`. With
   `joystick_only_loss=True`, only the 4 joystick dims (21–24) are supervised
   (now that buttons are mapped, this can be turned off to supervise buttons too).
4. `plan_dropout` (~0.15) also drops PLAN examples to null → classifier-free-guidance
   training so inference can amplify the plan via `get_action_with_cfg`.

**Intent window — how much of the chunk the plan controls (THE key design knob):**
Current = **hard split** via `SyntheticPlanSampler(n_intent_steps=6, blend_real_tail=True)`:
the **first `n_intent_steps`=6 of 18** steps are set to the plan's direction; the
remaining 12 steps are filled with the streamer's **real** action. So a plan
"redirects the immediate ~6 steps, then the reactive policy resumes." It is NOT
first-action-only, NOT the whole chunk, NOT decaying — it's a configurable hard
split. Our design choice (no paper precedent — NitroGen has no plan conditioning).
Alternatives to evaluate: first-action-only; full-chunk; smooth **decaying** weight;
per-plan-type windows. OPEN design question.

**Control-scheme ambiguity (d-pad vs stick) — modality-aware plans:** "go left" is
ambiguous — d-pad-left (dim 2) and left-stick-left (dim 21) are both "left", and
which moves the character is game-dependent (3D→stick, many 2D/retro→d-pad; in our
40-chunk sample 39 stick / 1 d-pad, but the full 1000-game set has far more d-pad).
`SyntheticPlanSampler(modality="auto")` detects the streamer's movement modality
per chunk (`actions.movement_modality`) and drives the matching control(s), so
directional counterfactuals stay on-distribution. Requires button supervision
(`joystick_only_loss=False`, valid now the button order is verified).

**Optimizer:** AdamW (β 0.9/0.95, wd 1e-3), split LR (plan head high, DiT+vl-mix
low), vision frozen, planner frozen, WSD schedule, EMA (with decay-warmup).

---

## EXP-000  Action-layout investigation (CRITICAL)  — 2026-06-16

**Question:** What is NitroGen's true 25-d action layout, and is our assumed
`[buttons(17), j_left(2), j_right(2), pad(4)]` (joysticks at dims 17–20) correct?

**Method:** Ran pretrained NitroGen (null/no-planner) `get_action` on 12–70 real
frames from cached dataset slices; dumped per-dim output stats; correlated model
output dims vs ground-truth actions from the chunk parquet (action_shift≈3).

**Findings:**
- Per-dim stats (216 steps): dims **0–20 binary** (0/1, occasional 1.0) → 21
  button dims. dims **21–24 continuous ≈0.50** → joysticks.
- Joystick correlation (pretrained pred vs ground truth), dims **21–24**:
  - `j_left_x` corr **+0.475**, `j_left_y` **+0.425** (630 samples).
  - `j_right_*` std≈0 (camera idle in these 2D/movement games) → uninformative.
  - mean corr **+0.45** → **CONFIRMS** layout + frame→action time alignment.
- Earlier assumption (joysticks at 17–20) was **WRONG**. True layout:
  **`[buttons(21) @0–20, j_left @21–22, j_right @23–24]`** — matches the shipped
  `unpack_actions` (`buttons=[:-4]`, `j_left=[-4:-2]`, `j_right=[-2:]`).
  The shipped `pack_actions` (17 buttons → pad to 25) is the inconsistent piece.

**Button identity (17 dataset cols → 21 model slots): RESOLVED.** The authoritative
order is `nitrogen/shared.py:BUTTON_ACTION_TOKENS` (21 entries), used by
`scripts/play.py:180` `zip(TOKEN_SET, button_vector)` to decode model output →
gamepad. Model dims 0..20:
`back, dpad_down, dpad_left, dpad_right, dpad_up, east, guide, left_shoulder,
left_thumb, left_trigger, north, right_bottom, right_left, right_right,
right_shoulder, right_thumb, right_trigger, right_up, south, start, west`.
The dataset's 17 columns are a subset; 4 model slots
(`right_bottom/left/right/up`) have no dataset column (left 0). Our weak
correlation independently corroborated this: confident matches dim8→left_thumb,
dim14→right_shoulder, dim16→right_trigger all land exactly on these indices.
Fixed in `actions.py` (`MODEL_BUTTON_TOKENS`, `DATASET_COL_TO_MODEL_DIM`); button
plans now valid (jump=south=dim18, attack=west=dim20).

**Impact:** Invalidates EXP-001 (trained joystick targets into button dims 17–18).
Action: fix packing to the true 25-d layout; redo Stage 1 reading sticks at 21–24.

**Artifacts:** `planner_poc/verify_button_order.py`, `/tmp/button_corr.npy`.

---

## EXP-001  First Stage-1 alignment (INVALID — wrong action layout)  — 2026-06-16

**Config:** `train_planner.py`, 100 chunks / 5 YouTube videos (real frames,
controller-masked), 200 steps, batch 8, warmup 20, plan_ratio 0.35,
lr_plan 2e-4, lr_dit 1e-5, vision frozen, EMA 0.9999 (no warmup), K=8 plan tokens,
plan vocabulary = {left,right,up,down,jump,attack,idle}.

**Result (as measured, on WRONG dim):** "go left" raw weights `Δ(dim17) = −0.052`
vs null; ‖Δaction‖ 4.68 vs baseline 1.57. EMA ≈ baseline (decay too high for 200
steps).

**Why invalid:** dim 17 is a *button* slot, not `j_left_x` (which is dim 21). The
joystick targets were packed into dims 17–18. The loss-down / influence-up shows
the channel *can* learn, but the steering target was the wrong dimension.

**Fixes triggered:** (a) correct action packing (EXP-000 layout); (b) EMA decay
warmup added; (c) eval must read sticks at dims 21–24.

---

## EXP-002  Stage-1 corrected layout, varied counterfactuals  — 2026-06-16

**Config:** `train_planner.py`, 66 chunks / 11 YouTube videos (real masked frames),
**corrected 18×25 layout** (joysticks @21–24, 21-button order from `BUTTON_ACTION_TOKENS`),
modality-aware directional plans (`modality="auto"`, `joystick_only_loss=False`),
11 grounded plans (8 directional + 3 compound) + idle, plan_ratio 0.5,
n_intent_steps 6 (hard split), 400 steps, batch 8, warmup 40, lr_plan 2e-4,
lr_dit 1e-5, EMA 0.9999 (+warmup), K=8. Loss 0.4→0.04.

**Counterfactual eval** (`eval_counterfactual.py`, 10 frames, EMA weights):
- **Steering (Δ plan − null, first 6 steps):**
  - `go left` → dpad_left **+1.00**, stick proj +0.21 ✓
  - `go up`   → dpad_up **+0.05**, stick proj +0.14 ✓
  - `go right`/`go down` → dpad ~0, stick proj NEGATIVE (wrong sign).
  - **Stick Δ is IDENTICAL across all four plans** ((−0.21,−0.14)): the stick
    pathway learned a *generic* "plan-present" shift, NOT a direction-specific one.
    The **d-pad** pathway IS direction-specific (left/up correct).
- **Null-invariance ‖null − base‖:** baseline-untrained 1.4 → trained 2.2. The null
  branch perturbs the base policy somewhat more after training (mild concern).

**Interpretation / honesty:**
- The plan channel **demonstrably learned** and is **partially** direction-specific
  (d-pad left/up). But it is NOT yet a clean per-direction steerer: the continuous
  stick collapsed to a generic shift, and right/down are weak.
- Likely causes: only 66 chunks / 400 steps (undertrained); continuous-stick
  steering harder than binary d-pad; direction imbalance in the tiny dataset.
- **Eval caveat:** ‖null−base‖ uses *sampled* trajectories (get_action draws noise);
  plan-model vs base-model consume RNG differently → metric is noisy (baseline
  varied 1.4–2.2 across runs). TODO: measure null-invariance on **velocity at fixed
  noise/timestep** (deterministic), per-plan steering likewise.

**Next:** (a) deterministic velocity-based eval; (b) scale data (more videos/chunks)
and steps; (c) revisit intent-window (decaying vs hard-split); (d) balance plan
direction sampling.

---

## FUTURE IDEAS (untested — do not implement yet)

- **Intra-chunk action mapping → simulated inter-chunk.** NitroGen implicitly
  predicts how the game state evolves while executing its 18-action chunk (the DiT
  must, to output coherent multi-step actions). Hypothesis: this implicit
  state-prediction can be *exploited synthetically*. If the model's behavior over
  18 actions is (under perfect state prediction) consistent with its behavior over
  36, we could **simulate inter-chunk / longer-horizon** plan conditioning without
  real rollouts by chaining chunks. MUST be tested empirically before relying on it
  (it's an assumption about the model's internal consistency, not a fact).
  Raised by @namak-kun 2026-06-16.

- **Cross-chunk plan phase (THE core long-horizon gap).** The VLM/planner fires
  only every A chunks, so a static plan token conditions multiple consecutive
  chunks identically — the only per-chunk difference is the current frame. WITHIN a
  chunk, temporal structure ("left then right") works via the DiT's positional
  embeddings on the 18 action tokens; ACROSS chunks there is no shared clock, so a
  static token cannot make chunk i ≠ chunk i+1 except reactively. Missing ingredient
  = a **plan-phase/progress signal**. Design options (untested):
  (1) feed a `t_plan ∈ [0,1]` progress scalar to the DiT (AdaLN, like the flow
      timestep) that advances per chunk;
  (2) VLM emits a SEQUENCE of plan tokens (one slice per upcoming chunk), a cursor
      selects chunk i's slice — decouples VLM frequency from per-chunk conditioning;
  (3) re-run the VLM each chunk with growing frame history (expensive).
  Training any of these needs cross-chunk supervision → ties directly to the
  multichunk-simulation hypothesis above. Raised by @namak-kun 2026-06-16.

---

## EXP-003  Schedule-based plans, deterministic eval  — 2026-06-16

**Config:** schedule-based plans (HOLD/TAP/SEQ2/SEQ3/IDLE + jump/attack), balanced
group sampling, modality-aware, 66 chunks / 11 videos, 800 steps, batch 8, plan_ratio
0.6, lr_plan 2e-4, lr_dit 1e-5, EMA(+warmup), K=8. Loss 0.3→0.03.

**Deterministic eval** (`eval_deterministic.py`: fixed noise + fixed timestep,
read predicted VELOCITY, avg over 5 timesteps × 12 frames; proj>0 = correct dir):
| plan | trained EMA | base(untrained) |
|---|---|---|
| hold_left | proj +0.035, dpad **+0.080** ✓ | +0.019 |
| hold_right| proj −0.043, dpad −0.034 ✗ | −0.019 |
| hold_up   | proj −0.066, dpad +0.028 ✓(dpad) | −0.014 |
| hold_down | proj +0.066, dpad +0.050 ✓ | +0.010 |
| seq_left_right | dpad 1st +0.038 / 2nd +0.031 | — |
- **null-invariance ‖v_null − v_base‖: 1.92 (untrained) → 3.18 (trained) — WORSE.**

**Honest interpretation:**
- The plan channel learns *some* direction structure (left/down/up d-pad correct,
  larger magnitude than baseline), but it is **biased** (right is wrong-signed) and
  **SEQ halves don't clearly differentiate** (both halves push similar dirs). This
  echoes EXP-002's left/up bias.
- **null-invariance DEGRADED** (1.9→3.2): tuning the DiT on 66 chunks drifts the
  base policy under the null plan — we are damaging system-1. This is the most
  important negative result.
- Root cause is almost certainly **data scale** (66 chunks is tiny) + DiT drift.

**Actions to consider (not yet run):**
- Protect null-invariance: freeze DiT (train only plan_head+adapter), OR LoRA on
  DiT, OR much lower lr_dit, OR raise null-example ratio.
- Scale data 10–50× (more videos/chunks); balance directions in *data* not just
  plan sampling.
- The velocity-projection metric is still noisy; consider many more frames.

- **CHOSEN cross-chunk design (K×A blocks).** @namak-kun 2026-06-16: resampler emits
  K×A query tokens, split into A blocks of K. A per-chunk cursor a∈[0,A) selects
  block a to inject for chunk i. Advantage over a t_plan scalar: physically swapping
  the injected K tokens needs NO phase signal in the DiT — the existing K-token
  injection path is reused. Architecture change is tiny (num_queries K→K*A + block
  selector). BUT: training the blocks to differ ("phase 0=left, phase 1=right")
  requires multi-chunk targets (single-chunk Stage 1 degenerates to K, trains only
  one block). So: implement K×A only once multi-chunk supervision exists (real
  consecutive chunks OR the state-prediction simulation hypothesis). Until then it
  adds untrained params.

---

## Null-plan implementation notes + bug (2026-06-16)

- **Null = learned (K,1024) param** `PlanHead.null_plan`; substituted when
  `dropped=True` via `plan = (1-m)*plan + m*null`.
- **`dropped` source:** dataset `plan_ratio` (≈1-plan_ratio of examples are explicit
  nulls with real-action targets). 
- **BUG (dead code):** model-level CFG dropout `nitrogen.py:563 if dropped is None
  and self.training: dropped = rand < plan_dropout` NEVER fires, because the
  collator always sets `plan_dropped` (dataset.py:222) → `data.get('plan_dropped')`
  is never None. So `planner_cfg.plan_dropout` is inert; we are NOT training proper
  CFG dropout on plan examples → `get_action_with_cfg` won't guide as intended.
  FIX: apply dropout in collator, or let model-level override even when provided.
- **Null-invariance metric was unfair:** base model has 256 VL tokens, null-plan
  model has 256+K (K null embeddings still cross-attended). "trained-null vs base"
  conflates the K-extra-token effect with training drift. Correct metric =
  **trained-null vs untrained-null** (same 256+K arch). Perfect null-invariance is
  impossible by construction; null embedding must learn to cancel the K tokens.

---

## EXP-004  Null-mode A/B (learned vs masked), fair metric  — 2026-06-16

**Setup:** identical data (66 chunks/11 vids), 600 steps, plan_ratio 0.5, lr_plan
2e-4, **lr_dit 1e-5** (DiT trainable), schedule plans. Two null modes:
- `learned`: dropped rows substitute a learned (K,1024) null embedding.
- `masked`: dropped rows keep plan tokens but mask the K plan positions out of BOTH
  the VL self-attention AND the DiT cross-attention. Verified (get_action): masked
  null == base to 5e-3 (bf16) when DiT weights are unchanged.

**Deterministic eval** (fixed noise+timestep, velocity; both vs base AND vs their
own untrained ref to isolate *training* drift):
| metric | learned | masked |
|---|---|---|
| ‖v_null − v_base‖ | 3.39 | **2.63** |
| ‖v_null − v_untrained‖ (training drift) | 3.49 | 2.63 |
| hold_left/up/down steering | OK | OK |
| hold_right | weak (persistent right bias) | weak |
| seq halves differentiate | partly | partly |

**Key interpretation (important):**
- **Masked mode preserves the base policy better under null (2.63 vs 3.39 drift)**
  with comparable steering → masked wins this A/B.
- BUT masked does NOT give *free* null-invariance once the DiT is trained: masking
  removes plan-token *contamination*, but the **shared DiT weights still drift**
  (we train lr_dit=1e-5). For masked, vs-base == vs-untrained == 2.63 = pure weight
  drift (confirms masked-null with base weights == base exactly).
- ⇒ **To get true null-invariance, FREEZE the DiT** (+ vl-mixing): then masked-null
  == base *architecturally*, and only plan_head/adapter learn to steer through the
  frozen DiT cross-attention. This is the clean next experiment.
- Persistent `hold_right` weakness across all runs (EXP-002/003/004) → likely data
  direction imbalance (stick-dominant, few right-movement chunks) — fix in data.

**Bug fixed during eval:** the manual velocity path wasn't applying the VL
self-attention mask (only the DiT cross-attn mask) → masked-null looked wrong
(5.9). After masking BOTH, numbers are coherent (2.63).

---

## FORMALISM & IMPLEMENTATION (exact, as coded)

### Flow-matching objective (NitroGen.forward)
Notation: action chunk `a ∈ R^{H×A}` (H=18, A=25), Gaussian noise `ε ~ N(0,I)`,
flow time `t ∈ [0,1]`.

1. **Time sampling** (`sample_time`): `s ~ Beta(α=1.6, β=0.8)`; `t = (1 − s)·s_noise`
   with `s_noise = noise_s = 0.999`. (Beta(1.6,0.8) is right-skewed → 1−s is
   left-skewed → prioritizes SMALL t, i.e. near-noise. Matches GR00T/π0.)
2. **Noisy trajectory**: `a_t = (1 − t)·ε + t·a`  (linear interpolation, per-sample
   t broadcast over H,A).
3. **Target velocity** (conditional vector field): `v* = a − ε`.
4. **Model prediction**: `v_θ = action_decoder( DiT( action_encoder(a_t, t_disc),
   cross-attn → VL tokens [plan(K)] + [img(256)] ), t_disc )`, where
   `t_disc = round(t · 1000)` (num_timestep_buckets=1000) is the AdaLN timestep.
5. **Loss** (`F.mse_loss(pred, v*, reduction='none')`, masked):
   ```
   raw = (v_θ − v*)^2                        # (B,H,A)
   m   = has_real_action[:,None,None] * actions_mask   # (B,H,A)
   L   = (has_real_action[:,None,None] * raw * m).sum() / (m.sum() + 1e-6)
   ```
   So it's a **masked mean-squared error on the velocity field**, averaged only over
   unmasked (valid) action dims. `actions_mask` = which of the 25 dims are real
   (joystick-only if `joystick_only_loss`, else all 25). `has_real_action`=1 here.
   NO separate plan/contrastive term — plan vs null behavior is induced entirely by
   the *targets* (null→real action; plan→scheduled counterfactual) + which tokens
   condition the DiT.

### Inference (get_action): Euler integration of the flow ODE
`a_0 = N(0,I)`; for i in 0..k-1 (k=num_inference_timesteps=16):
`t = i/k`, `a_{t+1/k} = a_t + (1/k)·v_θ(a_t, t)`. Returns `a_1`.
Joysticks unpacked from dims 21–24 → (j+? )... `unpack_actions`: j_left=a[...,-4:-2],
j_right=a[...,-2:], rescaled `j*2−1`, clipped [−1,1]; buttons=a[...,:-4] thresholded >0.5.

### Action layout (verified)  A=25
`[ buttons(21) @ dims 0..20 (binary, order = shared.py BUTTON_ACTION_TOKENS),
   j_left_x @21, j_left_y @22, j_right_x @23, j_right_y @24 ]`, joysticks packed to
[0,1] via `(j+1)/2` (center 0.5). Dataset has 17 of the 21 buttons; 4 model slots
(right_bottom/left/right/up) are always 0.

### Plan conditioning (our additions)
- Frozen Qwen3.5-0.8B encodes plan text → last hidden states `H_plan ∈ R^{B×L×1024}`
  (cached per unique text).
- `PlanResampler`: K=8 learned queries, 2 cross-attn layers over H_plan → R^{B×K×1024}.
- `PlanAdapter`: LN→Linear(1024→2048)→GELU→Linear→LN → R^{B×K×1024}.
- Injected at `_PLAN_TOKEN`(=7) positions, PREPENDED to the 256 image tokens in the
  VL stream (`max_sequence_length` = 256+K). Refined by `vl_self_attention` (4L),
  then the DiT's cross-attention (8L, alternating self/cross) attends to it.
- **Null modes:** `learned` = substitute learned `null_plan ∈ R^{K×1024}` for dropped
  rows; `masked` = keep tokens but mask their K positions out of BOTH vl_self_attn
  (additive key mask) AND DiT cross-attn (additive mask via DiT.forward). Masked-null
  == base exactly iff DiT weights unchanged.

### Counterfactual targets (schedule-based, plans.py)
A plan = list of `Segment(start,end,kind,payload)` over normalized chunk [0,1]:
kind ∈ {dir, button, free(=streamer real action), idle}. Direction segments drive
stick and/or d-pad per detected `movement_modality`. Groups: HOLD (whole-chunk dir),
TAP (brief dir + free), SEQ2 (halves), SEQ3 (thirds), IDLE, BUTTON. Sampling is
balanced across groups, then uniform within group.

### Stage-1 optimizer (train_planner.py)
AdamW(β=(0.9,0.95), wd=1e-3); param groups: plan_head lr=2e-4, {DiT, vl_mixing,
action enc/dec} lr=1e-5; vision tower + planner frozen. WSD schedule (linear warmup,
constant, linear decay over last 10%). grad clip 1.0. EMA decay 0.9999 with warmup
`d=min(0.9999,(1+n)/(10+n))`. bf16 autocast. plan_ratio = fraction of examples that
get a synthetic plan (rest are null→real-action). plan_dropout config currently inert
(known bug: collator always sets plan_dropped).

### Eval metrics (eval_deterministic.py)
Fixed ε and fixed t_disc ∈ {100,300,500,700,900}; read v_θ directly (no sampling).
- STEERING(plan,region) = mean_t [ (v_plan − v_null) · expected_dir ] over region
  (stick: projection on (Δ21,Δ22)·(dx,dy); dpad: Δ on the dpad dim). >0 ⇒ correct.
- NULL-INVARIANCE = mean ‖v_null − v_ref‖ over frames×t; ref=base (absolute) or
  ref=untrained-same-arch (isolates training weight-drift).

---

## EXP-005  Direction proportion stats (data audit)  — 2026-06-16

Over 25,674 chunks / 28.1M frames (SHARD_0000), streamer REAL actions:
- **Modality (per chunk):** stick 89.4%, dpad 4.3%, both 1.3%, none 5.0%.
  → data is overwhelmingly stick-driven; d-pad (which steered BEST in evals) is ~5.6%.
- **Stick direction** (|defl|>0.3, 44.8% of frames active; axis-wise share of
  directional): right 31.2%, up 29.8%, left 25.7%, **down 13.3%**.
- **D-pad direction** (1.7% of frames): right 32.0%, down 28.9%, left 22.3%, up 16.8%.

**Implications (correcting an earlier claim):**
- `hold_right` weakness is **NOT** a scarcity issue — right is the MOST common stick
  dir (31%). So the persistent right-failure (EXP-002/003/004) is a genuine
  model/training effect, not data imbalance. Higher-value to investigate directly.
- `down` is the genuinely rare stick dir (13%) yet steered OK → steering quality does
  not track direction frequency.
- D-pad steered best despite ~2% of frames → binary targets likely easier to learn
  than continuous-stick steering (consistent with "stick collapsed to generic shift").

---

## EXP-006  WHY "right fails": stick collapses to a constant shift  — 2026-06-16

**Probe:** trained masked model (EXP-004), raw (un-projected) velocity deltas
(v_plan − v_null) on j_left dims, per hold-direction, fixed noise, 5 timesteps×10
frames:
| plan | Δvx(21) | Δvy(22) |
|---|---|---|
| hold_left  | −0.075 | −0.109 |
| hold_right | −0.082 | −0.110 |
| hold_up    | −0.088 | −0.114 |
| hold_down  | −0.077 | −0.110 |

**Finding (corrects EXP-002/003/004 "right is weak"):** the stick velocity delta is
a near-CONSTANT vector ≈(−0.08, −0.11) **independent of the plan direction**. The
apparent "left/up OK, right/down weak" was an ARTIFACT of the projection metric: a
constant negative (Δx,Δy) aligns with left(−x)/up(−y) and anti-aligns with
right/down. So **the continuous STICK is not being steered directionally at all** —
it learned a generic "a plan is active" shift. The **d-pad (binary) IS
direction-specific** (EXP-004: hold_left→dpad_left, hold_up→dpad_up). 

(For reference, the true per-direction stick target delta in velocity space is
~±0.5 on the relevant axis — the model produces ~0.08, i.e. it largely regressed to
a direction-agnostic mean shift. Base null velocity mean vx=+0.97, vy=+0.20.)

**Hypotheses for stick-collapse:** (a) continuous regression that must FLIP SIGN by
direction is much harder than binary d-pad; with 66 chunks/600 steps the model takes
the easy path (small constant shift + use d-pad for direction); (b) under-training of
the resampler/adapter to map left/right text → opposite continuous outputs.

**Implications for next experiments:**
- This is a DATA/COMPUTE-scale + optimization issue, not direction imbalance
  (EXP-005: right is the most common dir).
- Try: (1) freeze DiT, train only plan_head/adapter, more steps — does stick become
  direction-specific? (2) scale data 10–50×; (3) upweight stick dims in the loss or
  increase counterfactual stick magnitude/consistency; (4) sanity: confirm the
  resampled plan tokens for 'left' vs 'right' are linearly separable.

---

## RESEARCH NOTE: ByteDance Lumine (arXiv:2511.08892) — usability for our project

**What it is:** A VLM generalist agent for **3D open-world games**, trained inside
**Genshin Impact** (Mondstadt 5h storyline), generalizing zero-shot to Honkai: Star
Rail and Wuthering Waves. Pixels in at 5 Hz → **keyboard+mouse** actions out at
30 Hz, with adaptive "reasoning" (hybrid think mode). Data: 1731 h gameplay
(pretrain) + 200 h instruction-following + 15 h reasoning.

**Availability (as of 2026-06):** Paper + project page only. **No official code,
weights, or dataset released.** Only an unofficial WIP reimpl (`zlc1004/Lumine`),
not from the authors. So we **cannot** currently obtain Lumine demonstrations/actions.

**Compatibility assessment (even if data were released):**
- **Action space MISMATCH (major).** Lumine = **keyboard + mouse** (key events,
  relative mouse deltas, GUI clicks). NitroGen = **gamepad** (21 buttons + 2
  analog sticks, 25-d). There is no clean bijection: mouse-look ≠ right stick
  (relative vs absolute-ish, different sensitivity), WASD ≠ left stick (binary vs
  continuous), GUI mouse has no gamepad analog. A mapping would be lossy and
  game-specific.
- **Game distribution MISMATCH.** Lumine = gacha action-RPGs (Genshin/HSR/WuWa);
  NitroGen's eval/data = speedrun/action titles via gamepad overlays. Little overlap.
- **Frequency:** Lumine 30 Hz vs NitroGen ~30 Hz chunked (18 actions, frame_stride
  2 from 60fps). Comparable order; not a blocker by itself.

**Verdict / how it COULD help (future, conditional on release):**
1. **Not as drop-in actions** for the gamepad DiT (action space + games differ).
2. **Most relevant as a PLAN source for Stage 2:** Lumine's *instruction-following*
   and *reasoning* episodes are exactly the "language plan ↔ gameplay" pairing we
   want for the VLM-planner side. If released, the (instruction, gameplay) pairs
   could supervise our planner's plan-generation — independent of the action-space
   mismatch, since we'd use the *language/reasoning*, not the kbm actions.
3. **Architecturally**, Lumine validates our overall bet (VLM perceives+reasons →
   low-level control), and its "adaptive reasoning invoked only when necessary" is a
   useful prior for our cross-chunk planner-firing cadence (the K×A design).
**Action item:** monitor lumine-ai.org / arXiv for a data/code drop; revisit if the
instruction-following corpus is released. Not actionable now.

---

## EXP-007  ROOT CAUSE of stick-collapse: collinear conditioning  — 2026-06-16

**Probe 1 — plan-token directional separability** (inter-class centroid dist /
intra-class spread; >1 = well separated):
| representation | inter/intra ratio |
|---|---|
| frozen-VLM raw hidden (mean-pooled) | 0.65 |
| untrained plan tokens | 0.59 |
| **trained (masked) plan tokens** | **0.91** (improved, still <1) |

But the PAIRWISE collinearity is the killer:
- cosine(trained left-token, right-token) = **0.999**
- cosine(trained up-token, down-token)  = **0.986**
→ opposite directions get NEARLY IDENTICAL conditioning ⇒ DiT can't emit opposite
stick velocities ⇒ constant shift (explains EXP-006).

**Probe 2 — where the collinearity originates (frozen VLM, opposite phrases):**
cosine in VLM hidden space (mean / last-token / max pooling):
| phrase pair | mean | last | max |
|---|---|---|---|
| "go left"/"go right" | 0.972 | 0.922 | 0.965 |
| "keep going left/right" | 0.994 | 0.959 | 0.990 |
| "move to the left/right" | **0.998** | 0.983 | 0.996 |
| "go up"/"go down" | 0.963 | 0.901 | 0.954 |
| "left"/"right" (bare) | **0.879** | 0.879 | 0.879 |
Per-token "go left" vs "go right": tok0("go")=1.000, tok1(dir word)=0.922.

**Conclusions:**
1. The frozen VLM barely separates opposite directions; the discriminative signal is
   ONE token (the direction word) and is diluted by (a) longer phrasings — "move to
   the left/right" cosine 0.998! — and (b) mean-pooling in the resampler.
2. This is the TRUE bottleneck for continuous-stick steering, NOT data scale or
   direction imbalance. d-pad survives because binary outputs tolerate weak/collinear
   conditioning; opposite continuous targets do not.
3. **Levers (in increasing generality):** (a) shorter, direction-focused plan text
   (0.879 vs 0.998); (b) last-token / direction-token pooling instead of mean; (c) a
   contrastive/separation auxiliary loss on plan tokens to force opposite plans apart
   (most general, works with long Stage-2 transcripts); (d) accept stick-shift (user:
   "stick shifting is also okay") and rely on d-pad for crisp direction.

---

## OPS NOTE 2026-06-16 (overnight): cookies re-blocked
Mid-overnight, YouTube began returning "Sign in to confirm you're not a bot" for ALL
new videos again (cookies aged/rotated from datacenter IP). Could not expand the
frame set beyond the existing 66 chunks / 166 frames. Overnight experiments therefore
run on the EXISTING data, prioritizing the EXP-007 collinearity fix (contrastive
plan-token loss + frozen-DiT), which does not need more data. Re-export cookies to
resume data scaling.

---

## EXP-008  Overnight matrix: frozen-DiT & contrastive (4 runs)  — 2026-06-17

All on existing 66 chunks, masked null, 800 steps. Eval: plan-token cos(L,R)/cos(U,D)
(collinearity, lower better), stick direction-specificity (x_split=Δvx(R)−Δvx(L),
y_split=Δvy(D)−Δvy(U); >0 = direction-specific), null-invariance vs base.

| run | cos(L,R) | x_split | y_split | null-inv vs base |
|---|---|---|---|---|
| untrained | 0.994 | — | — | (masked→0) |
| frozen-DiT (plan head only) | 0.999 | +0.008 | −0.004 | **0.000** |
| contrastive(0.5), DiT-train | 0.997 | −0.007 | +0.000 | 3.41 |
| contrastive+frozen-DiT | 0.997 | −0.004 | +0.004 | **0.000** |
| contrastive(2.0), DiT-train | 0.999 | +0.004 | −0.009 | 3.18 |

**Two clean results:**
1. **FROZEN-DiT + masked null ⇒ EXACT null-invariance (0.000).** Architectural
   guarantee: we can train the plan head with ZERO damage to system-1. (vs DiT-train
   which drifts ~3.2–3.4.) Important for safe Stage-1.
2. **NONE of the runs broke the collinearity** — cos(L,R) stayed 0.997–0.999, stick
   splits ≈0 (no direction-specific continuous steering). The EXP-007 bottleneck is
   ROBUST.

**Why the contrastive loss failed (diagnosed):**
- Raw (non-EMA) cos(L,R): con0.5=0.991, con2.0=0.996 — barely moved from 0.994.
- **Directional plans are too RARE:** over 200 sampled examples the label mix was
  null 107, idle 24, seq 19, seq3 13, and only left 7 / right 10 / up 11 / down 9
  (~4% each). Only **23/100** batches (B=8) even contain ≥2 same-direction examples
  (SupCon positives). So the "pull-together" term almost never fires.
- **SupCon is also untargeted:** it separates ALL labels equally (null/idle/seq
  dominate the gradient), not specifically left↔right. 
- Plus we fight the frozen VLM's collinear text rep (cos 0.97–0.99 for "keep going
  left/right"); the resampler *could* separate them but needs DENSE, TARGETED signal.

**Next (higher-confidence) experiment:** directional-heavy sampling
(group_weights favor hold/tap so left/right/up/down are dense) + contrastive →
should finally co-occur opposite directions in batches and separate them. Also
consider a targeted opposite-pair repulsion loss and shorter direction text.

---

## EXP-009  BREAKTHROUGH: direction-heavy + contrastive fixes stick steering  — 2026-06-17

**Hypothesis (from EXP-008):** the contrastive loss failed only because directional
plans were too rare to co-occur in batches. Fix: `--direction-heavy` (group_weights
hold×4/tap×2) makes left/right/up/down dense (label mix left91/right87/up80/down102
per 100 ex; **70/100 batches now have same-direction positives**, was 23) + contrastive
weight 1.0, masked null, 1000 steps.

**Config:** /tmp/stage1_v2 (66 chunks), masked null, plan_ratio 0.6, direction-heavy,
contrastive-weight 1.0, 1000 steps. Two variants: DiT-trainable (lr_dit 1e-5) and
frozen-DiT (plan head only, lr_plan 3e-4).

**Eval (EMA, fixed noise+timestep, 10 frames × 3 timesteps):**
| run | cos(L,R) | cos(U,D) | x_split(R−L) | y_split(D−U) |
|---|---|---|---|---|
| (all prior runs) | 0.997–0.999 | ~1.0 | ≈0 | ≈0 |
| **DH+con, DiT-train** | **0.003** | −0.581 | **+0.196** | **+0.395** |
| **DH+con, frozen-DiT** | **−0.199** | 0.163 | **+0.184** | **+0.342** |

Per-direction raw stick Δv (DiT-train): left Lx=−0.152, right Rx=+0.045 (x flips
sign!); up Uy=−0.158, down Dy=+0.237 (y flips sign!). 

**RESULT: continuous-stick directional steering now WORKS.** The plan-token
collinearity (cos L,R 0.999→~0) was THE bottleneck (EXP-007), and dense directional
co-occurrence + contrastive broke it. Both DiT-train and frozen-DiT succeed; frozen
additionally guarantees exact null-invariance (EXP-008) — so **frozen-DiT +
direction-heavy + contrastive is the current best Stage-1 recipe**: steers the stick
AND leaves system-1 untouched under null.

**Caveats:** cos values noisy at this scale (cos(U,D) −0.58 vs +0.16 across the two
runs — directionally right, magnitude unstable); 66 chunks only. Next: rerun on the
larger fetched set (225 chunks/18 videos now cached) for stability; sweep
contrastive weight; confirm steering on held-out frames.

**Key recipe flags:** `--null-mode masked --freeze-dit --direction-heavy
--contrastive-weight 1.0 --lr-plan 3e-4`.

---

## EXP-010  Best recipe confirmed on larger data (225 chunks)  — 2026-06-17

Re-ran the EXP-009 best recipe on the larger fetched set (225 chunks / 18 videos;
fetched overnight after cookie refresh). Recipe: **frozen-DiT + masked-null +
direction-heavy + contrastive(1.0)**, 1500 steps, lr_plan 3e-4. Eval on the LAST 12
frames (different subset), fixed noise (seed 1), 3 timesteps.

| metric | EXP-009 (66ch) | **EXP-010 (225ch)** |
|---|---|---|
| cos(L,R) | 0.003 / −0.199 | **0.103** |
| cos(U,D) | −0.581 / 0.163 (unstable) | **0.151** (stabilized) |
| x_split (R−L) | +0.196 | **+0.405** |
| y_split (D−U) | +0.395 | **+0.326** |
| per-dir Lx/Rx | −0.15/+0.05 | **−0.288/+0.116** |
| per-dir Uy/Dy | −0.16/+0.24 | **−0.141/+0.185** |
| null-invariance vs base | 0.000 | **0.0000** |

**Confirmed:** more data stabilized cos(U,D) (no longer flips) and strengthened the
x-axis steering (x_split +0.41). All four directions show correct opposite-sign stick
velocity. Null-invariance is EXACT (frozen DiT + masked null = base policy untouched).

**STAGE-1 RECIPE (locked):**
`--null-mode masked --freeze-dit --direction-heavy --contrastive-weight 1.0
 --plan-ratio 0.6 --lr-plan 3e-4` on direction-heavy schedule plans (hold/tap/seq),
contrastive SupCon on pooled plan tokens grouped by direction. This gives:
(a) direction-specific continuous-stick AND d-pad steering, (b) exact null-invariance
(no system-1 damage), (c) the K-token plan channel ready for Stage-2 transcripts.

**Open follow-ups (not blocking):** contrastive-weight sweep; verify SEQ temporal
structure (1st vs 2nd half) under this recipe; scale data further; then Stage-2
(real transcripts) on this frozen-DiT base.

---

## EXP-011  SEQ within-chunk temporal structure: did NOT learn  — 2026-06-17

**Question (from @namak-kun):** did we ever train on / learn complex temporal
actions (e.g. SEQ "left then right" routing within a chunk)? We DO sample SEQ/SEQ3
plans, so they were in training — but were they learned?

**Eval** (`eval_seq_temporal.py`): for SEQ plans, compare (v_plan−v_null) in the
FIRST half vs SECOND half of the 18-step chunk on the relevant stick axis. A working
SEQ shows OPPOSITE signs (e.g. left->right: 1st −x, 2nd +x).

| checkpoint | left->right 1st/2nd | up->down 1st/2nd | verdict |
|---|---|---|---|
| best_large frozen (225ch) | −0.058 / −0.068 | +0.099 / +0.054 | NO (same sign) |
| dh_con DiT-train (66ch) | +0.076 / +0.016 | +0.111 / +0.104 | NO |
| dh_con frozen (66ch) | −0.050 / −0.099 | +0.234 / +0.084 | NO |

**Finding: NO checkpoint learned within-chunk temporal structure.** For SEQ plans
the two halves have the SAME sign (a generic shift, like the old stick-collapse) —
the model averages "left then right" into one direction rather than sequencing.

**Why (analysis):**
- The plan is a STATIC set of K tokens (no temporal index). For SEQ to work, the DiT
  must route: action-token i (early, via its positional embedding) extracts "left"
  from the plan tokens; action-token j (late) extracts "right". That requires the K
  tokens to encode BOTH sub-directions AND the DiT cross-attention to learn
  position-dependent attention.
- In the frozen-DiT runs the cross-attention weights CANNOT change → no temporal
  routing possible; only the static plan_head moved.
- SEQ was also under-sampled (direction-heavy weighted hold×4 vs seq×1).
- HOLD works because it needs no temporal routing (same target all 18 steps).

**Implication:** within-chunk temporal (and later cross-chunk) "complex influence"
likely REQUIRES unfreezing the DiT (or LoRA on its cross-attention) + SEQ-heavy
sampling, so the DiT can learn position-dependent plan extraction. Tests this next
(EXP-012). Trade-off: unfreezing breaks the exact null-invariance that frozen-DiT
gave (EXP-008/010); LoRA is the compromise.

---

## EXP-012  Intra-chunk temporal: unfreeze-DiT + SEQ-heavy  — 2026-06-17

**Goal:** get within-chunk SEQ ("left then right") to sequence, which failed under
frozen-DiT/hold-heavy (EXP-011). Two runs, warm-started from best_large (aligned
HOLD), SEQ-heavy sampling (seq×4/seq3×2), contrastive 1.0, masked null, 1500 steps:
- SEQ-A: **unfrozen DiT** (lr_dit 2e-5, 207M trainable)
- SEQ-B: frozen DiT (plan head only, control)

**Eval** (`eval_seq_temporal.py`, half-chunk Δv split = 2nd_half − 1st_half):
| run | L→R split | U→D split | U→D vs D→U | null-inv |
|---|---|---|---|---|
| best_large (hold-heavy, EXP-011) | −0.01 | −0.05 | ~same | 0.000 |
| **SEQ-B frozen + SEQ-heavy** | −0.13 | −0.16 | nearly identical | **0.0000** |
| **SEQ-A unfrozen + SEQ-heavy** | −0.27 | −0.48 | nearly identical | **4.47** |

**Findings:**
1. **Within-chunk temporal differentiation EMERGED.** Halves now differ by 0.13–0.53
   (vs ~0.01–0.06 in EXP-011). SEQ-heavy data alone helps (frozen −0.13/−0.16);
   unfreezing the DiT helps MORE (−0.27/−0.48). So the DiT CAN learn position-
   dependent plan routing when given capacity + dense temporal targets. Progress on
   P1 (intra-chunk).
2. **But the temporal pattern is NOT plan-specific yet.** up→down ≈ down→up (nearly
   identical Δv) — the model learned "a SEQ plan means +y-early/−y-late" generically,
   not the specific order. **Root cause = a labeling bug:** `plan_label_id` maps ALL
   `seq_*` to ONE label "seq", so the contrastive loss PULLS seq_up_down and
   seq_down_up TOGETHER, actively preventing differentiation. Same failure mode as the
   EXP-007 direction-collapse, one level up. FIX: per-variant SEQ labels.
3. **Trade-off (as predicted):** unfreezing the DiT broke exact null-invariance
   (0→4.47); frozen kept it exact (0.0000) but weaker temporal. ⇒ **LoRA on the DiT
   cross-attention** is the right compromise (temporal capacity without full drift) —
   exactly @namak-kun's suggestion.

**Next:** (a) fix SEQ labels (per-variant) — cheap, likely unlocks plan-specific
sequencing; (b) LoRA on DiT instead of full unfreeze; (c) then re-eval SEQ ordering.

---

## EXP-013  LoRA + per-variant SEQ labels — temporal ordering still hard  — 2026-06-17

Implemented **LoRA on DiT cross-attention** (nitrogen/flow_matching_transformer/lora.py;
rank 8, 16 projections, +0.26M params; loads ng.pt via key remap; B=0 init ⇒ identical
to base at start) and **fixed the contrastive labels** (hybrid: HOLD/TAP grouped by
direction, each SEQ/SEQ3 VARIANT distinct — so seq_up_down ≠ seq_down_up). Run:
frozen base + LoRA-8 + SEQ-heavy + contrastive 1.5, warm-start best_large, 1200 steps,
lr_plan 1e-4, lr_lora 1e-4.

**Eval (seq_lora EMA):**
| SEQ plan | 1st half | 2nd half | split |
|---|---|---|---|
| left->right (x) | +0.088 | −0.156 | −0.245 |
| right->left (x) | +0.013 | −0.168 | −0.182 |
| up->down (y) | +0.390 | +0.253 | −0.136 |
| down->up (y) | +0.408 | +0.174 | −0.234 |
- null-inv vs base = **4.88** (LoRA at lr 1e-4 drifted it; not small).

**Findings (honest):**
1. **Per-variant labels did NOT fix SEQ ordering.** up->down (1st +0.39) ≈ down->up
   (1st +0.41) — still nearly identical; the model applies a generic temporal pattern
   (e.g. y decreasing over the chunk) largely independent of the specific order.
   left/right differ slightly but weakly.
2. **Why (deeper analysis):** the contrastive loss separates the *mean-pooled* plan
   tokens. But temporal ORDERING info must live in the SEQUENCE of K tokens (some
   tokens = first sub-dir, others = second) for the DiT to decode it positionally;
   mean-pooling for the contrastive objective can't enforce that, and the DiT isn't
   translating distinct pooled tokens into distinctly-ORDERED outputs.
3. **LoRA at lr 1e-4 drifts null-invariance (4.88).** LoRA on cross-attn affects the
   image-token path too, so the null moves; use a much lower LoRA LR (or gate LoRA to
   plan-token key positions) for near-exact null-invariance.

**Conclusion on within-chunk temporal (P1):** unfreeze/LoRA + SEQ-heavy makes the two
halves DIFFER (EXP-012/013), but plan-SPECIFIC ordering (up->down ≠ down->up) is NOT
solved by data density + contrastive + LoRA. Likely needs an ARCHITECTURAL change:
**sub-chunk plan blocks (mini K×A inside one chunk)** or **temporal position tags on
the K plan tokens** (MULTICHUNK_DESIGN.md §4 opts 3–4), so ordering is explicit rather
than something the DiT must infer from pooled static tokens. This is the clean next
build. (Stick DIRECTION steering, HOLD, remains solved — EXP-009/010.)

---

## EXP-014  CORRECTION + precise SEQ diagnosis: order lost at the VLM  — 2026-06-17

**Correction to EXP-013 wording:** I wrote "pooled static plan tokens" — WRONG. We
DO have learnable query tokens (Perceiver/Q-former/BLIP-2 style): `PlanResampler` =
K=8 learnable queries cross-attending to VLM hidden states → K DISTINCT plan tokens
(not pooled at production). Pooling only happens (a) inside the contrastive loss
(`.mean(dim=1)`) and (b) in my separability probes. The K tokens DO specialize
(within seq_up_down, mean pairwise token cosine 0.379 — they differ from each other).

**Per-token probe (seq_lora EMA), seq_up_down vs seq_down_up plan tokens:**
- mean-pooled cosine **0.996**, per-token [0.90,1.00,0.92,1.00,0.98,1.00,1.00,1.00],
  flattened **0.974**. → near-identical token SETS for opposite orderings.
- Reference hold_up vs hold_down: mean cosine **0.316** (well separated → HOLD works).

**Where the order info is lost — it's the FROZEN VLM, upstream of the resampler:**
- VLM hidden states "go up then down" vs "go down then up": mean-pool cosine **0.975**,
  last-token (causal, saw full ordered seq) **0.945**. Same for left/right-then:
  0.979 / 0.971.
- So the 0.8B VLM barely encodes WORD ORDER for these short phrases; the order
  difference (which dir comes FIRST) is a tiny cos≈0.95 signal — even smaller than the
  direction-CONTENT difference (cos≈0.97, EXP-007) that we DID amplify.

**Precise conclusion:** SEQ ordering fails NOT because we lack query tokens (we have
them) and NOT because tokens are pooled — but because the **frozen VLM's order signal
is tiny (cos≈0.95) and the contrastive loss (mean-pooled, sparse seq examples) did not
force the resampler to amplify it**, unlike directions where dense sampling +
contrastive DID amplify the (slightly larger) content signal (EXP-009). Same mechanism
as EXP-007, one notch harder (order ≪ content separability).

**Concrete fixes (ordered by confidence):**
1. **Order-sensitive contrastive:** use the FLATTENED/sequence K-token representation
   (not mean) in SupCon + dense seq-variant co-occurrence + higher weight → forces the
   resampler to amplify the VLM's tiny order signal (mirrors the EXP-009 direction win).
2. **Explicit sub-chunk query blocks (intra-chunk K×A):** partition the K queries so
   queries 0..K/2 encode the FIRST sub-segment, K/2..K the SECOND, with the DiT action
   positions attending to the matching block. Builds in order rather than hoping the
   resampler amplifies a 0.95-cosine difference. MOST robust.
3. **Stronger order signal into the VLM:** longer/explicit plan text, last-token
   (causal) conditioning, or feed frames (it's a VLM) to ground the ordering.

---

## EXP-015  Cross-model probe: plan separability vs scale & family  — 2026-06-17

**Question (@namak-kun):** is the "two semantically-similar plans look nearly
identical" problem specific to the 0.8B backbone, or universal? Does scale/family
fix it? → Probed 8 frozen LMs/VLMs across 4 families and 0.6B–9B.

**Method** (`/tmp/probe_vlm_separation.py`): mirror `PlanEncoder.encode_text` (raw
text, last hidden layer). For a 20-plan bank, compute **mean-centered** cosine
(subtract the bank mean to remove transformer anisotropy — raw cosine is inflated;
ALL sentences look ~0.7-0.99 similar). Report per category, mean-pool over tokens:
- opp_content: "go up" vs "go down"  (content flip)
- opp_order:   "go up then down" vs "go down then up"  (ORDER flip; the hard case)
- unrelated:   "go up" vs "open the inventory"  (baseline ≈ 0 after centering)

**Centered cosine (mean-pool); higher = MORE entangled = harder to steer apart.
unrelated baseline ≈ 0 by construction.**

| model | family | size | opp_content | **opp_order** |
|---|---|---|---|---|
| Qwen3-0.6B        | Qwen3   | 0.6B | 0.485 | 0.908 |
| Qwen3.5-0.8B      | Qwen3.5 | 0.8B | 0.653 | 0.772 |
| Llama-3.2-1B-Inst | Llama3.2| 1B   | 0.616 | 0.696 |
| gemma-4-E2B-it    | Gemma4  | ~2B  | 0.861 | 0.915 |
| Qwen3.5-2B        | Qwen3.5 | 2B   | 0.826 | 0.938 |
| gemma-4-E4B-it    | Gemma4  | ~4B  | 0.772 | 0.779 |
| Qwen3.5-4B        | Qwen3.5 | 4B   | 0.792 | 0.899 |
| Qwen3.5-9B        | Qwen3.5 | 9B   | 0.773 | 0.836 |

**Findings:**
1. **UNIVERSAL.** Every model, every family, every size: opposite-CONTENT plans are
   highly entangled (cen 0.49-0.86) and opposite-ORDER plans are **even more**
   entangled (0.70-0.94) — opp_order > opp_content in all 8/8 (gemma-E4B barely).
   The unrelated baseline sits at ~0, so this is real entanglement, not anisotropy.
2. **Scale does NOT help.** Within Qwen3.5, opp_order cen across 0.8→2→4→9B =
   0.772, 0.938, 0.899, 0.836 — flat/non-monotonic, never drops. Bigger LMs cluster
   movement commands *tighter* (they "know" up/down/left/right are all the same kind
   of action), so the discriminative subspace stays tiny or shrinks.
3. **Family does NOT help.** Gemma-4 and Llama-3.2 show the same pattern as Qwen.
4. **Minor lever:** the causal LAST-token rep separates order somewhat better than
   mean-pool for the larger models (Qwen3.5-9B last opp_order 0.748 vs 0.836 mean;
   gemma-4-E4B last 0.423 — the single best across the sweep). The last token has at
   least *seen* the full ordered sequence. Cheap auxiliary worth trying.

**Conclusion / implication (validates @namak-kun's intuition):** a frozen backbone
gives the planner "no reason to distinguish semantically similar plans," and this is
intrinsic — NOT fixable by a bigger/better frozen VLM. BUT high cosine ≠ unusable:
EXP-009/010 already got robust CONTENT steering despite opp_content cen 0.65, because
the contrastive resampler *amplifies the tiny discriminative subspace*. So the order
fix is the **same recipe aimed at order**, not a backbone swap:
  (a) order-aware contrastive (flatten/pertoken — implemented EXP-016, training now);
  (b) architectural order injection (sub-chunk query blocks); and/or
  (c) last-token / causal conditioning; escalating to (d) LoRA-on-VLM only if needed.

**Addendum — gemma-4-12B-it (largest probed):** mean-pool centered cosine
opp_content **0.882**, opp_order **0.570**; last-token opp_content 0.735, opp_order
**0.489**. This is the ONE model where ORDER is *more* separated than CONTENT
(opp_order < opp_content) — echoing gemma-4-E4B's last-token (0.423, best order
separation in the sweep). So the Gemma-4 family encodes word-order somewhat better
than Qwen3.5, but opp_order gap-vs-unrelated is still large (+0.76..0.94). Net: scale
within a family doesn't help; the Gemma-4 family is marginally better at order. Still
not a substitute for the order-aware contrastive / architectural fix.

---

## EXP-016  ✅ SEQ within-chunk ordering SOLVED via order-aware contrastive  — 2026-06-17

**The fix for the EXP-011/012/013/014 SEQ-ordering failure.** Root cause (EXP-014):
the SupCon plan-token loss pooled over the K tokens (`.mean(dim=1)`), so opposite
orderings (seq_up_down vs seq_down_up) shared a mean and the resampler never learned
to encode order — the K tokens for opposite orderings stayed collinear (cos 0.996).

**Change:** added `contrastive_mode` to the SupCon loss (planner.py + nitrogen.py
`_plan_contrastive_loss`):
  * `mean`     — pool over K (old; order-blind).
  * `flatten`  — concat the K tokens → (n, K·d); order info MUST live in
                 position-specific tokens to satisfy the loss. **(used here)**
  * `pertoken` — SupCon independently at each query position, averaged.
Wired `--contrastive-mode` into train_planner.py. seq_* plans already get distinct
per-ordering labels (dataset `_plan_intent` keeps seq_* names whole), so SupCon has
the right negatives once it stops mean-pooling.

**Run** `runs/seq_flatten`: frozen-DiT + masked-null + **seq-heavy** +
**contrastive-weight 1.0 + contrastive-mode flatten** + plan-ratio 0.6 + lr-plan 3e-4,
1500 steps, batch 64, init from `best_large`. (~9.5 min after fixing the data-loader
stall — see note below.)

**Result 1 — REPRESENTATION (token probe, seq_up_down vs seq_down_up):**
| metric | seq_lora (mean, EXP-014) | **seq_flatten (flatten)** |
|---|---|---|
| mean-pooled token cosine | 0.996 | **0.422** |
| flattened token cosine   | 0.974 | **0.162** |
| per-token cosine | all 0.90–1.00 | **−0.68,1.0,−0.47,0.98,−0.69,0.92,−0.68,1.0** |
The resampler now dedicates ~half the query tokens to ORDER (they flip sign when the
ordering flips) and half to shared content. Within-plan token specialization 0.031.

**Result 2 — BEHAVIOR (`eval_seq_temporal.py`, Δv 1st-half vs 2nd-half on the stick
axis; plan-specific ordering required):**
| plan | seq_flatten 1st/2nd | split | mean-ref split | best_large split |
|---|---|---|---|---|
| left→right  | −0.342 / +0.279 | **+0.62 OK** | −0.13 ✗ | −0.01 ✗ |
| right→left  | +0.489 / −0.380 | **−0.87 OK** | −0.12 | −0.02 ✗ |
| up→down     | −0.182 / +0.364 | **+0.55 OK** | −0.16 ✗ | −0.05 ✗ |
| down→up     | +0.295 / −0.622 | **−0.92 OK** | −0.16 | −0.06 ✗ |
All four pass with LARGE, plan-specific splits (left→right is the mirror of
right→left). The mean-contrastive ref and best_large show no real ordering.

**Result 3 — NO REGRESSION on the locked wins (`eval_matrix.py`):**
| metric | seq_flatten | best_large |
|---|---|---|
| cos(L,R) / cos(U,D) | 0.294 / 0.207 | 0.103 / 0.151 |
| x_split / y_split (HOLD steering) | +0.208 / +0.413 | +0.217 / +0.468 |
| **null-invariance vs base** | **0.000** | 0.000 |
HOLD direction steering retained; null-invariance still EXACT (frozen DiT + masked
null). Minor: cos(L,R) rose 0.10→0.29 (seq-heavy spent some capacity on order) but
direction is still clearly separated.

**Conclusion:** within-chunk temporal ordering is achievable on a FROZEN DiT — no
unfreeze/LoRA needed. The representation fix (order-aware contrastive separating the
plan tokens) was sufficient; the DiT's existing action-position embeddings route the
order-carrying tokens to the right half of the chunk. This confirms the EXP-015
reframe: the frozen-VLM's tiny order signal IS amplifiable, exactly like content
(EXP-009) — the only thing missing was an order-aware contrastive objective.

**Data-loader note:** the first launch looked stalled (GPU 0% for 11 min). py-spy
showed the main thread in PIL-decode + SigLIP-preprocess inside `__getitem__`
(num_workers=0 is required because collate calls the CUDA plan encoder). Fix: cache
SigLIP `pixel_values` per-uuid in the dataset (225 unique frames) → 0→2.6 it/s. Also
launch with `python -u` (logs were block-buffered, hiding progress).

**STAGE-1 RECIPE (updated):** locked direction recipe (EXP-010) + for temporal
ordering add `--seq-heavy --contrastive-mode flatten`. Best ckpt:
`runs/seq_flatten/plan_stage1_1500.pt`.

### EXP-016b  flatten vs pertoken contrastive — pertoken wins overall

Ran `runs/seq_pertoken` (same recipe, `--contrastive-mode pertoken`: SupCon applied
independently at each of the K query positions, averaged).

**Token separation (seq_up_down vs seq_down_up):**
| | flatten | pertoken |
|---|---|---|
| mean-pooled token cosine | 0.422 | **0.076** |
| flattened token cosine | 0.162 | **0.076** |
| within-plan token specialization | 0.031 (diverse) | 1.000 (collapsed) |
pertoken separates orderings harder but collapses the K tokens to ~identical (every
position is forced to carry the same label signal); flatten keeps the K tokens diverse
(order in ~half, content in the other half).

**SEQ ordering behavior (split = 2nd−1st half Δv; |split| big + correct sign = good):**
| plan | flatten | pertoken |
|---|---|---|
| left→right  | +0.62 | **+0.93** |
| right→left  | −0.87 | −0.90 |
| up→down     | **+0.55** | +0.43 |
| down→up     | −0.92 | **−0.97** |
Both SOLVE all 4 (plan-specific). pertoken stronger on x, flatten slightly on up→down.

**Direction steering + null-invariance regression:**
| metric | flatten | **pertoken** | best_large |
|---|---|---|---|
| cos(L,R) / cos(U,D) | 0.294 / 0.207 | **0.116 / 0.004** | 0.103 / 0.151 |
| x_split / y_split | 0.208 / 0.413 | **0.494 / 0.411** | 0.217 / 0.468 |
| null-invariance | 0.000 | **0.000** | 0.000 |

**Verdict:** **pertoken is the new best Stage-1 recipe** — it matches best_large's
direction steering (and actually beats it: x_split 0.49 vs 0.22, cos(U,D) 0.004) AND
adds within-chunk temporal ordering, all on a frozen DiT with EXACT null-invariance.
The token-collapse it induces is harmless here (all tokens carry the order+direction
signal; the DiT positional embeddings do the routing). flatten is the pick if future
plans need the K tokens to carry MULTIPLE independent attributes (diversity preserved).

**Best checkpoint: `runs/seq_pertoken/plan_stage1_1500.pt`.**
Recipe: `--null-mode masked --freeze-dit --seq-heavy --contrastive-weight 1.0
--contrastive-mode pertoken --plan-ratio 0.6 --lr-plan 3e-4`, init from best_large.

---

## EXP-017  SEQ3 (3-segment) ordering generalizes — 2026-06-17

Does the order-aware fix scale beyond a single flip? Tested 3-segment plans
("go a, then b, then c") by splitting the 18-action chunk into THIRDS and checking
each third steers toward a/b/c on the relevant stick axis
(`eval_seq3_temporal.py`). seq3 plans were in training (seq-heavy weight 2.0).

| plan (thirds) | **pertoken** | **flatten** | best_large |
|---|---|---|---|
| left→up→right | −0.48 / −0.63 / +0.54  **3/3** | −0.30 / −0.57 / +0.37  **3/3** | 1/3 |
| up→right→down | −0.62 / +0.54 / +0.52  **3/3** | −0.61 / +0.43 / +0.33  **3/3** | 1/3 |
| down→left→up  | +0.29 / −0.48 / −0.70  **3/3** | +0.21 / −0.37 / −0.45  **3/3** | 2/3 |

Both order-aware models route **three** sequential sub-segments correctly within one
chunk (large, correct-signed deltas in each third); best_large shows no ordering.
The frozen DiT's per-action-position embeddings are expressive enough to place 3
distinct directions across the 18 steps, driven purely by the now-order-separated
plan tokens. So within-chunk plan structure spans the full spectrum: HOLD (whole
chunk), SEQ2 (halves), SEQ3 (thirds) — all working, frozen DiT, null-inv exact.

---

## EXP-018  SEQ4 (4-segment) ordering — ZERO-SHOT compositional generalization — 2026-06-17

Question (user): does SEQ3 generalize to SEQ4? Tested 4-segment plans
("go a, then b, then c, then d") by splitting the 18-action chunk into QUARTERS
(boundaries round(0.25/.5/.75 * 18) = 0,4,9,14,18 -> lengths 4,5,5,4) and checking
each quarter steers toward a/b/c/d on the relevant stick axis
(`planner_poc/eval_seq4_temporal.py`).

CRUCIAL: SEQ4 is fully OUT OF DISTRIBUTION. The checkpoints trained on SEQ2 (halves)
+ SEQ3 (thirds) ONLY — never 4 segments, and the plan text "...then d" is novel. This
is a pure compositional-generalization test of the position-routing mechanism.

Plans rotate through all 4 cardinals (consecutive segments alternate stick axis).

| plan (quarters) | **pertoken** | flatten | best_large (no SEQ) |
|---|---|---|---|
| left→up→right→down | −.42/−.33/+.34/+.14  **4/4** | −.42/−.46/+.15/−.03  3/4 | 1/4 |
| up→right→down→left | −.25/+.18/+.12/+.21x **3/4** | −.43/+.41/+.19/+.04x 3/4 | 2/4 |
| down→left→up→right | +.38/−.27/−.42/+.09  **4/4** | +.29/−.23/−.46/+.08  **4/4** | 2/4 |
| **TOTAL** | **11/12** | 10/12 | 5/12 (chance) |

**Verdict: SEQ3 generalizes to SEQ4 zero-shot.** pertoken routes 11/12 quarters
correctly on a 4-segment plan it never trained on; flatten 10/12; the no-SEQ baseline
is at chance (5/12). The model learned "map plan-segment i -> chunk-region i" as a
GENERAL operation (compositional), not a memorized 2/3-segment template. pertoken wins
again, consistent with EXP-016b/017.

**Failure mode = the LAST quarter (steps 14-18).** Every miss is the 4th segment, and
steering magnitude decays across the chunk (pertoken Q1..Q4 mean |delta|: .42,.33,.34,
.14 — the tail is ~3x weaker). Causes: (a) segments are only 4-5 steps so the signal is
diluted, (b) step 14-18 are the action positions furthest from / least covered by the
injected plan tokens, (c) the contrastive labels never supervised a 4th segment. This is
the expected zero-shot ceiling; adding SEQ4 to seq-heavy training should close it. NOT
done yet — left as the obvious next training run if airtight 4/4 is needed.

Implication for cross-chunk: the routing is compositional within a chunk, which is
encouraging for the K*A + cursor design (each chunk reuses the same "segment i ->
region i" operation one level up). But the tail-decay warns that the FAR end of any
single conditioning window is the weak spot — relevant to where we place the cursor.

---

## EXP-019  Heterogeneous (dir+button) sequencing — zero-shot FAILS; root cause found — 2026-06-17

User question: try HARDER intra-chunk — mix MODALITIES in sequence ("go left then
jump"). This is the hardest routing test: the two segments live in DIFFERENT parts of
the 25-dim action vector (stick dims 21-24 vs button dims 0-20), so the model can't
reuse one "stick axis" — it must route segment-i to the right region AND modality.
Eval: `planner_poc/eval_seqhet_temporal.py` (halves; dir-half scored by stick-axis sign,
button-half by mean delta > 0).

Zero-shot result (segments correct / 12):
| plan | pertoken | flatten | best_large |
|---|---|---|---|
| left,jump | 2/2 | 1/2 | 2/2 |
| jump,left | 2/2 | 1/2 | 1/2 |
| right,attack | 1/2 | 2/2 | 2/2 |
| attack,right | 0/2 | 2/2 | 0/2 |
| up,jump | 0/2 | 2/2 | 2/2 |
| jump,down | 1/2 | 0/2 | 1/2 |
| **TOTAL** | 6/12 | 8/12 | 8/12 |

**Verdict: heterogeneous sequencing does NOT generalize zero-shot** (unlike SEQ4, which
hit 11/12). The SEQ-trained models are NO BETTER than the no-SEQ baseline here, and the
DIRECTION halves carry the score — every BUTTON half is near-zero (jump deltas +0.004 to
+0.08, attack -0.01 to +0.19, signs flip). 

**ROOT CAUSE (decisive): button plans were never trained.** `grounded_only=True` is the
PlanDatasetConfig default and train_planner.py never overrides it, so the only button
plans in the library (jump/attack, both grounded=False) were EXCLUDED from every Stage-1
run. The model has literally never been supervised to press a button under a plan — so
the button half is pure base-model noise, and dir+button SEQUENCING was never seen at
all. SEQ4 generalized because it was "more of the same operation" (route a direction to
a region); dir->button is a NEW operation pairing + a new (late-chunk) button position,
so it's genuinely OOD and needs explicit training.

**Next:** add heterogeneous plans (seqhet_*: dir+button and button+dir, both orders),
enable buttons in training (grounded_only=False via --het-heavy), give them contrastive
labels, oversample, and retrain. Re-eval with eval_seqhet_temporal.py.

---

## EXP-020  Heterogeneous training installs button steering (partial) + sampler-eval methodology — 2026-06-17

Acted on EXP-019: added seqhet_* plans (dir+button & button+dir, both orders, 12 plans),
enabled buttons in training via `--het-heavy` (grounded_only=False; group_weights
{hold1.5,tap0.5,seq1.5,seq30.5,idle0.3,button2.0,seqhet4.0}), contrastive labels per
variant. Trained `runs/het_flatten` (1500 steps, init from seq_flatten, --null-mode
masked --freeze-dit --contrastive-mode flatten --contrastive-weight 1.0 --plan-ratio 0.6
--lr-plan 3e-4). flatten chosen over pertoken because heterogeneous plans need the K
tokens to carry TWO independent attributes (dir + button) and EXP-016b showed pertoken
collapses the tokens (flatten preserves diversity).

**METHODOLOGY NOTE (important): the velocity-delta proxy is UNRELIABLE for buttons.**
`eval_seqhet_temporal.py` (single-timestep velocity delta at fixed t, sign test) gave
HET-flatten 8/12 and looked no better than baseline — MISLEADING. Buttons are binary
0/1 targets; a single-step velocity sign doesn't reflect the final press. Built
`eval_seqhet_sampler.py` which runs the FULL 16-step flow sampler (get_action) and
scores the ACTUAL sampled action chunk, with temporal localization for buttons (press
must be positive AND larger in its target half than the other half). Use the sampler
eval for any discrete/button behavior; the velocity proxy is fine only for continuous
stick directions.

**Sampler-based result (segments correct / 12; button=localized press):**
| plan | HET-flatten (trained) | SEQ-flatten | best_large |
|---|---|---|---|
| left,jump | 1/2 | 1/2 | 1/2 |
| jump,left | 2/2 | 0/2 | 1/2 |
| right,attack | 2/2 | 1/2 | 2/2 |
| attack,right | 2/2 | 1/2 | 1/2 |
| up,jump | 1/2 | 1/2 | 1/2 |
| jump,down | 1/2 | 0/2 | 1/2 |
| **TOTAL** | **9/12** | 4/12 | 7/12 |

**Verdict: training installs button steering, but heterogeneous routing is NOT yet
airtight on a frozen DiT.** Decisive evidence it worked: for untrained models the jump
delta is CONSTANT across plans (-0.065/-0.066/-0.067 = plan has no effect on the button);
after HET training it VARIES by plan (+0.275, +0.083, -0.066, -0.025) and localizes to
the correct half in 4/6 cases vs 0/6 untrained. Directions preserved (5/6; one
regression: up in up,jump). Remaining failures: left,jump and jump,down don't press jump;
attack (west) magnitudes are tiny vs jump (south). 

**Hypothesis for the ceiling:** directions steer well on a FROZEN DiT because the DiT has
strong built-in stick priors; pressing a SPECIFIC button in a SPECIFIC temporal half is a
finer, cross-modal routing the frozen DiT cross-attention may lack capacity for. Next:
`--het-heavy` + LoRA on the DiT cross-attention (user flagged this: "more complex
influence... might require unfreezing the dit / lora"). Trade-off: LoRA perturbs exact
null-invariance (EXP-012/013) — watch that metric.

---

## EXP-021  LoRA gives the DiT capacity to route buttons -> strong, localized presses — 2026-06-17

Hypothesis from EXP-020: the frozen DiT can't route a SPECIFIC button to a SPECIFIC
temporal half (it presses buttons only marginally). Test: add LoRA (rank 16) on the DiT
cross-attention. Trained `runs/het_lora` (1500 steps, init from het_flatten, --het-heavy
--null-mode masked --freeze-dit --lora-dit 16 --contrastive-mode flatten
--contrastive-weight 1.0 --plan-ratio 0.6 --lr-plan 3e-4 --lr-dit 1e-4). Sampler eval
(eval_seqhet_sampler.py, N_SEED=4, 8 frames).

| plan | HET-LoRA | HET-flatten(frozen) | best_large |
|---|---|---|---|
| left,jump | 2/2 (jump +0.92) | 1/2 (jump -0.07) | 1/2 |
| jump,left | 2/2 (jump +0.54) | 2/2 (jump +0.08) | 1/2 |
| right,attack | 2/2 (attack +0.91) | 2/2 (attack +0.001) | 2/2 |
| attack,right | 2/2 (attack +1.00) | 2/2 (attack +0.10) | 1/2 |
| up,jump | 1/2 (jump +0.94) | 1/2 | 1/2 |
| jump,down | 1/2 (jump +0.22) | 1/2 | 1/2 |
| **TOTAL** | **10/12** | 9/12 | 7/12 |
| **null-invariance |null-base|** | **0.0224** | 0.0008 | 0.0008 |

**Verdict: LoRA confirms the capacity hypothesis.** The binary 10-vs-9 undersells it: the
FROZEN model presses buttons marginally (target-half delta +0.001 to +0.27), while LoRA
presses them to NEAR-FULL 0.9-1.0 in the correct half and ~0 in the other (jump +0.92/
+0.54/+0.94, attack +0.91/+1.00). So routing a discrete button to a temporal half is a
DiT-capacity problem that direction-steering (frozen-OK) is not — directions ride the
DiT's strong built-in stick priors; arbitrary button x half does not.

**Cost: null-invariance lost exactness, 0.0008 -> 0.0224.** LoRA modifies the DiT
cross-attention for ALL examples, including the null/unconditional branch (masked null
only removes the K plan tokens, not the LoRA delta). 0.022 mean-abs on [0,1] actions is
small but not 0 — relevant for CFG (the unconditional branch drifts from base). Mitigation
options (future): much lower --lr-dit, gate LoRA to plan-token key positions only, or
accept the small drift.

**Remaining failures are DIRECTIONAL, not button:** up,jump (up +0.206 wrong) and
jump,down (down -0.254 wrong). The Y-AXIS stick regresses specifically when paired with
jump; X-axis (left/right) is fine in all 4 of its combos. A button<->y-axis interaction
(jump=south button may be entangled with vertical-stick statistics in the data, or LoRA
over-allocated to the button at the expense of y-steer in these combos). Candidate fixes:
ensure up/down co-occur with buttons more in sampling, longer training, or pertoken for
the direction component. NOT yet addressed.

**Checkpoints:** runs/het_flatten (frozen, null-inv exact, weak buttons),
runs/het_lora (LoRA, strong buttons, null-inv 0.022). Pick per whether exact null-inv or
strong button-pressing matters more for the downstream task.

---

## EXP-022  Het y-axis fix (balance) + lr-dit trade-off frontier — 2026-06-17

Acted on EXP-021's two open issues: (1) up,jump & jump,down regressed the y-axis;
(2) LoRA cost null-invariance (0.0224).

**(1) Root cause of y-axis regression = direction imbalance.** The old seqhet set had
left/right in 4 plans each but up/down in only 2 each (2:1 under-representation). Fixed:
seqhet now = all 4 cardinals x 2 buttons x 2 orders = 16 plans, every direction in 4,
every button in 8 (plans.py het_pairs). Retrained `runs/het_lora_bal` (balanced data,
lr-dit 1e-4, else identical).

**(2) lr-dit sweep** for null-inv recovery: `runs/het_lora_bal_lowlr` (lr-dit 3e-5).

Sampler eval (eval_seqhet_sampler.py):
| metric | het_lora_bal (lr1e-4) | het_lora_bal_lowlr (lr3e-5) | het_lora (unbal,1e-4) | frozen het_flatten |
|---|---|---|---|---|
| total het routing | **11/12** | 9/12 | 10/12 | 9/12 |
| jump,down (was the regress) | 2/2 (down +0.19) | 1/2 | 1/2 (down -0.25) | 1/2 |
| up,jump | 1/2 (up +0.09 marginal) | 1/2 | 1/2 (up +0.21) | 1/2 |
| button press strength | 0.43-0.92 (sharp) | 0.13-0.67 (bleeds) | 0.22-1.00 | ~0.1 (marginal) |
| **null-invariance |null-base|** | 0.0207 | **0.0034** | 0.0224 | 0.0008 |

**Verdict 1: balancing fixed the y-axis.** jump,down fully recovered (down -0.254 ->
+0.194) and up,jump's error shrank 2.4x (+0.206 -> +0.087, now ~neutral not reversed).
balanced lr1e-4 is the best het model: 11/12 with strong, localized button presses. Only
up+jump remains marginal (up steering suppressed when jump is requested same-chunk).

**Verdict 2: lr-dit is a single knob on a Pareto frontier** — it controls BOTH button-
routing sharpness AND null-inv perturbation (both come from the same DiT LoRA delta).
lr3e-5 recovers null-inv to 0.0034 (near the frozen 0.0008) but the LoRA can no longer
route buttons SHARPLY to one half — presses bleed into the other half (attack,right:
attack +0.30 here vs +0.40 other-half = not localized). So:
  - exact-ish null-inv + no real button control -> frozen het_flatten (0.0008)
  - near-exact null-inv + soft buttons          -> het_lora_bal_lowlr (0.0034, 9/12)
  - strong sharp buttons + small null-inv drift  -> het_lora_bal (0.0207, 11/12)  [PICK
    this when buttons matter]

**Open:** up+jump still marginal (the only y-axis holdout). Likely needs up/down+button
co-occurrence even denser, or a touch more capacity (rank 32), or accepting it. Buttons
otherwise solid.

**Best heterogeneous checkpoint: runs/het_lora_bal/plan_stage1_1500.pt** (11/12, sharp
buttons, null-inv 0.0207).

---

## EXP-023  Uneven-duration plans (seqdur) — content vs fixed even-split prior — 2026-06-17

User option 2: does routing key off plan CONTENT (duration words) or a fixed even 50/50
split? Added seqdur_* plans: same two directions, but TEXT implies first-segment length
("briefly go a, then b" / "go a then b" / "go a for a long time, then b") with TARGET
split at 25/50/75% (steps 4/9/14). 4 dir-pairs x 3 splits = 12 plans, each a distinct
contrastive label. Eval (eval_seqdur_temporal.py) samples the chunk and finds the
TRANSITION STEP (best boundary separating a-sign-early from b-sign-late).

**Zero-shot (seq_pertoken, trained ONLY on even splits): FIXED EVEN PRIOR.** Transitions
cluster at 9-11 regardless of intended split (right->left 9/9/10; down->up 6/9/9). Weak
noisy sensitivity (a few 75% push later: up->down 17) but nothing reliable. "briefly"/
"long time" carry no learned duration meaning. Expected — only even splits were trained.

**Trained (dur_pertoken, --dur-heavy --freeze-dit --contrastive-mode pertoken, init
seq_pertoken): PARTIALLY content-driven.**
| pair | zero-shot 25/50/75 | trained 25/50/75 | expect 4/9/14 |
|---|---|---|---|
| right->left | 9/9/10 (stuck) | 2/10/16 (tracks, monotonic) | 4/9/14 |
| down->up | 6/9/9 (stuck) | 2/12/16 (tracks, monotonic) | 4/9/14 |
| left->right | 11/9/14 | 1/17/11 (extremes ok, mid broke) | 4/9/14 |
| up->down | 10/10/17 | 2/1/10 (extremes ok, mid broke) | 4/9/14 |

**Verdict: duration is PARTIALLY learnable on a frozen DiT.** The model goes from a fixed
even prior to spreading transitions across 1-17 by the duration WORD: "briefly" -> early
(t~=1-2), "for a long time" -> late (t~=16). 2/4 pairs track monotonically; the other 2
get the extremes right but the EVEN 50% case is unstable, and the extremes OVER-shoot
(briefly -> t1 vs expected 4; long -> t16 vs 14). So the model learned the DIRECTION of
the duration effect but not precise placement. Mirrors EXP-020/021: coarse routing
(early/late) works frozen; FINE temporal placement (exact transition step) likely needs
more capacity. Testing LoRA next (runs/dur_lora) to see if it sharpens + stabilizes the
middle.

---

## EXP-024  LoRA sharpens uneven-duration transition timing — 2026-06-17

Tested the "fine temporal placement needs capacity" hypothesis from EXP-023: add LoRA
(rank16, lr-dit 1e-4) to the seqdur recipe. `runs/dur_lora` (--dur-heavy --lora-dit 16
--contrastive-mode pertoken, init from dur_pertoken).

Transition step (eval_seqdur_temporal.py), expected 4/9/14 for 25/50/75%:
| pair | frozen dur_pertoken | LoRA dur_lora | monotonic (LoRA) |
|---|---|---|---|
| up->down | 2/1/10 (mid broke) | 3/10/14 (near-exact) | YES |
| down->up | 2/12/16 | 4/9/17 | YES |
| left->right | 1/17/11 (mid broke) | 1/9/11 (mid fixed) | YES |
| right->left | 2/10/16 (was ok) | 16/9/16 (regressed) | no |

**Verdict: LoRA sharpens + stabilizes duration placement.** 3/4 pairs clean monotonic
(vs 2/4 frozen), up->down nearly exact (3/10/14 vs target 4/9/14), and the frozen middle-
case instability (50% landing at 1 or 17) is fixed in 3/4 pairs. One pair (right->left)
regressed (25% -> 16), residual instability. Same recurring theme as EXP-020/021: COARSE
content routing (briefly=early, long=late) emerges on a FROZEN DiT; PRECISE temporal
placement (exact transition step) needs DiT capacity via LoRA. Expect the same null-inv
cost as het_lora (~0.02); not separately measured.

**Summary of the "harder intra-chunk" arc (EXP-018..024):**
- SEQ4 (more same-type segments): generalizes ZERO-SHOT (frozen, 11/12). Routing is
  compositional for the operation it already knows.
- Heterogeneous (new modality - buttons): NOT zero-shot (buttons never trained); frozen
  training installs weak presses; LoRA -> strong localized presses (11/12 balanced).
- Uneven duration (new temporal-placement skill): NOT zero-shot (fixed even prior);
  frozen training -> coarse (briefly/long); LoRA -> sharper, 3/4 near-exact.
- RULE OF THUMB: a frozen DiT generalizes to harder versions of skills it ALREADY has
  (more segments) and learns COARSE new routing; genuinely NEW fine control (specific
  button x half, exact transition step) needs LoRA capacity, at a small null-inv cost.

---

## EXP-025  Chasing the up,jump holdout — diagnosis — 2026-06-17

Diagnostic (diag_upjump.py) on het_lora_bal: FIRST-HALF direction steering for every
(cardinal x button) seqhet plan + pure holds. Data check first: in the real chunks,
pressing a button correlates with DOWN stick (south: stick-y mean +0.188, 46% down vs 24%
up; west: +0.227) vs ~neutral (+0.05) when no button — a genuine button->down behavioral
prior in the base data.

First-half direction delta (sign must match the cardinal):
| dir | +jump | +attack | pure hold |
|---|---|---|---|
| left | -0.126 OK | -0.058 OK | -0.122 OK |
| right | +0.325 OK | +0.319 OK | +0.011 OK |
| up | +0.087 x | -0.045 OK | -0.142 OK |
| down | +0.552 OK | -0.083 x | +0.266 OK |

**Diagnosis:** NOT global up-weakness — pure "hold up" steers correctly (-0.142). Only 2
of 8 combos fail: up+jump (+0.087) and down+attack (-0.083), BOTH marginal (near zero),
both on the weaker y-axis. The clean button->down coupling does NOT fully explain it
(down+attack failing is the OPPOSITE of what coupling predicts, since attack couples to
down which should HELP "down"). So the holdouts are best read as under-capacity / residual
noise on the two weakest y-axis (dir,button) combos, not a systematic prior conflict. Fix
under test: rank 32 + 2500 steps (runs/het_lora_r32) to give the LoRA more room for all 8
combos. Note: the y-axis is intrinsically weaker than x (pure-hold up/down |v|~0.14-0.27
but the combos sit near the sign boundary).

---

## EXP-026  Holdout root cause = DATA (up is the rarest direction); rank32 regresses — 2026-06-17

Chased the up,jump / down,attack holdouts. Two findings:

**(1) rank32 + 2500 steps makes it WORSE, not better.** Expanded the het eval to 8 combos
(added down,attack, up,attack):
| model | total | regressions | null-inv |
|---|---|---|---|
| het_lora_bal (r16, 1500) | **14/16** | — | 0.0207 |
| het_lora_r32 (2500) | 11/16 | jump,left 2->0, up,attack 2->1, jump,down 2->1 | 0.0317 |
r32 fixed down,attack but broke 3 others + worse null-inv. The failing combos SHUFFLE
between runs -> they're at the decision boundary, not a fixable capacity deficit.

**(2) ROOT CAUSE: "up" is the rarest action in the data.** Stick usage over all chunks:
- X-axis active (|x|>0.2) 47.1%, Y-axis 28.9% (X used 1.63x more).
- strong-up (<-0.5): 5.1%  | strong-down: 11.6% | strong-left: 17.0% | strong-right: 17.8%.
UP is ~3.5x rarer than left/right. The base DiT has the weakest prior for up, so synthetic
plans cannot robustly install up-steering — the base model barely produces up. This
explains the ENTIRE holdout pattern: x-axis (abundant) combos always work; y-axis (sparse,
up especially) sit at the sign boundary and 1-2 fail per run depending on noise. The
buttons themselves always fire strongly (it's only the first-half y-DIRECTION that's
marginal).

**Verdict: the het holdouts are DATA-LIMITED, not recipe-limited.** Best het model stays
**het_lora_bal (14/16, r16)**; rank/steps don't help. To actually fix up-steering you'd
need data with more vertical-stick usage (different games) or accept up is weak. The
remaining dur holdout (right->left 25% under dur_lora) is DIFFERENT — right/left is x-axis
(abundant), so that's LoRA training noise, and the FROZEN dur_pertoken already handles it
(2/10/16). Net: stop chasing y-axis marginals (data floor); for duration, frozen+LoRA are
complementary (frozen better on right->left, LoRA better on up/down/left->right).

---

## EXP-027  Cross-chunk cursor mechanism (K*A blocks + cursor) WORKS — 2026-06-17

First cross-chunk result (MULTICHUNK_DESIGN.md R0 machinery). Built K*A plan tokens +
per-chunk cursor:
- planner.py: PlannerConfig.num_chunks (A); PlanHead resampler emits K*A queries,
  reshapes to (B,A,K,d), a per-example cursor gathers block a -> (B,K,d). Downstream
  injection UNCHANGED (still K tokens). A=1 is byte-identical to before (backward-compat).
- nitrogen.py compute_plan_tokens: passes data["plan_cursor"] to plan_head.
- plans.py: CrossChunkPlan + CROSS_CHUNK_PLANS (8 plans, A=2: cc_left_right = chunk0
  hold-left, chunk1 hold-right) + build_cc_target + CrossChunkPlanSampler. Each chunk is a
  whole-chunk HOLD; the SEQ happens ACROSS chunks via the cursor (vs SEQ2 which sequences
  WITHIN a chunk via action-position embeddings).
- dataset.py: cross_chunk mode samples (plan, cursor, target, dir); label = dir_<chunk
  direction> (reuses the direction contrastive); adds plan_cursor to example+collate.
- train_planner.py: --num-chunks A, --cross-chunk; warm-start now SKIPS shape-mismatched
  params (so init from a single-chunk K-query ckpt skips the wider K*A resampler.queries).

Trained runs/cc_pertoken (1500 steps, A=2, --null-mode masked --freeze-dit --cross-chunk
--contrastive-mode pertoken --plan-ratio 0.6 --lr-plan 3e-4, init seq_pertoken; contrastive
0.79->0.23). MVP uses the SAME context frame for every cursor (synthetic forcing is
frame-independent), isolating the cursor mechanism.

**Result (eval_crosschunk.py): 16/16 — PERFECT.** Same plan text + same frame, cursor
flips the whole-chunk direction:
| plan | cursor0 | cursor1 |
|---|---|---|
| cc_left_right | left -0.41 | right +0.59 |
| cc_right_left | right +0.59 | left -0.41 |
| cc_up_down | up -0.38 | down +0.61 |
| cc_down_up | down +0.61 | up -0.38 |
| cc_left_up | left -0.41 | up -0.38 |
| cc_up_right | up -0.38 | right +0.59 |
| cc_right_down | right +0.59 | down +0.61 |
| cc_down_left | down +0.61 | left -0.41 |

All 16 (8 plans x 2 cursors) route correctly, strong/correctly-signed (0.38-0.61). The
resampler's 2 blocks specialize and the cursor selects the right one. "up" works here
(whole-chunk holds, not the data-limited button combos of EXP-026). Frozen DiT + masked
null -> null-invariance expected exact (cursor never touches the null path).

**This proves the core long-horizon machinery: one sparse plan -> A blocks -> per-chunk
cursor -> different per-chunk behavior.** Next: (1) R0 with REAL consecutive frames (need
to extract frame at chunk0-start and chunk1-start per window; currently 1 PNG/dir);
(2) scale A>2; (3) combine cross-chunk with within-chunk SEQ (each block can itself
sequence). Caveat: this MVP teaches "follow the plan" with synthetic forcing on a fixed
frame; it does NOT yet teach overriding the streamer across REAL frame transitions (R2).

---

## EXP-028  Cross-chunk R0 on REAL per-chunk frames — cursor robust to frame transitions — 2026-06-17

EXP-027 used ONE shared frame for every cursor (synthetic forcing is frame-independent).
R0 step: use the REAL frame at each chunk's start (cursor a -> frame at start + a*H*stride).
- scripts/extract_cc_frames.py: pre-extracted 1350 masked frames (225 dirs x 3 window
  starts x 2 chunk offsets) from the cached mp4 slices -> /tmp/frames_cc/<uuid>__<idx>.png.
- dataset.py: make_dir_frame_provider_multi + cc_real_frames mode (samples cursor FIRST,
  fetches real frame at chunk-a-start, caches pixel by (uuid, frame_idx)).
- train_planner.py: --cc-real-frames.
Trained runs/cc_realframes (1500 steps, init cc_pertoken, A=2, real frames). Eval uses the
REAL frame per cursor (different frame for chunk0 vs chunk1).

**Result (eval_crosschunk.py, real frames): 16/16 — cursor routing robust to real frame
transitions.** Every cross-chunk plan routes correctly under both cursors with strong
deltas (0.38-0.60), matching the shared-frame MVP (also 16/16). So the per-chunk cursor
selects the right block regardless of which real frame the chunk sees — the routing is a
property of the BLOCK, not the frame. This is the R0 machinery validated on real
multi-chunk frame transitions (still synthetic FORCED targets; R0-post-hoc with real
actions, and R2 counterfactual, are later).

---

## EXP-029  NESTED cross-chunk + within-chunk SEQ — two-level temporal hierarchy — 2026-06-17

Combine the two routing mechanisms: the CURSOR routes across chunks, and WITHIN each
chunk the action-position embeddings sequence a within-chunk SEQ. Generalized
CrossChunkPlan to per-chunk SCHEDULES + labels (plans.py): hold plans (chunk = whole-chunk
hold, label dir_x) AND nested plans ccnest_* (chunk = within-chunk SEQ a->b, label
seq_a_b). Sampler `pool` = hold|nested|both; train flag --cc-pool. Trained runs/cc_nested
(1800 steps, --cc-pool both --cc-real-frames, A=2, init cc_realframes; contrastive
0.66->0.10).

**Result (eval_crosschunk.py, real per-chunk frames):**
- **NESTED: 24/24.** Every ccnest plan routes BOTH levels: e.g. ccnest_left_right__up_down
  -> cursor0 does left(-0.49)->right(+0.39) WITHIN chunk0, cursor1 does up(-0.38)->down
  (+0.57) WITHIN chunk1. All 6 nested plans x 2 cursors x 2 within-chunk halves correct.
- **HOLD retained: 16/16.** The same model still does cross-chunk hold routing perfectly.

**Verdict: two-level temporal hierarchy COMPOSES cleanly.** The cursor (cross-chunk) and
the DiT action-position embeddings (within-chunk) are orthogonal routing mechanisms that
stack: one sparse plan -> A blocks (cursor picks the chunk) -> each block carries a
within-chunk SEQ (positions sequence inside). A single frozen-DiT model does cross-chunk
holds AND nested cross-chunk+within-chunk SEQ on real frames. This is the full long-
horizon machinery: a plan can specify a program of (chunk -> ordered sub-actions) and the
model executes it chunk-by-chunk via the cursor.

**Best cross-chunk ckpt: runs/cc_nested/plan_stage1_1800.pt** (hold 16/16 + nested 24/24,
real frames, frozen DiT). Caveats unchanged: synthetic FORCED targets (proves the routing
machinery); R0-post-hoc (real actions) and R2 counterfactual (real frame rollouts under
the plan) are the remaining steps toward overriding the streamer across chunks.

---

## EXP-030  Cross-chunk scales to A=4 horizon — 2026-06-17

Does the cursor generalize beyond A=2? Added A=4 hold plans (cc4_*, rotate the 4 cardinals
so each of 4 cursors holds a distinct direction) via a generalized _make_cc_hold_A builder
+ sampler pool "hold4". num_queries = K*A = 32. Trained runs/cc_a4 (1500 steps, num_chunks=4
--cc-pool hold4, shared frame, init cc_pertoken; warm-start skips the wider resampler.queries
16->32). Eval eval_crosschunk_a4.py.

**Result: 16/16.** Every cc4 plan routes all 4 cursors to 4 distinct directions correctly
(deltas 0.29-0.71), e.g. cc4_left_up_right_down -> c0 left, c1 up, c2 right, c3 down. The
resampler's 32 blocks (4 x K) specialize and the cursor selects the right one with NO
degradation at the longer horizon. So the K*A + cursor design scales: one sparse plan can
program A=4 consecutive chunks. (Synthetic forced targets, shared frame; same caveats as
EXP-027.) A could go higher (limited by num_queries width + plan-token capacity per chunk).

---

## EXP-031  R0 post-hoc: cursor follows REAL per-chunk plans, overrides the frame — 2026-06-17

The synthetic experiments (EXP-027..030) used FORCED targets (constant holds). R0 post-hoc
trains on REAL action targets: the plan describes what the streamer ACTUALLY did per chunk
(post-hoc dominant direction), target = the REAL action chunk, frame = real frame at
chunk-a. Mixed 50/50 with synthetic forced plans so the plan stays CAUSAL (else the model
could ignore the plan and reproduce the chunk from the frame).
- actions.py: chunk_dominant_dir (stick+dpad -> cardinal, or None if unclear).
- dataset.py: cc_posthoc mode (per-chunk dominant dirs -> plan "go d0, then d1"; target =
  real chunk[cursor]; falls back to synthetic if any chunk lacks a clear dir).
- train_planner.py: --cc-posthoc --cc-posthoc-ratio.
Data: 45% of episodes have both chunks clearly directed, 21% with DISTINCT dirs (the clean
test set). Trained runs/cc_posthoc (2000 steps, init cc_realframes, A=2, real frames,
posthoc-ratio 0.5).

**Eval 1 — post-hoc causality (eval_crosschunk_posthoc.py): 16/16 both cursors.** On 16
real episodes with DISTINCT per-chunk dirs (d0 != d1), given the post-hoc plan
"go d0, then d1" and a FIXED chunk0 frame: cursor0 -> d0 (16/16), cursor1 -> d1 (16/16).
Crucially, in many episodes the FRAME (null/frame-only baseline) suggests a DIFFERENT
direction than d1, yet cursor1 produces d1 anyway, e.g. null:left->c1:down,
null:right->c1:down, null:left->c1:up. So the cursor+plan CAUSALLY OVERRIDES the frame on
REAL data — the model isn't just reading the frame.

**Eval 2 — synthetic routing retained (eval_crosschunk.py): 16/16.** The same model still
routes the synthetic forced cross-chunk plans perfectly, so post-hoc real-target training
did NOT break plan-causality.

**Verdict: R0 installed on REAL data.** A single frozen-DiT model: (a) follows synthetic
counterfactual cross-chunk plans (cursor routing), AND (b) follows post-hoc plans
describing real per-chunk behavior, overriding the frame when the plan disagrees — trained
on real action-chunk targets. This is the R0 milestone: the cursor/phase machinery works
on real multi-chunk data, not just synthetic forcing. Remaining toward R2 ("override the
streamer using the FUTURE the plan implies"): counterfactual frames via a world model /
env (the frame here is still the real one; we override the OUTPUT, not roll forward).

**Best cross-chunk ckpts:** cc_posthoc (real-data R0, synth+posthoc), cc_nested (nested
hierarchy), cc_a4 (A=4 horizon), cc_realframes (R0 synthetic real-frame), cc_pertoken (MVP).

---

## EXP-032  Query self-attention (Q-former) A/B — no benefit on synthetic, marginally hurts — 2026-06-17

User ask: A/B the resampler query self-attention (--resampler-self-attn: true Q-former,
self-attn->cross-attn->FFN per layer) vs the default Perceiver cross-attn-only. User
hypothesis: gains likely on REAL data, slight/none on synthetic. Tested OFF vs ON,
matched recipe/steps, both from base (clean from-scratch A/B), frozen DiT.

| task (from scratch) | OFF (cross-attn only) | ON (q-former self-attn) |
|---|---|---|
| Direction steer (easy) | 4/4, mean|d|=0.171 | 3/4, mean|d|=0.131 |
| Cross-chunk A=4 (32 queries->4 cursor groups, hard coordination) | 13/16, mean|d|=0.179 | 8/16, mean|d|=0.103 |

**Verdict (synthetic): query self-attention gives NO benefit and marginally HURTS** at
matched training budget. Consistent across the easy (direction) and the hardest
coordination task (A=4, where one might most expect a Q-former to help). The ON steering
magnitudes are consistently LOWER (0.131<0.171, 0.103<0.179).

**Why it hurts on these tasks (mechanism):** self-attention among the K (or K*A) queries
pulls them toward each other (coordination = partial homogenization). But our tasks need
the OPPOSITE: cross-chunk needs the A blocks to be DISTINCT (cursor specialization), and
pertoken contrastive already wants per-position discriminability. Homogenizing queries
works against specialization. Plus the fresh self-attn module (+8.4M params) is
undertrained at matched steps, adding optimization burden. So on synthetic, forced,
single-frame-ish tasks the extra capacity is unused-to-harmful. Matches the user's
expectation (slight/none on synthetic) — actually slightly negative.

**CAVEATS (important):**
1. All ON models have a RANDOM-init self-attn that must be learned; at matched 1500-2000
   steps it may be undertrained vs the cross-attn. This is a "does adding self-attn at
   equal budget help" test, not an asymptotic-capacity test.
2. A=4 from scratch is undertrained for BOTH (contrastive ~1.0); weak operating point.
3. THIS IS ALL SYNTHETIC. The user's actual hypothesis — Q-former helps on REAL data —
   is NOT tested here. The real test is Stage-2 (real transcripts as plans, variable
   length, semantic) where the resampler must distill a genuinely variable-length,
   information-rich VLM hidden sequence — exactly where query self-attn should pay off.
   On fixed short synthetic plans the resampler has little to coordinate.

**Conclusion:** keep self-attn OFF (default) for Stage-1 synthetic work — it's the current
best and the Q-former adds nothing here. Revisit ON for Stage-2 real transcripts, where
the input is long/variable and coordination among queries is more likely to matter. The
toggle (--resampler-self-attn) is ready for that A/B.

### Side finding (REGRESSION, needs fixing): order-aware contrastive no longer LEARNS from scratch
While setting up the SEQ A/B, found that a fresh seq_pertoken-recipe run (init best_large,
seq-heavy, pertoken) FAILS to separate opposite orderings: cos(seq_left_right,
seq_right_left)=0.999 (vs the original runs/seq_pertoken ckpt = 0.165, still 4/4 in eval).
DIRECTION (content) separation still learns fine from scratch (fresh cos(L,R)=0.24,
cos(U,D)=-0.50). So the regression is SPECIFIC to ORDER separation (same content, opposite
order) via the pertoken/flatten contrastive. Cross-chunk models still show ordering ONLY
because they INHERITED it from seq_pertoken (init chain). Excluding seqdur from seq-heavy
(it had polluted the grounded pool) did NOT fix it. Root cause not yet found — a code
change since EXP-016 (cross-chunk/het/seqdur refactors) broke order-contrastive LEARNING.
The seq_pertoken/cc_* checkpoints are UNAFFECTED (already trained). TODO: bisect via
rewind-snapshots; suspect the dataset __getitem__ refactor or label/co-occurrence change.

### EXP-032 regression addendum (root-cause scoping)
Both order-aware modes fail to LEARN order from scratch now: pertoken cos(LR,RL)=0.999,
flatten=0.887 (orig seq_pertoken=0.165, seq_flatten=0.16). Verified NOT the cause:
- contrastive fn logic (diff vs working snapshot = intended EXP-016 refactor only; supcon core identical)
- contrastive invocation block (IDENTICAL to working snapshot)
- compute_plan_tokens (only diff = inert cursor pass-through, skipped for num_chunks=1)
- dataset labels (seq_left_right=12 vs seq_right_left=15, consistent), targets (seq_left_right
  target IS left-then-right), contrastive_mode reaches model as pertoken, LR/RL co-occur ~11%/batch.
So the model + data are correct. Remaining hypotheses: (a) training-distribution/optimization
fragility — order-separation (same content, opposite order) is a harder, higher-variance
optimization than direction (content), and 2 reruns failed where the originals succeeded;
(b) a subtler distribution shift from the plan-library growth (more label classes dilute
per-class positives in a batch). Existing seq_pertoken/seq_flatten/cc_* checkpoints are
UNAFFECTED (already trained). To resolve: 2-3 reruns to test variance, or git-bisect the
dataset sampling. Does NOT block the attention conclusion (direction + cc4 A/B use content
labels that learn reliably from scratch).

### EXP-032 regression update — batch size partially recovers order separation
3 from-scratch reruns of the seq_pertoken recipe all COLLAPSE order separation
(cos(LR,RL): b8 rerun#1 0.985, EXP-032 runs ~0.999/0.887) vs original 0.165 -> a REAL
regression, not variance. Batch-size test: b32 -> cos(LR,RL)=0.490, cos(UD,DU)=0.642 —
BETTER than b8 (0.985) but still short of original (0.165). So SupCon positive-scarcity
(8 seq-types in batch-8 -> opposite orderings rarely get same-label positives) is a
CONFIRMED CONTRIBUTOR but not the whole story (b32 still short). Actionable fix direction:
larger batch, fewer seq-types per run, or explicit positive mining. Existing ckpts
unaffected. Probe (probe_order_sep.py) added for fast re-testing. Full root-cause deferred
(not blocking; existing checkpoints carry the capability).

---

## EXP-033  Stage-2 transcript feasibility probe — captions EXIST but are chit-chat-heavy — 2026-06-18

GO/NO-GO probe for Stage 2 (real transcripts as plans) BEFORE building the alignment
pipeline (planner_poc/probe_transcripts.py). For the 18 unique source videos: check en
caption availability + measure motor-plan-word density.

- **Availability: 15/18 videos have English captions** (mix of manual + YouTube auto).
  Good — the raw material exists and is fetchable with the existing cookie/yt-dlp infra.
- **Plan-word density: ~1.5% median (range 0.7-3.0%).** Fraction of transcript words that
  are directional/action terms (left/right/jump/dash/attack/boost/...). LOW — transcripts
  are dominated by commentary/chit-chat, not near-term motor narration. The most
  action-narrated clips (QOBW 3.0%, 9-FusL 2.2%, TrZlF4 2.0%) are the exceptions.
- NOTE: the `wpm` column in the probe is BOGUS (full-video caption words / chunk-slice
  duration -> inflated). Ignore it; the plan-word RATE is per-word and valid. Also auto-VTT
  repeats rolling-display lines, inflating raw word counts (not the rate).

**Verdict (honest): Stage 2 is a DATA-QUALITY problem, not a modeling one.** Transcripts
exist but most words don't describe what the hands are doing. Naively conditioning on raw
transcript text would feed the planner mostly-irrelevant chit-chat. Implications/options
for tomorrow:
1. **Relabel, don't raw-condition:** use an LLM to compress each transcript window into a
   terse intent ("approaching boost, going for aerial") — turns 1.5% signal into a dense
   plan. This is the GR00T/Hi-Robot "language relabeling" pattern.
2. **Curate channels:** some streamers narrate intent densely; filter to those.
3. **Hybrid:** keep synthetic plans as the backbone; use transcripts only where plan-word
   density clears a threshold (a few % of windows).
This also re-frames the Q-former hypothesis (EXP-032): on RAW noisy transcripts the
resampler must do heavy filtering — exactly where query self-attn MIGHT help — but on
LLM-relabeled terse plans the input is short/clean again (closer to synthetic, where
self-attn didn't help). So test the Q-former A/B on RAW transcript conditioning, not
relabeled.

---

## EXP-034  Probe H1 — NO latent world model in NitroGen's DiT (env-free) — 2026-06-18

Tested the hypothesis (MULTICHUNK_DESIGN.md §5) that NitroGen's DiT implicitly models how
the state evolves over an 18-action chunk, so we could decode the NEXT frame from DiT
internals -> a free 1-step world model -> enables R2 offline. Probe
(planner_poc/probe_h1_worldmodel.py): on 675 REAL consecutive-frame pairs (/tmp/frames_cc;
frame_t = chunk-0 start, frame_t1 = chunk-1 start = result of executing chunk 0), predict
mean-pooled SigLIP features of frame_t1 from pooled DiT internals (conditioned on frame_t +
the REAL chunk-0 actions). Ridge + PCA(64), 80/20 split, vs a persistence baseline
(predict frame_t1 from frame_t's own features).

| predictor -> next-frame SigLIP feats | held-out R^2 |
|---|---|
| persistence (frame_t feats) | **+0.468** |
| DiT internals only | +0.289 |
| DiT internals + persistence | +0.423 |
| **DiT adds over persistence** | **-0.045** |

**Verdict: no evidence of a forward-predictive world model.** The DiT internals carry
frame information (R^2 0.29 alone) but it is a SUBSET of what the current frame already
provides — adding them to persistence does NOT improve next-frame prediction (it slightly
hurts). So NitroGen encodes the CURRENT visual state, not a lookahead. Consistent across
n=120 (X-only R^2 negative from overfitting) and n=675+PCA (clean). 

**Caveats (honest):** linear + mean-POOLED probe; a world model's signal could be
nonlinear and/or spatial (WHERE things move), which pooling+linear destroys. Horizon is 36
source frames (~0.6s); persistence R^2=0.47 means consecutive frames are similar, so the
predictable "delta" is small. A stronger test = spatial (un-pooled) features + a conv/MLP
decoder predicting the frame DELTA. But for the cheap-readout question the answer is clear:
**extracting a world model from NitroGen internals is NOT cheap.** Combined with the
cross-game dynamics difficulty (one action space, many games), this argues against the
offline-world-model route; a real environment (games) remains the principled R2 path.

## Strategic synthesis (EXP-033 + EXP-034, for tomorrow)
- Stage 2 transcripts: captions EXIST (15/18) but are chit-chat (plan-word ~1.5%) ->
  need LLM RELABELING into terse intents, not raw conditioning.
- World model: NOT latent in NitroGen (cheap probe) -> offline WM is not free; training one
  across many games with a shared action space is hard (low data/compute).
- => The two tractable frontiers are (a) Stage 2 with relabeled transcripts (env-free,
  on-thesis) and (b) a real game environment for genuine R2 (infra-heavy but principled).
  "Abstractions across the many games" (user's note) is the interesting third axis to think
  about tomorrow — a game-agnostic plan/skill space.
