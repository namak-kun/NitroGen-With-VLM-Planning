# Agreed Architecture Experiments (consensus)

This is the final, minimal, ordered experiment list from the two-round GPT55/OPUS48 debate. It does **not** re-open the architecture debate. It only records the simple experiments both sides agreed are worth doing first, grounded in the actual repo paths:

- `nitrogen/planner.py`: `PlanEncoder.encode_text`, `PlanEncoder.encode_multimodal(..., text_only=True)`, `PlanEncoder.generate_plan(..., max_new_tokens=24)`.
- `planner_poc/eval_policy.py`: `NitroGenPolicy.regenerate_plan`, `NitroGenPolicy.act`, `_prep`, `_sample_chunk`.
- `nitrogen/flow_matching_transformer/nitrogen.py`: `NitroGen.compute_plan_tokens`, `NitroGen.apply_null_mask`, `PlanHead.adaln_cond`, DiT `plan_cond`.

## Decision thresholds used by the list

Let:

- `T_decode` = wall-clock for current autoregressive VLM plan generation: `PlanEncoder.generate_plan(frames, device, instruction, prev_plan, max_new_tokens=24)`.
- `T_prefill` = wall-clock for one decode-free full-depth VLM prefill: `PlanEncoder.encode_multimodal(frames, cached_plan_text, device, text_only=True)`, followed by `NitroGen.compute_plan_tokens` if timing the full usable latent.
- `T_chunk` = wall-clock for one NitroGen 18-step action chunk: the DiT flow loop in `NitroGenPolicy._sample_chunk`, with plan hidden already cached so the VLM is not counted.
- `r = T_prefill / T_chunk`.

Branch choice:

- **Branch A: prefill every chunk** if `r < 0.5` **and** decode-free prefill behavioral parity holds.
- **Branch A, guarded/overlapped** if `0.5 <= r < 1.0`, parity holds, and an async worker misses the next chunk boundary on `<1%` of chunks; overrun chunks reuse the previous latent.
- **Branch B: cache + localizer** if `r >= 1.0`, parity fails, or prefill-overlap misses are `>=1%` under the target deployment loop.

Parity definition for E0:

- Existing direction/button eval scores with decode-free prefill are at least **95% of decoded-plan scores** on the same frames/seeds/prompts.
- Plan flip rate is no worse than decoded-plan by **+5 percentage points absolute**.
- Mean action-vector delta versus decoded-plan policy is not decision-changing on the existing steering/button probes; report it, but use the 95% task-score rule as the decision criterion.
- Exact null-invariance passes: with `plan_dropped=True`, masked plan-token attention via `apply_null_mask`, and `PlanHead.adaln_cond(..., dropped=True)`, outputs match the base/no-plan path with `max_abs <= 1e-6` for deterministic float outputs, or bit-for-bit where the code path is deterministic enough to assert it.

---

## E0 — Latency + decode-free-prefill parity selector

**Goal:** Pick Branch A vs Branch B by measurement, not opinion.

**What to measure/build:**

1. Use `btn_s600` on **one A6000**, same frame resolution/batch size as closed-loop eval.
2. Time `T_decode` by calling `NitroGenPolicy.regenerate_plan()` or directly `PlanEncoder.generate_plan(frames, device, instruction, prev_plan, max_new_tokens=24)` on a fixed recent-frame buffer. Warm up first; report median, p90, p95 over at least 50 calls.
3. Time `T_prefill` by calling `PlanEncoder.encode_multimodal(frames, cached_plan_text, device, text_only=True)`; then pass the returned `(plan_hidden, plan_key_padding_mask)` through `NitroGen.compute_plan_tokens` to time the usable-token path. Warm up first; report median, p90, p95 over at least 100 calls.
4. Time `T_chunk` by running the DiT flow sampler in `NitroGenPolicy._sample_chunk` for one 18-step chunk with hot/cached plan hidden. Do not include `generate_plan`; ideally do not include VLM prefill. The relevant code is the `num_inference_timesteps` Euler loop around `_sample_chunk` lines 207-243.
5. Behavioral parity: run the same frames/prompts/seeds through:
   - decoded path: `generate_plan(...) -> encode_text(...) / PlanHiddenCache -> compute_plan_tokens -> _sample_chunk`;
   - decode-free prefill path: `encode_multimodal(..., text_only=True) -> compute_plan_tokens -> _sample_chunk`.
   Compare existing `eval_buttons.py`, `eval_balance.py`, and simple plan-flip/probe metrics.
6. Null-invariance inside E0: run the same deterministic frame/noise seed with plan present and with `plan_dropped=True`; verify `apply_null_mask` removes `_PLAN_TOKEN` attention and `adaln_cond` returns zeroed conditioning for dropped rows.

**Success / decision criterion:**

- `r < 0.5` + parity + null-invariance => **Branch A**.
- `0.5 <= r < 1.0` + parity + async miss rate `<1%` => **Guarded Branch A** with fallback to previous latent on overrun.
- `r >= 1.0`, parity failure, null-invariance failure, or async miss rate `>=1%` => **Branch B**.

**Effort:** 0.5-1 day; inference-only; near-zero build, mostly benchmark harness/instrumentation.

**Unblocks:** The only architecture fork: per-chunk full-depth prefill vs slow cached latent + 5-20M localizer.

---

## E-null — Null-invariance regression test, mandatory after every change

**Goal:** Preserve the core contract: a null/masked plan reproduces base NitroGen exactly.

**What to measure/build:**

1. Add a small deterministic test/harness that constructs one encoded frame using `NitroGenPolicy._prep` and fixed random seed/noise.
2. Compute conditional and null plan tokens via `NitroGen.compute_plan_tokens` using `plan_dropped=False` and `plan_dropped=True`.
3. Verify `NitroGen.apply_null_mask(vl_token_ids, vl_attn_mask, dropped=True)` zeros only `_PLAN_TOKEN` positions.
4. Verify `PlanHead.adaln_cond(plan_tokens, dropped=True)` is exactly zero for dropped rows.
5. Run one deterministic DiT velocity call or a full `_sample_chunk(..., null=True, noise_seed=...)` and compare against the no-plan/base path.

**Success / decision criterion:**

- `max_abs <= 1e-6` for deterministic float outputs, or bit-for-bit equality where practical.
- Any new modulator/localizer/fast-token path must multiply all outputs by `plan_valid` or equivalent so this test still passes.

**Effort:** 2-4 hours; inference-only; no training.

**Unblocks:** Safe refactors for async decode, K widening, prefill path, and any future modulator. This is a required gate, not an optional experiment.

---

## E-async — Move VLM decode off the control loop

**Goal:** Remove the current blocking `generate_plan` stall regardless of Branch A or B.

**What to measure/build:**

1. In `NitroGenPolicy.act`, replace the blocking replan call:
   - current: if `replan_every` fires, `self._plan_text = self.regenerate_plan()` before `_sample_chunk`;
   - new: submit `regenerate_plan()` to a single async worker and immediately continue using the latest completed `self._plan_text`.
2. Store a versioned buffer: `{plan_text, plan_id, started_at_step, completed_at_step, source_frames}`.
3. At each `act` call:
   - if a future is done, atomically swap in the new plan;
   - if no future is running and the cadence/event trigger fires, launch one;
   - never wait for the future inside the control loop.
4. Keep behavior unchanged when `replan_every=0`.
5. Log per-chunk `act` wall-clock, `plan_age_chunks`, future start/end, and whether the chunk used a stale plan.

**Success / decision criterion:**

- With `replan_every>0`, p95 `act` time is within **5%** of `replan_every=0` plus the normal DiT chunk cost.
- No visible `T_decode`-sized spikes at replan boundaries.
- Plan text eventually updates and plan versions are monotonic.
- E-null passes unchanged.

**Effort:** 0.5-1 day; inference-only; minimal code change.

**Unblocks:** Both branches. Branch A uses async decode as the slow semantic writer; Branch B uses it to refresh the cached latent/text without stalling.

---

## E-K32 — K=8 -> K=32 bandwidth probe

**Goal:** Test whether the known bridge bandwidth bottleneck improves with more plan tokens before building larger fast modules.

**What to measure/build:**

1. Instantiate the same policy/model with `num_plan_tokens=32` by setting `PlannerConfig.num_plan_tokens=32` and `NitrogenTokenizerConfig(num_plan_tokens=32, max_sequence_length=256+32)`.
2. Initialize from `btn_s600` where possible:
   - copy/tile the first 8 learned resampler queries into 32 query slots or load non-shape-mismatched weights with `strict=False`;
   - keep the base DiT and existing LoRA initialization otherwise unchanged.
3. Run a short continuation/probe training on the existing Stage-1 synthetic-plan alignment mixture, training only the plan head/adapter/resampler and optional existing rank-16 DiT-LoRA. Do not train the VLM.
4. Compare K=8 vs K=32 on existing direction, button, plan-flip, and token-separability probes (`eval_buttons.py`, `eval_balance.py`, `probe_*`).
5. Include E-null in the same run.

**Success / decision criterion:**

- Adopt K=32 if it gives **>=10% relative improvement** on the weakest agreed steering/button metric or materially reduces plan flip rate without regressing any already-passing button/direction metric by more than **5% relative**.
- Reject/defer K=32 if it needs substantially more training or causes null-invariance/parity regressions.

**Effort:** 0.5-1 day for a short probe if data/checkpoints are local; requires training, but small/frozen-VLM training only.

**Unblocks:** Higher-bandwidth Branch A prefill tokens and Branch B cached latents; may reduce need for a semantic localizer.

---

## E1 — Does control-state localization add value beyond fresh prefill?

**Goal:** Decide whether the small control-state localizer earns its keep when Branch A already provides fresh full-depth prefill latents.

**What to measure/build:**

1. Only run after E0 shows Branch A or Guarded Branch A is viable.
2. Use one emulator-backed environment with save/load and visible recovery signal; if recovery labels are not available, use scripted perturbations plus environment reward/progress. This likely needs emulator save-state data, not just offline demos.
3. Create paired rollouts from the same save-states:
   - A: fresh prefill latent every chunk via `encode_multimodal(..., text_only=True) -> compute_plan_tokens -> DiT`, with only the existing null-safe plan-AdaLN/CFG gate;
   - B: same as A plus a tiny zero-init control-state modulator reading action trace, chunk phase, plan age, and event bits; first emit only CFG/AdaLN gates, not extra tokens.
4. Perturbation/recovery test: force a wrong-action prefix for 0.5-1.0 seconds from the save-state, then release control to the policy. Measure recovery over a fixed horizon.
5. Metrics: recovery success rate, progress/reward delta, time-to-recover, chunk-boundary action discontinuity, plan over/under-guidance, added latency, and E-null.

**Success / decision criterion:**

- Keep the localizer in Branch A if B improves recovery/progress by **>15-20% relative** over A, adds **<5 ms** median latency, and passes E-null.
- Otherwise keep Branch A minimal: fresh prefill + K32 if useful + existing null-safe plan-AdaLN/CFG, no larger FastPlanMod.
- If E0 selected Branch B, this experiment becomes the Branch-B localizer validation/training test rather than a Branch-A ablation.

**Effort:** 1-3 days for a minimal perturbation/recovery harness if emulator save-states are ready; requires some training for the modulator. If using GRPO-style reward ranking instead of labels, effort increases and depends on emulator rollout stability.

**Unblocks:** Whether the consensus modulator stays a <1M gate in Branch A or becomes the 5-20M cached-latent FastPlanMod in Branch B.

---

## Explicitly deferred / not first

- Runtime shallow-layer VLM early-exit taps: both sides rejected depth-wise early exit as the wrong lever; the blocker is autoregressive decode.
- VLM-LoRA: allowed but higher-cost; defer until frozen prefill/cached-latent experiments fail for representation reasons.
- Full YouTube/IDM mega-pipeline as an architecture prerequisite: useful later, not needed to choose Branch A vs B.
- Emulator save-state GRPO for broad recovery: likely needed for real recovery labels, but only after E0/E-async/E-null and the smallest E1 perturbation test.
