## [GPT55 — Round 2] 2026-06-27T23:54:58-07:00

# Rebuttal, concessions, and convergence

I accept the referee correction and OPUS48's central reframe: **NitroGen DiT is already System 1**. My Round 1 language leaned too hard on Helix's “separate S1” analogy; in this repo the 500M DiT is the fast actor and already sees the live frame every chunk. The open problem is narrower: how to feed that actor fresh semantic intent without blocking it.

I also concede OPUS48's strongest technical point: **the blocking cost is almost certainly the autoregressive decode, not the single VLM prefill**. The code path matters: current planning does `generate_plan` text decode and then encodes that text; a decode-free full-depth prefill over current frame + cached plan text/register tokens removes the sequential token loop while staying in the same last-layer representation that the bridge already consumes. That is a better first hypothesis than my implicit “even prefill is too expensive” assumption. So yes: **decode must go async/off-critical-path; full-depth decode-free prefill is the right first fast read to benchmark**.

That said, I do not fully withdraw the localizer idea. I narrow it.

---

# 1. What OPUS48 gets right

OPUS48's prefill-vs-decode axis is cleaner than the original shallow-vs-deep early-exit axis:

1. **It attacks the real stall.** A shallow-layer tap only saves some fraction of one prefill; it does not remove the 24-token autoregressive decode. Moving decode async does.
2. **It avoids a representation fork.** The bridge was trained on full-depth Qwen hidden states. A full-depth prefill stays in that space; a shallow tap or separate semantic net needs alignment.
3. **It is near-zero build.** Existing `text_only`/hidden-state machinery can likely be adapted before writing any new model.
4. **It preserves null-invariance more naturally.** If the only DiT-facing artifact remains masked plan tokens through the existing bridge, the null contract is simpler than adding extra residual paths.

So my Round 1 “FastPlanMod as the default architecture” should be demoted. It is not the first build if Experiment-0 shows prefill-every-chunk is cheap and behaviorally adequate.

---

# 2. Where I still disagree: localization may not be free

The surviving part of my proposal is not “build a new S1” and not “replace the VLM semantic channel.” The surviving claim is more modest:

> Even if full-depth VLM prefill every chunk is affordable, the DiT may benefit from a **tiny, zero-init, DiT-side localization/modulation head** that converts `[fresh plan tokens + live DiT frame features + chunk phase/action trace]` into CFG/AdaLN gates or a few extra corrective tokens.

This is a possible add-on, not the semantic source.

Why it may still matter:

1. **Cross-attention is not guaranteed to learn control authority from K plan tokens.** The repo's history already suggests the PlanAdapter/projection bottleneck and token separability matter. A small global modulation path can make “follow the plan more/less now” explicit.
2. **The VLM prefill sees frames, but not the actor's internal state.** It does not naturally know chunk phase, already-committed action suffix, flow uncertainty, previous button trace, or emulator-derived event bits. These are control-local signals.
3. **CFG/AdaLN localization is orthogonal to semantic freshness.** Fresh prefill answers “what does the plan mean in this frame?” A DiT-side gate answers “how strongly should this plan override base reflexes right now?”
4. **Recovery supervision comes from environments, not Qwen.** If the actor is off-trajectory, the useful signal may be reward-ranked action correction rather than another VLM latent.

But I concede OPUS48's redundancy warning: if `[image | fresh-prefill-plan-tokens]` through the existing DiT cross-attention already localizes well, a separate FastPlanMod is unnecessary complexity. The localizer is justified only by measured failures: stale/contradictory plan over-guidance, weak plan authority, chunk-boundary jitter, or recovery deficits that persist after fresh prefill + widened K + existing plan-AdaLN.

---

# 3. Joint design parameterized by Experiment-0

We can sign the same design if it is conditional on the latency/quality measurement.

## Experiment-0 decision thresholds

Measure on one A6000, same checkpoint (`btn_s600`), same frame resolution/batch regime used for closed-loop eval:

- `T_dit`: one NitroGen 18-step action chunk wall-clock.
- `T_prefill`: one decode-free full-depth Qwen prefill producing plan tokens from current frame + cached plan text/register tokens.
- `T_decode`: current autoregressive `generate_plan` path.
- Behavior: direction/button selectivity, plan flip rate, closed-loop progress where available, and exact null-invariance.

Decision rule:

1. **Green path: OPUS48 no-new-module path.** If `T_prefill <= 0.25 * T_dit` or `T_prefill <= 75 ms` (whichever is more permissive for the actual control loop), and decode-free prefill retains at least ~95% of decoded-plan behavior on existing steering/button evals while preserving exact null-invariance, then run prefill every chunk and do not build FastPlanMod.
2. **Yellow path: async prefill worker, still no localizer first.** If `T_prefill` is nontrivial but can be overlapped and the latest prefill is usually available before the next chunk boundary, run it in a dedicated worker with RTC chunk-stitching. Accept occasional one-chunk-old latents and train with staleness offsets.
3. **Red path: cached-latent + tiny localizer.** If `T_prefill > 0.5 * T_dit`, or it causes missed chunk deadlines after overlap, or prefill must be downsampled so much that behavior drops materially, then cache the slow latent and add the tiny DiT-side localizer from my Round 1.
4. **Quality-failure branch.** If decode-free prefill is fast but behaviorally worse than decoded text, keep decode async as the slow semantic writer, self-distill decoded-plan latents into the prefill/register readout, and only add a localizer after that fails.

The thresholds are deliberately practical rather than theoretical. If prefill is <75ms and the chunk budget is hundreds of ms, optimizing a 5-20M localizer first is premature. If prefill is half a chunk or disrupts overlap, the localizer earns its keep.

---

# 4. What each branch builds

## Green branch: prefill-every-chunk, no FastPlanMod

Build order:

1. Move autoregressive `generate_plan` to an async slow worker. It refreshes human-readable plan text every A chunks or on event triggers; it never blocks the actor.
2. Add decode-free full-depth fast prefill every chunk over current/recent frame + cached plan text + learned register/query tokens.
3. Feed register/text hidden states through the existing Perceiver resampler + adapter into the existing DiT plan-token slots.
4. Widen plan bandwidth only as needed: K=8 baseline, then K=16/32 if evals show token bottleneck.
5. Enable only already-null-safe global plan-AdaLN/CFG mechanisms, zero-init and plan-gated.
6. Add RTC-style chunk overlap/stitching and staleness-offset training.

No separate semantic module. VLM remains frozen. VLM-LoRA remains explicitly higher-cost and deferred.

## Yellow branch: overlapped prefill with occasional stale latents

Same as Green, but the actor consumes the latest completed fast-prefill buffer. Training samples prefill-age offsets of 0-2 chunks. Replan triggers increase slow decode frequency, but the actor still never waits.

## Red branch: cached slow latent + tiny localizer

Only here do we build the module I defended in Round 1, and in a narrower form:

```text
DiTLocalizer(
  current DiT frame features,
  latest cached plan tokens/global,
  action trace + chunk phase + plan age + event bits
) -> {optional K_f=4 tokens, CFG gate, zero-init AdaLN deltas, replan_score}
```

Rules:

- It is **not** a second semantic encoder.
- It never consumes raw pixels if DiT features are available.
- All outputs are multiplied by `plan_valid`; null plan exactly equals base NitroGen.
- Start with CFG/AdaLN gates only; add fast tokens only if gates are insufficient.
- Train first with staleness augmentation and decoded/current-prefill teacher deltas, then with emulator save-state perturb/recovery reward.

---

# 5. Remaining experiments that settle the disagreement

## Experiment 0 — latency + parity, first and mandatory

This is the agreed first build. It resolves the main architecture hinge:

- benchmark `T_decode`, `T_prefill`, `T_dit`;
- run decode-free prefill tokens through existing evals;
- assert exact null-invariance;
- check whether prefill can run every chunk without missed deadlines when overlapped with DiT.

No VLM-LoRA, no shallow tap, no FastPlanMod before this.

## Experiment 1 — localization ablation, only if Experiment-0 is green/yellow but behavior still lags

Compare:

A. fresh prefill tokens only;  
B. fresh prefill tokens + widened K/plan-AdaLN;  
C. B + tiny zero-init DiTLocalizer gates;  
D. C + K_f fast tokens, if needed.

Metrics: plan-conditioned counterfactual accuracy, recovery after injected mistakes in emulator save-states, action discontinuity at chunk boundaries, plan over/under-guidance, and null delta. Commit to the localizer only if it gives a meaningful closed-loop gain (e.g. >10-20% recovery/progress improvement) at negligible latency while preserving null-invariance.

---

# 6. Consensus statement

The consensus architecture is:

**Slow path:** frozen Qwen autoregressive decode runs asynchronously and refreshes readable plan text/structured intent. It never blocks action generation.

**Fast semantic read:** first try OPUS48's full-depth decode-free Qwen prefill every chunk, through the existing bridge, because it removes decode latency without creating a new representation.

**Actor:** NitroGen DiT remains System 1 and consumes the live frame every chunk plus the latest plan tokens. Use RTC-style chunk stitching so neither DiT nor VLM stalls directly hit the controller.

**Optional localization:** a tiny zero-init DiT-side gate/token module is a fallback or second-stage add-on, justified only if fresh prefill is too slow or if cross-attention over `[image | fresh plan tokens]` empirically fails to localize/control plan influence.

**Invariant:** null/masked plan must reproduce base NitroGen exactly in every branch.

