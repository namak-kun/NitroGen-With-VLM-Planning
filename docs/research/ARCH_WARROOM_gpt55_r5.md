# ARCH WARROOM R5 — GPT-5.5: System-1/System-2 Attribution in Demo Narration

## Stance

The S1/S2 split is **partly recoverable, but only operationally, not psychologically**. The owner’s narration cannot reveal what he “really thought”; introspection is famously unreliable and post-hoc (Nisbett & Wilson 1977, https://doi.org/10.1037/0033-295X.84.3.231). The usable split is instead: **does this text add information not already recoverable from the current frame by the frozen actor?** That makes the best definition of type-(A) “frame-underdetermined action-relevant residual.” It matches the project’s constraints: one generalist, frozen-VLM-first, exact null-invariance, and a measured SMW/Sonic dissociation.

## Q1 — Recoverability and ranking of methods

**1. Frame-counterfactual filter (best, primary).** A narration line is type-(B) if the base/null DiT already predicts the demonstrated action from the frame. The repo can measure this two ways: `base_dit_perdim.py` estimates null/base fidelity from frame alone; `eval_policy.py` exposes the exact CFG decomposition `v = v_u + w*(v_c-v_u)`, so `||v_c-v_u||` under a candidate plan is plan authority and `||a_demo-a_null||` is frame-underdetermined residual. This is the rigorous operationalization: type-(A) is whatever remains after subtracting the frozen System-1’s frame prediction. It is not “true cognition,” but it is the only split that matters for training plan tokens.

Failure modes: base DiT errors can misclassify type-(B) as type-(A) when the actor is undertrained; stochastic sampling/noise can inflate residuals; a frame may visually imply “duck bullet bill” but base DiT fails due to actor-OOD; and line text may describe multi-second intent while the chunk window only sees 18 frames. Mitigation: use per-dim calibrated thresholds, multiple seeds, and require residual consistency across adjacent chunks.

**2. Temporal abstraction (second, useful prior).** Aggregate over an A-chunk window and keep the slowest-varying intent: “enter pipe,” “chase not worth it,” “set up flagpole jump.” This aligns with options/temporal abstraction (Sutton, Precup & Singh 1999, https://doi.org/10.1016/S0004-3702(99)00052-1). But slow does not equal S2: “run right” is slow in Sonic and still frame-obvious. Therefore temporal abstraction should propose candidates for the counterfactual filter, not replace it.

**3. VLM re-abstraction (third, useful but unsafe alone).** Frozen Qwen can consume frames + raw narration and emit an 8-word objective using `generate_plan`/`encode_multimodal(text_only=True)`. This is attractive: narration becomes grounding hint, not plan target. But Qwen will parrot common platformer reactions unless forced by prompt and verified by the DiT residual. Use decode-free hidden states only after filtering or use generation with a strict system prompt: “remove immediate reactions already visible in the frame; output only persistent objective/route choice.” Still, popular-game VLM priors contaminate this path.

**4. Don’t split (last as training, good as eval).** Using narration only as eval/judge/checkpoint labels is safe, but leaves the main data bottleneck unsolved. I would keep raw narration for human-auditable labels and not feed it directly to the plan bridge.

## Q2 — Supervision target and loss

The target should not be the raw narration. The target is a **residual objective token**: a short VLM-reabstracted phrase or hidden-state target for only those windows where `(demo action - null/base action)` is large and plan-conditioned action improves toward the demo/reward. Concretely:

1. Parse narration to frame spans (`demo_narration.py`).
2. For each span, sample `a_null = _sample_chunk(frame, "", null=True)` and optionally `a_plan = _sample_chunk(frame, filtered_text, w=1/8)`.
3. Compute residual score `R_A = ||a_demo - a_null||_weighted`, with button AUC/separation thresholds and stick MSE/correlation calibration from `base_dit_perdim.py`.
4. Keep spans above threshold and not marked/noise-classified as “flair/no real reason.”
5. Ask frozen Qwen to re-abstract retained spans into objective-only text, then train bridge/plan-head with masked-null preserved.

Loss: **gated residual behavior cloning / denoising loss** on action chunks plus a contrastive token-separation term. Weight each example by `stopgrad(R_A)` and optionally by return advantage when env reward exists. Negative controls are null and type-(B) lines: plan tokens should produce `v_c≈v_u` there. This differs by game: SMW should yield many high-weight type-(A) examples; Sonic should mostly be gated out, leaving actor adaptation/RL to handle locomotion. One model, game-dependent weights.

## Q3 — Confabulation, flair, VLM priors, guardrails

(C) and (D) are not harmless label noise; they can teach the bridge to inject arbitrary plan authority. Guardrails:

- lexical hard filters for “no real reason,” “maybe,” “flair,” “vain attempt”; these become eval notes, not training targets;
- residual gate: if base DiT already matches, zero the plan loss even if narration sounds intentional;
- null-invariance test: filtered-out text must produce `v_c≈v_u` and no action shift at CFG=1;
- cross-game holdout: evaluate on obscure/homebrew games because Qwen may know SMW/Sonic from pretraining;
- counterfactual environment validation: “enter pipe” is accepted only if alternative route/reward differs, not because Qwen knows Mario lore.

Synthetic-data-in-VLM-training makes chain-of-thought-like explanations suspect; popular-game prior contamination makes Qwen’s “objective” sometimes memorized rather than perceived. Therefore VLM text is a proposal generator, not an oracle. DAgger’s lesson is relevant: imitation targets drift unless corrected by on-policy data/aggregation (Ross, Gordon & Bagnell 2011, https://arxiv.org/abs/1011.0686). Here, residual narration must eventually be validated in env rollouts.

## Q4 — Architecture implication

I would soften the hard S1/S2 framing into a **timescale split with an S1 residual gate**. Human cognition is entangled at one-second narration granularity, so forcing a psychological module boundary is brittle. Keep the current NitroGen DiT as fast frame-fresh actor, but add two plan timescales:

- **fast latent every chunk**: frame-grounded, low-authority, allowed to encode immediate affordance/context but regularized toward null when base DiT suffices;
- **slow objective every A chunks**: persistent route/task token, high authority only when counterfactual residual/reward says the frame underdetermines behavior.

This is closer to Helix’s advertised split between fast visuomotor control and slower semantic goals (Figure AI Helix, https://www.figure.ai/news/helix) and to π0.5’s hierarchical/generalist robot-policy framing (Physical Intelligence, https://www.pi.website/blog/pi05), while preserving this repo’s null-invariant CFG design. I would not abandon System-2; I would define it by **timescale + counterfactual necessity**, not by human introspective category.

## Q5 — Cheap experiment

Run an attribution-validation experiment on existing SMW narration.

Metric: **Type-A residual fraction** per demo/game:
`A_frac = mean_span[ I(||a_demo-a_null||_weighted > tau AND text not C/D) ]`, with `tau` set from the null-vs-null noise floor in `base_dit_perdim.py` / repeated `_sample_chunk(null=True)` seeds. Also compute `plan_grad = ||a_plan-a_null||` and reward benefit `Δ_plan = R_oracle_plan - R_live/null` from the discriminator harness.

Procedure: parse `demo_explanations.md` into frame spans (or mirror into narration.txt then `demo_narration.py parse`), compute residuals for SMW spans, and create a Sonic control narration template of “run right / jump obstacles” over matched frames. Confirmation: SMW `A_frac ≥ 0.30` and Sonic `A_frac ≤ 0.10`, with game-level ordering matching `Δ_plan` (SMW positive, Sonic near-zero/negative). Refutation: `A_frac` similar across SMW/Sonic or no correlation with per-state `Δ_plan`. A stronger validation: manually inspect 20 top-residual and 20 low-residual lines; ≥75% top are pipe/chase/routing/setup, ≥75% low are dodge/stomp/duck/flair.
