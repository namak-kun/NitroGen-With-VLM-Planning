# ARCH WARROOM — Round 9 seed: demo-fit FIXES planner-variance but DESTROYS bridge expressiveness

The owner is back and wants the architecture discussion re-opened, now grounded in HARD new empirical findings
(not latency speculation). The night's routed recipe WORKS and GENERALIZES — but a decisive probe exposed a
**central tension** that the recipe trades away. This round: both agents debate how to keep the plan-following
win WITHOUT destroying the bridge's broader expressiveness. Toward ONE generalist (owner rejects per-game/
per-genre models).

## What's VALIDATED (this session — build on it, do NOT re-litigate)
- **Dissociation:** SMW/MMX/SMB1 = plan-OOD (oracle plan helps; RWBC actor-adapt collapses). Sonic = actor-OOD
  (plain local RWBC robustly +1.44, 3/3 seeds; plan redundant).
- **Routed recipe (ONE model, two MODES):** plan-OOD -> SUPERVISED demo-fit of PLAN-HEAD ONLY (LoRA+base frozen,
  fit to human demo actions on the correct plan, NO env reward). actor-OOD -> plain local RWBC (lora). Merge =
  disjoint params (plan_head + lora) -> keeps both gains, null-invariant 0.0.
- **Generalizes + scales:** MMX per-game demo-fit Δ_plan +117% (3/3). ONE POOLED plan-head fit on SMW+MMX+SMB1
  widens Δ_plan on all (SMW +26, MMX +31, 3/3 seeds, CIs>0). Pooled fit, not averaging/sequential.
- **Planner-variance / staleness: SOLVED as a side effect.** The LIVE VLM plan (generate_plan every 2 chunks,
  real deployment) on the BASE bridge HURTS on SMW (Δ_plan-vs-null = -10.7); after pooled demo-fit it's +56.3
  (≈oracle). Replicated MMX (+38.5->+61.5), SMB1 (+10.2->+53.3). Text staleness mild. So demo-fit makes the
  bridge ROBUST to the VLM's imperfect live plans. (This was a big open worry; consider it addressed.)
- **Trustworthy eval = Δ_plan** = (plan-advance − null-advance) screen_x from FIXED demo save-states (low var).
  >=5 seeds + bootstrap CIs mandatory; single-seed retro deltas burned us repeatedly.
- **DEAD (refuted multi-seed, don't propose):** RWBC for plan-OOD games (all variants); rtg+advnorm;
  residual-mask-as-loss-gate.

## THE NEW DECISIVE FINDING (this is the round's center of gravity)
The owner's GOLD narration mentions DUCKING; BATTERY['smw']['correct'] (the training "correct plan") is terse
and never does. We measured DOWN-press rate vs plan content (4-6 SMW demo save-states, oracle/explicit plans):

DUCK-ON-COMMAND (does the bridge press DOWN when the plan says to?):
| plan text                       | BASE bridge DOWN-dpad | POOLED demo-fit DOWN-dpad |
|---------------------------------|-----------------------|---------------------------|
| terse "move right, jump..."     | 1.2%                  | 0.0%                      |
| "duck to dodge the bullet"      | 3.4%                  | 0.0%                      |
| "press down to duck under it"   | 10.8%  (9x base!)     | 0.0%                      |
Human demos duck 1.7%; BASE ducks 2.9% (≈human); POOLED emits **0% DOWN for EVERY plan**.

INTERPRETATION (agree or challenge): the BASE bridge IS plan-steerable for rare situational actions (explicit
"press down" -> 9x more ducking). The supervised plan-head demo-fit, trained on ONE terse directional plan that
never says "duck", didn't just bias toward right+jump — it **CATASTROPHICALLY OVERWROTE the bridge's
plan-responsiveness**: duck-conditioning is entirely GONE (0% even when explicitly commanded). We bought
plan-following on the TRAINED axis (advance) by destroying expressiveness on every axis the training plan never
named. This is a generalist-killer: a real game needs situational moves (duck, wait, retreat, climb) that no
single terse plan enumerates.

## Hard constraints (carry forward)
- ONE generalist model. No per-game/per-genre models, no routing around interference.
- VLM planner training is now ALLOWED (owner approved a Qwen-LoRA) — but the bridge (resampler/adapter/plan-
  head) + DiT-LoRA remain the cheap levers. Frozen-VLM-first still preferred for variance.
- Exact null-invariance must be preserved (null plan == base DiT). GPU-frugal. Recoverable/additive only.

## Questions for Round 9 (be concrete — the orchestrator RUNS your pick tonight on 4×A6000)

**Q1 — Root cause of the expressiveness collapse.** Is it (a) DATA (one terse plan -> no gradient ties DOWN to
any plan token, so plan_head free to drop it), (b) OBJECTIVE (plain BC mode-collapses rare actions; no anchor
to base), (c) CAPACITY/ARCH (K=8 plan tokens + overwriting plan_head is too low-bandwidth / destructive), or
(d) all three? Rank them. What single measurement would discriminate (a) vs (b) vs (c)?

**Q2 — The fix to TEST TONIGHT.** Pick ONE and define it precisely (so it can be coded):
  (1) **Richer plans:** re-fit on PER-SITUATION plans — for chunks where the human pressed DOWN, condition on a
      duck-mentioning plan (mine from the gold narration / a templated "...duck under the bullet" when DOWN is
      held); else the directional plan. Cheapest. Does Δ_plan stay AND DOWN-on-command return?
  (2) **KL-anchor / behavior-preserving demo-fit:** add a KL (or L2-on-logits) penalty pulling the demo-fit
      policy toward the BASE bridge on a held-out plan distribution (incl. duck plans), weighted so advance-
      following survives but rare-action conditioning is preserved. Keeps the single terse plan, fixes the
      collapse via regularization.
  (3) **ADD not OVERWRITE — zero-init modulator (FastPlanMod):** freeze plan_head, learn a zero-init FiLM/AdaLN
      side-path that ADDS plan-conditioned advance bias, leaving the base bridge's duck-responsiveness intact by
      construction. Preserves null-invariance trivially.
  (4) **Bandwidth K=8->32:** widen plan tokens so situational commands have room; re-fit.
  State the EXACT pass criterion: e.g. "DOWN-on-command for the demo-fit recovers to >=5% under 'press down to
  duck' (vs base 10.8%) WHILE SMW Δ_plan stays within CI of the pooled +26." Name the falsifier.

**Q3 — Is the terse BATTERY plan the real culprit, or is BC fundamentally mode-collapsing?** A cheap diagnostic:
re-run the SAME demo-fit but with the per-chunk plan VARIED (directional when moving, duck when DOWN held,
wait/retreat when idle). If DOWN-on-command returns, it's DATA (Q1a) and richer-plan demo-fit is the answer; if
it STILL collapses, it's the OBJECTIVE/ARCH (Q1b/c) and we need KL-anchor or the modulator. Should we run this
diagnostic FIRST tonight before committing to a fix? (orchestrator leans yes.)

**Q4 — Generalize the metric.** DOWN/duck is one rare action. The real claim is "demo-fit must preserve
plan-conditioned expressiveness for ALL situational actions". Propose a small EXPRESSIVENESS BATTERY (a handful
of plan->action contrasts the BASE bridge already obeys: duck-on-DOWN, jump-on-"jump now", left-on-"go back
left", wait-on-"stop and wait") to score base vs any demo-fit, so we catch collapse for the whole action space,
not just DOWN. Which 4-5 contrasts, and what's the aggregate pass bar?

**Q5 — Does this change the SCALING story?** If the fix is "richer per-situation plans", the pooled generalist
must be re-fit on per-situation-plan-labeled demos for ALL plan-OOD games (the gold narration exists for SMW;
MMX/SMB1 have only directional plans). Does the VLM (frozen, or the now-allowed Qwen-LoRA) generate the
per-situation plan labels for free (it already emits grounded objectives)? Or is light human narration needed?
Keep short — this is the path, not tonight's run.

## What the orchestrator will run tonight (your job: confirm/redirect + sharpen criteria)
Default plan unless you override: (1) Q3 diagnostic first — re-fit SMW demo-fit with VARIED per-chunk plans
(duck-labeled when DOWN held), 3 seeds, measure DOWN-on-command + Δ_plan. (2) Based on the result, run the
winning Q2 fix (richer-plan re-fit if data, KL-anchor or zero-init modulator if objective/arch). Every headline
>=3 seeds + the duck-on-command contrast + Δ_plan CI + null-invariance check. Videos for the owner.
