# ARCH WARROOM — Round 11 seed: GROUND the System-2→System-1 design in the LITERATURE (TTT & friends)

The owner asked for a round with EXPLICIT literature review — specifically Test-Time Training (TTT) "and other
related things". The prior rounds reasoned mostly from first principles + our own experiments; R11 must ground
the architecture in real published work and say which techniques actually transfer. Both agents: READ the three
lit-review files below (gathered by research subagents, citation-dense), then answer Q1-Q5 by MAPPING specific
papers to our concrete problems and PICKING what to adopt. Be concrete + runnable (the orchestrator runs your
pick). Keep ONE-generalist + null-invariance constraints.

## REQUIRED READING (read these FIRST — they are the point of this round)
- files/LIT_R11_ttt.md          — Test-Time Training (Sun 2020; TTT-layers/RNN 2024; ARC-TTT per-task LoRA), test-time adaptation (TENT, CoTTA, SAR), online-TTT collapse/stabilization.
- files/LIT_R11_distill.md      — self-improvement & System-2→System-1 distillation (Expert Iteration/AlphaZero, STaR, ReST/ReST-EM, "Distilling System 2 into System 1" Yu 2024, Quiet-STaR), DAgger, amortized/sleep-time compute.
- files/LIT_R11_fastweights.md  — fast weights (Schmidhuber 1992, Ba 2016, Schlag 2021), modern Hopfield, MAML, dual-process VLA (Helix, GR00T, pi0/pi0.5, RT-H, latent actions), AdaLN-zero/CFG conditioning, "transformers do ICL by GD".

## VALIDATED CONTEXT (build on it; don't re-litigate — see DISCRIMINATOR_RESULTS.md + ARCH_WARROOM.md R3-R10)
- Routed recipe: plan-OOD games -> KL-anchored plan-head demo-fit (R9 winner: SMW Δ_plan +45.1, battery R=0.83,
  null-invariant); actor-OOD -> local RWBC lora. Merge = disjoint params.
- R10 unified the owner's "short/long path" as EVOCATION (bridge summons a maneuver the DiT HAS) vs ADDITION
  (DiT-LoRA expands the manifold for one it lacks, e.g. grab-mesh). Router = 2-stage gate (null-AUC screens
  reflexive; evoc-ratio + EMULATOR counterfactual decides evoke/add). Outcome maneuvers (spin-jump-kills-Rex)
  decided by save-state P(outcome|plan)-P(outcome|null).
- R9 LESSON (critical for any TTT/consolidation proposal): on-policy RWBC on these retro envs is SEED-VARIANCE-
  DOMINATED and COLLAPSES in every variant unless: plan-head-only (not LoRA), KL-anchored to base, RTG+death-
  penalty reward, >=5 seeds+CIs, supervised-flavored (BC-on-winners, not policy-gradient). Any test-time/online
  weight update inherits this collapse risk -> address it explicitly with the lit's stabilizers.
- OWNER CONSTRAINT (new): do NOT use the raw hand-written narration as training plans (too granular, mixes
  S1/S2). Real System-2 plans must be VLM-ABSTRACTED (frozen Qwen -> clean grounded objective), narration used
  only as a SOURCE/locator. VLM-LoRA is allowed.
- Open problems the lit should attack: STALE PLANS (~50-60% replan flip), actor-OOD capability ADDITION, and
  "carry over learnings across attempts" = consolidation.

## Questions for Round 11 (map LIT -> our design; be concrete, cite the papers you rely on)

**Q1 — TTT for stale plans / online episode adaptation.** Does test-time TRAINING (per-episode weight updates)
apply to our stale-plan + drift problem? In our deployment we PAUSE the game (latency is free) and have a
frame-exact emulator. Options from the lit: (a) TTT-layers-style fast/slow (treat the K plan-tokens or a small
adapter as a fast state updated by a self-supervised loss during the episode); (b) TENT-style entropy/uncertainty
minimization on the actor; (c) ARC-style per-EPISODE LoRA fit from the few frames we've seen. What is the
SELF-SUPERVISED signal at test time in OUR setting (no labels — but we DO have the emulator: forward-consistency,
plan-vs-action agreement, CFG-disagreement)? Pick ONE TTT variant + define the runnable prototype + the metric
that shows it reduces replan-flip / drift WITHOUT the R9 collapse. Name the falsifier.

**Q2 — The consolidation loop = which self-improvement algorithm, exactly?** Our loop (save-state -> VLM mints K
plans/learnings -> roll frozen DiT -> keep top-RTG winners -> KL-anchored plan-head BC conditioned on the winning
plan) — is this Expert Iteration, ReST-EM, or STaR? Cite the closest precedent and BORROW its concrete recipe
detail (e.g. ReST-EM grow/improve split + reward threshold + restart-from-base each round to avoid drift; STaR
rationalization; ExIt's search-as-S2-distilled-into-policy-net framing). What do these papers predict about
collapse / distribution-narrowing / reward-hacking, and do our R9 guards (KL-anchor, RTG, plan-head-only) match
their stabilizers? Define tonight's MINIMAL runnable version (Q1 of R10 already mints+scores rollouts for free).

**Q3 — Is the SHORT path a "fast weight" / continuous latent, and does the lit say to lean on it?** Map the K=8
plan-token bridge to fast-weights / Helix continuous-latent / linear-attn-as-fast-weight-programmer / "ICL = GD".
Does the theory ("transformers do in-context learning by gradient descent") imply our continuous conditioning is
ALREADY implicit test-time optimization that explicit TTT would only make heavier? Recommendation: lean harder
on the continuous short path (and how — pooled latent, FastPlanMod AdaLN-zero) vs invest in explicit TTT? Pick.

**Q4 — Capability ADDITION without forgetting (the long path), per the lit.** For grab-mesh (router-confirmed
"add"): is DAgger / on-policy distillation the right mechanism (cite Ross 2011 covariate-shift O(εT²))? How do
the VLA papers (RT-H "language motions", latent-action methods) add a new low-level skill under a high-level
command? Confirm/deny our "DiT-LoRA on add-chunks only, masked-null, disjoint-merge" plan against the lit, and
borrow one concrete safeguard against catastrophic forgetting (LoRA + replay? EWC? null-anchor?).

**Q5 — The ONE coherent "System-2 becomes System-1" architecture, lit-grounded + the experiment tonight.**
Synthesize Q1-Q4 into a single design and name the SINGLE highest-value experiment that is BOTH lit-grounded and
runnable tonight on 4×A6000 (the R10 maneuver-router is already queued and gates this). Specify metric + pass bar
+ falsifier + seed count. Respect: VLM-abstracted plans (not raw narration), KL-anchor, null-invariance, >=5
seeds for any reward claim.

## Format
Each agent: TL;DR + Q1-Q5 (cite specific papers from the LIT files for every recommendation) + a steelman of the
other agent's likely pick + the single experiment tonight. Terse, technical, numeric, runnable. Call out where
the literature CONTRADICTS a prior war-room conclusion or the orchestrator's framing.
