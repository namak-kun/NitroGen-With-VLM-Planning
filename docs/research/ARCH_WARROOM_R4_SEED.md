# ARCH WARROOM — Round 4 seed: from diagnosis to PRESCRIPTION + a RED-TEAM of the consensus

R3 reached consensus (see ARCH_WARROOM.md "ROUND 3" + ARCH_WARROOM_{gpt55,opus}_r3.md):
- Subtle drift on a TRAINED game = **DiT-actor OOD** (BC covariate shift), not plan-OOD.
- A fresher PLAN is **not load-bearing** for drift (near-zero plan gradient); only on-policy emulator state
  coverage cures it. VLM = objective selection + drift detection/replan triggering, not motor correction.
- Online gate: CFG residual `r_t` + K-sample disagreement `D_t`, conformal p95, null-invariant.

A discriminator experiment is RUNNING right now (`rwbc_actor_adapt.py`, SMW+Sonic, btn_s600): 3 rungs
R_A=live-plan/no-adapt, R_B=oracle-plan/no-adapt, R_C=oracle-plan/`--lora-only`-adapt; Δ_plan=R_B−R_A,
Δ_actor=R_C−R_B. Predicted: Δ_actor ≫ Δ_plan. (Metric = native env screen_x cumulative reward; judge by
RELATIVE magnitude/sign, not absolute 0.10.) Results will be appended before Round 4b.

This round is NOT a re-agreement. Two assigned ROLES, then cross-rebuttal. Latency stays OFF the table.

## THE HOLE BOTH MUST CONFRONT (this is the point of R4)
The "near-zero plan gradient" evidence (base DiT reproduces the action from the frame alone, buttons AUC
0.82–0.96) was measured on **ON-TRAJECTORY / near-expert** states (the dataset is expert demos). But the
drift regime is **OFF-TRAJECTORY** — exactly the states we have **NO measurement** for. So "the plan doesn't
help actor-OOD" is an **extrapolation from on-manifold data into the off-manifold regime.** Off-manifold, the
frame may be genuinely ambiguous/novel and the plan gradient could be **non-zero** — i.e. the plan might
re-acquire authority precisely where the actor is lost. Also: a **stale/wrong plan can CAUSE drift** (actor
pursues the wrong objective → leaves the manifold), so the two OODs may be **entangled**, not independent.

## ROLE A — OPUS-4.8: the PRESCRIPTION (constructive). Write `ARCH_WARROOM_opus_r4.md`.
Design the concrete minimal training loop that cures actor-OOD and yields ONE generalist. Must resolve:
1. **No queryable expert.** Classic DAgger (Ross 2011) needs an oracle to LABEL the actor's visited states;
   we have only (a) fixed human demos and (b) emulator reward+reset — NOT an interactive expert. So spell out
   the REALISTIC on-policy scheme: reward-ranked self-imitation (RWBC, already in repo) vs save-state GRPO vs
   reward-weighted regression vs a learned/queryable expert (e.g. a stronger search policy or the emulator
   itself via lookahead). Which, why, and the exact data/loss. Cite GRPO (DeepSeek arXiv:2402.03300 / 2501),
   AWR (Peng 2019 arXiv:1910.00177), DAgger (1011.0686), π0.5.
3. **Active loop = the gate doubles as a data sampler.** Does the Q3 detector (`r_t`,`D_t`) become an ACTIVE
   sampler: detect drift online → snapshot the save-state → emulator branch-search K recoveries → add the best
   to the training set? This targets exactly the high-uncertainty off-manifold states (the DAgger insight
   without a labeling oracle). Design it: trigger, search budget, what gets added, how it interleaves with BC.
4. **ONE generalist, no collapse.** How to train across games without per-game interference (the multigenre
   "interference" was partly a metric artifact, but state it): curriculum, reward normalization per game,
   balancing, shared-vs-per-game LoRA. Frozen-VLM-first; VLM-LoRA cost-flagged.
5. **Honesty:** if the plan re-acquires authority off-manifold (the HOLE), where does that change the recipe?

## ROLE B — GPT-5.5: the RED-TEAM (adversarial). Write `ARCH_WARROOM_gpt55_r4.md`.
Try to BREAK the R3 consensus. Your job is the strongest steelman that the consensus is wrong or incomplete:
1. **Steelman "the plan IS load-bearing for drift."** Push the HOLE hard: the near-zero-gradient is
   on-manifold-only; off-manifold the plan may dominate. What measurement would expose this? Could the
   discriminator (oracle plan on drifted eval states) be MISSING it (e.g. oracle plan helps only if the actor
   is ALSO adapted — an interaction Δ the 3-rung ladder doesn't isolate)? Propose the 4th rung if so:
   R_D = oracle-plan + actor-adapt + FRESH-replan-at-drift, to catch plan×actor interaction.
2. **Entanglement.** Argue stale-plan→drift causation: if a wrong objective drives the actor off-manifold,
   then "fixing the plan" (staleness-offset, replan-on-drift) reduces actor-OOD INDIRECTLY, making the
   plan-vs-actor split a false dichotomy. What experiment separates cause from correlate?
3. **Failure modes of the prescribed cure.** Where does reward-ranked self-imitation / save-state GRPO
   FAIL or self-deceive (reward hacking screen_x, survival-sprint, distributional collapse, exploiting the
   emulator)? Tie to the eval bug we just fixed (expert-denominator). What guardrails?
4. **What would make you CONCEDE the consensus stands?** State the number/observation from the running
   discriminator (and your proposed 4th rung) that would settle it for you.

## Constraints (unchanged)
ONE generalist model (no per-game/per-genre); VLM-LoRA allowed but flag cost; frozen-VLM-first; preserve
exact null-invariance; kill by numeric PID; no git commit; cite PRIMARY sources only.

## Mechanics
Opus → `ARCH_WARROOM_opus_r4.md`; GPT-5.5 → `ARCH_WARROOM_gpt55_r4.md` (absolute paths in each agent's
instructions). Orchestrator injects the discriminator RESULTS, then runs a short 4b cross-rebuttal + referee.
