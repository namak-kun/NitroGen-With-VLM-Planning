# GPT-5.5 Round 4b — Discriminator Closeout

## 1. Vindication scorecard

**Red-team was vindicated, but not globally.** My R4 concession required both games to show `R_B−R_A<=+0.05` and `R_C−R_B>=+0.10`; SMW breaks both (`Δ_plan=+0.20`, `Δ_actor=-0.01`), so R3's “plan not load-bearing” claim is false as a universal statement. Sonic goes the other way (`Δ_plan=-0.18`, `Δ_actor=+1.90`): actor-OOD dominates there.

Attack results:
- **Plan load-bearing off-manifold: HELD for SMW, NOT for Sonic.** Oracle plan helps SMW and hurts Sonic. The authority map also shows deployment CFG=8 plan authority is non-zero (~1.0–1.4), so “plan inert” is only a cfg=1/on-manifold statement.
- **Plan×actor entanglement: STILL OPEN.** The 3-rung ladder found no SMW actor-only gain and a huge Sonic actor gain, but it did not test replan-at-drift after adaptation. Authority only rises mildly with drift in SMW (+14%) and is flat in Sonic, so the mechanism is weaker than my strongest version.
- **Discriminator-blindness: HELD.** The discriminator did what it should: it falsified one global answer. R3 was over-aggregated across game genres/states.
- **Reward-hack worry: PARTLY DID NOT HOLD.** Sonic raw `screen_x` ceiling at 3.22 remains suspicious, but survival-weighted reach_eval still improves (`rwbc 0.584 > base 0.250`), so the Sonic actor win is not purely a camera/survival hack. Video audit still advised.

## 2. Is R_D worth running?

Yes, but **run it first on SMW**. SMW is the game where plan signal already has behavioral value while actor-only adaptation failed; that is exactly where `oracle-plan + actor-adapt + replan-AT-drift` can expose a missed plan×actor interaction. Sonic is lower priority: the plan currently hurts and actor adaptation already solves much of the objective.

Metric: paired starts/seeds, compare `R_D` vs `R_C`. Primary threshold: `Δ_interaction = mean(Return_D−Return_C) >= +0.05` native normalized return. Gate-local threshold: `Δ_recovery = mean(PostNReward_D−PostNReward_C) >= +0.05` over N=6 chunks after first gate, or `escape_rate_D−escape_rate_C >= +0.10`. Any of these validates replan-at-drift as useful even if full-episode return is noisy.

## 3. Minimal robustness check

Cheapest before trusting “SMW=plan, Sonic=actor”: **repeat the same 3-rung ladder on 2 additional seed blocks with paired starts** and report bootstrap CIs for `Δ_plan`/`Δ_actor`. Add survival-weighted reach only for SMW if the sign persists; held-out SMW starts are next, not first.

## 4. Replacement for R3

The data-backed claim is now: **plan vs actor OOD is game/state dependent, not globally actor-dominated.** In obstacle/decision-point games like SMW, frozen-VLM plan text can be behaviorally load-bearing and actor-only LoRA may do nothing; in open locomotion/speed games like Sonic, frame-conditioned actor coverage dominates and plan text can be redundant or harmful. Therefore keep the one-generalist, frozen-VLM-first, exact-null-invariant architecture, but make the training loop self-measuring: estimate per-game/per-state `Δ_plan` and `Δ_actor`, keep plan-head trainable where plan helps, allow `--lora-only` where actor-OOD dominates, and log `Δ_plan-off` during drift-triggered branch search rather than enforcing a global prescription. Source: `DISCRIMINATOR_RESULTS.md`.
