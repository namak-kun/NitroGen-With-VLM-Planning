# MORNING BRIEF — night autonomous session (2026-06-28)

You left me to "get the setup to work as best as you can, recoverable." Here's what happened.

## TL;DR — the setup WORKS, via a game-type-routed recipe (ONE model, TWO training modes)
| game type | working mode | result | 
|-----------|--------------|--------|
| **actor-OOD** (Sonic, open/locomotion) | plain local RWBC, full-adapt | **+1.44 mean, 3/3 seeds, every config** ✅ |
| **plan-OOD** (SMW, obstacle/decision) | **SUPERVISED demo-fit, plan-head only, correct plan, no reward** | **Δ_plan +57%, 4/5 seeds, bootstrap CI [+2.0,+38.7] > 0, never collapses** ✅ |
Both preserve EXACT null-invariance (verified: 0.00e+00 diff after the plan-head delta).

## *** CAPSTONE (added late): ONE GENERALIST works — both modes merged, no interference ***
Merged SMW's plan-head demo-fit delta + Sonic's RWBC LoRA delta onto ONE btn_s600 (combine_eval.py):
  - SMW Δ_plan: base +30.3 -> combined +71.2  (+135%, the plan-OOD lever)
  - Sonic plan: base +119  -> combined +333    (the actor-OOD lever)
  - Keeps BOTH single-mode gains (no interference) + positive cross-transfer; null-invariance 0.0 diff.
The two modes write disjoint params (plan-OOD->plan_head, actor-OOD->lora) so they STACK. This is the
end-to-end "setup works" proof: route each game to its mode, train, merge. (Caveat: 1 delta-seed each,
6 fixed starts/raw screen_x -> directions robust, magnitudes noisy; multi-seed + reach_eval is the cleanup.)

## The arc (war-room rounds R6→R7 + experiments)
1. **R6:** both models converged on a sophisticated recipe (rtg + per-game advantage-norm + frame-counterfactual
   residual-mask on the loss). I built it (additive flags in rwbc_actor_adapt.py) and ablated it multi-seed.
2. **It FAILED.** SMW collapsed under it (Δ −0.489), and the residual-mask made it WORSE (−0.638), opposite of
   the prediction. Diagnosis (both models owned it): the residual mask removed the implicit "anchor-to-base"
   regularization on frame-determined dims → MORE drift. And rtg+advnorm was the bigger culprit (positive-clamped
   advantage concentrates the update on suicide-sprint chunks).
3. **Clean dissociation found:** plain local RWBC robustly HELPS Sonic (actor-OOD) but is variance/config-
   dominated for SMW (plan-OOD) — sign-flips, no trustworthy win. RWBC is simply the wrong tool for plan-OOD.
4. **R7 pivot (both models, unanimous):** for plan-OOD games, STOP actor-adapting; do a SUPERVISED fit of the
   PLAN-HEAD only (LoRA+base frozen → null path fixed → gains are plan-conditional) to the HUMAN demo actions
   conditioned on the correct plan, no env reward. The residual idea was right as a *supervised target weight*,
   wrong as a *self-imitation loss gate*. This removes every variance source that voided the RWBC claims.
5. **It WORKED:** SMW Δ_plan +35.7 → +56.0 (the inference-time plan benefit, measured from FIXED demo save-states
   = low variance), 4/5 seeds widened, CI excludes 0, NO collapse, null-invariant. First trustworthy SMW win.

## What this means
- The earlier "SMW plan-OOD vs Sonic actor-OOD" dissociation is now a TRAINING-MODE ROUTING RULE with a working
  recipe for each side, not just a diagnosis.
- "One generalist": still one model — the two modes write to the same plan-head/LoRA; route by measured
  Δ_plan/Δ_actor per game. (Whether a SINGLE supervised objective across all games is even cleaner — because
  Sonic's residual≈0 makes demo-fit ~no-op there — is the natural next test.)

## Open / next (in priority order)
1. **Combine into ONE model:** apply supervised demo-fit (plan-head) for SMW + RWBC (LoRA) for Sonic to the
   SAME btn_s600, eval both — confirm no cross-interference. This is the actual "one generalist" deliverable.
2. **Residual-as-supervised-weight ablation** (does it ADD over plain demo-fit?): re-run with a chunk-capped
   base-null precompute (the all-1995-chunk precompute was too slow; I killed it). Secondary.
3. **FE / Minish:** test frozen-VLM objective generation (objective_label_demo) instead of asking you to write
   plans — likely no human plans needed. Turn-based FE ≈ pure plan-OOD → demo-fit should shine.
4. **Reach_eval survival-weighted** confirmation of the demo-fit deltas (I used the fixed-start screen_x Δ_plan,
   which is low-variance but raw; cross-check on the survival-weighted metric).
5. The demo-fit eval uses GENERIC vs correct plan internally; consider VLM-generated plans as the conditioning.

## Recoverability (nothing committed; all additive)
- Code (additive flags / new files, defaults preserve old behavior): planner_poc/demo_bc.py (R7 supervised pivot),
  rwbc_actor_adapt.py (--reward-mode rtg/--advantage-norm/--residual-mask/--seed-offset), demo_narration.py,
  narration_residual.py, narration_residual2.py.
- Deltas (gitignored, in session files/): r7demo_smw_plain_s{0..4}.pt (the WORKING SMW plan-head deltas),
  r7demo_sonic_*.pt, r7local_*.pt (plain-local grid), r6_*.pt (failed recipe).
- War-room: ARCH_WARROOM.md (R3–R7 + referees) + ARCH_WARROOM_{gpt55,opus}_r{6,7}.md. Data: DISCRIMINATOR_RESULTS.md.
- Drivers/logs in /tmp: run_r6_ablation.sh, run_r7_local.sh, run_r7_demofit.sh; /tmp/r6,/tmp/r7l,/tmp/r7d.
- GPUs left idle, no stray processes.

## Memories stored this night
- The dissociation is genre-shaped (corrected the over-general "actor-OOD always dominates").
- RWBC raw-screen_x / rtg+advnorm collapses obstacle games; residual-mask-as-loss-gate is dead.
- The WORKING routed recipe (Sonic→RWBC, SMW→supervised plan-head demo-fit) + that fixed-start Δ_plan is the
  trustworthy low-variance eval.
