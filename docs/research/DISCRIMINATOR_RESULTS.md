# Plan-OOD vs Actor-OOD discriminator — RESULTS (run #2, robust) — 2026-06-28

Harness: planner_poc/rwbc_actor_adapt.py, btn_s600, native screen_x cumulative reward, eval n=10-12, seed0=100.
(Run #1 died from SESSION-TEARDOWN, not a segfault — `nohup &` in a sync shell got killed; run #2 used
detached async and completed. The Δ_plan sign/magnitude reproduced across both runs.)

3 rungs: R_A=live-plan/no-adapt, R_B=oracle-plan/no-adapt, R_C=oracle-plan + `--lora-only` actor-adapt.
Δ_plan = R_B − R_A (plan-OOD lever);  Δ_actor = R_C − R_B (PURE actor-OOD lever, plan-head frozen & oracle).

| game  | R_A live | R_B oracle | R_C oracle+adapt | Δ_plan  | Δ_actor | dominant      |
|-------|----------|------------|------------------|---------|---------|---------------|
| SMW   | +0.052   | +0.254     | +0.243           | **+0.20** | **−0.01** | **PLAN-OOD**  |
| Sonic | +1.268   | +1.089     | +2.991           | −0.18   | **+1.90** | **ACTOR-OOD** |

## HEADLINE: the answer is GAME-DEPENDENT — a clean DISSOCIATION (neither R3 nor naive-plan is universally right)
- **SMW = PLAN-OOD dominates.** The oracle plan HELPS (+0.20) and `--lora-only` actor adaptation does
  NOTHING (−0.01). SMW has pits/enemies/obstacle decision points where "jump over pits and enemies" supplies
  information the frame-only actor lacks. The R3 consensus ("plan not load-bearing") is FALSE here.
- **Sonic = ACTOR-OOD dominates.** Actor adaptation HELPS MASSIVELY (+1.90) and the plan slightly HURTS
  (−0.18). Sonic is an open "always run right" speed game — direction is obvious from the frame, so the plan
  adds nothing, but self-imitation on screen_x easily lifts raw progress. The R3 consensus holds here.
- This MAPS onto prior memories: left/right (Sonic's axis) is frame-obvious / env-recoverable; SMW's value is
  in obstacle-timing decisions the markov frame underdetermines. The plan-vs-actor question is GENRE-SHAPED.

## GUARDRAIL FLAG (red-team was right to worry)
Sonic R_C post-eval list = [2.31, 3.21, 3.22, 3.22, 3.22, 3.03, 2.73, 3.22, 2.53, 3.22] — saturates at ~3.22.
This looks like a CEILING / possible screen_x reward-hack (survival-sprint / camera-scroll exploitation, the
exact failure GPT-5.5 R4 flagged). The Sonic actor-OOD win MUST be audited on video / a survival-weighted
metric before it's trusted. Do NOT take +1.90 at face value yet.

## Against the agents' concession criteria
- GPT-5.5 R4 said it concedes R3 only if R_B−R_A ≤ +0.05 AND R_C−R_B ≥ +0.10 (both games). SMW VIOLATES this
  (Δ_plan=+0.20 ≫ 0.05, Δ_actor=−0.01 < 0.10) → on SMW the consensus does NOT stand; plan is load-bearing.
- Sonic SATISFIES the actor-OOD prediction (Δ_actor ≫ Δ_plan) but is confounded by the ceiling/hack.

## Implication for the cure (Opus R4 recipe)
Opus's recipe is SELF-CORRECTING by design (measures Δ_plan-off during branch-search; keeps plan-head
trainable if the plan helps off-manifold, else `--lora-only`). This dissociation VALIDATES that design choice:
a single fixed answer is wrong — the loop must MEASURE plan-vs-actor PER GAME/STATE and route accordingly.
=> Don't lock `--lora-only` globally (it zeroed SMW). Keep plan-head trainable + per-game advantage norm.

## Next corroboration: plan_authority_map.py (running)
Drives a null/base rollout (actor drifts on its own), measures plan authority ||chunk_correct − chunk_null||
stratified by actor-disagreement D_t and chunk index. Prediction from the dissociation: SMW authority RISES on
drift (plan re-acquires authority); Sonic authority stays low (frame-obvious). Results: authmap_{smw,sonic}.json.

## plan_authority_map RESULTS (corroboration) — null-driven drift, cfg=8, n=40 chunks/game
Measures ||chunk_correct − chunk_null|| (auth) and ||chunk_correct − chunk_wrong|| (dirsep) in stick+jump
space, stratified by actor-disagreement D_t (the agreed OOD detector) and chunk index t.

| game  | auth_dir low-D_t | mid | high-D_t | verdict (rise threshold 1.15x) |
|-------|------------------|-----|----------|--------------------------------|
| SMW   | 1.219            |1.270| 1.391    | mild RISE (1.14x) — borderline |
| Sonic | 1.171            |0.946| 1.118    | FLAT/non-monotone              |

### Interpretation (important nuance)
1. **The plan is NOT inert at deployment CFG.** auth_dir ~1.0–1.4 (sticks are [0,1]) at BOTH on-manifold
   (low-D_t/early-t) AND drift (high-D_t) states, for BOTH games. At cfg=8 the plan measurably moves the
   action everywhere. (Caveat: cfg=8 AMPLIFIES v_c−v_u 8x; the RAW per-step plan gradient at cfg=1 is smaller.
   But deployment uses cfg=8, so deployment-relevant authority is real, not ~0.)
2. **Authority ≠ benefit.** The plan moves the action everywhere, but whether that movement HELPS is the
   behavioral question — and that's the clean game-dependent dissociation (SMW plan helps; Sonic plan hurts).
   The authority map can't see direction-correctness; the discriminator reward can.
3. **Red-team's "authority rises off-manifold" is only WEAKLY supported (SMW, via D_t, +14%) and flat for
   Sonic.** So the mechanism is mild, not dramatic. The real signal is behavioral, not authority-magnitude.

## Sonic actor-OOD win is NOT purely a screen_x hack (guardrail partly cleared)
The Sonic R_C=+1.90 raw-screen_x gain is CORROBORATED by today's SURVIVAL-WEIGHTED reach_eval (scripted-RIGHT
ceiling, suicide-penalized): rwbc Sonic=0.584 > base=0.250. The actor-adapt Sonic win survives the
survival-weighted audit, so it's not solely the camera-scroll/sprint exploit the red-team feared (though the
3.22 ceiling on raw screen_x still warrants a video spot-check before final claims).

## BOTTOM LINE (what we gained beyond R3)
The R3 consensus ("actor-OOD dominates, plan never load-bearing") is OVER-GENERALIZED. The truth is a
GENRE-SHAPED DISSOCIATION:
- Obstacle/decision-point games (SMW): the PLAN is the lever; lora-only actor-adapt does nothing.
- Open speed/locomotion games (Sonic): the ACTOR (on-policy data) is the lever; the plan is redundant.
=> The cure must be PER-GAME/PER-STATE adaptive (Opus R4's self-correcting Δ_plan-off design), NOT a global
`--lora-only`. Keep the plan-head trainable; let the active loop measure plan-vs-actor and route. This is the
concrete, data-backed refinement of the R3/R4 consensus.

## MISSING CELL (Opus r4b's proposed experiment): oracle plan + FULL adapt (plan-head+LoRA trainable)
Drops --lora-only so plan_head.* AND lora_* train (107.57M params). Same RWBC self-imitation on screen_x.
| game  | R_B oracle/no-adapt | R_C lora-only | R_C FULL-adapt (plan-head trainable) |
|-------|---------------------|---------------|--------------------------------------|
| SMW   | +0.254              | +0.243 (Δ−0.01)| **−0.125 (Δ=−0.379, COLLAPSE)**     |
| Sonic | +1.089              | +2.991 (Δ+1.90)| +3.054 (Δ+1.965, ≈same; ceiling 3.22)|
SMW training trajectory: baseline +0.254 → step49 +0.084 → step99 +0.010 → post −0.125 (MONOTONIC COLLAPSE).

### DECISIVE FINDING: SMW's plan-lever is NOT trainable into the actor via RWBC — it COLLAPSES.
Making the plan-head trainable does not amortize SMW's +0.20 oracle-plan benefit into the actor; reward-ranked
self-imitation on screen_x ACTIVELY DESTROYS it (−0.38, worse than R_A=+0.052 AND lora-only R_C=+0.243). The
mechanism: RWBC BC-targets the actor's OWN top-screen_x chunks; for an obstacle game, max-screen_x = sprint
into pits/enemies (the survival problem). So self-imitation drags the plan-head toward survival-sprint and
corrupts the very plan-conditioning that helped. Sonic (open) LOVES that same signal (+1.97) → ceiling/hack.

### Upshot (sharpens the cure): the REWARD SIGNAL is the crux, not the trainable param set.
- Naive RWBC / raw-screen_x self-imitation is ACTIVELY HARMFUL for obstacle games (SMW) and hack-prone for
  open games (Sonic). This is the strongest evidence yet for the agents' guardrails: the GRPO/AWR loop MUST
  use a MULTI-COMPONENT reward (screen_x + survival + checkpoint), NOT raw progress, BEFORE any plan-head
  adaptation — otherwise SMW collapses.
- SMW's oracle-plan benefit is currently an INFERENCE-TIME property of the FROZEN btn_s600 plan-head; it is
  not (yet) trainable into the actor with the available reward. Options: (a) keep inference-time planning with
  frozen plan-head for obstacle games; (b) fix the reward (survival-weighted) THEN retry plan-head GRPO;
  (c) the adapter-collapse fix (token-sep probe + label cleaning) before adaptation.
- Sonic full-adapt == lora-only (plan-head trainable adds nothing) → Sonic is pure actor lever, and the 3.22
  ceiling persists → multi-component reward + audit mandatory.

### Net (final): the cure is GATED ON FIXING THE REWARD FIRST.
Before per-game GRPO with plan-head trainable can work as designed, the loop needs the survival-weighted /
multi-component reward (which we ALREADY built today: reach_eval's scripted-ceiling + survival-weighting).
Re-running this missing cell with THAT reward (not raw screen_x) is the single highest-value next step.

## REWARD FIX: return-to-go (RTG) credit assignment — the collapse was an ARTIFACT (2026-06-28)
Root cause of the SMW collapse: rwbc_actor_adapt collect() ranked chunks by LOCAL per-chunk env reward. The
SMW env DOES have a death penalty (life_var="lives" default, death_penalty=5.0), but it lands ONLY on the
chunk where the life decrements — the risky forward-sprint chunks BEFORE the death keep their +screen_x and
survive the top-40% filter. Credit-assignment failure.
FIX (implemented): added `--reward-mode rtg` to rwbc_actor_adapt.py — each chunk is scored by the DISCOUNTED
RETURN-TO-GO of its episode (G_t = r_t + gamma*G_{t+1}, gamma=0.95), so a sprint chunk that causes a death
2-3 chunks later inherits the -5 (discounted) and is correctly DROPPED. This is the AWR/GRPO return-based
advantage the agents prescribed; it's a real harness improvement for all future RL.

### RTG re-run of the missing cell (btn_s600, --use-correct-plan, n=10):
| config                         | LOCAL reward    | RTG reward       |
|--------------------------------|-----------------|------------------|
| SMW full-adapt (plan-head+LoRA)| -0.125 (COLLAPSE)| **+0.359 (Δ=+0.105 over R_B=+0.254)** |
| SMW lora-only                  | +0.243          | -0.033           |
| Sonic full-adapt               | +3.054          | +3.164 (still ceiling 3.22) |

### FINDINGS
1. **RTG FIXES the SMW collapse and FLIPS it to a WIN.** SMW full-adapt: -0.125 -> +0.359, a +0.48 swing,
   now BEATING the oracle-plan-alone baseline (+0.254). So SMW's plan-lever IS trainable into the actor —
   the earlier "collapse" was a local-reward credit-assignment artifact, NOT a fundamental limit.
2. **The lever for SMW is full-adapt (plan-head trainable), not lora-only.** lora-only+RTG = -0.033 (lora-only
   can't access the plan->action pathway that carries SMW's obstacle signal). Confirms: keep plan-head
   trainable; don't global-`--lora-only`.
3. **Sonic 3.22 is metric SATURATION, not a death-hack.** RTG (which would penalize deaths) left Sonic
   unchanged at the 3.22 ceiling -> the actor genuinely runs screen_x to the segment limit in 14 chunks; the
   raw-screen_x metric just maxes out. (Still worth a longer-horizon / level-checkpoint metric, but it's not
   the survival-sprint hack the red-team feared.)

### NET (updated bottom line)
The agents' core prescription is now DATA-VALIDATED end-to-end: keep the plan-head trainable globally + use a
RETURN-BASED (GRPO/AWR) advantage, NOT local-reward RWBC. With that one fix, the SAME setup that COLLAPSED
SMW (-0.38) now IMPROVES it (+0.36 > oracle plan), while Sonic continues to win via the actor. One generalist,
plan-head trainable, return-based per-game advantage-norm — handles BOTH genres. The single most decisive
lever turned out to be the REWARD/credit-assignment, exactly as the missing cell predicted.

## *** RETRACTION: the RTG "fix" was a SINGLE-SEED ARTIFACT *** (multi-seed robustness, 2026-06-28)
The "RTG fixes the SMW collapse (+0.359)" result above held ONLY at seed 0. Paired RTG-vs-local re-runs at
seeds 1 and 2 (SMW full-adapt, --use-correct-plan, n=10 eval each) REVERSE it:
| seed | RTG (post)        | local (post)      |
|------|-------------------|-------------------|
| 0    | +0.359 (good)     | -0.125 (collapse) |
| 1    | -0.209 (collapse) | +0.583 (good)     |
| 2    | -0.250 (collapse) | +0.696 (good)     |
The sign of Δ FLIPS with seed for BOTH reward modes. RTG: {+0.36,-0.21,-0.25}; local: {-0.13,+0.58,+0.70}.

### CORRECTED CONCLUSION
SMW RWBC actor-adaptation (120 steps, either reward mode) is **DOMINATED BY SEED VARIANCE** — you cannot
conclude RTG>local OR local>RTG, and you cannot conclude the SMW plan-lever is reliably trainable into the
actor. The earlier +0.48 RTG swing was NOISE. This also means the discriminator's single-seed SMW cell
(Δ_actor=-0.01) and the dissociation's SMW side are WITHIN the noise band and need multi-seed CIs before trust.
(Sonic's actor win +1.9 is far larger and reproduced across runs + survives reach_eval, so Sonic is the more
robust leg; SMW is in the noise.)

### Lesson (meta)
Single-seed RL/adaptation deltas on these high-variance retro envs are NOT trustworthy — exactly the trap the
fixed-eval work warned about. Require >=5 seeds + bootstrap CIs before any plan-vs-actor or RTG-vs-local claim.
The --reward-mode rtg FEATURE remains a principled, valid harness addition (return-based credit assignment is
correct in general); what is RETRACTED is the empirical claim that it fixes SMW. Running seeds 3-4 next for n=5.

## FULL-ACTION residual v2 (narration_residual2.py, 2026-06-28) — game-level STILL WEAK
Fixed v1's direction-only proxy: full semantic action {LEFT,RIGHT,UP,DOWN,JUMP} residual + plan-authority.
| signal                         | SMW    | Sonic  | SMW/Sonic | verdict |
|--------------------------------|--------|--------|-----------|---------|
| sem_mean (||a_real-a_null||)   | 0.189  | 0.166  | 1.14x     | WEAK (<1.3x) |
| auth_mean (||a_correct-a_null||)| 0.696 | 0.720  | 0.97x     | REFUTE — Sonic HIGHER |
- The full-action frame-counterfactual residual does NOT cleanly separate SMW (plan-OOD) from Sonic
  (actor-OOD) at the GAME level (1.14x). And plan AUTHORITY is slightly HIGHER for Sonic (0.97x) despite
  Sonic being the game where the plan does NOT help => **authority is anti-correlated with benefit** (strongly
  confirms 'authority != benefit'; the correct plan moves Sonic's action MORE, but in a way that doesn't help).
- IMPLICATION: the residual-mask's GAME-LEVEL auto-allocation claim (R5/R6) is NOT supported by the cheap
  proxy. BUT its PER-DIM, PER-CHUNK collapse-prevention is a LOCAL mechanism that does NOT require game-level
  separation: on frame-determined chunks (sprint-right) the residual on the direction dim is ~0 -> the
  plan-head gets ~no gradient there -> it can't be dragged toward the suicide-sprint that caused the SMW
  collapse. So the recipe's COLLAPSE-PREVENTION is intact even though AUTO-ALLOCATION is weak.
- base DiT reproduces ~81-83% of the human's semantic action in BOTH games (sem residual ~0.17-0.19) =>
  confirms near-zero-plan-gradient at the action level; the type-A residual to train the plan on is SMALL.
  Risk: residual-weighted training may give the plan-head little gradient (no collapse, but maybe little gain).
- DECISION: BUILD + TEST the recipe anyway (C2 naive vs C3 residual, multi-seed) — the empirical ablation is
  the real test; the game-level proxy weakness is logged as a caveat, not a blocker.

## *** R6 RECIPE ABLATION — THE CONSENSUS RECIPE FAILED ON SMW *** (multi-seed, 2026-06-28 night)
Ran the unanimous R6 recipe (rtg + per-game advantage-norm + residual-mask). C2=naive (no residual mask),
C3=+residual mask. SMW seeds 0-4 (paired), Sonic seeds 0-2. Native env reward; baseline = R_B oracle-plan no-adapt.
| condition         | post mean (std) | Δ vs oracle-no-adapt baseline |
|-------------------|-----------------|-------------------------------|
| SMW C2 naive      | -0.113 (0.098)  | **-0.489 (collapse, all 5 seeds)** |
| SMW C3 residual   | -0.261 (0.250)  | **-0.638 (collapse, WORSE than naive)** |
| Sonic C3          | +1.710 (0.832)  | +0.585 (helps, noisy)         |
Paired SMW C3-C2 (post): [-0.143,-0.325,-0.167,+0.217,-0.325] mean **-0.149**, bootstrap 95% CI [-0.293,+0.037]
=> the residual mask is WORSE than naive in 4/5 seeds (marginally significant).

### THE RECIPE WAS REFUTED. Two hard findings:
1. **The residual mask HURTS, opposite of the R5/R6 prediction.** Hypothesized to prevent the SMW collapse;
   instead it DEEPENED it (C3<C2, 4/5 seeds). LIKELY MECHANISM: the ones-mask implicitly ANCHORS the policy to
   reproduce base behavior on frame-determined dims (regularization); the residual mask REMOVES that anchor and
   trains ONLY the deviation dims → the plan-head drifts MORE, not less. The "collapse-prevention" reasoning
   was backwards: masking out the frame-determined dims removes the very anchor that limits drift.
2. **rtg + advantage-norm is WORSE for SMW than plain local RWBC.** This ablation's C2 (rtg+advnorm, no mask)
   SMW Δ=-0.489 is much worse than the earlier plain-local-RWBC SMW (post mean ~+0.42 across seeds 1-4).
   Advantage-norm + rtg concentrates/sharpens the update and collapses obstacle-game capability.

### What actually WORKS for SMW (from all runs to date): DON'T ADAPT.
The robust, low-variance SMW win is the INFERENCE-TIME oracle plan, NO adaptation: Δ_plan=+0.20 (R_B std ~0.06).
EVERY on-policy RWBC variant tried (local / rtg / rtg+advnorm / +residual-mask) either fails to beat it or
actively collapses it. For the plan-OOD game, the plan benefit is an inference-time property of the frozen
plan-head that on-policy self-imitation DEGRADES. Sonic (actor-OOD) is the opposite: adaptation helps (+0.585).

### Net: the generalist "one recipe" does NOT exist in the RWBC-adaptation family.
SMW wants the frozen plan-head (inference-time plan, no actor update); Sonic wants actor adaptation. A single
RWBC knob can't serve both. NEXT (R7): the models must pivot — likely (a) for plan-OOD games, do NOT actor-
adapt; keep the frozen plan-head and instead improve the PLAN (VLM re-abstraction / plan-head Stage-1 style
supervised, NOT reward-RWBC); (b) reserve on-policy RWBC for actor-OOD games; (c) the residual mask idea is
dead as a collapse-preventer (or needs inversion: anchor-to-base, not train-the-residual).

## Q3 ATTRIBUTION (from existing data) — rtg+advnorm is the CULPRIT; PLAIN LOCAL RWBC ACTUALLY WORKS for SMW
Comparing SMW full-adapt (--use-correct-plan, plan-head+LoRA) across reward/loss configs, same eval protocol:
| config (SMW full-adapt)     | post mean | Δ vs oracle-baseline | seeds positive |
|-----------------------------|-----------|----------------------|----------------|
| **plain local** (no rtg/advnorm/residual) | **+0.559** | **+0.236** | **3/4** ✅ |
| rtg + advnorm (C2)          | -0.113    | -0.489               | 0/5 ❌ COLLAPSE |
| rtg + advnorm + residual (C3)| -0.261   | -0.638               | 0/5 ❌ |
**=> The damage is rtg+advantage-norm (a +0.236 → -0.489 swing, ~0.73). Plain local RWBC HELPS SMW.**
Each war-room "sophistication" (rtg, then advnorm, then residual-mask) made it monotonically WORSE. Classic
over-engineering: the SIMPLEST recipe was the best. This CORRECTS the earlier "all RWBC variants fail SMW" —
plain local does NOT fail; only the rtg+advnorm+residual stack does.

### Revised picture of "the setup that works"
- **Plain local RWBC, full-adapt (plan-head+LoRA), --use-correct-plan** is neutral-to-POSITIVE on SMW
  (Δ+0.236) AND strongly positive on Sonic (the original +1.9). It may already be the one-generalist recipe —
  WITHOUT rtg/advnorm/residual.
- Why advnorm hurts obstacle games: standardizing advantage within a noisy small batch sharpens the update
  toward the highest-screen_x (sprint) chunks and removes the gentle min-max scaling that kept updates small;
  rtg compounds it. On Sonic (open) that's fine; on SMW (obstacles) it over-commits to the risky chunk.
- NEXT: a clean PAIRED 5-seed re-run of {plain-local vs rtg+advnorm} on identical eval settings to lock the
  attribution, + confirm plain-local Sonic. If plain-local wins both, "the setup works" = the simplest recipe.

## PLAIN-LOCAL CONFIRMATION GRID (2026-06-28 night) — the CLEAN dissociation
Plain local RWBC full-adapt (--use-correct-plan, lr 3e-5, anchor 0.01, no rtg/advnorm/residual), native reward:
| game  | per-seed Δ                                  | mean   | verdict |
|-------|---------------------------------------------|--------|---------|
| SMW   | {-0.177,-0.351,-0.216,-0.374,-0.595}        | -0.343 | HURTS (all 5) |
| Sonic | {+1.567,+1.582,+1.176}                      | +1.442 | HELPS (all 3) ✅ robust |

### THE HONEST CONSOLIDATED PICTURE (across ALL configs this session)
- **Sonic (actor-OOD): RWBC WORKS ROBUSTLY.** +1.44 here (lr3e-5/anchor), +1.9 earlier (lr1e-4/lora-only/rtg/
  advnorm) — POSITIVE in every config and seed tried. This leg of "the setup works" is SOLID and reproducible.
- **SMW (plan-OOD): RWBC is UNRELIABLE / variance-dominated.** Earlier lr1e-4-no-anchor: Δ+0.236 (3/4 pos);
  now lr3e-5+anchor0.01: Δ-0.343 (0/5). Sign flips with SEED *and* CONFIG. So the earlier "plain local helps
  SMW" was itself a config artifact — there is NO trustworthy RWBC win for SMW. (rtg/advnorm/residual all
  worse still.) RETRACT any "plain local fixes SMW" claim.
- This is the cleanest possible vindication of the R7 PIVOT: route actor-OOD games (Sonic) to RWBC (works);
  do NOT use RWBC for plan-OOD games (SMW) — use the SUPERVISED demo-fit (no reward/rollout/drift → no
  variance). The dissociation is now a TRAINING-MODE routing rule, empirically grounded.

### "The setup works" status
- Sonic (actor-OOD) via RWBC: WORKS (robust, +1.44, reproducible). ✅
- SMW (plan-OOD): RWBC fails reliably; SUPERVISED demo-fit (demo_bc.py R7) is the candidate, testing now.

## *** R7 SUPERVISED DEMO-FIT WORKS for SMW (the pivot succeeded) *** (2026-06-28 night)
demo_bc.py R7: train PLAN-HEAD ONLY (LoRA+base frozen → null path fixed) on human demo actions, conditioned on
the correct plan, no env reward. Eval = Δ_plan = (plan-advance − null-advance) screen_x from 8 FIXED demo start
states (low-variance). 5 SMW seeds:
| seed | Δ_plan PRE | Δ_plan POST | change | null PRE->POST (should be ~flat) |
|------|-----------|-------------|--------|----------------------------------|
| s0   | +40.5     | +32.0       | -8.5   | 40.1->36.5 |
| s1   | +11.4     | +53.4       | +42.0  | 47.6->46.4 |
| s2   | +41.6     | +88.6       | +47.0  | 44.1->32.0 |
| s3   | +60.1     | +66.0       | +5.9   | 29.2->37.5 |
| s4   | +24.8     | +40.1       | +15.4  | (flat)     |
**Mean Δ_plan +35.7 -> +56.0 (change +20.3, 4/5 widened, bootstrap 95% CI [+2.0,+38.7] EXCLUDES 0). +57% relative.**

### THE PIVOT SUCCEEDED — first TRUSTWORTHY SMW win this session
- Supervised demo-fit WIDENS the plan channel (+57%) where RWBC actor-adaptation COLLAPSED or sign-flipped it.
- Improvement is PLAN-CONDITIONAL: null path stays ~flat (LoRA frozen), the plan path rises (e.g. s2 plan
  +85.8->+120.6). So the plan-head genuinely learned to use the correct plan better.
- NO collapse in any seed (worst -8.5, vs RWBC's reliable negative blowups). Bootstrap CI excludes 0.
- WHY it works where RWBC failed: trusted human-demo target (not reward-selected self-imitation), no env
  rollout (no covariate-shift drift), no reward (no hacking), plan-head only (no LoRA leakage into null) ->
  none of the variance sources that voided every RWBC claim. Confirms the R7 unanimous diagnosis exactly.

### "THE SETUP WORKS" — both legs now demonstrated
- Sonic (actor-OOD) via plain local RWBC: +1.44 mean, 3/3 seeds, every config. ROBUST. ✅
- SMW (plan-OOD) via supervised demo-fit (plan-head): Δ_plan +57%, 4/5 seeds, CI>0, no collapse. ✅
=> ONE model, TWO training MODES routed by the dissociation: actor-OOD->RWBC, plan-OOD->supervised demo-fit.
Pending: residual-as-supervised-weight ablation (does it add over plain demo-fit?) + Sonic no-harm guard
(demo-fit should be ~no-op for Sonic, residual≈0).

## R7 follow-ups (2026-06-28 late)
- **NULL-INVARIANCE VERIFIED INTACT after supervised demo-fit:** loaded a plan-head demo-fit delta onto
  btn_s600, compared null-path _sample_chunk (null=True, fixed seed) to base: **max|diff| = 0.00e+00 (exact)**.
  The plan-head delta does NOT touch the masked-null path → the core contract holds. (Confirms the Sonic guard's
  null +2.8->+161 swing was EVAL RNG from 4 noisy starts, not a contract break.)
- **Sonic no-harm guard (demo-fit, 2 seeds):** plan path stays LARGE (s0 +151->+215, s1 +294->+214) — NOT
  collapsed; demo-fit doesn't catastrophically harm Sonic. Δ_plan appears to shrink but that's the noisy null
  path (4 starts). Inconclusive on a clean no-op claim, but MOOT: Sonic's assigned MODE is RWBC, not demo-fit.
- Residual-as-supervised-weight ablation: NOT completed (the all-chunks base-null precompute was too slow —
  1995 chunks x2 forwards before training; killed to keep the box clean). FOLLOW-UP: cap precompute to the
  sampled chunks (or --max-chunks) and re-run; secondary to the headline (plain demo-fit already works).

## ============ NIGHT BOTTOM LINE: "THE SETUP WORKS" (game-type-routed, one model, two modes) ============
| game type | example | WORKING training mode | result (multi-seed) | null-invariant |
|-----------|---------|-----------------------|---------------------|----------------|
| actor-OOD (open/locomotion) | Sonic | plain local RWBC (full-adapt) | +1.44 mean, 3/3 seeds, every config | yes |
| plan-OOD (obstacle/decision) | SMW   | SUPERVISED demo-fit (plan-head only, correct plan, no reward) | Δ_plan +57%, 4/5 seeds, bootstrap CI [+2.0,+38.7]>0, NO collapse | yes (0.0 diff) |
What does NOT work (all refuted multi-seed this session): RWBC for plan-OOD games (SMW) in EVERY variant
(local/rtg/advnorm/residual-mask) — variance-dominated, sign-flips, collapses. The residual-mask-as-loss-gate
DEEPENED the collapse. The fix was the R7 pivot: route by game type; use the SUPERVISED, trusted-target,
no-reward, plan-head-only fit for plan-OOD games.

## *** CAPSTONE: ONE GENERALIST (two training modes merged) WORKS — no interference *** (2026-06-28 late)
combine_eval.py: merge SMW plan-head demo-fit delta (plan_head.* from r7demo_smw_plain_s2) + Sonic RWBC LoRA
delta (lora_* from r7local_sonic_s0) onto ONE btn_s600. Eval from 6 fixed demo starts (deterministic seed):
| cell        | SMW Δ_plan | Sonic plan-advance |
|-------------|-----------|--------------------|
| base        | +30.3     | +119.2             |
| smw_only    | +70.2     | +333.0             |
| sonic_only  | +60.5     | +200.5             |
| **combined**| **+71.2** | **+333.0**         |
=> COMBINED KEEPS BOTH GAINS (SMW +71.2 >= smw_only +70.2; Sonic +333 >= sonic_only +200). NO INTERFERENCE.
Bonus: POSITIVE cross-transfer — the SMW plan-head delta also boosts Sonic (+119->+333) and the Sonic LoRA
also boosts SMW Δ_plan (+30->+60), because both deltas strengthen the "advance right" response and the plan
channel. The two modes write mostly-disjoint params (SMW->plan_head, Sonic->lora) so they stack cleanly.

### => "THE SETUP WORKS" — END TO END, ONE MODEL:
btn_s600 + SMW-mode plan-head demo-fit (plan_head) + Sonic-mode RWBC (lora) = ONE generalist that improves
BOTH the plan-OOD game (SMW Δ_plan +30->+71, +135%) AND the actor-OOD game (Sonic +119->+333) with null-
invariance intact. Caveats: eval is 6 fixed starts (small-sample, raw screen_x; directions robust, magnitudes
noisy); one delta-seed each (proof-of-concept; multi-seed averaging is the cleanup). The ROUTING (which mode
per game) + the MERGE (disjoint params) are the recipe.

## *** GENERALIZATION CONFIRMED: supervised demo-fit WORKS on MMX (a NEW game) *** (2026-06-28 night-2)
demo_bc --game mmx --train plan_head --use-correct-plan (trimmed demos), 3 seeds, Δ_plan from 2 MMX demo starts:
| seed | Δ_plan PRE | POST | change |
|------|-----------|------|--------|
| s0   | +26.5     | +80.5| +54.0  |
| s1   | +26.5     | +69.5| +43.0  |
| s2   | +49.5     | +72.5| +23.0  |
Mean Δ_plan +34.2 -> +74.2 (change +40.0, 3/3 widened, +117% relative). Null path stays ~flat (LoRA frozen).
=> The R7 supervised plan-head demo-fit recipe TRANSFERS to a new side-scroller OUT OF THE BOX — even cleaner
than SMW (3/3 vs 4/5). Confirms the recipe is game-general for plan-OOD platformers, not SMW-specific.
(Caveat: only 2 MMX demo start states -> eval is coarse; but the effect is large + consistent.)

## *** R8 HEADLINE: the POOLED GENERALIST works — ONE plan-head fit across 3 games *** (2026-06-28 night-2)
pooled_demofit.py: ONE supervised plan-head fit on POOLED SMW+MMX+SMB1 demos (each chunk on ITS game's correct
plan), LoRA+base frozen. Eval Δ_plan from fixed demo starts on SMW (screen_x) and MMX (xpos). 3 seeds:
| seed | SMW Δ_plan PRE->POST | MMX Δ_plan PRE->POST |
|------|---------------------|---------------------|
| s0   | +29.3 -> +59.3 (+30.0) | +40.0 -> +70.5 (+30.5) |
| s1   | +29.3 -> +58.3 (+29.0) | +40.0 -> +57.0 (+17.0) |
| s2   | +29.3 -> +48.2 (+18.9) | +40.0 -> +85.5 (+45.5) |
SMW change mean +26.0 (3/3>0, bootstrap CI [+18.9,+30.0]); MMX change mean +31.0 (3/3>0, CI [+17.0,+45.5]).
NULL-INVARIANCE: max|diff| = 0.00e+00 (exact). Delta = 38 plan_head tensors (recoverable).

### Against both models' PASS criteria (Opus: retention >=0.8x per-game, CI>0, null 0.0; GPT55: +10 both, null drift<=10):
- SMW: pooled +26 vs per-game +20 = **1.3x** (EXCEEDS per-game) ✅; CI>0 ✅; >+10 ✅
- MMX: pooled +31 vs per-game +40 = **0.78x** (~at the 0.8x bar) ✅~; CI>0 ✅; >+10 ✅
- null-invariance 0.0 ✅; no collapse seed ✅
=> PASS. ONE supervised objective over pooled obstacle-game demos GENERALIZES across games (SMW & MMX both
widen, 3/3 seeds). The same-mode-merge question (Q3) is ANSWERED: pooled fit > averaging/sequential; it ships
ONE plan-head delta that serves multiple plan-OOD games. SMW even BENEFITS from pooling (1.3x vs solo) =
positive cross-game transfer at the dataset level (SMB1+MMX demos help SMW).

### "THE SETUP WORKS AND GENERALIZES" — consolidated
- plan-OOD games (SMW, MMX, SMB1): ONE pooled supervised plan-head demo-fit widens Δ_plan on all tested
  (SMW +26, MMX +31, 3/3 seeds, CIs>0, null-invariant). Generalizes to NEW games out of the box.
- actor-OOD games (Sonic): plain local RWBC (+1.44, 3/3).
- merge: plan_head (pooled plan-OOD) + lora (actor-OOD RWBC) = disjoint, no interference (capstone).
- One model, two training modes, pooled within-mode. Artifacts: files/pooled_planfit_s{0,1,2}.pt.

## *** FULL ONE-GENERALIST: pooled plan-head + Sonic lora, 4 games, no interference *** (2026-06-28 night-2)
full_merge_eval.py: merge POOLED plan-OOD plan-head delta (SMW+MMX+SMB1, 38 tensors) + Sonic actor-OOD lora
(32 tensors) onto one btn_s600. Δ_plan (plan-advance − null-advance) from 6 fixed demo starts per game:
| game  | base | pooled-planhead | sonic-lora | FULL MERGE |
|-------|------|-----------------|------------|------------|
| SMW   | +29  | +59             | +36        | **+52**    |
| MMX   | +40  | +70             | +56        | **+70**    |
| Sonic | +72  | +285            | +173       | **+321**   |
=> FULL ONE-GENERALIST OK: every game RETAINED or improved over base (SMW +29->+52, MMX +40->+70, Sonic
+72->+321). No catastrophic interference. (SMW dips solo+59->full+52: the lora adds minor noise to SMW but
it's still +80% over base.) BOTH deltas help ALL games via positive cross-transfer (the pooled plan-head even
lifts Sonic +72->+285; the Sonic lora lifts SMW/MMX). Disjoint params (plan_head + lora) stack cleanly.
NOTE: earlier full_merge run had a loader bug (lora keys are 'model...attn1.to_q.lora_A', not startswith
'lora_'); fixed to substring match -> the corrected run above is authoritative.

## ============ NIGHT-2 BOTTOM LINE: the setup WORKS, GENERALIZES, and MERGES into one model ============
- plan-OOD games (SMW, MMX, SMB1): ONE pooled supervised plan-head demo-fit widens Δ_plan on all (SMW +26,
  MMX +31, 3/3 seeds, CIs>0, null-invariant). Generalizes to NEW games (MMX +117%).
- actor-OOD games (Sonic): plain local RWBC (+1.44 env reward, 3/3 seeds).
- ONE GENERALIST = pooled plan-head (plan_head.*) + Sonic lora (model...lora) merged on btn_s600: all 4 games
  retained/improved, no interference, null-invariance exact. ARTIFACT: files/{pooled_planfit_s*.pt, r7local_sonic_s0.pt}.
- 3D-ready plan (both models): demo-fit transfers wholesale; swap screen_x eval -> 3D distance-to-waypoint,
  directional plan -> landmark-relative; new N64 env on emulator_env base.

## Pooled generalist — RIGOROUS eval (8 SMW starts, 2 MMX starts, 3 seeds, paired bootstrap): MMX sig, SMW noisy
pooled_eval_rigorous.py (per-state paired Δ_plan, more starts): base SMW +15.6 -> pooled +33.8 (~2.2x, change
+18.1, CI [-15.4,+55.8] = ns at n=8, high cross-LEVEL variance); base MMX +38 -> pooled +69 (change +31.2, CI
[+6.3,+56.0] SIG>0). => the pooled generalist's MMX win is statistically clean; SMW mean clearly up (~doubled)
but variance across 8 levels widens the CI -> needs more starts/seeds for sig. The qualitative result (pooled
fit ~doubles Δ_plan on both) holds; SMW significance is a sample-size cleanup, not a failure.

## SMB1 x-address SOLVED + demo-fit confirmed (2026-06-28 night-2, user-requested)
SMB1 All-Stars world-x RAM = 0x7E0042 (<u2), found by CORRELATING WRAM bytes vs measured frame-scroll (0.96+)
+ smooth-monotone validation (70->105->166->...->1309 over a level; coarse 256-block counters rejected).
Online RAM maps were WRONG (gave 0x6D/0x86/0xD4 = all flat 0 in All-Stars). SMB1 env now eval-ready
(xpos 70->195 on right-run). make_smbas reads tmp/retro_data/smbas_progress.json.
SMB1 demo-fit (plan-head, 3 seeds, 8 eval starts): Δ_plan +35.6->+91.4, +17.9->+82.2, +33.9->+43.4
(mean +29.4->+72.3, change +43, 3/3 widened). => recipe now validated on a THIRD plan-OOD game (SMW, MMX, SMB1).

## 3-GAME pooled generalist (SMB folded in, paired bootstrap CIs, 2026-06-29)
pooled_eval_rigorous.py --games smw,mmx,smbas (the SAME pooled plan-head delta, now eval'd on all 3):
| game  | base Δ_plan | pooled Δ_plan | change | 95% CI        | sig |
|-------|-------------|---------------|--------|---------------|-----|
| SMW   | +4.5        | +27.2         | +22.8  | [+2.7,+42.5]  | SIG |
| SMB1  | +25.4       | +57.9         | +32.5  | [+2.1,+73.9]  | SIG |
| MMX   | +61.5       | +63.5         | +2.0   | [-11,+15]     | ns (base already high this run, n=2 starts) |
=> 2/3 plan-OOD games SIGNIFICANTLY improved by ONE pooled plan-head fit. SMB1 (the new, x-addr-solved game)
is now a clean SIG eval game. MMX ns is an n=2/high-base artifact, not a regression (per-game MMX was +117%).

## OPEN PROBLEMS NOT ADDRESSED BY THE DEMO-FIT WORK (user flagged 2026-06-29)
All demo-fit evals use --use-correct-plan (FIXED ORACLE plan) over a SHORT horizon (16 chunks) from fixed
save-states. This DELIBERATELY assumes away:
- STALE PLANS / plan flip-rate (we held the plan fixed+fresh; never tested staleness)
- PLANNER VARIANCE (used oracle plans, never the live VLM generate_plan)
- CLOSED-LOOP ACTOR DRIFT over long horizons (short save-state rollout; only Sonic-RWBC touches actor-OOD)
What we proved is the PREREQUISITE: the bridge CAN learn to use a good plan better. The R3/R5 solutions for
the open problems (Helix staleness-offset training; replan triggers; dynamic-CFG gate; r_t/D_t detection;
save-state GRPO for drift) are DESIGNED but UNIMPLEMENTED. Next: a staleness-robustness probe (eval Δ_plan
with a DELAYED plan + the LIVE VLM plan vs the fixed oracle) to quantify the gap.

## *** STALENESS/PLANNER-VARIANCE PROBE: demo-fit fixes the LIVE-plan deployment gap *** (2026-06-29)
staleness_probe.py: same model from fixed SMW demo starts, 4 plan regimes (n=6 starts, 1 seed). The crux:
deployment uses the LIVE VLM plan (generate_plan), NOT the fixed oracle we trained/eval'd on.
| metric                              | BASE   | POOLED demo-fit |
|-------------------------------------|--------|-----------------|
| oracle plan (ceiling)               | +8.5   | +62.7           |
| **LIVE VLM plan (real deployment)** | **-10.7** | **+56.3**    |
| planner-quality gap (oracle-live)   | +19.2  | +6.3            |
| staleness cost (live-stale)         | +2.8   | +0.2            |
FINDINGS (directly addressing the user's "stale plans still remain?" concern):
1. On BASE, the LIVE VLM plan HURTS (-10.7 < null) -> the oracle ceiling (+8.5) was misleading; real
   deployment was net-negative. This IS the planner-variance problem.
2. Demo-fit UNEXPECTEDLY FIXES it: live plan -10.7 -> +56.3 (nearly = oracle +62.7); planner-quality gap
   collapses +19.2 -> +6.3. So demo-fit's real value = making the bridge ROBUST to the VLM's imperfect live
   plans, not just "use the oracle better". The live-case improvement (+67) exceeds the oracle improvement (+54).
3. TEXT STALENESS is MILD for right-running platformers (+0.2-2.8): "move right jump obstacles" doesn't go
   stale between replans.
STILL OPEN: long-horizon closed-loop actor DRIFT (our rollouts = 16 chunks from save-states, not minutes);
latent-staleness / replan-FLIP jitter in a true real-time loop. Need the parked Helix staleness-offset
training + r_t/D_t drift gate. CAVEAT: n=6/1-seed/SMW -> replicate on MMX/SMB + more seeds.

## Staleness/planner-variance REPLICATED across 3 games + VIDEOS recorded (2026-06-29)
LIVE VLM plan (real deployment, generate_plan every 2 chunks) — does it help? (Δ_plan LIVE-null):
| game | BASE | POOLED demo-fit |
|------|------|-----------------|
| SMW  | -10.7 (HURTS) | +56.3 |
| MMX  | +38.5 | +61.5 |
| SMB1 | +10.2 | +53.3 |
=> demo-fit makes the LIVE VLM plan substantially more effective on ALL 3 games (most dramatic on SMW where
base was NEGATIVE). Planner-quality gap (oracle-live) on pooled: SMW +6.3, MMX +12, SMB -1.2 (live even beats
oracle). Text staleness small everywhere. So the planner-variance problem (live plan != oracle) is largely
ADDRESSED by demo-fit as a side effect; it makes the bridge robust to the VLM's imperfect live plans.
VIDEOS (docs/demo_videos_base/smw/ + docs/demo_videos_pooled/smw/, 20s each, live plan text overlaid):
state{0,1}__base.mp4 (null), base/state*__plan.mp4 (live plan on BASE, scored -10.7), pooled/state*__plan.mp4
(live plan on demo-fit, scored +56.3). Visual proof of the planner-variance fix. (user scp's them off.)
STILL OPEN: long-horizon closed-loop drift; latent-staleness/replan-flip jitter in true real-time. n small (2-6
starts, 1 seed) -> replicate w/ more seeds.

## DUCKING GAP — demo-fit ELIMINATES ducking (user-spotted in videos, 2026-06-29)
DOWN-press rate (6 SMW demo starts, oracle plan, 16 chunks):
| model  | DOWN d-pad | DOWN stick |
|--------|-----------|-----------|
| HUMAN demos | 1.7% | - |
| BASE   | 2.9%      | 28.8%     |
| POOLED demo-fit | 0.0% | 0.0% |
=> The base model ducks ~at human rate; the demo-fit emits ZERO DOWN -> it collapsed to pure right+jump. Cause:
ducking is rare in demos (1.7%) AND the "correct plan" ("move right, run, jump over pits and enemies") never
mentions ducking, so demo-fit concentrated mass on right+jump and dropped the rare defensive move. The Δ_plan
win came partly at the cost of behavioral DIVERSITY (situational survival moves the plan doesn't name).
FIXES (increasing effort): (1) plan mentions it ("duck under bullet bills") - cheap test of whether the bridge
CAN emit DOWN on command; (2) KL-anchor demo-fit to base to preserve rare actions; (3) more duck data / a
dedicated objective. VIDEOS: docs/demo_videos_{base,pooled}/smw/ all 8 states (base: null+plan; pooled: plan).

## *** DUCK-ON-COMMAND: demo-fit DESTROYS the bridge's plan-responsiveness (2026-06-29, decisive) ***
User noted their GOLD plan mentions ducking; BATTERY['smw']['correct'] (terse) does NOT. Tested DOWN-rate vs
plan content (4 SMW starts, 12 chunks):
| plan                         | BASE DOWN-dpad | POOLED DOWN-dpad |
|------------------------------|----------------|------------------|
| terse "move right..."        | 1.2%           | 0.0%             |
| "duck to dodge the bullet"   | 3.4%           | 0.0%             |
| "press down to duck..."      | 10.8% (9x!)    | 0.0%             |
=> The BASE bridge IS plan-steerable for ducking (1.2%->10.8% as the plan explicitly commands it). The POOLED
demo-fit is COMPLETELY UNRESPONSIVE (0% for ALL plans) -> the supervised demo-fit on ONE terse directional plan
CATASTROPHICALLY OVERWROTE the bridge's plan-responsiveness for rare actions. It didn't just bias toward
right+jump; it BROKE duck-conditioning entirely.
ARCHITECTURE IMPLICATION (central tension): demo-fit improves plan-following on the TRAINED axis (advance) but
DESTROYS the bridge's broader expressiveness (situational actions the training plan never named). Fixes to
weigh: (1) train on RICHER per-situation plans (the gold narration mentions ducking) not one terse plan;
(2) KL-anchor / preserve base plan-responsiveness; (3) ADD plan steering via a zero-init modulator (FastPlanMod)
instead of OVERWRITING plan_head; (4) higher plan bandwidth K=8->32 for situational commands. => re-open arch.

## *** R9: BOTH expressiveness fixes WORK (single-seed s0; confirms DATA root cause) (2026-06-29) ***
War-room R9 (GPT-5.5+Opus converged): the ducking collapse is a DATA artifact (one constant terse plan -> plain
BC ignores the plan -> drops rare duck). Ran the crossed design on SMW (plan_head-only, 600 steps, 8 demo
starts, eval Δ_plan + duck-on-command). DPAD channel (dim1, clean binary, matches the orig finding):
| cell                              | Δ_plan PRE->POST  | press_down DOWN-dpad | monotone | verdict |
|-----------------------------------|-------------------|----------------------|----------|---------|
| P0A0 control (terse, plain BC)    | +35.0 -> +30.1 (-4.9)  | 0.0%            | (collapse) | reproduces the 0% collapse |
| P1A0 situational (action-as-label)| +35.0 -> +53.6 (+18.6) | 4.8%            | terse 0 -> pd 4.8 | duck RECOVERS + Δ_plan widens |
| P0A1 KL-anchor (terse + L2 to base)| +35.0 -> +75.5 (+40.5)| 6.7%            | 0.2->3.0->6.7 | BEST: duck>5% monotone + biggest Δ_plan |
=> The control (the collapse recipe) reproduces 0% duck. BOTH fixes RESTORE duck-on-command (dpad, monotone in
plan-explicitness) AND WIDEN Δ_plan -- confirming the DATA root cause AND that fixing it does NOT cost advance.
KL-anchor is the standout (biggest Δ_plan +40.5, cleanest duck recovery, keeps the cheap single-plan data;
matches Opus's taxonomy-free steelman). anchor loss stayed ~0.001 (plan tokens already near ref -> non-distorting).
CAVEATS being resolved: (a) the STICK channel (dim22, what map_action ACTUALLY trains for human DOWN) was
measured at a noisy 0.5 thresh in these s0 logs -> re-probing at >0.6 + a combined stick|dpad battery;
(b) the old rollout 'null-invariance |8.75|' is SAMPLING NOISE not a real break (fixed to an action-level
bit-identity check: null chunk on fixed frame+seed); (c) 3-seed (s1,s2) + full 5-contrast expressiveness battery
(duck/retreat/jump/wait/up) RUNNING. KEY DIM NOTE: human DOWN -> STICK (dim22) in map_action, NOT dpad_down
(dim1); the demo-fit can only TRAIN the stick channel, so stick is the faithful 'did training restore duck'
signal; dpad recovery is incidental (and still present).

## *** R9 FINAL: KL-anchor WINS — fixes ducking collapse w/o destroying expressiveness (3-seed) (2026-06-29) ***
Crossed design fully run (SMW, plan_head-only, 600 steps, 8 demo starts). THREE measures: Δ_plan (advance),
duck-on-command (the trained STICK channel dim22>0.6 + the base-native DPAD dim1), and a 5-contrast
EXPRESSIVENESS BATTERY (duck/retreat/jump/wait/up; responsiveness R = mean clip(d_i/b_i,0,1) over the 4 clean
contrasts retreat/jump/wait/up vs base).

| cell | Δ_plan (3 seeds) | mean | duck-on-command | expr-battery R (retreat/jump/wait/up) |
|------|------------------|------|-----------------|----------------------------------------|
| P0A0 control (terse, plain BC) | -4.9 (s0)        | -4.9 | 0% BOTH channels (collapse) | 0.165 (0.09/0.50/FLIP/0.07) = kills ALL |
| P1A0 situational (action-label) | +18.6/+46.9/+32.9 | +32.8 | STICK terse~2-5% -> press_down 12.6-98.2% MONOTONE (restored!) | 0.53 (0.03/1.0/1.0/0.09): keeps jump+wait, LOSES retreat+up |
| P0A1 KL-anchor (terse + L2-to-base) | +40.5/+42.6/+52.1 | +45.1 | DPAD 0.2->3.0->6.7% monotone; combined +0.186 | 0.83 (1.0/0.31/1.0/1.0): PRESERVES retreat+up+wait, PASS |

CONCLUSIONS:
1. ROOT CAUSE = DATA, CONFIRMED. The control (constant terse plan, plain BC) reproduces the 0% duck collapse AND
   kills the whole action space (battery R=0.165: wait FLIPPED, up/retreat ~gone). The war-room called it.
2. BOTH fixes restore duck-on-command AND widen Δ_plan (advance NOT sacrificed). Fixing expressiveness is FREE.
3. **KL-anchor is the WINNER on BOTH axes:** biggest+tightest Δ_plan (+45.1 mean, 3/3) AND best expressiveness
   (R=0.83, the only PASS) -- it preserves retreat+up that the situational TAXONOMY lost (those are
   under-labeled: 'up' has no SIT plan, 'retreat' washed out). Taxonomy-free L2-to-base preserves the WHOLE
   action space by construction (exactly Opus's steelman). It also keeps the cheap single terse plan (no
   per-chunk labeler/threshold to maintain) and anchor loss stays ~0.001 (non-distorting).
4. Situational is a clear 2nd (R=0.53 >> control 0.165): good where the action is labeled (jump/wait/duck
   restored, duck STICK press_down up to 98%!), but the taxonomy is a treadmill (loses unlabeled actions).
RECOMMENDATION: adopt KL-anchor (lambda=0.3) as the demo-fit objective; optionally STACK situational labels for
extra duck/stick authority. Re-pool the generalist with --kl-anchor. KEY DIM NOTE confirmed: human DOWN->STICK
(dim22); the base ducks via DPAD (dim1); a COMBINED stick|dpad metric mis-reads BASE duck as -0.109 (base stick-y
rests high) so report per-channel. null-invariance now an ACTION-LEVEL bit-identity check (not rollout noise).
Deltas: files/r9_smw_{ctrl,situ,kl}_s*.pt ; battery JSON: files/expr_battery_r9_smw.json.
Videos (all 4 games, ALL demo states, base+pooled): docs/demo_videos_{base,pooled}/{smw,smbas,mmx,sonic}/ (132 mp4).

## *** R11 STALENESS FORWARD-PASS TTA: staleness is TEXT-bound, NOT token-staleness (2026-06-29) ***
Built fresh/cached/fresh_ema modes into staleness_probe.py (added plan_hidden_override to eval_policy so the K
plan tokens can be cached/EMA'd; the DiT still sees the live frame). KEY ENCODING FINDING: the OLD 'stale' mode
already re-grounded tokens on the fresh frame every chunk (text_only encode is frame-CONTEXTUALIZED) -> the TRUE
fully-stale baseline (cache tokens at t0) was missing. Added it as 'cached'. Modes: null / oracle (curated
BATTERY text, re-grounded) / live (regen text every 2 chunks) / cached (tokens fixed at t0) / fresh (t0 text,
re-ground tokens every chunk = forward-pass TTA) / fresh_ema (fresh + EMA + 4-frame aug). Survival-aware advance
(death cutoff). SMW, 3 seeds, n=6 starts, frozen btn_s600:
| seed | null | oracle | live | cached | fresh | fresh-cached | oracle-fresh |
|------|------|--------|------|--------|-------|--------------|--------------|
| 0    | 44.8 | 53.8   | 41.0 | 32.7   | 34.2  | +1.5 (7%)    | +19.7        |
| 1    | 36.8 | 85.7   | 48.3 | 49.5   | 48.2  | -1.3 (-4%)   | +37.5        |
| 2    | 50.2 | 55.5   | 52.7 | 40.3   | 49.7  | +9.3 (62%)   | +5.8         |
| mean | 43.9 | 65.0   | 47.3 | 40.8   | 44.0  | +3.2         | +21.0        |
DECISIVE FINDINGS (robust across all 3 seeds even though magnitudes are noisy):
1. **tok-drift ~0 EVERYWHERE** (fresh 0.0007-0.0012, cached 0.0000). The K plan tokens barely change frame-to-
   frame -> the continuous SHORT PATH IS REMARKABLY STABLE. The owner's '50-60% plan flip' is NOT in the plan
   tokens; it's in the TEXT (live mode has the highest drift 0.0097 because the text regenerates).
2. **fresh ~= cached ~= null (44/41/44), all << oracle (65, oracle best in 3/3 seeds).** Re-grounding a stale
   t0-text on fresh frames does NOT recover staleness -> there is no 'token staleness' to fix. The gap is plan
   TEXT QUALITY (curated oracle text vs VLM/t0 text), oracle-fresh = +21 mean.
3. live ~= fresh (regenerating the text every 2 chunks barely beats holding the t0 text, +3.3) AND live has the
   highest token drift -> regenerating churns the tokens without an advance payoff.
=> Opus's R11 FALSIFIER FIRED: 'fresh ~= stale ⇒ staleness is TEXT-bound.' The von Oswald 'short path is already
TTT' re-grounding does NOT help here because the tokens are already stable. REDIRECT: the lever is PLAN TEXT
QUALITY (VLM-abstracted plans / Qwen-LoRA / better prompting), NOT token-level forward-pass TTA and NOT actor
weight-TTT. This matches the owner's 'use VLM-abstracted clean plans' instinct and motivates training the VLM.
(MMX 3-seed cross-game check + maneuver router RUNNING.) Files: r11_stale_smw_s{0,1,2}.json.

## R11 staleness MMX CONFIRMS (cross-game) + the redirect (2026-06-29)
MMX (3 seeds, null is ~1.5 so the plan is very load-bearing): cached/fresh/oracle all ~40-49; fresh-cached
-1.5/-2.0/-7.5 (re-grounding does NOT help); **live-fresh -0.5/-11.0/-8.0 (LIVE re-planning HURTS vs a FIXED
plan, by up to 11)**; tok-drift cached 0.000, fresh ~0.0013, live HIGH (0.022-0.031), EMA->~0.0001. =>
CONSISTENT WITH SMW: the continuous short path is stable; staleness is TEXT-bound. STRONGER on MMX: the VLM's
live re-planning is the WORST non-null mode -- it churns the plan text (high drift) without payoff. A FIXED good
plan beats live re-planning. REDIRECT (both games): the stale-plan lever is (a) plan TEXT QUALITY/STABILITY
(VLM-abstracted plans, Qwen-LoRA, or just replan-less/hold a stable plan), NOT token forward-pass TTA and NOT
actor weight-TTT. Forward-pass token-TTA (Opus's tonight pick) is REFUTED as the staleness fix BECAUSE there is
no token-staleness to fix (tokens already stable); the named falsifier fired cleanly.

## *** R11 MANEUVER ROUTER: does the DiT know spin-jump-kills-Rex? (the owner's question) (2026-06-29) ***
Frozen btn_s600. Reconstructed save-states ~30 frames BEFORE each narrated maneuver (replay demo.npz actions
from initial.state). NOTE: SMW 'score' var does NOT register kills (verified on the human's own spin-jump-kill ->
score flat), so the emulator outcome = SURVIVE + ADVANCE past the threat (>=25px, no life lost) = 'handled the
rex' (kill OR precise dodge), NOT strictly 'killed' (true kill needs sprite-status RAM, follow-up).
| maneuver | n | reflex(null) | evoke(plan) | evoc_ratio | outcome null->plan (Δ) | read |
|----------|---|--------------|-------------|-----------|------------------------|------|
| duck (bullet-bill states) | 3 | 0.457 | 0.571 | 1.25 | - | REFLEXIVE at bullet states (DiT already ducks 46%); plan adds little. (Contrast R9: duck EVOCABLE from generic starts.) State-dependent. |
| spinjump_rex | 10 | 0.039 | 0.281 | 7.24 | 0.04 -> 0.07 (+0.04) | JUMP button strongly EVOCABLE (7.2x) BUT the OUTCOME (handle the rex) barely moves (both ~5%). |
ANSWER to 'does the DiT know spin-jump-kills-Rex?': it knows how to JUMP ON COMMAND (7.2x evocable), but it does
NOT reliably execute the spin-jump-KILL OUTCOME (survive+advance only 3.75%->7.5%, both near chance). So the
precise rex-dispatch maneuver is an ADDITION at the OUTCOME level even though the jump primitive is evocable ->
route to demo-fit/DiT-LoRA on the human's rex-kill chunks (the long path), NOT plan-head evocation alone.
CAVEAT: outcome absolute rates are low for both (hard reconstructed scenario, 4 chunks to reach+pass the rex);
the RELATIVE +0.04 is the signal. verdict-gate logic over-weighted the button evoc_ratio (labeled EVOKE) -- for
OUTCOME maneuvers the outcome-Δ should decide (=ADD here). Tools: planner_poc/maneuver_router.py,
r11_router_smw.json. duck-evocability is STATE-DEPENDENT (reflexive near a real bullet, evocable from generic).
