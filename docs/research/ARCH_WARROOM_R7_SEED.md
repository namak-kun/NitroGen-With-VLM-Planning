# ARCH WARROOM — Round 7 seed: the R6 recipe FAILED. Diagnose + pivot.

The unanimous R6 recipe was IMPLEMENTED and ablated multi-seed. It FAILED on the target game. This round:
diagnose WHY honestly, and pivot to something that actually works. No face-saving — the data is the boss.

## WHAT WE BUILT (exactly) and RAN
Recipe = `--reward-mode rtg --gamma 0.97 --advantage-norm --residual-mask --anchor 0.01 --lr 3e-5 --steps 120`,
plan-head+LoRA trainable, --use-correct-plan, on btn_s600. Residual mask: per-(B,H,25) actions_mask =
clip((|a_chunk − a_base_null| − |a_base_null − a_base_null2|)/0.3, 0.05, 1) — plan/LoRA gets gradient ONLY
where the reward-kept chunk deviates from the FROZEN base-DiT null action. C2 = same WITHOUT the residual mask
(ones-mask). Eval = native env reward, baseline = R_B (oracle-plan, NO adapt), 5 paired seeds SMW / 3 Sonic.

## THE RESULTS (multi-seed, paired, bootstrapped)
| condition         | post mean (std) | Δ vs oracle-no-adapt | verdict |
|-------------------|-----------------|----------------------|---------|
| SMW C2 naive      | -0.113 (0.098)  | **-0.489**           | COLLAPSE (all 5 seeds negative) |
| SMW C3 residual   | -0.261 (0.250)  | **-0.638**           | WORSE collapse |
| Sonic C3          | +1.710 (0.832)  | +0.585               | helps (noisy) |
Paired SMW C3−C2 (post) mean **−0.149**, bootstrap 95% CI [−0.293, +0.037] → residual mask WORSE in 4/5 seeds.
Cross-run: this C2 (rtg+advnorm) SMW −0.489 is MUCH worse than earlier PLAIN LOCAL RWBC SMW (~+0.42 post,
seeds 1-4). So BOTH new ingredients (rtg+advnorm AND residual-mask) HURT SMW.

## THE TWO HARD FINDINGS (both refute R5/R6 consensus)
1. **The residual mask DEEPENED the collapse (opposite of prediction).** Leading hypothesis: the ones-mask
   implicitly ANCHORS the policy to reproduce base behavior on frame-DETERMINED dims (a regularizer toward the
   base DiT); the residual mask REMOVES that anchor and trains ONLY the deviation dims → the plan-head drifts
   MORE. The "collapse-prevention" logic was backwards: the frame-determined dims were the ANCHOR, not noise.
2. **For the plan-OOD game (SMW), ALL on-policy RWBC variants (local / rtg / rtg+advnorm / +residual) fail to
   beat — and usually COLLAPSE — the inference-time oracle-plan benefit (Δ_plan=+0.20, low-variance).** The
   plan benefit is an inference-time property of the FROZEN plan-head that reward self-imitation DEGRADES.
   Sonic (actor-OOD) is the opposite: adaptation helps.

## Established context (don't re-derive)
- Dissociation: SMW=plan-OOD (oracle plan +0.20, actor-adapt ~0/negative), Sonic=actor-OOD (+1.9, plan redundant).
- Near-zero plan gradient on factual data: base DiT reproduces the action from the frame alone.
- Full-action residual does NOT separate SMW vs Sonic at the game level (1.14x); authority anti-correlates with benefit.
- Repo: rwbc_actor_adapt.py (flags above; nitrogen.py:690 raw_loss*actions_mask is the loss gate); plan-head =
  resampler+PlanAdapter; scripts/train_planner.py = the SUPERVISED Stage-1 plan trainer (the OTHER way to
  improve the plan, NOT reward-RWBC); reach_eval survival-weighted metric; emulator save-states.

## The questions for Round 7 (be decisive; the orchestrator will run your pivot tonight)

**Q1 — Diagnose the residual-mask failure.** Is the "anchor removal" hypothesis right? If so, the FIX is the
INVERSION: instead of masking the loss toward the residual, ADD an explicit anchor-to-base regularizer on the
frame-determined dims (train the deviation dims normally BUT keep frame-determined dims pinned to base). Or is
the residual mask simply the wrong tool? State the concrete corrected loss (it must be a small edit to the
existing actions_mask path) OR declare it dead.

**Q2 — The real pivot: STOP actor-adapting plan-OOD games.** The data says SMW wants the FROZEN plan-head +
better PLAN, not RWBC. Concretely, which of these tonight:
  (a) **Supervised plan-head improvement** via scripts/train_planner.py-style Stage-1 (the residual as a
      SUPERVISED target/weight on the DEMO action, not on reward-kept self-imitation chunks) — i.e. fit the
      plan-head to the HUMAN demo action at frame-underdetermined states, no env reward, no drift.
  (b) **VLM re-abstraction of the plan** (text_only objective) to raise inference-time Δ_plan, no adaptation.
  (c) **Inference-time only**: accept SMW = frozen plan-head, and route ONLY actor-OOD games to RWBC.
  Pick the ONE to run tonight + the exact command/metric. Reserve RWBC for Sonic-like games.

**Q3 — Was rtg+advnorm the bigger culprit than the residual mask?** The cross-run evidence says plain local
RWBC was the least-bad for SMW. Should the actor-OOD (Sonic) recipe drop advnorm too, or keep it? Propose the
cleanest control to attribute the damage (e.g. SMW local-no-advnorm-no-residual vs rtg-advnorm), runnable now.

**Q4 — Is "ONE generalist recipe" even the right target,** given SMW and Sonic want OPPOSITE updates? Options:
a single SUPERVISED objective (plan-head fit to demo at high-residual states for ALL games — which naturally
does nothing for Sonic where residual≈0, and helps SMW) might be the true unifier that RWBC isn't. Argue for
the unifying SUPERVISED objective vs accepting per-game-MODE routing (still one model, two training modes).
Take a stance.

**Q5 — The ONE experiment to run tonight** that most advances "the setup works", given the failure. Concrete
command on the existing harness + the PASS number. (Likely: supervised plan-head fit to demo actions weighted
by the frame-counterfactual residual — no env reward — evaluated by inference-time Δ_plan / reach_eval, multi-seed.)

## Constraints
ONE model (per-game training MODES ok, NOT per-game weights/heads); frozen-VLM-first; exact null-invariance;
recoverable (additive flags / new files); no commits; cite primary sources. Be willing to DISCARD the residual
mask and the RWBC-for-everything framing — the data earned that.

## Mechanics
GPT-5.5 → `ARCH_WARROOM_gpt55_r7.md`; Opus-4.8 → `ARCH_WARROOM_opus_r7.md`. The orchestrator implements the
agreed pivot tonight and reports back. CONVERGE on ONE runnable pivot.
