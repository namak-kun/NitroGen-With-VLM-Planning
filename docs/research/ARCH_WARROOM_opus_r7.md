# ARCH WARROOM R7 — Opus-4.8: I was wrong about the residual mask. The pivot.

**Own it up front.** I proposed residual-mask-as-loss-gate in R5 and made it the "unifier" centerpiece in
R6. Multi-seed paired ablation refuted it: SMW C3 (+residual) Δ=−0.638 < C2 (naive) Δ=−0.489, residual
WORSE in 4/5 seeds (CI [−0.293,+0.037]). The data is the boss. The residual idea was right as a
**supervised target weight** and wrong as a **self-imitation loss gate**. That distinction is the whole
pivot.

## Q1 — Diagnose my residual-mask failure. "Anchor removal" is right — plus a second mechanism.

Yes, anchor-removal is correct, and there is a compounding second effect I missed. Look at the loss
(`nitrogen.py:690-693`): `action_loss = Σ(m·raw_loss) / (Σm + 1e-6)`. The mask both (1) **down-weights the
frame-determined dims by ~20×** (floor 0.05 vs 1.0) — removing the implicit anchor — AND (2) **renormalizes
by Σm**, so the surviving deviation dims get a *larger* effective LR. With ones-mask the BC target on
sprint-right dims ≈ base behavior (base DiT already emits it), so that term is an implicit **KL-to-base
trust region** (the same regularizer AWR's exp-advantage and GRPO's KL-to-ref enforce explicitly —
Peng 2019 1910.00177; Shao 2024 2402.03300). My mask deleted the trust region *and* amplified gradient on
exactly the reward-hacked deviation dims the top-frac filter selected. Under seed noise those deviations
are noise → self-imitation amplifies its own covariate shift (Ross & Bagnell 2010; Ross et al. 2011
1011.0686): unconstrained BC on policy-generated rollouts compounds error.

**The literal corrected loss (the inversion), as a small edit to the actions_mask path:** keep ones-mask
BC (restore the anchor) and *add* an explicit pin-to-base on low-residual dims —
`L = mean[MSE(pred,vel)] + λ·mean[(1−m)·‖pred − v_base‖²]`, with `v_base` the cached base-null velocity
(we already cache `base_null`). I.e. use `(1−m)` to gate an **anchor-up**, not `m` to gate the BC **down**.

**But I concede it's not enough, so the gate is dead.** The inversion at best recovers C2 (ones-mask),
which *also* collapsed SMW (−0.489). The deviation-dim target is still the reward-selected self-imitation
chunk. Fixing the anchor doesn't fix the poisoned target. **Verdict: kill residual-mask-as-loss-gate in
RWBC. Move the residual to where the target is trusted.**

## Q2 — The pivot: STOP actor-adapting plan-OOD games. Run (a).

Pick **(a): supervised plan-head fit to HUMAN demo actions, residual-weighted, no env reward.** This is my
residual idea resurrected in its *correct* form. The harness exists: `demo_bc.py` already loads gold demos
(`docs/demos/.../demo.npz`, 12-button→25-dim) and reuses `build_batch` — whose `base_null` path *is* the
residual weight. Four additive changes (recoverable, no commits): train **plan-head** not `--lora-only`;
add `base_null/base_null2` per demo chunk (mirror `collect()`); condition on `BATTERY['smw']['correct']`;
`--save-delta`. Keep **null_mode=masked** so null stays bit-exact while we fit only conditional rows.

```
RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
for s in 0 1 2 3 4; do
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/demo_bc.py --game smw \
    --train plan_head --residual-mask --use-correct-plan --lr 2e-5 --steps 600 \
    --seed-offset $s --save-delta runs/r7/smw_planfit_s$s.pt
done
$RUN .venv/bin/python planner_poc/reach_eval.py --games smw sonic \
  --policies base planfit:smw=runs/r7/smw_planfit_s0.pt --horizons 450 900 --n-starts 8 --seeds 5
```
**Metric:** the low-variance inference-time `Δ_plan = R_B(delta+correct plan) − R_A(base+live)` (frozen
plan-head Δ_plan=+0.20, std~0.06), plus reach_eval headline (reach×surv). Reject (b) VLM re-abstraction
(bigger change; defer) and (c) route-only (that's the *fallback*, not an advance).

## Q3 — Was rtg+advnorm the bigger culprit? Yes.

Cross-run: plain-local RWBC SMW ≈ noisy-around-0 ({−0.13,+0.58,+0.70}); rtg ≈ noisy-around-0; **rtg+advnorm
C2 = −0.489 negative in ALL 5 seeds.** advnorm turned "noise around zero" into a *consistent* collapse — it
+positive-clamps the group-relative advantage, concentrating the update on the most-extreme screen_x sprints
(the suicide mode). So rtg+advnorm (−0.9 swing) ≫ residual mask (−0.149) as the damage source.
**Cleanest control runnable now:** 3-arm paired ladder on SMW, ≥5 seeds, no residual —
`{--reward-mode local}`, `{--reward-mode rtg}`, `{--reward-mode rtg --advantage-norm}` — isolates rtg then
advnorm. **Keep advnorm for Sonic** (actor-OOD; its win +0.585 survived it; group-standardization is the
principled GRPO baseline). The harm is specific to obstacle games we are no longer adapting.

## Q4 — Is "ONE generalist recipe" right? Yes — but the SUPERVISED objective, not RWBC.

**Stance: the single supervised residual-weighted demo-fit IS the true unifier.** It self-allocates by
construction: where the frame underdetermines the expert action (SMW pits/enemies) the residual is large →
the plan-head gets gradient; where the frame determines it (Sonic "run right") residual≈0 → m→floor →
**near-no-op**. One loss, one data type (expert demos), all games — exactly the auto-allocation R5 claimed,
but now on **trusted** targets where it actually works (no reward, no rollout, no drift, no seed-variance —
the thing that voided every RWBC claim). RWBC-for-everything was the wrong target because SMW and Sonic want
opposite *self-imitation* updates; a supervised objective doesn't pull them apart because Sonic's residual
is ~0. RWBC stays as a **second training MODE** for actor-OOD games that must *exceed* the demo (Sonic
+1.9), routed by the measured Δ_plan/Δ_actor — still ONE model, two modes, no per-game weights. Primary =
the unified supervised objective; routing is an additive refinement, not the headline.

## Q5 — The ONE experiment tonight + PASS number.

**Run the Q2 command:** supervised, residual-weighted plan-head fit to SMW human demos on the correct plan,
≥5 seeds, eval by inference-time Δ_plan + reach_eval; Sonic as the no-op guard.

**PASS (decisive):**
1. **SMW Δ_plan ≥ +0.30**, 95% bootstrap-CI lower bound **> +0.20** (beats the frozen plan-head beyond its
   ~0.06 noise) over 5 seeds. *(This is the advance: the plan channel improved, drift-free.)*
2. **No-harm guard:** Sonic reach_eval headline within ±0.05 of frozen (residual≈0 ⇒ near-no-op); SMW
   reach_eval headline ≥ frozen plan-head (no collapse); generalist rollup ≥ frozen.
3. **Invariant:** null output bit-identical (masked null-mode + `apply_null_mask`), delta additive/recoverable.

**Honest risk (logged):** base DiT already reproduces ~81-83% of the semantic action (`narration_residual2`)
⇒ the type-A residual is *small* ⇒ the gain may be modest. Mitigate by lowering `--residual-scale` (0.3→0.15)
to sharpen the weight. Even a modest **reproducible** +Δ_plan is the win the project needs after RWBC's
seed-variance debacle: it shows the plan-OOD lever is improvable **without** the actor-adapt collapse.

---
**Citations.** AWR — Peng et al. 2019, arXiv:1910.00177. GRPO — Shao et al. 2024, arXiv:2402.03300. DAgger /
no-regret online IL — Ross, Gordon, Bagnell 2011, arXiv:1011.0686. BC covariate-shift / compounding error —
Ross & Bagnell 2010 (AISTATS, *Efficient Reductions for Imitation Learning*). Confabulated post-hoc reports
— Nisbett & Wilson 1977. Repo: `rwbc_actor_adapt.py` (build_batch residual path:118-146 / advnorm:263-266);
`nitrogen.py` raw_loss·mask:690-693 / apply_null_mask:669; `demo_bc.py` (demo loader + build_batch reuse);
`plan_graded_test.py` BATTERY smw correct:43; `reach_eval.py` headline reach×surv:175-180 / delta load:89-94.
