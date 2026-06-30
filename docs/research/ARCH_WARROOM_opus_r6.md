# ARCH WARROOM R6 — Opus-4.8: THE one-generalist recipe + the residual-mask unifier

**Convergent thesis.** Keep the plan-head trainable globally, gate its gradient with a **cheap cached
frame-counterfactual residual mask**, weight chunks by a **return-based GRPO/AWR advantage**, and judge by
**reach_eval survival-weighted headline from fixed save-states**. The mask is the *unifier*: it
auto-allocates plan capacity SMW-vs-Sonic with no per-game knob AND structurally prevents the SMW collapse,
independent of reward seed-noise. All additions are additive flags / cached tensors — recoverable, no
commits, frozen-VLM, exact null-invariance.

## Q1 — THE RECIPE (single, minimal, runnable)

**Params that train (one generalist):** `plan_head.*` (resampler + **PlanAdapter** + null) **+** `lora_*`,
**globally**. NEVER global `--lora-only` — the discriminator showed it zeroes SMW (Δ_actor=−0.01;
lora-only+rtg=−0.033). Base DiT frozen, Qwen frozen (`mm_text_only=True`, decode-free prefill, EXP-052).
The adapter (`plan_head.adapter.*`, nitrogen.py:283) is where steering localizes (R5) and it trains here.

**Reward / weighting:** `--reward-mode rtg --gamma 0.95` (return-to-go propagates death back to the
sprint-into-pit chunk — AWR/GRPO return advantage, arXiv:1910.00177 / arXiv:2402.03300). Replace the
harness min-max `rw` (rwbc_actor_adapt.py:236) with a **per-game group-relative advantage**
`Â_i=(R_i−mean)/(std+ε)`, positive-clamped `max(Â,0)+0.1` (AWR form, GRPO group baseline). Per-game
standardization dissolves the cross-game `screen_x`-scale "interference" artifact (R4). **Decision metric
is reach_eval `headline = reach×surv`, never raw screen_x.**

**The residual-weighted plan loss (THE UNIFIER — concrete + cheap):**
1. **Precompute once per collected chunk** in `collect()`, right after the conditional `_sample_chunk`:
   `a_base = pol._sample_chunk(frame,"",cfg,null=True)` (FROZEN base-DiT null action) plus a second null
   seed `a_base2` for the noise floor. Cache `s["base_null"]`, `s["base_null2"]`. Cost = +1–2 null
   forwards per *collected* chunk (collection is the cheap leg); **cached** → reused for all 120 BC steps.
2. In `build_batch` (rwbc_actor_adapt.py:124–125) **replace `actions_mask = ones`** with the per-dim mask
   `m = clip( (|a_real − a_base| − |a_base − a_base2|) / s_dim , 0.05 , 1.0 )`, shape (25,), broadcast
   over H → (H,25). Subtracting the null-vs-null spread strips sampler-stochasticity false positives (the
   `base_dit_perdim.py` noise floor); optional global per-dim `(1−AUC)` prior from `base_dit_perdim` over
   the demo set as an extra multiplier.
3. This mask flows **unchanged** into `raw_loss * mask` at nitrogen.py:690–692 — **zero model change**;
   `actions_mask` is already the per-(B,H,25) loss gate.
4. **Null-invariance untouched:** the mask only reweights the *conditional* flow-BC target; `plan_dropped`
   rows still hit `apply_null_mask` (nitrogen.py:669) → base DiT bit-identical under null.
5. **Why it unifies + kills the collapse:** the collapse mechanism (DISCRIMINATOR_RESULTS) is RWBC
   dragging the plan-head toward max-screen_x = sprint-into-pits. But "sprint right" is **frame-determined**
   → base DiT already emits it → `m→0` on those dims → **plan-head gets no gradient to sprint**. It is
   trained ONLY on frame-*underdetermined* obstacle-timing dims (SMW's +0.20 lever). Sonic is
   frame-determined everywhere → `m≈0` → plan-head starved → Sonic rides the actor (LoRA). The residual
   mask **is** the type-A fraction → no-knob SMW-vs-Sonic allocation, and it gates on the *frame*, not the
   noisy *reward*, so it survives the seed variance that dominates the reward.

**Data:** `--use-correct-plan` (BATTERY[game]['correct'], avoids wrong-direction plan noise) as the action
source; frozen-VLM text_only as the encoder; the mask **is** the residual span-filter (soft, no separate
pass).

**Seeds/eval/"works":** ≥5 paired seeds, reach_eval headline from fixed mined save-states, bootstrap CIs.
**WORKS =** generalist headline (SMW+Sonic equal-weight) of residual-adapt > both {no-plan base} and
{naive-adapt} with 95% bootstrap-CI lower bound > 0, SMW not collapsed (≥ fixed-plan-no-adapt within CI),
null output bit-identical.

## Q2 — Tame the seed variance (the real blocker)

SMW post-adapt Δ std ~0.4, sign flips per seed. Variance-reduction plan:
1. **Eval from FIXED mined save-states** (`ordinal_sanity.mine_starts`/`rollout`, snes_env
   `save_state`/`load_state`:189/193) — NOT `env.reset()` relaunch. Removes start-state randomness, the
   dominant variance source.
2. **Paired seeds + bootstrap CIs (Efron 1979):** same seed → identical start + sampler noise across
   conditions; report per-seed paired ΔΔ=(treatment−control), then mean ± 95% bootstrap CI. Pairing cancels
   shared-start variance.
3. **Scale:** 5 seeds × 8 starts × 2 sampler seeds ≈ 80 paired rollouts/condition.
4. **Lower LR 1e-4→3e-5 + `--anchor 0.01`** (L2-to-init trust region, KL proxy) shrinks post-adapt drift.
5. **Report the low-variance leg too:** no-adapt Δ_plan=R_B−R_A (std ~0.06, trustworthy) anchors the noisy
   post-adapt Δ. If post-adapt CI straddles 0 but Δ_plan>0 and residual-adapt ≥ Δ_plan, the inference-time
   plan benefit is preserved — that already counts as PASS (no collapse).

## Q3 — The minimal ablation (runnable tonight, ≥5 seeds)

4 conditions × {SMW, Sonic} × 5 paired seeds, reach_eval survival-weighted headline from fixed starts:
- **C0** no-plan base (`plan="" null=True`)
- **C1** fixed-correct-plan, NO adapt (inference-time plan only)
- **C2** plan + **naive** full-adapt (rtg, ones-mask) ← collapse control
- **C3** plan + **residual-weighted** full-adapt (THE recipe)

```
for s in 0 1 2 3 4; do
  for game in smw sonic; do
    rwbc_actor_adapt.py --env $game --use-correct-plan --reward-mode rtg --advantage-norm \
      --residual-mask --anchor 0.01 --lr 3e-5 --seed-offset $s --save-delta runs/r6/${game}_c3_s$s.pt
    # C2: same line WITHOUT --residual-mask  -> runs/r6/${game}_c2_s$s.pt
  done; done
reach_eval.py --games sonic smw --policies base c1plan c2:smw=...,sonic=... c3:smw=...,sonic=... \
  --n-starts 8 --seeds 2   # paired bootstrap over the 5 deltas/seed
```
(`--advantage-norm` and `--residual-mask` are the two additive flags the orchestrator wires.)

**PASS number:** C3 generalist headline **≥ 0.45** AND **C3 − C2 ≥ +0.08** with 95% bootstrap-CI lower
bound > 0 AND **C3_SMW ≥ C1_SMW − 0.03** (no collapse). Scale anchor: today rwbc Sonic=0.584, base=0.250,
so ≥0.45 generalist with a non-collapsed SMW leg (≥0.30) is the bar that means "the setup works."

## Q4 — FE / Minish: don't ask the owner; generate + validate

**Recommend (b): frozen-VLM re-abstraction + validate.** BATTERY already has `fireemblem`+`minish`
'correct' plans and demos exist for both; FE is turn-based (≈pure type-A) so a VLM objective should be
load-bearing. **Deciding check (one run, reuses the mask):** compute per-game `f_A` + the grounding check
on FE/Minish demos. If the VLM objective **shrinks the base-DiT residual toward the real action** AND
`Δ_plan=R_B−R_A>0` from a fixed start → keep it, **no owner plan needed**; ask the owner only IF grounding
fails. Not on tonight's critical path — defer until SMW/Sonic passes.

## Q5 — New data tonight: SKIP

The existing substrate (SMW8+Sonic4+FE2+Minish1 + emulator save-states) is sufficient AND superior here:
the recipe is on-policy save-state self-imitation — the emulator **is** the labeler (DAgger-without-oracle,
arXiv:1011.0686), generating its own data. YouTube needs the not-yet-built IDM plus a domain-bridge and
VLM-memorization risk; tonight's blocker is **seed variance + mask validation, not data scarcity**.
`mine_starts` already covers start diversity → **skip new pulls, spend the GPU on the 5-seed ablation.**
Pull YouTube only once the IDM exists (a multi-day item).

---
**Citations.** GRPO — arXiv:2402.03300. AWR — Peng 2019 arXiv:1910.00177. DAgger — Ross/Gordon/Bagnell
2011 arXiv:1011.0686. Bootstrap — Efron 1979 (Ann. Statist. 7(1):1–26). Helix fast/slow split —
figure.ai/news/helix. Repo: rwbc_actor_adapt.py (build_batch mask:124 / rw:236 / flags:181–192);
nitrogen.py (raw_loss*mask:690 / apply_null_mask:669 / plan_head:283); eval_policy.py `_sample_chunk`
null:230 / v_c,v_u:237–239; reach_eval (reach×surv); mine_starts:134; snes_env save/load:189/193.
