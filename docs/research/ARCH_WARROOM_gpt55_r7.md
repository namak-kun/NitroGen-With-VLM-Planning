# ARCH WARROOM R7 — GPT-5.5: discard RWBC-for-plan-OOD, pivot to supervised residual weighting

## Q1 — Residual-mask diagnosis

The residual-mask failure is real, not an unlucky implementation detail. The anchor-removal hypothesis is the best explanation: in the current `rwbc_actor_adapt.py` path, `build_batch()` either sends `actions_mask=ones` or `m=clip((|a_selected-a_base|-|a_base-a_base2|)/scale,floor,1)` into NitroGen, where `nitrogen.py` multiplies `raw_loss * actions_mask` before reduction. The all-ones mask was not just “noise”; it was BC pressure to keep the conditional policy close to the selected chunk on every frame-determined dimension. Because the selected chunk itself is usually close to frozen base-null on those dims, it acted as an implicit base anchor. The residual mask removed that anchor and left only high-residual dims to steer the plan-head/LoRA, so the unfrozen plan path could drift without being pinned by easy dimensions. The prediction was backwards.

Corrected loss, if we ever revisit actor adaptation for plan-OOD, is not “mask to residual”; it is **residual target plus complement anchor**:

`target = m * a_selected + (1-m) * stopgrad(a_base_null)`

`actions_mask = m + lambda_anchor * (1-m)` with `lambda_anchor in [0.25,1.0]`, not floor `0.05`.

That is a small edit in the existing `actions/actions_mask` construction: when `base_null` exists, blend the action target toward `base_null` on frame-determined dims and keep nontrivial mask mass there. But for tonight I would treat the R6 residual mask as dead as a collapse-preventer. The anchor fix is a control, not the pivot.

## Q2 — Pivot: supervised residual-weighted plan-head fit

Stop actor-adapting SMW-like plan-OOD games. The clean signal is frozen inference-time plan benefit (`Δ_plan≈+0.20`); reward-RWBC corrupts it. Run **supervised plan-head fitting to human demo actions**, with the residual used as a supervised weight, not as a self-imitation gate. This is BC on gold `(frame, plan, human action chunk)` from `docs/demos` / `docs/demo_plans`, i.e. `scripts/train_planner.py` style flow loss, but on human demos rather than reward-kept on-policy chunks. Freeze Qwen and base DiT/LoRA; train `plan_head.*` only if the harness supports it. If using the existing no-edit harness tonight, the closest runnable smoke test is:

```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
$ENVP CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/demo_train_stack.py \
  --games smw --ckpt ckpts/btn_s600_full.pt --steps 800 --lr 3e-5 --bs 4 \
  --eval-game smw --eval-chunks 16 --A 2 --cfg 8 \
  --save docs/rwbc_deltas/r7_smw_supervised_demo_stack.pt
```

Metric: evaluate saved delta against `oracle-plan/no-adapt` on SMW demo starts and `reach_eval` fixed starts; primary is preservation/improvement of inference-time plan reach, not native raw reward. The better follow-up is adding `--plan-head-only --residual-weight-demo` to this harness: compute `m` from human chunk vs frozen base-null exactly as R6 did, but use it as sample/dim weight on gold action, never on reward-selected self-imitation.

## Q3 — rtg+advnorm attribution

Both ingredients hurt. C2 proves `rtg+advnorm` without residual already collapsed SMW by `-0.489`; C3 proves residual made that worse by another `-0.149` paired mean. The clean runnable attribution is a 5-seed paired SMW grid with no residual: `local/no-advnorm` vs `rtg/no-advnorm` vs `local/advnorm` vs `rtg/advnorm`, same `--use-correct-plan --anchor 0.01 --lr 3e-5 --steps 120`. That isolates credit assignment from advantage sharpening. Sonic should keep advnorm only if its own paired control says so; AWR (Peng et al., 1910.00177) and GRPO-style group baselines (2402.03300) reduce variance in principle, but obstacle-game data says normalization can over-sharpen bad self-imitation.

## Q4 — One generalist target

One **RWBC recipe** is the wrong target. One **model** is still right. The unifier should be supervised: fit the shared frozen-VLM→plan-head interface to demo actions at high frame-counterfactual residual states, and let it be naturally near-no-op where residual is low. That matches BC’s strength and avoids covariate/reward hacking. Ross et al. DAgger (1011.0686) is the warning: offline BC can drift under its own state distribution, so evaluation must be on-policy/fixed-start, but the label itself must be expert/human, not the model’s reward-selected echo. Per-game training modes are acceptable: SMW uses supervised plan-head/no actor RWBC; Sonic can still use actor-OOD RWBC/LoRA after survival-weighted audit. Same weights/heads, different mode selection.

## Q5 — Tonight’s decisive experiment

Run the supervised SMW demo-stack above, 3 seeds if time (`CUDA seed`/`PYTHONHASHSEED` plus `np.random.seed` if added), and evaluate against `btn_s600` oracle-plan no-adapt. PASS: SMW post supervised delta is **not below oracle-plan/no-adapt by more than 0.03** and improves live/null by **≥ +0.15 reach_eval** on fixed starts. Strong PASS: beats oracle-plan/no-adapt by **≥ +0.05** paired without Sonic regression when loaded as the same model. This most advances “the setup works” because it tests the actual hypothesis: plan-OOD should improve through supervised plan grounding, not reward self-imitation.
