# ARCH WARROOM R6 — GPT-5.5 converged recipe

## Q1 — THE recipe: residual-gated return-weighted BC, one generalist

Run one frozen-VLM-first generalist from `btn_s600`: keep Qwen frozen, preserve `null_mode="masked"`, train **DiT LoRA + `plan_head.*`** together, but with a residual gate so plan parameters receive gradient only where the frame-only actor cannot explain the action. Do **not** use global `--lora-only`; Sonic needs LoRA, SMW needs the plan path, and the same model should decide by state weight.

Minimal harness additions to `planner_poc/rwbc_actor_adapt.py`:

1. Add flags: `--residual-plan-loss`, `--adv-norm game`, `--eval-metric reach`, `--reach-out <json>`, `--residual-cache <jsonl optional>`, `--residual-tau 0.15`, `--residual-floor 0.05`, `--residual-ceil 1.5`.
2. In `collect()`, for each kept sample compute a cheap residual once, under `torch.no_grad()`: `a_base = pol._sample_chunk(frame, "", cfg, plan_frames=[frame], null=True, noise_seed=fixed)`; `rho = weighted_action_dist(sample["action"], a_base)`. Use all 25 dims, not the weak direction-only test: buttons weighted by `(1 - AUC_dim)`/separation calibration from `base_dit_perdim.py`; sticks by normalized MSE/noise; jump/buttons get explicit weight because buttons had good AUC. Clip and store `sample["residual_w"] = clip((rho - tau)/(ceil - tau), floor, 1)`.
3. In `build_batch()`, add `batch["residual_w"]` shaped `(B,1,1)` or `(B,H,25)` if per-dim is implemented. In the loss path, first runnable version may multiply the scalar model loss by `mean(reward_w * residual_w)` only for full-adapt runs. Better small edit: have NitroGen return per-token/action MSE (`reduction=none`) and apply `actions_mask * residual_w` before reduction. Anchor still applies to all trainables.
4. Use return-based ranking: `--reward-mode rtg --gamma 0.97`, but standardize advantages **within game/run** before top-k (`A=(G-mean)/std`, keep `--top-frac 0.35`). This is AWR-style return-weighted regression (Peng et al., arXiv:1910.00177) without unstable log-probs; the per-game/group normalization is the same variance-control idea as GRPO (Shao et al./DeepSeekMath, arXiv:2402.03300).

Exact first run command template:

```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
$ENVP .venv/bin/python -u planner_poc/rwbc_actor_adapt.py \
  --ckpt ckpts/btn_s600_full.pt --env smw --use-correct-plan \
  --collect-eps 16 --eval-eps 24 --chunks 14 --A 2 --cfg 8 \
  --reward-mode rtg --gamma 0.97 --top-frac 0.35 \
  --steps 120 --bs 4 --lr 3e-5 --anchor 1e-6 \
  --residual-plan-loss --adv-norm game --seed-offset SEED \
  --save-delta docs/rwbc_deltas/r6_smw_seedSEED.pt
```

Then run the same for Sonic. For tonight, use `--use-correct-plan` as the clean plan-channel test. VLM-generated/replanned plans are **phase 2**, after this passes: `encode_multimodal(text_only=True)` is the right input, but letting generation noise enter tonight will confound the recipe. Existing demo spans can seed residual caches; no new labels required. “Works” means: equal-weight SMW+Sonic reach_eval improves over base and oracle/no-adapt, while Sonic does not regress.

## Q2 — tame seed variance: measure paired, fixed-start, reach-weighted deltas

The blocker is not theory; it is SMW post-adapt std ≈0.4. Treat every comparison as **paired**: same collect seed offset, same eval start states, same horizons, same policy noise seeds. Save deltas, then evaluate all deltas in one `reach_eval.py` invocation so denominator mining and start states are identical. Use `--eval-eps 24` during RWBC only as a smoke signal; headline comes from `reach_eval.py --n-starts 8 --seeds 4 --horizons 450 900`, which already uses survival-weighted `reach × survived` and scripted-RIGHT normalization. Report paired per-start deltas and percentile/bootstrap CIs (Efron 1979 bootstrap; resample game/start/horizon/seed tuples 10k times). Lower LR (`3e-5`) plus `--anchor 1e-6` is the anti-collapse default; if gradients are tiny, raise steps to 200 before raising LR. Also report the low-variance no-adapt `Δ_plan = correct-plan - null/live` as a sanity covariate; if `Δ_plan` is positive but post-adapt loses, blame adaptation, not plan-conditioning.

## Q3 — minimal ablation, >=5 seeds, PASS number

Run 5 seeds first: `SEED={0,1000,2000,3000,4000}` for both `smw` and `sonic`.

Ablation cells:

1. `base/null`: reach_eval policy `base` with empty/null plan.
2. `oracle-plan/no-adapt`: `base` with correct fixed plan, no training.
3. `naive-full`: current full RWBC, `--reward-mode rtg`, no residual gate.
4. `r6-residual-full`: recipe above.
5. `lora-only-control`: recipe reward settings with `--lora-only`.

Evaluate saved deltas together:

```bash
$ENVP .venv/bin/python -u planner_poc/reach_eval.py \
  --games smw sonic --ckpt ckpts/btn_s600_full.pt --cfg 8 \
  --policies base oracle r6_s0:docs/rwbc_deltas/r6_smw_seed0.pt ... \
  --horizons 450 900 --n-starts 8 --seeds 4 \
  --out docs/graded/r6_reach_eval.json
```

If `reach_eval.py` cannot express oracle-plan/no-adapt directly, add the tiny policy alias: `oracle:<game>=<correct-plan>` and pass that plan into `_sample_chunk`. **PASS R6 number:** residual-full beats base by **+0.10 generalist headline** with bootstrap 95% CI lower bound > +0.02, SMW improves over oracle/no-adapt by ≥ +0.05 median paired delta, and Sonic is not worse than base by more than −0.03. Also require residual-full > naive-full by ≥ +0.05 on SMW, proving the mask matters.

## Q4 — FE / Minish

Do **not** wait for owner-written plan-level plans tonight. Generate objectives from frozen Qwen (`text_only=True` / `generate_plan`) and validate them by residual/reward. Cheap deciding check: for each FE/Minish demo, sample 32 narrated/demo states, compute full-action residual fraction and `Δ_plan` between Qwen objective, owner-style generic objective, and null. If Qwen plan gives positive low-variance `Δ_plan` or high residual alignment, include as eval-only; otherwise skip training. Ask the owner later only for audit labels on failures. This follows DAgger’s lesson (Ross et al., arXiv:1011.0686): offline labels are proposals; on-policy/environment validation decides.

## Q5 — new data tonight

Skip YouTube/IDM tonight. The recipe’s uncertainty is seed variance and loss routing, not data volume. YouTube adds compression/domain and pseudo-action confounds before the residual gate is proven. Smallest useful new data, if GPU idle after ablation: emulator save-state starts for SMW only, 8 fixed mid-level starts around pits/enemies, to reduce eval variance and improve reach_eval coverage. Otherwise existing SMW/Sonic/FE/Minish demos + emulator starts are enough. This keeps the architecture aligned with Helix/pi0.5: frozen slow VLM objective, fast visuomotor actor, residual-gated coupling, one recoverable generalist.
