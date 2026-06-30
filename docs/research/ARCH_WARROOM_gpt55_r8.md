# ARCH WARROOM — GPT-5.5 Round 8: generalization + scaling call

## Bottom line

Yes: my prior is that the R7 recipe generalizes from SMW to MMX and SMB1, because the recipe is not reward-shaping and not game-specific. It is supervised imitation of human action chunks through the plan-conditioning pathway only: `demo_bc.py` loads exact demo `(frame, action)` chunks, conditions on `BATTERY[game]["correct"]`, and can train only `plan_head.*`; eval is `Δ_plan = plan_advance - null_advance` from fixed demo start states. R7 showed the exact signature we wanted: SMW `Δ_plan` widened from +35.7 to +56.0, 4/5 seeds positive, bootstrap CI [+2.0,+38.7], with exact null-invariance after loading the delta. That is the correct primitive for plan-OOD games. It is also consistent with DAgger's diagnosis of closed-loop BC failure (Ross et al., arXiv:1011.0686): do not use self-generated reward-ranked data where the reward is noisy; use trusted demonstrations for the planning channel.

## Q1 — MMX + SMB1 prediction and falsifier

Prediction: MMX should widen `Δ_plan`; SMB1 should train similarly, but its eval is blocked until progress is reliable. Both are right-running obstacle platformers: the plan is not just “go right,” it disambiguates jump/shoot/avoid timing where a single Markov frame is underdetermined. MMX is the stronger immediate test because `new_demo_envs.make_mmx()` exposes `xpos/health/lives`; SMB1 has much more demo data but no trusted world-x yet.

Falsification threshold: call the recipe **not generalized** if MMX plan-head-only demo-fit, with correct plan, gives mean `Δ_plan_post - Δ_plan_pre <= 0` across 3 seeds **or** fewer than 2/3 seeds widen by at least +10 screen-x units. If the 3-seed mean is positive but weak, expand to 5 seeds and require bootstrap 95% CI lower bound > 0, matching the SMW standard (Efron bootstrap, 1979). If MMX passes but SMB1 cannot be evaluated, do not penalize the recipe; mark SMB1 as train-only until x-progress is solved.

## Q2 — SMB1 eval call

Tonight: do **(a) 2-byte page+offset monotone scan** first. SMB1 world progress is almost certainly `256*page + x_in_page`, not a single byte. Search candidate byte pairs against demo time using monotonicity, high correlation with frame index during rightward segments, reset/drop behavior on death/level transition, and sanity against known page wrapping. This is the only option that preserves the same low-variance fixed-state `Δ_plan` metric used for SMW/MMX.

Fallback: **(d) use SMB1 for training only; eval on MMX + SMW.** Do not use score/coins/time as the primary metric: they are sparse, confounded, and can reward standing still or waiting. Do not make optical-flow the primary pass/fail tonight: it is useful as a video sanity check, but it changes the metric from game-state progress to visual similarity and will punish legitimate alternate trajectories.

## Q3 — multi-game plan-head scaling

Same-mode deltas can conflict. The capstone merge was easy because SMW wrote `plan_head.*` and Sonic wrote `lora_*`; SMW+MMX+SMB1 all write the same plan-head parameters. Averaging deltas is a fragile linear-mode-connectivity bet; sequential fitting is order-dependent and will overfit the last game. The right test is **one pooled supervised demo-fit** over all plan-OOD demos, with per-game balanced sampling.

Test tonight: train one plan-head delta on pooled trimmed SMW+MMX+SMB1 chunks, all with each game's correct plan string. Balance batches by game, not by raw chunk count, because SMB1 has ~7700 chunks and would otherwise dominate the two MMX demos. Keep LoRA frozen, base frozen, VLM frozen, null path masked. This is the real generalist objective: one shared plan encoder learns “rightward obstacle-platformer intent” across games. If conflict appears, the next tool is multi-task gradient surgery/weighting (e.g. PCGrad-style conflict handling or GradNorm), not per-game weights; but do not build that before the pooled baseline.

Runnable shape:

```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
# Implement as additive --game planood_pool or --games smw,mmx,smbas with balanced sampling.
$ENVP CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/demo_bc.py \
  --game planood_pool --train plan_head --use-correct-plan \
  --steps 600 --lr 2e-5 --bs 4 \
  --save-delta /home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files/r8_pool_planood_s0.pt
```

## Q4 — 3D prep

Transfers: the training recipe is reward-free and game-agnostic: frozen VLM, correct text objective, human action chunks, plan-head-only supervised fit, null-invariance. That should apply to Mario64/OoT-style demos.

Breaks: `screen_x` is invalid; “move right” is not a universal correct plan; camera-relative controls make action labels context-dependent; a Markov frame is weaker in 3D because intent may depend on route, camera, and room memory. Minimal adaptation: replace progress eval with demo-continuation metrics available from emulator state (position-to-waypoint, room/door/checkpoint, health/survival), and replace the correct plan with waypoint/objective language (“reach the door,” “collect the star,” “cross the bridge”). Do not use GRPO/AWR (arXiv:2402.03300; arXiv:1910.00177) before the supervised floor is measured.

## Q5 — highest-value experiment tonight

Run the **single pooled plan-OOD demo-fit** beyond the already-running MMX seeds.

Metric: evaluate one pooled delta on SMW and MMX fixed demo starts, reporting `Δ_plan_pre`, `Δ_plan_post`, `change`, null drift, and per-state post plan advance. Pass number: pooled delta passes if it improves mean `Δ_plan` by **>= +10 screen-x on both SMW and MMX**, with null path drift `|null_post-null_pre| <= 10`, and does not underperform the best per-game delta by more than 25% on either game. Strong pass: 3 seeds, at least 2/3 positive on both games. That answers the true scaling question: one model, one plan-head, multiple plan-OOD games.
