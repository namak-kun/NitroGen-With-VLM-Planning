# ARCH WARROOM — GPT-5.5 Round 9: preserve expressiveness while keeping demo-fit

## TL;DR
Root cause is primarily **data/conditioning collapse**: we trained every SMW chunk on one terse “advance” plan, so plan-head BC learned an almost plan-constant right+jump adapter and erased the base bridge’s latent DOWN affordance. The experiment to run tonight is **richer per-situation demo-fit**: same plan_head-only BC, but label each chunk with an action-derived plan template (`duck`, `jump`, `retreat`, `wait`, default advance) and oversample rare situational chunks. Pass: explicit duck command recovers to **DOWN-dpad >=5%** and explicit-vs-terse separation >=3pp while SMW Δ_plan retains **>=75% of pooled-demo-fit gain** (target: post change >=+20 screen-x, or within CI of pooled +26) over >=3 seeds, with null maxdiff 0.

## Q1 root cause
Rank:
1. **DATA / label support**: 0% DOWN after demo-fit is exactly what one-plan BC predicts. The training plan never names duck; rare DOWN frames get averaged into the dominant right+jump mode.
2. **OBJECTIVE**: plain MSE/flow BC on an imbalanced action distribution amplifies the data problem; it has no term preserving base plan-sensitivity on held-out commands.
3. **CAPACITY/ARCH**: least likely tonight. Base K=8 already implements `press down` -> 10.8% DOWN, so bandwidth is sufficient for at least this contrast before demo-fit.

Single discriminating measurement: **same architecture/objective, varied per-chunk plans**. If DOWN-on-command returns, it was data/conditioning. If it stays ~0 despite duck-labeled training examples and positive training fit, it is objective/arch destruction; then move to KL-anchor or zero-init add-on.

## Q2 the fix to test tonight
Pick: **(1) richer per-situation plans**, with rare-action balanced sampling. Do not build KL/FastPlanMod/K=32 first.

Minimal code shape in `planner_poc/demo_bc.py`:
- Add `--plan-label-mode {constant,action_templates}` default `constant`.
- Add `--duck-min-frames 3 --rare-action-frac 0.25`.
- In `load_demo_chunks`, after building the 18x25 mapped action chunk, choose `sample["plan"]`:
  - `duck`: if `(chunk[:,22] > 0.75).sum() >= 3`: `"Move right, press down to duck under hazards, then keep advancing."`
  - `retreat`: elif `(chunk[:,21] < 0.25).sum() >= 4`: `"Move left briefly to avoid danger, then recover and advance."`
  - `wait`: elif neutral/no-progress chunk: `"Stop and wait briefly; avoid the hazard before advancing."`
  - `jump`: elif `(chunk[:,18] > 0.5).sum() >= 3`: `"Run right and jump over gaps and enemies to advance."`
  - default: `BATTERY[game]["correct"]`.
- Oversample situational chunks (`duck|retreat|wait`) so each batch has ~25% if available; keep batch-balanced, not loss-weighted.
- Train only `plan_head.*`; LoRA/base/VLM frozen; masked null unchanged.

Runnable target after adding flags:
```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
for s in 0 1 2; do
  $ENVP CUDA_VISIBLE_DEVICES=$s .venv/bin/python planner_poc/demo_bc.py \
    --game smw --train plan_head --steps 600 --lr 2e-5 --bs 4 \
    --plan-label-mode action_templates --duck-min-frames 3 --rare-action-frac 0.25 \
    --eval-starts 8 --seed-offset $s \
    --save-delta /home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files/r9_smw_richplan_s${s}.pt
done
```

Pass criterion:
- SMW Δ_plan with default/correct advance plan: post gain **>= +20** over pre on mean, or >=75% of pooled +26, 2/3 seeds positive.
- Duck probe on same 4-6 SMW starts, 12 chunks: `"press down to duck under it"` gives **DOWN-dpad >=5%** and at least **+3pp over terse plan**.
- Terse/default plan does not overduck: DOWN-dpad <=3% unless the state itself requires it.
- Null-invariance: max absolute null-path diff vs base **0.00e+00**.

Falsifier: varied-plan training still yields explicit-duck DOWN <2% or explicit-vs-terse <1pp in 2/3 seeds, while Δ_plan advance remains high. That means the objective/overwriting mechanism, not the terse label alone, is destroying expressiveness.

## Q3 diagnostic-first?
Yes — run it first, and treat it as the first fix. The diagnostic is cheap and isolates data without introducing new architecture.

Per-chunk labels:
- Use mapped NitroGen actions, not text narration timing.
- `duck` if `JLY=DOWN` (`action[:,22] > 0.75`) for **>=3/18** sampled steps.
- `retreat` if `JLX=LEFT` (`action[:,21] < 0.25`) for **>=4/18** steps.
- `wait` if `abs(JLX-0.5)<0.15`, no jump/run, and low button activity for **>=8/18** steps.
- `jump` if `south/jump` (`action[:,18] > 0.5`) for **>=3/18** steps.
- Else default advance plan.
Priority order: duck > retreat > wait > jump > advance. This avoids “jump” swallowing duck+jump chunks.

## Q4 expressiveness battery
Score base vs each demo-fit on fixed starts; report action-rate contrast `effect = rate(command_plan)-rate(control_plan)`.

1. **Duck/DOWN (SMW)**: `press down to duck under it` vs terse advance; metric DOWN-dpad rate (`JLY>0.75`).
2. **Jump now (SMW/SMB1)**: `jump now over the pit/enemy` vs `keep running, do not jump yet`; metric first-6-step jump/south rate.
3. **Retreat left (SMW/MMX)**: `move left briefly to avoid danger` vs `run right`; metric `JLX<0.25` rate / mean left displacement.
4. **Wait/hold (SMW)**: `stop and wait briefly` vs `run right`; metric neutral-stick + no-jump/no-run rate and lower x-advance for first chunk.
5. **Shoot/attack (MMX)**: `shoot the enemy now` vs `do not shoot, keep moving`; metric attack button rate (use the MMX mapped fire dim from demos; if uncertain, report both Nitro dims 5 and 10).

Aggregate pass bar: no contrast inverted; at least **4/5 effects retain >=50% of base effect**, mean normalized retention >=0.6, and the headline advance Δ_plan remains >=75% of pooled-demo-fit. Null maxdiff must stay 0.

## Q5 scaling impact
This does **not** break the pooled-generalist story; it makes the pooled dataset higher entropy and less destructive. For scaling, start with action-derived templates from demos/emulator labels (free, exact buttons), then let frozen Qwen add nouns/objects from the frame (`duck under bullet`, `shoot enemy`, `climb ladder`) as optional text enrichment. Qwen-LoRA is useful later to standardize labels across games, but tonight does not need it. Light human narration is only needed to seed template vocabulary for genuinely new mechanics; the supervision signal is still buttons + frames, not per-game heads.

## Red-team my pick
Strongest objection: richer action-template labels may teach a low-level button-command language rather than true System-2 objectives, and duck data may be too rare even with oversampling. It could recover DOWN by overduking while silently weakening long-horizon advance/generalization.

Abandon this pick if, after 3 seeds with rare-action oversampling, explicit-duck DOWN remains <2% or the expressiveness battery retains <3/5 contrasts, **or** if SMW Δ_plan drops below +15 / more than 25% below pooled baseline. If training loss on duck chunks is low but plan-swap effect is absent, move to **KL-anchor** first (preserve base outputs on held-out plan distribution, λ tuned so advance gain stays >=75%); only then consider FastPlanMod/K=32.
