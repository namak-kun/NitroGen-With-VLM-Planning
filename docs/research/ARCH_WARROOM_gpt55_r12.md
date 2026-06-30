# ARCH WARROOM — GPT-5.5 Round 12

## TL;DR
Tonight should **not** start with Rex LoRA. Run the text-source experiment first: **R12-GoldPlan-KL-Pool**, i.e. generate gold-action-interleaved VLM abstract plans, attach them per chunk, then re-run **KL-anchored plan-head demo-fit** against the R9 terse+KL winner. This directly attacks the R11 result: staleness is **text-bound**, not token-bound.

**Pass bar:** vs R9 terse+KL, keep `Δ_plan >= 0.9x` and preferably `+10`, improve/retain expressiveness `R >= 0.85` (R9=0.83), duck-on-command monotone with `press_down >= 8% dpad or >=15% stick`, null action maxdiff `0.00e+00`, 3 seeds paired bootstrap CI not below terse by >5. Falsifier: rich VLM plans reduce `R < 0.80` or `Δ_plan < 0.8x terse`; then use gold-action VLM only as an offline planner-distillation target, not direct conditioning.

## Q1 — Gold-action VLM plans as training conditioning

**Position:** highest-value immediate move. R9 already proved the objective: **KL-anchor** fixes conditioning collapse (`Δ_plan +45.1`, battery `R=0.83`, null exact). R11 proved the remaining deployment failure is **plan text quality/stability**. Today's gold-action narration is exactly the missing labeler: it makes the VLM say *duck/spin/stomp* from actions, not human prose.

**Minimal code addition:** extend `load_demo_chunks()` / `pooled_demofit.py` with:
- `--plan-source battery|situational|vlm_json`
- `--vlm-plan-json-glob 'docs/demos/demos/*/*/vlm_gold_actions.json'`
- attach each chunk to nearest/covering VLM plan; fallback to `BATTERY[game]['correct']` if blank.

**Generate plans:**
```bash
PY=.venv/bin/python
$PY planner_poc/vlm_narrate_demo.py --demo <SMW_DEMO_DIR> --game smw \
  --interleave-actions --seg-seconds 2.0 --seg-frames 8 --every 1.0 \
  --qwen Qwen/Qwen3.5-2B --out <SMW_DEMO_DIR>/vlm_gold_actions_qwen2b.json
# repeat with Gemma-4-12B or Qwen3.5-9B only if 2B word-recall is weak.
```

**Training A/B/C, paired seeds 0/1/2:**
```bash
# A: R9 winner baseline
$PY planner_poc/demo_bc.py --game smw --train plan_head --use-correct-plan \
  --kl-anchor 0.3 --kl-anchor-n 32 --duck-probe --steps 600 --seed-offset $s \
  --save-delta files/r12_terse_kl_s${s}.pt

# B: new condition
$PY planner_poc/demo_bc.py --game smw --train plan_head --plan-source vlm_json \
  --vlm-plan-json-glob 'docs/demos/demos/*/*/vlm_gold_actions_*.json' \
  --kl-anchor 0.3 --kl-anchor-n 32 --duck-probe --steps 600 --seed-offset $s \
  --save-delta files/r12_goldvlm_kl_s${s}.pt

# C: optional ablation: gold plans without KL, to verify KL still matters
$PY planner_poc/demo_bc.py --game smw --train plan_head --plan-source vlm_json \
  --vlm-plan-json-glob 'docs/demos/demos/*/*/vlm_gold_actions_*.json' \
  --duck-probe --steps 600 --seed-offset $s \
  --save-delta files/r12_goldvlm_nokl_s${s}.pt
```

**Metrics:** `Δ_plan` from `demo_bc.py`, `planner_poc/expressiveness_battery.py --game smw --deltas ...`, duck probe per-channel, action-level null maxdiff. Also report plan-word recall on held-out chunks: `duck/spin/stomp/wait/retreat/up` from VLM json vs action-derived labels.

**Prediction:** gold-VLM+KL will not massively beat terse+KL on advance because KL already recovers the plan channel; expect `Δ_plan +40..60`. The gain should be in **semantic specificity**: duck/stick authority and fewer lost rare actions (`R 0.85..0.90`). Gold-VLM without KL may regress to taxonomy collapse; KL remains the guard.

**Why before Rex addition:** Rex LoRA is long-path capability addition and will need plan-gated LoRA plumbing + sprite RAM audit. Gold-action VLM plans are the cross-cutting text source for SMW, Metroid, Minish, and future ReST-EM. Since R11 says text quality is the lever, this is the root experiment.

## Q2 — Capability ADDITION: spin-jump-kills-Rex

Router result: jump button is evocable (`7.2x`), but **outcome** is not (`survive+advance 0.04 -> 0.07`). Treat Rex as ADDITION.

**Minimal runnable long-path:** add `--maneuver spinjump_rex --train lora --plan-token-gated-lora` to a Rex-specific trainer, seeded by `maneuver_router.py` reconstruction. Train rank-16 DiT-LoRA only on human Rex chunks, conditioned on VLM plan text: “spin-jump the Rex and keep moving right”. Keep KL-anchored plan-head frozen/loaded.

**Null exactness requirement:** ordinary `lora.py` notes it can break null invariance. The LoRA must be gated to plan-token K/V positions or multiplied by `plan_valid`; null path has no plan tokens, so delta=0. Verify action-level `max|chunk_null_after - chunk_null_before| == 0.00e+00`.

**Forgetting guard:** 50% replay non-Rex SMW chunks + pooled platformer chunks; action/logit L2 to frozen combined model on replay; optional Fisher/EATA weights later. Merge is expected clean because R8/R9 showed `plan_head.*` and `lora_*` stack.

**Eval:** from the 10 reconstructed Rex save-states, `k=16` stochastic rollouts, `chunks=4..6`, `A=2`, death-aware `survival_advance`. Primary pass now: `P(survive+advance>=25px) >= 0.25` and `Δ >= +0.15` over null/base, with life loss not increasing. Better follow-up: add sprite-status RAM so pass becomes `P(Rex sprite removed AND survived AND advanced) >= 0.20`. Guard: SMW/SMB/MMX `Δ_plan` retains `>=0.8x` R12-GoldPlan-KL, battery `R>=0.80`, null exact.

## Q3 — Plan-text quality/stability levers ranked

1. **Gold-action VLM plans as offline training conditioning** — do now. It directly improves the text distribution the bridge sees; no test-time gold actions needed.
2. **Replan-less / hold stable abstract plan** — cheap deploy default. R11: live replanning can hurt; fixed good text beats churn. Add `--replan-every 999` / “hold until gate fires” baseline to `staleness_probe.py` and record flip-rate + live-null advance.
3. **Bigger planner as evaluator/deployment A/B** — test Qwen3.5-9B/27B or Gemma-4 on fixed save-states, but not first if 2B+gold labels already train the bridge. Metric: frozen actor `survival_advance(live_plan)-null`, plan flip rate, maneuver-word recall against action-derived labels; no oracle text required.
4. **Qwen-LoRA distill to 2B** — after collecting gold-action VLM plans. Train 2B LoRA to emit the same abstract plan from frames/control schema/history; deploy cheap. Pass if it recovers `>=90%` of big-planner live-null advance and cuts flip-rate by `>=30%`.
5. **Gold-action prompting at inference** — not available causally. Do not feed predicted actions as fake gold unless explicitly framed as retrospective re-narration for the next high-level segment and validated against null.

**Plan-quality without oracle:** freeze actor+bridge, evaluate plan text by outcome: `survival_advance(plan)-survival_advance(null)` across fixed save-states; plus text stability (semantic edit/embedding flip), maneuver recall from held-out demos where action labels are known, and contradiction rate (`r_t` high + low progress).

## Q4 — Exploration demos: Metroid + Minish

Fold them into the **pooled plan-OOD generalist as positive diversity**, but do **not** evaluate them with screen_x. They add missing plan verbs: wait, backtrack, aim/up/down, morph/crouch, talk, read, pick/throw, room transition. That is exactly what R9’s battery said platformer-only terse data under-covers.

**Auto-narrating unannotated Ridley:** yes, this is the point of gold-action narration. The human omitted boss/escape because prose was hard; action labels still exist. Add control schemas `smetroid` and `minish`, run:
```bash
$PY planner_poc/vlm_narrate_demo.py --demo docs/demos/demos/SuperMetroid-Snes/20260629-132934 \
  --game smetroid --interleave-actions --every 1.0 --seg-seconds 2.0 \
  --out .../vlm_gold_actions_smetroid.json
$PY planner_poc/vlm_narrate_demo.py --demo docs/demos/demos/LegendOfZeldaTheMinishCap-GbAdvance/20260627-113636 \
  --game minish --interleave-actions --every 1.0 --seg-seconds 2.0 \
  --out .../vlm_gold_actions_minish.json
```
Pass for the auto-narrator: boss/escape plan-word recall includes action-implied maneuvers (`aim`, `shoot`, `jump`, `morph`, `dodge`, `escape/climb`) at `>=60%` of action-derived maneuver windows and does not devolve into enemy-noun hallucination that changes the maneuver. Enemy nouns can be wrong; maneuver must be right.

**Eval for Minish/top-down:** use a demo-progress coordinate, not x. First choice: RAM/event progress: map id/room transition, dialogue flag advanced, item/rupee/heart/object state, grass-pick/throw count. Immediate generic fallback: nearest-neighbor progress along the human trajectory in embedding/RAM/position space; report `demo_index_advance` before death/menu reset. `survival_advance` should become `survival_progress(metric='demo_index|room|dialogue')`.

## Q5 — coherent next phase + tonight

**Priority order:**
1. Text source: gold-action VLM abstract plans + KL-anchored plan-head demo-fit.
2. Re-pool generalist over SMW/SMB/MMX plus Metroid/Minish chunks once eval coordinates exist.
3. Stable deployment: hold good plan; replan only on gate/fire or room/objective transition.
4. Long-path ADD: plan-token-gated Rex LoRA, then other router-confirmed additions.
5. Qwen-LoRA distill 2B planner from gold-action/big-planner outputs if deployment planner quality is the bottleneck.

**Tonight’s single experiment:** `R12-GoldPlan-KL-Pool` on SMW first, 3 seeds, A/B against R9 terse+KL; optional C no-KL. It is one experiment because it answers whether the new breakthrough (gold-action VLM plans) should replace terse plans as training conditioning.

**Pass/fail:**
- `Δ_plan_goldKL >= Δ_plan_terseKL - 5` and preferably `>= +10` over terse; paired bootstrap lower bound > `-5`.
- `R_goldKL >= 0.85` and no contrast < `0.5` except known noisy duck channel.
- Duck-on-command monotone; `press_down >= 8% dpad` or `>=15% stick`.
- Live-plan staleness probe on the resulting delta: `live-null >= terseKL live-null -5` and oracle-live gap not larger.
- Null action maxdiff `0.00e+00`.

**Falsifier:** gold plans are too noisy/granular: `R<0.80` or `Δ_plan<0.8x terse+KL` or non-monotone duck. Then keep KL terse for bridge training and use gold-action VLM outputs only for (a) Qwen-LoRA planner distillation and (b) routing/ADD data mining.

## Steelman of Opus’s likely pick

Opus will likely pick **Rex plan-token-gated DiT-LoRA** because the router produced the first clean ADDITION result: evocable primitive, missing outcome. That is strong and directly advances “System 2 becomes System 1.” I still sequence gold-action plans first: Rex LoRA needs new null-safe LoRA plumbing and a better kill signal, while gold-action conditioning is already validated by today’s narration breakthrough and attacks the R11 text-bound bottleneck across every future game.
