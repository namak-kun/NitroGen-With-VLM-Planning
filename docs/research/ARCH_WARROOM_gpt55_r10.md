# ARCH WARROOM — GPT-5.5 Round 10: short-path evocation vs long-path addition

## TL;DR
Root framing: **yes, but route by capability, not by game.** Short path = **evocation**: KL-anchored bridge/demo-fit turns VLM/narration knowledge into better selection among maneuvers the frozen DiT can already emit. Long path = **addition**: only for chunks where the base actor cannot emit the human maneuver even from the exact save-state.

**ONE experiment tonight:** wire the owner's gold narration into `demo_bc.py` and run **GoldNarr+KL vs Terse+KL** while simultaneously running the **maneuver router probe** on duck / spin-jump-Rex / grab-mesh save-states.

Pass bar: GoldNarr+KL beats Terse+KL by **+10 Δ_plan** or **+0.15 maneuver-evocation score** on >=2/3 seeds, keeps expressiveness **R>=0.80**, duck preserved, null action maxdiff **0.0**. If no new maneuver improves and Δ_plan is within CI, richer real plans are not the current bottleneck.

## Q1 — evocation/addition split + router
I validate the split. The router should be **base actor competence from the exact state**, then **explicit-plan evocation**:

Maneuvers:
1. **duck bullet**: known evocable; positive control.
2. **spin-jump-kills-Rex**: load a Rex save-state; unknown.
3. **grab/climb mesh**: state 7; owner's likely actor-OOD negative control.

Probe per maneuver from saved states, no need to reach them by rollout:
- `actor_fit`: compare base null/terse DiT chunk to human chunk on maneuver dims over first 18 steps. Use per-dim target-rate / AUC where possible: duck `JLY22>0.6` and dpad1; spin-jump = jump button + spin/action button + upward/forward timing; mesh = UP/JLY-up + grab/run/action hold + low horizontal drift at mesh.
- `evoke_gain`: run base and trained model with explicit plan vs neutral/null from same state. Metric = target-action rate or signed margin. Report `gain = rate(explicit)-rate(null)`, `ratio = rate(explicit)/(rate(null)+1e-3)`.

Classification:
- **EVOKE** if `actor_fit >=0.65` OR (`ratio>=2.0` AND explicit reaches `>=0.5*human_rate` and `>=5pp` absolute gain). Train only short path.
- **ADDITION REQUIRED** if `actor_fit <0.60` AND explicit-plan `ratio<2.0` OR explicit rate `<0.25*human_rate` in >=2/3 seeds/states.
- **AMBIGUOUS** otherwise: collect 2 more states, do not build long path yet.

Expected: duck=evoke; Rex likely evoke if base has spin-jump in manifold; mesh likely addition.

## Q2 — consolidation loop
`ClosedLoopPlanner(mode="learn")` is the right **episodic-memory substrate**, but not by itself a training objective. Minimal runnable consolidation is:

1. From a save-state, run K=8 attempts with `learn` mode; carry `<learnings>` across attempts.
2. Score attempts by maneuver/progress/survival event, not raw screen_x alone. For tonight's candidates: duck avoids hit, Rex enemy removed + survival, mesh y/up progress + attached/climb proxy.
3. Keep top 25-40% chunks and write training tuples: `(frame, action_chunk, plan_text = gold narration span + current <learnings>)`.
4. Distill with **KL-anchor short path** if router says evoke; use long-path module only for addition chunks.

This is **save-state RWBC/GRPO with a narration-conditioned actor**, not a separate critic/value-net method. The distinction from failed RWBC is the target/conditioning/guardrails: fixed save-state, explicit learning text, maneuver reward, KL-anchor, and no plan-head overwrite on broad plan distribution. Seed count: >=5 for any reward claim; tonight can only smoke-test 3 seeds for supervised narration.

## Q3 — two-path bandwidth / long carrier
Pick **dedicated plan-gated DiT-LoRA** as the true long path, but **do not run it tonight unless the router flags addition**.

Why not K=32: R9 refuted bandwidth for duck at K=8 and K=32 breaks `btn_s600` plan-head shape.

Why not FastPlanMod as the final carrier: zero-init `adaln_proj` is a good authority booster and null-safe, but it still modulates a frozen actor. If grab-mesh is genuinely absent from the action manifold, it is evocation-with-more-gain, not repertoire addition.

Long-path spec when needed:
- Add a **plan-token-gated LoRA** on DiT cross-attn/action routing so LoRA delta is zero for masked/null plan rows and, ideally, only active for plan-token attention positions. Current `lora.py` explicitly says ordinary LoRA does **not** preserve exact null-invariance; do not ship ungated LoRA as the long path under this contract.
- Keep `plan_head.*` trained with `--kl-anchor 0.3` frozen/reference-preserved; train long LoRA only on addition-labeled chunks (`grab_mesh`, maybe Rex if router says so).
- Validation: null bit-identity maxdiff 0.0, Terse+KL battery R not lower by >0.05, addition maneuver explicit-plan rate >=0.5*human_rate.

Tonight: classify first. If addition is real, tomorrow implement gated LoRA. FastPlanMod is fallback if gated LoRA is too invasive, not my first long-path carrier.

## Q4 — wire in gold narration
Run this as the main experiment.

New flags/code shape:
- Ingest all `demo_explanations.md` via `demo_narration.py ingest` so all 8 demos have `narration.json`.
- Add to `demo_bc.py`: `--plan-source {terse,narration,narration_sit}` and `--narration-tolerance-sec 1.5`.
- For each chunk, choose the narration span overlapping the chunk midpoint; if no span, fall back to `BATTERY[game][correct]`. For `narration_sit`, append action label only when label is rare: `" Also: duck / wait / retreat / jump now."`
- Always use `--kl-anchor 0.3 --kl-anchor-n 64 --duck-probe`.

Commands:
```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
$ENVP .venv/bin/python planner_poc/demo_narration.py ingest --file demo_explanations.md
for mode in terse narration narration_sit; do
  for s in 0 1 2; do
    $ENVP CUDA_VISIBLE_DEVICES=$s .venv/bin/python planner_poc/demo_bc.py \
      --game smw --train plan_head --steps 600 --lr 2e-5 --bs 4 \
      --plan-source $mode --kl-anchor 0.3 --kl-anchor-n 64 \
      --duck-probe --eval-starts 8 --seed-offset $s \
      --save-delta /home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files/r10_smw_${mode}_kl_s${s}.pt
  done
done
```
If only 4 GPUs/time: run `terse` and `narration` first; `narration_sit` is optional.

Metrics:
- Primary: Δ_plan on 8 SMW starts; bootstrap paired CI over starts/seeds.
- Preserve: `expressiveness_battery.py` R; pass `R>=0.80`, no contrast inverted, duck monotone.
- New maneuver probe: duck/Rex/mesh explicit-vs-null target-action rate from save-states.

Prediction: narration helps **Rex/duck** more than plain Δ_plan; Δ_plan may only modestly improve because terse already encodes right-runner objective. Falsifier for “richer real plans help”: narration Δ_plan within CI of terse **and** maneuver-evocation gain `<+5pp` for Rex/mesh **and** battery not better.

## Q5 — Super Metroid in pooled training-only
Add it only if plan conditioning is **objective/narration/VLM-goal**, never `advance right`.

I expect it to **help** the generalist if game-balanced and KL-anchored: exploration adds non-right-runner coverage (wait, backtrack, climb, door/room objectives), exactly the plan diversity the bridge needs. It will **hurt** if mislabeled with screen_x/advance or over-sampled.

Training-only recipe:
- `pooled_demofit.py --train-games smw,mmx,smbas,smetroid --game-balanced --kl-anchor 0.3`
- Super Metroid plan source priority: gold narration if available; else frozen-VLM objective per segment (`reach door`, `climb shaft`, `collect item`, `avoid enemy`); append action-derived SIT for low-level buttons.
- Cap per-game chunks equal; no eval claim until a RAM progress/objective metric exists.

## Steelman of Opus's likely pick
Opus will likely pick **zero-init FastPlanMod + KL-anchor**: R9 showed overwriting plan_head is destructive, so add a null-safe side module with extra authority and keep the short bridge pristine. That is a strong safety argument: code exists (`adaln_proj`), null rows are masked to zero, and it can be trained on gold narration without breaking `btn_s600` shapes.

My disagreement: FastPlanMod is the best **evocation amplifier**, not the cleanest **addition** mechanism. If the router shows mesh is base-actor-OOD, frozen-DiT modulation may still fail. Use FastPlanMod as fallback only after the router says the maneuver is evocable but underpowered.

## Single highest-value experiment tonight
**GoldNarr+KL plus maneuver-router probe**. It is cheapest, uses the owner's missing gold System-2 knowledge, preserves the R9 winner, and decides whether the long path is needed at all.

Decision table tomorrow morning:
- Narration improves Rex/mesh: short path is enough; re-pool with gold/VLM objectives + KL.
- Narration fails but router says actor can emit maneuver: add FastPlanMod authority.
- Narration fails and router says actor cannot emit maneuver: implement plan-gated DiT-LoRA for addition chunks.
