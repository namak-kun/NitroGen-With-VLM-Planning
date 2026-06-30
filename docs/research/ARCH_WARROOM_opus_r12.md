# ARCH WARROOM R12 — Opus-4.8: the action IS the label — distill gold-action VLM plans, fix the duck-collapse FIRST, defer the Rex add

*This round I OWN an R11 falsification: my forward-pass token-TTA was the wrong staleness lever (R11 fired my own falsifier — staleness is TEXT-bound, tok-drift~0). R12 pivots to the thing R11 redirected us to: PLAN-TEXT QUALITY. Gold-action VLM narration is exactly that source, and it doubles as the fix for the R9 duck-command collapse. One experiment buys both.*

## TL;DR — the next-phase plan + tonight's ONE experiment + pass bar

**Next-phase order (Q5):** (1) **TONIGHT — gold-action VLM plans as KL-anchored demo-fit conditioning** (replace the terse BATTERY plan with per-chunk gold-action-grounded VLM plans; vs the R9 terse+KL winner). (2) **Qwen-LoRA distill** those gold-action plans into the 2B *deployment* planner (we have NO gold actions at test time — distillation is how the live 2B emits gold-action-quality plans from frames alone; this is the R11 plan-text-quality lever). (3) **Rex DiT-LoRA addition** (plan-token-gated, sprite-status RAM, EATA/replay guard) — the long path, gated on building the kill-detector. (4) cheap deployment default: **`--replan-every N` + hysteresis** (R11: live re-planning HURTS, MMX −11). (5) **fold exploration demos** into the pooled expressiveness training; **debug + re-run the Ridley auto-narration** (the existing `vlm_narration_ridley.json` is ALL-EMPTY — the hypothesis is currently UNVALIDATED).

**Tonight's single experiment — GOLD-ACTION-PLAN ⊗ KL crossed demo-fit on SMW (then MMX replication of the winner).**
3 arms × 3 seeds, plan-head-only, frozen LoRA+base (null-exact by construction):
- **A = terse+KL** (R9 winner control): `BATTERY['smw']['correct']` + `--kl-anchor 0.3`. Reproduce R9 (Δ_plan +45.1, battery R=0.83, duck-dpad 6.7%).
- **B = gold+KL** (treatment): per-chunk gold-action VLM plans + `--kl-anchor 0.3`.
- **C = gold−KL** (ablation): per-chunk gold-action plans, no anchor — isolates whether per-chunk specificity ALONE prevents collapse (like R9 situational P1A0, R=0.53) or whether KL still does the breadth work.

**Pass bar — B passes iff ALL hold (3/3 seeds, bootstrap 95% CI):** Δ_plan(B) ≥ **+40** (CI>0; must not cost advance vs A's +45.1) **AND** duck-on-command on the TRAINED STICK channel (`duck_probe` dim22>0.6) escalates monotone with plan-explicitness and reaches **≥30%** press_down at the explicit duck plan (A's terse+KL is ~0% on stick — it only preserves the base-native dpad) **AND** expressiveness battery **R(B) ≥ 0.80** (matches/beats A's 0.83; must NOT regress to situational's 0.53 that dropped retreat+up) **AND** null action-bit-identity **= 0.0**.
**Named falsifier:** R(B) < 0.80 OR Δ_plan(B) < +40 ⇒ open-vocab gold narration re-imports the situational taxonomy-treadmill loss (owner's "too granular, mixes S1/S2") ⇒ DEMOTE gold-action to a duck-authority STACK on top of terse+KL (R9's "optionally stack situational"), keep terse+KL as the deployment base. Secondary: if C also passes battery breadth ⇒ KL-anchor wasn't load-bearing for breadth ⇒ cheaper to keep terse+KL.

**Why this over the Rex add:** it is a GENERALIST fix on the critical path (the duck-collapse, battery R=0.165 for plain BC, poisons EVERY multi-game demo-fit deployment), it directly serves R11's text-quality redirect, and it reuses validated infra (R9 KL-anchor + `demo_bc.py`) at ZERO new tooling. The Rex add is one maneuver, one game, near-chance outcome rates (3.75→7.5%, rel +0.04), and needs an unbuilt sprite-status detector — that's multi-night, not tonight.

---

## Q1 — Gold-action VLM plans as TRAINING conditioning (the immediate high-value move)

**Claim:** gold-action narration is the *per-chunk specificity of situational labels WITHOUT the fixed-taxonomy ceiling*. R9 gave us the exact prior to beat:

| R9 cell (SMW, plan-head, 600 steps, 3 seeds) | Δ_plan | duck-on-command | battery R (retreat/jump/wait/up) |
|---|---|---|---|
| P0A0 terse, plain BC (collapse) | −4.9 | 0% both channels | **0.165** (kills ALL — wait FLIPPED) |
| P1A0 **situational** (4-cat action labels) | +32.8 | STICK press_down → **98%** (trained) | **0.53** (keeps jump+wait, LOSES retreat+up) |
| P0A1 **terse+KL** (winner) | **+45.1** | DPAD 0.2→3.0→**6.7%** (base-native preserved) | **0.83** (PASS — preserves retreat+up+wait) |

Read the dissociation precisely (R9 KEY DIM NOTE, DISCRIMINATOR:532,560): **situational TRAINS the stick channel** (human DOWN→stick dim22 in `map_action`) → stick-duck recovers to 98%, but the 4-category taxonomy under-labels retreat/up → breadth collapses to R=0.53. **terse+KL preserves the base-native dpad duck** (6.7%) and the whole space (R=0.83) but never trains the stick-duck. These are DIFFERENT channels — they are not directly comparable, and neither arm does both.

**Prediction for B (gold+KL):** gold-action plans NAME the duck *per chunk, grounded in the chunk's own gold DOWN→stick action* (today's result: Gemma-4-12B SMW maneuver-word counts went frames-only {duck 0, spin 0, stomp 0} → +schema {0,2,12} → +schema+GOLD **{8,5,17}**; it produced "Duck to slide under the moving Bullet Bill" — the duck it never named from frames). So B should **train the stick-duck like situational (≥30%, ideally ≫)** AND, because the vocabulary is OPEN (covers retreat/up/wait, not just 4 buckets) and is held broad by the KL-anchor, **preserve battery R≥0.80 like terse+KL**. That is the 2-for-1 *inside Q1*: the win is getting situational's trained-stick duck without paying situational's breadth tax.

**Exact run (orchestrator codes the one small wiring piece — a `--vlm-plans` per-chunk plan source in `load_demo_chunks`):**
```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
PY=.venv/bin/python ; F=/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files

# (0) LABEL PASS — frozen VLM, one-time, ~minutes. The 4 existing SMW vlm_narration.json are STALE
#     (interleave:None, schema:None, empty) — regenerate ALL 8 with gold-action interleave + schema.
#     Use the BIG labeler (Gemma-4-12B or Qwen3.5-9B) here (offline teacher; deployment planner stays 2B).
for d in docs/demos/demos/SuperMarioWorld-Snes/*/; do
  $ENVP CUDA_VISIBLE_DEVICES=0 $PY planner_poc/vlm_narrate_demo.py --demo "$d" \
     --interleave-actions --game smw --seg-seconds 2.0 --seg-frames 8 \
     --qwen google/gemma-4-12b --out "$d/vlm_narration.json" ; done   # enable_thinking=False is default

# (1) CROSSED DEMO-FIT — 3 arms x 3 seeds (plan-head-only -> null-exact). New flag: --vlm-plans maps each
#     chunk start-frame i to the covering narration entry [frame,frame_end); fallback = BATTERY['correct'].
for s in 0 1 2; do g=$((s%4))
 $ENVP CUDA_VISIBLE_DEVICES=$g $PY planner_poc/demo_bc.py --game smw --train plan_head --seed-offset $s \
    --use-correct-plan --kl-anchor 0.3 --duck-probe --save-delta $F/r12_A_terse_kl_s$s.pt &        # A
 $ENVP CUDA_VISIBLE_DEVICES=$g $PY planner_poc/demo_bc.py --game smw --train plan_head --seed-offset $s \
    --vlm-plans 'docs/demos/demos/SuperMarioWorld-Snes/*/vlm_narration.json' --kl-anchor 0.3 \
    --duck-probe --save-delta $F/r12_B_gold_kl_s$s.pt &                                             # B
 $ENVP CUDA_VISIBLE_DEVICES=$g $PY planner_poc/demo_bc.py --game smw --train plan_head --seed-offset $s \
    --vlm-plans 'docs/demos/demos/SuperMarioWorld-Snes/*/vlm_narration.json' --kl-anchor 0.0 \
    --duck-probe --save-delta $F/r12_C_gold_nokl_s$s.pt ; done                                      # C

# (2) BREADTH GUARD — the decisive metric, all arms:
$ENVP $PY planner_poc/expressiveness_battery.py --game smw --tag r12 \
   --deltas $F/r12_A_terse_kl_s*.pt $F/r12_B_gold_kl_s*.pt $F/r12_C_gold_nokl_s*.pt
# (3) Winner only: cross-game replication on MMX (3 seeds) to claim generalist (R8 pattern).
```
**Metrics:** Δ_plan (survival-aware advance, plan−null, 8 fixed SMW starts via `eval_common.survival_advance`); `duck_probe` per-channel (STICK dim22>0.6 trained; DPAD dim1>0.5 base-native) across `DUCK_PLANS` (terse / "duck to dodge" / "press down to duck"); `expressiveness_battery` R + sign-match; null bit-identity.

**Is this higher value than the Rex add? YES, decisively, for tonight** — it's a generalist regression-fix on validated infra serving two open problems (duck-collapse + R11 text-quality), vs a one-maneuver point-gain at near-chance SNR needing unbuilt tooling. Q1 also produces the *training data* Q3 needs (the gold-action plans) and the *label source* Q2 needs (the "spin-jump the Rex" conditioning) — Q1 is upstream of both. Run Q1 first; it de-risks everything downstream.

---

## Q2 — Capability ADDITION (Rex): the long path, concretized but DEFERRED to post-Q1

R11 router verdict (DISCRIMINATOR:606-622): the JUMP primitive is **evocable 7.2×** by an explicit plan, but the OUTCOME (survive+advance past the Rex) is flat **3.75%→7.5%** (both near chance) ⇒ ADDITION at the outcome level, route to DiT-LoRA on the human's Rex-kill chunks. I CONFIRM the routing and the mechanism, with four sharpenings:

1. **Eval = emulator outcome, not button-rate.** Survive+advance ≥25px past the Rex with no life lost, from the router's reconstructed save-states. SMW `score` does NOT register kills (verified, DISCRIMINATOR:608) → build a **sprite-status RAM kill-detector** (the SMW sprite-status table; the missing tooling) to separate kill-vs-dodge. Until that exists, use survive+advance as the proxy. **This is the gating blocker — it's the reason Q2 is not tonight.**
2. **Null-exactness ⇒ plan-token-GATED LoRA, not ordinary LoRA.** `lora.py:8-12` is explicit: ordinary cross-attn LoRA perturbs *image-token* attention → null only approximately preserved. Gate the LoRA to plan-token KEY positions only → null path has no plan tokens → exact-zero contribution → null-invariance restored by construction. Verify with the action-level bit-identity check (must = 0.0), same as the plan-head deltas (DISCRIMINATOR:303-306).
3. **Forgetting guard = EATA Fisher (arXiv:2204.02610)** on ~100 frames of baseline play + our *demonstrated* structural guards (add-chunks-only, disjoint-merge). DAgger (arXiv:1011.0686, O(εT²)→O(εT)) says collect corrections at *policy-visited* states (router save-state replay), not human-visited — this is the EXP-011 "left-then-right" covariate-shift fix.
4. **Disjoint-merge is already proven** (R8/R9 capstone, DISCRIMINATOR:324-343,384-397): plan-head delta (plan_head.*) + LoRA (model…lora) write mostly-disjoint params and STACK with no interference, even positive cross-transfer. So a Rex-LoRA merges cleanly onto the Q1 plan-head generalist.

**Minimal runnable (post-detector):** DiT-LoRA-gated, conditioned on the **gold-action plan** "spin-jump the Rex" (Q1's label source — Q1⊗Q2 share the plan source), ≥5 seeds (any outcome/reward claim, R9 discipline), metric = ΔP(handled-rex|plan)−P(|null) from reconstructed states, pass = Δ ≥ +0.15 (3× the current +0.04 relative) with battery R retained ≥0.80 (forgetting guard). **Caveat I flag loudly:** the router's absolute rates are LOW for both arms on a hard 4-chunk reconstructed scenario (router's own caveat, DISCRIMINATOR:619) — improve save-state reconstruction (seed nearer the Rex) BEFORE trusting any LoRA delta, or the SNR will void the result like the single-seed RTG artifact did (DISCRIMINATOR:150-172).

---

## Q3 — Plan-text quality/stability levers, RANKED (the R11 staleness redirect)

R11 nailed the lever: staleness is TEXT-bound (tok-drift~0; fresh≈cached≈null≪oracle; oracle−fresh +21 SMW; live re-planning is the WORST non-null mode on MMX, −11; DISCRIMINATOR:566-604). My R11 forward-pass token-TTA is REFUTED — there is no token-staleness to fix. Ranked levers:

1. **(b) Qwen-LoRA distill of gold-action plans into the 2B deployment planner — HIGHEST.** At test time we have NO gold actions (seed's caveat — they exist only on demos). The move: generate gold-action-grounded plans offline (Q1's label pass), then **LoRA-distill them into the 2B** so the *live* 2B emits gold-action-QUALITY text from frames alone. This is RT-H / LAPA "language motion" distillation (arXiv:2410.11758) and is the owner's VLM-abstracted-plan axis. Qwen-LoRA is now allowed; keeps inference cheap (2B, no 12B at deploy). **Directly attacks the oracle−live gap** (the +21 R11 ceiling). Q1 produces its training data → Q3 is the natural successor.
2. **(c) replan-less / hold a stable plan — CHEAP, ship NOW as a deployment default.** R11: live re-planning churns tokens (drift 0.022-0.031) without payoff and can HURT (MMX −11). `--replan-every N` (large N) + hysteresis is near-free. But it is a STABILITY band-aid, not a QUALITY lever (a stable bad plan is still bad) — pair with (b), don't substitute.
3. **(a) bigger planner as DEPLOYMENT planner — use as the offline TEACHER only.** Gemma-4-12B/Qwen-9B/27B generate the good gold-action labels (the teacher) but are expensive at inference and violate GPU-frugal. Subsume into (b): big planner = teacher, 2B+LoRA = student/deployment. Do NOT deploy the 12B live.
4. **(d) gold-action prompting at inference — REJECT for deployment.** No gold actions at test time (seed's own caveat). It is a TRAINING/labeling tool (the Q1/Q3 data source), full stop.

**How to measure plan quality WITHOUT the oracle:** the emulator's survival-weighted advance from the LIVE plan vs null — `Δ_live = survival_advance(live_plan) − survival_advance(null)`, the R11 `staleness_probe` `live` mode, 3+ seeds, bootstrap CI. A better live planner RAISES Δ_live toward the oracle ceiling (shrinks oracle−live: SMW +6.3, MMX +12 already on pooled demo-fit, DISCRIMINATOR:473-474). Hack-proof (emulator outcome), needs no oracle. Secondary stability proxy: plan-token drift / action-direction flip-rate (R11 instrumented both).

---

## Q4 — Exploration demos (Super Metroid, Minish Cap): what they ADD + how to use them

**(i) Fold into the pooled plan-OOD generalist as POSITIVE plan-diversity — yes, for TRAINING; separate coordinate for EVAL.** The exploration actions (wait, climb/up, talk, pick-throw, backtrack/retreat) are EXACTLY the situational behaviors platformers under-label — and they are the SPECIFIC battery contrasts R9 found hardest to preserve (retreat R=0.03, up R=0.09 under situational; DISCRIMINATOR:545). So these demos supply real retreat/up/wait chunks that strengthen the expressiveness battery the Q1 generalist must pass. FOLD their chunks into the pooled demo-fit. But EVAL needs a per-genre coordinate (Metroid is metroidvania = horizontal+vertical; Minish is top-down = NO advance axis) — keep training pooled, eval per-genre.

**(ii) Does gold-action narration auto-solve the un-annotated Ridley fight? UNVALIDATED — and the one existing attempt FAILED.** I checked: `docs/demos/demos/SuperMetroid-Snes/20260629-132934/vlm_narration_ridley.json` exists (interleave=True, schema=True, 24 entries) but **ALL 24 text fields are EMPTY**. So the seed's hypothesis "the action IS the label → auto-narrate the boss the human couldn't" is currently *unproven by a failed run*, not demonstrated. **The experiment (do this, don't assume it):** re-run `vlm_narrate_demo.py --interleave-actions` on the Ridley frames+gold-actions, **DEBUG the empty output first** (likely causes: no Super-Metroid control-schema key in `--game` so `schema=''`; `enable_thinking` not forced False on the labeler; or the seg window landing on the death/escape tail). **Metric = maneuver-word coverage** (jump/shoot/aim/dodge counts, the SMW {duck,spin,stomp} method) + non-empty ≥80% of segments + a human spot-read. Pass: ≥1 grounded maneuver word per segment, non-empty ≥80%. If it passes, it's a STRONG generalist argument — auto-label the un-annotatable hard parts (bosses, escapes) for free — and it directly addresses the owner's two skipped sections. If it fails, the gold-action labeler has a coverage hole on chaotic combat (worth knowing before relying on it).

**(iii) Eval for top-down Minish (no advance axis):** ranked by infra cost — (a) **room-reached** = map-id RAM transition count (cleanest; analogous to the Solarus `map` var and the SMB1 x-addr hunt, DISCRIMINATOR:415-419); (b) **dialogue-advanced** = text-box state RAM; (c) **item-count** = rupee/grass-throw counter RAM. All need a RAM-address hunt (the `gba_minish_cap` env exists, gba_env.py:68). **Defer the eval infra**: Minish is a TRAINING-diversity contributor first (enriches the pooled battery), an eval target second. Don't block tonight on a RAM hunt.

---

## Q5 — The ONE coherent next-phase plan + tonight's experiment

**The architecture (unchanged shape; this round sharpens the TRAINING SOURCE and kills my R11 staleness lever):**
```
System-2 teacher (BIG VLM, frozen)  ── gold-action INTERLEAVED narration ──▶  VLM-abstracted per-chunk plans   (Q1 label pass; action IS the label)
        │ distill (Qwen-LoRA)                                                                │ condition
        ▼                                                                                    ▼
System-2 deploy (Qwen3.5-2B + LoRA) ── frames-only plan (gold-action QUALITY) ─▶  SHORT PATH: K=8 plan tokens (stable, tok-drift~0)
        │ replan-every N + hysteresis (stability)                                            │ inject (cross-attn, null-exact)
        ▼                                                                                    ▼
CONSOLIDATION = KL-anchored plan-head demo-fit on gold-action plans (R9 winner ⊗ Q1)   ──▶  FROZEN DiT actor → 18-step chunk
        │ disjoint-merge (R8/R9 proven)
        ▼
LONG PATH (ADD only, router-gated) = plan-token-GATED DiT-LoRA on Rex chunks + EATA Fisher (Q2, post-detector)
GUARDS: KL-anchor (=CoTTA functional restore, arXiv:2203.13591); ≥3 seeds + bootstrap (≥5 for any outcome claim); null bit-identity=0.0; survival-aware advance.
```
ONE generalist (pooled, per-chunk abstracted conditioning — no per-game weights), VLM-frozen-deploy (LoRA-distilled, teacher stays frozen), null-invariant.

**Priority:** (1) Q1 gold⊗KL demo-fit TONIGHT → (2) Q3 Qwen-LoRA distill → (3) Q2 Rex-LoRA (post sprite-status detector) → (4) replan-every default + (5) fold-exploration / Ridley-auto-narration debug.

**Tonight's experiment is Q1** (full spec in TL;DR): 3 arms (terse+KL / gold+KL / gold−KL) × 3 seeds on SMW + MMX winner-replication; metrics Δ_plan + per-channel duck_probe + battery R + null bit-identity; pass bar B: Δ_plan≥+40 ∧ stick-duck≥30% ∧ R≥0.80 ∧ null=0.0; named falsifier R<0.80 ⇒ demote gold to a duck-stack on terse+KL. GPU: 9 small plan-head fits + 1 frozen label pass ≪ 4×A6000 overnight.

---

## Steelman of GPT-5.5's likely pick

GPT-5.5 has been the **audit-the-metric / emulator-outcome** voice (caught the Sonic screen_x hack R4; forced ≥5 seeds+RTG R9; exposed the RTG single-seed artifact). Its likely R12 pick: **run the Q2 Rex DiT-LoRA ADDITION tonight**, arguing — "the router CONFIRMED a real capability GAP (spin-jump-rex outcome flat 3.75→7.5%); capability addition is the actual frontier; Q1's duck-fix merely RESTORES expressiveness the base model already had (battery R=0.165 is a demo-fit regression, not new capability). Build the sprite-status kill-detector, plan-token-gated LoRA on the human Rex chunks conditioned on the gold-action 'spin-jump the Rex' plan, ≥5 seeds, emulator P(handled-rex) — and don't trust any behavioral duck-RATE, it's hackable; demand the emulator outcome." It will also likely note Q1 and Q2 FUSE (gold-action narration is the label source for the Rex LoRA too) and push to do the harder one.

**Where I concede:** (1) emulator outcome is the hack-proof metric — folded in as Q2's gate (P(survive+advance past rex), sprite-status for true kill). (2) Gold-action narration IS the right label source for the Rex chunks — Q1⊗Q2 share it (nice convergence). (3) ≥5 seeds for any outcome/reward claim, non-negotiable.

**Where I push back (tonight = Q1, not Q2), three data-grounded reasons:** (a) **SNR.** The Rex outcome is near-chance for BOTH arms (3.75% vs 7.5%; rel +0.04) on a hard 4-chunk reconstructed scenario — the router's OWN caveat (DISCRIMINATOR:619). Training a LoRA against that tonight risks exactly the single-seed-artifact trap that voided the RTG "fix" (DISCRIMINATOR:150-172). The reconstruction must improve AND the sprite-status detector must be BUILT first — multi-night, not one. (b) **Critical path.** The duck-collapse is NOT "just restoring base expressiveness" — it's a GENERALIST-BLOCKING regression (plain BC battery R=0.165 KILLS the whole action space; DISCRIMINATOR:544) that every multi-game demo-fit deployment inherits, *including the generalist the Rex-LoRA would merge into*. Fixing it is upstream of Q2. (c) **ROI.** Q1 reuses the validated R9 KL-anchor + `demo_bc` at zero new tooling AND tests R11's text-quality redirect — strictly dominant tonight. Correct sequence: Q1 tonight (cheap, decisive, generalist) → sprite-status detector tomorrow → Q2 with ≥5 seeds + emulator outcome.

---

## Contradictions called out (this round vs prior conclusions, incl. my own)

1. **MY OWN R11** (forward-pass token-TTA `fresh`/`fresh_ema` recovers staleness) — **FALSIFIED by the experiment I proposed.** R11 result: fresh≈cached≈null≪oracle, tok-drift~0, live can HURT (MMX −11). My named falsifier ("fresh≈stale ⇒ TEXT-bound") fired cleanly. I OWN it: token-TTA is dead as a staleness fix; R12 pivots to plan-TEXT quality (gold-action plans + Qwen-LoRA distill). von Oswald "short path is already TTT" (arXiv:2212.07677) is moved from my "lean on it" column to the "true but irrelevant here — tokens already stable" column.
2. **Seed's Q1 framing presumes gold-action ≻ terse.** I partly DISAGREE: R9 showed the closest existing analog (fixed-taxonomy situational) LOST breadth (R=0.53 vs terse+KL 0.83). Gold-action's OPEN vocab MIGHT fix that, but it is an empirical question — hence the explicit battery-R≥0.80 pass bar and the named falsifier. I refuse to assume the answer the seed leans toward.
3. **R9's "optionally STACK situational for duck authority"** — gold-action is the better stack source, BUT only if it doesn't re-import the taxonomy loss; the C arm (gold−KL) tests whether per-chunk specificity alone suffices or whether the KL-anchor is still load-bearing for breadth. If C passes breadth, we DON'T need gold's open vocab — terse+KL stays.
4. **The discriminator's per-game ROUTING** (SMW=plan-OOD→supervised demo-fit; Sonic=actor-OOD→RWBC; DISCRIMINATOR:314-322): gold-action conditioning is a PLAN-OOD lever — expect it to lift SMW/MMX, do NOTHING for Sonic (frame-obvious direction). Don't mis-eval it on the actor-OOD leg.
5. **Seed's optimism that auto-narration solves the un-annotated Ridley boss** — the ONE existing attempt (`vlm_narration_ridley.json`) is ALL-EMPTY. The capability is UNVALIDATED; Q4(ii) is "debug + prove it," not "use it."

### Repo anchors (all runnable)
`planner_poc/demo_bc.py` (`--kl-anchor` =CoTTA functional restore, `--use-correct-plan`, `--situational-plans`, `--duck-probe`, `--save-delta`; `load_demo_chunks`/`build_anchor_pool`/`duck_probe` dim22 stick + dim1 dpad; NEW `--vlm-plans` per-chunk source) · `planner_poc/vlm_narrate_demo.py` (`--interleave-actions --game --seg-seconds`; `render_action`; `PlanEncoder.generate_interleaved`) · `planner_poc/expressiveness_battery.py` (5-contrast R) · `planner_poc/maneuver_router.py` (evoc_ratio + emulator outcome; Rex routing) · `planner_poc/eval_common.py` `survival_advance` (death-aware) · `planner_poc/staleness_probe.py` (`live` mode = no-oracle plan-quality metric) · `nitrogen/flow_matching_transformer/lora.py:8-12` (gate LoRA to plan-token positions for null-exactness) · `docs/demos/demos/{SuperMetroid-Snes,LegendOfZeldaTheMinishCap-GbAdvance}` (exploration demos; the empty `vlm_narration_ridley.json`).
Constraints honored: ONE generalist (pooled, per-chunk abstracted plans); VLM-frozen-deploy (teacher frozen, 2B LoRA-distilled); exact null-invariance (plan-head-only fit + gated-LoRA + bit-identity=0.0); VLM-abstracted (not raw) plans; survival-weighted death-aware eval; ≥3 seeds + bootstrap CIs; GPU-frugal (tonight all frozen-base plan-head fits); kill by numeric PID; no commits.
