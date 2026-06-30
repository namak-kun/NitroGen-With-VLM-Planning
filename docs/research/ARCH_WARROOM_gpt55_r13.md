# ARCH WAR ROOM R13 — GPT-5.5 position memo
## System-2 → System-1 knowledge transfer and the recursive skill hierarchy

**Position:** the project now has a real transfer primitive, but it is not “the VLM teaches the actor by naming everything.” The measured taxonomy is: **evocation** when S2 names a behavior already latent in S1's action manifold, and **addition/consolidation** when the desired *outcome* is not reliably produced by that manifold. The hierarchy loop should therefore be router-gated: evoke cheaply, verify in emulator, then KL-anchored consolidate successful evocations into null/instinct so S2 can climb to higher abstractions. I would lead with P5+P3/P4, not P1 alone: P1 proves the path exists; P5 proves it can be made persistent without destroying breadth; P3/P4 prove this can remain one generalist.

Ground truth used: R13 seed P1-P7 and Q1-Q6; `DISCRIMINATOR_RESULTS.md` duck-on-command, R9 KL-anchor, capstone/pooled, staleness, and maneuver-router sections; `ARCH_WARROOM.md` R10-R12 settlement. Key citations are inline as file:line ranges.

---

## 0. What P1-P7 actually establish

1. **P1, evocation exists but is narrow/state-dependent.** Base SMW duck-on-command moves DOWN from 1.2% under terse text to 10.8% under explicit “press down to duck,” while the post-demo-fit pooled head collapses to 0% for all duck texts (`DISCRIMINATOR_RESULTS.md`:497-508). This is a clean positive: text can call a primitive already present in S1. It is also not a long-horizon claim.
2. **P2, outcome addition is distinct from button evocation.** In the Rex test, jump/spinjump button behavior is 7.24x evocable, but the emulator outcome “survive + advance past threat” moves only ~0.04→0.07, near chance (`DISCRIMINATOR_RESULTS.md`:606-621). Therefore route by *outcome*, not by button proxy, for outcome skills.
3. **P3, shared concept transfer is real.** One pooled plan-head fit on SMW+MMX+SMB1 widens plan-vs-null on SMW and MMX, 3/3 seeds, exact null invariance; SMW even beats solo by 1.3x (`DISCRIMINATOR_RESULTS.md`:357-375). Later 3-game eval shows 2/3 significant and SMW mean up but high variance (`DISCRIMINATOR_RESULTS.md`:423-431).
4. **P4, one-model/no-interference is plausible.** Plan-head pooled delta plus Sonic LoRA merged into one btn_s600 retains or improves SMW, MMX, Sonic with exact null invariance (`DISCRIMINATOR_RESULTS.md`:384-404). Earlier capstone shows disjoint param groups stack (`DISCRIMINATOR_RESULTS.md`:324-343).
5. **P5, KL-anchor is the consolidation primitive we were missing.** Plain BC on a constant plan destroys expressiveness (battery R=0.165, duck 0%). KL-anchor gives Δ_plan +45.1 mean, 3/3, duck monotone, expressiveness R=0.83, only pass (`DISCRIMINATOR_RESULTS.md`:536-563; `ARCH_WARROOM.md`:599-614). This is the strongest mechanism-level result.
6. **P6, do not chase token staleness.** The K-token path is stable: tok-drift ~0.001; fresh≈cached≈null and all below oracle in SMW; MMX confirms fresh re-grounding does not help and live re-planning hurts by up to 11 (`DISCRIMINATOR_RESULTS.md`:566-604). The bottleneck is text quality/stability, not token TTA or actor-weight TTT.
7. **P7, live-plan robustness can be trained.** Demo-fit turns live SMW plan from harmful (-10.7) to +56.3 and improves live plans on SMW/MMX/SMB1 (`DISCRIMINATOR_RESULTS.md`:444-480). Caveat: those runs are short-horizon fixed starts, not minutes-long closed loop.

---

## Q1. Cheap decision rule: evocable vs addition

**Decision rule:** classify a wanted skill at the *owner's desired outcome level*, not at the first visible button. Use a three-stage online router:

1. **Reflex/no-op:** if null already succeeds, do not train. For primitive actions use null reflex/AUC; for outcomes use emulator success. Threshold: null outcome success ≥0.5 or null action reflex ≥0.6 on the relevant save-state set. Duck near bullet states illustrates this: null already ducks 0.457 and plan only adds 1.25x (`DISCRIMINATOR_RESULTS.md`:611-614), so that local bullet-state behavior is already partly instinct.
2. **Evocation:** if null is low but explicit plan raises the *same-level metric* reliably. For primitive/action skill: evoc_ratio ≥2x and absolute planned action rate ≥5% over null. P1 duck from generic starts passes: 1.2→10.8% under explicit text (`DISCRIMINATOR_RESULTS.md`:497-505). For outcome skill: planned emulator outcome must improve by at least +0.20 absolute or ≥2x with CI>0.
3. **Addition:** if explicit plan moves buttons but not outcome, or if plan does not move either. Rex is the canonical split: button ratio 7.24x but outcome +0.04 near chance, so addition (`DISCRIMINATOR_RESULTS.md`:611-621).

**Minimal online test using existing tool:** run `planner_poc/maneuver_router.py --game smw --k 4 --out files/r13_router_<skill>.json` from reconstructed save-states. For each skill, collect at least 10 states for outcome skills, 3-5 for cheap primitive probes. Report: null reflex/action rate, plan action rate, evoc_ratio, null outcome, plan outcome, Δoutcome. **Pass/fail:** EVOKE only if the metric the owner cares about passes; ADD if outcome Δ<+0.20 or CI crosses 0 despite button evoc_ratio. Use 3 random seeds for sampling where stochasticity matters.

**Mechanism:** S2 text is a retrieval key into S1's existing manifold. The router measures whether retrieval is enough. The emulator outcome test measures whether the retrieved primitive composes into the intended game-state transition.

**Falsifier:** if multiple supposedly evocable primitives (duck, wait, retreat, jump) show high action evoc_ratio in offline probes but fail in live save-state outcomes, then “evocation” is just open-loop action bias, not skill transfer. In that case move the router's first-class metric entirely to emulator outcomes and demote action probes to diagnostics.

---

## Q2. Persistent evocation → instinct: rank mechanisms and pick first

### Ranking

**1. Try first: (a) amortized KL-anchored consolidation / self-distillation into the null policy.**

This is the best-supported path because P5 already demonstrates the hard part: adding behavior while preserving expressiveness. The mechanism is: S2 evokes a behavior on verified states; emulator filters winners; then a sleep-time M-step trains the bridge/actor so the behavior fires with null or a shorter/higher-level plan. Crucially, use KL-anchor to preserve the pre-fit plan-token map over a broad plan distribution, not naive BC. This is policy distillation/ReST-EM style: E-step propose/score, M-step distill winners with a KL/functional anchor. It matches Sutton-Precup-Singh options (skills become temporally extended policies), policy distillation, ReST-EM, and VOYAGER's skill-library loop, but here the “library write” is into the S1 bridge/null habit rather than a code file.

**Concrete first experiment:** SMW duck instinct.

- Use `planner_poc/maneuver_router.py --game smw --k 4 --out files/r13_duck_router.json` on bullet-bill or low-ceiling duck states to identify states where explicit duck plan helps or where null reflex is incomplete.
- Build training chunks from top successful explicit-duck rollouts / human duck chunks.
- Fit plan-head-only with `planner_poc/demo_bc.py --game smw --train plan_head --steps 600 --use-correct-plan --kl-anchor 0.3 --kl-anchor-n 32 --duck-probe --save-delta files/r13_duck_instinct_s<S>.pt`, three seeds. If using explicit duck text for the positive arm is not already wired, use the existing duck-probe/eval text as the evaluation and keep training on the available terse+KL baseline; the minimal claimed effect is null-policy ducking at the selected states.
- Evaluate on the same held-out save-states and fresh starts: null duck/action rate, explicit-plan duck rate, survival/advance, expressiveness battery R, and action-level null bit identity for unrelated null states.

**Pass bar:** on held-out bullet states, null duck rate increases by ≥+20 percentage points or reaches ≥0.65; explicit-plan duck remains ≥base explicit plan (no loss of evocation); expressiveness R≥0.80; Δ_plan for advance not below R9 KL by >20%; null bit-identity remains exact on masked-null states not in the consolidated skill set. Across 3 seeds, median pass and no catastrophic seed.

**Falsifier:** if KL-anchored consolidation cannot raise null ducking on held-out skill states without lowering expressiveness below R=0.80 or erasing explicit plan responsiveness, then P5 is a plan-following regularizer, not an instinct-write mechanism. Then do not recurse yet; switch to option-token codebook or gated LoRA for state-conditioned execution.

**2. (c) skill-token codebook as the second mechanism, not first.**

A codebook is the right compression target, but it needs verified skills to put in the codebook. After consolidation has shown that “duck” can become owned, mint a skill token such as `<smw.duck_bullet>` or a learned vector macro and train S2/Qwen-LoRA to emit it from visual contexts. This is options-as-tokens: S2 discovers; emulator verifies; S1 learns the option; S2 reuses it in higher plans. It addresses token dilution, but by itself does not prove persistence.

**Falsifier:** if codebook tokens improve prompt bandwidth but not null/no-plan execution or outcome success, they are just shorter prose, not instinct.

**3. (b) S2-written fast weights/hypernet online tweak, defer.**

Fast weights are theoretically elegant (Ha/Schmidhuber hypernetworks; Schlag fast-weight programmers; cross-attention-as-implicit-optimization per von Oswald), and the R11 memo rightly says the K-token path is already a fast-weight programmer (`ARCH_WARROOM.md`:663-687). But P6 empirically says token re-grounding is not the current bottleneck; text quality is. Online weight updates also risk reproducing R9 collapse. Use fast weights only later as a router-gated execution adapter, with Fisher/EWC and null-safe plan-token gating.

**Where I defer to Opus-4.8:** Opus will likely prefer the KL-anchor/CoTTA framing and may push codebook earlier as the clean HRL abstraction. I agree on the framing but hold firm that codebook comes after the first null-instinct write, because without persistence it does not answer the owner's “S2 can stop naming it” requirement.

---

## Q3. Token-dilution wall: bandwidth path preserving exact null invariance

**Do not lead with widening K.** R9 already refuted K=8 as the immediate duck bottleneck: base K=8 can evoke duck; collapse came from data/objective, not capacity (`ARCH_WARROOM.md`:563-569). Widening K=8→32 is a later bounded capacity ablation, but it changes plan-head shape and risks retraining churn. It does preserve masked-null if plan tokens are absent under null, but it is the least semantically efficient answer.

**Recommended bandwidth stack:**

1. **Short term: stabilize and improve text.** Because P6 shows text-bound churn, use better abstracted plans / Qwen-LoRA distillation, and a replan hysteresis/hold policy; do not spend on token TTA.
2. **Medium term: skill-token codebook.** Each learned option token expands to a K-token macro or retrieves a codebook vector appended to the plan-token sequence. Null-invariance is preserved by construction if the codebook path is behind the same masked-null gate: under null, no plan tokens and no codebook tokens are injected, so the base DiT path is bit-identical.
3. **Long term: hierarchical plans.** S2 emits abstract option tokens; a small mid-layer expands them into per-state execution tokens. This is the recursive hierarchy: “cross gap” → `<duck_bullet>` + `<jump_rex>` rather than prose over every primitive.

**Architecture detail:** keep K=8 base tokens, add M optional skill-code tokens (e.g., M≤4) generated by a learned codebook/resampler. Concatenate `[image | plan_K | skill_M]` only in non-null mode. Add router-gated skill masks so irrelevant options do not dilute attention. This preserves exact null because null mode masks all plan/skill tokens. Measure null bit-identity after every change.

**Falsifier:** if adding codebook tokens lowers the expressiveness battery R below 0.80 or reduces Δ_plan/live outcome versus prose at equal cfg, the codebook is causing attention dilution rather than compression. If codebook helps but null bit-identity fails, the gating implementation is invalid regardless of performance.

---

## Q4. Subtly different execution: shared concept channel + small execution add, one model

**Position:** yes: shared concept channel plus router-gated execution residual, but not per-game models.

**Concrete architecture:**

- **Shared concept channel:** the existing generalist plan-head / resampler / adapter trained pooled with KL-anchor. This learns broad semantic plan directions and already transfers across SMW/MMX/SMB1 (P3) and merges with Sonic LoRA (P4).
- **Execution residual channel:** a single library of small adapters, not separate checkpoints. Use plan-token-gated DiT-LoRA or AdaLN/FastPlanMod modules with a router gate `g(s, plan, game_context, skill_id)`. The gate can be produced by a shared hypernet conditioned on frame embedding, game-context embedding, and option token. The residual is sparse/top-k: only the relevant execution adapter fires.
- **Null-safe construction:** residual keys attend only to plan/skill-token positions; when null plan is masked, the residual receives no key/value or gate=0, preserving exact null. Verify action-bit identity.
- **One model:** all adapters/codebook entries live in one checkpoint with shared router and shared base. “Per-game” is only metadata/context in the router, analogous to a mixture-of-experts index, not a separate model. Capacity can be allocated per skill, but optimization and deployment are global.

**Mechanism:** P3/P4 show broad concept vectors are shared; P2 shows precise Rex dispatch needs game-state execution detail. The execution residual supplies only the missing motor nuance while the concept channel remains common.

**Experiment:** after duck-instinct, build Rex as first ADD path only after sprite-status detector exists. Train rank-16 plan-token-gated DiT-LoRA on human Rex-kill chunks with KL/Fisher guard; evaluate `maneuver_router.py` outcome success. **Pass bar:** P(survive+advance or true sprite-kill) improves by ≥+0.20 absolute over null and over plan-head-only, 5 seeds, no expressiveness R<0.80 on bridge battery, null bit-identical. **Falsifier:** if adapter improves button jump but not emulator outcome, the execution channel is still not learning the skill; need DAgger on policy-visited save-states.

---

## Q5. Recursion loop with our tools + minimal this-week SMW experiment

**Loop:**

1. **S2 proposes.** Use VLM/gold-action plans or explicit prompts to propose a maneuver from a save-state: “duck under bullet,” “spin-jump Rex,” later “cross this enemy corridor.”
2. **Emulator verifies.** Use `maneuver_router.py` and emulator save-states to score outcome, not just actions. This is the E-step / option-discovery filter.
3. **Route.** If null succeeds: no-op. If explicit plan succeeds: evocation. If outcome fails: addition.
4. **Consolidate.** For evocation, distill successful S2-conditioned behavior into null/shorter-plan policy via `demo_bc.py --kl-anchor`. For addition, train gated DiT-LoRA only on ADD chunks.
5. **Abstract.** Mint a skill token / update Qwen-LoRA so S2 emits higher-level plans that assume the new instinct.
6. **Repeat at higher level.** Once duck is instinct, S2 stops saying “press down to duck” and can say “cross the bullet corridor”; once that is instinct, it composes with Rex and platform timing.

**Minimal this-week experiment: SMW duck becomes instinct.** This is the right experiment because it uses P1 (evocable), P5 (KL-anchor), P6 (avoid token-TTA distraction), and existing tools; it does not require the unbuilt Rex kill detector.

**Protocol, no new GPU-heavy infrastructure:**

- **States:** bullet-bill/duck-relevant SMW save-states from demos. Split by state, not chunk: train 70%, holdout 30%. Include generic starts as negative controls.
- **Baseline classification:** `planner_poc/maneuver_router.py --game smw --k 4 --out files/r13_duck_router.json` to measure null and explicit duck plan.
- **Consolidation arms, 3 seeds:**
  - A: R9 control/reference: `demo_bc.py --game smw --train plan_head --steps 600 --use-correct-plan --kl-anchor 0.3 --kl-anchor-n 32 --duck-probe --save-delta files/r13_duck_kl_s<S>.pt`.
  - B, if explicit-duck training text wiring exists/gets minimally reused from duck-probe: same plus explicit duck plan on duck chunks; otherwise defer B and keep A as minimal.
  - C: no-KL or situational baseline only if time permits, to ensure we did not rediscover R9 collapse.
- **Evaluation:** on held-out bullet states: null duck rate, explicit duck rate, death/survive+advance, plan-vs-null advance on generic starts, expressiveness battery, and null bit-identity. Then run `staleness_probe.py --game smw --delta files/r13_duck_kl_s<S>.pt --modes null oracle live cached fresh --n-starts 6 --seed <S> --out files/r13_duck_stale_s<S>.json` to ensure live text behavior did not regress.

**Pass/fail:**

- PASS: null-policy duck at bullet states +20pp or ≥0.65; survival/advance not worse; explicit plan still evokes; expressiveness R≥0.80; generic Δ_plan retains ≥80% of R9 KL; null bit identity exact; 2/3 seeds pass.
- FAIL/FALSIFIER: null duck does not improve despite explicit-plan winners, or improvement only occurs by collapsing to always-down / hurting advance / R<0.80. Then consolidation into null is not solved; next mechanism is codebook token `<duck_bullet>` rather than true instinct.

**Disagreement with seed:** I would not make Rex the minimal recursion demo. P2 is too close to chance and lacks a true kill detector; R12 already deferred Rex behind sprite-status RAM and better save-state reconstruction (`ARCH_WARROOM.md`:735-744). Duck is the only this-week experiment that directly tests “evoked behavior becomes instinct” without conflating with unlearned outcome addition.

---

## Q6. Highlight framing

**Lead with this:** “We have the first measured path from S2 language to S1 skill transfer in a frozen actor: S2 can evoke a latent primitive; KL-anchored consolidation can add supervised behavior while preserving the rest of the plan/action vocabulary; and pooled training/merge keeps it one generalist.”

I would structure the highlight reel as:

1. **P5 as the lead mechanism.** KL-anchor is the most important positive result because it solves the failure mode that would kill recursive transfer: naive demo-fit overwrote duck and the whole action battery; KL-anchor added advance (+45.1 mean, 3/3) while preserving expressiveness R=0.83 (`DISCRIMINATOR_RESULTS.md`:536-563). This is “add without forgetting,” the prerequisite for instinct.
2. **P1 as the intuitive demo.** Duck-on-command is the clean visual proof that S2 text can summon an S1 primitive: 1.2→10.8% under explicit text (`DISCRIMINATOR_RESULTS.md`:497-505).
3. **P3/P4 as the one-generalist proof.** Pooled plan-head transfer and disjoint merge show this is not per-game one-off tuning (`DISCRIMINATOR_RESULTS.md`:357-404).
4. **P7 as deployment relevance.** Live VLM plans were harmful on base SMW and become useful after demo-fit (`DISCRIMINATOR_RESULTS.md`:444-480), but be honest that this is short-horizon.
5. **P2 as discipline.** Not every named thing transfers. Rex is addition, not evocation, at the outcome level.
6. **P6 as a negative but valuable settlement.** Do not spend Round 13 on token staleness; the problem is text quality and persistent skills.

**Honest caveats:** most wins are oracle-plan or fixed-save-state, short horizon, and platformer-biased. Live-plan robustness is measured over short rollouts, not full games. P1 is an action primitive, not a verified long-horizon option. P5 is consolidation-without-forgetting, but not yet proof that a transient evocation can be written into the null policy and then used by S2 as a higher abstraction. That is exactly why the SMW duck-instinct experiment is the next decisive test.

**Final stance:** do the recursion in this order: router → evocation verified → KL-anchored null consolidation → skill token/Qwen-LoRA abstraction → gated execution adapters for true additions. This is the smallest path that respects the owner constraints: one generalist, exact masked-null, no per-game checkpoints, and no relitigation of text-bound staleness or naive RWBC collapse.
