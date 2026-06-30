# ARCH WARROOM R10 — Opus-4.8: EVOCATION-FIRST. The router is a 2-STAGE gate, not a 1-probe; the long path is DiT-LoRA, not FastPlanMod.

## TL;DR
**Root framing call: VALIDATE the evocation/addition split, SHARPEN the router, PARTIALLY REJECT the FastPlanMod-as-long-path reading.**
The orchestrator's split is real and unifies the open problems — but its proposed router (the actor-OOD per-dim probe alone) is
**wrong by our own R9 data**. The null-reproduction probe measures *"is this the DiT's DEFAULT reflex,"* not *"is this IN the DiT's
manifold."* Duck is the counterexample: 9× evocable (1.2%→10.8% on command, R9) yet a LOW default DOWN-rate (~2.9%) → the null probe
would misroute duck to **ADDITION** when we KNOW it's **EVOCATION**. So the router must be a **2-stage gate**: (1) null-AUC screens
"already reflexive → no-op"; (2) the **evocation-ratio** (explicit-plan vs null, behavioral + *emulator-outcome*) decides evoke-vs-add.
My thesis: **most of the owner's examples (duck, spin-jump-kills-Rex) are EVOCATION — the DiT has the buttons; the gold narration is
the missing ASK. Only grab-mesh (a climb trajectory the DiT never represents) is genuine ADDITION.** Therefore wire the gold narration
into the **KL-anchored short path FIRST** (cheap, we PROVED it works in R9: +45.1 Δ_plan, battery R=0.83), and reserve the **DiT-LoRA
long path** for the router-confirmed "add" set only. Continuity: R9 my taxonomy-free KL-anchor WON; R10 it becomes the short-path carrier.

**The ONE experiment tonight (run BOTH halves in parallel on 4×A6000 — the R9 pattern):**
- **PRIMARY / GPU0 (zero training): the maneuver ROUTER (Q1).** Reconstruct frame-exact save-states at duck / spin-jump-Rex /
  grab-mesh from the frame-aligned gold narration; on the FROZEN DiT measure null-AUC + evocation-ratio + the **emulator counterfactual**
  P(outcome|plan)−P(outcome|null). Classify each maneuver evoke/add/reflexive. Decisive, hack-proof, answers the owner's literal question
  ("does the DiT know spin-jump kills a Rex?"), gates Q3.
- **COMPANION / GPU1-3 (the cheap short-path fix, 3 seeds): gold-narration KL-anchored demo-fit (Q4).** Arm A2=gold-narration vs
  A0=terse vs A1=SIT, all +`--kl-anchor 0.3`. This is the actual "make System-2 text reach System-1" test.

**Pass bar (combined):**
- **Q1 controls land:** duck classifies **EVOKE** (AUC<0.6 AND evoc-ratio ≥2×) [positive control]; grab-mesh classifies **ADD**
  (AUC<0.6 AND evoc-ratio <2× AND emulator climb-Δ <+0.3) [negative control]. spin-jump-Rex = the owner's real unknown, reported by the
  emulator kill-Δ (≥+0.3 ⇒ evocable).
- **Q4 short-path win:** A2 Δ_plan POST **≥+20** (3/3 seeds, bootstrap CI>0), null action-bit-identity **=0.0**, expressiveness battery
  **R≥0.83** (no regression vs R9 KL-anchor), AND on **≥1 situational maneuver A2 evoc-ratio > A0 terse** (gold narration surfaces a
  maneuver terse doesn't). **Falsifier:** A2 within-CI of A0 on Δ_plan AND no maneuver evoked → System-2 text is NOT becoming
  System-1-relevant via conditioning → the bottleneck is the manifold → pivot to the Q3 DiT-LoRA long path on the router-"add" chunks.

---

## Q1 — Evocation/Addition split + the router (VALIDATE split, SHARPEN router)

**Split: VALIDATED.** EVOCATION = bridge (resampler→adapter→K plan-tokens→cross-attn) + KL-anchored demo-fit summons a maneuver the
DiT's action manifold already contains. ADDITION = a maneuver the DiT physically can't synthesize (grab-mesh climb) → needs a carrier
that expands the manifold (DiT-LoRA, Q3). This is the correct unification of (stale plans / actor-OOD / capability transfer).

**Router: SHARPENED to a 2-stage gate (reject the 1-probe null-AUC version).** Why: the null per-dim probe (`base_dit_perdim.py`:
AUC of frozen-null pred vs human-on, given the frame) measures *default reflex*, not *manifold membership*. Duck proves these dissociate
(evocable 9× but low default-rate). A 1-probe router would send every "in-manifold-but-not-default" maneuver to the expensive,
null-invariance-risky long path. The gate:

| stage | metric | verdict |
|---|---|---|
| 1. reflexive? | null-AUC ≥ 0.6 (DiT raises this dim unprompted at the maneuver frame) | **NO-OP** — already a reflex |
| 2a. evocable? | AUC<0.6 **AND** evoc-ratio = rate_d(explicit plan)/max(rate_d(null),ε) **≥ 2.0** | **EVOKE** (short path: KL-anchored demo-fit) |
| 2b. add? | AUC<0.6 **AND** evoc-ratio < 2.0 **AND** emulator outcome-Δ < +0.3 | **ADD** (long path: DiT-LoRA) |

**For OUTCOME maneuvers (spin-jump-kills-Rex), the per-dim button probe is insufficient — the "kill" is an outcome, not a press.**
Use the owner's save-state idea as the *decisive* metric: load the Rex state, roll the explicit "spin jump on the rex" plan vs null
K=8×, score **P(Rex removed AND Mario survives contact | plan) − P(·| null)** from frame-exact RAM (enemy-slot/coin/life vars via
`env._var`). ≥+0.3 ⇒ evocable (the DiT *can* be steered to the killing maneuver); ≈0 ⇒ addition.

**Maneuvers + dims:** duck → DPAD_DOWN(dim1)/stick-Y(dim22>0.6) [known evocable, positive control]; spin-jump → SNES-A ≈ EAST(dim5)
(resolve from `BUTTON_ACTION_TOKENS`, *verify by the demo chunk at the narrated frame*, but DECIDE on the emulator kill-Δ); grab-mesh →
UP-hold(stick-Y dim22<0.4) + sustained screen-Y climb on the fence [owner says DiT lacks it → expected ADD, negative control].

**Save-state reconstruction (no new recording needed):** the demos already carry full per-frame `observations`+`actions` (demo.npz:
7678×{12 actions, 224×256×3 obs}) and a loadable `initial.state`. `demo_narration.py` frame-aligns the owner's per-second narration →
`narration.json` gives the FRAME INDEX of each maneuver. Reconstruct: `env.load_state(initial)`, replay `demo.npz['actions'][:f]` to the
narrated frame `f`, `env.save_state()`. Gives "a state AT a Rex / AT the mesh." ≥6 states per maneuver (the two narrated SMW demos
105913/105939 have 10+80 narrated notes incl. duck@00:08/00:14/01:21/01:29, spin-jump-Rex@00:16/01:35, climb@state-7).

**Runnable (GPU0, frozen, ~1–2 GPU-hr, NO seeds — average over 6 states × K=8 samples):**
```bash
RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
PY=.venv/bin/python ; F=/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files
$RUN $PY planner_poc/demo_narration.py ingest --file demo_explanations.md     # hygiene: 2/8 -> 8/8 narration.json
# new planner_poc/maneuver_router.py: reconstruct states -> null-AUC (base_dit_perdim core) + evoc-ratio (duck_probe core) + emu outcome-Δ
$RUN CUDA_VISIBLE_DEVICES=0 $PY planner_poc/maneuver_router.py --game smw \
   --maneuvers duck,spinjump_rex,grab_mesh --states-per 6 --k 8 --cfg 8 --out $F/router_r10.json
```
Reuses `base_dit_perdim.auc()`, `demo_bc.duck_probe()` (generalize `DUCK_PLANS`→`MANEUVER_PLANS`), `snes_env.save_state/load_state`,
`env._var`. **Threshold for "addition required": AUC<0.6 AND evoc-ratio<2× AND emu outcome-Δ<+0.3** (exactly the seed's proposal, plus
the emulator outcome leg so outcome-maneuvers aren't misjudged on a button).

---

## Q2 — The consolidation loop (concretely): save-state RWBC with a NARRATION-CONDITIONED actor — NOT a new objective

**Pick ONE form: it is save-state Reward-Weighted BC with a narration-conditioned actor — the R9 KL-anchored demo-fit with two swaps:
(a) targets come from *self-generated winning rollouts* instead of human demos; (b) conditioning text comes from learn-mode *learnings*
instead of a fixed terse plan.** No separate consolidation loss — R9 already proved plain-BC-overwrite collapses and the KL-anchor fixes
it taxonomy-free; a bespoke loss would re-open the trust-region problem we solved.

**learn-mode is the DISCOVERY substrate, not the weight-update substrate.** `game_planner.ClosedLoopPlanner(mode='learn')` already does
inference-time episodic memory (System-2 reviews System-1's execution, carries `<learnings>` bullets). It MINTS the System-2 hypothesis
("spin jump to kill the rex"); it does not itself touch weights. The consolidation step distills the *behavior those learnings produced*
into the bridge.

**Minimal runnable form (`demo_bc.py --consolidate`):**
1. Load a save-state (Q1's reconstructed Rex/mesh states are the seeds).
2. `ClosedLoopPlanner.plan()` in learn-mode → sample **K=8** candidate plans/learnings.
3. Roll each through the **FROZEN** DiT from the save-state; score by **RTG** (return-to-go, γ=0.95 — R9's credit-assignment fix) with
   the death penalty + progress, frame-exact from the emulator.
4. **KL-anchored (λ=0.3), plan-head-ONLY BC** on the top-quartile chunks, **conditioned on the learning text that produced them.**

**Mandatory guards (the R9 variance lesson — non-negotiable):** RWBC actor-adaptation on these envs is SEED-VARIANCE-DOMINATED (R9:
sign-flips across seeds, every variant). So consolidation must: (i) adapt **plan-head only**, never LoRA (plan-head demo-fit was the
robust leg; LoRA RWBC is the variance trap); (ii) **≥5 seeds + bootstrap CIs**; (iii) **RTG** scoring with death penalty (not local/raw
screen_x — that collapsed SMW); (iv) keep it **supervised-flavored** (BC-on-winners + KL-anchor), not policy-gradient. With these it is
exactly save-state GRPO-without-a-critic (rank K plans by reward, BC the winners) — the form the war-room converged on, made
collapse-safe by the KL-anchor.

**Tonight's scope:** DO NOT run the full closed loop tonight (too many moving parts + the variance trap). Q1 already executes steps 2–3
(mint plans, roll, emulator-score) for free. RUN the discovery+score half tonight (=Q1); DEFER the BC-on-winners half until Q4 validates
that narration-conditioning lands at all. This is a prerequisite ordering, not a punt.

---

## Q3 — Two-path bandwidth: the long path is a DiT-LoRA (reject K=32; FastPlanMod is a short-path amplifier, not the addition carrier)

**Recommend ONE: a dedicated rank-16 DiT-LoRA as the long-path/addition carrier; the short path stays the KL-anchored plan-head.**

- **K=8→32: DEAD.** R9 settled it — the base bridge does duck-on-command AT K=8, so bandwidth is not the bottleneck; and K=32 changes
  `resampler.queries`/`null_plan` shapes → btn_s600 `plan_head` won't load → forces a full Stage-1 retrain (not a one-night run). Reject.
- **FastPlanMod (zero-init adaln_proj, planner.py:305–345): the right carrier for added SHORT-PATH AUTHORITY, but NOT for addition.**
  It is a plan→`dit_temb` FiLM offset (null-masked → null-invariant *by construction*: null rows give a 0 offset). It can add advance/
  steering authority without touching the duck pathway — but a global FiLM gain on the conditioning **cannot synthesize a climb trajectory
  the DiT never represents.** It modulates the manifold; it doesn't expand it. PARK it as a short-path amplifier; pull it in only if
  evocation needs more authority than the K=8 cross-attn tokens give.
- **DiT-LoRA: the only candidate that modifies the action-generation manifold itself** (writes `model.*.attn.to_{q,k,v}.lora_{A,B}`,
  disjoint from `plan_head.*`). It is therefore the addition carrier. **Coexistence with the KL-anchored short path is PROVEN** — the R8/R9
  capstone merged pooled plan-head + Sonic-LoRA on one btn_s600 with **no interference** (disjoint params, all games retained/improved).

**The one real risk — null-invariance — and its fix.** LoRA is always-on, so a naively-trained added-capability LoRA *would* touch the
null path. Keep null exact by training it with the SAME masked-null discipline as the bridge: `plan_dropout` nulls the plan on a fraction
of chunks and the null target is base, so null rows get ~zero LoRA delta; then VERIFY with R9's action-level bit-identity check
(null chunk, fixed frame+seed, max|diff|=0.0). Train the LoRA ONLY on router-"add" chunks (grab-mesh), never on the broad demo set, so it
adds a capability rather than re-biasing advance. Net: short path (evocation, default, KL-anchored plan-head) + long path (addition,
DiT-LoRA on the "add" set only, masked-null) — additive, recoverable, null-invariant, merged disjointly. **Pick: DiT-LoRA.**

---

## Q4 — Wire in the gold narration (the cheap short-path test) — ENDORSE as companion, with a per-chunk-timing sharpening

The owner's `demo_explanations.md` is rich, per-second, owner-written ("00:14 duck", "00:16 spin jump on the rex to instantly kill",
"grab the mesh and climb"). `demo_narration.py` already frame-aligns it. Map each 18-step chunk → the narration text covering its frame
window; condition KL-anchored demo-fit on it (replacing terse BATTERY). New flag `demo_bc.py --gold-narration` (in `load_demo_chunks`,
attach `narration.json` text per chunk window as `s["plan"]`; `build_batch` already encodes per-sample plans → no batch change).

**Arms (3 seeds each, plan-head only, 600 steps, KL-anchor λ=0.3, 8 demo starts):** A0=terse (R9 winner, baseline) · A1=SIT
action-derived · **A2=gold-narration (NEW).**

**Sharpening — separate "richer text" from "correctly-TIMED text":** run A2 two ways: **A2-perchunk** (the maneuver text active in that
chunk's window) and **A2-global** (one demo-summary plan, a staleness control). The per-chunk version is the real test of "naming the
maneuver at the right frame evokes it"; the gap A2-perchunk − A2-global isolates timing from richness.

**Prediction:** A2-perchunk **beats** A0/A1 on (b) the expressiveness battery and (c) the maneuver-evocation probe, but is **within-CI of
A0** on (a) Δ_plan(advance). Rationale: Δ_plan measures advance, which terse already nails (+45.1, R9); the gold narration's marginal value
is the *situational* maneuvers it NAMES (duck/spin-jump/grab) that terse never does — so its lift appears in battery-R and per-maneuver
evocation, not raw advance. A2-global ≈ A0 (one summary plan ≈ constant plan = the R9 collapse precursor; the KL-anchor saves it from
collapse but it won't add timed maneuvers).

**Metrics:** (a) Δ_plan from fixed demo starts (bootstrap CI); (b) `expressiveness_battery.py` R (=mean clip(d_i/b_i,0,1)); (c) the
NEW maneuver-evocation probe (extend `duck_probe`/the battery with spin-jump→Rex-kill-Δ via emulator and grab-mesh→UP-hold+screen-Y climb).

**FALSIFIER for "richer real plans help":** A2-perchunk Δ_plan within the bootstrap CI of A0 **AND** battery-R within CI of A0 **AND**
spin-jump & grab evoc-ratio neither ≥2× nor > A0. ⇒ richer System-2 text does NOT become System-1-relevant via conditioning ⇒ the
limiter is the manifold, not the bridge ⇒ route to Q3 (DiT-LoRA addition) on the router-"add" chunks.

**Runnable (GPU1-3):**
```bash
for s in 0 1 2; do $RUN CUDA_VISIBLE_DEVICES=$((s+1)) $PY planner_poc/demo_bc.py --game smw --train plan_head \
   --gold-narration --kl-anchor 0.3 --steps 600 --seed-offset $s --save-delta $F/r10_gold_s$s.pt --duck-probe; done
# baselines = reuse R9 deltas r9_smw_kl_s* (A0) and r9_smw_situ_s* (A1)
$RUN CUDA_VISIBLE_DEVICES=3 $PY planner_poc/expressiveness_battery.py --game smw \
   --deltas r10_gold_s0.pt r9_smw_kl_s0.pt r9_smw_situ_s0.pt --maneuvers duck,spinjump_rex,grab_mesh --tag r10
```

---

## Q5 — Super Metroid into the POOLED generalist (training-only): ADD it, but ONLY under per-chunk situational/narration conditioning

**Verdict: HELPS under the KL-anchor + per-chunk situational plans; HURTS only under a global "advance" plan (which is the wrong
conditioning anyway).** R9 nailed the mechanism: collapse/off-axis drift comes from a CONSTANT plan. Super-Metroid (an exploration game,
not a right-runner) under a global "advance" plan would inject left/up/down/morph/aim directional noise that, averaged with the
platformers' "advance," pulls the shared advance policy off-axis — exactly the seed's worry, but ONLY in the constant-plan regime. Under
**per-chunk action-derived SIT** (extend `label_chunk` with aim_up/morph_down/shoot/explore_left) or **gold/VLM narration**, each
Super-Metroid chunk is conditioned on its OWN action, the plan→action map stays consistent, and the KL-anchor preserves the base manifold.
Super-Metroid then becomes POSITIVE evidence for evocation — it exercises left/up/down/aim/morph contrasts the platformers don't, widening
the battery's coverage.

**Plan conditioning (ranked):** Tier-0 = action-derived SIT (free, generalizes, the R9 mechanism) — primary. Tier-1 = frozen-VLM grounded
objective (`objective_label_demo.py`) for the situation ("door ahead → shoot", "ledge above → morph-jump"). NOT a global advance plan.

**Success gate (training-only, eval deferred — no progress-addr):** the PLATFORMER expressiveness battery R must NOT regress below the R9
KL-anchor PASS (R≥0.83) when Super-Metroid is pooled (Super-Metroid must not corrupt SMW/MMX/SMB advance or duck), AND new
Super-Metroid-specific contrasts (aim-up, morph-down) should appear. If platformer R drops <0.83 → it hurt → revert to platformers-only.
Keep short: pool via `pooled_demofit.py` with the per-chunk labeler + `--kl-anchor`; defer Δ_plan eval until a Super-Metroid progress var
exists.

---

## Steelman of GPT-5.5's likely pick + the single highest-value experiment tonight

**GPT-5.5 will likely pick "probe/router FIRST, and trust only the emulator counterfactual."** It has been the audit-the-metric / reward-
is-the-crux voice (R4 flagged the Sonic screen_x ceiling-hack; demanded multi-component reward before any plan-head adaptation). It will
argue: (1) classify before you build — don't wire gold narration into training until the router proves which maneuvers are even evocable;
(2) the gold narration is owner-written prose, NOISIER than terse — it risks re-introducing constant-plan / label-noise failure, so the
behavioral battery is not enough; the **frame-exact emulator P(kill|plan)−P(kill|null)** is the only hack-proof metric; (3) consolidation
must be save-state GRPO with a **multi-component RTG reward + ≥5 seeds**, or it inherits the R9 variance trap.

**This is largely RIGHT, and I concede:** Q1 (router) MUST gate Q3 — do NOT build the DiT-LoRA long path until the router confirms
grab-mesh is genuinely "add"; and the emulator counterfactual IS the decisive metric for outcome-maneuvers (I've folded it into the Q1
gate above). Where I push back: **Q1 alone produces no model tonight, and we have 4 GPUs.** Don't serialize. Run Q1 (frozen, GPU0) AND Q4
(KL-anchored gold-narration demo-fit, GPU1-3) IN PARALLEL — Q1 classifies, Q4 attempts the short-path fix, and they inform each other by
morning (Q1's grab-mesh "add" verdict predicts exactly whether Q4 could possibly evoke grab-mesh; if Q4 fails to evoke a maneuver Q1
labeled "add," that's CONFIRMATION, not a surprise).

**Single highest-value experiment tonight = the maneuver ROUTER (Q1), with the gold-narration KL-anchored demo-fit (Q4) as its free
parallel companion.** The router is the round's actual question (validate the framing), it's zero-training-risk, it answers the owner's
literal new questions from save-states we already have, and it routes everything downstream (evoke→short path we proved; add→long-path
DiT-LoRA). Q4 is the cheapest possible test that System-2 text becomes System-1-relevant, and it costs nothing extra on the idle GPUs.
**Combined pass bar:** duck=EVOKE & grab-mesh=ADD (controls land) · spin-jump kill-Δ reported · A2-gold Δ_plan ≥+20 (3/3, CI>0) · null
bit-identity 0.0 · battery R≥0.83 · ≥1 maneuver where gold > terse. Miss the last clause on every maneuver ⇒ falsified ⇒ the limiter is
the manifold ⇒ next night = DiT-LoRA addition on the router-"add" set.

---

### Grounding (repo anchors, all runnable)
`base_dit_perdim.py` (null-AUC core, the reflexive screen) · `demo_bc.py`:183 `duck_probe`/`DUCK_PLANS` (evoc-ratio core, generalize to
`MANEUVER_PLANS`), :259 `--kl-anchor` (R9 winner, λ=0.3), :104 `load_demo_chunks` (attach gold/SIT plan per chunk), :210
`build_anchor_pool` · `demo_narration.py ingest` (full demo_explanations.md → frame-aligned narration.json; 2/8→8/8) ·
`expressiveness_battery.py` (battery R; add spin-jump/grab-mesh contrasts) · `snes_env.save_state/load_state` + `env._var` (frame-exact
counterfactual; replay demo.npz actions to a narrated frame to reconstruct a state AT a Rex) · `planner.py`:305–345 `adaln_proj`
FastPlanMod (PARKED short-path amplifier) · `lora.py` (long-path DiT-LoRA carrier) · `game_planner.ClosedLoopPlanner` learn-mode (Q2
discovery substrate) · `pooled_demofit.py` (Q5, per-chunk labels + KL-anchor).
Constraints honored: ONE generalist (per-chunk labels + pooled, not per-game weights), VLM frozen-first (Tier-0 labels need no VLM; LoRA
optional), exact null-invariance (plan-head + masked-null short path; masked-null discipline + bit-identity check for the LoRA long path),
additive/recoverable deltas, GPU-frugal (frozen Q1 + 600-step Q4), no commits.
