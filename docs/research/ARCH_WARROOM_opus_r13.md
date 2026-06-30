# ARCH WARROOM R13 — Opus-4.8: discovery→instinct is a CHANNEL-MOVE, not a policy-overwrite. The construction-safe consolidation target is the ALWAYS-ON game-context channel, NOT the null.

*My lens this whole project: make the property hold BY CONSTRUCTION, not by training luck. masked-null gives exact null-invariance by construction; KL-anchor's functional-L2-to-base preserves the whole action space by construction (P5). R13's north-star ("discovery becomes instinct") has a buried construction trap — **"consolidate the skill into the null policy" and "masked-null exactness" are the SAME object, so you cannot do both.** The resolution is not a better optimizer; it is choosing the right surface. I resolve it below and build the entire recursion on it.*

---

## TL;DR

- **The crux (Q2):** "evocation → instinct" literally = self-distilling the plan-conditioned policy into the *unconditioned* policy on the skill's states. But our unconditioned policy is the **masked-null base** (`apply_null_mask` zeroes the K `_PLAN_TOKEN` positions → bit-identical to the frozen DiT, `nitrogen.py:594-605`). Distilling instinct *there* means changing the frozen DiT's image-only forward → **masked-null exactness dies** (the action-bit-identity-vs-base check goes nonzero). **You cannot consolidate into the null and keep null==base; they are one object.**
- **The construction-safe fix:** consolidate into the **always-on game-context channel** — the `_GAME_ID_TOKEN` embedding (`nitrogen.py:264-269, 477-479`), a *shared table indexed by game-id* that `apply_null_mask` **never masks** (it only masks `_PLAN_TOKEN`, `nitrogen.py:602`). The skill then fires **without the plan token** (instinct ✓, frees bandwidth ✓) while the K plan tokens stay cleanly maskable → **plan-relative null-invariance holds EXACTLY by construction** (masking the K tokens is a no-op vs the game-context policy; CFG plan-guidance still extrapolates along the K tokens). What we deliberately relax is *which* policy is the CFG reference: null moves from "pretrained-base" to "base+game-instincts." That redefinition is exactly what "instinct" means.
- **Q2 ranking (construction-safety first):** **(a) amortized KL-anchored self-distillation into the always-on game channel > (c) skill-token codebook (the compression/Q3, also null-safe by construction) > (b) S2-written fast-weights (best for Q4 execution-nuance, worst for instinct permanence + null-exactness).** P5 (KL-anchor adds-without-forgetting, R=0.83, only PASS) is what makes (a) safe; masked-null + the disjoint always-on row is what makes it *exact*.
- **This-week experiment (Q5):** **CONSOLIDATE-DUCK.** Evoke duck (P1, R9 `--duck-probe`), verify the bullet-state (`maneuver_router.py` DUCK_FRAMES save-states), KL-anchored self-distill the plan-conditioned duck into a **zero-init game-id instinct residual**, then show the **null policy ducks at bullet-states without the plan token** and the **plan-marginal duck → 0** ("S2 stops naming it"). New flag `--consolidate game_id` in `demo_bc.py` (≈30 lines, reuses the KL-anchor pool + `duck_probe`). 3 seeds; bars + falsifiers below.
- **Highlight (Q6):** lead with **P1→P5 as the recursion's first hop** — P1 = S2 evokes a skill S1 owns (the "discovery"), P5 = consolidate-without-forgetting by construction (the "becomes instinct") — supported by P3/P4 (it transfers across games and merges as ONE generalist). Honest caveats: oracle-plan, ≤16-chunk horizon, fixed save-states, closed-loop drift unproven, and the always-on-channel consolidation is **proposed, not yet run.**

I **disagree** with the seed's framing that consolidation is "just self-distillation into the null policy" (it is self-distillation, but into the *always-on* channel, not the null — the seed's own parenthetical is right and I make it load-bearing). I **expect to disagree with GPT-5.5** on tonight's pick (it will again push the Rex *addition*; I argue the consolidation *mechanism* is the higher-leverage, construction-decidable result and is the literal north-star).

---

## P1–P7 verified against `files/DISCRIMINATOR_RESULTS.md` (read, not asserted)

| P | claim | anchor | verified value |
|---|---|---|---|
| P1 | **evocation works** (S2 summons a primitive S1 owns) | DISCRIMINATOR:497-508 | BASE DOWN-dpad **1.2→3.4→10.8%** (9×) as the plan names duck explicitly; POOLED demo-fit **0%** (collapse, see P5) |
| P2 | **addition needed for OUTCOMEs** | DISCRIMINATOR:606-622 | spinjump_rex: JUMP **evoc_ratio 7.24** (button evocable) BUT survive+advance **0.04→0.07** (both near chance) ⇒ ADD at the outcome level |
| P3 | **positive cross-game transfer** | DISCRIMINATOR:357-375, 423-431 | ONE pooled plan-head: SMW **+26**, MMX **+31** (3/3, CI>0, null 0.0); SMW **1.3×** vs solo (pooling HELPS SMW); MMX new-game **+117%** |
| P4 | **no-interference merge** | DISCRIMINATOR:324-343, 384-397 | combined SMW **+71.2** ≥ smw_only +70.2; Sonic **+333**; 4 games retained/improved; disjoint `plan_head.*` vs `lora_*`; null exact |
| P5 | **consolidation WITHOUT forgetting** | DISCRIMINATOR:536-563 | KL-anchor Δ_plan **+45.1** (3/3), expr-battery **R=0.83 (only PASS)**, anchor loss **~0.001 (non-distorting)**; plain-BC control R=0.165 (kills all) |
| P6 | **staleness is TEXT-bound** | DISCRIMINATOR:566-604 | tok-drift **~0.0007-0.0012**; fresh≈cached≈null (**44/41/44**) ≪ oracle **65**; LIVE re-planning HURTS (MMX **−11**) |
| P7 | **demo-fit fixes the live-plan gap** | DISCRIMINATOR:444-475 | BASE live SMW **−10.7 (HURTS)** → demo-fit **+56.3** (≈ oracle +62.7); all 3 games (MMX +61.5, SMB1 +53.3) |

Architecture facts I re-derived from code (these decide the construction arguments):
- **masked-null** = `apply_null_mask` clears only `_PLAN_TOKEN` positions for dropped rows → exact base (`nitrogen.py:594-605`; `null_mode='masked'` `planner.py:39`).
- **`_GAME_ID_TOKEN`=6 is ALWAYS-ON:** never masked (apply_null_mask touches only `_PLAN_TOKEN`=7, `nitrogen.py:602`); `game_embedding = nn.Embedding(len(game_mapping), vision_hidden, padding_idx=0)` with **"0 = unconditional"** (`nitrogen.py:264-269`). A **shared table indexed by game-context** — exactly the "single hypernet/codebook indexed by game-context, not separate weights" the owner allows.
- **`plan_adaln`/FastPlanMod is PLAN-gated** (`adaln_cond(plan_tokens, plan_dropped)`, `nitrogen.py:673`; "null-masked → base-exact" `planner.py:47`) → it is a *second PLAN channel*, NOT always-on. (Important: this is the Q4 execution-authority channel, NOT the Q2 instinct channel.)
- **Cross-attn LoRA is NOT null-exact** (perturbs image-token attention); "gate LoRA to plan-token key positions" = the named construction fix (`lora.py:8-12`).
- **Resampler already emits K·A tokens** (A blocks of K, per-example cursor selects block a, `planner.py:33,328-343`) — the latent slots a skill-token codebook reuses for free.
- **maneuver_router** computes reflex_rate (null-AUC), evoc_ratio, emulator_outcome (survive+advance ≥25px, no life lost); gate `evoc_ratio≥2 OR outcome_Δ≥0.3` (`maneuver_router.py:6-11,140`); `r_down`=dim22>0.6 | dim1>0.5, `r_jump`=dim18>0.5.

---

## Q1 — The evocable-vs-addition classifier + the exact decision boundary

**Mechanism.** Evocation and addition are not a label we assign; they are a *measurable property of the frozen DiT's manifold* at a state. The router already measures three things — make them a **3-feature, 2-stage gate** keyed off the right one:

1. **reflex_rate** = `r_skill(null)` (does S1 already do it unprompted?) — the **null-AUC**.
2. **evoc_ratio** = `r_skill(plan) / r_skill(null)` (does the plan SUMMON the *button*?).
3. **outcome_Δ** = `P(success|plan) − P(success|null)` from an emulator save-state (does the plan summon the *result*?).

**The decision boundary (sharpened from R10/R11, the router's own mis-route lesson):**

```
if reflex_rate ≥ 0.6:                          → ALREADY INSTINCT  (no-op; candidate to FREE from plan, see Q2)
elif maneuver is OUTCOME-typed:                 → decide on outcome_Δ ONLY
        outcome_Δ ≥ +0.30  → EVOCABLE (short path: KL-anchored plan-head)
        outcome_Δ <  +0.30 → ADDITION (long path: plan-token-gated DiT-LoRA)
else (PRIMITIVE/button-typed):
        evoc_ratio ≥ 2.0   → EVOCABLE
        evoc_ratio <  2.0   → ADDITION
```

The non-obvious, load-bearing rule: **for OUTCOME maneuvers, `evoc_ratio` is a TRAP.** spinjump_rex had evoc_ratio 7.24 (button strongly evocable) yet outcome_Δ +0.04 — the router's verdict-gate *over-weighted the button ratio and mislabeled it EVOKE* (DISCRIMINATOR:620). The boundary that routes effort correctly is: **a maneuver is "evocable" iff the property the owner CARES about is evocable.** For "press down" the property is the button (evoc_ratio decides). For "kill the Rex" the property is the world-state change (outcome_Δ decides). duck = evocation (P1, button is the goal); Rex-kill = addition (P2, the goal is the kill, and the kill is flat).

**Minimal online test (exists today, zero new tooling):** `planner_poc/maneuver_router.py --game smw --k 4` on `btn_s600`, reconstructing the save-state from `maneuver_router.py:reconstruct` (replay demo.npz to the narrated frame). One frozen pass yields all three features. Cost ≈ one router run (already done for duck+rex).

**Named falsifier for the boundary itself:** take **3 maneuvers of each predicted class**, route them, then *actually* try the cheap path (KL-anchored plan-head, R9 recipe) on the EVOCABLE-labeled ones and measure outcome_Δ-after. **FALSIFIER: an "evocable"-labeled maneuver whose outcome_Δ stays <+0.15 after a 600-step plan-head fit** ⇒ the boundary is mis-calibrated (evoc_ratio is over-predicting transferable capability) ⇒ raise the outcome_Δ gate or demote button-evoc_ratio to a *necessary-not-sufficient* pre-filter. This is decidable in one night on the existing router + `demo_bc.py`.

**Where I disagree with the seed:** the seed lists "null-AUC reflex + evoc-ratio + emulator outcome" as co-equal. They are not — **null-AUC and evoc_ratio measure DEFAULT REFLEX and BUTTON-LEVEL routing, neither of which is manifold membership for an outcome.** Only the emulator outcome_Δ is hack-proof for the maneuvers the owner actually wants (kills, dodges, dispatches). The router must *route on outcome_Δ for outcome maneuvers* or it will keep calling Rex "evocable."

---

## Q2 (CORE) — Persistent evocation → instinct: the null-invariance-vs-consolidation tension, resolved by construction

### The tension, stated exactly

"Make the evoked skill OWNED by S1 so S2 can stop naming it" = make the policy fire the skill **without the plan token**, i.e. change the **unconditioned** policy on the skill's states. Formally this is **self-distillation**: teacher `π(a | s, plan=skill_plan)` (the evoked behavior, P1) → student `π(a | s, ∅)` (no plan token), on the states where the skill should fire. **The seed asks: "is that literally self-distillation of (plan-conditioned → unconditioned)?" YES.** And: "does it preserve null-invariance?" **Here is the trap, made precise:**

> Our "unconditioned policy" is the **masked-null base**: `apply_null_mask` zeroes the K `_PLAN_TOKEN` positions and the DiT is frozen, so the null forward is **bit-identical to the pretrained DiT** (`nitrogen.py:594-605`; the project's action-level bit-identity check, DISCRIMINATOR:303-306). To make the null policy *duck*, you must change that image-only forward — i.e., put a non-plan-gated delta into the frozen DiT. **That delta also fires on every null example, so `null ≠ base` → masked-null exactness is destroyed.** Consolidating-into-the-null and masked-null-exactness are **the same object pulled in opposite directions.** No optimizer escapes this; it's structural.

### The construction-safe resolution: consolidate into the ALWAYS-ON game-context channel, not the null

The frozen DiT already has a conditioning surface that is **present in BOTH the null and plan paths**: the `_GAME_ID_TOKEN` embedding (`nitrogen.py:264-269,477-479`), which `apply_null_mask` **does not touch** (it masks only `_PLAN_TOKEN`, `nitrogen.py:602`). Move the instinct there:

- **Add a zero-init `InstinctResidual: game_id → R^{vision_hidden}`**, added at the always-on `_GAME_ID_TOKEN` position (a separate, disjoint param group; zero-init ⇒ identity at start, so the current `null==base` bit-identity check passes *at init* and migrates cleanly).
- **Self-distill into it (KL-anchored, P5):** teacher = `π(·|s, plan="press down to duck")` at duck-states **and** `π_base(·|s)` elsewhere; student = `π(·|s, game-context-on, plan tokens MASKED)`. The KL-anchor-to-base over a broad state×plan pool is precisely the "elsewhere = base" pin (so the always-on channel doesn't make the policy duck *everywhere* and tank advance). This is mechanically the R9 winner (`demo_bc.py --kl-anchor`), just writing the always-on row instead of `plan_head`.

**What is preserved BY CONSTRUCTION (the whole point):**
- **plan-relative null-invariance, EXACT.** Masking the K `_PLAN_TOKEN` positions is *still* a no-op relative to the game-context policy (the game-id token is unchanged by masking). CFG plan-guidance still extrapolates cleanly along the K maskable tokens. The action-bit-identity check survives — **redefined** to "masking the K plan tokens reproduces the game-context policy," which is true by construction because the only thing masking removes is the K plan tokens.
- **ONE generalist, no per-game weights.** It's a *row* of a shared table (≤1024 floats), the codebook the owner explicitly allows. The DiT stays frozen/shared.
- **No cross-game forgetting, BY CONSTRUCTION.** SMW's instinct lives in SMW's row; MMX's row is untouched; the frozen DiT is shared. Consolidating a skill in one game **cannot** overwrite another game's skills (disjoint rows + frozen backbone). This is the *structural* answer to the owner's "intra-genre interference."

**What we deliberately relax:** the CFG *reference point*. null was "pretrained-base"; it becomes "base + game-instincts." That is not a bug — **it is the definition of instinct.** CFG only needs a *stable, well-defined* unconditional anchor to extrapolate from (Ho & Salimans, arXiv:2207.12598); it never required that anchor be specifically the pretrained weights. We keep the exact, maskable K-token contrast; we move the baseline. (For an unseen/held-out game, `game_id=0` is the zero "unconditional" row → base-exact preserved → un-consolidated games still reproduce the original base by construction.)

### Ranking (a)/(b)/(c) — construction-safety first

1. **(a) amortized KL-anchored self-distillation into the always-on game channel — PICK.** It is the *only* candidate that achieves true instinct (fires without the plan token) AND keeps an EXACT (plan-relative) null-invariance by construction AND stays one generalist with structurally-disjoint per-game rows. P5 is exactly its no-forget guarantee in functional form (= CoTTA stochastic-restoration, arXiv:2203.13591). This is policy distillation (Rusu, arXiv:1511.06295) onto a frozen-backbone always-on code — the safest possible surface.
2. **(c) skill-token codebook — the COMPRESSION layer (Q3), complementary not competing.** A learned vocab of option-tokens that occupy the SAME maskable K (or K·A) plan-token slots → **null-safe by construction** (mask them → base). But a codebook does NOT free the bandwidth in the instinct sense — S2 still must *emit* the token. It's the bridge *representation* between transient evocation and the consolidation target: discover → mint a skill-token → (once it fires reliably and context-cued) **migrate it from the codebook into the always-on row** = (a). Ranked 2nd because it's load-bearing for Q3/Q5 but is compression, not consolidation.
3. **(b) S2-written fast-weights/hypernet — best for Q4 EXECUTION-nuance, worst for instinct.** Online per-episode weight writes are either (i) general → break masked-null, or (ii) plan-gated → null-safe but then NOT "without the plan token," defeating the instinct goal. Fast-weights also want *transience* (per-episode tweak), the opposite of *permanent* instinct. Reserve (b) for the FastPlanMod/`plan_adaln` execution-authority channel (Q4), keyed by game-context, plan-gated → null-exact.

**P5 most enables (a)** — directly. The seed asks "which does P5 most enable?" P5 *is* (a)'s safety proof: KL-anchor adds the new skill while functional-L2-to-base preserves the whole action space (R=0.83). Without P5, distilling into an always-on channel would re-trigger the plain-BC collapse (R=0.165, duck→0 for all plans) on the always-on surface. With P5, it adds-without-forgetting by construction.

### First experiment + named falsifier (the consolidation primitive in isolation)

This is Q5's experiment — see there for the full spec. The **falsifier for the whole "evocation→instinct" mechanism**: after KL-anchored self-distillation into the game channel, **if the null-path (no plan token) skill-rate does NOT rise toward the previously plan-evoked rate** (duck null stays ~base while plan still evokes), then the always-on channel cannot carry this instinct ⇒ the skill is not a per-game-constant policy bias (it's state-conditional in a way the game-id code can't express) ⇒ fall back to (c): keep it a skill-token S2 emits. Secondary falsifier: **battery R < 0.80 OR advance Δ_plan drops** ⇒ the "elsewhere=base" KL pin failed (instinct leaked into non-skill states) ⇒ raise λ or rebalance the distillation pool.

**Lit grounding:** this is option-discovery/consolidation in the Sutton-Precup-Singh (1999) sense (a discovered option becomes a callable primitive), realized as ReST-EM (Singh, arXiv:2312.06585: mint→roll→score→KL-anchored-BC, restart-from-base each round = our KL-anchor-to-base) with the *output surface* chosen for construction-safety. The "skill-library that grows and gets called at higher abstraction" is VOYAGER (Wang, arXiv:2305.16291), here grounded in a continuous always-on code rather than a text library.

---

## Q3 — The token-dilution wall, and a null-safe skill-token codebook

**The wall (real, measured):** image tokens swamp the K=8 plan tokens (EXP-050/051 image-overwhelm); live text churn HURTS (P6: live re-plan MMX −11). As games recurse, S2 must say MORE in the same K. Verbose prose is out (P6 text-quality is the bottleneck, and prose dilutes).

**The codebook beats prose, and stays null-exact BY CONSTRUCTION.** Mint a small **learned vocabulary of skill-tokens**, each token = a **K-token (or K·A-token) macro** the bridge expands into plan-token slots. Crucial construction property: the expanded macro lands in the **same `_PLAN_TOKEN` positions that `apply_null_mask` clears** → masking them → exact base → **null-invariance preserved with zero new machinery.** This is strictly better than widening K (8→32 dilutes the cross-attention and was rejected R10) because the codebook **compresses** (one discrete index → K learned vectors) rather than **lengthens**.

- **Why it beats prose (mechanism):** prose must be re-encoded by the frozen VLM every chunk (drift, P6) and competes with image tokens token-for-token; a skill-token is a *direct* learned write into the K slots — no VLM re-encode, no text drift, maximal information-per-slot. It is the "options discovered by S2" the owner names: S2 emits indices, not sentences.
- **Reuse existing structure:** the resampler already produces K·A tokens with a per-example cursor (`planner.py:33,328-343`). A codebook entry is a learned (A-block) macro; chaining `c1 c2 c3` = emitting three indices across A blocks — the owner's "chain combos, not press-A-then-B."

**Falsifier:** mint a 16-entry codebook, fit it (plan-head-only, KL-anchored) so each index reproduces a curated maneuver, then compare **maneuver-fidelity (router evoc_ratio / outcome_Δ) and Δ_plan of `[skill-token c_duck]` vs the prose "press down to duck"** at equal sequence budget. **PASS:** codebook ≥ prose on duck evoc_ratio AND ≥ on battery R AND null bit-identity = 0.0. **FALSIFIER:** codebook < prose on fidelity at equal budget ⇒ discrete macros lose the VLM's compositional grounding ⇒ keep prose for *novel* maneuvers, reserve the codebook only for *consolidated, high-frequency* ones (which then migrate to the Q2 always-on channel anyway). Tooling: `demo_bc.py` plan-head fit with a codebook embedding in front of the resampler; eval via `maneuver_router.py` + `expressiveness_battery.py`. **Defer** behind Q5 — the codebook only pays off once we have ≥a few consolidated skills to compress.

---

## Q4 — Subtly-different execution: shared concept channel + a router-gated per-state execution ADD, as ONE model

**The owner's structure ("broad concept holds, execution differs"), mapped onto our surfaces by construction:**

| layer | what | surface | transfers? | null-safe? |
|---|---|---|---|---|
| **concept** | "advance / jump / duck" (game-general) | generalist **plan-head** (pooled, P3/P4) | YES (P3: pooled SMW+26/MMX+31; cross-game lift P4) | plan-gated → exact |
| **instinct** | per-game *constant* bias the skill became | **always-on game-id row** (Q2) | per-game row, frozen backbone | plan-relative exact |
| **execution** | per-*state* motor nuance (spin-jump timing/sprite-interaction = the Rex ADD, P2) | **plan-token-gated DiT-LoRA** + `plan_adaln` FiLM, **gated to plan-token key positions** (`lora.py:8-12`) | router-gated, ADD-chunks only | **exact by construction** (null path has no plan tokens → zero contribution) |

**One model, no per-game weights — the construction that makes this legal:** every per-game thing is an *index into a shared structure*, never a separate network. The execution ADD is a **single hypernet/gate indexed by game-context** (the `game_id`/concept embedding conditions the LoRA gate), not a per-game LoRA. The router (Q1) decides *whether* the execution-add fires (outcome_Δ gate); the game-context decides *how* it's parameterized (shared hypernet). The DiT stays frozen and shared. This is Progressive-Networks-style add-without-forget (Rusu, arXiv:1606.04671: frozen columns + lateral connections) but with the "column" being a *shared, game-indexed* gated-LoRA rather than a new column per task — which is precisely how we keep ONE generalist.

**Disjoint-merge is already proven** (P4: plan_head.* ⊕ lora_* stack with no interference, even positive cross-transfer). So: concept (plan-head) ⊕ instinct (game-id row) ⊕ execution (gated-LoRA) are three disjoint param groups that merge into one btn_s600.

**Falsifier:** train the execution-add (gated-LoRA, Rex chunks, conditioned on the gold-action plan) and check **(i)** outcome_Δ on the held-out Rex save-states rises ≥+0.15 (3× the current +0.04) AND **(ii)** a DIFFERENT game's plan-advance and battery R are unchanged within noise (shared-hypernet didn't leak) AND **(iii)** null bit-identity = 0.0 (gating worked). **FALSIFIER on the "one-model" claim specifically:** if making the gated-LoRA game-indexed (shared) instead of SMW-only *degrades* the Rex outcome_Δ vs an SMW-only LoRA by >1 SE ⇒ the execution nuance genuinely needs per-game capacity the shared hypernet can't express ⇒ escalate the hypernet rank before conceding per-game weights. (I expect the shared hypernet to hold; P3/P4 show execution-direction capacity transfers.) **This is the long path — gated behind the Q5 consolidation demo and a sprite-status kill-detector (P2 caveat), not this week.**

---

## Q5 — The recursion with OUR tools: the MINIMAL this-week SMW demonstration (CONSOLIDATE-DUCK)

**The loop, one full turn, all existing tools:** evoke (P1) → verify with a save-state (router) → consolidate into the always-on channel (P5, Q2) → S2 stops naming it (measured) → bandwidth freed for a higher abstraction. duck is the right first skill: **evocable** (P1, 9×), **verifiable** (router DUCK_FRAMES bullet-states), and the **construction-safe target is identified** (game-id row, Q2).

**Step 0 — evoke + verify (today, frozen, zero training).**
```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
PY=.venv/bin/python ; F=/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files
# Confirm duck is evocable from generic starts AND measure the bullet-state null reflex (the consolidation target gap):
$ENVP $PY planner_poc/maneuver_router.py --game smw --k 4 --out $F/r13_router_duck.json   # reflex(null) vs evoke(plan)
```

**Step 1 — consolidate (the one new flag; ≈30 lines in `demo_bc.py`).** Add `--consolidate game_id`: trains a **zero-init `InstinctResidual(game_id)`** (added at the `_GAME_ID_TOKEN` position) by **self-distillation** — teacher action = the plan-conditioned chunk under `DUCK_PLANS[-1]` ("press down to duck") at duck-relevant frames (the existing `--duck-probe` frame set + router bullet-states); student = same frames with the **K plan tokens masked**; plus the existing `--kl-anchor` pool as the "elsewhere = base" pin. Everything else (KL-anchor machinery, duck_probe dim22/dim1, save-delta, survival_advance eval) already exists.
```bash
for s in 0 1 2; do g=$((s%4))
 $ENVP CUDA_VISIBLE_DEVICES=$g $PY planner_poc/demo_bc.py --game smw --consolidate game_id \
    --kl-anchor 0.3 --kl-anchor-n 32 --duck-probe --steps 600 --seed-offset $s \
    --save-delta $F/r13_consol_duck_s$s.pt ; done
```

**Step 2 — measure the recursion (frozen eval on the deltas).**
```bash
# (a) instinct: null-path duck at bullet-states WITHOUT the plan token (the headline);
# (b) "S2 stops naming": plan-marginal duck (plan-null) shrinks toward 0;
# (c) no-forget: expressiveness battery R; (d) advance retained; (e) plan-relative null bit-identity.
$ENVP $PY planner_poc/maneuver_router.py --game smw --k 4 --ckpt-delta $F/r13_consol_duck_s0.pt --out $F/r13_router_post_s0.json
$ENVP $PY planner_poc/expressiveness_battery.py --game smw --tag r13 --deltas $F/r13_consol_duck_s*.pt
```

**Metrics, bars, seeds (all on btn_s600; 3 seeds for the mechanism, ≥5 for any emulator-outcome claim, R9 discipline):**

| signal | tool | PRE | PASS bar (POST) |
|---|---|---|---|
| null-path duck @ bullet-states (no plan) | `maneuver_router` r_down(null) | ~0.46 (already reflexive there) → use **bullet-APPROACH** states where null is low | rises to ≥ plan-evoked rate at the matched states |
| **plan-marginal duck = r_down(plan) − r_down(null)** | router | + (plan needed) | **→ ≤ 0.25× of PRE** ("S2 stops naming it") |
| advance retained | `survival_advance` plan−null, 8 SMW starts | Δ_plan baseline | **≥ PRE − 1 SE** (no advance cost) |
| no-forget | `expressiveness_battery` R | base 1.0 | **R ≥ 0.80** (match P5's 0.83; NOT plain-BC 0.165) |
| **plan-relative null-invariance** | action-bit-identity (mask K plan tokens vs game-context fwd) | 0.0 | **= 0.00e+00 (exact, by construction)** |
| no cross-game leak | run held-out game `id`/`id=0` | base-exact | unchanged within noise (disjoint-row guarantee) |

**Then the higher abstraction (closes the loop, same week if Step 2 passes):** re-run the live-plan deployment probe (`staleness_probe.py live`, P7) with a plan that **omits duck entirely** ("advance, jump pits"). **PASS:** the consolidated model survives bullet-bills (null-duck instinct fires) at ≥ the duck-naming model's survival, while S2's plan budget now carries only the higher-level "advance" — *empirical proof that S2 stopped naming duck and the freed budget is usable.* This is the literal "discovery→instinct→S2 moves up a level."

**Named falsifiers (decisive):**
- **Null-duck does NOT rise** ⇒ duck isn't a per-game-constant bias the game-id code can carry ⇒ the always-on channel is the wrong surface for *this* skill ⇒ fall back to Q3 codebook skill-token (S2 keeps emitting it, but compactly). *This would falsify my Q2 pick for duck specifically; I'd report it honestly.*
- **R < 0.80 or advance drops** ⇒ the KL "elsewhere=base" pin failed; instinct leaked into non-bullet states (ducking when it shouldn't) ⇒ raise λ / rebalance the distillation states.
- **Cross-game leak** ⇒ the residual isn't properly game-indexed ⇒ fix the indexing before any multi-game claim.

**Cost:** 3 plan-head-scale fits + frozen evals ≪ one overnight A100 slice. No new env, no kill-detector, no RL. **This is the smallest thing that demonstrates the entire north-star loop and is construction-decidable.**

---

## Q6 — Highlight reel + honest framing

**Lead with P1→P5 as the recursion's first hop, framed as a matched pair:**
- **P1 (evocation works):** S2 summons a primitive S1 already owns — duck 1.2→10.8% on command (DISCRIMINATOR:497-508). This is the cleanest existence-proof of S2→S1 *knowledge transfer*: the plan token is a *pointer* into S1's manifold.
- **P5 (consolidate without forgetting, BY CONSTRUCTION):** KL-anchored functional-L2-to-base adds a skill while preserving the whole action space (R=0.83, only PASS; anchor loss ~0.001; DISCRIMINATOR:536-563). This is the "becomes instinct" half — and it's *construction-guaranteed*, not tuned.

Together they are the two ends of the owner's "discovery becomes instinct." **Support with P3/P4:** the transfer is not a one-game fluke — one pooled plan-head transfers across SMW/MMX/SMB1 (P3, SMW even BENEFITS from pooling 1.3×, new-game MMX +117%) and merges as ONE generalist with no interference (P4). That is the "shared capabilities across games" the owner wants, demonstrated.

**Why this lead over the alternatives:** P3 alone is the strongest *transfer* number, but it's data-level (concept) and doesn't show *instinct*; P2 (Rex) is a *negative* result (addition needed). The P1→P5 pair is the only thing that evidences the **full forward arc** (evoke → consolidate-safely) that R13 is a report about — and P5 is *my* construction lens delivering the no-forget guarantee the whole recursion needs.

**Honest caveats (state them up front, don't bury):**
1. **Oracle-plan.** Most Δ_plan evals use curated BATTERY/oracle text; deployment uses the live 2B. P7 shows demo-fit closes most of this gap (live −10.7→+56.3) but the residual oracle−live gap is real (+6.3 SMW).
2. **Short-horizon.** Rollouts are ≤16 chunks from save-states (DISCRIMINATOR:461). Closed-loop, minute-scale **drift is UNPROVEN.**
3. **Fixed save-states.** The router/consolidation verify on reconstructed states; absolute outcome rates are low and reconstruction-sensitive (Rex 3.75→7.5% on a hard 4-chunk scenario, DISCRIMINATOR:619). Any outcome claim needs ≥5 seeds and better reconstruction (the RTG single-seed artifact, DISCRIMINATOR:150-172, is the cautionary tale).
4. **The consolidation result is PROPOSED, not yet run.** Q2/Q5 are a construction *argument* + a falsifiable experiment, not a measured win. The honest framing: "P5 proves consolidation-without-forgetting on the plan-head; R13 proposes (and Q5 tests this week) that the *same* primitive, redirected to the always-on game channel, yields instinct while preserving plan-relative null-invariance by construction."
5. **Channel-measurement subtlety.** human DOWN→stick (dim22), base ducks via dpad (dim1); report per-channel or a combined metric mis-reads base duck (DISCRIMINATOR:560-562). Carry this into the Q5 bars.

---

## Where I disagree (seed + expected GPT-5.5)

**With the seed:** (1) "consolidation = self-distillation into the *null* policy" — half-right and the dangerous half: it IS self-distillation, but into the null it **breaks masked-null by construction**. The seed's own parenthetical ("maybe consolidate into an always-on game-context channel, not the null") is the correct answer and I make it the load-bearing mechanism, with the exact surface (`_GAME_ID_TOKEN`, never masked) and the redefined-but-exact invariant. (2) The seed treats null-AUC, evoc_ratio, outcome_Δ as co-equal router features (Q1); I argue **only outcome_Δ is hack-proof for outcome maneuvers** and the other two mis-route (the router already proved this on Rex).

**With GPT-5.5 (expected):** GPT-5.5 has reliably pushed the **emulator-outcome / capability-addition** frontier (caught the Sonic screen_x hack R4, forced ≥5-seed+RTG R9, exposed the RTG artifact). I expect its R13 pick to be **build the Rex addition now** ("the router found a real capability gap; addition is the actual frontier; consolidation just re-files capability S1 already has"). **Where I concede:** emulator outcome_Δ is the only hack-proof metric (it gates my Q1 and Q4); ≥5 seeds for any outcome claim; the Rex add IS the eventual long path (Q4). **Where I push back:** (a) the Rex outcome is near-chance for both arms (3.75→7.5%) on a reconstruction the router itself flags as low-SNR, and needs an **unbuilt sprite-status kill-detector** (P2, DISCRIMINATOR:608) — that's multi-night and artifact-prone *now*. (b) The owner's north-star is **discovery→instinct→higher discovery**, and the *instinct* hop is exactly the thing we have NOT yet demonstrated and CAN this week, construction-decidably, on validated infra (P5 KL-anchor + duck_probe). (c) Consolidation is **upstream of the generalist** the Rex-LoRA would merge into: if we can't safely turn an evoked skill into instinct without breaking null-invariance, every downstream add inherits an unstable base. **Correct sequence:** CONSOLIDATE-DUCK this week (cheap, decisive, the literal north-star) → sprite-status detector → Rex execution-add with ≥5 seeds + outcome_Δ. I'll also note our **convergence**: gold-action plans (R12) are the shared label source for BOTH the consolidation teacher and the eventual Rex-add conditioning, so this isn't either/or — it's sequencing the *decidable, construction-safe* result first.

---

### Repo anchors (all runnable today)
`nitrogen/flow_matching_transformer/nitrogen.py:594-605` (masked-null: only `_PLAN_TOKEN` cleared) · `:264-269,477-479,602` (`_GAME_ID_TOKEN` always-on shared embedding = the consolidation target) · `:673` (`plan_adaln` is plan-gated → the Q4 execution channel, not the Q2 instinct channel) · `nitrogen/planner.py:33,328-343` (resampler K·A latent slots for the Q3 codebook) · `:39,47` (`null_mode='masked'`, `plan_adaln` null-masked) · `lora.py:8-12` (gate LoRA to plan-token keys → null-exact ADD, Q4) · `planner_poc/maneuver_router.py:6-11,140` (reflex/evoc_ratio/outcome_Δ gate, Q1) · `planner_poc/demo_bc.py` (`--kl-anchor`,`--kl-anchor-n`,`--duck-probe`,`--save-delta`,`--seed-offset`; NEW `--consolidate game_id`, Q5) · `planner_poc/expressiveness_battery.py` (R, the no-forget guard) · `planner_poc/eval_common.py:48 survival_advance` (death-aware) · `planner_poc/staleness_probe.py live` (no-oracle deployment metric, Q5 closeout).

### Constraints honored
ONE generalist (game-indexed shared rows/hypernet, never per-game weights) · **plan-relative null-invariance EXACT by construction** (always-on channel leaves the K `_PLAN_TOKEN` mask contract intact; the relaxation — null = base+instinct — is named explicitly) · VLM frozen-first (Qwen-LoRA reserved for the Q3/text-quality successor) · ≥3 seeds + bootstrap (≥5 for any emulator-outcome claim) · survival-aware death-aware eval · GPU-frugal (Q5 = 3 plan-head-scale fits + frozen evals) · no relitigating P6 (staleness text-bound), RWBC-for-plan-OOD-collapse, or P1 (evocation works) · kill by numeric PID; no commits.
