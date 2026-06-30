# ARCH WARROOM R11 — Opus-4.8: LEAN ON THE IMPLICIT TTT (the continuous short path), DON'T BOLT ON A WEIGHT-TTT LOOP

*Lit-grounded round. Every call cites a paper from LIT_R11_{ttt,distill,fastweights}.md. arXiv IDs are the ✓-verified ones.*

## TL;DR — the design call
**The literature says our short path is ALREADY test-time training; the right move is to lean on it (forward-pass), not bolt on
explicit weight-TTT.** von Oswald 2023 (arXiv:2212.07677) proves cross-attention over in-context tokens = implicit gradient descent;
Schlag 2021 (arXiv:2102.11174) shows the resampler→K-token write + DiT cross-attn read IS a fast-weight programmer. So the K=8 plan
tokens are an implicit per-step optimizer over the frozen DiT. Therefore:
- **Q1 (stale plans): REJECT weight-TTT on the actor** — TENT/SAR entropy-min (arXiv:2006.10726 / 2302.12400) is undefined for our
  continuous flow field (the lit's own confirmed gap) AND its documented collapse mode (TENT §e, SAR §a) is exactly the R9 duck→0%
  collapse. PICK a **zero-training forward-pass plan-token TTA**: re-encode the K tokens on the *fresh* frame every chunk (Gandelsman
  2023 video-locality, arXiv:2307.05014) + EMA/aug-average across replans (CoTTA 2203.13591 + MEMO 2110.09506). Collapse-immune (no
  optimizer).
- **Q2 (consolidation): it is ReST-EM** (Singh 2023, arXiv:2312.06585), not STaR, not literal ExIt. Borrow restart-from-base + reward-τ.
- **Q3: LEAN ON the continuous path.** Our R9 KL-anchor is literally CoTTA stochastic-restoration (arXiv:2203.13591) in functional form.
- **Q4 (addition): DAgger (arXiv:1011.0686, O(εT²) covariate shift) + plan-token-GATED DiT-LoRA + EATA Fisher guard (arXiv:2204.02610).**
- **Q5: a sleep-time amortized ReST-EM loop over a fast-weight short path; Fisher-guarded gated-LoRA long path; forward-pass TTA for
  staleness. No online weight-TTT on the actor.**

**ONE experiment tonight (runs alongside the already-queued R10 maneuver-router; both frozen, ~1 GPU each):** the **staleness forward-pass
TTA test** on the existing `planner_poc/staleness_probe.py` — add a `fresh` mode (hold the t=0 VLM *text*, re-encode the K plan tokens on
the live frame every chunk) and a `fresh_ema` mode (+EMA pooled token across replans + M=4 aug-average). This is the cleanest possible
falsification of von Oswald's "the short path is already TTT" claim, on the owner's stated #1 bottleneck (50–60% replan-flip), at ZERO
training risk.

**Pass bar:** `fresh` recovers ≥50% of the (oracle−stale) advance gap on SMW **and** ≥1 other plan-OOD game (MMX/SMB1; SMW & MMX are
where DISCRIMINATOR proved the plan is load-bearing); `fresh_ema` cuts the per-replan action-direction **flip-rate ≥30% relative** vs
`live` with advance(fresh_ema) ≥ advance(live); **null action-bit-identity = 0.0**; **≥3 seeds** for VLM-text variance.
**Falsifier:** `fresh ≈ stale` ⇒ staleness lives in the TEXT, not token-staleness ⇒ von Oswald's re-encode-as-GD does NOT help here ⇒
escalate to replan-hysteresis on text + GPT-5.5's emulator-supervised weight-TTT (Q1 Tier-2, ≥5 seeds, KL-anchored).

---

## Q1 — TTT for stale plans / online adaptation: PICK forward-pass plan-token TTA; REJECT actor entropy-TTT

**The seed offers (a) TTT-layers fast/slow, (b) TENT/SAR entropy-min, (c) Akyürek per-episode LoRA. I reject all three as the tonight
weight-update and pick a fourth: forward-pass token TTA.** Reasoning, each lit-grounded:

**REJECT (b) TENT (arXiv:2006.10726) / SAR (arXiv:2302.12400) entropy-min on the actor — two independent killers.**
1. *Continuous-output gap (the lit's confirmed gap, ttt §Gaps-1).* The DiT emits a continuous flow velocity, not a softmax; Shannon
   entropy is undefined. Redefining via Gaussian/KDE entropy is unproven for flow-matching (no paper). MEMO's marginal entropy
   (arXiv:2110.09506) has the same problem.
2. *Collapse = R9.* TENT §e and SAR §a both document entropy-min collapsing to a single output under single-sample online batches —
   which is *literally* our deployment (batch=1 per chunk) and *literally* the R9 failure (demo-fit collapsed duck 10.8%→0.0%,
   DISCRIMINATOR:504). SAR's SAM-flat-minima + grad-filtering is a real mitigation, but it is still a self-generated-signal weight update
   = the seed's own warning ("ANY test-time weight update inherits R9 collapse"). **I am contradicting the LIT_R11_ttt.md headline
   recommendation (Recipe A = "SAR-style LN adaptation, strongest P1 match") using the lit's own caveats + R9.** Park SAR as the *only*
   acceptable weight-TTT IF we ever do one (Q1 Tier-2), never as the default.

**REJECT (c) Akyürek per-episode LoRA (arXiv:2411.07279v1) for STALE PLANS** (it's the right tool for Q4/ADDITION, wrong for staleness):
needs labeled (frame,action) pairs + 100–500 LoRA steps, and ordinary LoRA breaks exact null-invariance (`lora.py:9` — affects image-token
attention). The ARC gains also lean on a *synthetic-task pretrain* slow-weight prior (Akyürek §e) we don't have for maneuvers.

**REJECT (a) TTT-Linear (arXiv:2407.04620) for tonight** — it's the principled long-term answer ("plan tokens as a fast-weight hidden
state that carries over"), but it requires replacing the Perceiver resampler with a TTT-Linear layer + the JAX dual-form custom kernel
(ttt §Gaps-6) and an end-to-end retrain that breaks `btn_s600` shapes. Not a one-night run. Roadmap it.

**PICK: forward-pass plan-token TTA (no optimizer → collapse-immune).** The self-supervised test-time signal in OUR setting is NOT entropy
— it is **temporal/augmentation consistency of the plan-token encode**, plus (Tier-2) **emulator forward-consistency**. Two moves:
1. **`fresh` (Gandelsman locality, arXiv:2307.05014):** hold the VLM *text* plan, but re-encode the K=8 tokens through
   `pl.encode_multimodal([live_frame], text)` **every chunk** on the fresh frame (the 18-frame chunk ≈ the paper's optimal ~2 s sliding
   window). von Oswald (arXiv:2212.07677) predicts this re-grounding IS an implicit GD step → it should recover staleness *without*
   regenerating (and thus flipping) the text.
2. **`fresh_ema` (CoTTA aug-pseudo-label, arXiv:2203.13591 §b + MEMO consistency, arXiv:2110.09506):** EMA the pooled plan token across
   replans (`z̄_t = β z̄_{t-1} + (1−β) z_t`, β≈0.7) and average tokens over M=4 frame jitters. Smooths the 50–60% flip in token space.

**Runnable tonight (zero training, frozen `btn_s600`; extends existing harness):**
```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
PY=.venv/bin/python ; F=/home/t-nagupta/.copilot/session-state/dddebd2a-be84-47f6-814d-d5e5cf9059b5/files
# add modes {fresh,fresh_ema} to staleness_probe.rollout() (it already has null/oracle/live/stale + replan_every);
# report advance (env._var) AND a per-replan action-direction flip-rate over stick+jump argmax.
for g in smw mmx; do for s in 0 1 2; do
  $ENVP CUDA_VISIBLE_DEVICES=$s $PY planner_poc/staleness_probe.py --game $g --seed $s \
     --modes null oracle live stale fresh fresh_ema --ema 0.7 --aug 4 --out $F/r11_stale_${g}_s$s.json ; done; done
```
**Metric:** staleness cost recovered = (advance(fresh)−advance(stale))/(advance(oracle)−advance(stale)); flip-rate(live) vs
flip-rate(fresh_ema); Δ_plan preserved; null bit-identity (reuse `demo_bc.py` null_ref check, must = 0.0).
**Pass:** fresh recovers ≥50% of (oracle−stale) on SMW+MMX; fresh_ema flip ≤0.7×live; advance(fresh_ema) ≥ advance(live); null 0.0.
**Falsifier (named):** fresh ≈ stale on both games ⇒ staleness is TEXT-bound ⇒ re-encode-as-GD insufficient ⇒ Tier-2.

**Q1 Tier-2 (gated, only if Tier-1 fails; this is where I CONCEDE to GPT-5.5):** an *emulator-supervised* per-episode adaptation — NOT
entropy-TTT, because we have the frame-exact emulator as a real (non-collapsing) signal. Sample K plans/encodes from the paused
save-state, rank by emulator forward-consistency (advance+survival), BC the winner into the **plan-head only** with `--kl-anchor 0.3`
(=CoTTA functional restore, Q3) — this is Akyürek's per-task adaptation (arXiv:2411.07279) with *emulator-reward filtering replacing
labels* and *plan-head replacing LoRA* (so null stays exact). Mandatory R9 guards: ≥5 seeds, RTG+death, supervised-flavored. This is
identical to the Q2 M-step → do it as Q2, not a separate mechanism.

---

## Q2 — The consolidation loop = ReST-EM (arXiv:2312.06585), not STaR, not literal ExIt

Our loop (save-state → VLM mints K plans → roll frozen DiT → keep top-RTG → KL-anchored plan-head BC on the winning plan) maps cleanly:

| precedent | fit | verdict |
|---|---|---|
| **ReST-EM** (Singh 2023, arXiv:2312.06585) | E=generate+score from save-states (Grow); M=SFT on filtered top-τ (Improve); EM w/ Jensen lower-bound | **EXACT MATCH** (distill §5d: "essentially our loop exactly described") |
| ExIt (Anthony 2017, arXiv:1705.08439) | needs a *search expert* emitting improved action **distributions** (MCTS visit counts) as soft targets | structural template only — we have no tree; the VLM plan is "a much weaker oracle" (distill §1e) |
| STaR (Zelikman 2022, arXiv:2203.14465) | binary-correctness filter, fine-tune **full** model on (problem,rationale,answer) | partial — we fit plan-head only; RTG is a noisy proxy not objective correctness |

**BORROW these concrete ReST-EM/ReST recipe details:**
1. **Restart-from-base each round** (ReST-EM §a; Gulcehre ReST 2308.08998 §d: "freshly sampled on-policy data crucial each round").
   Our KL-anchor-to-**base** (not to last round) IS this anti-drift restart, in gentler functional form — keep anchoring to the original
   `btn_s600` plan_head every round, never to the previous round's delta.
2. **Reward threshold τ** (ReST 2308.08998 §d): τ too low → trains on mediocre, slow; τ too high → data starves, stalls. Use top-quartile
   (τ=p75) AND require ≥1000 winning chunks/round (distill §19 "target ≥1000 successful episodes per M-step").
3. **Soft targets, not argmax** (AlphaZero, distill §2e): BC-on-winners is temperature-0 distillation → narrows the policy. Mitigant: the
   KL-anchor preserves base-distribution breadth (this is *why* R9 kept battery R=0.83 instead of collapsing to right+jump). So
   "BC-on-winners + KL-anchor" ≈ poor-man's full-softmax distillation.
4. **STaR rationalization trick** (arXiv:2203.14465 §d) — for the counterfactual-override case only: when a save-state attempt fails,
   condition the VLM on the *desired outcome* ("you needed to kill the Rex") and ask for the plan → a synthetic winning (plan, action)
   pair even with zero successful rollouts.

**What the papers predict vs our guards (they line up ~1:1):**
| predicted failure | source | our R9 guard | match |
|---|---|---|---|
| reward-hacking | STaR/ReST/ExIt | RTG+death-penalty + VLM-judge ("did it follow the plan?", distill §5d) | confirmed by the real Sonic screen_x ceiling-hack, DISCRIMINATOR:26 |
| distribution narrowing | AlphaZero/BC-theory | KL-anchor + τ≤p75 + diverse save-states | ✓ (R9 R=0.83) |
| covariate shift O(εT²) | DAgger 1011.0686 | collect winners at policy-visited save-states | ✓ (see Q4) |
| on-policy PG instability | ReST-EM (why EM>PPO) | supervised-flavored BC-on-winners, not policy-gradient | ✓ (R9: PG variants all collapsed) |

**Minimal runnable tonight = the E-step only (the M-step is gated by the router, exactly as R10).** The R10 maneuver-router already mints
K plans per save-state, rolls the frozen DiT, and emulator-scores them — that IS ReST-EM's E-step/Grow, for free. The M-step (KL-anchored
plan-head BC on top-quartile winners, conditioned on the **VLM-abstracted** plan — not raw narration, per owner) is DEFERRED until the
router confirms which maneuvers are evocable: ExIt §b warns that if the VLM oracle is no better than chance, "distillation teaches noise."
So: run E+score tonight (=router); run M only on router-confirmed-evocable maneuvers tomorrow, ≥5 seeds.

---

## Q3 — Is the short path a fast weight, and does the lit say to lean on it? YES, and YES — lean hard.

**The fast-weight cluster is unanimous that the K=8 short path is an implicit test-time optimizer, so explicit weight-TTT is redundant
*and* riskier:**
- **von Oswald 2023 (arXiv:2212.07677) — the decisive theory.** Cross-attention over in-context tokens computes an implicit GD update;
  "cross-attending to plan tokens is implicit test-time optimization" (fastweights §2.3d, robust even for softmax). ⇒ the short path
  *already does* what TENT/SAR would do, every chunk, for free, with no collapse risk. Explicit weight-TTT "would only make it heavier"
  (seed's words) and inherits R9. **This is the round's central lit-grounded conclusion.**
- **Schlag 2021 FWP (arXiv:2102.11174):** resampler+adapter writing K (k,v) slots = a fast-weight *write*; DiT cross-attn = the *read*.
  The short path is a fast-weight programmer; the long path (explicit symbolic decomposition) is the slow-weight path.
- **Ba 2016 (arXiv:1610.06258) / Schmidhuber 1992 / Ramsauer Hopfield (arXiv:2008.02217):** K=8 plan tokens = ephemeral fast weights /
  a small Hopfield memory over the frozen DiT; **CFG scale s = the Hopfield retrieval temperature β** (fastweights §1.3d) — a free
  test-time knob (`get_action_with_cfg`) we already have.
- **Helix / π₀ (arXiv:2410.24164) / GR00T N1 (arXiv:2503.14734):** continuous-latent conditioning at fixed K, cross-attention-only (π₀
  uses NO AdaLN for language → validates `plan_adaln=False` default); Helix holds one latent across fast-path steps = our cross-chunk
  token persistence.

**Is our KL-anchor literally CoTTA stochastic-restoration? — YES, functionally, and this reframes the R9 winner in the lit's language.**
CoTTA (arXiv:2203.13591 §b) resets each weight to its source θ⁰ with prob p to bound drift. Our KL-anchor (`demo_bc.py:347–354`) is a
functional L2 (MSE) pulling the *current* plan_head's plan-token output back to a **frozen reference** plan_head's output over a broad
plan pool (`build_anchor_pool`). Same anti-drift-to-source idea; ours is the **deterministic, output-space (functional)** version — and
arguably *better* here because it anchors the plan→token *map* (so duck-on-command survives) rather than random weights. **Lit-grounded
upgrade:** add CoTTA's actual stochastic weight-restore (p≈0.01–0.1) as a near-free *extra* guard inside any Q1-Tier2/Q2 weight update;
EATA's Fisher-weighted anchor (arXiv:2204.02610) is the heavier upgrade (Q4).

**Recommendation: LEAN ON THE CONTINUOUS SHORT PATH; do NOT invest in explicit short-path weight-TTT.** Concretely:
1. Fresh-frame re-encode every chunk (von Oswald = more in-context grounding = more implicit GD; Q1 `fresh`).
2. Use CFG as the Hopfield-β authority knob (Ramsauer).
3. **Elevate FastPlanMod** (`planner.py:305–345`, zero-init `adaln_proj`) from "parked amplifier" to a principled **second fast-weight
   write channel** — global FiLM (Perez 2018, arXiv:1709.07871) via AdaLN-zero (Peebles&Xie DiT, arXiv:2212.09748), null-invariant by
   construction (null rows → 0 offset). **This CONTRADICTS my own R10**, where I parked it as merely an amplifier; the FWP/DiT lit says
   it's a legitimate, null-safe authority channel worth testing as the staleness fix's booster (the K-token KV channel + the FiLM channel
   are the two fast-weight writes GR00T/N1.5 also combine).

---

## Q4 — Capability ADDITION without forgetting: DAgger + plan-token-GATED DiT-LoRA + EATA Fisher

**CONFIRM the R10 plan with three lit-grounded sharpenings.**
- **Mechanism = DAgger (Ross 2011, arXiv:1011.0686).** BC's covariate shift is **O(εT²)** (quadratic); DAgger → **O(εT)** (linear).
  distill §13d ties this *directly* to our EXP-011 "left-then-right" SEQ failure: the actor reaches step 9 having gone left, and steps
  9–17 are OOD with a "right" instruction. Fix = collect corrections at states the **plan-conditioned policy** visits (save-state replay
  from the router), NOT human-visited states. `record_play_server.py` + save-state reconstruction is the queryable oracle.
- **VLA precedent for "new low-level skill under a high-level command" = RT-H** (Belkhale 2024; arXiv ID unconfirmed in lit, flag it) and
  **LAPA** (arXiv:2410.11758): an auto-generated intermediate "language motion" / latent action conditions the low-level policy. Our
  VLM-abstracted plan = RT-H's language motion (owner-constraint-compatible: abstracted, not raw narration). The DiT-LoRA learns
  command→new-trajectory for the add-chunk.
- **Confirm/deny "DiT-LoRA on add-chunks, masked-null, disjoint-merge":** CONFIRM disjoint-merge (R8/R9 proved no-interference) and
  add-chunks-only (no advance re-biasing). **DENY that masked-null alone makes it null-exact** — `lora.py:8–12` is explicit: ordinary
  cross-attn LoRA perturbs *image-token* attention, so null is only approximately preserved. **Fix (the lit-aligned one lora.py itself
  names as "future refinement"): gate the LoRA to plan-token KEY positions only** → null path has no plan tokens → exact-zero LoRA
  contribution → null-invariance restored by construction. Ship the gated LoRA, not ungated.
- **Forgetting safeguard (pick ONE, per seed) = EATA Fisher/EWC (Niu 2022, arXiv:2204.02610).** Estimate diagonal Fisher importance of the
  add-LoRA params from ~100 frames of baseline gameplay; add `λ·Σ F_i(θ_i−θ_i⁰)²` so the add-LoRA can't move params critical to base
  behavior (ttt §T6d: "cleanest learn-a-new-maneuver-without-forgetting method"). Caveat (ttt §T6e): diagonal Fisher is coarse for
  transformers ⇒ COMBINE with our proven structural guards (add-chunks-only + disjoint-merge + plan-token-gating). Fisher is the lit's
  forgetting *bound*; the structural guards are our *demonstrated* ones. (CoTTA stochastic-restore is the cheaper alternative if Fisher
  estimation is too slow.)

---

## Q5 — The ONE coherent "System-2 becomes System-1" architecture + the experiment tonight

**Architecture (each layer lit-anchored):**
```
System-2 (frozen Qwen, VLM-LoRA optional) ── mints VLM-ABSTRACTED plans
   = sleep-time compute (Lin 2025, arXiv:2504.13171: sparse VLM)  +  amortized inference (VAE, arXiv:1312.6114: plan-head ≈ encoder)
        │
        ▼  SHORT PATH (fast weights — LEAN HERE)
   K=8 plan tokens  = fast-weight program (Schlag 2102.11174, Ba 1610.06258) = implicit TTT (von Oswald 2212.07677)
                    = Hopfield memory, CFG=β (Ramsauer 2008.02217)
   • re-encode on fresh frame/chunk (Gandelsman 2307.05014)  • EMA+aug (CoTTA 2203.13591 + MEMO 2110.09506)  → STALENESS FIX
   • + FastPlanMod AdaLN-zero FiLM channel (DiT 2212.09748 + Perez 1709.07871)  → null-safe authority
        │
        ▼  CONSOLIDATION  = ReST-EM (Singh 2312.06585): E=save-state rollouts(router), M=KL-anchored plan-head BC on top-τ winners
                            KL-anchor = functional CoTTA restore (2203.13591); restart-from-base each round (Gulcehre 2308.08998)
        │
        ▼  LONG PATH (ADD only, router-gated)  = plan-token-GATED DiT-LoRA (Akyürek 2411.07279) + DAgger (1011.0686) + EATA Fisher (2204.02610)
GUARDS: KL-anchor + ≥5 seeds + RTG+death + plan-head-only for any reward claim (R9); null action-bit-identity = 0.0.
```
This is ONE generalist (per-chunk/abstracted conditioning, pooled — no per-game weights), VLM-frozen-first, null-invariant.

**Single highest-value experiment tonight = the Q1 staleness forward-pass TTA test** (commands + pass bar in TL;DR/Q1). It runs IN
PARALLEL with the already-queued R10 maneuver-router (router = GPU0 frozen, answers evoke/add → gates Q3/Q4; staleness-TTA = GPU1–2
frozen, answers Q1 = the owner's stated #1 bottleneck). Why this over anything else:
- It is the round's actual thesis: the cleanest falsification of "the continuous short path is already TTT" (von Oswald 2212.07677).
- ZERO training ⇒ structurally immune to the R9 collapse that would confound any weight-TTT run tonight.
- Reuses `staleness_probe.py` (modes/replan_every/`--delta` already there) → ~2 hr, no new infra.
- Decisive for the whole roadmap: if forward-pass re-encode fixes staleness, we NEVER build short-path weight-TTT (saves the collapse
  fight); if it fails, it hands GPT-5.5 the exact next step (emulator-supervised Tier-2).

**Metric / pass bar / falsifier / seeds:** see TL;DR (≥50% oracle−stale recovery on SMW+MMX; flip ≤0.7×live; null 0.0; ≥3 seeds).

---

## Steelman of GPT-5.5's likely pick (and where I concede / push back)

**GPT-5.5 will likely pick the EMULATOR-SUPERVISED per-episode weight-TTT** — i.e., reframe Akyürek per-task adaptation (arXiv:2411.07279)
as save-state GRPO-without-critic (ReST-EM E+M, arXiv:2312.06585) on the plan-head, KL-anchored, ≥5 seeds — and argue my forward-pass
token-smoothing is a band-aid that *adds no capability*, only stability. It has been the audit-the-metric voice (R4 caught the Sonic
screen_x hack; R9 demanded ≥5 seeds + RTG). It will insist the staleness metric be the **emulator** P(progress|fresh)−P(progress|stale),
calling a behavioral flip-rate hackable, and will cite the lit that our loop IS ReST-EM so we should just *run a round*.

**I concede:** (1) the emulator P(outcome) is the hack-proof metric — I've folded it in as the primary advance signal AND as the Tier-2
supervisory signal (entropy is rejected precisely so the emulator can replace it); (2) ANY weight update is ≥5 seeds + KL-anchor(=CoTTA)
+ RTG, non-negotiable (R9). (3) ReST-EM is the correct consolidation precedent (Q2).

**Where I push back:** tonight's pick must be the **zero-training** test, *because* von Oswald (arXiv:2212.07677) says the short path is
already TTT. If we run a weight-TTT round first and it collapses (R9 says it will, ~50% of seeds), we'll wrongly conclude "TTT doesn't
work" when the real lesson is "we skipped the free forward-pass version." Correct sequence: falsify forward-pass re-encode tonight (cheap,
safe, decisive) → only then pay for emulator-supervised weight-TTT, KL-anchored, ≥5 seeds. Stability *is* capability here: the owner's
bottleneck is the 50–60% flip, not a missing maneuver (that's Q4/the router). Fixing the flip with no optimizer is strictly dominant.

---

### Contradictions called out (this round vs prior conclusions)
1. **Seed Q1 framing** (pick a weight-TTT variant) ⟂ **fast-weight lit** (von Oswald 2212.07677, Schlag 2102.11174): the short path is
   already implicit TTT → lean on it forward-pass, don't bolt on weight-TTT for the short path.
2. **LIT_R11_ttt.md Recipe A** (SAR LN-adaptation as the P1 fix) ⟂ **its own caveats** (TENT §e/SAR §a collapse; ttt §Gaps-1 continuous-
   output gap) + R9: reject actor entropy-TTT.
3. **My own R10** (FastPlanMod = "merely a parked amplifier") ⟂ **DiT/FWP lit** (Peebles&Xie 2212.09748, Perez 1709.07871, Schlag
   2102.11174): it's a principled null-safe second fast-weight channel — un-park it for the staleness booster.
4. **In-house framing of the R9 KL-anchor as a generic regularizer** → it is specifically **CoTTA stochastic-restoration**
   (arXiv:2203.13591) in functional/output-space form; unlocks the stochastic-restore + Fisher (EATA 2204.02610) upgrade path.

### Repo anchors (all runnable)
`staleness_probe.py` (null/oracle/live/stale + replan_every + `--delta`; ADD `fresh`/`fresh_ema`) · `demo_bc.py:347–354` `--kl-anchor`
(=functional CoTTA restore) + `build_anchor_pool` + `duck_probe` + `eval_from_states` · `planner.py:305–345` `adaln_proj` FastPlanMod
(AdaLN-zero FiLM channel) · `nitrogen.py` masked-null + `get_action_with_cfg` (CFG=β) · `lora.py:8–12` (gated-LoRA = null-exact path) ·
`snes_env.save_state/load_state` + `env._var` (frame-exact emulator signal) · `base_dit_perdim.py` (router null-AUC) ·
`game_planner.ClosedLoopPlanner` learn-mode (ReST-EM plan-minting substrate).
Constraints honored: ONE generalist; VLM-frozen-first; exact null-invariance (forward-pass TTA + masked-null + bit-identity; gated-LoRA
for the long path); VLM-abstracted plans (not raw narration); GPU-frugal (both tonight runs frozen); no commits.
