# Steering a Frozen Flow-Matching Game Actor with a Frozen VLM Planner: Plan Tokens, Demo-Fit, and the Limits of Env-Free Knowledge Transfer

*A research writeup of the NitroGen + VLM-Planner project. This is an internal academic-style report, not a
peer-reviewed publication. All quantitative claims link to the primary logs in [`research/`](research/) and the
experiment log [`EXPERIMENTS.md`](EXPERIMENTS.md). Numbers are from small-sample, mostly short-horizon evals;
see §9 Limitations before citing.*

---

## Abstract

We study whether a **frozen** vision-language model (VLM), acting as a deliberative *System 2* planner, can
steer a **frozen** ~500M-parameter flow-matching diffusion-transformer (DiT) game actor — NVIDIA's NitroGen, a
reactive *System 1* that maps the current frame to an 18-step gamepad chunk — without retraining the actor. Our
mechanism is a small trainable **bridge**: a Perceiver resampler distills the VLM's hidden states into **K=8
continuous "plan tokens"** that are injected into the DiT's vision-language cross-attention. A **masked-null**
construction makes a null plan reproduce the base actor *exactly*, enabling classifier-free guidance (CFG) on
the plan. In an env-free synthetic stage we show the bridge learns direction steering, exact null-invariance,
within-chunk temporal ordering (with zero-shot compositional generalization), cross-chunk and nested control —
all on a frozen DiT — with the key enabler being a **plan-token contrastive loss** that de-collinearizes
opposite-meaning plans. We then fit the bridge on human gameplay demos and find a clean dissociation: for
**plan-OOD** (obstacle/decision) games, supervised **plan-head demo-fit** widens inference-time plan-following
and *generalizes* across games via a single pooled fit; for **actor-OOD** (open locomotion) games the lever is
on-policy actor adaptation instead, and the two write disjoint parameters so they **merge into one generalist**.
We identify and fix a failure mode — fitting on one terse plan collapses the action vocabulary — with a
**KL-anchor** that adds a skill while preserving the rest by construction. Finally, we characterize what does
*not* transfer env-free: the VLM can **evoke** a primitive the actor already owns (e.g. ducking, 1.2%→10.8% on
command) but cannot install precise *outcomes* the actor lacks (a spin-jump kill is 7× evocable as a button yet
near-chance as an outcome); and counterfactual override robustly requires an environment. We close with a
forward proposal — consolidating evoked skills into an always-present per-game channel so they become
instinctual — and with RAM-independent VLM-judged video verification that reports honest, modest gains.

---

## 1. Introduction

Reactive imitation policies for games and robotics are typically **markov**: they map the current observation to
a short action chunk and cannot deliberate over long horizons. NitroGen is such a policy — a flow-matching DiT
that sees only the last frame. The question we pursue is whether *deliberation* can be supplied by a separate,
frozen VLM (the kind that is good at reading a scene and writing a plan) **without** the cost and instability of
fine-tuning the actor.

This is a **System-1 / System-2** decomposition (Kahneman): a fast reactive controller (S1) and a slow
deliberative planner (S2), connected by a learned bridge. Our design choices are driven by three constraints
that recur throughout:

1. **The actor stays frozen** (at least initially) — so the bridge must *steer*, not *retrain*.
2. **Exact null-invariance** — a "no plan" input must reproduce the base actor bit-for-bit, so that plan
   guidance can be expressed as CFG and so a generalist never regresses an untouched game.
3. **One generalist** — no per-game or per-genre models; cross-game interference is a problem to *solve*.

Our contributions:
- A **plan-token bridge** (resampler + adapter + injection) with a masked-null construction giving exact CFG
  guidance on a frozen DiT (§3).
- An **env-free synthetic alignment** result: a frozen DiT can be steered for direction, ordering, and
  cross-chunk control once a **contrastive loss de-collinearizes the plan tokens** (§4, §6.1).
- A **plan-OOD / actor-OOD dissociation** and a **one-generalist recipe**: pooled plan-head demo-fit for the
  former, on-policy adaptation for the latter, merged via disjoint parameters (§6.2).
- The **KL-anchor**: add a skill via demo-fit without collapsing the action vocabulary, by construction (§6.3).
- A characterization of **what does not transfer env-free**: evocation vs addition; counterfactual override
  needs an environment (§6.4–§6.5).
- A forward proposal — **consolidation into instinct** via an always-present per-game channel / learning tokens
  (§8) — and **RAM-independent video verification** of the demos (§7).

---

## 2. Related work

**Hierarchical / dual-system control.** Our split mirrors robotics VLAs that pair a slow VLM with a fast motor
policy: Figure **Helix** (async continuous-latent S2→S1), NVIDIA **GR00T** (V/L embeddings → connector →
flow-DiT with cross-attn + AdaLN), and **π0/π0.5** (text → 1s flow chunk). Unlike these, our actor is *frozen*
and reached only through a K-token bridge with *exact* null-invariance. Survey: [`research/ARCH_RESEARCH.md`](research/ARCH_RESEARCH.md).

**Plan/skill conditioning and resamplers.** The K-query resampler is Perceiver/Q-former-style (BLIP-2). Steering
a generative model via a guidance contrast is classifier-free guidance (Ho & Salimans, arXiv:2207.12598).

**Options / skills / language-as-action-abstraction.** The forward direction connects to the options framework
(Sutton-Precup-Singh), VOYAGER's growing skill library (2305.16291), RT-H "language motions" (2403.01823), and
discrete skill-token codebooks (LISA 2203.00054, PRISE 2402.10450). Consolidating a conditioned behavior into an
unconditioned policy is policy distillation / LwF (1606.09282) / Distral (1707.04175); the verify→distill→
compose loop is Expert-Iteration (1705.08439). Full grounding: [`research/LIT_R13_S2TOS1.md`](research/LIT_R13_S2TOS1.md).

**Test-time training.** A continuous prompt re-grounded each step is a form of TTT (von Oswald 2212.07677;
Schlag 2102.11174); we find (§6.4) that for our tokens this is a no-op because the tokens are already stable.

---

## 3. Method: the plan-token bridge

### 3.1 System 1 — the flow-matching DiT actor (frozen)

A reactive imitation policy for games — NitroGen, a ~500M-parameter flow-matching DiT that sees only the last
frame. It is trained by **flow matching**: for an action chunk `a` (18×25), noise `ε~N(0,1)` and `t~U(0,1)`,

```
noisy = (1−t)·ε + t·a ;   v_target = a − ε ;   loss = ||DiT(noisy, t, context) − v_target||²
```

The actor predicts the velocity field between noise and data; at inference it integrates that field from noise
(Euler) to produce the chunk. The frame becomes 256 image tokens (SigLIP); the action tokens **cross-attend** to
the VL context. The plan enters *only* through that context.

### 3.2 System 2 — the VLM planner (frozen)

A frozen Qwen3.5 VL model (0.8B → 2B). At training time `encode_multimodal(frame, plan)` yields last-layer
hidden states `(B, L, d_vlm)` (cached features — the VLM never gets gradient). At deployment `generate_plan`
autoregressively writes a short plan from recent frames.

### 3.3 The bridge (trainable): resampler → adapter → plan tokens

- **PlanResampler:** K=8 learned queries cross-attend (2 layers) to the VLM hidden states → `(B, 8, d_vlm)`. A
  Perceiver/Q-former distillation of variable-length VLM output into a fixed K. (Cross-chunk: emits K·A queries
  = A blocks; a per-example cursor selects this chunk's block — §6.1, EXP-027.)
- **PlanAdapter:** projects `d_vlm → d_dit` (e.g. 2048→1024) *after* the resampler (no lossy bottleneck before
  the cross-attention). Output: K=8 plan tokens in the DiT's vision space.
- **Injection + masked-null:** the 8 tokens are written into typed `_PLAN_TOKEN` slots of the VL sequence. For a
  null/dropped example, `apply_null_mask` zeroes those slots in the VL attention mask → the forward pass is
  **bit-identical to the bare DiT**. This is the CFG unconditional anchor; plan guidance is
  `v = v_null + w·(v_plan − v_null)`.
- **Optional global authority (plan-adaLN):** the pooled plan can additionally inject a zero-init FiLM/AdaLN
  offset into the DiT timestep embedding, gating every block (EXP-048). Null-masked → base-exact.

Exact code paths and tensor shapes: [`TRAINING_RECIPE.md §10`](TRAINING_RECIPE.md).

### 3.4 The losses

The action loss is the flow-matching velocity MSE (§3.1), masked by a per-dim `actions_mask`. Two auxiliaries
matter:
- **Plan-token contrastive (SupCon).** Groups plan tokens by a direction label and pushes apart opposite labels.
  The representation is taken **per-token** (independently at each of the K positions, then averaged), which
  forces every query position to be label-discriminative — the fix for the collinearity collapse (§6.1).
- **KL-anchor (demo-fit).** A functional L2 from the training plan-head's tokens to a *frozen pre-fit* copy over
  a broad plan distribution: `+λ·||plan_head(pool) − plan_head_ref(pool)||²` (λ=0.3). It pins "what plans mean"
  while the action loss adapts behavior — adding a skill without forgetting the rest (§6.3).

Full formulas: [`TRAINING_RECIPE.md §9`](TRAINING_RECIPE.md).

---

## 4. Training pipeline

Training is staged; each stage reuses the prior checkpoint and trains a small parameter set.

| Stage | Trains | Data | Loss | Output |
|---|---|---|---|---|
| 1. synthetic alignment | bridge (+opt LoRA) | synthetic counterfactual chunks (no env) | velocity MSE + contrastive | the capability suite (§6.1) |
| 2. real plans, 2B | bridge (+LoRA) | VLM-on-frames, action-conditioned, boundary-frame caches | + distillation | `btn_s600` (base eval model) |
| 3. demo-fit (plan-OOD) | **plan-head only** | human demos, each on its game's plan | velocity MSE (+ KL-anchor) | pooled / kl / situ deltas |
| 3′. actor adapt (actor-OOD) | **LoRA only** | env rollouts | reward-weighted BC | rwbc deltas |
| merge | — | — | overlay disjoint groups | one generalist |

Stage-2's central lesson (EXP-033..038): game-stream *transcripts* are chit-chat and *frame-only* VLM plans do
not predict the real action (chance-level); the fix is **action-conditioned** plan supervision, which grounds
the plan in what the player actually did. The 2B redesign (EXP-050) moved the resampler to the backbone's native
width and the adapter *after* it. Exact commands, hyperparameters, and the per-game text plans:
[`TRAINING_RECIPE.md §3–§4`](TRAINING_RECIPE.md); per-checkpoint detail: [`CHECKPOINTS.md`](CHECKPOINTS.md).

---

## 5. Experimental setup

**Games.** Super Mario World (SMW), Super Mario All-Stars/SMB1 (SMBAS), Mega Man X (MMX) — plan-OOD obstacle
platformers; Sonic 2 — an actor-OOD open game. Human demos recorded via a browser play-and-record server;
death-tails trimmed before chunking. **Substrate.** In-process emulator envs (stable-retro/mGBA) with
frame-exact save/load enable fixed-start evals and counterfactual checks.

**Metrics.** (i) **Δ_plan** = advance under the plan − advance under null, in the game's RAM progress variable,
from fixed demo save-states over 16 chunks — isolates the plan's contribution. (ii) **Survival-weighted reach**
= running-max progress *before* death (death = lives-decrement or a progress reset), so a suicide-sprint scores
zero. (iii) **RAM-free VLM video judgment** — a VLM watches sampled rollout frames (§7). We report multi-seed
means with bootstrap CIs where available. **Caveats on metrics are first-class findings — see §7, §9.**

---

## 6. Results

### 6.1 Env-free synthetic alignment works on a frozen DiT (Stage 1)

A frozen DiT can be steered once the plan tokens are de-collinearized. The decisive chain: opposite-direction
plans initially collapse to nearly the same token (cos(L,R)≈1.0), which collapses continuous-stick steering
(**EXP-006/007**); a **supervised-contrastive loss on the pooled plan tokens** separates them (cos(L,R)
0.999→0.10) and direction steering works with **exact null-invariance** (**EXP-009/010**). A later **per-token**
refinement (EXP-016b) additionally solves within-chunk ordering. On the frozen DiT we further obtain:
within-chunk ordering with **SEQ4 zero-shot compositional generalization** (EXP-016/018), cross-chunk control
via a cursor over K·A blocks (EXP-027), nested two-level control (EXP-029), and following **real** multi-chunk
action plans post-hoc (EXP-031). Fine cross-modal routing (a specific button in a specific half, exact
transition timing) is where DiT **LoRA** capacity helps (EXP-021/024). Capability→checkpoint map:
[`INDEX.md`](INDEX.md).

### 6.2 A plan-OOD / actor-OOD dissociation, and one generalist

Fitting the bridge on human demos reveals that "what is out of distribution" is **game-type-dependent**
([`research/DISCRIMINATOR_RESULTS.md`](research/DISCRIMINATOR_RESULTS.md)):

- **Plan-OOD (SMW, MMX, SMB1):** supervised **plan-head demo-fit** (DiT/LoRA/VLM frozen, no env reward) widens
  inference-time Δ_plan. It **generalizes** to a new game out of the box (MMX +117%, 3/3 seeds) and a **single
  pooled fit** over three games widens Δ_plan on all of them simultaneously (SMW +26, MMX +31, 3/3 seeds,
  CIs>0, null-invariance exact); SMW even *benefits* from pooling (1.3× vs solo).
- **Actor-OOD (Sonic):** the plan is near-redundant; the lever is on-policy **reward-weighted BC** (LoRA).
- **One generalist:** the plan-head delta (plan-OOD) and the LoRA delta (actor-OOD) write **disjoint
  parameters**, so merging them onto one base retains/improves all games with no interference and exact
  null-invariance.

What does **not** work: on-policy RWBC *for plan-OOD games* yields **no reliable win** — it collapses or
sign-flips across seeds and configs (raw progress reward = suicide-sprint; one apparent win was retracted as a
config artifact). The pivot to supervised, trusted-target, no-reward demo-fit is what made plan-OOD games
trainable.

### 6.3 The KL-anchor: add a skill without forgetting (R9)

Naïvely fitting on a single terse plan ("move right…") makes plain BC ignore the plan and collapse the action
vocabulary onto right+jump — it drops rare situational actions (ducking goes to 0% for *all* plans, breaking
even the base model's plan-responsiveness). The **KL-anchor** (functional L2 to a frozen pre-fit plan-head over
a broad plan distribution) **adds the advance skill while preserving the whole action space by construction**
(expressiveness battery R=0.83, the only passing variant; Δ_plan +45.1, 3/3 seeds; anchor loss ~0.001 →
non-distorting). This is policy-distillation/LwF/Distral in plan-token function space, and it is the
consolidation primitive the forward proposal relies on.

### 6.4 Evocation vs addition; staleness is text-bound

A measured dichotomy emerges (R11/R13, [`research/FORWARD_S2_TO_S1.md`](research/FORWARD_S2_TO_S1.md)):
- **Evocation (free):** the VLM can *summon* a primitive the DiT already owns. SMW ducking goes 1.2% → 3.4% →
  **10.8%** as the plan names it more explicitly — the actor already has "duck"; the plan token retrieves it.
- **Addition (costly):** a precise *outcome* the actor lacks is not installable by naming. A spin-jump *kill*
  is 7.2× evocable as a **button** yet stays near chance as an **outcome** (survive+advance past the enemy
  3.75%→7.5%). It must be *learned* into the actor, not named.

Separately, we find **staleness is text-bound, not token-bound**: the K plan tokens barely change frame-to-frame
(drift ~0.001), so re-grounding them every step (a TTT-style fix) is a no-op; live re-planning can even *hurt*
(it churns the plan text without payoff). The lever is plan **text** quality, not token- or weight-level TTT.

### 6.5 The env negative (robust): counterfactual override needs an environment

Two independent lines say long-horizon counterfactual capability cannot be manufactured purely offline: (i)
probing the DiT internals finds **no reusable latent world model** (EXP-034), so counterfactual futures can't be
rolled out in the actor; and (ii) explicit env-free **CFG override training fails and gets *worse* with data
scale** (EXP-047/049/049b/054). This motivates the emulator save-state substrate for any future RL.

### 6.6 Honest demo verdict (RAM-free verified)

Trained models go **further** than base (MMX ~3×, Sonic ~10× by rings, SMW kl ~4×), but the far-reaching runs
often **die**; no model beats a level. The clearest demos are the ones that go furthest *and* survive (Sonic
pooled: 40 rings + survives 90 s vs base 4 rings + death at 29 s; SMW kl: survives a lava fortress where base
dies at 13.6 s). See [`research/VERIFY_SMW.md`](research/VERIFY_SMW.md), [`VERIFY_CROSSGAME.md`](research/VERIFY_CROSSGAME.md).

---

## 7. Verification methodology (and a cautionary finding)

Because RAM progress signals can be unreliable, we verify rollouts RAM-free: a VLM watches sampled frames and
judges progress/death. This surfaced a **symmetric reliability problem**:
- **RAM** misses some deaths and its progress *magnitude* is not comparable across levels (warp/underground
  scaling; Sonic's `screen_x` is camera-inflated).
- **The VLM judge false-positives deaths** — it calls "died" when a hazard is merely near the sprite. We proved
  this on one rollout the judge flagged as a death: the in-game **TIME counter** (which resets on death in SMW)
  counts down continuously with no reset → the actor actually survived.

**Conclusion:** neither RAM nor a VLM judge is reliable alone; the trustworthy arbiter is a **game-intrinsic HUD
counter** (TIME/lives/rings) read from the pixels, cross-checked against both. This both validates the demo
verdicts and motivates a HUD-OCR death detector. ([`research/ORCHESTRATOR_VERIFY.md`](research/ORCHESTRATOR_VERIFY.md).)
A complementary open diagnostic distinguishes *plan-quality* from *DiT-adherence* failures: log the live plans
(`--log-plans`) and measure CFG plan authority `||v_plan − v_null||` (`plan_authority_map.py`).

---

## 8. The forward proposal: consolidation into instinct

The owner's north star is a recursion: *discovery → instinct → higher-level discovery*. A skill the VLM evokes
transiently (§6.4) should become **owned** by the actor so the VLM can stop naming it and compose at a higher
level. The war-room (R13) converged, with a construction caveat: consolidating into the *null* policy would
break masked-null exactness (the null policy **is** the base, on a frozen DiT). The construction-safe surface is
an **always-on per-game channel** — the game-id embedding (never masked), or, more ambitiously, a bank of N
persistent **"learning tokens"** per game that hold the game's learned capabilities. A zero-init, KL-anchored
residual there makes a skill fire *without* the plan token (instinct; frees plan bandwidth) while the K plan
tokens stay cleanly maskable → plan-relative null-invariance remains exact. An un-learned game stays base-exact
via a zero learning-id. The first minimal experiment is **consolidate-duck**: evoke ducking, verify the
bullet-state in the emulator, KL-anchored self-distill it into the persistent channel, then show the null policy
ducks without the plan token — and the VLM's freed budget carries a higher-level plan. The eventual compression
of many consolidated skills is a discrete **skill-token codebook** (LISA/PRISE). Full mechanism, rankings, and
falsifiers: [`research/FORWARD_S2_TO_S1.md`](research/FORWARD_S2_TO_S1.md).

---

## 9. Limitations

- **Small-sample, short-horizon evals.** Δ_plan uses ≤16-chunk rollouts from a handful of fixed save-states;
  many results are 3–6 starts × 1–3 seeds. Closed-loop long-horizon drift and real-time replan jitter are
  largely **unproven**.
- **Progress signals are imperfect** (§7): trust HUD counters, not raw RAM magnitude or a single VLM judgment.
- **The actor is weak in absolute terms** — no model beats a level; gains are "further, often riskier."
- **Plan-OOD evals use oracle/fixed plans**; the live-VLM deployment gap is only partially characterized.
- **One delta-seed each for the merge** proof-of-concept; the multi-seed averaging is a cleanup, not done.
- **No counterfactual override env-free** (§6.5) — by design the next capability needs the environment.

---

## 10. Conclusion

A frozen VLM can steer a frozen flow-matching game actor through a small plan-token bridge with exact CFG
guidance. Env-free, this yields real, composable *steering* (direction, ordering, cross-chunk) and a clean
recipe for *plan-following* that generalizes across games and merges into one model. But there is a sharp
boundary: the bridge can **evoke** what the actor already knows and cannot **install** precise new outcomes
without an environment, and counterfactual override is demonstrably env-bound. The path forward is to turn evoked
skills into *owned* ones via construction-safe consolidation, and to ground new outcomes in emulator
save-state RL. The map of every experiment, discussion, and artifact is in [`ATLAS.md`](ATLAS.md).

---

### Reproducibility
Code: GitHub `namak-kun/NitroGen-With-VLM-Planning` (branch `namak-kun/plan-conditioning`). Weights + demos
(private): HF `nmk-kun/nitrogen-vlm-planner-handoff`. One-command recovery: `scripts/recover_checkpoints.sh`.
Exact recipes/losses/data: [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md). Results: [`research/DISCRIMINATOR_RESULTS.md`](research/DISCRIMINATOR_RESULTS.md).

### Acknowledgements
Builds on NVIDIA NitroGen and Qwen3.5. Architecture debates were run as a structured "war-room" between GPT-5.5
and Opus-4.8 agents (R1–R13); literature reviews are arXiv-verified ([`research/LIT_R13_S2TOS1.md`](research/LIT_R13_S2TOS1.md)).
This is a research project for research purposes only; it uses free/FOSS/homebrew games, and gameplay footage of
third-party games is retained for private research use.
