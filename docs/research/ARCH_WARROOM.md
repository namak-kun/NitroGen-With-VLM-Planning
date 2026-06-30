# ARCH WAR ROOM — head-to-head architecture debate (GPT-5.5 vs Claude Opus 4.8, refereed by Opus-CLI)

Append-only. Topic: **a near-real-time DUAL-RATE coupling of a slow VLM planner and a fast flow-DiT actor.**
Each debater writes to its OWN per-round file (ARCH_WARROOM_<who>_r<N>.md, ABSOLUTE path given in prompt);
the referee (Opus-CLI) consolidates rounds here. Be concrete, technical, cite real systems, attack the
other's weak points. No style nitpicks.

================================================================================
## SEED / SHARED CONTEXT (read before writing)
================================================================================

### The system (CORRECTED facts — respect exactly)
- **System 1 / actor "NitroGen"**: a ~500M flow-matching Diffusion Transformer whose architecture is BASED
  ON NVIDIA GR00T (N1/N1.5) — a flow/DiT that denoises continuous action chunks, conditioned via
  CROSS-ATTENTION to vision/language embeddings + AdaLN diffusion-step conditioning. Input = the CURRENT
  game frame (the DiT ALREADY sees the live frame every chunk → it has fast VISUAL reactivity). Output = an
  18-step chunk of 25-dim gamepad actions (21 buttons + 2 sticks). ~16 flow inference steps per chunk.
- **System 2 / planner**: a frozen Qwen3.5-2B VLM → emits a short text plan + frame-grounded hidden states.
- **The bridge**: a Perceiver resampler with **K=8 QUERY TOKENS** + a linear adapter. CORRECTION: these
  K=8 "plan tokens" are CONTINUOUS LATENT VECTORS (resampler outputs), NOT discrete vocab tokens. So the
  issue is NOT continuous-vs-discrete. The real gaps are: **(a) BANDWIDTH** (only K=8 latent vectors carry
  all of System 2's intent), **(b) RATE** (plan recomputed only every A=2 chunks ≈ 1.2s), **(c) STALENESS**
  (the semantic/plan signal is frozen between replans even though the DiT's visual input is fresh), **(d)
  BLOCKING LATENCY** (every A chunks the VLM does a FULL forward generation that BLOCKS the control loop;
  NitroGen already doesn't run real-time, the VLM makes it worse).
- CFG scales plan influence; a null/masked plan reproduces base NitroGen EXACTLY (must preserve this).
  Only a rank-16 DiT-LoRA + the bridge are trainable; VLM frozen (a VLM-LoRA is now ALLOWED but
  higher-cost, flag it). Hardware: 4× RTX A6000.

### The OWNER'S dual-rate idea to design around (this is the thesis under debate)
An **EARLY-EXIT / depth-wise dual channel on a SINGLE VLM**: tap **EARLY-LAYER VLM latents as the FAST
path** (available early in the forward pass, cheap, low-latency) to give the DiT a quick coarse semantic
nudge EVERY chunk, while the **FULL VLM generation is the SLOW path** (refined plan, every A chunks). One
VLM, two depths: shallow=fast/coarse/frequent, deep=slow/refined/infrequent. Plus: start the slow plan
PREEMPTIVELY/async, and use LATENT-BASED COMPRESSION + caching as the dual channel so the VLM never blocks
the DiT. Inspiration: iSHIFT (Lightweight Slow-Fast GUI Agent with Adaptive Perception, arXiv:2512.22009).

### THE DEBATE QUESTION
Design the BEST architecture to make this slow-fast VLM→DiT system run in (or near) real time AND fix the
bandwidth/rate/staleness gaps, on 4×A6000, VLM frozen first. Specifically take a STANCE on:
1. Is the owner's EARLY-EXIT (shallow-VLM-latent fast path) the right fast channel — or is a separate tiny
   fast module (conditioned on cached deep latent + live frame) better? Defend one.
2. What EXACTLY flows on the fast path vs slow path (shallow hidden states? a compressed latent? FiLM/AdaLN
   gains? K extra tokens?), at what RATE, and how is null-invariance preserved?
3. How to make it NEAR-REAL-TIME: async/preemptive planning, latent caching/reuse, chunk-stitching — be
   concrete about what runs concurrently and how stalls are hidden.
4. The training objective for the fast/shallow path (what supervises it; where does the data come from given
   we have human demos + a frame-exact emulator + RL reward but NO mid-mistake recovery labels yet).
5. The single highest-leverage experiment to run FIRST, and what to NOT build yet.
Constraints to respect: preserve exact null-invariance; VLM frozen first; be honest about ROI for a
2D-game-from-pixels setting (not a real robot). Attack the other side's plan where it's weak.

(Prior survey context lives in files/ARCH_RESEARCH.md; a deeper cited survey is being written to
files/ARCH_RESEARCH_V2.md in parallel — you may read them if present.)

================================================================================
## ROUND 1 — REFEREE CONSOLIDATION (Opus-CLI, 2026-06-27)
================================================================================
Full positions: files/ARCH_WARROOM_gpt55_r1.md , files/ARCH_WARROOM_opus_r1.md

### Both debaters REJECT the owner's literal early-exit-by-depth (shallow VLM layers = fast path)
- GPT55: early layers ungrounded; proposes Helix-style split — slow frozen Qwen writes a cached deep
  continuous latent (K=32 + global latent, async every 2-4 chunks); a SEPARATE tiny FastPlanMod reads the
  live NitroGen frame features every chunk + emits 4-8 corrective tokens + CFG gate + zero-gated AdaLN.
- OPUS48: reframes the axis as PREFILL-vs-DECODE, grounded in THIS repo's code: the bridge already reads
  LAST-LAYER states from a single PREFILL (planner.py:173); the blocking cost in the live loop is the
  autoregressive generate_plan DECODE (24 sequential memory-bound token-forwards ~= a whole chunk;
  planner.py:238-275, eval_policy.py:124-126). Early-exit shaves the cheap prefill, not the decode. Fast
  path = the SAME frozen VLM run as a decode-free FULL-DEPTH PREFILL over [frames + cached plan-text + K
  learned register tokens], read every chunk through the EXISTING bridge; slow path = the async decode every
  A chunks. No separate module, no new representation -> no shallow/deep distribution shift.

### REFEREE FACT-CHECK — iSHIFT (arXiv:2512.22009), verified vs primary source (HTML v1)
Title: "iSHIFT: Lightweight Slow-Fast GUI Agent with Adaptive Perception" (Mehrotra, Rebbapragada, Bonthu,
Balasubramanian; cs.CV; 2025-12). Mechanism (Alg.2 + prompt templates, S.1/S.2.1): built on Qwen2-VL-2B +
frozen DINOv2-L. FAST path = latent-thinking tokens only (<bot><latent..><eot>, decode-free implicit CoT).
SLOW path = latent-thinking + adaptive LATENT PERCEPTION tokens (<bop><ctrl><eop>) + a detection image, for
grounding-critical actions, routed by a rule-based classifier f_cat(A). => iSHIFT is NOT depth-wise
early-exit; it is decode-free latent CoT + ADAPTIVE-PERCEPTION token routing. OPUS48's mechanism claim is
CORRECT and GPT55's framing is consistent; the owner's "early-LAYER latents = fast path" reading is NOT what
iSHIFT does. CAVEAT: iSHIFT routes PER-ACTION (fast OR slow by grounding need) -- it is adaptive-COMPUTE, a
DIFFERENT axis from a dual-RATE real-time controller (fast EVERY chunk + slow occasionally). Neither side
should cite iSHIFT as a real-time dual-rate precedent; it's an adaptive-perception precedent. Also OPUS48's
acronym expansion ("Implicit Slow-fast Hybrid Inference with Flexible Tokens") is NOT in the paper -> treat
as unverified gloss; the mechanism description is accurate.

### THE CRUX for Round 2 (empirically settleable, cheap)
Separate-tiny-fast-module (GPT55) vs same-VLM-prefill-every-chunk (OPUS48). The disagreement reduces to ONE
measurement: is a single full-depth Qwen-2B PREFILL cheap enough to run EVERY chunk (Opus: ~20-60ms << a
chunk), or must we cache the deep latent and read it with a tiny module because even the prefill is too slow
at chunk rate (GPT55's implicit assumption)? Both AGREE: (a) the autoregressive decode must go async/off the
critical path; (b) keep a small zero-init DiT-side AdaLN/CFG modulator for localization; (c) preserve exact
null-invariance; (d) RTC-style chunk-stitching (arXiv:2506.07339) + Helix staleness-offset training; (e) the
fast path's missing recovery supervision comes from emulator save-state RL, not demos.

DECISIVE NEXT STEP (both proposed a variant): EXPERIMENT-0 = measure per-chunk wall-clock of {autoregressive
generate_plan decode, single decode-free prefill, one DiT 18-step chunk} on btn_s600 on one A6000, + verify
decode-free-prefill behavioral parity & exact null-invariance vs the decoded plan. This settles the crux and
is near-zero build (uses the existing text_only path, planner.py:217). Round 2 should be framed AROUND this
measurement.

================================================================================
## REFEREE CLARIFICATION before Round 2 (Opus-CLI, 2026-06-27) — component roles + user note
================================================================================
USER NOTE: "gpt 5.5 might be slightly confused because the DiT is the small fast component." Both
directions are acceptable to the owner right now; he needs thinking time. Round 2 = rebut + converge.

PIN DOWN THE COMPONENT ROLES (so neither side drifts):
- **Qwen3.5-2B VLM = the SLOW, LARGE, BLOCKING component.** Its autoregressive DECODE (generate_plan, ~24
  tokens) is the real per-replan stall (~a whole chunk); its PREFILL is cheaper (one parallel pass).
- **NitroGen DiT (~500M) = the SMALL, FAST component = System 1 = the actor.** It ALREADY sees the live
  frame every chunk (fast visual reactivity exists). Its cost is ~16 flow steps/chunk — cheap vs the VLM.
  Real-time is hard because of (i) the VLM blocking and (ii) NitroGen's own flow-sampling infra.
- So in Helix terms: our **DiT already IS S1** (the 200Hz visuomotor reader). The open question is NOT
  "build an S1" — it exists — but "what feeds S1 the semantic intent, how cheaply, how fresh."

THE CRISP DISAGREEMENT for Round 2:
- OPUS48: the DiT (S1) should consume a cheap decode-free VLM PREFILL latent EVERY chunk, through the
  EXISTING bridge — NO new module. (DiT is the fast consumer; just feed it fresh latent.)
- GPT55: insert a TINY (5-20M) FastPlanMod BETWEEN the cached slow latent and the DiT, that localizes the
  cached intent against live NitroGen frame features + emits AdaLN/CFG/fast-tokens EVERY chunk; the VLM
  prefill is run only at the slow rate (cached), not every chunk.
- The hinge: GPT55 assumes running even the VLM PREFILL every chunk is too costly on 4xA6000, so it caches
  the latent and adds a tiny localizer; OPUS48 assumes the prefill is cheap enough (~20-60ms) to run every
  chunk, making the localizer redundant. => This is the SAME empirical question (EXPERIMENT-0: measure
  decode vs prefill vs DiT-chunk wall-clock). Note: even if prefill-every-chunk is affordable, GPT55's
  localizer could still help if the cached latent needs frame-specific re-grounding — so the two are not
  mutually exclusive; R2 should clarify whether the localizer adds value BEYOND fresh prefill latents.

================================================================================
## ROUND 2 — CONSENSUS (Opus-CLI referee, 2026-06-27)
================================================================================
Full R2: files/ARCH_WARROOM_{gpt55,opus}_r2.md. The debate CONVERGED. Both reject the owner's
early-exit-by-depth; both agree the real blocker is the autoregressive DECODE (not depth).

### Agreed SHARED CORE (build regardless of branch)
- **Wire the VLM decode ASYNC, off the control loop** (UNCONDITIONAL — both endorse; the actor never waits
  for generate_plan).
- One **small, plan_valid-gated, ZERO-INIT DiT-side modulator** that reads CONTROL-STATE the VLM can't see
  (action trace, chunk-phase, event bits: stuck/damage/death/room-change) → AdaLN/CFG gates. (Opus conceded
  GPT55's input list; GPT55 conceded the modulator should be tiny/optional.)
- Widen plan tokens **K=8→32**; RTC chunk-stitching (arXiv:2506.07339); Helix staleness-offset training;
  recovery supervision from **emulator save-state GRPO** (not demos); preserve EXACT null-invariance;
  VLM frozen first (VLM-LoRA = higher-cost, deferred).

### The ONE fork — decided by a measurement, not opinion
r = T_prefill / T_chunk (per chunk wall-clock):
- **Branch A** (r < 0.5 AND decode-free-prefill behavioral parity holds): run the decode-free full-depth VLM
  PREFILL EVERY chunk through the existing bridge; modulator shrinks to a <1M gate. (Opus's design; the
  prefill hides async on a dedicated idle GPU at DEPLOY → ~0 critical-path cost. "Scarce GPU" only bites at
  TRAINING.)
- **Branch B** (r ≥ 1, or parity fails, or training-GPU contention): CACHE the deep latent at the slow rate +
  a 5-20M FastPlanMod localizer that re-grounds it against live frame/control-state every chunk. (GPT55's
  design; prefill-every-chunk is ~360× the FLOPs / ~20-50× the latency of the localizer and double-encodes
  the frame.)

### Decisive experiments (agreed)
- **EXPERIMENT-0 (first, ~zero build):** on btn_s600, one A6000, measure per-chunk wall-clock of {async
  generate_plan DECODE, single decode-free PREFILL, one DiT 18-step chunk}; + verify decode-free-prefill
  behavioral PARITY and EXACT null-invariance vs the decoded plan (uses the existing planner.py:217
  text_only path). This picks Branch A vs B.
- **EXPERIMENT-1:** does the control-state localizer add value BEYOND a fresh prefill latent (perturbation/
  recovery test)? Decides whether the modulator earns its keep in Branch A.
- **AGREED FIRST BUILD:** run EXPERIMENT-0 **and** wire the decode async (unconditional) in parallel.

### Referee note
Clean convergence; the binary design choice is correctly DEFERRED to EXPERIMENT-0 (a cheap measurement), so
no commitment is needed now. Both fixed the iSHIFT acronym; Opus verified the A6000 throughput numbers
(~156 TFLOP/s BF16, 768 GB/s) that make the prefill-vs-decode asymmetry load-bearing. Owner: you can take
thinking time — the next action (measure latencies + async the decode) is valuable under BOTH branches.

---

# ===== ROUND 3 (2026-06-28): Actor-OOD drift & stale plans — LATENCY OFF THE TABLE =====
Seed: ARCH_WARROOM_R3_SEED.md. Full positions: ARCH_WARROOM_{gpt55,opus}_r3.md. Owner reframe: "assume we
PAUSE the game during inference like NitroGen normally does — latency is irrelevant. The real problems are
(1) stale plans and (2) the DiT drifting OOD even on a TRAINED game because the state is subtly different."

## UNANIMOUS CONSENSUS (no rebuttal round needed — both models converged independently)

**Q1 — Which OOD dominates? → DiT-ACTOR OOD (not plan-OOD).**
"Subtly different state on a trained game" = classic behavior-cloning COVARIATE SHIFT / compounding error.
NitroGen is a BC'd flow policy fit to near-expert demos; each chunk's small error nudges the agent off the
demo manifold where it never trained, and error compounds — Ross–Bagnell: BC error grows **O(εT²)** in
horizon vs **O(εT)** on-policy (Ross & Bagnell AISTATS 2010; DAgger, Ross-Gordon-Bagnell AISTATS 2011,
arXiv:1011.0686). It's an in-distribution GAME but OOD CLOSED-LOOP VISITATION.
- **Decisive in-repo evidence:** the frozen base DiT reproduces the streamer's action FROM THE FRAME ALONE
  (buttons AUC 0.82–0.96, sticks 0.4–0.76) → on on-trajectory states the **plan gradient ≈ 0**; if the action
  is fixed by the frame, drift can't be a plan failure. And on-policy RWBC beats base (Sonic 0.58>0.25, SMW
  0.53>0.41) — the *signature of covariate shift*, not a plan defect.
- **Plan-OOD is real but SECONDARY & categorically different:** it bites on novel (frame×plan) combos
  (stale/contradicting plans) as an INTERMITTENT failure at replan boundaries — already owned by the R1/R2
  consensus (staleness-offset training + dynamic-CFG). It is NOT the CONTINUOUS drift the owner means.

**Q2 — Is a fresher/better PLAN the lever for actor-drift? → NO (both, emphatic; Opus concedes to GPT-5.5).**
Where drift happens (on/near-trajectory states) the action is already determined by the frame, so improving
the plan CANNOT change it. The plan only moves the action where the frame is genuinely AMBIGUOUS (a real
fork) = **objective selection**, the markov gap the VLM exists to fill — NOT drift recovery. The ONLY cure
for actor-drift is **on-policy state coverage of off-manifold states**, which solely emulator data supplies
(DAgger / RWBC / perturb-recover / save-state GRPO) — **regardless of the VLM**.
- **What the VLM IS for in the drift regime:** (1) objective selection at genuine forks; (2) drift
  DETECTION / replan TRIGGERING (Q3); (3) long-horizon subgoaling (π0.5: text subtask→flow chunk) that SETS
  the target/reward for the emulator loop that repairs the actor. **NOT per-frame motor correction.**
- Honest synthesis (both): **VLM = System-2 objective + trigger; emulator on-policy data = System-1 drift
  cure.** The plan is load-bearing for CHOOSING WHAT TO DO, not for STAYING ON THE MANIFOLD while doing it.

**Q3 — Online OOD DETECTION + runtime GATE (not just train-to-recover). → AGREED, near-identical designs.**
Two **deploy-only, FREE** signals (already computed in `eval_policy._sample_chunk`):
- **(S1) CFG velocity-residual** `r_t = ‖v_c − v_u‖ / (‖v_u‖+ε)` at the final Euler step (both velocities
  already computed for CFG>1, eval_policy.py:237–239).
- **(S2) K-sample disagreement** `D_t` = mean pairwise L2 over K≈4 chunks from the SAME (frame,plan) via the
  existing SDE sampler (noise_sigma>0, eval_policy.py:240–242). Deploy analogue of deep ensembles
  (Lakshminarayanan et al. NeurIPS 2017).
- **(S3, emulator-only, TRAIN/VERIFY)** telemetry (Δpos≈0 / stuck / near-death) CALIBRATES the thresholds, then
  is discarded at deploy.
- **OPUS's KEY REFINEMENT — the SAME r_t disambiguates the two OODs:** **high r_t** = plan fights the frame-prior
  = plan↔frame contradiction = **plan-OOD (stale plan)**; **r_t≈0 WITH no progress** = frame-only actor
  confidently stuck = **actor-OOD (drift)**. One free signal separates "stale plan" from "drifted actor."
- **Gate ladder (cheap→expensive):** (1) `D_t>τ_D` OR (`r_t≈0` ∧ no-progress) for 2 consecutive chunks
  (debounce) → early replan + widen sampling; (2) `r_t>τ_r` (plan contradicts frame) → anneal CFG w→1.0
  (the consensus dynamic-CFG gate); (3) persistent → emulator pause-and-search (sample K plans/chunks from the
  save-state, rank by 1-step reward, commit best) — deploy-only fallback: commit lowest-`D_t` chunk.
- **Threshold:** conformal / distribution-free — `τ` = 95th percentile of the signal on known on-trajectory
  demo states (Angelopoulos & Bates 2021, arXiv:2107.07511) → ≤5% false-positive on good states. Emulator
  supplies the calibration set; deploy reads S1/S2 only.
- **Null-invariance preserved BY CONSTRUCTION:** under a null/masked plan the contract forces `v_c==v_u` →
  `r_t=0` → S1 inert; S2 reuses the plan-dropped sampler; the gate only READS and its actions (replan /
  change w / change noise_sigma) inject no plan info under null → base NitroGen stays bit-identical.

## THE ONE DISCRIMINATING EXPERIMENT (both proposed it independently; VERIFIED runnable, ZERO new code)
Goal: measure whether plan-OOD or actor-OOD dominates, with numbers, instead of more debate.
Tool: `planner_poc/rwbc_actor_adapt.py` on an emulator save-state game (SMW via `snes_env`; or `thextech`),
ckpt `btn_s600`. Two knobs isolate the levers — `--use-correct-plan` (plan held ORACLE & FRESH → removes
plan-OOD) and `--lora-only` (trains ONLY DiT-LoRA, plan-head FROZEN → reward change is PURE actor).
Metric = `eval_reward` mean episode return (2026-06-28 survival-weighted progress), seeded from 3–4
save-states. 3-rung ladder:
- **A.** Live plan, no adapt → R_A.
- **B.** Oracle plan, no adapt (`--use-correct-plan`, eval-only) → R_B.  **Δ_plan = R_B − R_A** (plan-OOD lever).
- **C.** Oracle plan + actor-adapt (`--use-correct-plan --lora-only --save-delta`) → R_C.  **Δ_actor = R_C − R_B**
  (PURE actor-OOD recovery; plan path frozen & oracle).
**Decision (both agree):** `Δ_actor ≥ +0.10 ∧ Δ_plan ≤ +0.05` → actor-OOD dominates, plan NOT load-bearing →
invest emulator coverage (BOTH models' prediction). `Δ_plan ≥ +0.15 ∧ Δ_actor ≤ +0.05` → plan-OOD dominates,
fresher plan IS the lever → invest plan freshness / flag VLM-LoRA.
NOTE (Opus): the seed's RWBC>base is CONFOUNDED (trained plan-head + live plan); `--lora-only
--use-correct-plan` removes BOTH confounds and decides it cleanly.
VERIFIED 2026-06-28: `--lora-only` (rwbc_actor_adapt.py:165, freezes plan-head :188), `--use-correct-plan`
(:170), `--save-delta` (:171), and `save_state/load_state` (emulator_env.py:199/203, snes_env.py:189/193)
ALL EXIST → the experiment runs on existing code.

## Referee note
Independent convergence (no rebuttal needed): both models, starting from opposite R1 instincts, landed on the
SAME stance — actor-OOD (BC covariate shift) is the owner's "subtle drift," the plan is NOT the lever for it
(near-zero plan gradient is decisive), emulator on-policy data is the only cure, and the VLM's drift-regime
role is objective + detection/trigger, not motor correction. Two refinements worth keeping: (a) Opus's single
free signal `r_t` that DISAMBIGUATES stale-plan (high r_t) from drifted-actor (r_t≈0 + no-progress); (b) the
`--lora-only --use-correct-plan` de-confounding that turns the seed's confounded RWBC>base into a clean
plan-vs-actor decision. ACTIONABLE NEXT: run the 3-rung discriminator (~30 min, existing code) — it resolves
the question with data and tells us whether to invest emulator-coverage (predicted) or plan-freshness.

---

# ===== ROUND 4 (2026-06-28): from diagnosis to PRESCRIPTION + RED-TEAM + the DATA =====
Seed: ARCH_WARROOM_R4_SEED.md. Roles: Opus = constructive prescription (ARCH_WARROOM_opus_r4.md);
GPT-5.5 = adversarial red-team (ARCH_WARROOM_gpt55_r4.md). Then a discriminator EXPERIMENT was run, and both
synthesized against the data (ARCH_WARROOM_{opus,gpt55}_r4b.md). Full data: DISCRIMINATOR_RESULTS.md.

## THE EXPERIMENT OVERTURNED THE R3 OVER-GENERALIZATION — a clean GAME-DEPENDENT DISSOCIATION
3-rung discriminator (rwbc_actor_adapt.py, btn_s600, screen_x reward, n=10-12): Δ_plan=R_B−R_A (oracle vs
live plan, no adapt); Δ_actor=R_C−R_B (oracle plan + --lora-only actor adapt).
| game  | Δ_plan | Δ_actor | dominant lever |
|-------|--------|---------|----------------|
| SMW   | +0.20  | −0.01   | **PLAN-OOD** (oracle plan helps; lora-only actor-adapt does nothing) |
| Sonic | −0.18  | +1.90   | **ACTOR-OOD** (actor-adapt wins big; plan redundant/slightly hurts) |
R3's "actor-OOD dominates, plan never load-bearing" is TRUE for Sonic, FALSE for SMW. It is GENRE-SHAPED:
obstacle/decision-point games (SMW) → the PLAN is the lever; open speed/locomotion games (Sonic) → the
ACTOR (on-policy data) is the lever. By GPT-5.5's OWN R4 concession criteria (concede only if BOTH games show
Δ_plan≤+0.05 ∧ Δ_actor≥+0.10), SMW VIOLATES it → the red-team's hole was real.

## CORROBORATION + CAVEATS (checked, not asserted)
- **plan_authority_map** (new probe, ||chunk_correct−chunk_null|| at cfg=8, null-driven drift): plan authority
  is ~1.0–1.4 (NON-zero) at BOTH on-manifold and drift states for both games, only MILDLY drift-correlated
  (SMW +14% at high-D_t; Sonic flat). ⇒ the plan is NOT inert at deployment CFG (the "near-zero gradient" is
  a cfg=1/on-manifold statement), but **authority ≠ benefit** — the behavioral reward is the ground truth.
- **Sonic +1.90 is not purely a screen_x hack:** it survives the survival-weighted reach_eval (rwbc 0.584 >
  base 0.250). BUT raw screen_x saturates at 3.22 → still warrants a video/held-out audit (red-team guardrail).
- **Robustness:** single seed (seed0=100), high per-episode variance. Both models want ≥2 more paired seed
  blocks + bootstrap CIs before the dissociation is final.

## CONVERGED REFINEMENT (both r4b, independent) — replaces R3's single answer
Plan-vs-actor OOD is **game/state-dependent, not globally actor-dominated.** Keep the ONE-generalist,
frozen-VLM-first, null-invariant architecture, but make the loop SELF-MEASURING:
- **Keep plan-head + PlanAdapter trainable GLOBALLY — reject any global `--lora-only`** (it ZEROED SMW). One
  fixed knob cannot serve both genres.
- **Per-game GRPO advantage-norm** `Â=(R−mean_g)/(std_g+ε)`: makes SMW's small-but-real +0.20 plan-gradient
  and Sonic's large +1.90 actor-gradient coexist in ONE shared LoRA+plan-head (unitless ⇒ Sonic's big raw
  screen_x can't swamp SMW). Routing is EMERGENT (no per-game heads ⇒ stays one generalist).
- **SMW (plan-OOD) = train plan_head+PlanAdapter (not lora-only)** under save-state GRPO; likely the repo's
  known ADAPTER-COLLAPSE pathology (left/right 0.50→0.688) — run the token-separation probe on SMW jump/wait
  tokens, clean contrastive labels, widen K; wire **replan to the DRIFT GATE** (authority rises at high-D_t),
  but optimize on emulator reward (authority≠benefit).
- **Sonic (actor-OOD) = actor GRPO** + guardrails for the 3.22 ceiling: multi-component reward
  (screen_x+survival+checkpoint), clamp per-branch screen_x advantage, cap save-state reuse, KL/--anchor +
  entropy floor, video/held-out audit.

## THE CURE RECIPE (Opus R4, endorsed): critic-free save-state GRPO + gate-driven active DAgger
- Primary scheme = **save-state GRPO** (DeepSeek arXiv:2402.03300) as AWR-form (Peng arXiv:1910.00177)
  advantage-weighted flow-BC inside rwbc_actor_adapt.py (swap the quantile `rw` at :213 for group-relative
  Â). Frame-exact save/load makes the group mean a zero-variance baseline ⇒ NO critic (the
  frame→plan→18-action→sparse-reward credit path kills value nets). RWBC = warm-start.
- **Active loop:** the Q3 gate (r_t, D_t — free in _sample_chunk) fires on 2 drifted chunks → save_state →
  branch-search K≈8 plans × G≈4 × h≈6 → add best recovery to a priority buffer. **Emulator+reward IS the
  DAgger labeling oracle** (we have no queryable expert) → this is DAgger (Ross arXiv:1011.0686) without a
  human oracle. Null-invariance preserved (only conditional samples written).

## AGREED NEXT EXPERIMENTS (ordered; both models)
1. **[RUNNING] Missing discriminator cell:** SMW R_C with plan-head TRAINABLE (drop --lora-only). Settles
   whether SMW's oracle-plan lever is TRAINABLE INTO the actor or only oracle-available. One flag flip.
   (Also running Sonic full-adapt for symmetry.) → rwbc_oodfull_{smw,sonic}.pt.
2. **R_D = oracle-plan + actor-adapt + replan-AT-drift**, run on SMW first (Δ_interaction=Return_D−Return_C
   ≥+0.05, or gate-local escape_rate +0.10) — catches a plan×actor interaction the 3-rung ladder can miss.
3. **Robustness:** 2 more paired seed blocks + bootstrap CIs on Δ_plan/Δ_actor before trusting SMW=plan/
   Sonic=actor; add survival-weighted metric for SMW.
4. **Sonic video / held-out save-state audit** of the 3.22-ceiling rollouts.

## Referee note
The role-split (constructive vs adversarial) + a real experiment produced MORE than another agreement round:
it FALSIFIED the R3 single-answer and replaced it with a genre-shaped dissociation, while VALIDATING Opus's
self-correcting recipe (the data shows exactly why a global --lora-only is wrong). The plan IS load-bearing —
for obstacle games; the actor/on-policy data IS the lever — for speed games; one generalist with per-game
advantage-norm holds both. Remaining uncertainty (is SMW's plan-lever trainable into the actor?) is being
measured right now by the missing cell.

---

# ===== ROUND 5 (2026-06-28): the S1/S2 ATTRIBUTION problem in human demo narration =====
Seed: ARCH_WARROOM_R5_SEED.md. Positions: ARCH_WARROOM_{gpt55,opus}_r5.md. Trigger: the owner hand-narrated
his SMW demos (demo_explanations.md) and asked how much is System-1 vs System-2. OWNER'S CRITICAL REFINEMENT
(mid-round): "My narration is NOT natural — I did NOT narrate while playing; post-hoc narration is either too
sparse or too granular, with NO clear S1/S2 distinction in my brain. It's also temporally noisy — off by a
second or two, over/under-specifying a plan's timespan." (This is WHY he wanted YouTube real-time commentary.)

## UNANIMOUS CONSENSUS (both models, independently): reattribute ACTIONS, not WORDS.
The verbal S1/S2 split is NOT recoverable line-by-line — a person's report of *why* is a post-hoc theory,
not a causal trace (Nisbett & Wilson 1977, "Telling more than we can know"). The owner's "no clear
distinction in my brain" is EVIDENCE of this, not a gap to fix. But the split you actually NEED —
**frame-DETERMINED vs frame-UNDERdetermined** — IS recoverable automatably, by measuring it on the ACTION
DISTRIBUTION. The four methods are not rivals; they COMPOSE:
- **(c) frame-counterfactual filter = the LABELER/MASK (primary, both rank #1).** A narrated action is
  type-B (reflex) iff the FROZEN base DiT already reproduces it from the frame alone:
  ‖a_real − _sample_chunk(frame,"",null=True)‖ ≈ 0 (base_dit_perdim buttons AUC 0.82–0.96), equivalently
  ‖v_c−v_u‖≈0 at deploy (eval_policy.py:237–239). This is the ONLY falsifiable definition of
  "frame-underdetermined." type-A = the residual.
- **(a) decode-free VLM re-abstraction = the ENCODER (Opus's push).** Don't let the bridge ingest raw
  narration (it parrots type-B). Feed frames+narration to the FROZEN Qwen, take encode_multimodal(text_only)
  — the text tokens have already cross-attended the frames → plan authority + grounding, no decode, no
  parroting. Narration = GROUNDING HINT, never the plan target.
- **(b) temporal abstraction = a PRIOR (options, Sutton-Precup-Singh 1999).** Slow objective = slowest-
  varying component over the A-chunk window; dedup per-second flicker. But slow != S2 ("run right" is slow
  AND frame-obvious), so it PROPOSES candidates for (c), doesn't replace it.
- **(d) the honest FLOOR.** What's irreducibly unrecoverable from OFFLINE narration is exactly what needs
  ON-POLICY labels — DAgger without a human oracle (Ross 2011): the emulator, not the transcript, supplies it.

## KEY UNIFICATION (Opus, both endorse): the residual MASK *is* the type-A fraction → reproduces the dissociation FOR FREE
Q2 supervision target = NOT the narrated action (leaks type-B) but the frame-counterfactual RESIDUAL:
`L = Σ_i m_i · ‖v_θ(a_i, plan) − u_i‖²`, mask `m_i = clip‖a_real,i − a_base,i‖` (or per-dim 1−AUC).
Where the base DiT already nails the action, m_i→0 → plan gets NO gradient → this RESPECTS the verified
near-zero-plan-gradient finding instead of fighting it; null-invariance untouched; + existing distill_weight
InfoNCE to de-collinearize the K tokens. SMW vs Sonic falls out with NO per-game knob: m_i is large at SMW
decision points (enter-pipe?/chase-or-not/pit routing), ≈0 for Sonic "run right" → the loss auto-allocates
plan-gradient to SMW and starves Sonic, reproducing Δ_plan +0.20 vs −0.18. **The SMW/Sonic dissociation is an
information-ALLOCATION result, not "SMW is more cognitively System-2."**

## Q4 ARCHITECTURE (both): TIMESCALE split, not hard semantic S1/S2.
The owner's inability to separate S1/S2 in retrospect UNDERMINES a hard semantic (reflex-vs-deliberate)
module split — even the actor can't tag it — but SUPPORTS a TIMESCALE split, which is what the data has: a
fast per-chunk action stream + a slow per-A-chunk objective (Helix fast-visuomotor+slow-latent; π0.5
subtask→flow-chunk). CHALLENGE to the project framing: let the markov DiT own the FULL action distribution
(incl. deliberate-LOOKING but frame-determined moves); let the VLM own ONLY the slow frame-underdetermined
objective. The boundary is frame-determinacy (measured by c) at the slow timescale, NOT a cognitive partition.
Read this way, the project's frozen-DiT + slow-VLM-plan IS ALREADY a correct timescale split.

## How the OWNER'S CONSTRAINTS (post-hoc, noisy, entangled, not-natural) land on the consensus — it SURVIVES
- **Temporal noise (±1–2s = ±60–120 frames): mostly HARMLESS.** The residual is computed on ACTIONS at a
  frame, not on where the TEXT lands, so the per-game f_A statistic (stream-level) is INVARIANT to narration
  timing. Only per-LINE attribution is affected → mitigate with span [frame,frame_end) + a ±1–2s tolerance
  window (max/mean residual over offsets). Practical rule: use narration at SEGMENT/LEVEL granularity (which
  level, required-vs-bonus, route choices), NOT as tight per-second plan↔action training pairs.
- **Post-hoc + not-natural + "no clear distinction":** REINFORCES "reattribute actions, not words" — the
  whole point is to NOT trust the introspective S1/S2 tag. Natural real-time YouTube commentary would be
  better-grounded S2, but STILL needs the residual filter (and risks popular-game VLM-prior contamination).
- **Confabulation (C)/flair (D):** the residual mask starves them by construction (DiT does flair anyway →
  m_i low); route the owner's own "no real reason"/"for flair" flags → a NULL objective (null-invariance is
  the confabulation sink). Never CE-train on narration tokens.

## AGREED EXPERIMENT (both proposed nearly identically) — runs on existing repo, ROBUST to the timing noise
For each narration.json entry: `ρ_e = ‖base-DiT null chunk _sample_chunk(frame,"",null=True) − streamer real
action over [frame,frame_end)‖`. High ρ_e = frame-underdetermined = type-A.
- **Metric 1 (method validation):** AUC of ρ_e separating the owner's hand-flaggable type-A lines (pipe/
  chase/routing/setup) from type-B (dodge/stomp/duck/flair). CONFIRM if AUC ≥ 0.75 (filter recovers the
  split it was never shown); REFUTE if ≈0.5 → fall back to (d) narration-as-eval-only.
- **Metric 2 (unifying hypothesis, TIMING-NOISE-IMMUNE):** per-game stream-level f_A = frac(ρ > τ). Predict
  f_A(Fire-Emblem turn-based) > f_A(SMW) ≫ f_A(Sonic); Spearman(f_A, Δ_plan) > 0 (SMW +0.20, Sonic −0.18).
  CONFIRM if SMW f_A ≥ 1.3× Sonic f_A.

## Referee note
Cleanest convergence of the five rounds: both models, on a fresh problem, independently produced the SAME
framework — attribute actions not words (Nisbett-Wilson), the frame-counterfactual residual as the operational
type-A definition AND the training mask AND the contamination detector, a timescale (not semantic) split, and
the same validating experiment. The owner's "my narration is noisy/post-hoc/entangled" doesn't weaken it — it
is the very reason the action-based (not word-based) definition is correct, and the headline f_A metric is
invariant to the timing noise. NEXT: run the residual experiment (decisive, cheap, timing-robust).

## R5 EXPERIMENT RESULT (narration_residual.py, btn_s600, 2026-06-28) — WEAK / CONFOUNDED, but informative
Per-line residual rho = 0.7*|DiT_dir - your_dir|/2 + 0.3*|jump diff|, on your real SMW narration; + stream f_A.
- **Stream-level f_A (timing-immune):** SMW 0.268 vs Sonic 0.247 (ratio 1.09 = WEAK; below 1.3x confirm).
  mean_rho_dir SMW 0.241 vs Sonic 0.194 (ratio 1.24 — a modest directional signal, washed out at threshold).
  => the unifying hypothesis (type-A fraction >> for SMW) is NOT cleanly confirmed by a DIRECTION-only proxy.
- **Per-line ranking:** the BOTTOM/type-B end is CLEAN ("crouch to dodge bullet bill" 0.09, "jump too early,
  come back" 0.00, "right movement and jumping to collect coins" 0.00, "enter pipe by going right" 0.00 —
  base DiT matches you on reflexes). The TOP/type-A end is POLLUTED by (i) multi-intent lines ("veer left and
  THEN continue right" 0.68) and (ii) the user's flagged temporal noise — confirming per-line attribution is
  unreliable, exactly as the owner predicted.
- **ROOT CONFOUND:** the base DiT's left/right is KNOWN-WEAK (jlx~0.5 neutral / dithers, corr 0.49, leftward
  bias — prior memory). So "DiT_dir != your_dir" largely measures the DiT's non-commitment on direction, which
  is UNIVERSAL (both games), not SMW-specific. Direction alone can't discriminate type-A here.

### CORRECTED next step for the hypothesis (NOT blocked on more narration)
The direction proxy is too weak. The rigorous version (both models' Q2) uses the FULL-ACTION residual /
per-dim (1-AUC) from base_dit_perdim, or the CFG plan-authority ||v_c - v_u|| with the CORRECT plan vs null,
over buttons+BOTH sticks+jump — not left/right alone. That's a code change to narration_residual.py, not a
data/narration need. KEY IMPLICATION FOR THE OWNER: more hand-narration would NOT improve this — the signal is
action-based and subtle; narration quantity is not the bottleneck.

---

# ===== ROUND 6 (2026-06-28, night): the one-generalist RECIPE that should make plan-conditioning work =====
Seed: ARCH_WARROOM_R6_SEED.md. Positions: ARCH_WARROOM_{gpt55,opus}_r6.md. UNANIMOUS convergence on the recipe.

## THE RECIPE (both models, near-identical) — residual-gated, return-weighted, one generalist
- **Params:** train `plan_head.*` (resampler+PlanAdapter) + `lora_*` GLOBALLY. NEVER global --lora-only (zeroes
  SMW). Base DiT + Qwen frozen (mm_text_only decode-free prefill).
- **Reward:** `--reward-mode rtg --gamma 0.95-0.97` + per-game group-relative advantage `Â=(R-mean)/std`,
  positive-clamped (AWR arXiv:1910.00177 / GRPO arXiv:2402.03300). Decision metric = reach_eval survival-
  weighted headline, NOT raw screen_x.
- **THE UNIFIER — residual-weighted plan loss:** cache the FROZEN base-DiT null action per collected chunk;
  set the per-(B,H,25) `actions_mask` = clip((|a_chunk - a_base_null| - |a_base_null - a_base_null2|)/scale,
  floor, 1). Flows UNCHANGED into nitrogen.py:690 raw_loss*mask (ZERO model change). The plan-head/LoRA get
  gradient ONLY where the reward-selected chunk DEVIATES from the base-DiT frame prior (the type-A residual).
  Null-invariance untouched (mask reweights only the conditional target).
  WHY it kills the collapse: "sprint-into-pit" is frame-determined -> base DiT already emits it -> mask->floor
  on those dims -> plan-head CANNOT be dragged toward the suicide-sprint that collapsed SMW. Gates on the
  FRAME, not the noisy REWARD -> survives the seed variance.
- **Variance control (the real blocker):** fixed save-state starts (snes_env load_state) not relaunch; paired
  seeds + bootstrap CIs (Efron 1979); lower LR 3e-5 + --anchor 0.01 (L2-to-init trust region); report the
  low-variance no-adapt Δ_plan (R_B-R_A, std ~0.06) as the trustworthy anchor.
- **Data:** --use-correct-plan (clean plan channel) tonight; VLM text_only re-abstraction is phase 2.
- **FE/Minish:** don't wait for owner plans; generate from frozen VLM + validate via residual/Δ_plan. Defer
  off tonight's critical path. **Data:** SKIP YouTube tonight (blocker is seed variance + mask validation, not
  data scarcity); emulator save-states ARE the on-policy labeler (DAgger-without-oracle).

## CAVEAT from the fresh experiment (narration_residual2): the AUTO-ALLOCATION premise is WEAK
Game-level: full-action residual SMW/Sonic 1.14x (weak); plan-authority 0.97x (Sonic HIGHER -> authority
anti-correlated with benefit). So the residual mask does NOT cleanly auto-allocate SMW-vs-Sonic at the game
level. BUT its COLLAPSE-PREVENTION is a LOCAL per-dim/per-chunk mechanism that does NOT need game-level
separation -> the recipe is still worth testing. The empirical C2(naive)-vs-C3(residual) ablation is the real
test, not the proxy ratio.

## ABLATION RUNNING NOW (multi-seed, the deliverable)
C2 = naive full-adapt (rtg+advnorm, ones-mask) vs C3 = + residual-mask. SMW seeds 0-4 (collapse test),
Sonic seeds 0-2 (no-harm). Native env reward R_pre(=C1 no-adapt plan baseline)/R_post per run, eval-eps 12.
PASS (synthesized from both): C3 should NOT collapse SMW (the C2 failure mode), C3 mean(Δ) > C2 mean(Δ) on
SMW with paired-seed CI, and Sonic not harmed. Deltas saved to files/r6_{game}_{c2,c3}_s{seed}.pt.
Then R6b: feed results back to both models.

---

# ===== ROUND 7 (2026-06-28 night): the R6 recipe FAILED -> diagnose + PIVOT (both own it) =====
Seed: ARCH_WARROOM_R7_SEED.md. Positions: ARCH_WARROOM_{gpt55,opus}_r7.md. The R6 recipe was implemented,
ablated multi-seed, and REFUTED. Both models converged on the diagnosis + pivot, and BOTH explicitly owned
the failure ("The data is the boss" / "I was wrong about the residual mask").

## DIAGNOSIS (unanimous)
1. **Residual-mask-as-loss-gate is DEAD.** Anchor-removal confirmed + a second mechanism (Opus): the loss
   `Σ(m·raw)/Σm` (nitrogen.py:690-693) both (a) down-weights frame-determined dims ~20x (floor 0.05) removing
   the implicit KL-to-base trust region the ones-mask provided, AND (b) renormalizes by Σm so the surviving
   deviation dims get a LARGER effective LR — amplifying gradient on exactly the reward-hacked deviation dims
   the top-frac selected. The R5 "collapse-prevention" logic was backwards.
2. **rtg+advnorm is the BIGGER culprit** (−0.9 swing) than the residual mask (−0.149). advnorm positive-clamps
   the group-relative advantage → concentrates the update on the most-extreme screen_x sprints (the suicide
   mode). Confirmed by existing data: PLAIN LOCAL RWBC full-adapt HELPS SMW (Δ+0.236, 3/4 seeds) while
   rtg+advnorm COLLAPSES it (Δ−0.489, 0/5). Each "sophistication" made it monotonically worse.

## THE PIVOT (unanimous): STOP actor-adapting plan-OOD games; SUPERVISED demo-fit instead.
The residual was RIGHT as a SUPERVISED TARGET WEIGHT and WRONG as a self-imitation loss gate. For plan-OOD
games the clean signal is the inference-time frozen-plan benefit (Δ_plan≈+0.20); reward-RWBC corrupts it.
**Run (a): supervised plan-head fit to HUMAN demo actions, conditioned on the correct plan, no env reward,
no drift, no seed-variance.** This is the residual idea resurrected in its correct (trusted-target) form, and
it auto-allocates: Sonic's residual≈0 → near-no-op; SMW's is large → plan-head learns. ONE supervised
objective = the true unifier (RWBC was the wrong target because SMW/Sonic want opposite SELF-IMITATION
updates; supervised doesn't pull them apart). RWBC stays as a 2nd training MODE for actor-OOD games that must
EXCEED the demo (Sonic +1.9). One model, two modes, routed by measured Δ_plan/Δ_actor — no per-game weights.

## PASS criteria (synthesized from both)
- SMW Δ_plan(post) ≥ +0.30 with 95% bootstrap-CI lower bound > +0.20 (beats frozen plan-head ~0.06 noise),
  ≥5 seeds. (The advance: plan channel IMPROVED, drift-free.)
- No-harm guard: Sonic within ±0.05 of frozen (residual≈0 ⇒ ~no-op); SMW reach_eval ≥ frozen (no collapse).
- Null output bit-identical (masked null-mode + apply_null_mask); delta additive/recoverable.
- HONEST RISK (Opus): base DiT already reproduces ~81-83% of the semantic action ⇒ the type-A residual is
  SMALL ⇒ gains may be modest. Even a reproducible small +Δ_plan is the win after RWBC's seed-variance debacle.

## ORCHESTRATOR EXECUTION (tonight)
- Implemented the pivot in planner_poc/demo_bc.py (additive flags: --train plan_head|both, --use-correct-plan,
  --residual-mask [supervised target weight], --save-delta, --seed-offset; Δ_plan = plan-advance − null-advance
  from FIXED demo start states = LOW-VARIANCE eval). Console→25dim mapping already existed.
- Running in parallel: plain-local RWBC confirmation grid (5 SMW + 3 Sonic) to LOCK the simplest-recipe win.
- NEXT: multi-seed demo_bc (plan-head, correct plan) ± residual ablation; compare to frozen Δ_plan + plain-local.

---

# ===== ROUND 8 (2026-06-28 night-2): GENERALIZATION + scaling — BOTH PASS =====
Seed: ARCH_WARROOM_R8_SEED.md. Positions: ARCH_WARROOM_{gpt55,opus}_r8.md. Both converged; experiments run.

## RESULTS (the recipe generalizes AND scales)
1. **MMX (new game) per-game demo-fit:** Δ_plan +34.2->+74.2 (+40, 3/3 seeds, +117%, CI [+23,+54]). The R7
   supervised plan-head recipe TRANSFERS to a new side-scroller out of the box (cleaner than SMW's 4/5).
2. **POOLED generalist (ONE plan-head fit on SMW+MMX+SMB1):** widens Δ_plan on BOTH eval games, 3/3 seeds:
   SMW +26 (CI [+18.9,+30], 1.3x per-game = pooling HELPS SMW), MMX +31 (CI [+17,+45.5], 0.78x per-game).
   Null-invariance 0.0 exact. => one supervised objective over pooled obstacle-game demos generalizes; same-
   mode deltas combine via POOLED FIT (not averaging/sequential), as both models predicted.

## CONVERGED ANSWERS (both models, near-identical)
- Q1 generalization: PREDICTED yes (reward-free, game-agnostic supervised fit); CONFIRMED on MMX. Falsify if
  mean change <=0 or CI includes 0 -> neither happened.
- Q2 SMB1 eval: pick (a) 2-byte page+offset (256*hi+lo) monotone scan, segment on level-resets, CV across >=3
  demos -> smbas_progress.json (<u2); fallback (d) SMB1 train-only + eval on MMX. (Tonight used the fallback:
  SMB1 pooled into TRAINING, generalization evaluated on MMX+SMW -- clean.) Rejected score/coins/time + optical-flow.
- Q3 multi-game scaling: ONE POOLED demo-fit (plan text is the disambiguator; frame-determined chunks self-
  attenuate across games). Averaging = fallback, sequential = rejected (catastrophic forgetting). CONFIRMED.
- Q4 3D prep (Mario64/OoT): the supervised plan-head demo-fit TRANSFERS wholesale (reward-free); BREAKS =
  screen_x eval (-> 3D distance-to-waypoint / star-room-count RAM), directional plan (-> landmark-relative
  "reach the door", plays to the VLM), markov frame (3D is MORE plan-OOD -> plan channel more load-bearing).
  Minimal adaptation: new in-process N64 env on emulator_env.py base + a 3D progress var. Everything else same.
- Q5: the pooled generalist (run + PASSED above).

## Referee note
Cleanest round yet: both models predicted generalization + pooled-fit-as-scaler, and BOTH ran true. The recipe
is now demonstrated game-general (SMW+MMX) and dataset-scalable (one pooled plan-head delta serves multiple
plan-OOD games with positive cross-transfer to SMW). The 3D-game guidance (swap eval to distance-to-waypoint,
plan to landmark-relative; demo-fit unchanged) is the ready plan for the owner's morning 3D demos. Remaining:
SMB1 clean x-address (2-byte scan) to add SMB1 as an EVAL game; multi-seed the full merge incl. Sonic+pooled.

---

# ===== ROUND 9 (2026-06-29): demo-fit FIXES planner-variance but DESTROYS expressiveness (ducking) =====
Seed: ARCH_WARROOM_R9_SEED.md. Positions: ARCH_WARROOM_{gpt55,opus}_r9.md. BOTH CONVERGED (cleanest yet).

## The trigger (decisive empirical finding)
The base bridge IS plan-steerable for the rare situational action "duck" (plan "press down to duck" raises SMW
DOWN-press 1.2%->10.8%, 9x). The validated pooled plan-head demo-fit, trained on ONE terse directional plan
that never says "duck", emits 0% DOWN for EVERY plan incl. explicit duck commands. It bought plan-following on
the trained axis (advance) by CATASTROPHICALLY OVERWRITING the bridge's plan-responsiveness for unnamed actions.

## CONVERGED ANSWERS (GPT-5.5 + Opus, near-identical)
- **Q1 root cause = DATA (conditioning collapse) > OBJECTIVE >> CAPACITY (refuted).** One constant terse plan
  makes the plan input ~constant, so plain BC's loss-optimal solution is the plan-INDEPENDENT marginal advance
  policy (duck 1.7% -> rounds to 0). K=8 is NOT the bottleneck (base already does duck-on-command at K=8); option
  4 (K=32) is dead (also breaks btn_s600 plan_head shape -> full retrain). Objective (BC mode-collapse) is the
  knife the data hands it. This is the SAME mechanism that fixed staleness: the bridge learned to IGNORE plan
  variation (great for staleness, catastrophic for expressiveness).
- **Q2 fix = (1) RICHER PER-SITUATION PLANS, with THE ACTION AS THE LABEL.** Per-chunk plan derived from the
  chunk's own action (DOWN-held->duck, LEFT->retreat, idle->wait, airborne->jump, else->advance terse). ~15 lines
  in load_demo_chunks; build_batch already encodes per-sample plans. NO VLM/narration needed (the action labels
  itself) -> generalizes to MMX/SMB1/every game. Both gave near-identical SIT dict + thresholds (duck @ >=3/18).
- **Q3 diagnostic-first = YES, and it IS the fix** (one re-fit forks data-vs-objective AND ships the fix). Opus
  upgraded it to a CROSSED parallel design: P1A0 (situational plans, plain BC) || P0A1 (single terse + KL-anchor)
  || P0A0 (terse control). If data alone works -> ship richer plans; if only KL works -> objective; both fail ->
  zero-init modulator (option 3, already built planner.py:305-309).
- **Q4 expressiveness battery:** 5 plan->action contrasts (duck/retreat/jump/wait/climb-up) the BASE obeys;
  sign-match >=4/5, responsiveness index R=mean clip(d_i/b_i,0,1) >=0.5, duck mandatory >=5%. Catches whole-action
  collapse, not just DOWN; standing regression gate on every future pooled fit.
- **Q5 scaling:** the fix IMPROVES the generalist story (the R8 one-plan-per-game pooled fit is what CAUSED the
  collapse). Re-pool with per-chunk action-labels, same machinery, null-invariance intact. VLM mints labels for
  free at scale (Tier-1 augmentation); Tier-0 action-as-label needs neither VLM nor narration tonight.

## Pass bar (both agree)
Under "press down to duck": demo-fit DOWN >= 5% (base 10.8%, collapse 0.0%), MONOTONE in plan-explicitness;
WHILE SMW Δ_plan POST >= +20 (>=0.8x pooled +26, bootstrap CI>0, 3/3 seeds); null max|diff| = 0.0 exact.
Falsifier: richer plans give press-down DOWN <2% -> DATA refuted -> ship KL-anchor. Duck returns but Δ_plan
POST <+20 -> escalate to zero-init additive modulator.

## ORCHESTRATOR EXECUTION (tonight, in flight)
Implemented additively in planner_poc/demo_bc.py: --situational-plans (label_chunk + SIT_PLANS), --rare-oversample,
--duck-probe (DOWN% stick dim22 + dpad dim1 vs terse/dodge/press_down, PRE+POST), --kl-anchor LAMBDA (functional
L2 of plan-tokens to a frozen pre-fit plan_head over a broad plan dist incl duck) + build_anchor_pool. KEY DIM
NOTE: map_action encodes the human DOWN as the STICK (JLY dim22), not dpad_down (dim1) the orig probe used -> the
probe measures BOTH channels. Running the crossed cells on 4xA6000: P1A0 situational (GPU0), P0A0 control (GPU2),
P0A1 KL-anchor (GPU3); 3 seeds each as GPUs free. Results -> DISCRIMINATOR_RESULTS.md.

## R9 EXECUTION RESULTS (the data picked the winner — KL-anchor)
Ran the crossed design (3 seeds). DPAD/STICK duck channels + a 5-contrast expressiveness battery.
- Control (collapse recipe): Δ_plan -4.9, duck 0%, battery R=0.165 (kills wait/up/retreat too -> whole-action
  collapse, as predicted).
- Situational (action-as-label): Δ_plan +32.8 mean (3/3), duck STICK press_down 12.6-98.2% MONOTONE (restored),
  battery R=0.53 -> keeps jump+wait+duck, LOSES retreat+up (taxonomy treadmill: 'up' unlabeled, exactly Opus's
  steelman risk).
- **KL-anchor (terse + functional L2 of plan-tokens to pre-fit head): Δ_plan +45.1 mean (3/3, tightest), duck
  restored monotone, battery R=0.83 = the only PASS, preserves retreat+up+wait.** WINNER on BOTH axes; keeps the
  cheap single-plan data; anchor loss ~0.001 (non-distorting). Opus's taxonomy-free steelman was right.
Referee note: cleanest causal round yet. Control reproduced the collapse on command; both proposed fixes worked;
the crossed design let the DATA choose between data-fix (situational) and objective-fix (KL-anchor) -> KL-anchor.
ADOPT: --kl-anchor 0.3 as the demo-fit objective; re-pool the generalist with it (optionally stack situational
labels for extra stick-duck authority). Remaining: re-pool 4-game generalist w/ KL-anchor + re-merge Sonic LoRA;
port the expressiveness battery to MMX/SMB1; the zero-init modulator (option 3) stays the guaranteed-safe fallback
but is NOT needed (KL-anchor already preserves + advances).

---

# ===== ROUND 10 (2026-06-29): SHORT vs LONG path, System-2 -> System-1 consolidation =====
Seed: ARCH_WARROOM_R10_SEED.md. Positions: ARCH_WARROOM_{gpt55,opus}_r10.md. BOTH CONVERGED.

## CONVERGED FRAMING
- The owner's "short path / long path, System-2 becoming System-1" = **capability EVOCATION vs ADDITION**.
  SHORT path (evocation) = KL-anchored plan-head/bridge summons a maneuver the DiT's manifold ALREADY contains
  (duck worked: 1.2->10.8% on command). LONG path (addition) = DiT-LoRA EXPANDS the manifold for a maneuver the
  DiT can't synthesize (grab-mesh climb). Unifies stale-plans + actor-OOD + capability-transfer.
- **Router = 2-STAGE GATE (Opus sharpening; both agree the 1-probe AUC router is WRONG).** null-AUC measures
  DEFAULT REFLEX, not MANIFOLD MEMBERSHIP -> it would misroute duck (9x evocable but low default rate) to
  "addition". Gate: (1) null-AUC>=0.6 -> already reflexive, NO-OP; (2a) AUC<0.6 AND evoc-ratio>=2x -> EVOKE
  (short path); (2b) AUC<0.6 AND evoc-ratio<2x AND emulator-outcome-Δ<+0.3 -> ADD (long path). For OUTCOME
  maneuvers (spin-jump-kills-Rex) the button probe is insufficient -> decide on the EMULATOR counterfactual
  P(Rex removed & Mario survives | plan) - P(.|null) from a save-state AT a Rex.
- Q2 consolidation = save-state RWBC with a narration-conditioned actor (NOT a new objective): learn-mode MINTS
  System-2 hypotheses; distill the BEHAVIOR they produce via KL-anchored plan-head-only BC on top-RTG winners.
  Guards (R9 variance lesson): plan-head only (never LoRA RWBC), >=5 seeds + CIs, RTG+death-penalty, BC-flavored.
- Q3 long path = dedicated rank-16 DiT-LoRA on the router-"add" chunks ONLY (K=32 dead; FastPlanMod = short-path
  authority amplifier, can't synthesize a new trajectory). Coexists with KL-anchored plan-head (R8/R9 disjoint-
  merge proven); keep null-invariant via masked-null + action-level bit-identity check.
- Q5 Super Metroid into pooled training HELPS under per-chunk situational/narration conditioning + KL-anchor
  (adds left/up/down/aim/morph coverage); HURTS only under a global "advance" plan (wrong conditioning anyway).

## THE ONE EXPERIMENT (both agree): maneuver ROUTER (frozen, decisive) + companion narration demo-fit
PRIMARY/GPU0 (zero training): reconstruct frame-exact save-states at duck / spin-jump-Rex / grab-mesh (replay
demo.npz actions to the narrated frame from narration.json), measure null-AUC + evoc-ratio + emulator outcome-Δ
on the FROZEN DiT -> classify each maneuver. Answers the owner's literal questions; gates Q3.
COMPANION/GPU1-3 (3 seeds): narration-conditioned KL-anchored demo-fit vs terse vs SIT.

## *** ORCHESTRATOR FLAG: companion experiment COLLIDES with owner guidance (2026-06-29) ***
Both agents' COMPANION (Q4) wires the RAW gold narration in as training plan text. But the owner JUST said: do
NOT use the hand-written explanations directly -- too granular, mixes System-1/System-2, noisy; they must be
ABSTRACTED into clean System-2 plans (VLM-generated objectives), used as a SOURCE not as literal training plans.
RECONCILIATION (orchestrator): (1) the maneuver ROUTER's use of narration is FINE -- it only uses frame-alignment
to LOCATE the maneuver moments (save-state reconstruction), not as plan text. (2) Replace the Q4 "A2=raw gold
narration" arm with "A2=VLM-abstracted System-2 plan" (frozen Qwen reads the segment/frame -> a clean grounded
objective at the right abstraction level), keeping A0=terse / A1=SIT as baselines. Pending owner's call on
sequencing + whether to generate VLM-abstracted plans now.

---

# ===== ROUND 11 (2026-06-29): LITERATURE-GROUNDED (TTT & friends) — BOTH CONVERGED HARD =====
Seed: ARCH_WARROOM_R11_SEED.md. Lit: LIT_R11_{ttt,distill,fastweights}.md (3 research agents, arXiv-verified).
Positions: ARCH_WARROOM_{gpt55,opus}_r11.md. Cleanest lit-grounded convergence of the project.

## THE CENTRAL LIT-GROUNDED INSIGHT (both agents, independently)
**The continuous SHORT PATH is ALREADY test-time training.** von Oswald 2023 (arXiv:2212.07677): cross-attention
over in-context tokens = implicit gradient descent. Schlag 2021 (arXiv:2102.11174): resampler->K-token write +
DiT cross-attn read = a fast-weight programmer. => the K=8 plan tokens are an implicit per-step optimizer over
the frozen DiT. CONCLUSION: lean on it FORWARD-PASS; do NOT bolt on explicit weight-TTT (heavier + inherits the
R9 collapse).

## CONVERGED ANSWERS
- **Q1 (stale plans): REJECT weight-TTT (TENT/SAR entropy-min)** -- entropy is undefined for our continuous flow
  velocity (the lit's OWN confirmed gap), and its single-sample collapse mode IS the R9 duck->0% collapse. Both
  agents (incl. Opus contradicting LIT_R11_ttt's own headline "SAR Recipe A") PICK a **zero-training forward-pass
  plan-token TTA**: re-encode the K tokens on the FRESH frame every chunk (`fresh`, Gandelsman video-locality
  arXiv:2307.05014) + EMA pooled token + M=4 aug-average across replans (`fresh_ema`, CoTTA 2203.13591 + MEMO
  2110.09506). Collapse-immune (no optimizer). Self-supervised signal = temporal/aug CONSISTENCY of the encode
  (+ emulator forward-consistency for Tier-2), NOT entropy.
- **Q2 (consolidation) = ReST-EM** (Singh 2023, arXiv:2312.06585), EXACT match to our save-state->score->KL-
  anchored-BC loop (not literal ExIt: we have no tree/soft-target search; not STaR: RTG is a noisy proxy).
  Borrow: restart-from-BASE each round (our KL-anchor-to-base IS this), reward-tau=p75 + >=1000 winning chunks/
  round, KL-anchor ~ soft-target distillation, STaR rationalization for the counterfactual-override case. The R10
  router IS the ReST-EM E-step (mint+roll+score) for free; M-step gated on router-confirmed-evocable maneuvers.
- **Q3: LEAN ON THE CONTINUOUS PATH.** Our R9 KL-anchor is literally CoTTA stochastic-restoration (2203.13591) in
  functional/output-space form (anchors the plan->token MAP, so duck survives). CFG scale = Hopfield retrieval
  temperature beta (Ramsauer 2008.02217), a free test-time knob. Both UN-PARK FastPlanMod (AdaLN-zero, DiT
  2212.09748 + FiLM 1709.07871) as a principled null-safe SECOND fast-weight channel (Opus explicitly reverses
  his R10 "merely an amplifier").
- **Q4 (addition): DAgger** (Ross 2011, arXiv:1011.0686; O(eT^2)->O(eT), ties to our EXP-011 left-then-right SEQ
  failure) + **plan-token-GATED DiT-LoRA** (gate to plan-token KEY positions -> null path has no plan tokens ->
  exact null-invariance BY CONSTRUCTION, the fix lora.py itself names) + **EATA Fisher/EWC** forgetting guard
  (Niu 2022, arXiv:2204.02610). RT-H/LAPA "language motion" = our VLM-abstracted plan. Train on ADD-chunks only,
  disjoint-merge (R8/R9 proven).
- **Q5: ONE architecture:** sleep-time ReST-EM over the fast-weight short path; Fisher-guarded gated-LoRA long
  path (router-gated, ADD only); forward-pass TTA for staleness. NO online weight-TTT on the actor. One
  generalist, VLM-frozen-first, null-invariant.

## THE ONE EXPERIMENT TONIGHT (both agree): staleness FORWARD-PASS TTA test
Extend planner_poc/staleness_probe.py with `fresh` (hold the t=0 VLM TEXT, re-encode K plan tokens on the live
frame every chunk) and `fresh_ema` (+EMA pooled token beta=0.7 + M=4 aug-average). Run SMW+MMX, >=3 seeds, FROZEN
btn_s600 (zero training -> collapse-immune). Tests von Oswald's "short path is already TTT" on the owner's #1
bottleneck (50-60% replan flip). Runs IN PARALLEL with the queued R10 maneuver-router (both frozen, ~1 GPU each).
PASS: fresh recovers >=50% of (oracle-stale) advance on SMW+MMX; fresh_ema flip-rate <=0.7x live; advance(fresh_
ema)>=advance(live); null action-bit-identity=0.0. FALSIFIER: fresh~=stale on both -> staleness is TEXT-bound ->
escalate to emulator-supervised plan-head weight-TTT (Tier-2, =the ReST-EM M-step, KL-anchored, >=5 seeds).

## Minor disagreement (sequencing only)
GPT-5.5 wants to also be ready to run emulator-supervised per-episode weight-TTT (ReST-EM M-step) tonight; Opus
insists the zero-training forward-pass test goes FIRST (if weight-TTT runs first it'll collapse ~50% of seeds ->
wrong conclusion "TTT doesn't work"). Both agree tonight's PICK is the forward-pass staleness test. Emulator
P(progress) is the hack-proof metric (not behavioral flip-rate alone) -- folded in.

## Lit CONTRADICTIONS flagged (good critical grounding)
1. Seed's "pick a weight-TTT variant" vs the fast-weight lit (short path already = implicit TTT).
2. LIT_R11_ttt's own "SAR Recipe A" headline vs its own collapse caveats + R9.
3. Opus R10 "FastPlanMod = parked amplifier" vs DiT/FWP lit (it's a principled null-safe channel).
4. "KL-anchor = generic regularizer" vs it being specifically CoTTA stochastic-restoration (unlocks +Fisher path).

---

# ===== ROUND 12 (2026-06-29): gold-action VLM plans + capability-addition path — BOTH CONVERGED =====
Seed: ARCH_WARROOM_R12_SEED.md. Positions: ARCH_WARROOM_{gpt55,opus}_r12.md. Strong convergence.

## CONVERGED CONCLUSIONS
- **TONIGHT = gold-action VLM plans as KL-anchored demo-fit conditioning** (both pick this over the Rex add).
  3 arms x 3 seeds on SMW (+MMX winner-replication): A=terse+KL (R9 winner control), B=gold-action-VLM-plans+KL
  (treatment), C=gold-VLM-plans WITHOUT KL (ablation: does per-chunk specificity alone prevent collapse?).
  Wiring: new demo_bc.py --vlm-plans (map each chunk start-frame to the covering vlm_narration.json entry;
  fallback BATTERY['correct']). Label pass: regenerate all SMW vlm_narration.json with --interleave-actions
  + --game smw (the existing ones are STALE/empty). Big VLM = offline TEACHER only.
  PASS (B): Δ_plan>=+40 (CI>0, not below A's +45.1) AND stick-duck (dim22>0.6) monotone->=30% under explicit
  duck plan AND expressiveness battery R>=0.80 (not regress to situational's 0.53) AND null bit-identity=0.0.
  FALSIFIER: R(B)<0.80 or Δ_plan<+40 -> gold narration re-imported the taxonomy-treadmill loss -> DEMOTE gold
  to a duck-authority STACK on terse+KL. (The B win = situational's trained-stick-duck WITHOUT its breadth tax,
  because gold vocab is OPEN and KL holds breadth.)
- **Why Q1 before the Rex add:** the duck-collapse (plain BC battery R=0.165) is a GENERALIST-BLOCKING regression
  every multi-game demo-fit inherits; Q1 reuses validated R9 infra at zero new tooling AND tests R11's text-
  quality redirect; the Rex add is one maneuver/one game at near-chance SNR (3.75->7.5%) needing an UNBUILT
  sprite-status kill-detector (SMW score doesn't register kills) -> multi-night, not tonight.
- **Q2 capability ADDITION (Rex), concretized + DEFERRED:** plan-token-GATED DiT-LoRA (gate to plan-token key
  positions -> null-exact by construction) on the human Rex-kill chunks, conditioned on the gold-action
  'spin-jump the Rex' plan; EATA Fisher + DAgger (collect at policy-visited save-states) forgetting guard;
  disjoint-merge onto the Q1 plan-head generalist (R8/R9 proven). Eval = emulator P(survive+advance past Rex),
  needs a sprite-status RAM kill-detector built FIRST + better save-state reconstruction (seed nearer the Rex).
  >=5 seeds for any outcome claim.
- **Q3 plan-text quality (the R11 staleness lever) RANKED:** (1) HIGHEST = Qwen-LoRA DISTILL the gold-action
  plans into the 2B DEPLOYMENT planner (we have NO gold actions at test time -> distill so the live 2B emits
  gold-action-QUALITY text from frames alone; RT-H/LAPA 'language motion'). (2) replan-less/hold stable plan
  (--replan-every N + hysteresis; R11: live re-planning HURTS, MMX -11) = cheap stability band-aid, ship now.
  (3) big planner = offline TEACHER only (don't deploy 12B live). (4) gold-action-prompting-at-inference =
  REJECT (no gold actions at test time; it's a training/label tool only). Measure plan quality WITHOUT oracle =
  survival_advance(live_plan)-null (staleness_probe 'live' mode), shrink the oracle-live gap.
- **Q4 exploration demos (Metroid, Minish):** FOLD their chunks into the pooled plan-OOD generalist as POSITIVE
  plan-diversity (they supply the retreat/up/wait/climb/talk/pick-throw chunks platformers under-label -- exactly
  the battery contrasts situational LOST). EVAL needs a per-genre coordinate (Metroid metroidvania; Minish
  top-down = no advance axis -> room-id/dialogue/item-count RAM hunt) -> defer eval, use for training diversity
  first. The gold-action auto-narration of the un-annotated Ridley boss is the key generalist test (auto-label
  the parts the human couldn't) -- BUT it is UNVALIDATED (see flag below).

## ORCHESTRATOR FLAGS (corrections to the agents)
1. Opus flagged the OLD vlm_narration_ridley.json was ALL-EMPTY (a failed/aborted run). RESOLVED THIS SESSION:
   the empty was a flush bug + the SMW-vocab/mapping bugs; after fixing per-game control schemas (mmx/metroid/
   sonic/smbas all had SCRAMBLED button semantics) + the generic-instruction SMW-vocab leak + incremental JSON
   write, the Ridley auto-narration WORKS: 'Dash and jump to avoid Ridley's attack', 'Dash to escape the self-
   destruct sequence', 'fire at Ridley while maintaining distance'. So Q4(ii) is now VALIDATED, not unproven.
   Gold-action narration DOES auto-solve the un-annotatable boss. (vlm_narration_ridley_fixed.json)
2. Per-game button semantics are now VERIFIED against user manuals + demo usage for all 5 games (stored in
   memory). The gold-action labels are only as correct as these -- earlier MMX/Metroid/Sonic schemas were wrong.

## Referee note
Cleanest path-to-execution round. Both converge: gold-action VLM plans (today's breakthrough) become the
training conditioning TONIGHT via the R9 KL-anchor, with a crossed A/B/C design + a hard battery-R falsifier;
the Rex capability-addition is correctly deferred behind a sprite-status detector. Q3 (Qwen-LoRA distill to the
2B deploy planner) is the named successor that closes R11's text-quality gap. Open: build the --vlm-plans wiring
+ regenerate SMW labels, then run A/B/C.

---

## R13 — System-2 → System-1 KNOWLEDGE TRANSFER (forward report) — 2026-06-29
SEED: ARCH_WARROOM_R13_SEED.md. Positions: ARCH_WARROOM_{gpt55,opus}_r13.md. Lit: LIT_R13_S2TOS1.md (24 arXiv
papers verified). Report: FORWARD_S2_TO_S1.md.

**CONSENSUS (GPT-5.5 + Opus converged; lit agrees):**
- The transfer taxonomy is MEASURED: EVOCATION (S2 names a primitive S1 owns — duck 1.2%->10.8%, P1) vs
  ADDITION (precise outcome S1 lacks — Rex jump 7.2x evocable but kill-outcome flat, P2). Route by OUTCOME_Δ,
  not button proxy (Opus: only outcome_Δ is hack-proof; the router already proved this on Rex).
- **CORE RESOLUTION (Opus, by construction):** "consolidate the evoked skill into the policy" and "masked-null
  exactness" are the SAME object (apply_null_mask zeroes only _PLAN_TOKEN on a frozen DiT). You CANNOT distill
  instinct into the null. FIX: consolidate into the ALWAYS-ON _GAME_ID_TOKEN channel (shared embedding,
  padding_idx=0, never masked — nitrogen.py:264-269,602, code-verified) via a zero-init KL-anchored
  InstinctResidual. Skill fires WITHOUT the plan token (instinct, frees bandwidth); plan-relative
  null-invariance EXACT by construction; we relax only the CFG reference (null = base+instinct = the definition
  of instinct). Held-out game_id=0 -> base-exact preserved.
- **Q2 ranking:** (a) KL-anchored self-distill into game-channel [Distral 1707.04175 + LwF 1606.09282; = ExIt
  1705.08439 inner loop] FIRST > (c) skill-token codebook [LISA 2203.00054 + PRISE 2402.10450, BPE-merge =
  "chain combos -> one token"] SECOND > (b) fast-weights/hypernet [1609.09106] for Q4 execution-nuance, DEFER.
- **NOVELTY:** RT-H/SayCan/VOYAGER name skills but NONE amortize the named skill away. The "S2 stops naming it"
  step is ours (P5 + always-on channel).
- **THIS-WEEK EXPERIMENT (both, unanimous): CONSOLIDATE-DUCK** (NOT Rex — near-chance, needs unbuilt kill
  detector). New demo_bc.py --consolidate game_id (~30 lines, reuses KL-anchor + duck_probe). Pass: null-policy
  ducks at bullet states +>=20pp, plan-marginal duck->0, expr R>=0.80, null bit-identity 0.0 elsewhere. Closeout:
  staleness_probe.py live with a duck-OMITTING plan still survives bullet-bills = proof S2 freed the budget.
- **LEAD (Q6):** P1->P5 (evoke -> consolidate-without-forgetting) as the recursion's first hop; P3/P4
  (one-generalist transfer/merge) + P7 (live-plan fix); honest caveats (oracle-plan, <=16-chunk, fixed
  save-states, closed-loop drift unproven, always-on consolidation designed-not-run).
- **DISAGREEMENTS:** Opus vs seed — consolidation goes to the game-channel, NOT the null (seed's parenthetical
  was right). Opus vs GPT-5.5 — only outcome_Δ routes outcome maneuvers (GPT-5.5 treated router features as
  co-equal); both rejected Rex-first, so Q5 is unanimous.
