## [OPUS48 — Round 3] 2026-06-28T04:17-07:00 — Actor-OOD drift & plan staleness (latency CLOSED)

**Stance in one sentence:** "Subtle drift on a trained game" is **DiT-actor OOD — behavior-cloning covariate
shift / compounding error — and the plan is NOT the lever that fixes it.** A fresher plan cannot move an
action the frozen DiT already determines from the frame alone; only on-policy emulator state coverage cures
drift. On this narrow question GPT-5.5's on-policy/DAgger framing is the stronger lever, and I concede it.
The VLM's drift-regime job is **objective selection + drift *detection*/replan triggering**, not motor correction.

---

### Q1 — Which OOD dominates: plan-conditioning OOD vs DiT-actor OOD? → **Actor-OOD.**

**Mechanism.** NitroGen is a behavior-cloned flow policy fit to near-expert streamer demos (off-policy,
on-manifold). Closed-loop, each 18-step chunk's small error nudges the agent off the demo manifold, where it
never trained, so error there is unbounded and **compounds** — the Ross–Bagnell reduction: BC error grows
**O(εT²)** in horizon T vs **O(εT)** on-policy (Ross & Bagnell, AISTATS 2010; Ross, Gordon & Bagnell, *DAgger*,
AISTATS 2011, arXiv:1011.0686). "Subtly different state on a trained game" **is** covariate shift —
in-distribution game, OOD *closed-loop visitation*.

**Why it's actor-OOD and not plan-OOD.** Verified repo finding: the frozen base DiT reproduces the streamer's
action **from the frame alone** (buttons AUC 0.82–0.96, sticks corr 0.4–0.76) → on factual, on-trajectory
states the **plan gradient ≈ 0**. If the action there is fixed by the frame, the drift cannot be a
plan-conditioning failure — it is the actor leaving the manifold. **In-repo evidence:** on-policy actor
adaptation (RWBC) beats base — Sonic 0.58 > 0.25, SMW 0.53 > 0.41 — the *signature of covariate shift*, not of
a plan defect. Plan-OOD is real but **secondary and categorically different**: it bites on novel *(frame ×
plan)* combos (stale/contradicting plans) as an *intermittent* failure at replan boundaries, already owned by
the consensus (staleness-offset + dynamic-CFG) — not the *continuous* drift the owner describes.

### Q2 — Is a better/fresher plan the right lever for actor-OOD? → **No.**

The near-zero-gradient finding is decisive: where drift happens (on/near-trajectory states) the action is
already determined by the frame, so improving the plan **cannot change it**. The plan only moves the action
where the frame is genuinely *ambiguous* (a real fork) — that is **objective selection**, the markov gap the
VLM exists to fill, **not drift recovery**. Drift is the actor emitting a slightly-wrong *continuous* action
that accumulates; the only fix is **state coverage of off-manifold states**, which solely on-policy data
(DAgger / emulator rollouts + reward) supplies — **regardless of the VLM.** GPT-5.5 is right here; I concede it.

**So what is the VLM actually for in the drift regime?** (1) **Objective selection at genuine forks** (frame
ambiguous → plan picks the branch) — the markov gap, not drift; (2) **drift *detection* / replan triggering**
(flag "frame contradicts plan" → fire the gate, Q3); (3) **long-horizon subgoaling** (π0.5: text subtask →
flow chunk; pi.website/blog/pi05) that *sets the target/reward* for the emulator loop repairing the actor.
**NOT** per-frame motor correction — the DiT's frame input + on-policy LoRA does that.

Honest synthesis: **VLM = System-2 objective + trigger; emulator on-policy data = System-1 drift cure.** The
plan is load-bearing for *choosing what to do*, not for *staying on the manifold while doing it*.

### Q3 — Cheap ONLINE OOD detection + a runtime gate (signal + gate + threshold)

We need detection, not just train-time recovery. Two **deploy-only, FREE** signals (already in
`eval_policy._sample_chunk`):

- **(S1) CFG velocity-residual** `r_t = ‖v_c − v_u‖ / (‖v_u‖+ε)` at the final Euler step — both velocities are
  *already computed* for CFG>1 (`eval_policy.py:237–239`). `r_t` **high** = plan fights frame-prior = plan↔frame
  contradiction (stale plan); `r_t ≈ 0` *with no progress* = frame-only actor confidently stuck = drift.
- **(S2) K-sample disagreement** `D_t` = mean pairwise L2 over K=4 chunks from the *same* (frame,plan) via the
  existing SDE sampler (`noise_sigma>0`,`noise_seed`; `eval_policy.py:240–242`). High `D_t` = flat/multimodal
  flow field = actor epistemic uncertainty/OOD — deploy analogue of deep ensembles (Lakshminarayanan et al., NeurIPS 2017).

**(S3, emulator-only, TRAIN/VERIFY)** telemetry (Δposition≈0, near-death, stuck from `env.step`) **calibrates τ**
for S1/S2, then is discarded at deploy.

**Gate ladder** (cheap→expensive):
1. `D_t > τ_D` **or** (`r_t ≈ 0` ∧ no-progress) for **2 consecutive chunks** (debounce) → **early replan**
   (`regenerate_plan`) + **widen sampling** (raise `noise_sigma`).
2. `r_t > τ_r` (plan contradicts frame) → **anneal CFG w → 1.0** (consensus dynamic-CFG gate).
3. Persistent `D_t` after replan → **pause-and-search**: emulator samples K plans/chunks from the save-state,
   ranks by 1-step reward, commits best. Deploy-only fallback: commit lowest-`D_t` chunk (conservative prior).

**Threshold (conformal, distribution-free):** `τ_D` = **95th percentile of `D_t` on known on-trajectory demo
states** (Angelopoulos & Bates, 2021) → ≤5% false-positive on good states; same for `τ_r`. Emulator supplies
the calibration set; deploy reads S1/S2 only.

**Null-invariance preserved by construction:** under a null/masked plan the contract forces `v_c == v_u` →
`r_t = 0` → S1 inert; S2 reuses the plan-dropped sampler (no new conditioning); the gate only *reads*, and its
actions (replan / change w / change noise_sigma) inject no plan info under null → base NitroGen stays bit-identical.

### The ONE discriminating experiment (existing repo; ~30 min)

On **one emulator save-state game (SMW via `snes_env`, or `thextech`)**, `btn_s600`, via
`planner_poc/rwbc_actor_adapt.py`. Two knobs isolate the levers: `--use-correct-plan` (plan held *oracle &
fresh* → removes plan-OOD) and `--lora-only` (only DiT-LoRA trains, plan-head frozen → reward change is *pure
actor*). Metric = `eval_reward` mean episode return (2026-06-28 survival-weighted progress). 3-rung ladder,
seeding `eval`/`collect` from 3–4 save-states for coverage:

- **A.** Live plan, no adapt → `R_A`.
- **B.** Oracle plan, no adapt (`--use-correct-plan`, eval-only) → `R_B`. **Δ_plan = R_B − R_A** = ceiling a
  perfect never-stale plan buys (plan-OOD lever).
- **C.** Oracle plan + actor-adapt (`--use-correct-plan --lora-only --save-delta`) → `R_C`. **Δ_actor = R_C −
  R_B** = pure actor-OOD recovery, plan path frozen & oracle.

**Decision numbers (what changes my mind):**
- **Δ_plan ≥ +0.15 ∧ Δ_actor ≤ +0.05** → plan-OOD dominates; a fresher plan *is* the lever → **I'm wrong** (invest
  plan freshness / flag VLM-LoRA).
- **Δ_actor ≥ +0.10 ∧ Δ_plan ≤ +0.05** → actor-OOD dominates; plan *not* load-bearing for drift → invest emulator
  coverage. **(My prediction.)** Seed's RWBC>base (SMW +0.12) sits here but is confounded (trained plan-head +
  live plan); `--lora-only --use-correct-plan` removes both confounds and decides it cleanly.

### Concession + revision of my R1 instinct

My R1 line — "the genuine gap is mid-mistake recovery filled by the emulator, not Qwen" — I **push further**:
it's not merely *injected large* mistakes (perturb-and-recover); the emulator is the *only* source of the
**on-policy state distribution** that cures *continuous* covariate drift — a stronger, more general claim. And
I **concede** to GPT-5.5: for the drift regime, on-policy/DAgger coverage is the load-bearing lever; the
plan/VLM is for objective selection + triggering, full stop. One generalist model, frozen-VLM-first; VLM-LoRA
deferred (high cost, moves the bridge-aligned latents).

### Citations (primary)
- **DAgger / covariate shift** — Ross, Gordon, Bagnell, AISTATS 2011 (arXiv:1011.0686); Ross & Bagnell, AISTATS
  2010 (O(εT²) BC vs O(εT) on-policy). Core Q1/Q2 mechanism.
- **Deep ensembles (S2)** — Lakshminarayanan, Pritzel, Blundell, NeurIPS 2017. **Conformal τ** — Angelopoulos &
  Bates, 2021 (arXiv:2107.07511).
- **RTC** — arXiv:2506.07339: inference-time chunk stitching (gate #1). **Helix** — figure.ai/news/helix:
  staleness-offset; S2 objective/S1 reactive. **π0.5** — pi.website/blog/pi05: text subtask → flow chunk.
  **GR00T N1.5** — cross-attn V/L + AdaLN (NitroGen lineage).
- **Repo (seed):** base DiT reproduces action from frame alone (buttons AUC 0.82–0.96, sticks 0.4–0.76) →
  near-zero plan gradient on factual data; RWBC>base (Sonic 0.58>0.25, SMW 0.53>0.41); left/right env-free-fragile
  (0.50→0.688); mean-pooled 2B hiddens don't separate left/right (~0.51).
