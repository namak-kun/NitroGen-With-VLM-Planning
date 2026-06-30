# ARCH WARROOM — Round 3 seed: Actor-OOD drift & plan staleness (LATENCY OFF THE TABLE)

This is a **focused** round, NOT a re-debate. Rounds 1–2 (latency / decode-vs-depth / async / prefill /
early-exit) reached consensus and are CLOSED. Do not re-open them.

## The reframing (verbatim intent from the project owner, 2026-06-28)

> "That [decode] is the blocker on a *latency* level. But let's assume latency is out of the picture — we
> **pause the game during inference like NitroGen normally does** (frame-exact pause). The real problems are
> (1) **stale plans** and (2) **adapting the DiT to OOD scenarios**. It's very easy to drift OOD because
> **even in a game it was trained on, the state can be subtly different.** The real-time stuff was only ever
> a lens to think about depth/integration."

So: **assume inference is free** (we pause). No async, no prefill-rate, no early-exit, no real-time. The
question is purely **capability / adaptation / distribution-shift**.

## What Rounds 1–2 already settled — do NOT re-litigate
- Latency/decode/prefill/async/early-exit: OFF the table. Assume free.
- **Plan staleness mechanism** is considered solved-in-principle: Helix-style **staleness-offset training**
  (train the DiT on plan latents delayed 0–A chunks so deploy staleness is in-distribution) + **replan
  triggers** + a **dynamic-CFG gate** that down-weights plan guidance when the live frame contradicts the
  stale plan. Only revisit staleness if you have a *concrete new* improvement beyond this.
- **Null-invariance contract** (a null/masked plan reproduces base NitroGen exactly) must be preserved by
  any proposal.

## The narrow questions for Round 3 (answer ALL THREE, take a stance)

**Q1 — Decompose the OOD.** Cleanly separate:
  - **(a) PLAN-conditioning OOD** — the bridge/VLM is fed a novel *(frame × plan)* combination it wasn't
    trained on (the conditioning is wrong/stale/unfamiliar), vs
  - **(b) DiT-ACTOR OOD** — the flow-matching *actor* visits game states absent from its own training
    distribution, **independent of any plan** (the classic behavior-cloning **covariate shift / compounding
    error** problem: small per-step errors move the agent into states the expert demos never covered).
  Which one dominates the owner's "subtle drift on a *trained* game"? Give the mechanism and evidence.

**Q2 — Is the plan even the right lever for actor-OOD?** Prior project finding (verified, in memory +
EXPERIMENTS): on factual data the **frozen base DiT already reproduces the streamer's action from the frame
alone** (buttons AUC 0.82–0.96, sticks corr 0.4–0.76) → factual plan-conditioning is a **near-zero-gradient**
objective; real navigational/counterfactual capability "can't escape envs." Given that:
  - Does a *better or fresher plan* actually reduce **actor-OOD** drift, or is actor-drift fundamentally an
    **on-policy / DAgger state-coverage** problem that only emulator-generated on-policy data fixes —
    *regardless of the VLM*?
  - If the plan does **not** help actor-OOD, then **what is the VLM/plan actually for** in the drift regime
    (objective selection? replan triggering? long-horizon credit? nothing?) — be specific and honest.

**Q3 — Detection & gating, not just recovery.** Rounds 1–2 only proposed *training to recover* from large
*injected* mistakes (perturb-and-recover / save-state GRPO). The owner's worry is **subtle, continuous**
drift that compounds. Do we ALSO need cheap **online OOD *detection*** to *gate* behavior the instant drift
starts — e.g.:
  - flow-matching **energy / velocity-residual / likelihood** of the sampled chunk,
  - **K-sample ensemble disagreement** (the DiT is stochastic — sample K chunks, measure variance),
  - **plan↔frame contradiction** score (cosine of plan tokens vs frame embedding),
  - emulator telemetry (stuck / no-progress / near-death)?
  And on detection, what's the *gate*: trigger a replan, raise/lower CFG, fall back to a conservative prior,
  pause-and-search over K plans, or widen sampling? Propose the **concrete signal(s) + the gate + a
  threshold**. Distinguish what needs the emulator (train/verify) vs what is deploy-time-only.

## REQUIRED deliverable from each side
1. A **stance** on Q1 (which OOD dominates) and Q2 (is the plan load-bearing for drift, or is it an actor-
   data problem) — with the evidence/mechanism.
2. A **concrete gating proposal** for Q3 (signal + gate + threshold), preserving null-invariance.
3. **ONE cheap, falsifiable experiment** on the EXISTING repo that *discriminates* plan-OOD vs actor-OOD —
   with a decision threshold. Must use what exists: emulator save-states (`nitrogen/eval/envs/{emulator_env,
   snes_env,mgba_env}.py`, frame-exact save/load), the `btn_s600` checkpoint, and the on-policy adaptation
   harness `planner_poc/rwbc_actor_adapt.py` (`make_env`, `collect`, `eval_reward`, `--use-correct-plan`,
   `--save-delta`). Name the metric and the number that would change your mind.

## Project facts to ground every claim (do not contradict these)
- Architecture: **frozen Qwen3.5-2B** (System-2) → **PlanResampler** (K=8 **continuous-latent** query tokens,
  NOT discrete tokens) → **PlanAdapter** → **PlanHead** → injected into a **frozen ~500M flow-matching DiT**
  (System-1, **GR00T-N1.5 lineage**: cross-attn to V/L + AdaLN diffusion-step cond) that maps the **current
  frame → an 18-step gamepad action chunk**. The DiT is the **small/fast** component; the VLM is the
  **big/slow** one. NitroGen is **markov** (sees only the last frame → cannot plan; that's the VLM's job).
- Key files: `nitrogen/planner.py` (PlanResampler/PlanAdapter/PlanHead, `encode_multimodal(text_only=True)`,
  `generate_plan`, `distill_weight`, zero-init `plan_adaln`); `nitrogen/flow_matching_transformer/nitrogen.py`
  (`compute_plan_tokens`, `apply_null_mask`, `adaln_cond`, plan-token injection ~L482–495); `eval_policy.py`
  (`NitroGenPolicy`); `scripts/train_planner.py` (Stage-1 trainer); `rwbc_actor_adapt.py` (RWBC on-policy
  adaptation — already collects actor rollouts, reward-weights, trains LoRA+plan-head).
- `btn_s600` = main eval ckpt: frozen base DiT **+ active rank-16 DiT-LoRA + plan-head**. "null/base" in
  experiments = this ckpt with the plan **dropped** (not raw NitroGen).
- Eval just fixed (2026-06-28): scripted-RIGHT-ceiling + idle-floor + survival-weighting; **RWBC beats base**
  on Sonic (0.58>0.25) and SMW (0.53>0.41) — i.e. **on-policy actor adaptation already measurably helps**,
  which is itself evidence for the actor-OOD hypothesis.
- Emulator substrate: in-process **frame-exact save/load** (`emulator_env.py` base; `snes_env.py`,
  `mgba_env.py`). Dense checkable reward + reset → the project's standing conclusion "we can't escape envs."
- Prior verified findings: left/right is env-free-fragile (token sep collapses in the adapter; cleaning
  contrastive label noise lifts it 0.50→0.688); up/down is easy (whole-scene correlated). Mean-pooled 2B
  frame-conditioned hiddens do NOT linearly separate left/right (acc~0.51).

## Constraints
- Goal = **ONE generalist model** (NO per-game / per-genre models).
- VLM-LoRA is **allowed** but must be flagged as higher-cost; frozen-VLM-first.
- Cite **primary sources** for any external claim (DAgger/Ross 2011, RTC arXiv:2506.07339, Helix
  figure.ai/news/helix, π0 / π0.5, GR00T, conformal/OOD-detection refs, etc.). No hand-wavy citations.
- Be concrete, grounded in the repo paths above, and willing to say "the plan doesn't help here" if true.

## Mechanics
- GPT-5.5 writes its opening position to `ARCH_WARROOM_gpt55_r3.md`; Opus-4.8 to `ARCH_WARROOM_opus_r3.md`
  (ABSOLUTE paths given in each agent's instructions). Then the orchestrator referees and runs a rebuttal
  round where each reads the other's r3 and converges or sharpens the disagreement.
