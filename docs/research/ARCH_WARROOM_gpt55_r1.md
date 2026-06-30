## [GPT55 — Round 1] 2026-06-27T23:34:57-07:00

# Position: do **not** put the fast channel inside shallow Qwen. Build a tiny System-1-side fast module conditioned on cached deep VLM latents + the live frame.

My stance is deliberately against the owner's early-exit thesis as the *runtime* fast path. The right architecture for NitroGen is closer to Figure Helix than to a shallow-exit VLM: a slow semantic writer produces a stable continuous intent latent asynchronously, and a small high-rate visuomotor reader localizes that intent against the current frame/action trace. Helix explicitly uses this split: S2 is a 7B VLM at 7-9 Hz, S1 is an 80M latent-conditional visuomotor transformer at 200 Hz, S2 distills task-relevant information into a single continuous latent vector, and deployment has S2 update shared memory asynchronously while S1 consumes the latest observation plus latest S2 latent; Helix also trains with temporal offsets to match deployment latency. That is the pattern to copy, scaled down for 2D games.

Early-exit VLM taps are attractive on paper, but they are the wrong first bet here:

1. **Shallow VLM states are not guaranteed to be task-grounded.** Early layers tend to encode local visual/text features; the plan-relevant binding often appears later. If the problem is “jump after the enemy, not before,” a shallow tap may know pixels but not the temporal objective.
2. **It still contends with the slow model.** To get shallow VLM states every chunk you still run the VLM vision tower / tokenization / early transformer prefix at actor rate. On a scarce 4xA6000 box, that is a bad use of a GPU when NitroGen already sees the current frame.
3. **It creates a training distribution mismatch.** Slow path uses final/deep plan latents; fast path would use shallow hidden states with different semantics and scale. You then need a nontrivial alignment loss just to make shallow latents mean what the bridge expects.
4. **The supervision signal is unclear.** We do not have mid-mistake recovery labels. A shallow-tap student can be trained to imitate deep latents, but then it inherits stale/deep behavior rather than learning true reactive correction.
5. **It risks breaking the clean null contract.** Multiple hidden-state taps and conditional exits make it easier to accidentally leak non-null conditioning into the base path.

I would still run *offline probes* on Qwen layer depth. I would not build the early-exit fast path until a probe proves that shallow layers predict action-relevant plan deltas better than a cheap frame-conditioned module.

---

# Proposed architecture: Async SlowPlan + FastPlanMod + real-time chunk stitching

## Slow path: Qwen as asynchronous semantic writer

Rate: **event-driven / every 2-4 NitroGen chunks**, not blocking. With the current seed saying A=2 chunks is about 1.2 s, I would target a slow update every 1.2-2.4 s plus immediate replans on high-confidence events (death, damage, stuck, room transition, objective completion, explicit uncertainty spike).

Inputs to slow Qwen worker:

- recent frames: current frame plus a short history or montage at low fps;
- previous plan text and plan latent id;
- executed action summary over the last 1-3 chunks;
- optional game state export when available, but do not depend on it for the architecture;
- prompt asking for a short objective, termination condition, and hazards.

Outputs written to a versioned shared cache:

```text
PlanCacheEntry {
  plan_id: int64,
  frame_t0: int,
  wallclock_t0: float,
  text_plan: str,
  slow_tokens: [K_s=32, d_dit],      # Perceiver-resampled continuous tokens
  slow_global: [d_dit],              # pooled plan/intent latent
  term_hint: small vector/text,       # optional predicted subgoal termination
  confidence: scalar,
}
```

I would widen the slow bridge from **K=8 to K_s=32** first. K=8 has already proven useful, but it is an unnecessary bottleneck for “objective + obstacle + temporal ordering + button intent.” GR00T N1.5's public model card says its flow matching transformer conditions action chunks on vision/language/proprioception with cross-attention to vision/language embeddings and uses an MLP connector and AdaLN for diffusion-step conditioning. That validates using richer V/L token memory and modulation, not just a tiny token bottleneck.

The slow path remains interpretable: text is logged and can be used for debugging, but the actor consumes continuous latents. This follows Helix's continuous latent interface and Latent Codes as Bridges (LCB), which argues for latent action bridges rather than forcing all planner-to-controller communication through language.

## Fast path: FastPlanMod as the high-rate semantic localizer

Rate: **every NitroGen chunk** immediately; later every frame only if chunk-level correction is insufficient. NitroGen already sees the live frame, so the fast module should not duplicate a VLM. It should read existing NitroGen frame features and cached slow plan latents.

Trainable module:

```text
FastPlanMod(
  frame_feat_t,              # frozen NitroGen vision features from current frame
  slow_tokens[plan_id],      # latest cached deep Qwen/Perceiver tokens
  slow_global[plan_id],
  action_trace_{t-H:t},      # previous executed buttons/sticks + chunk phase
  chunk_phase,               # index in chunk, time since plan, plan age
  event_bits                 # stuck/death/damage/velocity if available
) -> {
  fast_tokens: [K_f=4 or 8, d_dit],
  adaln_delta: per selected DiT block (gamma,beta,gate),
  cfg_gate: scalar in [0,1] or small vector by token group,
  replan_score: scalar,
}
```

Implementation details:

- Use a 2-4 layer transformer or Perceiver over `[CLS, slow_global, slow_tokens, pooled frame tokens, action trace tokens]`, ~5-20M params.
- Reuse frozen NitroGen image features; no second image encoder unless profiling proves NitroGen features are inaccessible/too slow.
- Append `fast_tokens` to the DiT cross-attention memory after image and slow tokens: `[image_tokens | slow_tokens | fast_tokens]`.
- Feed `adaln_delta` into selected DiT blocks as a **zero-initialized residual** on the existing diffusion-step AdaLN parameters. FiLM showed that feature-wise affine modulation is a general conditioning mechanism; GR00T already uses AdaLN inside the flow-DiT, making modulation a native insertion point.
- Use `cfg_gate` to reduce plan guidance when the live frame contradicts the stale plan or when plan confidence is low.
- `replan_score` notifies the slow worker; it must never block the actor.

Expected latency: the fast module should be low single-digit milliseconds on an A6000 because it runs on already-computed frame features plus tens of latent tokens. The slow Qwen worker can take hundreds of ms without blocking. The DiT still dominates action generation, so this architecture is honest: it does not magically make NitroGen 60 FPS, but it removes VLM blocking and reduces stale-plan failures.

## Exact null-invariance contract

This is non-negotiable. Null/masked plan must reproduce base NitroGen exactly.

Rules:

1. `plan_valid=0` masks `slow_tokens`, `slow_global`, and `fast_tokens` out of cross-attention memory.
2. All FastPlanMod outputs are multiplied by `plan_valid`.
3. AdaLN/FiLM residual heads are zero-initialized and gated: `h = h + plan_valid * gate * Delta(h)` with `gate=0` at init.
4. DiT-LoRA, if used, is also plan-gated. For null plan, LoRA contribution is exactly zero.
5. Unit tests compare base NitroGen logits/actions/velocity predictions against plan-null with bitwise or tight fp tolerance across several diffusion seeds.
6. Training includes a null batch path and a large penalty/CI assertion for any nonzero delta under null.

This is cleaner than early-exit because the only route by which plan information enters is the existing masked bridge plus explicitly gated residuals.

---

# Near-real-time runtime schedule on 4xA6000

Use three asynchronous loops and versioned memory. Never wait for Qwen in the control loop.

**GPU 0: Slow planner worker**

- Maintains cached prompt/KV where possible.
- Reads latest frame/history from a lock-free queue.
- Runs Qwen3.5-2B generation and hidden-state extraction.
- Runs Perceiver resampler to produce `K_s=32` slow tokens and `slow_global`.
- Writes `PlanCacheEntry(plan_id++)` into shared CPU/GPU memory.

**GPU 1: Actor/chunk worker**

- For each chunk request, reads latest committed `PlanCacheEntry`.
- Computes NitroGen frame features for the current frame.
- Runs FastPlanMod.
- Runs flow-DiT denoising for the next 18-step chunk with `[image | slow | fast]` memory and modulated AdaLN.
- Exposes chunk to the executor.

**GPU 2: Speculative / overlapping chunk worker**

- Generates the next chunk before the current chunk is exhausted using the latest plan.
- If a newer plan arrives mid-generation, it can either finish old generation or restart if remaining horizon is long enough.
- Can also produce a null/base fallback chunk for safety comparison.

**GPU 3: spare for training/eval or duplicate actors**

- Use for batched emulator rollouts, profiling, or online RL later. Do not couple the first architecture to needing all four GPUs at runtime.

**Chunk stitching**

Apply the “Real-Time Execution of Action Chunking Flow Policies” idea: a controller-facing `GetAction(obs)` returns the next action from the current chunk while a background inference loop prepares the next chunk. The paper's algorithm preserves unexecuted actions from the previous chunk and uses inpainting/soft masking during flow denoising to improve cross-chunk continuity. For NitroGen:

- keep `A_cur[18]` and execution index `i`;
- when generating `A_new`, preserve already-committed prefix and optionally the first `m=2-4` future actions from the previous chunk as soft constraints;
- if a plan update arrives, blend plan influence over the first few actions rather than hard-switching;
- if DiT inference misses deadline, continue executing the safe suffix of the old chunk or a base/null fallback.

This addresses blocking latency and staleness more directly than early-exit. Staleness is handled by (a) the actor always seeing the live frame, (b) FastPlanMod translating stale intent into current-frame modulation, and (c) the slow worker updating opportunistically.

---

# Training objective and data plan

The fast module needs to learn “given a stale-but-semantically-correct plan, what should I do from the current frame?” We have no mid-mistake recovery labels yet, so the first objective must be robust imitation with staleness augmentation, then emulator-generated counterfactuals.

## Stage A: supervised flow-matching BC with stale-latent augmentation

Data:

- human demos from record/play server;
- frame-exact emulator data where actions are known;
- existing Stage-1 synthetic plan/action chunks;
- later YouTube pseudo-labels only after IDM is credible, not first.

For a demo trajectory, sample time `t` and a plan latent computed from an earlier frame `t-lag`, where lag is drawn from the deployed latency distribution: e.g. 0, 1 chunk, 2 chunks, random 0-2 s. The current frame is `obs_t`; target action chunk is human/emulator action chunk from `t:t+18`.

Loss:

```text
L = L_flow_velocity(action_chunk_target)
  + lambda_null * ||DiT(null_plan) - base_DiT||^2
  + lambda_plan_dropout * BC under random stale/null/dropout plans
  + lambda_fast_distill * ||FastPlanMod(...).summary - TeacherDelta||^2
  + lambda_cfg_smooth * temporal smoothness(cfg_gate)
```

`TeacherDelta` can be a non-runtime teacher computed offline: run the full slow Qwen on the current frame and compare its bridge latent to the stale latent, or compare action predictions under current-vs-stale plan. This does **not** mean deploying early-exit. It is just a training signal saying what a corrected latent would look like.

## Stage B: emulator perturb-and-recover without human recovery labels

Use frame-exact save/load to synthesize recovery data:

1. Start from states along human/emulator demos.
2. Inject 0.5-2 seconds of noisy/off-policy actor actions to create plausible mistakes.
3. From the saved mistaken state, run short-horizon action search / actor sampling / GRPO-style ranking with dense game reward or hand-coded progress signals.
4. Use the best short rollout as a pseudo-recovery target.
5. Train FastPlanMod and DiT-LoRA on these states with the same stale plan latent.

This is where emulator access matters. It is more relevant to 2D games than robot-style internet semantics. It also avoids pretending Qwen knows the right recovery action from a single shallow hidden state.

## Stage C: optional RL, not first

After BC + perturb recovery works, train plan-head/FastPlanMod/LoRA with save-state GRPO. Actor = bridge + FastPlanMod + rank-16/32 DiT-LoRA. VLM remains frozen. VLM-LoRA is allowed later only if probes show the frozen Qwen cannot represent the needed objectives; it is expensive and risks overfitting to game pixels.

---

# Failure modes and mitigations

1. **FastPlanMod ignores slow plan and becomes a BC reflex.** Mitigate with counterfactual plan swaps, plan-contrastive losses, and evaluation where same frame requires different action under different plan.
2. **Slow plan is semantically wrong.** Fast module cannot fix a bad objective. Mitigate with replan triggers, plan confidence, held-out obscure/homebrew evals, and eventually VLM-LoRA only if needed.
3. **Modulation destabilizes DiT.** Start with cross-attention fast tokens only; add AdaLN deltas with zero gates and per-block clamps. Keep exact null tests in CI.
4. **Chunk boundaries jitter.** Use real-time chunking/inpainting and soft blending; penalize action discontinuities.
5. **Latency still too high because DiT is slow.** Reduce flow steps, distill DiT sampler, batch/speculate next chunks. Do not waste effort optimizing Qwen early exits before profiling DiT.
6. **2D games do not need a VLM.** Honest point: for many 2D games, a small goal-conditioned policy + emulator RL may beat a frozen VLM. The VLM is valuable mainly for human-readable objectives, cross-game priors, and sparse high-level planning; the fast loop should remain a game policy, not a miniature chatbot.

---

# Anticipating the opponent's pro-early-exit argument

A strong opponent will say: “Early-exit is elegant: one VLM, two depths, shallow latents every chunk, deep latents occasionally. It shares representations and avoids another model.”

My rebuttal:

- **Shared weights are not free if the shared model is the bottleneck.** Running even part of Qwen every chunk is more expensive than a 5-20M module over existing NitroGen features.
- **The fast path must be grounded in the live control state.** Shallow Qwen latents do not naturally include executed action trace, chunk phase, flow sampler uncertainty, or emulator event bits. FastPlanMod does.
- **Helix's lesson is separation, not early-exit.** Helix S2 writes a continuous latent; S1 is a separate high-rate policy with its own visual stream. That is the closest real deployed system to this problem.
- **iSHIFT is GUI action selection, not continuous action-chunk control.** Its arXiv page describes a compact 2.5B GUI agent with adaptive perception tokens and strong GUI benchmark scores, but that does not establish that shallow VLM layers are the right fast controller for frame-reactive flow-DiT games.
- **Early-exit can be tested later as a teacher/student source.** First prove that fast correction helps at all. If it does, then compare shallow-Qwen features to FastPlanMod features under identical latency budgets.

---

# Single highest-leverage first experiment

**Experiment: StalePlan-FastMod ablation on one emulator-backed game and one existing synthetic steering suite.**

Build the smallest FastPlanMod:

- keep Qwen frozen;
- widen slow tokens K=8 -> K=32;
- add `K_f=4` fast tokens from current NitroGen frame features + cached slow_global + action trace;
- initially skip AdaLN deltas or enable them with zero gate only after token path passes null tests;
- implement async plan cache but allow offline precomputed slow latents for the first benchmark;
- train with stale-latent augmentation on human/emulator action chunks;
- evaluate three variants:
  1. current blocking/stale K=8 bridge;
  2. async K=32 slow tokens only;
  3. async K=32 + FastPlanMod K_f=4.

Metrics:

- exact null-invariance delta vs base NitroGen;
- action BC/flow loss under artificial plan lags;
- direction/button counterfactual accuracy;
- plan flip rate and action discontinuity at chunk boundaries;
- closed-loop emulator reward/progress after injected 0.5-1.0 s mistakes;
- wall-clock chunk latency with Qwen running concurrently.

Decision rule: if FastPlanMod improves stale-lag closed-loop reward or recovery by >20% relative to async K=32 alone while preserving null invariance, commit to this architecture. If it does not, do not jump to early-exit; first inspect whether slow latents are wrong, DiT lacks capacity, or data lacks recovery states.

## What not to build yet

- No runtime shallow-Qwen early-exit path.
- No VLM-LoRA.
- No full end-to-end VLM+DiT training.
- No massive YouTube/IDM pipeline as a prerequisite for the architecture test.
- No elaborate option hierarchy or language tree search.
- No frame-rate fast path until chunk-rate FastPlanMod fails.
- No GRPO/prime-rl loop until BC + emulator perturb-recovery produces a stable null-invariant actor.

---

# References checked

- Figure Helix blog: S2 VLM 7-9 Hz, S1 80M transformer at 200 Hz, continuous S2 latent, async shared-memory deployment, temporal-offset training.
- NVIDIA GR00T N1.5 Hugging Face model card: pre-trained vision/text encoders, flow matching transformer over action chunks, cross-attention to V/L embeddings, MLP connector, AdaLN diffusion-step conditioning.
- Physical Intelligence π0 blog: VLM adapted to output continuous motor commands via flow matching at up to 50 Hz.
- ACT/ALOHA project page: action chunking with transformers, fixed chunk size 90, direct 50 Hz joint prediction.
- Real-Time Execution of Action Chunking Flow Policies arXiv HTML: controller `GetAction`, background inference loop, preservation of previous chunk suffix, flow-policy inpainting/soft masking.
- FiLM paper: feature-wise affine conditioning as a general conditioning layer.
- Latent Codes as Bridges arXiv page: latent-code bridge from LLMs to actions in hierarchical robot control.
