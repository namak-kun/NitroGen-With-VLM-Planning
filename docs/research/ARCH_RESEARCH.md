# Architectural options for coupling a slow VLM planner with a fast reactive policy

_Date: 2026-06-27_

This note surveys recent dual-system robotics/VLA, action-chunking, async control, latent-bridge, and fast-adaptation ideas, then maps them to the NitroGen + frozen Qwen3.5 planner system.

## Executive conclusion

The owner's hypothesis is sound: the best current embodied systems do **not** put a slow VLM in the innermost control loop. They use a **dual-rate controller**:

- a slow semantic path that runs a VLM/MLLM at low Hz or at subtask boundaries, and
- a fast visuomotor path that runs from the latest observation at high Hz, conditioned by the latest cached semantic latent/goal.

The strongest precedent is [Figure Helix](https://www.figure.ai/news/helix): System 2 is an onboard VLM at **7-9 Hz**, System 1 is an **80M latent-conditional visuomotor transformer at 200 Hz**, and deployment is explicitly asynchronous: S2 continuously updates a shared-memory latent vector while S1 consumes the latest frame/state plus the most recent S2 latent. Helix's interface is not text tokens at the policy boundary; S2 distills task-relevant information into a **continuous latent vector** that is projected into S1 token space and concatenated with S1 visual features.

For our system, the current `K=8` cross-attention-only bridge is likely too narrow and too slow-changing. The highest-leverage next architectural experiment is therefore:

> **Add a zero-initialized, plan-conditioned DiT modulation path in parallel with wider plan tokens:** keep null invariance, widen K from 8 to 32/64, and feed the cached slow plan latent through a tiny fast module that sees the current frame/action trace every chunk and emits per-block AdaLN/FiLM gains plus a few corrective tokens. Train only the bridge/modulator and optional LoRA first; do not train the full VLM initially.

This directly tests whether the bottleneck is channel bandwidth + missing fast corrective feedback, while staying feasible on 4x RTX A6000.

---

## 1. Most relevant real systems

| System | Link | System2→System1 interface | Rate / timing | Key idea to borrow |
|---|---|---|---|---|
| **Figure Helix** | [Figure blog](https://www.figure.ai/news/helix) | S2 VLM distills image/state/text into a **single continuous latent vector**; S1 projects it into token space and concatenates with S1 visual features. S1 also gets its own high-rate image/state stream. | S2 **7-9 Hz**; S1 **200 Hz**. Asynchronous shared-memory latent. Training adds temporal offset matching deployment latency. | Closest match to owner's hypothesis: slow semantic latent + fast reactive visual loop; train with latency offset; don't force all feedback through slow VLM. |
| **Physical Intelligence π0** | [π0 blog](https://www.pi.website/blog/pi0), [paper PDF](https://www.pi.website/download/pi0.pdf) | VLM backbone adapted to output **continuous action chunks via flow matching**. The action pathway is continuous, not just language tokens. | Robot motor commands up to **50 Hz**; uses chunks/flow policy. | Flow-matching action experts can be conditioned by VLM semantics without discretizing action. Relevant because NitroGen is already a flow/DiT action chunker. |
| **Physical Intelligence π0.5** | [π0.5 blog](https://www.pi.website/blog/pi05), [paper PDF](https://www.pi.website/download/pi05.pdf) | Same model first emits a **high-level text action**, then follows it with a **50-step / 1-second continuous action chunk** via flow matching. | Slow high-level text step; low-level 1s action chunks. | A clean hierarchical pattern: text is useful for subtask decomposition, but final control is continuous action chunks. Similar to Qwen plan → NitroGen chunk. |
| **Hi Robot** | [project page](https://www.pi.website/research/hirobot), [paper](https://arxiv.org/abs/2502.19417) | High-level VLM “whispers” **intermediate natural-language steps** to π0; can incorporate user feedback and scene context. | High-level at subtask/feedback times; π0 executes low-level actions. | Natural-language inner voice is valuable for long-horizon semantics, but the low-level controller must already understand those short instructions. Good precedent for plan text + trace feedback. |
| **NVIDIA GR00T N1.5** | [HF model card](https://huggingface.co/nvidia/GR00T-N1.5-3B/blob/main/README.md), [repo](https://github.com/NVIDIA/Isaac-GR00T) | Vision/text transformers produce embeddings; a flow-matching DiT interleaves self-attention over proprio/action tokens with **cross-attention to V/L embeddings**. Diffusion step uses **AdaLN**. Connector MLP between V/L features and DiT. | Chunked continuous action inference; model card does not specify deployed Hz. | Very close architecturally: V/L embeddings → connector → DiT cross-attn, plus AdaLN inside DiT. Borrow multi-token V/L conditioning and AdaLN/FiLM modulation in addition to cross-attn. |
| **LCB: Latent Codes as Bridges** | [project](https://fredshentu.github.io/LCB_site/), [arXiv](https://arxiv.org/abs/2405.04798) | LLM emits or exposes a learned **latent code** / `<ACT>` token embedding as bridge to low-level policy, avoiding pure-language bottleneck. | High-level planner lower-rate; low-level policy higher-rate / asynchronous possible. | Replace “plan text only” with learned action-grounded latent slots. Strong precedent for trainable bridge tokens that do not need to be human-readable. |
| **OpenHelix** | [project](https://openhelix-robot.github.io/), [survey](https://arxiv.org/html/2505.03912v1) | Learned token + linear projection from MLLM to diffusion policy; prompt tuning; auxiliary action-prediction task to force multimodal reasoning. | Dual-system; discusses async strategies. | Important caution: action token embeddings may mostly encode static instruction semantics and ignore visual changes. Need probes and auxiliary grounding losses. |
| **Fast-in-Slow VLA** | [project](https://fast-in-slow.github.io/) | Embeds fast System 1 execution module inside VLM-based System 2 with partial parameter sharing and asynchronous frequencies. | Reports **21.9 Hz** control without action chunking. | More coupled than separate modules; useful conceptually, but likely too invasive for NitroGen unless training a new integrated model. |
| **RT-2** | [project](https://robotics-transformer2.github.io/), [DeepMind blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/), [arXiv](https://arxiv.org/abs/2307.15818) | Converts robot actions into **tokens** so a VLM can output language and actions in one autoregressive stream; can emit CoT then `Action:` tokens. | Low-frequency robot control typical of Google robot settings; not a high-Hz dexterous loop. | Good for semantic transfer; less suitable as-is for reactive games because action token decoding is slow/coarse. |
| **OpenVLA** | [project](https://openvla.github.io/), [arXiv](https://arxiv.org/abs/2406.09246) | Open VLA with discretized action-token output; LoRA fine-tuning works well, only ~1.4% params in reported PEFT experiments. | Evaluated on 5 Hz Franka-Tabletop and 15 Hz Franka-DROID setups. | VLM-LoRA can adapt action grounding efficiently, but discrete action decoding is not the first choice for our existing flow action model. |
| **OpenVLA-OFT** | [project](https://openvla-oft.github.io/) | Optimized fine-tuning recipe emphasizes continuous action regression / chunking and faster inference; qualitative analysis contrasts language grounding vs visual feedback. | Real-robot rollouts; designed for practical speed. | Useful warning: architectural choices that overemphasize language can impair visual feedback. For games, preserve frame-reactivity. |
| **Octo** | [project](https://octo-models.github.io/), [paper](https://arxiv.org/abs/2405.12213) | Transformer diffusion policy; conditions on language or goal images, observation history, and multi-modal action distributions. | Robot-policy scale; efficient fine-tuning. | Goal-image or latent-goal conditioning can be richer than text; observation history matters. |
| **RoboFlamingo** | [project](https://roboflamingo.github.io/), [arXiv](https://arxiv.org/abs/2311.01378) | Pretrained VLM for single-step visual-language comprehension; explicit policy head models sequential history. | Designed for low-cost/single-GPU robot manipulation. | Decompose VLM comprehension from a smaller history-aware policy head. Good model for a cheap fast corrective head. |
| **ACT / ALOHA** | [project](https://tonyzhaozh.github.io/aloha/), [arXiv](https://arxiv.org/abs/2304.13705) | No VLM bridge; predicts action chunks with a transformer/CVAE and uses temporal action chunking. | ACT directly predicts joint positions at **50 Hz**, fixed chunk size **90**. | Chunking reduces effective horizon; chunk overlap / smoothing ideas are useful for NitroGen chunk stitching. |
| **Real-Time Chunking for flow policies** | [arXiv HTML](https://arxiv.org/html/2506.07339v1) | Background inference loop swaps in new action chunks; uses inpainting/soft masking to preserve already-planned suffix and improve cross-chunk continuity. | Controller calls `GetAction` every control tick while inference runs separately. | Directly relevant: async chunk generation + inpainting continuity can hide slow planning/inference delays. |
| **FiLM / HyperNetworks** | [FiLM](https://arxiv.org/abs/1709.07871), [HyperNetworks](https://arxiv.org/abs/1609.09106) | Conditioning network emits feature-wise affine gains/biases or weights for another network. | Per input / per episode / per step. | Low-bandwidth conditioning can modulate many layers more effectively than appending a few tokens. Use zero-init for null invariance. |
| **Options / Feudal HRL** | [Option-Critic](https://arxiv.org/abs/1609.05140), [FeUdal Networks](https://arxiv.org/abs/1703.01161) | Slow manager emits option/goal embedding; fast worker optimizes intrinsic/goal-conditioned behavior. | Manager low-rate; worker high-rate. | For 2D games, continuous goals/options may be more sample-efficient than free-form text plans. |

---

## 2. Is the dual-rate hypothesis sound?

Yes. The hypothesis is not only plausible; it is the dominant design in the systems closest to our setup.

### What surveyed systems actually do

#### Helix: slow latent writer, fast reactive reader

Helix is the cleanest example. Figure states that:

- System 2 is an internet-pretrained VLM running at **7-9 Hz** for scene understanding and language comprehension.
- System 1 is a fast reactive visuomotor policy running at **200 Hz**.
- S2 distills semantic task-relevant information into a **single continuous latent vector**.
- S1 receives the **same image/state inputs** as S2 but at higher frequency, plus the latest S2 latent.
- In deployment, S2 is an asynchronous background process that updates shared memory; S1 runs its real-time loop with the most recent latent.
- Training includes a temporal offset to match the S2/S1 deployment latency.

This is exactly “long slow path + short low-latency path.” Importantly, the fast path is not just stale plan replay: S1 keeps seeing the latest observation, so it can correct locally while maintaining the semantic objective encoded in the latent.

#### π0.5 / Hi Robot: text for subtasking, continuous policy for execution

π0.5 first asks the model for a high-level text action, then asks it to execute a **50-step / 1-second** continuous action chunk. Hi Robot uses a high-level VLM to “whisper” intermediate text commands to π0 and incorporate real-time user feedback. This supports the current Qwen-text-plan approach, but also exposes a limitation: the low-level policy must already be well grounded in the command interface. If “go left for two beats then jump” is not represented well in NitroGen's action-conditioned latent space, text alone is not enough.

#### GR00T: high-level embeddings feed a flow DiT through cross-attention and AdaLN

GR00T N1.5 is architecturally close: vision/text encoders produce embeddings, and a flow-matching transformer/DiT denoises continuous action chunks conditioned on those embeddings. The model card says the flow transformer interleaves self-attention over proprioception/actions with **cross-attention to the vision/language embeddings**, and implements diffusion-step conditioning using **adaptive LayerNorm**. This suggests two takeaways for us:

1. cross-attention tokens are a standard interface, but
2. modulation/AdaLN is already natural inside DiT-style action heads and should be considered as a parallel conditioning path.

#### LCB / OpenHelix: latent bridges, learned action tokens, and warnings

LCB argues that language is too restrictive as the high-low interface; it introduces learnable latent codes / `<ACT>` bridge tokens. OpenHelix surveys dual-system VLAs and identifies latent-feature representation and integration as core open questions. It specifically notes that some MLLM action embeddings mostly reflect instruction semantics and little visual adaptation; their project adds an auxiliary action-prediction task to force multimodal reasoning.

This matters for NitroGen: if Qwen hidden states collapse to “static plan semantics,” our K=8 bridge may be low-bandwidth **and** insufficiently frame-grounded. We should add probes and losses that measure whether plan latents change correctly with current frame, previous actions, and failures.

#### ACT / real-time chunking: chunked actions need stitching, not blocking

ACT predicts long action chunks at 50 Hz and reduces effective horizon. Real-time chunking for flow policies runs inference in a background loop, preserving a suffix of the previous chunk and swapping in a new chunk when ready. This is directly applicable to NitroGen because NitroGen already emits 18-step chunks. A blocking “wait for VLM then sample DiT” loop is the wrong abstraction; use versioned cached latents and chunk stitching.

### What should flow on slow path vs fast path?

**Slow path (every A chunks or asynchronously):**

- plan text for interpretability and logging;
- richer continuous plan latent, not only 8 resampled tokens;
- object/entity/goal slots if available;
- prior-plan summary and executed-action trace;
- optional predicted hazard/failure mode or subgoal termination condition.

**Fast path (every chunk or frame):**

- current-frame features from NitroGen's existing image encoder or a small separate encoder;
- latest cached plan latent;
- recent action trace / chunk progress;
- cheap corrective outputs: FiLM/AdaLN gains, cross-attention token deltas, CFG/gating scalars, residual actions, option gate, or replan trigger.

The key architectural shift is: **System 2 should set intent; a cheap System1-side adapter should localize that intent to the current frame at high rate.**

---

## 3. Concrete proposals for our system

### Proposal 1 — Widen the bridge and add multi-site conditioning

**Mechanism**

Replace the single `K=8` plan-token bottleneck with a richer conditioning bundle:

1. `K=32` or `K=64` plan tokens instead of `K=8`.
2. Use multiple VLM hidden layers, not only final text-token hidden states. Middle layers often preserve more visual grounding; OpenHelix notes GR00T uses middle-layer features in some designs.
3. Keep cross-attention injection, but add a parallel global vector path: pooled plan latent → MLP → per-block scale/shift/gate parameters.
4. Use zero-initialized gates so null/masked plan still exactly reproduces base NitroGen.
5. Add representation probes: direction/button/action separability, frame sensitivity, and plan-vs-null invariance.

**Fast vs slow path**

- Slow: Qwen emits text and hidden-state tokens; bridge resamples them into 32/64 tokens + pooled latent.
- Fast: none beyond ordinary NitroGen frame input in the simplest version.

**Trainable**

Perceiver/resampler, adapter, FiLM/AdaLN MLP, gates, optional existing DiT-LoRA. VLM remains frozen initially.

**Expected latency**

Almost unchanged VLM latency. DiT cross-attention grows moderately with K but should be cheap relative to Qwen. K=64 is likely affordable on 4x A6000.

**Addresses problems**

- Problem 1: directly widens the channel.
- Problem 2: not test-time adaptation yet.
- Problem 3: still mostly one-directional/infrequent.
- Problem 4: respects latency asymmetry.

**Precedent**

GR00T uses V/L embeddings through a connector and cross-attention; LCB/OpenHelix emphasize learned latent bridges; FiLM/AdaLN are standard conditioning layers.

**Risk**

Widening alone may only carry a better stale plan. If mid-chunk corrective feedback is the real issue, this under-delivers. Also, large K can tempt the model to ignore the DiT's visual pathway unless regularized.

---

### Proposal 2 — Fast plan-conditioned corrective modulator (**recommended first full architectural experiment**)

**Mechanism**

Add a tiny high-rate module, `FastPlanMod`, that runs every NitroGen chunk, or eventually every frame. It consumes:

- current frame features from NitroGen's frozen vision encoder,
- cached slow plan latent/tokens,
- current chunk index / time-since-plan,
- previous executed actions and optionally last predicted action chunk,
- simple discrepancy signals if available: stuck detector, damage/death flag, velocity estimate, action entropy.

It emits a small set of correction signals:

1. **DiT modulation:** per selected DiT block, AdaLN/FiLM `(gamma, beta)` or scalar gates.
2. **Corrective tokens:** 2-8 extra fast tokens appended to cross-attention memory.
3. **CFG/guidance gate:** dynamic plan-guidance weight instead of fixed CFG≈8.
4. Optional **action residual** on the 18-step output, clipped and zero-initialized.

All outputs are zero-initialized and masked so the base/null path remains exactly NitroGen at initialization.

**Fast vs slow path**

- Slow: Qwen plan text + rich plan latent every A chunks/asynchronously.
- Fast: `FastPlanMod(current_frame, cached_plan, action_trace)` every chunk/frame emits low-latency modulation and correction.

**Trainable**

FastPlanMod, plan resampler/adapter, modulation heads, optional rank-16/32 DiT-LoRA. Qwen frozen initially. Later add Qwen LoRA if needed.

**Expected latency**

A small transformer/MLP over existing frame features should be sub-millisecond to a few ms, far below VLM latency. It can run on the DiT GPU. No new slow path.

**Addresses problems**

- Problem 1: complements wider channel with continuous modulation.
- Problem 2: gives the DiT side per-chunk adaptive behavior without changing weights online.
- Problem 3: introduces a fast feedback path from current frame/action trace to plan-conditioned control.
- Problem 4: uses VLM only at slow rate.

**Precedent**

Helix's S1 uses high-rate observations plus latest S2 latent. GR00T's DiT already uses cross-attention and AdaLN. FiLM/HyperNetworks provide the mechanism for context-dependent modulation.

**Risk**

Needs training data where fast corrections matter. Existing synthetic steering may not supervise “recover after mistake.” Best paired with emulator data / counterfactual rollouts / DAgger-style corrections. There is also risk of destabilizing exact null invariance unless all gates are carefully zero-initialized and tested.

---

### Proposal 3 — Async planner + versioned plan cache + real-time chunk stitching

**Mechanism**

Move Qwen planning out of the blocking control loop:

1. A planner worker continuously consumes the latest `(recent frames, prior plan, executed action trace, event summary)` and writes `{plan_id, text, latent, timestamp, predicted horizon}`.
2. NitroGen never waits for Qwen. It always acts using the latest valid plan latent.
3. When a new latent arrives mid-chunk, apply it only at safe boundaries or blend/gate it over a few frames.
4. Use RTC-style suffix preservation/inpainting: when generating a new 18-step chunk, preserve a prefix/suffix from the currently executing chunk and denoise/fill the remainder for continuity.
5. Train with artificial temporal offsets like Helix so inference latency is in-distribution.

**Fast vs slow path**

- Slow: async Qwen latent updates.
- Fast: control loop uses last latent and chunk-stitching; optional FastPlanMod reads the same cache.

**Trainable**

Initially none, except maybe a chunk-blending/gating head. Later train the DiT/adapter with latency offsets and latent staleness augmentation.

**Expected latency**

Improves wall-clock responsiveness because VLM latency is hidden. No need to reduce Qwen latency. Memory overhead is a small latent cache and worker queue.

**Addresses problems**

- Problem 1: not directly.
- Problem 2: not directly.
- Problem 3: improves replan frequency and feedback freshness; avoids control stalls.
- Problem 4: directly handles latency asymmetry.

**Precedent**

Helix asynchronous shared-memory latent; Real-Time Chunking for flow policies; ACT/ALOHA action chunking.

**Risk**

If the bridge is weak, fresher latents still won't help. Plan switching can cause jitter unless blended or trained with staleness. Async engineering without architectural improvement may look good in demos but not solve left/right or fine control failures.

---

### Proposal 4 — Learned latent action slots / `<ACT>` tokens instead of only text-token resampling

**Mechanism**

Introduce trainable action-grounded latent slots in Qwen's prompt/adapter:

- append `<ACT_1> ... <ACT_M>` special tokens or soft prompt queries;
- train only embeddings + bridge initially, optionally Qwen LoRA later;
- use their hidden states as the plan latent, instead of compressing arbitrary text tokens;
- add auxiliary heads predicting coarse action summaries: direction histogram, button events, expected progress, termination/hazard.

This makes the bridge a learned protocol between VLM and DiT, not an accidental compression of natural-language hidden states.

**Fast vs slow path**

- Slow: Qwen produces text plus `<ACT>` hidden slots.
- Fast: same as current unless combined with Proposal 2.

**Trainable**

Soft tokens / token embeddings, resampler/adapter, auxiliary heads, optional Qwen LoRA. Keep main Qwen weights frozen first.

**Expected latency**

Adds a few prompt tokens only. LoRA training increases training cost but not much inference latency.

**Addresses problems**

- Problem 1: improves semantic density and action grounding.
- Problem 2: not per-episode test-time adaptation.
- Problem 3: still low-rate unless combined with FastPlanMod.
- Problem 4: compatible with slow planning.

**Precedent**

LCB latent codes / `<ACT>` bridge; OpenHelix learned token + prompt tuning + auxiliary action prediction.

**Risk**

OpenHelix's warning applies: learned action tokens can become static instruction embeddings unless trained with frame-sensitive auxiliary losses. Requires careful ablations: same text/different frames should produce different useful latents.

---

### Proposal 5 — Dynamic DiT adapters / hypernetwork fast weights driven by System 2

**Mechanism**

Have the slow plan latent generate small **per-plan weights** for the DiT side:

- LoRA scale vectors per block/head;
- low-rank delta matrices for cross-attention projections;
- adapter-bank mixture weights (`left-corridor`, `jump-gap`, `avoid-enemy`, etc.);
- per-layer AdaLN/FiLM parameters;
- steering vectors added to selected DiT activations.

Avoid generating full DiT weights. Use a tiny hypernetwork with strict norm clipping and zero-init.

**Fast vs slow path**

- Slow: plan latent → per-plan adapter/gate parameters.
- Fast: optional per-frame module adjusts mixture/gates based on current frame.

**Trainable**

Hypernetwork, adapter bank, LoRA scales, optionally DiT-LoRA. Qwen frozen or LoRA-tuned later.

**Expected latency**

Per-plan generation is cheap; applying LoRA/adapters adds modest DiT cost. Full hypernet-generated weights would be expensive and risky; avoid.

**Addresses problems**

- Problem 1: high-bandwidth conditioning through weights/modulation rather than tokens.
- Problem 2: gives DiT per-plan adaptation, but not true online learning unless updated during episode.
- Problem 3: can be combined with fast gates for feedback.
- Problem 4: slow generation, fast application.

**Precedent**

HyperNetworks, FiLM, AdaLN in diffusion/DiT models, adapter/LoRA routing. Conceptually similar to slow net emitting fast net modulation.

**Risk**

Glamorous but easy to overfit or destabilize. In a 2D game action space, full dynamic weights are likely lower ROI than FiLM/AdaLN gates and corrective tokens.

---

### Proposal 6 — Goal/option-conditioned low-level policy instead of free-form text plan only

**Mechanism**

Turn the bridge into a hierarchical RL/options interface:

- Qwen emits structured subgoals: `{verb, object/entity, direction, target coordinate/region, horizon, termination condition}`.
- A learned encoder maps this into a continuous goal embedding.
- NitroGen/LoRA is trained as a goal-conditioned worker.
- The fast path tracks progress toward the goal and can trigger early termination/replan.

For games, examples: `move_to(x=..., y=...)`, `jump_over_gap`, `avoid_enemy`, `collect_powerup`, `enter_door`, `climb`, `shoot_until_clear`.

**Fast vs slow path**

- Slow: semantic option/subgoal selection.
- Fast: goal-conditioned worker and termination/progress monitor.

**Trainable**

Goal encoder, option/progress head, adapter/LoRA. Later use emulator RL/GRPO to improve option policies.

**Expected latency**

Very cheap after Qwen. Structured goals can be cached and interpreted by small networks.

**Addresses problems**

- Problem 1: structured goals may be more information-dense than 8 tokens.
- Problem 2: goal embedding changes policy behavior without updating weights.
- Problem 3: progress monitor gives feedback and early replan.
- Problem 4: low-rate goal selection, high-rate execution.

**Precedent**

Options / Feudal Networks; SayCan/RT-1 long-horizon planning with skills; Hi Robot text subtasking.

**Risk**

Requires state/goal labels or reward functions. In pixel-only 2D games, extracting coordinates/entities may need VLM/object detectors or emulator state. However, with emulator envs this may be more tractable than in real robotics.

---

### Proposal 7 — True test-time training / online adaptation

**Mechanism**

During an episode, update tiny adapters or fast weights using self-supervised and reward-derived losses:

- consistency between predicted and observed frame/action outcomes;
- stuck/death negative reward;
- plan progress classifier;
- contrastive success/failure rollouts from emulator save-states;
- update only adapter norms/gates, never base NitroGen or Qwen.

**Fast vs slow path**

- Slow: Qwen proposes plan and maybe critiques failures.
- Fast/online: adapter weights update every few chunks or after failed rollouts.

**Trainable**

At test time: tiny adapter/LoRA scales only. Offline: train the TTT loss and update rule.

**Expected latency**

Potentially large if backprop is in the loop. On 4x A6000, save-state GRPO batches may be feasible offline/between attempts, but not inside a real-time game loop.

**Addresses problems**

- Problem 1: not primarily.
- Problem 2: directly.
- Problem 3: feedback if updates are frequent enough.
- Problem 4: risky; backprop conflicts with real-time latency.

**Precedent**

General TTT/fast-weights literature; GRPO/RL with save-states is more relevant to our emulator substrate than robotics VLA deployments.

**Risk**

High complexity and instability. For this project, prefer **dynamic modulation without gradient updates** first, then use emulator save-states for offline RL or between-episode adaptation.

---

### Proposal 8 — VLM LoRA for plan/action grounding

**Mechanism**

Now that VLM-LoRA is approved, train a small Qwen LoRA to make hidden states and text plans more action-grounded:

- supervised on emulator/action traces and synthetic plan labels;
- auxiliary predictions: next action histogram, button event, direction, progress, risk;
- preserve language/vision ability with mixed caption/VQA/objective data;
- keep LoRA rank low and isolate it to late/mid layers used by the bridge.

**Fast vs slow path**

- Slow only: better Qwen latents/plans.
- Fast path still needs Proposal 2 for low-latency correction.

**Trainable**

Qwen LoRA, bridge, auxiliary heads; optionally freeze resampler/adapter after initial alignment.

**Expected latency**

Inference overhead minimal. Training cost and data curation higher.

**Addresses problems**

- Problem 1: improves quality/density of the slow latent.
- Problem 2: not DiT-side adaptation.
- Problem 3: not fast feedback.
- Problem 4: compatible.

**Precedent**

OpenVLA PEFT/LoRA results; OpenHelix prompt tuning; π0/π0.5 co-training with web/robot data.

**Risk**

Could spend scarce time tuning VLM while the main bottleneck is still DiT-side interface/control. Do only after proving that better bridge capacity helps.

---

## 4. Ranking for a 4x RTX A6000 setup

### Ranking by value × tractability

1. **Proposal 2 + Proposal 1 hybrid: wider tokens + fast plan-conditioned FiLM/AdaLN modulator.**  
   Best value. Directly attacks bandwidth and fast feedback while keeping VLM frozen and base NitroGen intact.

2. **Proposal 3: async planner + chunk stitching.**  
   High tractability and necessary engineering. It will not solve representation bottlenecks alone, but should be built before serious real-time eval.

3. **Proposal 4: learned `<ACT>` latent slots with auxiliary grounding losses.**  
   Good way to make Qwen hidden states action-grounded without full VLM fine-tuning.

4. **Proposal 6: structured goal/options interface.**  
   Very promising for 2D games, especially with emulator state/reward, but requires designing labels/rewards and may shift the project toward HRL.

5. **Proposal 8: Qwen LoRA for action grounding.**  
   Now allowed and likely useful, but higher cost. Do after a frozen-VLM bridge/modulator baseline shows headroom.

6. **Proposal 5: dynamic adapters / hypernetwork.**  
   Use the tame version only: LoRA scales, adapter-bank mixtures, or FiLM. Do not generate full DiT weights.

7. **Proposal 7: true online TTT.**  
   Research-interesting but low first-pass ROI. Use emulator save-state RL/GRPO offline or between attempts before real-time gradient updates.

8. **Single unified VLA / Fast-in-Slow-style partial sharing.**  
   Scientifically attractive, but too invasive for a fork built around frozen NitroGen + frozen/LoRA Qwen. Not a first-stage architecture.

### Single highest-leverage experiment to try first

**Experiment: “Dual-rate Modulated Bridge v1.”**

Implement:

1. `K=32` plan tokens from the existing frozen Qwen bridge.
2. A pooled plan latent in parallel.
3. `FastPlanMod` every NitroGen chunk:
   - inputs: current NitroGen vision features, pooled plan latent, previous 18 actions, chunk index since plan;
   - outputs: zero-initialized per-block FiLM/AdaLN gates for 4-8 selected DiT blocks, dynamic CFG scalar, and 4 corrective tokens.
4. Preserve null invariance exactly: if plan is null/masked, all gates/tokens/residuals are zero and base NitroGen output matches bitwise/numerically within tolerance.
5. Train on existing synthetic plan tasks plus emulator-generated perturbation/recovery data. Include staleness augmentation: train with plan latents delayed by 0-3 chunks.
6. Evaluate against current K=8 bridge on:
   - direction/button selectivity,
   - left/right separation probes,
   - plan flip rate,
   - mid-chunk or next-chunk recovery after forced perturbations,
   - exact null invariance,
   - latency.

Why first: it gives a decisive answer to the owner's hypothesis with minimal architectural upheaval. If it fails, we learn that either the plan labels/data are insufficient or NitroGen's frozen DiT cannot be steered via small modulation; if it works, it becomes the platform for async planning, learned `<ACT>` tokens, and RL.

### What not to bother with first

- **Do not run the full Qwen VLM every frame.** Robotics systems avoid this; Helix proves async latent caching is the right pattern.
- **Do not generate full DiT weights from a hypernetwork.** Too unstable and unnecessary; FiLM/AdaLN/LoRA scales capture most benefits.
- **Do not switch to fully autoregressive action tokens for NitroGen.** RT-2/OpenVLA-style tokenization helps semantic transfer but sacrifices the flow-action advantage NitroGen already has.
- **Do not train the whole VLM.** If VLM adaptation is needed, use LoRA/prompt/action tokens with grounding auxiliaries.
- **Do not over-invest in real-robot-only complexity** such as proprioception/multi-view embodiment adapters. 2D games need fast visual correction, action timing, and option progress more than embodiment generality.

---

## 5. Specific design notes for NitroGen

### A. Preserve the exact-null property

The current null/masked-plan mode is a major asset. Every proposal should preserve it:

- zero-init modulation heads;
- gate all plan-conditioned paths by `non_null_plan`;
- test null equality after every change;
- keep base cross-attn memory ordering stable.

This lets us use CFG-style guidance and safe ablations.

### B. Use plan guidance as a dynamic control variable

Fixed CFG≈8 is blunt. A fast module can emit `w_plan(t)`:

- high when the plan is confident and the frame matches expected context;
- low when the plan is stale, contradictory, or the current frame indicates danger;
- spike for planned button events like jump/fire;
- decay over chunks since last successful replan.

This is a low-risk fast path even before adding action residuals.

### C. Add bidirectional feedback without putting Qwen in the loop

“Bidirectional” does not need to mean Qwen reacts every frame. Instead:

- fast path computes summaries: stuck, progress, damage/death, action mismatch, velocity, repeated no-op, plan confidence;
- these summaries feed the next slow Qwen prompt;
- the fast path can trigger early replan if confidence/progress collapses;
- Qwen reads compact trace, not raw every-frame history.

This mirrors Hi Robot's feedback handling, but uses cheap telemetry between slow calls.

### D. Treat the plan latent as a cache with versioning

Each plan should have:

- `plan_id`, timestamp, source frame ids;
- text plan;
- K tokens;
- pooled latent;
- predicted horizon / termination condition;
- confidence and expected action/event sketch;
- stale flag.

The DiT/fast modulator should know `age_chunks` and `phase_in_plan`.

### E. Add representation tests inspired by OpenHelix

OpenHelix's central warning is that dual-system latents can become static instruction semantics. Add tests:

1. Same text, different frame: latent/control should differ when the correct action differs.
2. Same frame, different text: latent/control should differ in intended direction/button.
3. Same text/frame, different action trace: latent/control should adjust if previous attempt failed.
4. Probe mutual information between plan latent and target action sketch.
5. Measure DiT-space token separation for left/right/up/down/buttons, as already done in this project.

### F. Use emulator data for recovery, not only imitation

Robotics papers lean on demos; this project has a better substrate: save-states. Use it to generate paired counterfactuals:

- same state + different plan latents;
- perturb action chunk halfway then ask policy to recover;
- sample K plan/modulation variants and rank by emulator reward;
- train progress and replan-trigger heads.

This is where the 2D-game setting is stronger than real robots.

---

## 6. Glamorous but low-ROI ideas for this setting

1. **Full integrated System1-in-System2 VLA** like Fast-in-Slow. Interesting, but it discards the main asset: pretrained NitroGen. Use it only as inspiration for partial parameter sharing or action-grounded latents.
2. **Large world model for video prediction.** Helpful eventually, but overkill before fixing the control interface. Use lightweight progress/hazard heads first.
3. **Human-readable micro-plans every frame.** Too slow and text is not precise enough for frame-level button timing.
4. **Pure language-token bridge.** RT-2-style action tokens are powerful for semantic transfer, but our action chunks are already continuous and high-frequency; keep the flow policy.
5. **Online gradient TTT inside the live control loop.** Save-state batched adaptation is promising, but real-time backprop is likely brittle and latency-heavy.

---

## 7. Final recommendation

Build a Helix-like dual-rate architecture around NitroGen rather than trying to make Qwen plan more often. Concretely:

1. **Widen** the slow latent channel (`K=32/64`, multi-layer features, pooled latent).
2. **Modulate** the DiT with zero-initialized FiLM/AdaLN/gates from the plan latent.
3. **Add a fast corrective module** that sees current frame + cached plan + action trace every chunk.
4. **Run Qwen asynchronously** and train with plan staleness/latency offsets.
5. **Use emulator save-states** to train and evaluate recovery/progress, not just plan following.
6. Only then add Qwen LoRA or learned `<ACT>` tokens if the frozen-VLM latent proves under-grounded.

This path is faithful to the best robotics precedent, addresses all four owner-identified problems, and is feasible on 4x RTX A6000 without abandoning NitroGen's pretrained action prior.
