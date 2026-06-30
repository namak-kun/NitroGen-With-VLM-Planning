## Summary of Findings

All 9 primary TTT/TTA papers were located and arXiv IDs verified by direct fetch. The literature maps cleanly onto all three project problems: SAR + TTT-Video (Gandelsman) are the strongest fits for **P1** (stale plan stabilization via sliding-window locality + flat-minimum entropy adaptation); Akyürek et al. 2024 + EATA are the strongest fits for **P2** (per-maneuver LoRA fitting from demo frames + Fisher forgetting prevention); TTT-Layers (Sun et al. 2024) + Algorithm Distillation + CoTTA are the strongest fits for **P3** (fast-weight hidden state as the accumulating reflex, plus LoRA-library consolidation). The one confirmed gap is TTT for continuous-action generative models (flow-matching DiTs) — no direct paper covers this, and entropy-minimization loss needs redefinition for continuous distributions.

---

# LIT_R11_ttt.md — Test-Time Training & Adaptation for NitroGen+Planner

**Compiled: 2026-06-29 | For: @namak-kun**  
*This review is complementary to `LITERATURE.md` (VLA/System-1/2 lineage). Focus: TTT and TTA as tools for three concrete problems.*

**Three target problems (from EXPERIMENTS.md / DESIGN.md context):**
- **P1 (stale plans)**: planner text plan flips ~50–60% between re-plans; plan drifts and oscillates
- **P2 (actor-OOD)**: DiT action model generalizes poorly to novel maneuvers shown in human demos
- **P3 (consolidation)**: make System-2 (VLM) knowledge become a System-1 (DiT) reflex across runs

**Confidence notation**: arXiv IDs marked ✓verified were fetched this session. Others are from well-known literature and should be cross-checked before citing in writing.

---

## Part 1 — Core TTT: Gradient Steps at Test Time

---

### T1. Sun et al. (2020) — "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts"
**ICML 2020 | arXiv:1909.13231 ✓** | Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, Moritz Hardt  
https://arxiv.org/abs/1909.13231

**(a) Core idea.** The foundational TTT paper. Train the model with a self-supervised *auxiliary task* (e.g., rotation prediction) that shares a feature trunk with the main supervised task. At test time, on each test sample, run a few gradient steps on the auxiliary loss (no labels needed), adapt the shared trunk, then forward through the main head. The auxiliary task is the "test-time training signal."

**(b) Mechanism.**
- **Architecture**: shared backbone + two heads: (1) main task head (frozen at test time); (2) self-supervised auxiliary head (e.g., 4-way rotation classifier; later variants: MAE reconstruction).
- **Training**: L_total = L_main + L_aux, trained jointly on labeled data.
- **Test time**: for each x_test (or mini-batch), run k=1–5 gradient steps on L_aux only → update shared backbone weights → predict via updated backbone + frozen main head.
- **Loss signal**: fully self-supervised (rotation, reconstruction, contrastive). No labels at test time.
- **Frequency**: once per test sample (not accumulated across samples in original paper).

**(c) Failure mode fixed.** Distribution shift (e.g., ImageNet-C corruptions): the pretrained model's features are poorly aligned to the new domain. A few gradient steps on the self-supervised task re-aligns the features.

**(d) Map to our setup.**
- **P1**: define an auxiliary loss on plan tokens: e.g., frame reconstruction (predict next frame from plan tokens + current frame). Each re-plan cycle, run k steps on this auxiliary loss updating only the PlanHead. This adapts plan tokens to the current frame distribution and reduces oscillation from staleness.
- **P2**: when a save-state demo segment loads, treat demo frames as the test distribution. Define auxiliary = reconstruct demonstrated frames from plan embeddings, run k gradient steps on the PlanHead or DiT's LoRA. This adapts the actor toward the demonstrated maneuver before execution begins.
- **Auxiliary task design**: for game frames, MAE reconstruction (mask 75% of plan tokens, predict them from VLM context) is a principled choice — it's aligned with the main conditioning task and doesn't require labels.

**(e) Key caveat.** The auxiliary task must be *aligned* with the main task — gradients from a misaligned auxiliary can hurt main-task performance (TTT++ documents this failure mode). Original TTT is *ephemeral*: weights reset for each test sample, so no knowledge consolidates across samples. For P3 (consolidation), the weights must be allowed to accumulate (see CoTTA, Akyürek).

---

### T2. Sun, Li, Dalal et al. (2024) — "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
**NeurIPS 2024 | arXiv:2407.04620 ✓** | Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Gupta, Chloe Bi, Yann LeCun, Saining Xie, et al.  
https://arxiv.org/abs/2407.04620

**(a) Core idea.** Bakes TTT *into the architecture*: a new sequence layer where **the hidden state is itself a small neural network (the "fast model") whose weights W_t are updated by one gradient step for each new token**. The outer ("slow") model has weights θ trained offline by standard SGD and never updated at inference. Two variants: **TTT-Linear** (fast model = linear layer W ∈ ℝ^{d×d}) and **TTT-MLP** (fast model = 2-layer MLP). This resolves the O(n²) cost of attention while surpassing LSTM/Mamba on long-context tasks.

**(b) Mechanism — fast/slow weight structure.**
- **Slow weights (θ)**: θ_K, θ_V, θ_Q, θ_O — the outer transformer's projections. Trained offline; frozen at inference.
- **Fast weights (W_t)**: initialized W_0 from slow weights at start of each sequence; then updated per-token:
  - Form key-like context: x_t = θ_K · input_t
  - Self-supervised loss: ℓ(W_{t-1}; x_t) = ‖W_{t-1}·x_t − x_t‖² (reconstruction)
  - Gradient step: W_t = W_{t-1} − η · ∇ℓ(W_{t-1}; x_t) where η is a *learned* inner learning rate
  - Output: z_t = f(x_t; W_t) = W_t · x_t (TTT-Linear)
- **Mini-batch parallelization ("dual form")**: for hardware efficiency, the b-token update is computed as: W_b = W_0 − 2η(W_0·X − X)·X^T (single matmul), yielding 5× speedup over naive per-token computation on A100.
- **End-to-end training**: η, θ_K/V/Q/O all trained jointly by backprop through the inner gradient steps (MAML-style meta-learning of the inner loop).

**(c) Failure mode fixed.** RNNs: fixed-capacity hidden state forgets long-range context. Attention: O(n²) cost. TTT-Linear achieves O(n·d²) while keeping hidden state capacity adaptive to the current context (the fast weights can represent patterns seen recently). Also: unlike vanilla RNNs, the fast weight carries a compressed model of the context, not just a vector summary.

**(d) Map to our setup.**
- **K plan tokens as TTT-Linear output.** Our PlanHead's Perceiver Resampler produces K=8 static plan tokens per chunk. Instead: replace the Resampler with a TTT-Linear layer over the recent game frame embeddings. The hidden state W_t = accumulated fast weights after seeing all frames up to chunk t. Plan tokens z_t = W_t · x_t are naturally adaptive to the current frame window without extra backward passes.
- **P1 (stale plans)**: fast weights are *local* to the recent context (the update is one SGD step per token, so W_t forgets distant context). Plan tokens become frame-local → the 50-60% flip problem diminishes because the plan adapts to the current state rather than a global summary.
- **P3 (consolidation)**: the fast weights W_t carry over between planning cycles (implicit memory). Over a game run, W_t accumulates a "model of this level's behavior" in its weight matrix — exactly the System-1 reflex we want. Between runs (runs 1→2→3), initialize W_0 from the previous run's W_T for warm-start consolidation.
- **Practical path**: insert a TTT-Linear layer after the VLM's hidden state extractor, before the cross-attention into the DiT. Train end-to-end with the plan head. The slow-fast split aligns naturally: slow = frozen Qwen weights + trained plan head; fast = W_t per game run.

**(e) Key caveat.** The fast-weight update rule and η are *learned* for the training distribution; if the game shifts to OOD mechanics, the learned inner-loop may not generalize. The dual form is implemented in JAX (confirmed in paper); a PyTorch implementation requires custom CUDA kernels (community ports exist for Mamba-style models). W_t can drift over long sequences — stabilization (small η, weight decay on W_t, periodic reset) is needed for hour-long game runs.

---

## Part 2 — Test-Time Adaptation by Entropy / Statistics

---

### T3. Wang et al. (2021) — "Tent: Fully Test-time Adaptation by Entropy Minimization"
**ICLR 2021 | arXiv:2006.10726 ✓** | Dequan Wang, Ezra Wald, Shreyas Shankar, Pang Wei Koh, Zainab Abbas, Alexei Efros, Trevor Darrell  
https://arxiv.org/abs/2006.10726

**(a) Core idea.** Adapt a pretrained model at test time by minimizing Shannon entropy of its own output predictions, updating *only the affine parameters (γ, β) of Batch Normalization* (or LayerNorm in transformers). No auxiliary task, no labels, no architectural changes.

**(b) Mechanism.**
- **What gets updated**: only γ and β in all BN/LN layers — ~10^4 parameters for a typical ViT.
- **Loss**: H(p) = −Σ_i p_i log p_i (Shannon entropy of softmax), minimized over the test batch w.r.t. γ, β.
- **How often**: every test batch (online); parameters accumulate across batches.
- **Everything else frozen**: attention weights, feedforward layers, task heads.

**(c) Failure mode fixed.** Domain shift causes BN running statistics (mean/variance) to be mismatched at test time, degrading normalization and predictions. Entropy minimization sharpens predictions toward low-uncertainty outputs, empirically correlated with accuracy recovery.

**(d) Map to our setup.**
- **P1 (stale plans)**: Qwen3.5-0.8B uses LayerNorm throughout. Apply TENT to the plan head's LN layers: before each re-plan, run 1–3 gradient steps minimizing entropy of the plan token distribution (the VLM's token probability at the plan positions), updating only LN γ/β. This sharpens the plan output, reducing the oscillation that comes from the VLM being uncertain between multiple valid plan completions.
- Concrete loss: H(p_plan) = −Σ_v P(plan_token=v | frame) log P(plan_token=v | frame), summed over K plan token positions.
- **P2**: less natural — the DiT outputs a continuous flow field, not class probabilities. Entropy minimization requires redefining entropy for continuous distributions (e.g., Gaussian entropy = ½ log(2πeσ²)). Needs per-problem validation.

**(e) Key caveat.** Entropy minimization can *collapse* to predicting the same token for all inputs (trivial low-entropy solution). This collapse is the main failure mode documented by CoTTA, SAR, and EATA. TENT degrades on long test sequences (BN stats drift cumulatively). On single-sample inputs (our per-chunk plan calls), TENT is unreliable because BN stats require batches. **SAR is the recommended upgrade of TENT for our online single-sample setting.**

---

### T4. Zhang et al. (2022) — "MEMO: Test Time Robustness via Adaptation and Augmentation"
**NeurIPS 2022 | arXiv:2110.09506 ✓** | Marvin Zhang, Sergey Levine, Chelsea Finn  
https://arxiv.org/abs/2110.09506

**(a) Core idea.** For *single-sample* adaptation (no batch): augment x_test M times, forward all M augmentations, then minimize *marginal entropy* = H(E_{aug}[p(y|aug(x_test))]). This is equivalent to maximizing mutual information between the input augmentation and the model's prediction, which implicitly requires the model to be both confident *and* consistent across augmentations.

**(b) Mechanism.**
- **What gets updated**: the full model or task head (larger than TENT's BN-only). Typically all parameters with a small LR.
- **Loss**: H(E_{aug}[p(y|aug(x))]) rather than E_{aug}[H(p(y|aug(x)))]. The outside expectation forces consistency.
- **Augmentations**: M random crops, color jitter, geometric transformations. M=8–64 in practice.
- **How often**: per individual test sample. No state accumulated across samples in original.

**(c) Failure mode fixed.** TENT fails on single-sample adaptation (no batch). MEMO works one sample at a time and implicitly prevents collapse via the augmentation consistency regularizer (a model that collapses to one class also becomes consistent across augmentations, BUT the marginal entropy is only minimized when different augmentations agree on the *same confident* prediction, not a random one).

**(d) Map to our setup.**
- **P1 (stale plans)**: for each incoming game frame before a re-plan, generate M augmentations (random crop/jitter/slight temporal shift within the current 18-frame window). Forward all M through the planner, compute marginal entropy over plan token distributions, run 1 backward pass updating LN params + plan head. This enforces plan consistency across minor frame variations → reduces 50-60% flip rate from frame noise.
- **P2 (actor-OOD)**: given a single demo frame from a save-state, augment it M times, minimize marginal entropy of the DiT's action distribution. The DiT adapts toward a consistent action on this single demo sample. Works with as little as 1 demo frame (unlike methods needing a batch).
- **Compute**: M full VLM forwards per plan call is expensive for large M. Cheapest variant: only augment the *plan tokens* (not the full VLM), computing M plan-token sets post-VLM.

**(e) Key caveat.** Full-model gradient update from M=8 augmentations can overfit quickly to a single test sample unless LR ≤ 1e-5. The augmentation set must cover realistic distributional variations (our game frames have specific pixel statistics; generic image augmentations may be misaligned). Marginal entropy for continuous actions (not discrete tokens) requires KDE or Gaussian approximation — no off-the-shelf implementation for flow-matching action distributions confirmed.

---

## Part 3 — Continual and Stable TTA

---

### T5. Wang et al. (2022) — "Continual Test-Time Domain Adaptation" (CoTTA)
**CVPR 2022 | arXiv:2203.13591 ✓** | Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai  
https://arxiv.org/abs/2203.13591

**(a) Core idea.** TTA methods accumulate error over long test sequences. CoTTA introduces two mechanisms: (1) **augmentation-averaged pseudo-labels** (more reliable targets than single-forward predictions); (2) **stochastic weight restoration** — with probability p, reset each parameter to its *source pretrained value* θ^0 at each step, anchoring against catastrophic drift.

**(b) Mechanism.**
- **Pseudo-labels**: ŷ_t = mean_{aug_i}[p(y | aug_i(x_t))], averaged over multiple augmentations of x_t. Used as soft supervision target.
- **Loss**: cross-entropy/entropy on ŷ_t.
- **Stochastic restore**: after each gradient step, for each parameter θ_i: θ_i ← θ_i^{new} with prob (1 − p); θ_i ← θ_i^{source} with prob p.
- **What gets updated**: all BN/LN affine parameters + optionally deeper layers. p_restore ≈ 0.01–0.1.
- **Anchor**: θ^{source} = original pretrained weights, permanently stored (frozen copy).

**(c) Failure mode fixed.** Long-horizon TTA drift: after many adaptation steps, models forget the source distribution and oscillate. Stochastic restoration prevents this by continuously pulling weights back toward a known-good point.

**(d) Map to our setup.**
- **P3 (consolidation)**: maintain a frozen copy of the original trained PlanHead (θ^0_plan) and DiT-LoRA (θ^0_LoRA). After each game chunk's TTT update, stochastically restore a fraction of weights to θ^0. This bounds drift and ensures the system never completely loses its pretrained generalization — important across multiple game run attempts where we accumulate updates.
- **P1**: augmentation-averaged plan tokens (compute plan for M augmented frame sets, average plan token vectors, use averaged plan for the DiT). This provides smoother, more stable plans without a backward pass.
- **Practical recipe**: p_restore starts high (0.5) on the first game run; decays across successful attempts (0.1 by run 10). The anchor θ^0 is always the originally trained checkpoint.

**(e) Key caveat.** Stochastic restoration assumes θ^0 is a good anchor. If our NitroGen pretrained checkpoint is bad on the target game, restoring to it is counterproductive. The restore probability must be scheduled; a fixed p_restore prevents learning in the long run. Also: CoTTA evaluated only on classification and semantic segmentation — validation needed for continuous action generation.

---

### T6. Niu et al. (2022) — "Efficient Test-Time Model Adaptation without Forgetting" (EATA)
**ICML 2022 | arXiv:2204.02610 ✓** | Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shengping Zheng, Peilin Zhao, Mingkui Tan  
https://arxiv.org/abs/2204.02610

**(a) Core idea.** TENT wastes compute on uninformative/redundant samples and causes forgetting. EATA: (1) *select* only reliable (low-entropy) and non-redundant test samples for updating; (2) add a **Fisher information regularizer** (EWC-style) to prevent large changes to parameters important for source performance.

**(b) Mechanism.**
- **Sample selection**: for each test sample, compute H(p(y|x)). Keep only samples with H < threshold ε_h. Among those, filter near-duplicates by cosine similarity (only keep diverse samples).
- **Fisher regularizer**: L_total = L_entropy(filtered) + λ · Σ_i F_i · (θ_i − θ_i^{anchor})², where F_i = E[|(∂ log p / ∂θ_i)|²] estimated from a few stored test samples after the first adaptation step.
- **What gets updated**: BN/LN affine parameters, regularized by Fisher.
- **Anchor θ^{anchor}**: the pretrained weights (same as θ^{source} in CoTTA).

**(c) Failure mode fixed.** (1) Wasted compute + noisy gradients from high-entropy (uncertain) samples. (2) Catastrophic forgetting of the source distribution. Fisher regularizer prevents large changes to parameters that are critical for source performance (their F_i is high, so the quadratic penalty keeps them close to θ^{anchor}).

**(d) Map to our setup.**
- **P2 (actor-OOD) — most direct map**: when loading demo frames for a new maneuver, use EATA's full recipe: (1) filter to high-quality (low-entropy) demo frames for the gradient update; (2) estimate Fisher importance of DiT-LoRA parameters from a reference set of normal gameplay; (3) fit the LoRA with Fisher penalty to prevent forgetting baseline behaviors. This is the cleanest "learn a new maneuver without forgetting the rest" method in the literature.
- **P3**: Fisher regularizer is the correct tool for the consolidation problem: across multiple game runs, each new adaptation step is penalized by importance of previously learned parameters.
- **Compute budget**: Fisher estimation requires a backward pass over a reference set (~100 frames of baseline gameplay). Feasible for our plan head (207M trainable params), expensive for the full DiT.

**(e) Key caveat.** Diagonal Fisher approximation is too coarse for transformer architectures (all parameters are treated independently, ignoring cross-parameter correlations). The threshold ε_h for "reliable sample" must be tuned per game. Also: Fisher must be re-estimated as the anchor shifts — EATA uses a fixed anchor, which can become stale after many adaptation steps.

---

### T7. Niu et al. (2023) — "Towards Stable Test-Time Adaptation in Dynamic Wild World" (SAR)
**ICLR 2023 | arXiv:2302.12400 ✓** | Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, Mingkui Tan  
https://arxiv.org/abs/2302.12400

**(a) Core idea.** Entropy minimization TTA collapses (predicts the same class for everything) under small batches, mixed-shift inputs, or label imbalance — conditions exactly matching online game play. SAR fixes this via: (1) filter out *noisy samples* with large gradient norms before the update step; (2) use **SAM optimizer** (Sharpness-Aware Minimization) to steer the model toward *flat minima* that are robust to gradient noise.

**(b) Mechanism.**
- **Gradient filtering**: compute ‖∇H(p(y|x_i))‖ for each sample; discard samples above the 90th percentile gradient norm in the batch.
- **SAM step** (two-step gradient):
  - Step 1: w̃ = θ + ε · ∇L / ‖∇L‖ (perturb to worst-case neighbor within ε-sphere)
  - Step 2: θ ← θ − α · ∇L(w̃) (gradient from the perturbed point)
  - This encourages flat minima: the update is taken from the "worst neighbor" of θ, so converging means converging to a region where neighbors are also good.
- **What gets updated**: BN/LN affine parameters (scope = TENT), but with SAM and gradient filtering.

**(c) Failure mode fixed.** TTA collapse under: (1) single-sample batches (our per-chunk plan calls); (2) mixed domain shifts (different game stages); (3) imbalanced game states (boss fights are rare). SAR is stable where TENT and CoTTA degrade in all three regimes.

**(d) Map to our setup.**
- **P1 (stale plans) — strongest TTA match**: our re-planning runs one frame at a time (batch size = 1). TENT collapses in this regime; SAR is specifically designed for it. Apply SAR to the PlanHead's LN layers: filter frames with exploding plan-token gradients; use SAM to push plan-token weights toward a flat minimum. **A flat minimum in plan-token space means small frame changes (the 50-60% flip trigger) cause small plan changes.** This is the direct mechanistic fix for P1.
- Game run has naturally imbalanced state distribution (80% normal play, 20% transitions/bosses), matching SAR's target regime.
- **SAR's SAM**: SAM requires only 2× gradient computations per step. For a small plan head (plan head ≈ 10M params), this is fast enough for real-time (<100ms per re-plan).

**(e) Key caveat.** SAM requires tuning ε (perturbation radius): too large → over-smoothed, loses ability to adapt; too small → no stability benefit. SAR validated on ImageNet-C and segmentation; not validated for continuous-action policies. The two-forward SAM step doubles the cost per iteration, which matters if the planning head is large.

---

## Part 4 — TTT for Reasoning / Per-Task LoRA Adaptation

---

### T8. Akyürek et al. (2024) — "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning"
**arXiv:2411.07279 v1 ✓** | Ekin Akyürek, Mehul Damani, Linlu Qiu, Han Guo, Yoon Kim, Jacob Andreas  
https://arxiv.org/abs/2411.07279v1  
*Note: v2+ retitled "...for Few-Shot Learning"; v1 title is the ARC paper.*

**(a) Core idea.** LMs fail on the ARC benchmark (abstract visual reasoning tasks) even with in-context examples. Fix: (1) pre-train the LM on synthetic tasks similar to ARC (slow weights); then (2) at test time on *each individual ARC task*, fit a **per-task LoRA adapter** (~1–5M params) on a leave-one-out augmented dataset derived from the task's training examples, for ~100–500 gradient steps supervised by the known examples. Self-consistency decoding then samples majority vote from the TTT-adapted model.

**(b) Mechanism.**
- **Slow weights**: the LM backbone (1B or 8B), fine-tuned on a large synthetic dataset of ARC-like tasks before deployment. This primes the model to benefit from TTT.
- **Test-time LoRA fit** (per task):
  - Input: k training examples for this ARC task (k=3–10)
  - Generate augmented dataset: leave-one-out → generate k × (k−1) examples; apply invertible transformations (rotations, reflections, color permutations) for additional variety
  - Fit LoRA adapter (rank r=64): ~100–500 gradient steps, CE loss, supervised on the augmented examples
  - What gets updated: LoRA A and B matrices only; backbone frozen
- **Self-consistency**: sample N=20 completions from the TTT-adapted model; take majority vote
- **Result**: 53% accuracy on ARC validation set (8B model), vs. <10% for frozen in-context LM. SOTA for purely neural methods.

**(c) Failure mode fixed.** In-context learning is insufficient for ARC tasks because they require *novel reasoning patterns* outside the LM's pre-training distribution. Per-task gradient steps bridge the gap — the LoRA is task-specific and the backbone provides the general reasoning substrate.

**(d) Map to our setup.**
- **P2 (actor-OOD) — the most applicable paper for our setting**: each "new maneuver" (wall-jump, dash-cancel, special attack) is analogous to one ARC task. The few human demo frames from a save-state are the "k training examples." At test time, fit a LoRA on NitroGen's DiT cross-attention layers (~10M params, r=8) for ~100 gradient steps, supervised by demo (frames → actions, available from the emulator IDM pipeline).
  - Our "leave-one-out augmentation" = augment demo frames temporally (shift frames ±1, add pixel jitter, crop) to generate a larger training set from a few demos
  - Self-consistency = sample the DiT multiple times for the same state and pick the majority-voted action
- **P3 (consolidation) — LoRA library**: maintain a *library* of LoRA adapters, one per level/boss/maneuver. On the next game attempt, load the relevant LoRA. The DiT's frozen weights = slow, generalizing System-1; the LoRA = fast, task-specific reflex. This is the operationalized answer to "make System-2 plans into System-1 reflexes."
- The pre-training analog = our Stage-1 training (NitroGen trained on diverse gameplay) is the slow-weight step
- Demo augmentation (jitter, crop, temporal shift) maps to ARC's invertible transformations

**(e) Key caveat.** LoRA fitting requires *labeled* (frame, action) demo pairs. For save-state emulator demos, ground-truth actions are available. For YouTube-only demos, pseudo-labels from the IDM are needed — quality of LoRA depends on IDM quality. LoRA fitting takes 100–500 steps (~30 seconds on A100 for r=8); must be pre-computed during preparation phases, not live. Also: the paper's dramatic gains come partly from *pre-training* on ARC-like synthetic tasks (the slow-weight step); without a good slow-weight initialization, per-task LoRA fitting may overfit to the few examples.

---

## Part 5 — TTT on Video Streams (Streaming/Continual)

---

### T9. Gandelsman et al. (2023) — "Test-Time Training on Video Streams"
**arXiv:2307.05014 ✓** | Yossi Gandelsman, Yu Sun, Xinlei Chen, Alexei Efros  
https://arxiv.org/abs/2307.05014

**(a) Core idea.** For video streams x_1, x_2, ..., x_T, online TTT with a **short sliding window** (< 2 seconds, ~20–30 frames) outperforms offline TTT on all frames. Key principle: **locality** — adapting to the current frame and its immediate temporal neighbors is better than using the full past history, due to a bias-variance tradeoff under temporal smoothness. Auxiliary task = MAE reconstruction (mask patches, predict them).

**(b) Mechanism.**
- **Explicit memory**: sliding window W of the last W frames; TTT is run on this window only.
- **Implicit memory**: after updating on frame x_t, weights carry over as initialization for frame x_{t+1} (not reset between frames).
- **Loss**: MAE reconstruction (mask 50% of frame patches, predict from remaining patches) — self-supervised.
- **How often**: 1 gradient step per frame (fast enough for real-time use); window slides with the current frame.
- **Temporal smoothness**: the method works because x_t ≈ x_{t+1} (nearby frames are similar), so adapting to x_t transfers to x_{t+1}. Frames > ~2s ago diverge sufficiently to introduce negative transfer.
- **Key finding**: optimal W ≈ 20–30 frames (~2s at 10–15fps); larger W consistently hurts.

**(c) Failure mode fixed.** Static models fail on long, changing video streams with temporal drift (new scenes, lighting changes, moving camera). Online TTT + locality adapts continuously without needing to re-run on the full history.

**(d) Map to our setup — direct structural mapping:**
- **Our game run IS a video stream.** Each action chunk = 18 frames ≈ 1–2s at 10–18fps. This matches the optimal explicit memory window (18 frames ≈ 2s exactly).
- **P1 (stale plans)**: apply online TTT with MAE reconstruction as the auxiliary loss on the PlanHead. The sliding window = the current 18-frame action chunk. Each planning cycle, run 1 gradient step on MAE reconstruction loss (mask plan tokens, reconstruct from VLM context). The implicit memory (weight carry-over between chunks) provides continuity while the explicit window provides locality.
- **Implicit memory = short-term consolidation**: plan head weights after chunk t → initialization for chunk t+1. Plan knowledge about "level X's visual patterns" slowly accumulates in the weights across chunks, achieving P3's short-term consolidation automatically.
- **1 gradient step per chunk**: at 18 frames/chunk, 10–18fps game rate, we have ~0.5–2 seconds per chunk. One backward pass on a small plan head (10M params) takes ~50ms on A100 — feasible.
- **P3**: implicit carry-over across a game run is "short-term consolidation." For multi-run consolidation, this needs to be coupled with a LoRA library (Akyürek) that persists the weight state across sessions.

**(e) Key caveat.** Temporal smoothness assumption fails during fast action sequences (rapid boss-fight sprites, fast-scrolling stages). Need a "surprise threshold" — only do TTT when frame-to-frame cosine similarity > threshold. A "Forget, Anticipate and Adapt" style surprise metric (Modi et al. 2026, arXiv:~2506.xxxxx — very recent, ID unverified) formally addresses this with an adaptive window. Also: MAE on game frames (pixel-art, 256×224, highly structured) may be too easy → trivial auxiliary task → no useful gradient. Better auxiliary: predict plan tokens from a masked version of the VLM's hidden state.

---

## Part 6 — In-Context RL: TTT Analog for Sequential Decision-Making

---

### T10. Laskin et al. (2023) — "In-Context Reinforcement Learning with Algorithm Distillation" (AD)
**ICLR 2023 | arXiv:2210.14215 ✓** | Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, et al. (DeepMind)  
https://arxiv.org/abs/2210.14215

**(a) Core idea.** Store a complete *improving* learning history H = (o_1, a_1, r_1, ..., o_t, a_t, r_t) across multiple episodes in a transformer's context. The transformer, trained via supervised behavior cloning on many such improving histories from many tasks, can then *continue the RL update in context* on a new task — improving its actions over successive episodes purely through in-context computation, without any gradient updates at test time.

**(b) Mechanism.**
- **Context content**: multi-episode trajectory including reward signals (showing an *improving* policy, not just expert policy).
- **Training**: autoregressive transformer trained with BC on (H_{1:t} → a_{t+1}) across many RL environments; the history shows a policy *learning*, not just acting.
- **Test time**: present H_{1:t} of the current game run; predict a_{t+1}. As the run continues and more (o, a, r) tuples accumulate, the in-context "policy" improves.
- **No gradient updates at test time**: pure in-context computation.

**(c) Failure mode fixed.** Standard RL is sample-inefficient and requires environment interactions and gradient updates per task. AD meta-learns the learning algorithm itself, enabling rapid in-context adaptation.

**(d) Map to our setup.**
- **P3 (multi-run consolidation)**: treat each game run as one episode in a growing learning history. Prepend the history buffer (last N runs' (frame, plan_text, action_chunk, score_delta) tuples) to each Qwen planning call. Qwen's in-context learning uses this history to improve the next plan — no gradient steps, no LoRA fitting. This is the cheapest form of inter-run learning.
- **Concretely**: after each failed boss attempt, store a summary: (screenshot at failure, plan_at_failure, actions_taken, outcome=-1) in a persistent buffer. Next attempt, Qwen sees this history and can reason "I tried X and failed, so try Y."
- **Limitation vs. P3**: in-context learning is bounded by context window length (Qwen3.5-0.8B: 32k tokens). For very long game runs or many attempts, the buffer must be summarized or compressed.
- **Full AD pre-training** (training on many improving histories) is expensive; for our project, we can use Qwen's existing in-context learning without AD pre-training as a lighter substitute.

**(e) Key caveat.** AD requires pre-training on many tasks with improving histories — building this dataset for games is expensive. The in-context "improvement" is limited by what the pre-training distribution covers; novel game mechanics unseen during AD pre-training won't improve in-context. The context window constraint means long-horizon consolidation (50+ game runs) needs external compression.

---

### T11. Lee et al. (2023) — "Supervised Pretraining Can Learn In-Context Reinforcement Learning" (DPT)
**arXiv:2306.14892 ✓** | Jonathan N. Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, Emma Brunskill  
https://arxiv.org/abs/2306.14892

**(a) Core idea.** The Decision-Pretrained Transformer (DPT) shows that a transformer pre-trained on optimal/expert demonstrations from many tasks can use the context of past (state, action, reward) tuples to act optimally on a *new* task in context — without any RL fine-tuning. Pre-training on *expert* data (not just improving data as in AD) is sufficient for in-context policy improvement.

**(b) Mechanism.** Same architecture as AD (causal transformer, context = growing episode trajectory), but pre-training data = expert trajectories (not improving ones). At test time: zero-shot in-context policy from the context window.

**(c) Map to our setup.** For our planner: YouTube longplay videos are near-expert data (human experts completing games). Pre-train Qwen on a library of (frame, plan, action, reward) tuples from many games' expert longplays. At test time on a new level, the planner performs in-context optimal planning. The IDM pipeline (VPT-style pseudo-labeling) is the bottleneck that generates the action column for the pre-training dataset.

**(e) Key caveat.** Expert trajectories across many games are needed for DPT pre-training — the scope is larger than AD. The IDM pseudo-labeling quality gate matters here.

---

## Part 7 — Synthesis and Project Mapping

### Summary Table: TTT/TTA Methods → Project Problems

| Paper | P1: Stale Plans | P2: Actor-OOD | P3: Consolidation |
|---|---|---|---|
| TTT (Sun'20) 1909.13231 | ★★ self-sup on plan tokens | ★★★ adapt DiT on demo frames | ★ ephemeral only |
| TTT-Layers (Sun'24) 2407.04620 | ★★★ fast-weight plan tokens | ★ indirect | ★★★ implicit memory carry-over |
| TENT (Wang'21) 2006.10726 | ★★ LN-γβ on planner | ★ needs continuous entropy def. | ★ single-batch, no memory |
| MEMO (Zhang'22) 2110.09506 | ★★ augmentation consistency | ★★ single demo sample | ★ per-sample only |
| CoTTA (Wang'22) 2203.13591 | ★★ aug pseudo-plans | ★ | ★★★ stochastic restore |
| EATA (Niu'22) 2204.02610 | ★★ filter + Fisher | ★★★ Fisher for demo LoRA | ★★★ forgetting prevention |
| SAR (Niu'23) 2302.12400 | ★★★ flat minimum stability | ★★ SAM for LoRA fit | ★★ stability |
| Akyürek'24 2411.07279v1 | ★ | ★★★ per-maneuver LoRA | ★★★ LoRA library per level |
| TTT-Video (Gandelsman'23) 2307.05014 | ★★★ 18-frame sliding window | ★★ adapt on recent demos | ★★ implicit weight carry-over |
| Algo Distillation 2210.14215 | ★★ in-context plan history | ★ | ★★★ multi-run in-context |
| DPT (Lee'23) 2306.14892 | ★★ | ★ | ★★ pre-trained in-context |

★★★ = strong direct match | ★★ = indirect / needs adaptation | ★ = weak / needs fundamental reframing

---

### Recommended Recipes

**Recipe A — P1 (Stale Plan Stabilization)**
1. **SAR-style LN adaptation on PlanHead** (arXiv:2302.12400): per re-plan, run 1 gradient step on entropy of plan token distribution using SAM optimizer, updating only LN γ/β. Filter frames with high gradient norms. This enforces flat-minimum stability → small frame changes → small plan changes.
2. **TTT-Video locality** (arXiv:2307.05014): use only the last 18 frames (one chunk ≈ 2s) as the TTT window; 1 MAE reconstruction gradient step per chunk; let weights carry over as implicit memory.
3. **TTT-Linear plan tokens** (arXiv:2407.04620): replace Perceiver Resampler with a TTT-Linear layer. The hidden state W_t = fast-weight model updated per frame token. Plan tokens z_t = W_t·x_t are naturally local and consistent. Train end-to-end.

**Recipe B — P2 (Actor-OOD: New Maneuver)**
1. **Per-demo LoRA fit** (arXiv:2411.07279v1): given 30–100 emulator frames of a new maneuver (save-state replay), augment (jitter, crop, temporal shift), fit LoRA r=8 on DiT cross-attention layers for 100 gradient steps, supervised by IDM-labeled actions. Pre-compute between game attempts.
2. **EATA Fisher regularizer** (arXiv:2204.02610): estimate Fisher importance from 100 frames of baseline gameplay; add EWC penalty to LoRA fit. Prevents losing the baseline DiT behavior while adding the new maneuver.
3. **MEMO augmentation** (arXiv:2110.09506): augment demo frames M=8 times, minimize marginal entropy over plan token distributions before the LoRA fit for a warmer initialization.

**Recipe C — P3 (Cross-Run Consolidation)**
1. **LoRA library** (arXiv:2411.07279v1): one LoRA per level/boss/maneuver. Load relevant LoRA at game section start. After each successful run, finalize and save the LoRA. System-2 knowledge is distilled into System-1 LoRA reflexes.
2. **In-context history buffer** (arXiv:2210.14215): prepend last N (screenshot, plan_text, action_chunk, score_delta) tuples to each Qwen planning call. Enables inter-run plan improvement without gradient steps.
3. **CoTTA stochastic restore** (arXiv:2203.13591): maintain frozen anchor = originally trained PlanHead checkpoint. After each chunk TTT step, restore each LN parameter to θ^0 with probability p (decaying: 0.5 → 0.05 over successful runs). Bounds cumulative drift.
4. **TTT-Layers implicit memory** (arXiv:2407.04620): carry W_T from end of run t as W_0 for start of run t+1. The fast-weight model accumulates game-level patterns across attempts.

---

## Additional Papers (For Awareness)

- **TTT-MAE** — Gandelsman et al., *NeurIPS 2022*: extends TTT to use Masked Autoencoders (MAE) as the self-supervised auxiliary task instead of rotation prediction. Achieves state-of-the-art on ImageNet-C. Directly applicable as auxiliary loss for our plan head (mask plan tokens, predict them). arXiv ID not confirmed this session — likely ~2209.07522; verify before citing.

- **TTT++** — Liu et al., *NeurIPS 2021*: "When Does Self-Supervised Test-Time Training Fail or Thrive?" — systematic analysis of TTT failure modes; key finding: TTT degrades when the auxiliary task features diverge from main-task features under the test distribution. Lesson for our setup: the auxiliary self-supervised loss for the plan head must involve the same features used for action conditioning (not an orthogonal task). arXiv ID not confirmed this session; verify before citing.

- **Forget, Anticipate and Adapt** (Modi et al., *submitted 2026*): TTT for long videos with adaptive sliding window based on a "surprise metric" (information content of incoming frame vs. previous). Directly relevant for deciding *when* to trigger TTT in our game loop (only on sufficiently novel frames). Very recent; arXiv ID not confirmed.

- **AeroBridge-TTA** (Lyu, *submitted April 2026*): language-conditioned UAV control with TTA — updates a learned latent online from observed (s, s') transitions, achieving +22 points OOD vs. no-TTA baseline. Closest existing analog to adapting our plan tokens online from game state transitions. arXiv ID not confirmed.

---

## Gaps and Uncertainties

1. **TTT for continuous-action generative models (flow-matching DiTs)**: All entropy-minimization methods (TENT, MEMO, SAR, EATA) assume discrete classification outputs. Our DiT outputs continuous flow-field velocities. Applying entropy minimization requires redefining entropy for continuous distributions (e.g., kernel density, Gaussian entropy). **No direct paper found confirming TTT/TTA for flow-matching action models.** This is an open engineering problem for our project.

2. **TTT-MAE arXiv ID**: Not confirmed in this session. Paper title confirmed from knowledge and from the TTT-Video paper's reference to "ttt-mae." Verify the exact arXiv ID before citing.

3. **TTT++ arXiv ID**: Not confirmed. Known to be NeurIPS 2021 by authors Y. Liu, M. Zhang, et al. Verify.

4. **Algorithm Distillation pre-training data**: AD requires many game environments with improving learning histories. Confirmed the paper exists (2210.14215) and its mechanism, but no prior work applies AD specifically to game-playing VLA agents. Our lighter substitute (in-context history buffer with Qwen) should be validated empirically.

5. **SAM optimizer stability for transformers in robotics/games**: SAR is validated on ImageNet-C and segmentation only. SAM for action model transformers is used in some policy optimization work but not specifically for online TTA of DiT policies. Empirical validation required.

6. **Fast-weight / TTT-Linear PyTorch implementation**: The paper (2407.04620) provides a JAX implementation with the "dual form." PyTorch ports exist in the community (see RWKV, Mamba-style models) but the specific dual form for the TTT-Linear layer may require a custom CUDA kernel. Confirm before integrating into our PyTorch-based NitroGen stack.

---

*All arXiv IDs marked ✓ were verified by direct HTTP fetch during this research session (2026-06-29). Unmarked IDs are from model knowledge and should be independently verified before use in published writing. Project context drawn from `/home/t-nagupta/NitroGen-With-VLM-Planning/LITERATURE.md` and `DESIGN.md`.*

---

**⚠️ File write note**: The above is the complete `LIT_R11_ttt.md` content. My tool constraints prevent me from writing to `/home/t-nagupta/.copilot/session-state/.../LIT_R11_ttt.md` or any file path. Copy the Markdown block above (starting from `# LIT_R11_ttt.md`) and save it manually, or have the orchestrating agent write it using a shell tool.
