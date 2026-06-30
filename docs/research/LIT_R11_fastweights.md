# Literature Review — Fast Weights, Meta-Learning, Dual-Process VLAs & Conditioning Mechanisms

*Compiled 2026-06-29. Focus: grounding for NitroGen + VLM Planner (frozen Qwen3.5 → Perceiver resampler → K=8 plan tokens → frozen flow-matching DiT cross-attention). Covers SHORT continuous path (plan tokens as implicit fast weights) vs LONG explicit path (planner-as-system-2), and mechanisms for fast test-time adaptation.*

*Verification note: arXiv IDs confirmed by live fetches this session (✓). Papers not on arXiv (Figure Helix, GR00T N1.5, Schmidhuber 1992) noted as ⚠️ blog/venue-only.*

---

## SECTION 1 — FAST WEIGHTS / ASSOCIATIVE MEMORY

### 1.1 Schmidhuber (1992) — The Original Fast Weights ⚠️

**Citation:** J. Schmidhuber, "Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks," *Neural Computation*, 4(1):131–139, 1992. [No arXiv; original venue: ICANN 1991 / Neural Computation 1992.]

**(a) Core idea:** A "slow" network trained by gradient descent writes temporary memories into a secondary "fast weight" matrix; a second network reads them. The slow weights store long-term structure; the fast weights store a rapidly-changing, context-specific working memory updated at inference time.

**(b) Exact mechanism:** Slow net (System 2 analogue) writes a key-value outer product `ΔW = η·v·kᵀ` into a fast weight matrix W; fast net (System 1 analogue) queries W with `Wx` at each step. The fast weights are overwritten at each context step — they are *per-example*, ephemeral parameters, not trained by gradient descent.

**(c) What it enables:** Decoupled timescales: slow weights generalize across episodes; fast weights specialize to the current episode/context. Avoids storing explicit copies of activations.

**(d) Map to our setup:**
- (i) ✅ **K plan tokens as fast weights**: The Perceiver resampler writes VLM hidden states into K fixed-size query outputs. At each new plan, those K tokens are overwritten. They are "fast" (re-computed per call) relative to the DiT's frozen weights. This is structurally identical to Schmidhuber's fast-weight write.
- (ii) Plan tokens are injected via cross-attention, not AdaLN/FiLM, so this is the cross-attention path. The plan_adaln=True flag (EXP-048) adds the additive-modulation path separately.
- (iii) ❌ No direct async/dual-rate angle here; this is the episodic memory framing.

**(e) Key caveat:** The 1992 paper uses outer products to write W, not a learned nonlinear resampler. Our resampler is far more expressive and the K tokens are high-dimensional embeddings, not scalar-weighted activity patterns. The analogy is structural, not mechanistic.

---

### 1.2 Ba, Hinton, Mnih, Leibo, Ionescu (2016) — Fast Weights Attend to the Recent Past ✓

**Citation:** J. Ba, G. Hinton, V. Mnih, J.Z. Leibo, C. Ionescu, "Using Fast Weights to Attend to the Recent Past," *NeurIPS 2016*. arXiv: [1610.06258](https://arxiv.org/abs/1610.06258). Submitted Oct 20, 2016.

**(a) Core idea:** Introduces a third tier of memory between activations and slow weights: "fast weights" W(t) updated at every step via `W(t) = λW(t-1) + η·h(t)·h(t)ᵀ`, i.e., a decaying outer-product accumulator of recent hidden states. A "fast net" then processes queries by attending to W(t) with a layer-norm inner loop.

**(b) Exact mechanism:** Slow net (LSTM or FF) produces hidden state h(t). Fast weight matrix W(t) accumulates recent h's via additive outer products (with decay λ, learning rate η). At each step, an inner loop: `s_{n+1} = LN(W(t)·s_n + Wh(t) + bs)` iterates ~1–3 times to retrieve from W. The fast weights act as an associative memory for short-horizon context; the slow weights act across episodes.

**(c) What it enables:** Sequence-level working memory without explicit memory slots; a learned "attend to the recent past" mechanism that can retrieve episodic cues without LSTM-style gating.

**(d) Map to our setup:**
- (i) ✅ **STRONG analogy**: Our K plan tokens are a *discrete*, higher-level analogue of Ba et al.'s W(t). At each new plan cycle, the resampler computes K plan token vectors from the current VLM hidden states — this is writing a new "fast weight" state. The DiT's cross-attention over plan tokens is the "read" step. The plan tokens are ephemeral (re-computed each call) relative to the frozen DiT, exactly as W(t) is ephemeral relative to LSTM slow weights.
- (ii) The null-mode='masked' design (EXP-048 plan_adaln=False baseline) is the pure cross-attention read path; plan_adaln=True adds a FiLM write directly into the DiT timestep conditioning.
- (iii) ❌ Ba et al. operate synchronously; no async dual-rate design.

**(e) Key caveat:** Ba et al.'s W(t) is a *dense matrix* computed over raw activations — high-dimensional and updated at every step. Our plan tokens are computed at sparse planner cadence (~1–10 Hz for the VLM) and held constant across the intervening DiT calls (the async latency-hiding pattern). This mismatch motivates the cross-chunk persistence design in `MULTICHUNK_DESIGN.md`.

---

### 1.3 Ramsauer et al. (2020) — Modern Hopfield / "Hopfield Networks Is All You Need" ✓

**Citation:** H. Ramsauer, B. Schäfl, J. Lehner, P. Seidl, M. Widrich, T. Adler, L. Gruber, M. Kopp, M. Klambauer, J. Brandstetter, G. Klambauer, J. Bock, A. Hochreiter, S. Hochreiter, "Hopfield Networks is All You Need," *ICLR 2021*. arXiv: [2008.02217](https://arxiv.org/abs/2008.02217). 2020.

**(a) Core idea:** Modern continuous Hopfield networks with exponential capacity can store and retrieve exponentially many patterns, and their retrieval dynamics are *exactly* softmax attention. The storage update rule `ξ_new = softmax(β·Ξᵀx)·Ξ` is one step of Hopfield retrieval, which is the softmax attention dot product.

**(b) Exact mechanism:** Stored patterns Ξ = [ξ₁,…,ξₙ] (the "memory"). Query x (the "retrieval state"). Update: `x_new = Ξ·softmax(β·Ξᵀx)`. This is exactly `x_new = V·softmax(Kᵀx/√d)` with K=V=Ξ and β=1/√d. Retrieval is attention; attention is associative retrieval.

**(c) What it enables:** Exponential capacity for pattern storage; retrieval in one attention step at high β; theoretical foundation for treating softmax attention as a content-addressable memory.

**(d) Map to our setup:**
- (i) ✅ **CRITICAL ANALOGY**: The K plan tokens (K=8) as keys+values for DiT cross-attention is a small Hopfield memory. The DiT's query (the noisy action) retrieves from {K=8 plan patterns} stored in the plan token matrix. High-temperature (low β) → distributed, smooth blending across plan tokens; high-temperature (high β) → sharp retrieval of the most-relevant plan token. This is exactly the modern Hopfield reading operation.
- (ii) The zero-init AdaLN/FiLM (plan_adaln=True) is *not* a Hopfield retrieval — it's a global modulation. The cross-attention injection is the Hopfield path.
- (iii) ❌ Not an async design.

**(e) Key caveat:** With K=8 plan tokens, capacity is tiny (modern Hopfield capacity ≈ d^(K/2); d=1024, K=8 → large, but only 8 distinct patterns can be stored). The DiT cross-attends to all 8 tokens + 256 image tokens, so plan tokens are a minority. This motivates the per-token contrastive loss (pertoken mode in PlannerConfig) to maximize per-position discriminability.

---

### 1.4 Schlag, Irie, Schmidhuber (2021) — Linear Transformers Are Fast Weight Programmers ✓

**Citation:** I. Schlag, K. Irie, J. Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers," *ICML 2021*. arXiv: [2102.11174](https://arxiv.org/abs/2102.11174). Feb 22, 2021.

**(a) Core idea:** Linear attention (softmax replaced by kernel φ(Q)φ(K)ᵀ) is formally equivalent to a Fast Weight Programmer (FWP): the key-value outer product `Σᵢ φ(kᵢ)·vᵢᵀ` is the fast-weight matrix W; the query φ(q)ᵀ·W is the fast-weight read. The "slow net" writes key-value pairs into W; the "fast net" reads them at query time.

**(b) Exact mechanism:** Standard linear attention computes `W = Σᵢ φ(kᵢ)⊗vᵢ` (outer product accumulation). Reading: `output = W·φ(q)`. This is a recurrent update to W that stores context. Schlag et al. replace pure additive outer products with a delta-rule update `ΔW = (v-Wk)kᵀ` to correct stale entries, improving capacity. The FWP framework also allows a learned dynamic learning rate.

**(c) What it enables:** Unifies linear recurrent models (Mamba, RWKV, RetNet), linear attention, and fast weights under one framework. Explains why linear attention degrades for long contexts (bounded capacity in W) and motivates delta-rule corrections.

**(d) Map to our setup:**
- (i) ✅ **THEORETICAL FOUNDATION for our short path**: The K plan tokens flowing into the DiT's cross-attention are, in the FWP frame, writing K key-value slots into the DiT's effective "context memory". The frozen DiT's cross-attention reads them with softmax (not linear), but the write is via the resampler + adapter projecting VLM states into (key, value) pairs — exactly a fast-weight write. **This gives us language to say: the short path (plan → resampler → K tokens → cross-attention) is a fast-weight program; the long path (VLM reasoning → explicit symbolic decomposition → language command → parser) is the slow-weight path.**
- (ii) AdaLN-zero (plan_adaln=True) is a different conditioning mechanism — it modulates the DiT's layer-norm scale/shift, not the KV memory.
- (iii) The delta-rule correction is relevant for EXP-052 text_only mode: if plan tokens overwrite each other (additive outer-product overflow at K=8), the delta rule could correct stale plan entries from previous timesteps.

**(e) Key caveat:** Our DiT uses softmax attention (not linear attention), so W is not explicitly maintained. The analogy is that the K plan tokens *function as* a finite FWP memory written by the planner and read by the DiT — but the write is nonlinear (resampler), not a literal outer product.

---

## SECTION 2 — META-LEARNING FOR FAST TEST-TIME ADAPTATION

### 2.1 Finn, Abbeel, Levine (2017) — MAML ✓

**Citation:** C. Finn, P. Abbeel, S. Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks," *ICML 2017*. arXiv: [1703.03400](https://arxiv.org/abs/1703.03400).

**(a) Core idea:** Meta-learn an initialization θ₀ such that a small number of gradient steps (e.g., 1–5) on a new task's support set produces a well-performing θ₀ + Δθ. The outer loop optimizes for post-adaptation performance, making θ₀ maximally "fast-adapting."

**(b) Exact mechanism:** Two loops. Inner: `θ' = θ₀ − α∇_θ L_task(θ₀)` (1–5 steps on task-specific data). Outer: `θ₀ ← θ₀ − β∇_θ₀ Σ_tasks L_task(θ')` — backprop through the inner update. The meta-gradient is a second-order gradient (through the inner-loop steps), though FOMAML approximates with first order.

**(c) What it enables:** A shared initialization that adapts in few steps at test time to new tasks. The model learns "how to be easily fine-tuned" rather than a fixed skill.

**(d) Map to our setup:**
- (i) ❌ K plan tokens are not a MAML inner loop — they are a forward-pass conditioning, not gradient updates.
- (ii) **RELEVANT to the "fast adaptation" framing**: If we freeze the DiT and train only the plan head (resampler + adapter), the plan head becomes a fast-adapting module. At test time, running a few gradient steps on plan head parameters given a new game = MAML-style fast adaptation of the conditioning module. This is a *future direction* for adapting to new games at test time.
- (iii) ❌ Not used in current architecture.

**(e) Key caveat:** MAML requires inner-loop gradients, which requires a differentiable loss signal at test time — we don't have ground-truth action labels at test time in the games setting. Would need a surrogate reward (e.g., game score via emulator, or GRPO ranking).

---

### 2.2 Nichol, Achiam, Schulman (2018) — Reptile ✓

**Citation:** A. Nichol, J. Achiam, J. Schulman, "On First-Order Meta-Learning Algorithms," OpenAI technical report. arXiv: [1803.02999](https://arxiv.org/abs/1803.02999). Mar 2018.

**(a) Core idea:** Reptile is a first-order meta-learning algorithm: for each task, take k steps of SGD; then move the meta-initialization toward the task-adapted parameters. No second-order gradients needed. Simpler and nearly as effective as FOMAML.

**(b) Exact mechanism:** `θ₀ ← θ₀ + ε(θ_task − θ₀)` where `θ_task` is the result of k SGD steps on the task. Equivalent to moving the initialization toward the manifold of task-specific optima.

**(c) What it enables:** Cheap meta-learning without second-order gradients; easier to implement in large-scale systems.

**(d) Map to our setup:**
- (ii) ✅ **RELEVANT to resampler/adapter as fast-adapting module**: If we treat different games as different "tasks" and want the resampler+adapter to rapidly specialize, Reptile-style training (k steps per game, then meta-update) is feasible without second-order gradients. More practical than MAML given our single-GPU constraint.
- (iii) Could be used with the emulator envs (§3B in AGENTS.md) as task environments.

**(e) Key caveat:** Both MAML and Reptile require multi-task data — at minimum one episode per game at meta-train time. Our current dataset is single-source (NitroGen YouTube shards, `game="other"` for many). Reptile becomes viable once we have labeled per-game data.

---

### 2.3 von Oswald et al. (2023) — Transformers Learn In-Context by Gradient Descent ✓

**Citation:** J. von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, M. Vladymyrov, "Transformers learn in-context by gradient descent," *ICML 2023*. arXiv: [2212.07677](https://arxiv.org/abs/2212.07677). Dec 2022.

**(a) Core idea:** In-context learning (ICL) in a transformer — providing a few (x,y) examples in the prompt and querying — is mechanistically equivalent to running implicit gradient descent on a linear regression problem. The attention heads implement the gradient descent update rule; in-context examples are the "training data" for an implicit fast model.

**(b) Exact mechanism:** For a single self-attention layer with a specific construction, the forward pass on a query x given in-context examples {(x_i, y_i)} computes `y_pred = W₀x + ΔW·x` where `ΔW` is the result of one step of gradient descent on in-context data under MSE loss. Linear attention is exactly this. Softmax attention performs a noisy/regularized version. The "weight update" ΔW is implicit in the attention activations, not stored.

**(c) What it enables:** Theoretical understanding of why ICL works and why more in-context examples improve performance. Bridges ICL and fast-weight/Bayesian meta-learning theory. Also implies that transformers performing ICL are implicitly performing rapid per-prompt optimization.

**(d) Map to our setup:**
- (i) ✅ **MOST DIRECTLY RELEVANT THEORETICAL PAPER**: Our K plan tokens, injected as keys+values into the DiT's cross-attention, are exactly an "in-context gradient descent" over the DiT's implicit regression problem. The plan tokens provide {(k_i, v_i)} pairs; the DiT's cross-attention query retrieves the "gradient descent update" that adapts the action-generation policy toward the plan. This makes the short path (K tokens → cross-attention) a **test-time implicit optimization**, not just static conditioning.
- (ii) The zero-init AdaLN path is a global modulation that does NOT have this ICL-as-gradient-descent property — it's a FiLM-style affine shift, not attention-based retrieval.
- (iii) Async: not addressed. But the theoretical framing suggests that a richer set of K tokens (more gradient descent steps) = more adaptation. This motivates K=8 vs K=4 ablations.

**(e) Key caveat:** The formal equivalence holds for *linear* attention and simple linear tasks. For softmax attention and complex nonlinear action generation (flow matching), the ICL-as-GD analogy is a theoretical parallel rather than an exact equivalence. However, the qualitative conclusion — "cross-attending to plan tokens is implicit test-time optimization" — is robustly supported.

---

## SECTION 3 — DUAL-PROCESS / HIERARCHICAL VLA SYSTEMS

### 3.1 NVIDIA GR00T N1 (2025) — System 1/2 DiT VLA ✓

**Citation:** NVIDIA et al. (Bjorck, Castañeda, Cherniadev, Da, Ding, Fan, Fang, Fox, …), "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots," 2025. arXiv: [2503.14734](https://arxiv.org/abs/2503.14734). Mar 2025.

**(a) Core idea:** Dual-system VLA. System 2 = a VLM (Eagle2 architecture) that processes vision + language instructions into semantic embeddings. System 1 = a flow-matching DiT action head that generates motor command chunks, cross-attending to System 2's output tokens. Both jointly trained end-to-end.

**(b) Exact mechanism:** VLM produces token sequence `z_s2` (vision-language tokens). DiT receives `z_s2` as keys+values in cross-attention layers. DiT also uses AdaLN conditioned on diffusion timestep embedding. Flow-matching objective: predict velocity field `v = a − ε`. Action chunking: predicts a chunk of T actions, uses receding horizon.

**(c) What it enables:** A generalizable robot foundation model that accepts free-form language instructions + vision and produces real-time motor control. NitroGen is explicitly adapted from this lineage (`README_UPSTREAM.md`).

**(d) Map to our setup:**
- (i) ✅ **DIRECT ARCHITECTURAL PARENT**: NitroGen = GR00T N1 System 1 (the DiT action head) with language/state encoders removed. Our project reconstructs GR00T's dual-system for games. The K plan tokens from our Qwen3.5 resampler play the role of `z_s2` in GR00T. The cross-attention injection mechanism in `nitrogen.py` (`prepare_input_embs`) mirrors GR00T's VLM→DiT wiring.
- (ii) GR00T uses AdaLN for timestep conditioning but cross-attention for VLM token conditioning. Our `plan_adaln=True` (EXP-048) explores adding AdaLN *also* for plan conditioning on top of cross-attention.
- (iii) GR00T runs System 2 and System 1 in a coupled, synchronous fashion. Our project explicitly explores async (VLM at planner cadence, DiT at action cadence) for latency hiding.

**(e) Key caveat:** GR00T trains end-to-end with System 1+2 from the start. We freeze the DiT (System 1) and only train the bridge (resampler+adapter). This risks misalignment between the planner's hidden-state manifold and the DiT's expected cross-attention conditioning distribution.

---

### 3.2 NVIDIA GR00T N1.5 (2025) ⚠️

**Citation:** NVIDIA, "NVIDIA GR00T N1.5," Technical blog/release, 2025. ⚠️ Not on arXiv as of 2026-06-29; see [developer.nvidia.com](https://developer.nvidia.com) or [huggingface.co/nvidia/GR00T-N1.5].

**(a) Core idea:** Incremental update to GR00T N1, improving VLM↔DiT alignment and generalization across robot embodiments. Key improvements include better cross-attention conditioning between the VLM token stream and the DiT, and expanded training data diversity.

**(d) Map to our setup:** GR00T N1.5's alignment improvements — *specifically how language tokens influence action generation more reliably* — are the most directly relevant technical details for our frozen-DiT plan injection problem. The key design question (8 plan tokens vs 256 image tokens, with plan tokens structurally "outvoted") is exactly the challenge N1.5 apparently addressed. ⚠️ **Unverified — read N1.5 technical report before citing mechanism details.**

---

### 3.3 Figure HELIX (2025) ⚠️

**Citation:** Figure AI, "Helix," technical description/blog, 2025. ⚠️ Not published as arXiv paper; see [figure.ai/helix](https://www.figure.ai/helix). Described as: "Helix controls the full loop: perception, movement, and reasoning, on board and in real time."

**(a) Core idea:** Dual-system humanoid robot controller. System 2 = a ~7B VLM running at ~7–9 Hz for semantic reasoning. System 1 = a ~80M visuomotor transformer running at ~200 Hz for reactive motor control. A **single continuous latent vector** from System 2 conditions System 1; the two run at different frequencies.

**(b) Exact mechanism:** System 2 (slow, semantic) emits a fixed-size continuous latent embedding z_s2 at ~7–9 Hz. System 1 (fast, reactive) consumes z_s2 as conditioning at every 200 Hz step. When S2 is computing the next z_s2 (latency ~100–130 ms), S1 keeps using the previous z_s2 — **async decoupling**. Both are trained jointly; the latent interface must carry sufficient state for S1 to stay coherent.

**(c) What it enables:** Real-time (<5 ms S1 latency) robot control with semantic grounding from a slow VLM; hides VLM latency behind async execution.

**(d) Map to our setup:**
- (i) ✅ **CLOSEST ANALOGUE to our cross-chunk design**: Helix's z_s2 = our K plan tokens held across multiple DiT calls. The K plan tokens are computed at VLM cadence (slow) and consumed by the DiT at action cadence (fast). MULTICHUNK_DESIGN.md's "cursor/phase" mechanism is implementing exactly this async decoupling.
- (ii) Helix uses a single continuous latent vector (not an AdaLN modulator explicitly, though likely FiLM-style injection). Our plan tokens go through cross-attention, not a single vector — higher bandwidth but more tokens.
- (iii) ✅ **DIRECTLY VALIDATES** the async dual-rate design in our cross-chunk path.

**(e) Key caveat:** Helix details are not publicly verified (no paper). The 7–9 Hz and 200 Hz numbers are from public descriptions. Their training procedure for the latent interface is not documented. ⚠️ **Unverified details — treat as directional validation, not methodological template.**

---

### 3.4 Physical Intelligence π₀ (2024) ✓

**Citation:** Physical Intelligence (Black, Brown, Driess, Escontrela, Finn, Florensa, Florence, He, Herrmann, Holt, Lee, Liao, Loquercio, Luo, Memmel, Niekum, Paxton, Pertsch, Reuss, Shi, Tian, Wulfe, Xiao, Yang, Yoneda, Burchfiel, Abbeel, Berg, Levine, Zeng), "π₀: A Vision-Language-Action Flow Model for General Robot Control," 2024. arXiv: [2410.24164](https://arxiv.org/abs/2410.24164). Oct 2024.

**(a) Core idea:** A VLA built on a pretrained VLM (PaliGemma 3B) with an attached "action expert" that uses flow matching to generate continuous action chunks. Language instructions condition both the VLM stream and, via cross-attention, the flow-matching action expert.

**(b) Exact mechanism:** PaliGemma encodes vision + language. A separate action expert transformer cross-attends to PaliGemma's output tokens and generates a T=50 action chunk via flow matching. The action expert uses full attention over its own action tokens + cross-attention to VLM tokens. No AdaLN in the action expert — conditioning is purely cross-attention.

**(c) What it enables:** Generalist manipulation across diverse robot embodiments; pre-training at scale on multi-robot data, then fine-tuning per task.

**(d) Map to our setup:**
- (i) ✅ NitroGen's DiT action head is the same lineage as π₀'s action expert (flow-matching, cross-attending to vision+language tokens). Our K plan tokens augment the cross-attention context exactly as π₀ uses VLM tokens as keys+values.
- (ii) π₀ uses **only cross-attention** for conditioning (no AdaLN for language/plan), validating our default cross-attention-only setup (plan_adaln=False).
- (iii) π₀ is synchronous (no async); the action chunk temporal offset serves as a buffer.

**(e) Key caveat:** π₀ is trained end-to-end (PaliGemma + action expert jointly). We freeze the DiT and train only the resampler+adapter. π₀'s action expert can adjust its cross-attention keys/queries during training to align with PaliGemma; ours cannot (frozen DiT). This is our core alignment challenge.

---

### 3.5 Physical Intelligence π₀.₅ (2025) ✓

**Citation:** Physical Intelligence, "π₀.₅: a Vision-Language-Action Model with Open-World Generalization," 2025. arXiv: [2504.16054](https://arxiv.org/abs/2504.16054). Apr 2025.

**(a) Core idea:** Extends π₀ with open-world instruction following by adding a high-level semantic reasoning step. A VLM-based "task planner" decomposes complex instructions into primitive skill sequences; a low-level π₀ policy executes each skill.

**(b) Exact mechanism:** An explicit two-stage hierarchy: (1) VLM planner emits a discrete subtask label or short language command; (2) π₀ conditioned on that command generates actions. The interface is *discrete language tokens*, not a continuous latent.

**(d) Map to our setup:**
- This is the **LONG PATH / SLOW PATH** analogue: planner → discrete language → policy. Our design deliberately uses a **continuous latent (short path)** instead of discrete language to avoid lossy serialization. π₀.₅ validates the concept of task decomposition, but their discrete interface is what we replace with K continuous plan tokens.

**(e) Key caveat:** The discrete interface (language) is interpretable but lossy; our K=8 continuous tokens are higher bandwidth but opaque. This is the core design trade-off.

---

### 3.6 Hi Robot (Physical Intelligence, 2025) ✓

**Citation:** Physical Intelligence (Shi et al.), "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models," 2025. arXiv: [2502.19417](https://arxiv.org/abs/2502.19417). Feb 2025.

**(a) Core idea:** Two-level hierarchy: a high-level VLM (System 2) receives complex natural-language instructions + real-time visual feedback and generates simpler *language subcommands* ("move arm forward"). A low-level VLA (π₀, System 1) executes those subcommands. "Thinking before acting" with language as the interface.

**(b) Exact mechanism:** System 2 produces a short natural-language string at each decision step; System 1 conditions on that string via its existing language-conditioned cross-attention. The bridge is **natural language** — interpretable, composable, but serialized through tokenization. System 2 runs at ~1 Hz; System 1 at ~10–50 Hz.

**(d) Map to our setup:**
- (i) ❌ Language interface = discrete tokens; our plan tokens = continuous latent. Hi Robot's bridge has lower bandwidth (serialized text) vs our K=8 dense vectors.
- (ii) ✅ The *training data structure* (instruction → decomposed subcommand → actions) is **directly applicable to our Stage-2 transcript training**: transcripts provide natural language plans; the streamer's actions provide the execution targets.
- (iii) ✅ Their async dual-rate (S2 at ~1 Hz, S1 faster) validates our planner-cadence design.

**(e) Key caveat:** Hi Robot's subcommands are interpretable and can be corrected by humans in language — an important practical advantage we lose with continuous plan tokens.

---

### 3.7 RT-2 (Google, 2023) ✓

**Citation:** A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," 2023. arXiv: [2307.15818](https://arxiv.org/abs/2307.15818). Jul 2023.

**(a) Core idea:** Fine-tune a large VLM (PaLM-E/ViT backbone) to directly predict robot actions as text tokens (discretized actions appended to the autoregressive token sequence). Web pretraining transfers semantic understanding directly into the action space.

**(d) Map to our setup:** RT-2 represents the **LONG PATH** (autoregressive VLM → discrete action tokens). NitroGen/our system is the opposite: a flow-matching DiT (short path, continuous). RT-2's insight that VLM pretraining transfers semantic knowledge is why we use a frozen Qwen3.5 (its pretraining knowledge of games is the free System-2 supervision, per AGENTS.md §3C).

---

### 3.8 RT-H: Action Hierarchies Using Language (Google, 2024)

**Citation:** S. Belkhale, T. Ding, T. Xiao, P. Sermanet, Q. Vuong, J. Tompson, Y. Chebotar, D. Dwibedi, D. Sadigh, "RT-H: Action Hierarchies Using Language," 2024. arXiv: March 2024 (arXiv ID unconfirmed from live search; submitted v1 4 March 2024 per arXiv metadata).

**(a) Core idea:** Builds a two-level "language of actions": a high-level policy predicts *language motions* ("move arm forward") from task + vision; a low-level policy conditioned on those language motions predicts actual motor actions. Language motions are intermediate between task-level language and low-level actions.

**(b) Exact mechanism:** Both levels are trained jointly as two query heads of the same VLM (PaLI-X). The language-motion query (π_h) predicts a language string describing the low-level motion. The action query (π_l) takes {task, vision, language_motion} and predicts motor actions. Interface is language tokens (discrete), but automatically generated — not requiring human labeling.

**(d) Map to our setup:**
- ✅ RT-H's **auto-generated intermediate language** = our **Qwen3.5-generated plan text**. Their "language motions" are programmatically generated from the dataset trajectory; our Stage-2 plans come from streamer transcripts or VLM-generated objectives. Same concept, different source.
- The discrete language-motion interface = RT-H's long path; our K plan tokens = short path. RT-H validates the concept of an intermediate layer; our system replaces their discrete tokens with a continuous latent.

**(e) Key caveat:** RT-H is trained end-to-end (both heads jointly). Our DiT is frozen. RT-H also operates in the robotics domain where language motions can be precisely defined; in games, "language motions" are harder to define (directional + button actions).

---

### 3.9 LAPA — Latent Action Pretraining from Videos (2024) ✓

**Citation:** S. Ye, J. Heo, M. Kim, J.-w. Ha, "Latent Action Pretraining from Videos," 2024. arXiv: [2410.11758](https://arxiv.org/abs/2410.11758). Oct 2024.

**(a) Core idea:** Learn a latent action space from video (without ground-truth actions) via an inverse dynamics model. A VQ-VAE tokenizes transitions (frame_t, frame_{t+1}) into discrete latent actions; a language-conditioned policy then maps language + frame → latent action → actual motor command.

**(b) Exact mechanism:** Two-stage: (1) Train a discrete latent action codebook on video frame transitions (unsupervised, IDM-style). (2) Pretrain the policy to predict latent action codes from language + vision; fine-tune to map latent codes to real actions via a small decoder.

**(d) Map to our setup:**
- ✅ LAPA's learned latent action space is a semantic relative of our plan tokens. Both replace discrete language with learned continuous/discrete latents as the high-level↔low-level interface.
- More importantly: LAPA's **IDM-from-video** approach is directly related to our VPT-style data bootstrap (AGENTS.md §3C, LITERATURE.md §13). LAPA extracts *implicit actions* from video; we extract *explicit gamepad actions* via an IDM trained on emulator ground truth. Complementary strategies.

**(e) Key caveat:** LAPA's latent actions are derived from frame transitions (visual IDM), not from semantic planning. Our plan tokens come from a VLM reading game context — higher semantic level.

---

### 3.10 Genie — Generative Interactive Environments (DeepMind, 2024) ✓

**Citation:** J. Bruce, M. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, Y. Aytar, S. Bechtle, F. Behbahani, S. Chan, N. Heess, L. Gonzalez, S. Osindero, S. Sherif, J. Requeima, L. Sifre, Y. Tang, H. Sadeghi, R. Pascanu, P. Purushwalkam, M. Saremi, K. Kavukcuoglu, T. Schaul, "Genie: Generative Interactive Environments," 2024. arXiv: [2402.15391](https://arxiv.org/abs/2402.15391). Feb 2024.

**(a) Core idea:** Learns a world model from unlabeled video of 2D platformer games. A latent action space (8 or 16 discrete latent actions) is inferred from video; a spatiotemporal transformer predicts future frames conditioned on latent actions. Enables controllable video generation and interactive environments from video alone.

**(b) Exact mechanism:** Three components: (1) Video tokenizer (VQ-VAE); (2) Latent action model (infers discrete action ID from frame pairs via IDM); (3) Dynamics model (predicts next frame tokens given current tokens + action). Trained entirely on unlabeled game video — no ground-truth game actions needed.

**(d) Map to our setup:**
- ✅ **DIRECTLY RELEVANT to MULTICHUNK_DESIGN.md Regime R2**: Genie provides a world model for "predict frame_{i+1} from frame_i + chunk_i", exactly what we need for counterfactual multi-chunk supervision (R2) without a live env. Genie trained on NitroGen-style game footage could provide the virtual environment for offline multi-chunk rollouts.
- The Genie latent action space is related to our plan tokens conceptually (both are compact, learned representations of "what to do") — but Genie's are inferred from frame transitions (visual IDM), while ours come from a text-conditioned VLM.

**(e) Key caveat:** Genie requires training a full world model — significant infrastructure. For our immediate needs, Regime R0 (real consecutive chunks) is free and is the recommended starting point per MULTICHUNK_DESIGN.md.

---

## SECTION 4 — CONDITIONING MECHANISMS IN DIFFUSION/FLOW POLICIES

### 4.1 Classifier-Free Guidance (Ho & Salimans, 2022) ✓

**Citation:** J. Ho, T. Salimans, "Classifier-Free Diffusion Guidance," *NeurIPS 2021 Workshop on DGMs*. arXiv: [2207.12598](https://arxiv.org/abs/2207.12598). Jul 2022 (arXiv version).

**(a) Core idea:** Train a diffusion model jointly in conditional and unconditional modes (randomly drop the conditioning signal c → replace with ∅). At inference, extrapolate: `v_guided = v_uncond + s·(v_cond − v_uncond)`. Scale s > 1 amplifies the conditioning effect; s = 0 = unconditional.

**(b) Exact mechanism:** Single model with conditioning dropout (`p_uncond` probability per example → c replaced by ∅ or a learned null token). Inference: two forward passes (cond + uncond), then linear extrapolation in velocity space. No separate classifier needed.

**(c) What it enables:** Tunable conditioning strength at inference; exact null-invariance when s=0; strong sharpening of conditional distributions at s>1.

**(d) Map to our setup:**
- (i) ✅ **DIRECTLY IMPLEMENTED** in NitroGen (`get_action_with_cfg`) and our plan head (`plan_dropout=0.15`, `null_mode='masked'`). The null plan is the `∅` of CFG; real plan = conditional c. The masked-null mode (`null_mode='masked'`) ensures exact base-model behavior under null plan (no residual plan signal).
- (ii) The zero-init AdaLN path (plan_adaln=True) also uses CFG: null plan → null AdaLN offset (zero-init guarantees this is the identity map at init).
- (iii) ❌ Not directly relevant to async.

**(e) Key caveat:** CFG amplification (`s > 1`) requires fixing the plan-dropout (dead-dropout bug noted in EXPERIMENTS). Also, the two-pass inference cost doubles action generation latency — important for real-time game control at 60 fps.

---

### 4.2 DiT — Scalable Diffusion Models with Transformers (Peebles & Xie, 2023) ✓

**Citation:** W. Peebles, S. Xie, "Scalable Diffusion Models with Transformers," *ICCV 2023*. arXiv: [2212.09748](https://arxiv.org/abs/2212.09748). Dec 2022.

**(a) Core idea:** Replaces the U-Net in diffusion models with a Vision Transformer (ViT) backbone. Introduces AdaLN-zero as the primary conditioning mechanism: scale γ and shift β are predicted from the conditioning signal (timestep + class label), and the initial projection of the conditioning residual is zero-initialized, making the block an identity map at initialization.

**(b) Exact mechanism:** `AdaLN-zero`: Given conditioning c, predict (γ, β, α) = MLP(c). Apply `y = γ·LN(x) + β + α·x_attn`. The α (gating) projection is **zero-initialized**, so at init `y = 0·x_attn = 0` → identity. After training, α grows to pass conditioning signal. This ensures null-conditioning → exact base behavior at init.

**(c) What it enables:** Stable conditional training starting from identity (no sudden conditioning disruption); graceful conditioning injection without overwriting frozen activations.

**(d) Map to our setup:**
- (ii) ✅ **CRITICAL DESIGN PRINCIPLE**: `plan_adaln=True` in `PlannerConfig` (EXP-048) implements precisely this zero-init AdaLN-zero pattern: the plan-to-temb projection is zero-initialized (`torch.nn.init.zeros_`), guaranteeing null-plan → exact DiT base behavior at init. This is "add steering without overwriting" — the zero-init ensures that before any training, the plan has zero net effect on the DiT, and the plan signal grows only as the adapter learns.
- Cross-attention (the primary K plan-token injection) is a separate mechanism — DiT paper uses AdaLN for *timestep* conditioning and cross-attention for *class/context* conditioning. We use both: cross-attention for the 8 plan tokens, optional AdaLN for global plan steering.

**(e) Key caveat:** The DiT paper uses AdaLN-zero for timestep and class, not for a separately-injected VLM plan. Our plan_adaln path adds plan conditioning *on top of* the existing timestep AdaLN — the two conditioning signals share the same modulation pathway and can interfere if not carefully scaled.

---

### 4.3 FiLM — Feature-wise Linear Modulation (Perez et al., 2018) ✓

**Citation:** E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville, "FiLM: Visual Reasoning with a General Conditioning Layer," *AAAI 2018*. arXiv: [1709.07871](https://arxiv.org/abs/1709.07871). Sep 2017.

**(a) Core idea:** FiLM applies a feature-wise affine transformation to a neural network's intermediate activations, conditioned on an external signal: `FiLM(F_i) = γ_i · F_i + β_i`. The (γ, β) pairs are predicted by a separate conditioning network from the conditioning input (e.g., a question, a language instruction).

**(b) Exact mechanism:** A "FiLM generator" network G takes conditioning input c and predicts per-feature (γ, β) vectors. These are applied to each feature map F_i: `F̃_i = γ_i(c) · F_i + β_i(c)`. This is a global affine modulation — it scales and shifts every feature in a layer, allowing the conditioning to "reweight" which features matter.

**(c) What it enables:** Strong, simple, differentiable conditioning; the conditioning network G can be of any architecture; FiLM injection is modality-agnostic.

**(d) Map to our setup:**
- (ii) ✅ **plan_adaln=True IS FiLM** in the DiT context: the plan-to-temb projection predicts (scale, shift) for the DiT's AdaLN, which is exactly FiLM applied to the hidden activations. The zero-init initialization is the key extension beyond vanilla FiLM.
- (i) Cross-attention with K plan tokens is NOT FiLM — it's a query-based selective retrieval, not a global affine shift. The two mechanisms are complementary: cross-attention = plan-relevant parts are retrieved selectively; FiLM/AdaLN = global plan signal modulates all activations.
- The combination (cross-attention + AdaLN-zero) provides both local (token-selective) and global (layer-wide) plan conditioning — analogous to how GR00T N1 uses both cross-attention for VLM tokens and AdaLN for timestep.

**(e) Key caveat:** FiLM with learned (γ, β) can disrupt pretrained features if the conditioning signal is too strong (γ deviates far from 1, β far from 0). Zero-initialization is the safeguard. Adding L2 regularization on (γ−1, β) during training is an optional stabilizer.

---

### 4.4 Perceiver IO (Jaegle et al., 2021) ✓

**Citation:** A. Jaegle, S. Borgeaud, J.-B. Alayrac, C. Doersch, C. Ionescu, D. Ding, O. Hénaff, W. Czarnecki, A. Zisserman, J. Carreira, "Perceiver IO: A General Architecture for Structured Inputs & Outputs," *ICML 2022*. arXiv: [2107.14795](https://arxiv.org/abs/2107.14795). Jul 2021.

**(a) Core idea:** A general architecture that handles arbitrary-length inputs and outputs by decoupling them via a fixed-size latent bottleneck. A set of K learned latent queries cross-attend to the (arbitrary-length) input to produce K latent vectors; these are then mapped to arbitrary outputs via output queries.

**(b) Exact mechanism:** Encoder: K learned queries × input → K latent vectors via cross-attention (O(KL) instead of O(L²)). Processor: optional self-attention among K latents. Decoder: output queries × K latents → outputs. The K queries are fixed parameters, not input-dependent.

**(d) Map to our setup:**
- (i) ✅ **OUR PlanResampler IS PERCEIVER IO**: `PlanResampler` in `planner.py:53–100` implements exactly Perceiver IO's encoder: K=8 learned queries cross-attend to VLM hidden states H (of variable length L), producing K=8 plan tokens via cross-attention. The query self-attention option (`resampler_query_self_attn=True`) is the Perceiver IO "processor" (self-attention among latents). The adapter MLP (`PlanAdapter`) is the output decoder.
- This is the canonical way to compress a variable-length conditioning signal into K fixed-size tokens — the K tokens are a learned compression of the plan semantics.

**(e) Key caveat:** K=8 is a tight bottleneck for long transcript plans (Stage 2). The K queries must learn to cover the semantic axes of game control (direction × buttons × timing). The per-token contrastive loss (PlannerConfig.contrastive_mode='pertoken') encourages each query to specialize to a different axis, mitigating the bottleneck.

---

## SECTION 5 — SYNTHESIS: SHORT PATH vs LONG PATH

```
SHORT (IMPLICIT) PATH                    LONG (EXPLICIT) PATH
━━━━━━━━━━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━━━━━━━━
Planner → Resampler → K plan tokens      Planner → Language reasoning →
  → DiT cross-attention                    Symbolic decomposition →
  (Perceiver IO compression)               Discrete command tokens →
                                           Policy parser

Mechanism: ICL-as-GD (von Oswald 2023)  Mechanism: discrete token routing
           Fast weights (Schmidhuber)             (RT-2, RT-H, Hi Robot)
           Hopfield retrieval (Ramsauer)           
           FWP (Schlag 2021)            

Latency: ~5 ms (frozen DiT, 1 fwd pass) Latency: VLM autoregressive decode
Bandwidth: K×1024 = 8K floats            Bandwidth: ~10-100 tokens × 4 bytes  
Interpretable: ❌ (continuous latent)    Interpretable: ✅ (natural language)
Async-friendly: ✅ (tokens held const)  Async-friendly: ✅ (new command issued)
Test-time control: CFG scale s           Test-time control: prompt engineering

Current implementation:                 Future / dual-mode:
  nitrogen/planner.py                     Qwen3.5 generation mode
  K=8, null_mode='masked'                 Stage-2 transcript training
  plan_adaln=True (EXP-048)               Hi Robot / π₀.₅ / RT-H style
```

### Zero-Init AdaLN as "Additive Steering That Preserves Null-Invariance"

The DiT paper (Peebles & Xie, arXiv 2212.09748) introduced AdaLN-zero specifically to ensure that a new conditioning signal starts as an identity and gradually learns influence. Our EXP-048 `plan_adaln=True` implements:

```python
# nitrogen/planner.py (PlannerConfig.plan_adaln=True path)
plan_offset = self.plan_adaln_proj(plan_pool)  # zero-init → 0 at start
temb_conditioned = temb + plan_offset           # additive: null plan → temb unchanged
```

This satisfies the null-invariance requirement: when plan tokens = null → `plan_pool` = null → `plan_offset` ≈ 0 (zero-init) → DiT sees exactly its original timestep embedding → base model behavior preserved exactly. As training progresses, `plan_offset` grows nonzero only for non-null plans.

The FiLM paper (Perez et al., arXiv 1709.07871) established the theoretical basis for this style of additive feature-wise modulation, and the DiT paper operationalized it with zero-initialization for stable fine-tuning.

---

## SECTION 6 — CITATION INDEX (verified)

| Paper | arXiv ID | Venue | Status |
|---|---|---|---|
| Schmidhuber (1992) Fast Weight Memories | — | *Neural Computation* 4(1):131–139 | ⚠️ pre-arXiv |
| Ba et al. (2016) Fast Weights Attend to Recent Past | [1610.06258](https://arxiv.org/abs/1610.06258) | NeurIPS 2016 | ✓ |
| Ramsauer et al. (2020) Hopfield Networks is All You Need | [2008.02217](https://arxiv.org/abs/2008.02217) | ICLR 2021 | ✓ |
| Schlag, Irie, Schmidhuber (2021) Linear Transformers as FWPs | [2102.11174](https://arxiv.org/abs/2102.11174) | ICML 2021 | ✓ |
| Finn et al. (2017) MAML | [1703.03400](https://arxiv.org/abs/1703.03400) | ICML 2017 | ✓ |
| Nichol et al. (2018) Reptile | [1803.02999](https://arxiv.org/abs/1803.02999) | OpenAI Tech Report | ✓ |
| von Oswald et al. (2023) Transformers learn ICL via GD | [2212.07677](https://arxiv.org/abs/2212.07677) | ICML 2023 | ✓ |
| NVIDIA GR00T N1 (2025) | [2503.14734](https://arxiv.org/abs/2503.14734) | Technical Report | ✓ |
| NVIDIA GR00T N1.5 (2025) | — | Blog/Release | ⚠️ no arXiv |
| Figure HELIX (2025) | — | Blog at figure.ai/helix | ⚠️ no arXiv |
| Physical Intelligence π₀ (2024) | [2410.24164](https://arxiv.org/abs/2410.24164) | Technical Report | ✓ |
| Physical Intelligence π₀.₅ (2025) | [2504.16054](https://arxiv.org/abs/2504.16054) | Technical Report | ✓ |
| Hi Robot (2025) | [2502.19417](https://arxiv.org/abs/2502.19417) | Technical Report | ✓ |
| RT-2 (2023) | [2307.15818](https://arxiv.org/abs/2307.15818) | CoRL 2023 | ✓ |
| RT-H (2024) Belkhale et al. | arXiv Mar 2024 (ID not confirmed) | — | ⚠️ title/authors confirmed |
| LAPA Latent Action Pretraining (2024) | [2410.11758](https://arxiv.org/abs/2410.11758) | Technical Report | ✓ |
| Genie (DeepMind, 2024) | [2402.15391](https://arxiv.org/abs/2402.15391) | ICML 2024 | ✓ |
| Peebles & Xie DiT (2023) | [2212.09748](https://arxiv.org/abs/2212.09748) | ICCV 2023 | ✓ |
| Ho & Salimans CFG (2022) | [2207.12598](https://arxiv.org/abs/2207.12598) | NeurIPS 2021 WS | ✓ |
| Perez et al. FiLM (2018) | [1709.07871](https://arxiv.org/abs/1709.07871) | AAAI 2018 | ✓ |
| Jaegle et al. Perceiver IO (2022) | [2107.14795](https://arxiv.org/abs/2107.14795) | ICML 2022 | ✓ |
| Chi et al. Diffusion Policy (2023) | [2303.04137](https://arxiv.org/abs/2303.04137) | RSS 2023 | ✓ |
| OpenVLA (2024) | [2406.09246](https://arxiv.org/abs/2406.09246) | CoRL 2024 | ✓ |

---

## SECTION 7 — GAPS AND UNCERTAINTIES

1. **RT-H arXiv ID unconfirmed**: Found by title/authors in arXiv search ("v1 submitted 4 March 2024") but the specific ID (format 2403.XXXXX) was not recovered from live fetches. ⚠️ Verify before citing: search `arxiv.org` for "RT-H action hierarchies Belkhale 2024".

2. **GR00T N1.5 technical details**: This is a product release (not a peer-reviewed paper). The specific improvements to VLM↔DiT alignment are not publicly documented in a paper. ⚠️ Check NVIDIA developer blog or HuggingFace model card for mechanism details.

3. **Figure HELIX frequencies (7–9 Hz S2, ~200 Hz S1)**: These are from public press material, not a peer-reviewed paper. The training procedure, the exact latent size, and the async protocol are not publicly verified. ⚠️ Treat as directional validation, not cite as technical spec.

4. **Schmidhuber 1992 venue**: Neural Computation vol 4(1):131–139, 1992. An earlier 1991 version appeared at ICANN. Both predate arXiv. The exact title varies: "Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks" — verify against the Neural Computation journal if citing.

5. **von Oswald et al. 2023 ICML**: The formal ICL-as-GD equivalence is proved for *linear* attention / *linear* regression. The extension to softmax attention + flow-matching DiTs is theoretical analogy, not proven equivalence. State this carefully in any paper.

6. **LAPA vs Genie latent actions**: Both learn latent action spaces from video, but differently. LAPA (2410.11758) uses a discrete VQ codebook from IDM; Genie (2402.15391) uses a 2D discrete latent space inferred from temporal differences. Both are related to our IDM bootstrap strategy (AGENTS.md §3C) but are not directly cited by NitroGen.

---

**Summary for main agent:** All key papers verified. The fast-weight cluster (Schmidhuber 1992, Ba 2016, Ramsauer 2020, Schlag 2021) provides the theoretical language for the K=8 plan tokens as a fast-weight conditioning state written by the planner and read by the DiT. The ICL-as-GD paper (von Oswald 2023, arXiv 2212.07677) is the strongest theoretical support for cross-attention plan injection as implicit test-time optimization. The dual-system VLA papers (GR00T N1 arXiv 2503.14734, Helix, π₀ arXiv 2410.24164, Hi Robot arXiv 2502.19417) validate the async dual-rate design and the continuous-latent interface choice. DiT AdaLN-zero (arXiv 2212.09748) and FiLM (arXiv 1709.07871) provide the "add steering without overwriting" conditioning mechanism for `plan_adaln=True`. Three items have unconfirmed details: RT-H arXiv ID, GR00T N1.5 mechanism, and Figure Helix technical specs.
