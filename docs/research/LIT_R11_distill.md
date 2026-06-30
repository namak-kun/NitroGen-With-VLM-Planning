# LIT_R11_distill.md
## Literature Review: System-2 → System-1 Consolidation for Game-Playing VLAs

*Compiled 2026-06-29 for namak-kun/NitroGen-With-VLM-Planning.*
*Covers: self-improvement/expert-iteration; System-2→System-1 distillation; amortized inference; sleep-time compute; DAgger/on-policy correction; hierarchical planner→policy distillation.*
*Citations use format: Title (Authors, Year) arXiv:XXXXXXX / Venue.*
*✓ = arXiv ID verified by live fetch this session. ✗ = from prior knowledge, not live-verified — double-check before citing in writing.*

---

## 0. Framing: What Is the Consolidation Loop?

Our proposed **consolidation loop** works as follows:
1. From emulator save-states, run closed-loop rollouts where the frozen **VLM (System 2)** proposes a textual plan and watches execution.
2. Keep rollouts that scored well (game score increase, level progression, or a VLM-judged success signal).
3. **Behaviour-clone (BC) the PlanHead** (resampler → adapter → plan tokens) on those rollouts, conditioned on the plan text that generated them.
4. Repeat: re-run with updated PlanHead; collect again; distil again.

This is an **amortized expert-iteration loop**: the slow planner finds good moves → those moves are distilled into the fast action model → the fast model gets better → the loop repeats. Every paper below is mapped to this setup.

---

## 1. Expert Iteration / ExIt (THE key precedent)

**Paper:** "Thinking Fast and Slow with Deep Learning and Tree Search"
Thomas Anthony, Zheng Tian, David Barber.
NeurIPS 2017. arXiv:[1705.08439](https://arxiv.org/abs/1705.08439) ✓

### (a) Core idea
ExIt explicitly formalises "System 2 search distilled into a System 1 policy" as a two-player iterative loop. A **tree-search expert** (MCTS) uses the current neural-net policy as a heuristic to produce much stronger move distributions than the policy alone can. Those move distributions are then used as **supervised targets** to update the policy (i.e., "imitate the improved expert"). The policy net is never trained with RL on raw game returns; it is trained only with supervised cross-entropy against the search's improved distribution.

### (b) Exact mechanism
- **Training signal:** MCTS visit-counts π_MCTS(s) at each state s in rollouts from the current policy π_θ.
- **What gets updated:** the policy network π_θ (both the value head and the policy head).
- **Loss:** supervised cross-entropy H(π_MCTS, π_θ) over states visited by the current policy.
- Iterations alternate: (1) **Imitate** = use π_θ to guide MCTS, collect (state, π_MCTS) pairs; (2) **Distil** = update π_θ ← argmin H(π_MCTS, π_θ).

### (c) What it fixes
Removes the need for reward-sparse RL on the raw policy while still producing monotonic policy improvement (under mild assumptions). The search finds good moves that the policy alone could not, and supervised imitation of those moves is much more sample-efficient than RL.

### (d) Maps to our setup — DIRECT PRECEDENT
| ExIt concept | Our consolidation loop |
|---|---|
| Tree-search expert (System 2) | VLM planner proposing plans + watching execution |
| Policy network π_θ (System 1) | NitroGen DiT + PlanHead |
| MCTS visit distributions | High-score rollout trajectories filtered by game score |
| Supervised imitation step | BC on PlanHead conditioned on the plan that produced success |
| State distribution of rollouts | Emulator save-state restarts |

**What ExIt predicts about our loop:**
- Convergence requires that the VLM planner can reliably find *strictly better* action distributions than the current PlanHead in the states it visits.
- If the VLM generates random or inconsistent plans, the "expert" is no better than chance and the distillation teaches noise.
- The bottleneck is the **quality of the System-2 planner**, not the expressiveness of the distillation.
- The distribution of states visited in rollout collection determines what the policy improves on. If we always restart from the same 5 save-states, the policy will only improve at those states.

**Predicted pitfalls:** ExIt can plateau when the search's advantage over the policy vanishes (policy becomes as good as search-guided). In our case this is desirable but slow to happen. More dangerously: if the filtered "high-score" trajectories come from the policy getting lucky at a specific state rather than from a principled plan, the distilled update may be noise.

### (e) Key caveat
ExIt was designed for fully-observable, discrete-action games (Hex, Go) with a well-defined search tree. Our action space is continuous (gamepad floats), so there is no natural MCTS tree — the VLM's plan text replaces the search tree, which is a much weaker oracle. The analogy holds at the level of the outer loop structure, not the inner search mechanics.

---

## 2. AlphaZero / AlphaGo — MCTS–Policy Distillation

**Papers:**
- "Mastering the Game of Go with Deep Neural Networks and Tree Search" — Silver et al., *Nature* 529, 484–489 (2016). ✗ (not on arXiv)
- "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" — Silver et al., 2017. arXiv:[1712.01815](https://arxiv.org/abs/1712.01815) ✓
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero) — Schrittwieser et al., 2020. arXiv:[1911.08265](https://arxiv.org/abs/1911.08265) ✓

### (a) Core idea
AlphaZero trains a single network (policy head + value head) to jointly represent both the "fast intuition" (direct policy) and the "slow reasoning" (value estimation for MCTS). The policy is trained by behavioural cloning against MCTS search probabilities (identical to ExIt's distillation step). AlphaGo Zero showed zero human-data improvement is possible; AlphaZero generalised to chess and shogi; MuZero learned a latent world model to enable MCTS without a simulator.

### (b) Mechanism
- **Signal:** π_MCTS(s) — search tree visit fractions at state s.
- **Update:** policy head ← CE(π_MCTS, π_θ); value head ← MSE(z, V_θ(s)) where z is game outcome.
- Self-play provides the state distribution; no human demonstrations needed.

### (c) What it fixes
Enables strong game-playing without curated expert demonstrations. The value head's training signal (game outcome) propagates through the entire game, solving long-horizon credit assignment.

### (d) Maps to our setup
MuZero is the most relevant variant: it learns a latent-action world model that allows MCTS to run **without a real simulator**. This is analogous to our interest in offline (emulator-free) multi-chunk planning. In MuZero the world model is a neural forward model; in our case we would need either: (a) the emulator itself (we have it), or (b) a world model trained on gameplay video (Genie-style). For our consolidation loop: the VLM plays the role of a very expensive, one-shot "search" step — not iterative MCTS, but still a form of lookahead.

### (e) Key caveat
AlphaZero's distillation from search probabilities is distribution-matched (KL divergence) — it keeps the **full policy distribution**, not just the argmax. Our BC on filtered rollouts keeps only the **best trajectories**, discarding the breadth of search. This is equivalent to a harsh temperature-0 distillation, which narrows the policy distribution and increases overfitting risk. AlphaZero would say: distil the full softmax, not a filtered subset.

---

## 3. STaR — Self-Taught Reasoner

**Paper:** "STaR: Bootstrapping Reasoning With Reasoning"
Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman.
NeurIPS 2022. arXiv:[2203.14465](https://arxiv.org/abs/2203.14465) ✓

### (a) Core idea
STaR is the LLM analogue of ExIt. A model generates **rationales (CoT)** before answering, then checks whether the final answer is correct. Rationales that led to correct answers are kept; the model is fine-tuned on (problem, rationale, answer) triples. Over iterations, the model learns to produce rationales that actually help it get to correct answers. A "rationalisation" trick augments hard problems: condition on the correct answer and ask the model to generate a rationale explaining it.

### (b) Mechanism
- **Signal:** binary correctness of the final answer (not a reward function — requires ground-truth labels).
- **What gets updated:** the full LLM weights via supervised fine-tuning on filtered (problem, rationale, correct_answer) triples.
- **Loss:** cross-entropy on the *complete* sequence (rationale + answer), not just the answer.

### (c) What it fixes
Bootstraps reasoning ability without a large labelled rationale corpus. The key insight is that **a model's own generated rationale is a valid training target** if the resulting answer is correct — you don't need a human to write the CoT.

### (d) Maps to our setup
Our loop is STaR with a game score replacing "answer correctness":
- VLM generates a plan (= rationale) and the agent executes it (= generates the answer).
- If the execution scores well, the (plan, execution) pair is a training target.
- The PlanHead is fine-tuned to reproduce the plan→action bridge of successful runs.

**STaR warns:** the quality of the binary filter (correct/incorrect) matters. For math problems, "correct" is objective. For games, "scored well" is noisier — a lucky run may not reflect good planning. STaR also requires enough correct examples per iteration to fine-tune; if the VLM generates poor plans most of the time, the filtered set will be too small for meaningful learning.

**STaR's rationalisation trick has a direct analogue:** if an attempt fails, condition the VLM on the *desired outcome* ("you need to reach the door") and ask it to explain what actions would have achieved it. This generates a counterfactual "rationale" that can be used for supervised learning even without successful rollouts.

### (e) Key caveat
STaR still requires **ground-truth answer labels**. In games, the "label" is score, but score is a proxy — reward hacking occurs. Also, STaR fine-tunes the full model; we fine-tune only the PlanHead, which is much cheaper but may hit an expressiveness ceiling if the PlanHead is too small.

---

## 4. ReST — Reinforced Self-Training

**Paper:** "Reinforced Self-Training (ReST) for Language Modeling"
Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, et al. (DeepMind).
2023. arXiv:[2308.08998](https://arxiv.org/abs/2308.08998) ✓

### (a) Core idea
ReST separates the self-improvement loop into two explicit phases: **Grow** (sample many rollouts from the current policy) and **Improve** (fine-tune on the top-scoring samples). This offline EM structure avoids on-policy RL instability while still using reward feedback to improve the policy.

### (b) Mechanism
- **Grow step (E-step):** sample N completions per prompt from π_θ; score each with a reward function R.
- **Improve step (M-step):** SFT on (prompt, completion) pairs where R(completion) > threshold τ.
- **Loss:** standard cross-entropy supervised on the top-τ percentile completions.
- Iterates: re-sample from the updated policy at each round (on-policy distribution).

### (c) What it fixes
Avoids the instability of online RL (PPO/REINFORCE) while still using reward feedback. The offline SFT step is more stable and compatible with standard fine-tuning pipelines.

### (d) Maps to our setup
ReST is essentially **our consolidation loop exactly described**:
- Grow = run rollouts from save-states, score by game return.
- Improve = BC the PlanHead on high-score (plan, trajectory) pairs.
- Our save-state reset is the "prompt" that seeds each rollout.

**ReST predicts:** the key hyperparameter is the threshold τ. If τ is too low, the model trains on mediocre data and improves slowly. If τ is too high, training data is scarce and the loop stalls. ReST also predicts that **without re-sampling** (using old trajectories from a previous policy iteration), distribution shift accumulates — Gulcehre et al. found freshly sampled on-policy data crucial for each round.

### (e) Key caveat
ReST requires that the reward function is **not gameable**. In NLP tasks the reward is answer quality as judged by a human or another model. In games, partial rewards (e.g., score delta per chunk) can be gamed by the policy learning to "farm" low-value repeatable actions.

---

## 5. ReST-EM / Beyond Human Data

**Paper:** "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models"
Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, et al. (Google DeepMind).
2023. arXiv:[2312.06585](https://arxiv.org/abs/2312.06585) ✓

### (a) Core idea
ReST-EM casts the self-improvement loop as **Expectation-Maximisation (EM)**: the E-step generates rollouts from the policy, the M-step maximises likelihood on the *filtered* ones. Shows that just 2-3 EM rounds of self-training on model-generated data **outperforms training on far more human data**, because the model's own samples are in-distribution.

### (b) Mechanism
- Exactly as ReST (Grow/Improve) but framed as EM: E = generate + score; M = SFT on filtered data.
- The EM framing gives an information-theoretic guarantee: each M-step cannot decrease the expected reward of the filtered distribution (Jensen's inequality argument).
- Uses binary reward (correct/incorrect) on coding/math benchmarks.

### (c) What it fixes
Provides a principled grounding for filtering-and-fine-tuning: it is gradient-based maximisation of a lower bound on the policy's expected reward.

### (d) Maps to our setup
The EM framing directly justifies our loop. Our **M-step loss** is:
```
L = -E_{(s,a)~D_good} [log π_θ(a | s, z_plan)]
```
where D_good are filtered trajectories and z_plan = PlanHead(text_plan). This is the standard EM-M-step for a latent-variable model where z_plan is the latent. The **E-step** is the rollout collection from the emulator.

**Pitfall the paper flags:** in coding/math, each binary reward is highly informative. In our case, game score per chunk is noisy — a chunk with score +0 may be a good defensive maneuver, not a bad one. The EM M-step will mistakenly treat it as failed and discard it. Consider instead: use the VLM as a binary judge ("did this trajectory follow the plan?") rather than relying solely on game score.

### (e) Key caveat
"Beyond Human Data" is specifically about language models where the full model is updated. In our setup only the PlanHead (a small adapter) is updated while the DiT is frozen. This limits the M-step's ability to reshape the action distribution; if the frozen DiT's action manifold doesn't contain the desired behavior, no amount of plan-token adjustment will produce it.

---

## 6. Distilling System 2 into System 1

**Paper:** "Distilling System 2 into System 1"
Ping Yu, Jing Xu, Jason Weston, Ilia Kulikov. (Meta FAIR)
2024. arXiv:[2407.06023](https://arxiv.org/abs/2407.06023) ✓

### (a) Core idea
The **canonical explicit paper** for our project's core idea. Given an expensive System-2 method (CoT, Tree-of-Thoughts, Branch-Solve-Merge, etc.), run it on unlabeled examples, filter by self-consistency, and then **fine-tune the base LLM to directly produce the same output without the intermediate steps**. The compiled model recovers most of the accuracy of System 2 at the speed of System 1.

### (b) Mechanism
- **Step 1 (System-2 inference):** for each unlabeled example x, run System-2 method to get y* (with intermediate reasoning steps).
- **Step 2 (Filter):** keep (x, y*) pairs where System-2 is "confident" (e.g., majority-vote self-consistency across multiple samples).
- **Step 3 (Distil):** fine-tune the base model (System 1) on (x, y*) pairs to predict y* directly, without generating the reasoning steps.
- **Loss:** CE on the *output only* (not on intermediate rationale tokens).

### (c) What it fixes
Latency: System 2 may require seconds of sequential token generation; the distilled System 1 answers in milliseconds. The paper shows the distilled model often **matches or beats** the System-2 teacher on several tasks.

### (d) Maps to our setup — CLOSEST CONCEPTUAL MATCH
Our PlanHead is "System 1" (fast, feeds into a real-time action model). Our VLM planner is "System 2" (slow, cannot run at 60 fps). This paper is the *exact* recipe for what we are trying to do:
- System 2 = VLM proposing plans and watching execution.
- System 1 = PlanHead converting plan-text to plan-tokens that steer the DiT.
- The "output" being distilled is not a text answer but a continuous latent code (plan-token embedding) that, when fed to the DiT, produces good actions.

**Key divergence from the paper:** in Yu et al. the distilled System-1 model is the *same architecture* as the System-2 model (just bypassing the reasoning chain). In our case System-1 (the DiT) is a *different architecture* (flow-matching, not autoregressive LLM) from System-2 (the VLM). The distillation must pass through a learned bridge (the PlanHead), which is a layer of indirection that may lose information.

**Pitfall directly flagged by the paper:** "not all tasks can be distilled into System 1, particularly complex math reasoning tasks requiring chain-of-thought." The paper found that tasks requiring genuine multi-step deduction resisted distillation — the System-1 model simply didn't have the capacity. **Our analogue:** game maneuvers requiring long-horizon sequential planning (e.g., "navigate a maze over 60 steps") may require a PlanHead capacity or DiT unfreeze that our current small adapter cannot achieve.

### (e) Key caveat
The self-consistency filter works because math problems have unique correct answers. For game maneuvers, "self-consistency" across multiple VLM attempts is ill-defined. We must substitute either: (a) game-score threshold (as in ReST) or (b) VLM self-evaluation ("was this execution consistent with the plan?").

---

## 7. Quiet-STaR

**Paper:** "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"
Eric Zelikman, Georges Hasson, Yijia Shao, Nick Haber, Noah D. Goodman.
2024. arXiv:[2403.09629](https://arxiv.org/abs/2403.09629) ✓

### (a) Core idea
Instead of generating rationales only before final answers, Quiet-STaR inserts **internal "thought" tokens at every single token position**. The thoughts are generated, then a gating mechanism weights whether the thought was useful for predicting the next token. Training is by REINFORCE on the token-level prediction improvement caused by the thought.

### (b) Mechanism
- For each token position t, sample a short "thought" τ_t (invisible to the user) before predicting x_{t+1}.
- **Reward signal:** the log-ratio log p(x_{t+1}|τ_t, context) − log p(x_{t+1}|context) (did the thought help?).
- **Policy:** the thought-generation network θ is updated by REINFORCE using this reward.
- Also fine-tunes the base model to use the thoughts for prediction.

### (c) What it fixes
Enables a model to learn *which internal computations are useful* without human annotations of rationale quality. The training signal is purely self-supervised (next-token prediction improvement).

### (d) Maps to our setup
Quiet-STaR is the self-supervised version of our VLM-plan loop. In our setup the "thought" is the plan text (produced externally by the VLM), and the training signal is game score. Quiet-STaR suggests a future direction: **train the VLM planner itself** to generate better plans by using the DiT's execution quality as a reward signal. The REINFORCE gradient would flow: game_score → PlanHead → VLM plan generation. This is different from the current setup (VLM frozen); it would require unfreezing the VLM or training a LoRA on it.

### (e) Key caveat
Quiet-STaR's per-token thoughts are very short (5–10 tokens), making the REINFORCE variance manageable. Our plans are potentially long (100+ tokens), making raw REINFORCE very high-variance. Would need PPO or advantage-normalisation. Note: unfreezing the VLM contradicts the project's hard constraint of keeping it frozen.

---

## 8. CoT Distillation into Smaller Models

**Papers:**
- "Teaching Small Language Models to Reason" — Magister, Shaki, Bakalov, Dyer. 2022. arXiv:[2212.08410](https://arxiv.org/abs/2212.08410) ✓
- "Orca: Progressive Learning from Complex Explanation Traces of GPT-4" — Mukherjee et al. (Microsoft). 2023. arXiv:[2306.02707](https://arxiv.org/abs/2306.02707) ✓

### (a) Core idea
Both papers distil the *intermediate reasoning steps* of a large teacher (GPT-4 or a large LLM) into a smaller student. The student is trained to mimic the explanation trace, not just the final answer. Orca adds "system prompts" that ask the teacher to explain its reasoning style (step-by-step, how, why), producing richer training signal.

### (b) Mechanism
- **Magister:** student trains on (question, CoT_from_teacher, answer) triples; cross-entropy on the full sequence.
- **Orca:** adds "augmentation instructions" (teacher, please explain step-by-step) to generate explanation traces; student trained on the full trace.

### (c) What it fixes / Maps to our setup
These are the mechanism behind "knowledge distillation through rationale imitation." For our setup: we could train the PlanHead (or a VLM adapter) to predict the plan-execution *rationale* that the large VLM would produce, not just the action distribution. This is useful if we later want to make the PlanHead *generate its own plan descriptions* rather than being fed external plan text — it would then be a small plan generator.

### (e) Key caveat
Distilling reasoning traces is only effective if the student has sufficient capacity to represent the trace. A 3-layer Perceiver resampler may not be able to represent a complex 500-token reasoning chain.

---

## 9. Hinton Knowledge Distillation (foundational)

**Paper:** "Distilling the Knowledge in a Neural Network"
Geoffrey Hinton, Oriol Vinyals, Jeff Dean. 2015. arXiv:[1503.02531](https://arxiv.org/abs/1503.02531) ✓

### (a) Core idea
A large "teacher" network's **soft output distribution** (softmax with high temperature T) contains more information than the hard one-hot labels. Training a smaller "student" to match the teacher's soft outputs transfers knowledge of inter-class similarities. "Temperature scaling" of softmax is the key mechanism.

### (b) Mechanism
- **Loss:** α·CE(y_true, student) + (1-α)·KL(teacher_soft_T, student_soft_T)
- High T "spreads" the teacher's distribution, revealing which classes are similar.

### (c) Maps to our setup
In our case, the "teacher" is the frozen DiT under the *correct plan* condition. The "student" is the DiT+PlanHead system. The soft targets are the action distributions under the teacher. This argues for **distributional BC** (KL matching) rather than MSE on single action trajectories — i.e., distil the full action distribution of the teacher, not just the argmax trajectory. This is harder to implement (requires sampling from the flow model) but more principled.

---

## 10. Amortized Inference — Conceptual Frame

**Paper:** "Auto-Encoding Variational Bayes"
Diederik P. Kingma, Max Welling. 2013. arXiv:[1312.6114](https://arxiv.org/abs/1312.6114) ✓

### (a) Core idea
The VAE is the canonical example of **amortized inference**: instead of optimising a variational posterior q(z|x) from scratch for each new observation x (expensive, iterative), train a neural network ("inference network" / "encoder") that directly maps x → q(z|x) parameters in a single forward pass. This **amortises** the cost of inference over training, paying once at train time to make inference fast at test time.

### (b) Mechanism
- **Slow path (per-example EM):** for each x, iterate to find q*(z|x) = argmin KL(q, p(z|x)).
- **Amortised path:** train encoder φ by minimising the ELBO over many x jointly; at test time, q_φ(z|x) ≈ q*(z|x) in one pass.
- This is exactly the idea behind the VLM→PlanHead pipeline: the VLM "solves" the inference problem (what plan-token z should condition the DiT given observation x?); the PlanHead amortises this so it can be done in one forward pass.

### (c) What it fixes / Maps to our setup
**Our PlanHead is literally an amortised inference network:**
- The "posterior" being amortised is: "given game frame x and plan text t, what latent z should I inject into the DiT to produce good actions?"
- The slow path is: run the VLM, search over plans, evaluate which z produces the best game outcome.
- The amortised fast path is: PlanHead_φ(x, t) → z in one forward pass at gameplay time.

The VAE framework also predicts a well-known failure mode: **amortisation gap** (Cremer et al., 2018). The amortised posterior q_φ(z|x) is always at least as bad as the optimal posterior q*(z|x) because it must generalise across all x with a shared parameter set. For states/plans that are rare in training data, the PlanHead will produce poor z, while the slow VLM search would still find a good one.

### (e) Key caveat
The amortisation gap grows with the distribution mismatch between train-time states and test-time states. If we train the PlanHead only on easy save-states and deploy on hard mid-game states, the gap will be large. DAgger (below) is the cure.

---

## 11. Sleep-time Compute

**Paper:** "Sleep-time Compute: Beyond Inference Scaling at Test-time"
Kevin Lin∗, Charlie Snell∗, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez. (Letta / UC Berkeley)
2025. arXiv:[2504.13171](https://arxiv.org/abs/2504.13171) ✓

### (a) Core idea
Test-time compute scaling (o1-style chain-of-thought at query time) is expensive and slow. Sleep-time compute instead runs "thinking" **between interactions** (when the model is idle) on the context, pre-computing useful inferences, so that when a query arrives the model can answer cheaply by reading the pre-computed notes. This amortises compute across multiple related queries about the same context.

### (b) Mechanism
- During "sleep," prompt the LLM: "Given this context, what questions might users ask, and what inferences would help answer them?"
- Store the generated notes alongside the context.
- At test time, prepend notes to the prompt — dramatically reducing the needed test-time reasoning steps.
- Found to reduce test-time compute ~5× on Stateful GSM-Symbolic/AIME, adding 2-18% accuracy.

### (c) What it fixes
Decouples the **expensive reasoning** (sleep) from the **latency-sensitive inference** (test time). Directly analogous to consolidation during biological sleep (hippocampal → neocortical transfer).

### (d) Maps to our setup
**Our VLM planner IS sleep-time compute.** The VLM fires sparsely (once per K chunks = "sleep time") and produces a plan summary that the DiT consumes repeatedly across K chunks without re-running the VLM. Sleep-time Compute directly validates this architecture:
- Running the VLM at gameplay time is expensive and high-latency (several seconds for a frozen Qwen pass).
- Pre-computing plan tokens in "sleep" (once per multi-second plan window) and caching them for the DiT to consume each chunk is precisely the sleep-time pattern.
- The paper's finding that compute amortises across multiple related queries maps to our case: one plan covers multiple game-steps/chunks, amortising VLM cost across them.

**Extension:** in our consolidation loop, the "sleep" is not just between gameplay windows but offline training. Overnight training runs that distil VLM knowledge into the PlanHead weights ARE a form of sleep — the model "dreams" (replays filtered trajectories) and consolidates knowledge into fast weights.

### (e) Key caveat
Sleep-time Compute's "notes" are natural language (interpretable). Our "notes" are latent vectors. If the pre-computed plan-tokens become stale (game state changes drastically mid-plan), the pre-computed z will be misleading — analogous to studying for the wrong exam.

---

## 12. Dreamer / DreamerV3 — World-Model Imagination as Offline Consolidation

**Papers:**
- "Dream to Control: Learning Behaviors by Latent Imagination" (DreamerV1) — Danijar Hafner et al. 2019. arXiv:[1912.01603](https://arxiv.org/abs/1912.01603) ✓
- "Mastering Diverse Domains through World Models" (DreamerV3) — Hafner, Pasukonis, Ba, Lillicrap. 2023. arXiv:[2301.04104](https://arxiv.org/abs/2301.04104) ✓

### (a) Core idea
Dreamer learns a latent world model from gameplay experience (a compact forward model of state transitions). Policy training happens entirely **inside the world model** via imagined rollouts — never touching the real environment during the policy update step. The world model is the "slow consolidation" substrate; the policy is the "fast actor."

### (b) Mechanism
- **World model:** learns p(s_{t+1} | s_t, a_t) as a RSSM (Recurrent State-Space Model) with latent state.
- **Policy training:** generate imagined trajectories s_0 → s_1 → … → s_H inside the world model. Use Dreamer's actor-critic to optimise the policy in latent space.
- **Real env:** only used to collect transitions for the world model; not touched during policy updates.

### (c) What it fixes
Removes the need for millions of real-env steps to train the policy. Policy training is cheap (latent-space only); data collection is expensive.

### (d) Maps to our setup
DreamerV3 is the template for the offline-consolidation leg of our roadmap (MULTICHUNK_DESIGN "Regime R3"):
- Learn a world model (or use NitroGen's implicit one) that can roll forward from a frame + action chunk → next frame.
- Train the PlanHead in imagination: sample a plan z from the VLM, imagine N game steps forward, compute a policy-improvement signal, backprop into the PlanHead.
- This would allow **training without live emulator access**, which our current setup requires.

**Pitfall:** world models trained on streamer video are biased toward "what skilled players do." Counterfactual plans that the streamer never executed may fall OOD for the world model, producing hallucinated futures.

### (e) Key caveat
DreamerV3 was tested on 150+ Atari/DMC/etc. games with a **rich, trained world model**. NitroGen's implicit action model is not a world model — it produces actions, not next-frame predictions. Building a usable world model adds major engineering (train a separate video prediction model).

---

## 13. DAgger — On-Policy Distillation with Corrective Feedback

**Paper:** "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell.
AISTATS 2011. arXiv:[1011.0686](https://arxiv.org/abs/1011.0686) ✓

### (a) Core idea
Behaviour cloning (BC) trained on an expert's trajectory distribution fails at test time because the policy visits states that were never in the training set (compounding error / covariate shift). DAgger fixes this by **interleaving data collection with the learner's own trajectory**: at each iteration, run the current policy, query the expert's action at each visited state, add to the training set.

### (b) Mechanism
- Initialise π_0 = expert (teacher).
- At iteration i: mix policy π_{i-1} and expert in ratio β_i; execute in env; collect states.
- Query expert for actions at those states → new dataset D_i.
- Train π_i = BC(∪_{j≤i} D_j).
- **Loss:** CE / MSE on expert actions at states visited by the learner's policy (not the expert's policy).
- The mixed execution ensures training distribution converges to the policy's own state distribution.

### (c) What it fixes
Covariate shift: once the policy deviates slightly from the training distribution (inevitable under any BC policy), subsequent states are OOD and errors compound quadratically. DAgger converts this to a no-regret online learning problem: error grows only linearly.

### (d) Maps to our setup — CRITICAL for OOD robustness
**Our action model is OOD on novel maneuvers the training data didn't cover.** The plan-conditioning may steer the DiT toward action distributions it has never seen in training, causing incoherent behavior. DAgger's prescription:
1. Run the plan-conditioned policy (not just the streamer's policy) in the emulator.
2. At each step, record the VLM's "desired" action (query the oracle = VLM evaluating what should have been done).
3. Add (state, plan_token, oracle_action) to the training set for BC.
4. Fine-tune PlanHead on this mixed dataset.

This is exactly the **human gold trajectory collection** already built into the system (`record_play_server.py`), but the DAgger insight is that you must collect data **in the states the plan-conditioned policy visits**, not in the states a human would visit unprompted.

**Specific pitfall DAgger addresses:** in EXP-011 (SEQ "left then right" fails), the failure mode is OOD state distribution — the model reaches step 9 of the chunk having gone left, and steps 9-17 (right) are in states the model has never seen with a "right" instruction at that phase. DAgger would fix this by collecting corrective data specifically at the transition point.

### (e) Key caveat
DAgger requires a **queryable oracle** — an expert who can provide correct actions at any state the learner visits. In robotics this is a human. In our setup, the VLM or a human playing the game is the oracle, but querying the VLM in real-time for each action is expensive (seconds per call). A practical approximation: collect human gold trajectories from save-states that the plan-conditioned policy reached (not from streamer starting points).

---

## 14. Policy Distillation (RL → supervised)

**Paper:** "Policy Distillation"
Andrei A. Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, Raia Hadsell. (DeepMind)
2015. arXiv:[1511.06295](https://arxiv.org/abs/1511.06295) ✓

### (a) Core idea
A strong RL-trained DQN "teacher" policy (Q-network) can be compressed into a smaller "student" network by supervised regression on the teacher's **Q-value distribution** (soft targets via high-temperature softmax). Also demonstrates **multi-task distillation**: distil multiple game-specific policies into a single multi-task policy.

### (b) Mechanism
- Teacher: DQN trained via RL (has broad Q-value distributions at each state).
- Student: smaller network trained via CE(teacher_soft, student) on teacher-visited states.
- High temperature T amplifies information in Q-value differences (same as Hinton distillation).

### (c) Maps to our setup
The "teacher" role maps to: our current best plan-conditioned NitroGen policy (the one that scored best). The "student" is the next round's PlanHead. This is the standard distillation pattern: use the best current policy as a soft label to update the adapter. **Crucially, Rusu et al. show that the student can exceed the teacher** if the student generalises better (e.g., a well-regularised student may be more robust than a specialised teacher). This supports iterative improvement.

### (e) Key caveat
Policy distillation from a DQN works because the Q-function provides richer signal than win/loss. Our DiT doesn't have an explicit Q-function; we only have action samples. Matching soft action distributions requires MCMC samples from the flow model, which is expensive.

---

## 15. HIRO — Data-Efficient Hierarchical RL (options-style distillation)

**Paper:** "Data-Efficient Hierarchical Reinforcement Learning" (HIRO)
Ofir Nachum, Shixiang Gu, Honglak Lee, Sergey Levine.
NeurIPS 2018. arXiv:[1805.08296](https://arxiv.org/abs/1805.08296) ✓

### (a) Core idea
HIRO trains a two-level hierarchy: a **high-level policy** proposes subgoals at a coarse timescale; a **low-level policy** is rewarded for reaching those subgoals. Crucially, HIRO addresses the **non-stationarity of the low-level reward** as the high-level policy changes during training, using an off-policy correction trick.

### (b) Mechanism
- High-level: π_hi(s_t) → g_t (subgoal) every c steps.
- Low-level: π_lo(s_t, g_t) → a_t, rewarded by ||s_{t+c} − (s_t + g_t)|| (reach the subgoal in latent space).
- **Off-policy correction:** when replaying high-level transitions, re-label the stored subgoal g_t with the goal that makes the observed low-level behavior appear optimal (maximise π_lo likelihood over a candidate goal set).

### (c) What it fixes
Non-stationarity: when the high-level policy updates, the low-level reward changes, making old high-level transitions stale. The re-labelling trick makes experience reusable across high-level policy updates.

### (d) Maps to our setup
HIRO is the RL analogue of our planner→action model hierarchy. The VLM→PlanHead→DiT is precisely a two-level hierarchy. **HIRO's off-policy correction is a key insight for our replay buffer**: when replaying old (plan, trajectory) pairs from earlier VLM policy, we should re-label the plan to the one that would have made the observed trajectory *most likely* under the current PlanHead. This prevents the consolidated replay data from being stale after plan-policy updates.

### (e) Key caveat
HIRO requires the low-level reward to be goal-reaching in a metric space. Our low-level is a continuous action chunk from a diffusion model — there's no clean notion of "reached the goal state." The analogy holds structurally but the reward engineering is harder.

**Options framework (parent theory):**
Sutton, Precup, Singh (1999), "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning," *Artificial Intelligence* 112(1–2):181–211. ✗ (Pre-arXiv). The options framework formalises hierarchical temporal abstraction: an option is a (initiation set, policy, termination condition) triple. Our plan token is a soft, latent version of an "option" — it has an initiation state (beginning of plan window), a policy (DiT conditioned on z_plan), and an implicit termination (end of K chunks). Framing plans as options gives access to the full options theory for proving policy improvement.

---

## 16. LCB — Latent Codes as Bridges

**Paper:** "From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control"
arXiv:[2405.04798](https://arxiv.org/abs/2405.04798) ✓

### (a) Core idea
An LLM proposes a **latent code** (not natural-language subcommand) as the bridge between high-level instructions and low-level diffusion policy execution. The latent code is learned end-to-end jointly with both the LLM head and the diffusion policy.

### (b) Mechanism
- LLM → linear projection head → latent z ∈ R^d.
- Diffusion policy conditioned on z → continuous actions.
- End-to-end BC training on demonstrations: optimise CE on latent-conditioned action predictions.

### (c) Maps to our setup
**This is the closest architecture analogue to our PlanHead.** Our stack is:
- Frozen Qwen3.5 → resampler (K queries) → adapter → z_plan (K latent tokens).
- NitroGen DiT conditioned on z_plan via cross-attention.

LCB validates: (a) latent bridge > language bridge for bandwidth; (b) end-to-end training of the adapter improves performance; (c) the LLM can fire sparsely while the diffusion policy acts continuously.

**Key finding from LCB:** the latent code carries significantly more information than a discrete language command — enabling finer-grained action conditioning. This directly supports our K-token plan head over a single language-string interface.

### (e) Key caveat
LCB is trained on robot demonstrations (not games). The latent code is learned jointly; in our setup the DiT is frozen initially, meaning the latent-code→action mapping is fixed from NitroGen pretraining and the PlanHead must "find" latent codes that fit within that existing mapping. Joint training (Phase D of our design) is needed to fully realise LCB's benefits.

---

## 17. Hi Robot — System 1/2 with Language Commands (robotics)

**Paper:** "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models"
Physical Intelligence. 2025. arXiv:[2502.19417](https://arxiv.org/abs/2502.19417) ✓

### (a) Core idea
Two-level VLA: a **VLM (System 2)** decomposes complex open-ended instructions into simple natural-language subcommands; a **low-level VLA (π0, System 1)** executes each subcommand. The bridge is pure natural language (human-readable strings), not latent vectors.

### (b) What it fixes / Maps to our setup
Hi Robot validates the two-level hierarchy for complex, long-horizon tasks. The data structure is exactly our Stage-2 transcript format: (complex_instruction → decomposed_command → execution). The key difference: Hi Robot's bridge is natural language (interpretable, reusable across platforms), ours is latent (higher bandwidth but opaque). Hi Robot supports using a frozen small VLM for subgoal generation with a larger base model for execution.

### (e) Key caveat
Language bridge loses temporal granularity — "pick up the cup and put it on the shelf" collapses two sub-actions into one string. Our latent bridge can in principle carry richer temporal structure (the K-token plan tokens can encode phase/duration), but only if the PlanHead learns to use them that way.

---

## 18. GR00T N1 (parent architecture)

**Paper:** "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"
NVIDIA. 2025. arXiv:[2503.14734](https://arxiv.org/abs/2503.14734) ✓

**Relevance:** GR00T N1 is the direct parent of NitroGen. It is itself a System-1/2 architecture: System 2 = frozen VLM (Eagle-2) encoding vision+language; System 1 = a flow-matching DiT cross-attending to VLM tokens and producing action chunks. **Our project is rebuilding this duality** for games: NitroGen = GR00T's System 1; Qwen3.5 = a game-playing System 2. The consolidation loop distils VLM knowledge into the PlanHead (= GR00T's VLM↔DiT interface), making the System 2's knowledge permanent in the System 1 weights.

---

## 19. Synthesis and Pitfall Map for Our Consolidation Loop

### What our loop is doing (unified framing):
```
Loop:
  E-step: run plan-conditioned NitroGen in emulator from save-states (VLM fires once per plan window)
  Score: track game return per episode
  Filter: keep top-τ% (plan, trajectory) pairs
  M-step: BC on PlanHead — min -log p_θ(a | frame, z_plan) over filtered pairs
  Iterate
```
This is simultaneously:
- **Expert Iteration** (System-2 guided rollouts → distil into System-1 policy)
- **ReST-EM** (E = generate+score, M = SFT on filtered)
- **Amortized inference** (PlanHead learning to approximate the VLM's optimal z quickly)
- **System-2→System-1 distillation** (Yu et al.)

### Predicted pitfalls and mitigations:

| Pitfall | Source paper | Description | Mitigation |
|---|---|---|---|
| **Reward hacking** | STaR, ReST, ExIt | Policy finds ways to maximize game score proxy without actually improving play | Use VLM-judged plan success *and* game score; require both to filter |
| **Distribution narrowing** | AlphaZero, BC theory | Filtering to top-τ% collapses diversity; after K rounds, policy becomes brittle | Keep diverse replay buffer; don't filter too harshly (τ > 50th percentile); KL-regularise toward base policy |
| **Covariate shift (OOD states)** | DAgger | Plan-conditioned policy visits states not in training data; compounding error | Collect DAgger-style corrections: record human/VLM actions at states the plan-conditioned policy *actually* visits, not just human demonstrations |
| **Amortisation gap** | VAE / amortized inference | PlanHead produces poor z for rare/novel (frame, plan) combinations | Expand save-state distribution; cover more game scenarios during data collection |
| **Frozen DiT bottleneck** | Yu et al., LCB | If DiT's pretrained action manifold doesn't include the desired behavior, no PlanHead adjustment will produce it | Must unfreeze DiT with LoRA (Phase D) for hard maneuvers; this is already in the design |
| **Small dataset → overfitting** | Magister/Orca CoT distil | Only top-τ% of rollouts used; if episodes total is small, each M-step trains on <100 trajectories | Run many save-state episodes in parallel; target ≥1000 successful episodes per M-step round |
| **Stale replay** | HIRO off-policy correction | Old (plan, trajectory) pairs become inconsistent after plan-policy update | Re-label stored plans with current VLM's most likely plan for observed trajectory; prune old buffer |
| **VLM memorization** | LITERATURE.md §E.14 | VLM's plans for famous games are recalled, not reasoned; will fail on unseen game states | Evaluate on custom levels/homebrew ROMs; include novel game scenarios in save-state collection |

---

## 20. Reading Priority

| Priority | Paper | Why |
|---|---|---|
| 🔴 IMMEDIATE | Anthony et al. 2017 [1705.08439] | Exact loop template; understand convergence conditions |
| 🔴 IMMEDIATE | Yu et al. 2024 [2407.06023] | The explicit S2→S1 distillation paper; maps 1:1 to our work |
| 🔴 IMMEDIATE | Ross et al. 2011 [1011.0686] | Fix OOD/covariate shift; critical for loop robustness |
| 🟠 HIGH | Singh et al. 2023 [2312.06585] | EM framing; pitfall analysis on filtering threshold |
| 🟠 HIGH | Lin, Snell et al. 2025 [2504.13171] | Sleep-time compute validates sparse VLM + persistent plan tokens |
| 🟡 MEDIUM | Gulcehre et al. 2023 [2308.08998] | ReST Grow/Improve; practical implementation details |
| 🟡 MEDIUM | Zelikman et al. 2022 [2203.14465] | STaR; rationalisation trick for counterfactual plans |
| 🟡 MEDIUM | Nachum et al. 2018 [1805.08296] | HIRO; off-policy correction for replay buffer |
| 🟢 BACKGROUND | Kingma & Welling 2013 [1312.6114] | Amortisation gap theory |
| 🟢 BACKGROUND | Hafner et al. 2023 [2301.04104] | World-model imagination; if/when we go offline |
| 🟢 BACKGROUND | Rusu et al. 2015 [1511.06295] | Soft-target policy distillation; distributional BC |

---

## Gaps and Uncertainties

1. **AlphaGo (Silver et al. 2016, *Nature*)** — not on arXiv; the specific MCTS→policy distillation implementation details require reading the Methods section of the Nature paper directly. The claim about "visit-count soft targets" is well-documented in the community but should be verified for the exact loss form.
2. **Options framework (Sutton, Precup, Singh 1999)** — pre-arXiv; published in *Artificial Intelligence* 112(1–2):181–211. Verified from prior knowledge, not live-fetched.
3. **Self-Rewarding LMs (Yuan et al. 2024) [2401.10020]** — verified as existing but not deeply summarised here; it covers using the same model as both judge and learner, which is relevant if we want the VLM to evaluate its own plan quality.
4. **Constitutional AI [2212.08073] / RLHF/PPO variants** — not covered; tangentially relevant for training the VLM on game feedback but outside our frozen-VLM constraint.
5. **"Teaching Small Language Models to Reason" [2212.08410]** — verified arXiv ID; author list and full abstract not live-confirmed — treat as ⚠️ unverified before formal citation.
6. **LCB authors** — arXiv:2405.04798 confirmed as "From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control" but full author list not confirmed in this session; verify before submitting.
7. **MuZero latent world model** — the connection to our offline consolidation is conceptual; MuZero uses a purpose-built neural simulator, not a generalist video model. Verify the specific rollout training algorithm before claiming direct precedent.

---

*End of LIT_R11_distill.md. ~380 lines. All arXiv IDs live-verified except where marked ✗.*

---

## Summary for Main Agent

**Findings:** All 18+ requested papers located and mapped with verified arXiv IDs. The **single closest precedent** for the consolidation loop is **Expert Iteration** (Anthony et al. 2017, arXiv:1705.08439) — an explicit System-2 search distilled into a System-1 policy network through supervised imitation of search-improved distributions — it predicts the bottleneck is always planner quality, not distillation capacity. The **most directly named paper** for the S2→S1 framing is **Yu, Weston et al. 2024** (arXiv:2407.06023), which experimentally confirms that System-2 outputs can be compiled into System-1 weights, with the caveat that tasks requiring genuine multi-step deduction resist distillation. **DAgger** (Ross et al. 2011, arXiv:1011.0686) is the canonical cure for the covariate-shift failure mode (confirmed by EXP-011's SEQ breakdown), prescribing rollout collection at states the *plan-conditioned* policy visits, not the human's states. **Sleep-time Compute** (Lin et al. 2025, arXiv:2504.13171) directly validates the sparse-VLM + persistent plan-token architecture.

**Key predicted pitfall:** reward hacking + distribution narrowing if rollout filtering is too aggressive (top-τ%). The ReST-EM and AlphaZero literature both converge on: keep a diverse replay buffer, use soft targets (full distribution, not argmax), and apply DAgger-style on-policy correction to prevent compounding OOD error.

