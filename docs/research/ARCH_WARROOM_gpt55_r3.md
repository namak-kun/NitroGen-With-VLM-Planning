# GPT-5.5 Round 3 opening: actor-OOD is the primary drift problem

## Q1 stance: separate plan-OOD from actor-OOD; subtle trained-game drift is mostly actor-OOD

I separate the two failures this way. **Plan-conditioning OOD** is when the bridge sees a novel or contradictory `(frame, plan)` pair: stale objective, wrong objective, or a plan latent whose adapter was never trained to mean the required counterfactual. That exists here: the project already found left/right env-free steering is fragile, the failure localizes to `PlanAdapter`, and cleaning label noise improves DiT-space token separation from 0.50 to 0.688. So plan-OOD is real.

But the owner's phrase “subtly different state even in a trained game” is more likely **DiT-actor OOD**: the Markov flow actor is still behavior cloning. A small chunk error changes position/velocity/enemy phase; the next frame is off the demonstration manifold; the next chunk is worse. This is the classic covariate-shift result formalized by Ross, Gordon, and Bagnell, **DAgger**, AISTATS 2011: imitation learners trained only on expert-state distributions incur compounding errors under their own induced state distribution, and aggregation of on-policy states is the fix. The seed's strongest repo evidence points the same way: after the eval fix, `rwbc_actor_adapt.py` on-policy reward-weighted BC improves Sonic **0.58 > 0.25** and SMW **0.53 > 0.41**. That improvement is not explained by a better VLM plan; it is the actor learning from its own visited states.

I concede Opus's full-depth-prefill framing covers one part of plan-OOD: if the slow plan text is correct but grounding is stale, recomputing a full-depth frame-conditioned latent is a clean implementation. It does not explain why fixed clean-plan RWBC improves closed-loop reward. That is actor state coverage.

## Q2 stance: a fresher plan is not load-bearing for actor-OOD

A better/fresher plan does **not** fundamentally reduce actor-OOD drift. The key prior finding is decisive: on factual data the frozen base DiT already reproduces streamer actions from the frame alone (buttons AUC 0.82-0.96; sticks corr 0.4-0.76), making factual plan conditioning near-zero-gradient. In repo terms, `NitroGenPolicy._sample_chunk` already compares conditional and null velocities through CFG (`v_u + w*(v_c-v_u)`), and `compute_plan_tokens`/`apply_null_mask` cleanly remove plan slots. If the null path already knows the factual local action, adding a more fluent VLM sentence cannot teach recovery from a new off-manifold Mario x-position, Sonic slope phase, or enemy timing.

So actor drift is an **on-policy state-coverage problem**. It is fixed by emulator data: DAgger-style collection, RWBC, perturb-and-recover, or save-state GRPO. This also matches the broader VLA literature: π0 and π0.5 (Physical Intelligence, 2024/2025) use VLM semantics plus flow action chunks, but still rely on action data; GR00T N1.5 uses V/L embeddings with a flow DiT and AdaLN, not magic language-only correction; Helix (Figure, 2025) uses a fast visuomotor S1 conditioned on S2 latent, but S1 is still a trained policy over state distributions.

What is the VLM for in drift, then? Three things, not low-level recovery: **(1)** objective selection and subgoal text/latent for long-horizon decisions NitroGen cannot infer from one frame; **(2)** replan-trigger context: “I am stuck / wrong room / should abandon current objective”; **(3)** reward/credit scaffolding for emulator search, e.g. propose K candidate objectives to rank from a save-state. VLM-LoRA is allowed later if frozen Qwen cannot represent objectives, but it is high-cost and should not be the first actor-OOD lever.

## Q3: yes, add online OOD detection and a null-preserving gate

Training-to-recover is not enough; subtle drift needs a detector before it compounds. I propose an **ActorRisk gate** calibrated on successful in-distribution emulator rollouts of `btn_s600` using empirical 95th/99th percentiles, conformal-style (Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, 2005; energy-style OOD precedent: Liu et al., NeurIPS 2020, “Energy-based Out-of-distribution Detection”).

Signals per chunk:

1. **CFG/flow residual ratio** (deploy-only): during the existing sampler, compute  
   `r_cfg = mean_i ||w*(v_c-v_u)|| / (mean_i ||v_u|| + eps)`. This is cheap because CFG already computes both velocities. If plan guidance is forcing a large delta relative to null, plan/frame or actor uncertainty is high.
2. **K-sample action disagreement** (deploy-only): sample `K=4` chunks with small late-step noise (`noise_sigma≈0.05`, already supported in `eval_policy.py`), measure first-`A` continuous stick std plus button vote entropy. High disagreement means the flow policy has no stable local action.
3. **Plan-frame contradiction** (deploy-only with VLM): cosine distance between current frame-conditioned plan tokens and a short moving average of successful tokens for the same plan; use only as a plan-gate, not as an actor-OOD oracle.
4. **Emulator telemetry** (emulator/train/verified deploy only): progress delta below calibrated p5 for 3 chunks, velocity sign contradicts objective for 2 chunks, damage/death/near-fall bits.

Threshold and action: trigger if `r_cfg > p99_success` OR `disagree > p99_success` OR telemetry fires twice in 3 chunks. Gate response is staged: **first chunk:** set plan CFG to `min(w,1.0)` and sample/select best of K under a conservative action prior (survive, keep moving objective direction); **second consecutive trigger:** force null-plan fallback for one chunk (preserves null-invariance exactly through `apply_null_mask`) and request a fresh VLM objective; **emulator mode:** branch 8 short rollouts from the save-state and execute the best-return first `A` actions. This preserves the null contract because the gate only chooses among already-valid conditional/null samplers; it does not change the base path.

## One cheap falsifiable experiment

Run a paired **CleanPlan-vs-RWBC discriminator** with existing code on SMW and Sonic. Use `btn_s600`, emulator reset/save-state support (`emulator_env.py`/`snes_env.py` expose `save_state/load_state`), and `rwbc_actor_adapt.py`.

Protocol: for 8 shared seeds/save-states, measure normalized reward with: **A)** baseline generated plans; **B)** `--use-correct-plan` fixed clean near-oracle plan, no training; **C)** `--use-correct-plan --lora-only --save-delta files/rwbc_actor_ood_delta.pt` after RWBC collection/eval. Metric: `ActorOOD_gap = reward(C) - reward(B)` and `PlanOOD_gap = reward(B) - reward(A)` under identical seeds.

Decision rule that would change my mind: if `PlanOOD_gap >= 0.15` normalized reward **and** `ActorOOD_gap < 0.05` on both games, plan-OOD dominates. If, as I expect, clean plans help little (`<0.10`) while lora-only RWBC adds `>=0.10` or recovers at least half the base-to-scripted-right gap, actor-OOD is dominant and emulator state aggregation is the main lever.
