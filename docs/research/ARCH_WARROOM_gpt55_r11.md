# ARCH WARROOM — GPT-5.5 Round 11: TTT/literature-grounded position

## TL;DR
Do **not** do TENT/SAR-style entropy minimization on NitroGen weights tonight. The TTT lit is classification-heavy, and the review explicitly found **no confirmed TTT method for continuous-action flow-matching DiTs**; entropy for our velocity field is undefined/nontrivial (TENT, arXiv:2006.10726; SAR, arXiv:2302.12400; LIT_R11_ttt gap). R9 says online weight updates collapse unless tightly supervised/KL-anchored.

Adopt **TTT-Layers/fast-weight TTT as a forward-pass fast state**, not persistent actor training: K plan tokens are an episodic fast-weight memory (Sun et al. 2024, arXiv:2407.04620; Schmidhuber 1992; Ba et al. 2016, arXiv:1610.06258; Schlag et al. 2021, arXiv:2102.11174; von Oswald et al. 2023, arXiv:2212.07677). Stabilize live replans with **local-window, augmentation/candidate consistency + KL anchor**, not actor entropy.

**ONE experiment tonight:** R11-FastState-Stability: from live save-states, compare baseline live replanning vs fast-state plan-token hysteresis/EMA selected over M=4 VLM-abstracted plan candidates by (i) plan-token consistency, (ii) CFG disagreement, (iii) 1-step emulator RTG. Pass: replan flip-rate 50–60% → **≤25%**, live Δ_plan no worse than baseline by >5 and ideally +10, expressiveness battery **R≥0.80**, duck preserved, null maxdiff **0.0**, 5 seeds for reward claims.

## Q1 — TTT for stale plans / online adaptation
**Pick:** TTT-Layers-style **fast-state TTT**, not TENT/SAR actor-weight TTT and not Akyürek per-episode DiT-LoRA for stale plans.

Why:
- TTT-Layers makes the hidden state a fast model updated by a self-supervised reconstruction loss while slow weights stay fixed (Sun et al. 2024, arXiv:2407.04620). That maps cleanly to our K plan tokens: update/choose the **fast conditioning state**; keep NitroGen and the KL-anchored bridge fixed.
- TTT-Video says adaptation should be **local**: last ~20–30 frames / ~2s is better than long history (Gandelsman et al. 2023, arXiv:2307.05014). Our 18-step chunk is exactly that window.
- MEMO says single-sample TTA should use augmentation consistency/marginal entropy, not batch entropy (Zhang et al. 2022, arXiv:2110.09506). For us, replace image augmentations with M VLM-abstracted/cropped/temporal-neighbor plan candidates.
- TENT is the wrong default: it assumes discrete predictive entropy and BN/LN adaptation; SAR fixes TENT collapse with gradient filtering + SAM but still assumes entropy outputs (TENT arXiv:2006.10726; SAR arXiv:2302.12400). Our actor is continuous flow matching, and R9 says weight updates are fragile.
- Akyürek-style per-task LoRA (arXiv:2411.07279v1) is for **few-shot supervised task examples**. Use it for ADDITION/manifold expansion, not stale-plan jitter.

**Self-supervised/test-time signals in our emulator setting:**
1. **Plan-token consistency:** M candidate VLM-abstracted plans from the same 18-frame window should produce nearby z-plan and nearby first action chunk; this is MEMO-style consistency (Zhang et al. 2022).
2. **Locality:** only use current/previous chunk frames; discard old frames per TTT-Video (Gandelsman et al. 2023).
3. **CFG disagreement penalty:** high `||v_c-v_u||` means plan fights frame-prior; penalize unless emulator reward confirms it. This reuses the R3/R4 gate and CFG theory (Ho & Salimans 2022, arXiv:2207.12598).
4. **Emulator forward check:** from save-state, roll 1–3 chunks and score survival/progress/goal; this is not a gradient through the DiT, just selection of the fast state.

**Runnable prototype:**
- At each replan, sample/construct M=4 **VLM-abstracted** plan candidates, not raw narration.
- Encode each to z via current KL-anchored PlanHead.
- Candidate score: `S = RTG_1to3 - α*CFG_disagree - β*||z-z_prev||² + γ*candidate_consensus`.
- If best score beats current by margin, switch; else keep/EMA z: `z_t = 0.8*z_{t-1}+0.2*z_best` for one chunk.
- No weights updated. Null plan bypasses all of this and must still produce base action exactly.

**Metric/falsifier:** flip-rate = fraction of replans whose semantic action class or z cosine changes beyond threshold. Pass ≤25% and no Δ_plan loss >5. Falsify if flip drops but expressiveness battery R<0.80 or duck-on-command falls below R9 KL-anchor baseline: that means hysteresis stabilized by suppressing plan responsiveness.

## Q2 — consolidation loop: which self-improvement algorithm?
Operationally our loop is **ReST-EM**, conceptually **Expert Iteration**, with one STaR trick.

- **Closest operational recipe: ReST / ReST-EM.** Grow = sample rollouts from save-states; Improve = supervised BC on filtered winners (Gulcehre et al. 2023, arXiv:2308.08998; Singh et al. 2023, arXiv:2312.06585). This is exactly our save-state → score → KL-anchored demo-fit loop.
- **Conceptual frame: ExIt.** Slow expert/search produces better behavior; System 1 is trained by supervised imitation of improved search distributions, not raw RL (Anthony et al. 2017, arXiv:1705.08439). Our VLM+emulator branch search is the expert.
- **Borrow from STaR only for counterfactual rationalization:** failed rollout + desired outcome → ask VLM to produce a clean abstract plan explaining the success, then only train if emulator/VLM-judge validates it (STaR, arXiv:2203.14465). Do not train on arbitrary rationales.

**Concrete recipe detail to borrow:** ReST-EM’s Grow/Improve split + fresh resampling each round + top-threshold filtering, but with AlphaZero/Hinton-style soft/distributional targets where feasible to avoid temperature-0 narrowing (AlphaZero arXiv:1712.01815; Hinton distillation arXiv:1503.02531). Keep a diverse replay buffer; do not top-5% filter only.

**Collapse/reward-hacking prediction from papers:** ReST/STaR warn that noisy reward filters train on lucky/hacked completions; AlphaZero warns hard argmax filtering narrows the policy; DAgger warns off-policy demos miss learner states (Ross et al. 2011, arXiv:1011.0686). Our guards match the lit: KL-anchor to base/R9 winner, plan-head-only for evocation, VLM plan-success judge plus game RTG, diverse save-states, >=5 seeds+CIs.

**Minimal runnable form tonight:** after the router, for EVOKE maneuvers only: K=8 plans × G=4 action samples from the same save-state; score by maneuver event+survival; keep top 25–40% **and** diversity buckets; train plan_head only with `--kl-anchor 0.3`; evaluate Δ_plan + expressiveness battery. Restart each Improve from the same R9 KL base, not from a drifted online checkpoint (CoTTA/EATA anchor logic: arXiv:2203.13591, 2204.02610).

## Q3 — short path as fast weights / continuous latent
Yes: the short path is already a fast-weight / implicit TTT mechanism.

Mapping:
- Perceiver resampler = fixed K latent bottleneck over arbitrary VLM states (Perceiver IO, arXiv:2107.14795).
- K plan tokens = fast episodic memory written by slow planner and read by frozen DiT (Schmidhuber 1992; Ba et al. 2016, arXiv:1610.06258).
- DiT cross-attention over plan tokens = Hopfield/associative retrieval (Ramsauer et al. 2020, arXiv:2008.02217) and a fast-weight-programmer read (Schlag et al. 2021, arXiv:2102.11174).
- ICL-as-GD says attention can implement an implicit gradient update in activations (von Oswald et al. 2023, arXiv:2212.07677). Caveat: proven for linear tasks, so cite as analogy, not theorem for flow DiTs.
- GR00T N1 and π0 validate VLM tokens → flow/DiT action chunk via cross-attention as the parent VLA pattern (GR00T arXiv:2503.14734; π0 arXiv:2410.24164). LCB validates latent bridges over language strings (arXiv:2405.04798).

Recommendation: **lean harder on the continuous short path before explicit TTT**. Preserve KL-anchored PlanHead; add fast-state hysteresis/EMA and optionally zero-init `FastPlanMod`/AdaLN as an evocation amplifier, not as a replacement. AdaLN-zero is literature-correct for null-safe additive steering (DiT, arXiv:2212.09748; FiLM, arXiv:1709.07871), but cross-attention remains the main fast-weight path.

Contradiction to prior framing: “TTT” should not mean “online gradient updates to actor weights” here. The fast-weight/ICL literature says forward-pass latent conditioning is already a form of per-context adaptation; explicit weight TTT is heavier and less justified for continuous flow actions.

## Q4 — capability ADDITION without forgetting
For router-confirmed ADD chunks, use **pooled plan-gated DiT-LoRA**, not per-game models.

Mechanism:
- DAgger says BC fails under learner-induced covariate shift; collect data at states the plan-conditioned policy actually visits (Ross et al. 2011, arXiv:1011.0686). Our emulator save-states are the queryable substrate.
- Akyürek per-task LoRA shows few-shot test-time LoRA can solve new abstract tasks when the backbone is a good slow initialization (arXiv:2411.07279v1). Use per-maneuver LoRA as a **scratch adapter** for grab-mesh-like additions, then distill/merge into a single generalist plan-gated LoRA; do not ship per-game adapters.
- VLA hierarchy papers support low-level skill addition under high-level commands: RT-H language motions (Belkhale et al. 2024, ID unconfirmed in lit), Hi Robot (arXiv:2502.19417), π0.5 (arXiv:2504.16054), LAPA/Genie latent actions (arXiv:2410.11758 / 2402.15391).

Safeguards:
1. **LoRA + replay + KL/null anchor.** Keep a reference replay of base/evocation chunks and penalize drift; this is EATA/Fisher/EWC-style forgetting prevention (Niu et al. 2022, arXiv:2204.02610) plus CoTTA source restoration logic (arXiv:2203.13591).
2. **Plan-gated or null-masked LoRA only.** Ordinary LoRA can break exact null-invariance per repo notes; require null action bit-identity maxdiff 0.0 after merge.
3. **Add-chunks only.** Router must show low null-AUC, low evoc-ratio, and low emulator outcome-Δ before LoRA. Duck/Rex-if-evocable stay plan-head-only.
4. **Soft-target distillation where possible.** Hinton/Rusu distillation says soft distributions preserve more behavior than single argmax traces (Hinton 2015, arXiv:1503.02531; Rusu et al. 2015, arXiv:1511.06295).

## Q5 — coherent System-2→System-1 architecture + tonight’s experiment
**Architecture:**
1. **Planner:** frozen Qwen3.5-2B emits/abstracts clean System-2 objectives; Qwen-LoRA allowed later only if VLM abstraction itself fails. Raw narration is locator/source only.
2. **Short/evocation path:** KL-anchored PlanHead (resampler+adapter) writes K=8 fast-weight plan tokens; optional zero-init AdaLN/FastPlanMod for null-safe authority. This is Perceiver/fast-weight/ICL-as-GD conditioning.
3. **Runtime fast state:** local-window candidate consistency + z-EMA/hysteresis; no persistent weight TTT for stale plans.
4. **Consolidation:** ReST-EM/ExIt sleep-time loop: Grow rollouts from emulator save-states; Improve with KL-anchored plan-head BC on validated winners; restart from anchor each round; diverse replay.
5. **Addition:** router-confirmed absent skills train pooled plan-gated DiT-LoRA with DAgger-style on-policy states + EATA/CoTTA replay/anchor; merge into one generalist.

**Single highest-value experiment tonight: R11-FastState-Stability.**
- Games/states: SMW, SMB1, MMX live-plan save-states used by staleness_probe; 5 seeds for reward claim.
- Conditions: baseline live replan every A=2 vs fast-state TTT candidate selection/EMA/hysteresis.
- Metrics: replan flip-rate; live Δ_plan-null; oracle-live gap; CFG disagreement; duck/retreat/jump/wait/up battery; null bit-identity.
- Pass: flip ≤25%; live Δ_plan ≥ baseline-5 and preferably +10; oracle-live gap not larger; battery R≥0.80; duck preserved; null maxdiff 0.0.
- Falsifier: flip reduction comes by plan suppression (battery <0.80 or duck collapse), or emulator-selected fast state improves one game but hurts another beyond CI → revert to pure KL-anchor and invest in ReST-EM consolidation instead.

## Steelman of Opus’s likely pick
Opus will likely pick **KL-anchored ReST-EM consolidation first**: R9 KL-anchor is the clean empirical winner; ExIt/ReST-EM are exact S2→S1 precedents; avoid speculative TTT code. That is strong. My only disagreement is sequencing: the owner explicitly asked for TTT/related things, and stale replan flip is still open. A no-weight fast-state TTT probe is cheap, null-safe, and directly lit-grounded; if it fails, Opus’s KL-ReST path becomes the default with no damage.
