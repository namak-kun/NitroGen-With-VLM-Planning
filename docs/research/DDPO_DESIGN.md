# DDPO / diffusion-policy-gradient for NitroGen — DESIGN (build it WITH @namak-kun)
Written 2026-06-26 night. Motivated by the night's strongest finding: RWBC self-imitation only works in a
COMPETENCE BAND (amplifies existing good behavior). To push a too-OOD actor (SMW +0.08) or a near-ceiling
actor, and to DISCOVER new motor skills, the actor must EXPLORE — which self-imitation BC cannot do. That
is exactly what diffusion-policy-gradient (DDPO/DPPO) provides. This is the next actor-RL investment.

NOT an implementation (you wanted the big build done together) — a concrete, review-ready design.

## 1. Why self-imitation isn't enough (recap, evidenced)
Filtered reward-weighted BC reweights the policy toward its OWN top rollouts → it SHARPENS the existing
action distribution. It cannot place probability mass on actions the current policy never samples. So:
- too-OOD actor (SMW): ~no good chunks in the rollouts → nothing to sharpen → no gain / drift.
- near-ceiling actor (TheXTech l8): already concentrated on its best → little to sharpen.
Real RL (policy gradient) moves mass toward HIGHER-reward actions whether or not they were likely,
via the score-function gradient — i.e. it explores.

## 2. The core obstacle: NitroGen samples actions DETERMINISTICALLY
`_sample_chunk` (eval_policy.py) integrates the flow-matching velocity field with Euler steps:
`actions += dt * vel(...)` for num_inference_timesteps — a deterministic ODE. No sampling stochasticity →
no log-prob → no policy gradient. To do PG we must make chunk sampling STOCHASTIC with a tractable
per-step log-prob. Two standard routes:
- **(A) SDE / DDPO-style**: treat each denoising step as a Gaussian policy
  `x_{k-1} ~ N(x_k + dt·v_θ(x_k,k,cond), σ_k² I)` with a chosen schedule σ_k>0. The chain log-prob is
  `Σ_k log N(x_{k-1}; μ_k, σ_k²)`. Gradients flow into v_θ (the LoRA params) through μ_k. This is the
  Black et al. DDPO formulation, adapted from DDPM to our flow field (μ from the flow velocity).
- **(B) Last-step Gaussian head**: keep the ODE deterministic for k>0, add a single Gaussian at the final
  step. Cheaper, lower-variance log-prob, but less exploration. Good for a first cut.
RECOMMEND start with (A) but only inject noise on the LAST few steps (σ_k=0 early) — most action variation
is determined late; this bounds variance while keeping exploration. Expose σ schedule as a knob.

## 3. The loop: SAVE-STATE GRPO (no critic) — fits our substrate
Our emulator envs have frame-exact save/load → the AGENTS.md "save-state GRPO" is the natural fit and
sidesteps a value network (the long credit path frame→plan→18 actions→sparse reward makes a critic hard):
```
for iter:
  sample a batch of START STATES s (env.save_state() snapshots; or your manually-authored level states)
  for each s, restore and sample K stochastic chunks {a_i} (same plan, same frame) -> rewards {r_i}
  advantage A_i = (r_i - mean_k r) / (std_k r + eps)          # GRPO group-normalized, no critic
  loss = - Σ_i A_i · logprob(a_i)  +  β · KL(π_θ ‖ π_base)    # PPO-clip optional
  step (LoRA-only, low LR)
```
- K plans-per-state and per-state grouping give a clean baseline (GRPO) — no critic net.
- Uses save_state/load_state we verified frame-exact (Sonic/SMW/GBA emulator envs; NOT proc TheXTech).
- Your manually-authored level start/end states are IDEAL start-state distributions here.

## 4. Anti-collapse (the duck's warning + our measured collapse)
- **LoRA-only, low LR** (we measured naive full/high-LR collapses; LoRA-only RWBC was stable).
- **KL-to-base** β·KL(π_θ‖π_base) on the denoising distribution (π_base = frozen DiT, no LoRA) — keeps the
  policy near the competent prior; the single most important regularizer for PG.
- **PPO-clip** ratio on the chain log-prob for trust-region stability.
- **Masked-null untouched**: PG trains LoRA only → the plan-head/null path is unchanged (CFG intact).
- **Reward shaping already done**: retro_rl_env has death/stuck penalties (anti-exploit).

## 5. Integration points (small, localized)
- `eval_policy.py`: add `sample_chunk_stochastic(frame, plan, cfg, sigma_schedule) -> (chunk, logprob,
  trace)` — a copy of `_sample_chunk`'s loop that (a) adds σ_k noise, (b) accumulates per-step log N. Keep
  the deterministic path for eval.
- New `planner_poc/ddpo_train.py`: the §3 loop, reusing make_env (ONE env, reset/load_state per group),
  the GT reward, and the LoRA param group from rwbc. Log: reward, KL, logprob, grad-norm, entropy.
- Reuse: build_batch is NOT needed (PG uses the sampling log-prob, not a BC target); the model forward for
  velocity is `m.model(...)` as in `_sample_chunk`'s `vel()`.

## 6. First experiments (when you're back)
0. **DONE — premise VALIDATED (planner_poc/ddpo_explore.py + eval_policy._sample_chunk noise_sigma):**
   late-step SDE noise makes the flow sampler stochastic; on an RWBC-adapted (competent) Sonic actor at
   σ=1.0 it EXPLORES (reward std 0.051) and best-of-8 = +0.065 vs greedy −0.020 (Δ+0.085) — exploration
   finds positive-reward chunks greedy misses. So diffusion-PG HAS signal. NUANCE: a sharpened policy
   needs HIGHER noise (σ~1.0; σ=0.3 was too peaked, std 0.006) than base (σ~0.6). The sampler foundation
   is built + de-risked; the remaining crux is the per-step log-prob math (§2A) + the loop (§3).
1. **Sanity**: stochastic sampler reproduces deterministic reward at σ→0; reward variance grows with σ. (✓ confirmed)
2. **Sonic save-state GRPO**, K=8, LoRA-only, β-KL swept — does it beat RWBC's +4.7 (PG should explore past
   the self-imitation plateau)? This is the decisive "exploration > sharpening" test. NOTE: the cheap
   shortcut (exploration-augmented self-imitation BC, RWBC --explore-sigma) was TESTED and FAILED (Sonic
   Δ+0.22 vs greedy +1.54 control) — BC imitates lucky-noise, so REAL PG (log-prob reweighting) is required,
   not BC-on-explored. This is exactly why §2's log-prob is the crux.
3. **SMW** (RWBC FAILED here, +0.08→−0.16): the cleanest test that PG can move a too-OOD actor that
   self-imitation could not. If DDPO lifts SMW, that's the headline.
4. Then **plan-conditioned** PG (does reward-driven exploration find plan→action couplings RWBC missed?).

## 7. Risks / open questions
- Flow-matching (not DDPM) log-prob: the Gaussian-per-step approximation on a flow ODE is heuristic;
  validate the log-prob gradient numerically (finite-diff on σ) before trusting it.
- Variance: chain log-prob over num_steps × 18 × 25 dims is high-variance; mitigations = late-steps-only
  noise, group-normalized advantage, PPO-clip, small K-but-many-states.
- Compute: K× the rollouts of RWBC, but each is a short chunk from a save state → tractable on 4×A6000.
- This is the bigger build — recommend pairing on the stochastic-sampler log-prob math first (the crux).
