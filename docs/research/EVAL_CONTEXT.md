# EVAL CONTEXT — for the GPT-5.5 rubber-duck (evals only)

This is the **evaluation problem** for a VLM-planner + diffusion-actor game agent. Goal of the discussion:
**how do we evaluate this model when our supervision is only expert human demonstrations and the
"reached level i+1" reward signal is too sparse?** Be concrete and skeptical.

## The system (1-paragraph)
A **frozen Qwen3.5-2B VLM (System 2 / planner)** reads recent frames (+ optionally its prior plan and an
executed action trace) and emits a short text **plan**. A **trainable bridge** (Perceiver resampler, K=8
queries, + linear adapter) turns the VLM's frame-grounded text-token hidden states into **K=8 "plan tokens"**
injected into the cross-attention of a **frozen ~500M flow-matching Diffusion Transformer (NitroGen, System
1 / actor)**, optionally with a small **DiT-LoRA (rank 16)**. The DiT maps the **current frame → an 18-step
chunk of 25-dim gamepad actions** (21 buttons + 2 analog sticks). It's markov (sees only the current frame),
fast/reactive; the planner runs every ~2 chunks (~1.2 s). CFG (classifier-free guidance, weight w≈8) scales
plan influence; a "null/masked" plan reproduces the base DiT exactly.

## What's trained, on what data (CONTAMINATION/VALIDITY flags — important for eval)
- **NitroGen DiT**: pretrained by NVIDIA on lots of gameplay (frame→action). Frozen. Some of our test games
  are IN-distribution, others (Genesis Sonic, GBA) are OUT-of-distribution for it.
- **The bridge + the default checkpoint `btn_s600`** (the "base"/"null" in all our experiments = this, with
  the plan dropped — NOT raw NitroGen): trained on **SYNTHETIC plans + counterfactual caches** — terse
  synthetic direction/button plans ("jump", "accelerate", "move left") paired with chunks whose dominant
  action matches, via an outcome-contrastive objective. **So the planner→action interface was tuned on
  synthetic text, not human demos or gameplay reward.** (FLAG: any eval that uses the planner's own plans is
  partly circular w.r.t. this synthetic training.)
- **The VLM-as-judge** (a separate scalar progress-head on the SAME frozen Qwen-2B; "did the agent make
  progress?" from before/after frames + an objective): trained on **GT-labeled before/after pairs collected
  from the reward envs** (~few hundred pairs, `docs/rl_data/{sonic,smw,minish,solarus,thextech}`), plus an
  OPSD privileged→unprivileged self-distillation variant. **NARROW training distribution**, works best where
  progress = visible MOTION (platformers), weaker on top-down. Zero-shot Qwen-judge is NOT reward-grade
  (over-credits, inverted on shmups). (FLAG: judge validated on few-hundred motion-correlated pairs; also
  these are POPULAR games the VLM has a pretraining prior on → may not transfer to obscure games like
  Witchblast.)

## The data we have for eval
- **15 human expert demos** across 4 games: SMW (SNES, 8 demos, 3–128 s), Sonic 2 (Genesis, 4, 89–300 s),
  Zelda Minish Cap (GBA, 1, 360 s), Fire Emblem Sacred Stones (GBA, 2, 127 & 417 s). Each demo dir
  (`docs/demos/demos/<Game>/<ts>/`) has: **`demo.bk2`** (full replayable libretro movie of the expert
  play), **`demo.npz`** (observations (N+1,H,W,3) + per-frame actions (N,12 buttons) + rewards + dones),
  **`initial.state` / `final.state`** (gzip'd frame-exact savestates), **`episode.mp4`**, **`meta.json`**
  (ROM sha1 — MATCHES our ROMs exactly — buttons, fps, steps).
- **Frame-exact emulator envs** (stable-retro): can load any savestate and step deterministically; we can
  also reproduce ANY point of the expert trajectory from the `.bk2` and snapshot a savestate there. So:
  from 15 demos we can derive **MANY (start-state, expert-continuation) eval points** by subdividing the
  multi-minute trajectories (the demos are NOT just 15 short clips). [user's correction: sample size is
  bigger than it looks.]
- **GT reward vars from game RAM** (per env): Sonic/SMW `screen_x` (horizontal progress) + `score`,`lives`,
  `rings`; Minish `0.01·euclidean(Δx,Δy) + 5.0·(room change)`; **Fire Emblem: NO reward yet (turn-based)**.

## Current eval methods (and their weaknesses)
1. **"progress" metric** = Δscreen_x (raw pixels). Used in gate/best-of-K/ablation. **EXPLOITABLE** — a
   run-right-into-death policy scores high. Only meaningful for horizontal platformers; meaningless for
   top-down (Minish) and turn-based (FE).
2. **"shaped" metric** = Σ step reward = `0.01·Δscreen_x + 0.01·Δscore − 5.0·death − 0.05·stuck`, episode
   ends on death/stuck/256-rows. Anti-exploit. BUT still progress-dominated, not normalized across start
   states, and tonight the shaped gains were tiny (≈0) even when raw-progress moved +194 → the two metrics
   DISAGREE and we don't know which to trust.
3. **Deterministic matched-seed paired eval**: same flow-noise seed per (state,chunk) across conditions
   (base vs plan vs adapted) → removes the ~2× stochastic swing. Good practice; we use it.
4. **Trained VLM-judge** (progress head): 0.76 acc on held-out pairs (motion-correlated genres); the
   intended reward/eval signal for games WITHOUT a RAM reward var (most obscure/test games).
5. We have NOT measured: held-out *games* (only held-out *levels* of the same game); completion / death-rate
   / success metrics; anything for FE/turn-based; whether the judge transfers to obscure (non-pretrained)
   games.

## The core eval questions (discuss these)
1. **What is the right reward/eval SIGNAL given only expert demos + sparse level-transition reward?** Is
   horizontal progress defensible for platformers? What replaces it for top-down (Minish) and turn-based
   (Fire Emblem)? Is "distance along the expert's trajectory" (demo-relative progress) better than raw
   screen_x — e.g. project the agent's state onto the nearest point of the expert's bk2 path and measure
   how far along it gets, with off-path penalty? Pitfalls?
2. **How to use the bk2 + multi-minute demos for a DENSE, less-sparse signal?** (subdivide into many start
   states; expert-relative progress; "does the agent reach the expert's state-T from state-0 within a
   budget?"; per-segment success.) Design this concretely.
3. **Genre-heterogeneous eval under ONE model** (the goal is a single generalist, NOT per-genre models):
   how do we get a comparable cross-game score when the reward axis differs per genre? Normalize how?
4. **Contamination**: these are popular games (VLM has a prior). How do we validate BOTH (a) the VLM-judge
   and (b) any rubric/LLM-judge approach against this prior? What obscure/homebrew held-out games (e.g.
   Witchblast) do we need, and what exactly does the obscure-game eval control for?
5. **Separating CAPACITY from DISTRIBUTION**: can we design an eval that distinguishes "the DiT actor lacks
   capacity" from "the actor is merely OOD for this game"? (e.g. in-distribution vs OOD game splits;
   demo-BC-fit headroom; oracle-plan-token search ceiling.)
6. **Anti-exploit & validity**: how to stop progress-hacking (run-into-death), and how to reconcile the
   progress-vs-shaped disagreement — which is the trustworthy headline metric?
7. **Judge validation**: given the judge was trained on ~few-hundred GT motion-correlated pairs AND the
   planner bridge was trained on SYNTHETIC plans, what's the minimal protocol to TRUST the judge as an eval
   signal (calibration vs GT, agreement with GPT-5.5, held-out-game transfer)?

## Files the duck can open
- Reward/env: `nitrogen/eval/envs/retro_rl_env.py` (screen_x reward, death/stuck shaping, step()),
  `nitrogen/eval/envs/gba_env.py` (reward_topdown, FE stub).
- Eval tools: `planner_poc/plan_graded_test.py` (the gate ladder), `planner_poc/best_of_k.py`,
  `planner_poc/latent_plan_search.py`, `planner_poc/demo_planner_ablation.py`,
  `planner_poc/held_out_eval.py` (progress + shaped), `planner_poc/eval_policy.py` (`_sample_chunk`).
- Judge: `planner_poc/judge_vlm_opsd.py`, `planner_poc/judge_calibrate.py`, data `docs/rl_data/*`.
- Demos: `docs/demos/demos/<Game>/<ts>/{demo.bk2,demo.npz,initial.state,meta.json,episode.mp4}`.
- Result JSONs: `docs/graded/*.json`.
- **Videos to inspect** (base vs plan vs learn, from demo start states, ~30s each):
  `docs/demo_videos/{smw,sonic,minish,fireemblem}/state*__{base,plan,learn}.mp4`. Extracted PNG frames for
  quick viewing: `docs/eval_review/frames/smw_state0_{base,plan,learn}_*.png` (more can be extracted via
  ffmpeg). These show qualitatively how base-DiT vs planner-conditioned rollouts differ.
- Prior eval/judge findings: session `files/{BENCH_RESULTS.md,RL_DESIGN_V2.md,MORNING_BRIEF_V2.md}`.
