# Morning brief — night of 2026-06-26 (for @namak-kun)

TL;DR (lead with the headline): **the DiT actor CAN be improved from emulator reward without collapsing** —
careful LoRA-only reward-weighted BC improved reward on BOTH the OOD Sonic actor (+1.5→+3.9, ~2.6×, gain
transfers to held-out start states) AND the actor-coherent TheXTech level20 (+9.85→+15.0, +52%, variance
tightened). This answers the pivotal "is the actor a dead end?" question with a clear preliminary YES. Also: built a real RL substrate (3 games with
ground-truth reward), ran a rigorous controlled plan-vs-null (plan HELPS — corrected an earlier wrong
finding), confirmed the raw 2B judge is unusable as reward but a cheap TRAINED judge is feasible, got a
sharp GPT-5.5 critique that reshaped the RL plan, and picked the first RL game. **Also built + validated
on a live env your "input format, revisited": the interleaved sub-chunk RL planner observation (K=8, A=2,
S=3) in THINK MODE (§10) — and found think mode is needed not just to hide analysis but to stop the 2B
echoing the action-summary format.** Details in
files/{RL_DESIGN,BENCH_RESULTS,GENRE_SPECTRUM_AND_RL,GROUNDED_PROMPT_FORMAT,RL_PROMPT_FORMAT}.md. Nothing committed (your rule).

## 0. SESSION 2 (evening, after you left for breakfast) — what's NEW (full detail: files/FEASIBILITY_JOINT_AND_JUDGE.md, files/DDPO_DESIGN.md + plan.md)
- **HEADLINE — RWBC works in a COMPETENCE BAND** (6 runs): filtered self-imitation BC amplifies EXISTING
  good behavior, so it only helps when the actor is mid-competence WITH headroom. Gains: TheXTech level20
  +9.85→+15.0, Sonic +1.5→+4.7, Solarus +16→+21. NO gain: SMW +0.08→−0.16 (too OOD, ~0 competence),
  TheXTech level8 +17.2→+15.8 (already near-ceiling), bonus1 (no headroom). → to push a too-OOD or
  near-ceiling actor you need actor-side EXPLORATION (DDPO), not self-imitation. **The DDPO build is the
  clear next actor-RL investment** (strongly evidenced now).
- **JUDGE (your OPSD direction) VALIDATED**: discriminative frozen-Qwen feature judge beats SigLIP —
  **0.761 vs 0.716 on 413 pairs/4 games (sonic+thextech+solarus+SMW), recall 0.79** (held on the small
  253-pair set too: 0.719 vs 0.688). No traces, think untouched. OPSD privileged→unprivileged distillation
  was FLAT on the larger set because the privileged teacher was trivial (hint=label sign); testing a
  NON-trivial privileged signal (future-frame look-ahead) next. judge_vlm_opsd.py.
- **ENV COVERAGE ++**: SMW (SNES, dense screen_x, works), Minish Cap (GBA top-down, movement reward, works
  + validated planner-in-loop = 3rd genre), Pokémon/Fire Emblem (GBA, load+save/load stubs — RPG/SRPG OOD),
  TheXTech graded suite level1→2→3→8 (+level20 ref). State-compat table + ingest_external_state.py for your
  manual external states (Genesis=GPGX1.7.5, SNES=snes9x snapshot-v9, GBA=mGBA — RetroArch RASTATE auto-unwrapped).
- **prev-plans knob: NEGATIVE** (naive "paste last 2 plans" LOWERS reward −7.45, causes plan oscillation not
  exploration). **deferral: untestable** on current substrate (needs save/load + competent actor together).
- **JOINT TRAINING feasibility**: simultaneous collapses; ALTERNATING two-timescale (LoRA ↔ plan-head) is
  the feasible route — masked-null preserved by construction in the plan-head phase. Prototype running.
- **BUGFIX**: stable-retro cores segfault on repeated construction in a CUDA process → refactored RWBC to
  reuse ONE env (reset per episode). Affects any emulator RL/eval loop.
- **DDPO PREMISE VALIDATED, but the cheap shortcut FAILS → real PG needed (de-risks + scopes the next build)**:
  added late-step SDE noise to the flow sampler (eval_policy._sample_chunk noise_sigma). (1) On a competent
  actor it EXPLORES and best-of-8 beats greedy Δ+0.085 (finds +reward chunks greedy misses) → diffusion-PG
  has signal. (2) BUT exploration-augmented self-imitation BC (--explore-sigma in RWBC) HURT vs plain greedy
  (Sonic Δ+0.22 vs +1.54 control, identical settings) — BC imitates lucky-noise targets. CONCLUSION: the
  cheap "DDPO-lite via BC" shortcut doesn't work; REAL policy-gradient (reward-weighted by the action
  log-prob) is necessary. The sampler foundation is built; the crux to pair on is the per-step log-prob math
  (files/DDPO_DESIGN.md).


## 1. CORRECTION (important): plan-conditioning HELPS — I was wrong last night
Earlier I wrote "plan hurts platformer/topdown". That was a metric artifact (mismatched episode lengths +
a per-cycle metric penalizing a stuck tail). RETRACTED. Truth (you were right):
- Without a plan the base DiT is INERT (stick pinned neutral, ~0 movement).
- With a plan it acts: length-matched, thextech moved 3× further (566 vs 186px).
- RIGOROUS confirmation: same-start, ground-truth reward (screen_x) on Sonic, 3 seeds → NULL +0.00
  (Sonic doesn't move), PLAN +1.28. The plan ACTIVATES the inert DiT. (Deferral idea is NOT supported by
  this — deferring = idling the DiT on platformers. De-emphasized.)

## 2. RL SUBSTRATE — BUILT + VERIFIED (this solves your 3 RL blockers)
stable-retro ships 1007 game integrations (reward vars + done + states). Our ROMs don't SHA-match the
shipped states, but the RAM reward addresses work and we generate our own start states.
- nitrogen/eval/envs/retro_rl_env.py — Sonic2 (Genesis): dense screen_x reward, FRAME-EXACT save/load
  (→ save-state GRPO), our-own start state. (anti-exploit reward: death+stuck penalties.)
- nitrogen/eval/envs/proc_rl_env.py — TheXTech (Δx) + Solarus (move+map) via the de-forked semantic state
  (env-builder subagent BUILT both engines: TheXTech RelWithDebInfo + Solarus/ZSDX). PPO-style (no save/load).
- So: common start state ✅, reward ✅, frame-exact verify ✅ — on REAL games, no IDM needed.

## 3. FIRST RL GAME = TheXTech (measured), not Sonic
rank_rl_games.py — btn_s600+plan GT-reward: thextech +15.75 (0% stuck) ≫ solarus +0.76 (67% stuck) ≫
sonic +0.11 (75% stuck). The DiT is coherent on TheXTech (Mario platformer ~ its pretraining); OOD on
Sonic. Start RL on TheXTech. (Tradeoff: TheXTech is proc → PPO/reward-weighted-BC, no save-state GRPO;
Sonic has the GRPO substrate but the actor is OOD there.)

## 4. GPT-5.5 RUBBER-DUCK reshaped the RL plan (files/RL_DESIGN.md) — key corrections
- The ACTOR (DiT) is the bottleneck, and "GRPO on the plan" does NOT fix it. The diffusion action sampler
  needs a real policy-gradient path → DDPO/DPPO-style or REWARD-WEIGHTED BC, NOT text-GRPO. Plan-RL and
  actor-RL are TWO different objectives; don't conflate.
- STAGE training: actor-LoRA warmup (fixed plans) → planner-RL → alternating → joint. With ablations.
- THE critical experiment: "can DiT-LoRA learn new motor skills from reward without collapsing?" —
  everything depends on this. Run it FIRST (reward-weighted BC on TheXTech, held-out start states).
- THINK MODE + in-prompt experience-memory = ABLATIONS, not pillars (do controlled exploration first:
  temperature, entropy bonus, plan-diversity, diffusion action noise). THINK MODE is currently OFF.
- OPSD (your dart): only helps if the PLANNER is the bottleneck (it isn't). DIAGNOSTIC RUN TONIGHT
  (planner_poc/opsd_diag.py, Sonic): fixed hand-written "good" plan ≈ Qwen generated plan (+1.25 vs +1.13,
  within noise), both ≫ null (−0.60). → the plan's value is BINARY (activate the inert actor), NOT graded
  (better plans don't raise return). => OPSD/GPT-5.5-plan-distillation is MOOT here (planner not the
  bottleneck; the actor can't exploit better plans). (GPT-5.5-teacher also can't token-KL — diff tokenizer.)
- verl/prime-rl fit the PLANNER (text logprobs), NOT the DiT diffusion actor (custom loop needed).

## 5. VLM-as-judge — NOT reward-grade (confirmed); a cheap TRAINED judge is feasible
Raw Qwen-2B judge over-credits + is INVERTED on hard genres (rated a flailing Sokoban agent 8/8 success;
shmup 0.94-vs-0 truth). GPT-5.5≈GT (0.94). I built collect_rl_trajectories.py (GT-labeled before/after
pairs from the reward envs) + judge_calibrate.py. RESULT: a CHEAP trained linear probe on frozen SigLIP
features beats the raw judge — 0.78 acc (platformer+Sonic, 169 pairs) / 0.656 (all 3 incl topdown solarus,
253 pairs) vs raw judge 0.56 and majority 0.50. So "the judge needs to be trained" is confirmed + tractable;
works best where progress correlates with visible MOTION (platformers), weaker on topdown exploration. For
RL reward: use GT state where available; a trained probe is the fallback for motion-correlated games.
(Gemini dropped — both versions fail the subagent file-write.)

## 5b. THE CRITICAL EXPERIMENT (duck's pivotal test) — CAN THE ACTOR LEARN FROM REWARD? → PRELIMINARY YES
Built planner_poc/rwbc_actor_adapt.py (roll out → keep top-return chunks → BC the DiT-LoRA on
(frame,plan→full 18-step chunk) via the model's flow-matching loss → re-eval reward).
- **Naive (LoRA+plan-head, lr1e-4, tiny data): COLLAPSE** +15.77→+1.58 (the duck's warning).
- **Careful (LoRA-ONLY=0.52M params, lr2e-5, more data):**
  - **Sonic (OOD actor, has headroom): ACTOR ADAPTS, ROBUSTLY + keeps climbing.** 150 steps +1.32→+2.80;
    300 steps +1.50→+3.88; **500 steps +2.14→+4.73 (+121%), trajectory +2.17→+3.27→+4.73, NOT plateaued**
    (several seeds converge on +5.14 = reliably reaching the next obstacle). held-out start states TRANSFER
    (+2.85→+3.47). → cheap self-imitation BC has substantial headroom before needing diffusion-PG.
  - **TheXTech level20 (COHERENT actor + headroom = the duck's ideal first RL game): BEST CASE —
    +9.85→+15.0 (+52%), variance tightened (baseline 5.7–13.1 → post 13.9–15.6).** This is the cleanest
    actor-adaptation result: when the game is well-chosen (actor coherent + headroom), the gain is large
    and consistent. **CONFIRMED with a longer 500-step run: +10.2→+13.6 (+33%); per-state spread BOTH
    lifted AND tightened (baseline [13.5,8.6,6.2,12.5] min 6.2 → post [15.7,12.9,10.8,15.0] min 10.8) —
    every state improved, none collapsed.** METHODOLOGY NOTE: the run's mid-evals (49:+9.3, 99:+8.3) used
    only 2 episodes and read as a DOWNWARD drift — pure noise; the 4-episode final showed clear +33%. Trust
    the full-eps final, not 2-ep mid-checks, on high-variance TheXTech. Longer lr2e-5 on the small 45-chunk
    keep set did NOT overfit/degrade.
  - TheXTech bonus1 (short level, at ceiling): flat +15.75 (no headroom — confirms the headroom need).
  - **Solarus/ZSDX (top-down zelda, CROSS-GENRE check, 200 steps): ACTOR ADAPTS +16.2→+21.2 (Δ=+5.0).**
    Third genre (after platformer Sonic + platformer TheXTech) where LoRA-only RWBC lifts reward → the
    actor-adaptation result is NOT platformer-specific. CAVEAT: Solarus reward is HIGH-VARIANCE (+5
    per map-change event); post per-state spread [46.0, 18.6, 2.6, 17.5] — the +46 is a multi-map-change
    outlier, so read this as "positive, noisy" not a clean +31%. Still: 3/3 genres adapt, none collapsed.
  - Plan-temperature exploration ≈ greedy (+3.85 vs +3.88) → the actor's execution is the lever, not plan
    diversity. OPSD diag: fixed-good plan ≈ Qwen plan → planner not the bottleneck.
- **TAKEAWAY:** the DiT actor IS adaptable from emulator reward WITHOUT collapse, given the duck's
  anti-collapse constraints (LoRA-only, low LR). Across **3 genres / 3 games (Sonic, TheXTech, Solarus)**
  reward-weighted self-imitation BC (0.52M params) lifts reward with NO collapse; even the cheapest run
  (150 steps) more-than-doubled reward on the OOD Sonic actor. This flips the pivotal
  question from "is the actor a dead end?" to "scale it up."
- CAVEATS (honest): self-imitation BC → SHARPENS existing behavior (can't discover NEW skills — need
  exploration [generate_plan temperature added] + real diffusion-PG for that); Sonic baseline low (still
  far from competent). HELD-OUT START-STATE TEST (planner_poc/rwbc_heldout.py): trained from ONE start
  state, the gain TRANSFERS to held-out level positions — held-out mean +2.85→+3.47 (3/4 states up, one
  −1.14 small-data variance) → NOT mere memorization. Methodology: actor-learnability needs coherence AND
  HEADROOM (TheXTech ceiling-capped → no signal; Sonic has headroom).

## 6. New reusable code (planner_poc/ + nitrogen/eval/envs/), all compile + smoke-tested
retro_rl_env.py, proc_rl_env.py, rl_eval_plan_vs_null.py, rank_rl_games.py, collect_rl_trajectories.py,
judge_calibrate.py, bench_rollout.py, build_judge_manifest_from_rollout.py, bench_judge_vlm.py,
bench_aggregate.py, merge_slim_to_full.py (fixed a LoRA-remap bug), generate_grounded_plan.py,
**rl_planner_prompt.py + rl_rollout_demo.py (the RL-time planner OBSERVATION format + live-env demo — see §10)**.

## Open decisions for you (morning brainstorm)
- First RL experiment: reward-weighted BC on TheXTech (actor-adaptation test) — agree?
- TheXTech (PPO, actor-coherent) vs Sonic (GRPO substrate, actor-OOD) for the first loop?
- Is the plan-token interface enough, or does the actor need more than LoRA (the fundamental risk the duck
  raised: plan-tokens may be too low-bandwidth to repair an OOD motor policy)?
- Reward shaping for TheXTech (it menu-strands at the level edge — fix the env's start level / objective).

## 7. Verified run commands (all use this prefix on a free GPU)
```bash
RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
PY=.venv/bin/python   # repo root; CUDA_VISIBLE_DEVICES=<0-3>
```
- Merge slim->full ckpt:  `$RUN $PY planner_poc/merge_slim_to_full.py ckpts/handoff_zips/btn_s600.pt ckpts/btn_s600_full.pt`
- RL env smoke (Sonic):   `$RUN $PY nitrogen/eval/envs/retro_rl_env.py`   (proc: `$RUN $PY nitrogen/eval/envs/proc_rl_env.py thextech`)
- Plan-vs-null (GT reward, same start): `$RUN $PY planner_poc/rl_eval_plan_vs_null.py --chunks 14 --seeds 3`
- Rank RL games:          `$RUN $PY planner_poc/rank_rl_games.py --games thextech sonic solarus_zelda`
- Collect GT-labeled traj:`$RUN $PY planner_poc/collect_rl_trajectories.py --env sonic --episodes 8 --out docs/rl_data/sonic`
- Train cheap judge:      `$RUN $PY planner_poc/judge_calibrate.py --data-dirs docs/rl_data/thextech docs/rl_data/sonic`
- ACTOR-ADAPT (the result): `$RUN $PY planner_poc/rwbc_actor_adapt.py --env sonic --collect-eps 16 --steps 300 --lr 2e-5 --lora-only`
- Held-out transfer:      `$RUN $PY planner_poc/rwbc_heldout.py --n-heldout 3 --steps 200`
- Bench judges (committed rollouts): build_judge_manifest_from_rollout.py -> bench_judge_vlm.py -> bench_aggregate.py
NOTE: stable-retro = 1 emulator/process (close env before opening another). Proc envs (thextech/solarus)
relaunch per reset (slow); Sonic (emulator) is fast + frame-exact (save/load -> GRPO). generate_plan now
takes temperature>0 for exploration (added this night). Env builds: .nitrogen-env-build/TheXTech, tmp/{solarus,zsdx}.

## 8. Watchdog
A detached process writes files/WATCHDOG_STOP after 8h (~20:30 PDT). I keep working until then.

## 9. NEXT STEPS (prioritized, for the brainstorm)
Given the actor IS adaptable (the night's headline), the path is clearer:
1. **Scale the actor-adaptation** (it was still climbing at 300 steps): more data + steps + LoRA rank sweep;
   find the plateau; track SUCCESS metrics (distance, death rate) not just reward. Add KL-to-base anchor.
2. **Exploration: plan-temperature does NOT help (tested tonight).** plan-temp=1.0 RWBC reached +3.848 vs
   greedy +3.878 — equal. Plan-SIDE exploration adds nothing to actor self-imitation BC, because the gain
   lives in the ACTOR's execution (LoRA), not plan diversity. → for NEW skills, explore ACTOR-side
   (diffusion action noise on the chunk sampler) + real diffusion-PG, NOT plan temperature. (Reinforces:
   the actor is the lever, not the planner — consistent with the failure taxonomy.)
3. **Real diffusion-policy RL** (DDPO/DPPO) once self-imitation BC plateaus — to push past the policy's
   current-best behaviors. This is the bigger build; self-imitation BC is the warm-up.
4. **Give TheXTech headroom** (it's the actor-coherent game but ceiling-caps on the short bonus1 level):
   VALIDATED tonight — `worlds/the first adventure/level20.lvlx` has headroom (policy drives x 260→827 then
   STUCK at an obstacle = a learnable challenge). Use THEXTECH_LEVEL env var (proc_rl_env supports it) to
   run RWBC there (coherent actor + headroom = best-case test; running tonight, see /tmp/rwbc_thextech_hr.log).
5. **Planner RL** (separate objective): once the actor is stronger, GRPO/PPO on plan tokens (prime-rl/verl)
   with the calibrated judge / GT reward. Then alternating planner/actor; joint last.
6. **Judge**: scale judge_calibrate (more data, MLP head, per-genre); use GT-state where available.
7. **Stabilize/ablate**: think-mode A/B, experience-memory A/B, plan-stability (replan-every N + hysteresis)
   — but only AFTER the basic actor+reward loop is solid (the duck: controlled exploration before memory).

## Honest bottom line
The project is in better shape than last night's (wrong) verdict suggested: plan-conditioning genuinely
helps (activates the inert DiT), the actor is adaptable from reward without collapse, the reward problem is
solved via emulator GT-state (+ a calibratable cheap judge), and the RL substrate is built and verified.
The main open risk (the duck's): self-imitation BC only sharpens — discovering genuinely NEW motor skills
needs exploration + real diffusion-PG, which is the next real build. But the foundation is solid.

## 10. RL PLANNER PROMPT FORMAT — built + verified tonight (your "input format, revisited")
`planner_poc/rl_planner_prompt.py` implements the exact closed-loop observation you specified, as the
planner's RL-time interface. Details in files/RL_PROMPT_FORMAT.md. Summary:
- **Layout**: [SYSTEM] intro+task / game-info / controls ; [USER] f0, then per chunk the SUB-CHUNK
  interleave s{c}_1,f{c}_1, s{c}_2,f{c}_2, s{c}_3,f{c}_3 — i.e. inputs-then-resulting-frame at the
  actor's intra-chunk sample rate (S=3 → subchunks of 6 of the 18 steps), spanning A=2 chunks so the
  prompt ends at f2_3 (where the planner is invoked). Then OUTPUT INSTRUCTIONS. Matches your spec exactly.
- **Config baked**: K=8 plan tokens, A=2 (replan every 2 chunks), S=3 intra-rate. `RLPlannerConfig`.
- **THINK MODE (your crucial ask)**: generates with the chat template's enable_thinking=True so analysis
  lands in <think>…</think> and the PLAN OUTPUT STAYS CLEAN. Verified: think trace holds the full
  per-subchunk analysis; plan = one grounded sentence ("Move RIGHT and JUMP …"). Qwen3.5-2B is verbose
  and won't close </think> in a small budget, so I added **budget forcing** (think_budget → force-close
  </think> with a neutral "Plan:" seed → bounded plan_budget) — guarantees clean separation AND bounds
  planner compute per RL call. `--no-think` is the ablation (empty think block, straight to plan).
- **Two ablation knobs you flagged**: `include_prev_plans` (append last ≤2 plans to incentivize
  exploration — OFF by default, ablatable) and `allow_defer` (lets the planner emit the sentinel
  "NO GUIDANCE NEEDED" → caller maps to null/masked plan = base-DiT-exact). The defer sentinel is the
  **deferral-via-output** mechanism you preferred over model-controlled CFG weight — SCAFFOLDED, but it
  needs an RL signal to learn WHEN to defer (you noted "we lack signal"); right now it's prompt-only.
  **PROBED tonight (0/5 zero-shot defer rate on Sonic): the frozen 2B NEVER defers on its own — it always
  produces a plan even in smooth System-1 moments.** So deferral is NOT elicitable zero-shot; it needs a
  TRAINING signal. GOOD NEWS — we already have the signal source: `rl_eval_plan_vs_null.py` measures
  per-situation plan-vs-null GT reward, so "defer is correct ⇔ null-plan reward ≈ plan reward" is a
  ready-made label. Train the defer token by rewarding it exactly when planning doesn't beat the null
  (System-1 moments). This is the concrete path for your deferral idea (do it after the actor+reward loop).
- **API for the RL loop**: `build_rl_messages(f0, chunks, genre=, game_info=, cfg=, include_prev_plans=,
  allow_defer=)` → (system, content, images, render); `generate_rl_plan(planner, system, content,
  images, device, enable_thinking=, think_budget=, plan_budget=, temperature=)` →
  {think, plan, raw, forced}. temperature>0 = plan sampling (exploration).
- Run: `$RUN $PY planner_poc/rl_planner_prompt.py --demo --genre platformer` (think demo);
  add `--print-prompt-only` (assemble only), `--no-think`/`--include-prev-plans`/`--allow-defer` (ablations).
- **REAL-ENV VALIDATION** (`planner_poc/rl_rollout_demo.py`): drives the REAL actor on a LIVE env, steps
  each 18-row chunk as S=3 sub-chunks of 6 capturing a frame after each, invokes the planner every A=2
  chunks via this format in think mode. Verified on Sonic: the planner GROUNDS ON REAL PIXELS (correctly
  reads "Sonic … ground hazards … hanging enemy … floating rings"), think stays in the trace, plans are
  sensible ("Move RIGHT to clear the ground hazards"), and GT reward climbs +4.1 → +10.6 over 2 cycles.
  `$RUN CUDA_VISIBLE_DEVICES=1 $PY planner_poc/rl_rollout_demo.py --env sonic --cycles 3`.
  Also validated on **TheXTech** (proc env, coherent actor): planner correctly read the real menu screen
  ("The screen shows a black menu") and planned "Restart the level" — real-pixel grounding on a 2nd env/genre.
- **THINK-MODE A/B (concrete evidence FOR think mode with this format)**: on identical real Sonic
  observations, WITHOUT think the 2B frequently DEGENERATES into echoing the interleaved action-summary
  style — e.g. plan = "RIGHT 6, DOWN 6, ACCELERATE/FIRE 6" / "RIGHT 1, UP 5, ATTACK 4, …" (copying the
  s{c}_{s} input format) — whereas WITH think it reliably emits a clean grounded plan ("Move RIGHT to
  avoid the floating orange ball and continue advancing"). The think block gives the model room to process
  the action trace so the PLAN doesn't collapse into action-token echoing. This validates your instinct
  ("we work in think mode … lest analysis dilutes the plan") AND adds a second reason: think mode prevents
  format-echo when the prompt interleaves action summaries. → keep think mode ON for the RL planner.
- **OPEN (morning)**: (a) wire this format into the rollout/RWBC loop so the planner actually consumes
  sub-chunk frames at RL time (currently RWBC uses the simpler generate_plan frame window); (b) the 2B's
  plans are coherent but generic — think-mode quality vs the stronger judges is worth an A/B; (c) decide
  the prev-plans + defer ablations once the actor+reward loop is solid (the duck: exploration before memory).
