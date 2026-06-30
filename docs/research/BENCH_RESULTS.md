# Benchmark results — btn_s600 across the genre spectrum (overnight 2026-06-26)

Checkpoint: **btn_s600** (merged → ckpts/btn_s600_full.pt; LoRA-remap bug fixed). Backbone Qwen3.5-2B,
K=8, A=2, cfg=8. Goal: feasibility of VLM-as-RL-reward + planner/actor/both failure taxonomy by genre.

## Pipeline (all built + de-risked this session)
1. `planner_poc/merge_slim_to_full.py` — slim handoff → loadable full ckpt (fixes LoRA `.base.` remap). ✓
2. `planner_poc/bench_rollout.py` — closed-loop btn_s600 rollout → per-cycle manifest (before/after
   frames, plan, executed-action summary, GT state). ✓ compiles
3. `planner_poc/bench_judge_vlm.py` — frozen Qwen3.5-2B judge → {plan_sensible,progress,plan_followed,
   failure,reason} per cycle. ✓ validated end-to-end on a synthetic before/after scene.
4. Strong-model judges — GPT-5.5 + Gemini-3.5-Flash SUBAGENTS view the same frames, same schema.
5. `planner_poc/bench_aggregate.py` — VLM-vs-strong agreement + failure taxonomy by genre. ✓ compiles

## Setup fixes found this session
- torchvision was MISSING (Qwen3VLProcessor dep) → installed 0.27.1+cu130.
- flash-linear-attention installed (Qwen3.5 Gated-DeltaNet); causal-conv1d fast path still warns
  (CUDA build skipped) — torch fallback works.
- merge LoRA-remap bug: `to_q.weight`→`to_q.base.weight`, 16 attn proj were loading random; fixed.

## Sanity (fake-frame probe, btn_s600_full)
- NULL plan sane: accel=1.0, neutral stick (proves base weights loaded). Jump steering crisp (0→1.0).
- Left/right inverted+saturated on a FAKE gradient frame @cfg8 — known-weak axis; characterize on real frames.

## Status: PIVOT — all 38 proc envs FAIL on this fresh box (game binaries/assets never installed here)
boot_results_strict.json (scout): 38/38 FAILS. Causes: apt game pkgs missing (supertuxkart/freedink/
mednafen/opentyrian/wesnoth → rc=127 not-found); built artifacts gone (Godot/jar/.exe/.love); cloned
game repos absent (/tmp/bab-be-u, /tmp/mmx8, /tmp/witchblast); mGBA ROMs missing (blind_jump,
notebook_adventure); missing dep evdev (cavestory); sokoban needs cavepacker binary. The dev box had
all these; this box has none. (Concretely demonstrates the user's "no start state / no install" problem.)

### PIVOT PLAN (unblocked)
1. EMULATOR path (CONFIRMED working: SNES stable-retro, frame-exact): ROMs in Game data/ (Super Mario
   All-Stars, Mega Man X, Final Fantasy III, ...). Wire SNES env into bench_rollout. Boots to MENUS (no
   save states) → script past menu OR accept early menu cycles. Gives platformer/action coverage NOW.
   GBA also works via stable-retro mgba core (Minish Cap, Pokemon Emerald, Fire Emblem) — topdown/RPG.
2. DELEGATE game-install to a subagent: apt quick-wins (supertuxkart=racing, freedink=topdown,
   opentyrian=shmup, wesnoth=strategy, mednafen=NES) + report which now boot via bench. Keeps proc-env
   coverage. Keep MY context clean.
3. Run full pipeline (rollout→3 judges→aggregate) on whatever works. Even 4-6 games across genres
   delivers all 4 goals (bench, judge-agreement, failure taxonomy, feasibility).
NOTE: leaked Xvfb :80-:97 from scout — clean by PID after scout finishes, before sweeps.

## Results (filled as rollouts + judging complete)

### Data source: REAL btn_s600 rollouts from docs/horizon_play_grounded/ (dev-box committed)
All 38 proc envs are uninstalled here, so instead of fresh rollouts we judge the COMMITTED annotated
rollouts: thextech/sdlpop/solarus_zelda/stk at A=2 (+A=4 available). Frames extracted from play.mp4,
cropped to remove the 120px overlay; per-cycle plan + executed actions + GT state parsed from actions.txt
(planner_poc/build_judge_manifest_from_rollout.py). 16 cycles/game. GT progress anchor (Δx / Δpos) on
thextech + solarus (16/16 each); stk + sdlpop have no GT state (state={}) -> visual-judge only.

### JUDGES: GPT-5.5 (strong) + Qwen3.5-2B (cheap) + ground truth. GEMINI DROPPED.
Gemini-3.5-flash AND gemini-3.1-pro both fail the multi-image file-writing subagent task ("completed,
no response", file never written) — a quick single-image probe worked but the 16-cycle judge does not.
Per user: drop Gemini. Comparison is Qwen-vs-GPT5.5-vs-GT.

### HEADLINE FINDING (thextech, GT-anchored, 16 cycles) — VLM-judge-as-RL-reward feasibility
| judge | progress vs GT exact | within1 | note |
|---|---|---|---|
| **GPT-5.5** | **0.94** | 1.00 | tracks real Δx almost perfectly; flagged ALL stuck cycles (8-15) as -1 |
| **Qwen-2B** | **0.38** | 0.94 | systematically OPTIMISTIC: never says -1, false-positive progress 3/16 |
- Qwen would REWARD A STUCK AGENT: cyc 12 it said progress=1/failure=none while agent was stuck (gt=-1).
- Qwen vs GPT5.5: progress-exact 7/16; failure binary(any-vs-none) 13/16 — agree on moving cycles,
  diverge on stuck ones (Qwen too lenient).
- **=> The cheap Qwen-2B judge is NOT reliable enough as an RL reward as-is.** Options: (a) strong-model
  reward (GPT-5.5-grade, expensive), (b) CALIBRATE/distill the cheap judge to GT/GPT-5.5 (user's "perfect
  the judge" idea — YouTube data), (c) use GT RAM state where available (thextech x-pos), which is exact.

### FAILURE STORY (thextech) — planner/actor/both, GT-confirmed
Cycles 0-7: real progress (x 235->~430), planner+actor both working. Cycles 8-15: STUCK (x flat, then
in_menu) while the planner emits "jump to the right" VERBATIM repeatedly (no correction/exploration) and
the DiT can't escape. GPT-5.5 attributes these to "both" (8x), never "planner"-alone (0x). This is
exactly the user's CORRECTIVE-PLANNING gap: no re-planning/exploration when stuck. Plan-repetition +
actor-stuck = compounding failure; ends in menu-stranding (the known thextech failure).

## VLM-judge vs strong-model agreement (RL-reward feasibility)
FINAL AGGREGATE across 10 games / 144 dual-judged cycles (docs/bench/AGGREGATE.json):
**Ground-truth anchor (objective, n=32, thextech+solarus Δstate):**
| judge | progress exact vs GT | within1 | false-positive-progress |
|---|---|---|---|
| **GPT-5.5** | **0.94** | 1.00 | **0.06** |
| **Qwen-2B** | **0.56** | 0.97 | **0.22** |

**Mean progress by genre — Qwen vs GPT-5.5(≈truth):**
| genre | Qwen | GPT-5.5 | gap |
|---|---|---|---|
| shmup | 0.94 | **0.00** | judge says 94% progress, truth 0% |
| puzzle(sokoban) | 1.00 | **0.00** | judge says 100%, truth 0% (Qwen rated ALL 8 cycles success; GPT-5.5 ALL fail) |
| platformer | 0.85 | 0.17 | 5× over-credit |
| topdown | 0.81 | 0.56 | |
| racing | 1.00 | 1.00 | only genre they agree (kart trivially moves forward) |
**Inter-judge (n=144):** progress exact 0.46, plan_sensible 0.37, plan_followed 0.38, failure_match 0.38.
=> The Qwen-2B judge is UNUSABLE as RL reward — worst exactly where the agent fails most (shmup/puzzle),
where it would hand MAXIMAL reward to a totally failing agent. GPT-5.5 ≈ ground truth (0.94).

## Failure taxonomy by genre
GPT-5.5 attribution, all 10 games (144 cycles):
| genre | none | planner | actor | both | read |
|---|---|---|---|---|---|
| **racing** (stk) | 11 | **5** | 0 | 0 | DiT races fine; plan DISTRACTS → PLANNER failures (only genre) |
| **topdown** (solarus/witchblast) | 15 | 1 | **14** | 2 | half ok, half DiT-can't-execute |
| **platformer** (thextech/sdlpop/blobwars/castlevania/pekka) | 27 | 0 | **36** | 9 | ACTOR-dominated + stuck-loops |
| **shmup** (chromium_bsu) | 0 | 0 | **16** | 0 | 100% ACTOR (DiT can't dodge/aim precisely) |
| **puzzle** (sokoban) | 0 | 0 | 2 | **6** | 100% FAIL; DiT mashes nonsense buttons, no puzzle prior |
- **ACTOR (DiT) is the dominant bottleneck across almost ALL genres** (planner failures only in racing,
  where the DiT is already competent). Strongly validates the user's "DiT isn't good → joint training".
- Plan-REPETITION (exploration deficit, consecutive-near-identical Qwen plans /15): chromium_bsu 15/15,
  castlevania ~10/15, witchblast 8/15, thextech 5/15, sdlpop 0/15 — worst plan-stagnation at the
  System-1-heavy/shmup end.

## Base (null-plan) vs A2 (plan-conditioned) — CORRECTED 2026-06-26 (earlier conclusion was WRONG)
**CORRECTION (user pushback + re-analysis):** an earlier version claimed "plan HURTS platformer/topdown".
That was a METRIC ARTIFACT and is RETRACTED. Evidence the plan HELPS:
- Base DiT is essentially IDLE without a plan: mean stick-deflection-from-neutral sdlpop **0.014**,
  thextech 0.061, solarus 0.037 (stick pinned ~0.5, ~0.3 buttons). It barely acts.
- With a plan the DiT ACTS: deflection sdlpop 0.417 / thextech 0.445 / solarus 0.366, 1.6-2.8 buttons.
- Length-MATCHED displacement (thextech, first 40 steps): base net Δx **186px** vs plan **566px** → the
  plan moves the agent ~3× further. (sdlpop README gif = the canonical example: base idle, plan moves it.)
- Why the earlier "plan hurts" was wrong: base episodes were 40 steps, A2 episodes 120 steps; A2's longer
  tail included a stuck/menu-stranding phase, and the PER-CYCLE progress metric averaged that tail down to
  -0.06 while base's short run stayed positive. Comparing mismatched-length episodes by per-cycle mean was
  invalid. The DiT does NOT do nothing-worse with a plan — it does MUCH MORE, then eventually gets stuck.
- CONSEQUENCE for DEFERRAL: WEAKENED/retracted as motivated by this data. Since the plan is what ACTIVATES
  the idle DiT, deferring to null = idle = WORSE on platformers/topdown. Deferral is only neutral-to-good
  where base is already competent (racing). So whole-genre deferral is NOT supported here; if deferral has
  value it is WITHIN-episode (suppress a specific bad/looping plan), not "defer on System-1-heavy games".
- WHAT SURVIVES: (1) the VLM-judge unreliability (GT-anchored, solid). (2) The real failure mode is
  "plan activates DiT → initial progress → DiT/agent GETS STUCK" (thextech: progress to x=424 then stuck→
  menu). That is a CORRECTIVE-PLANNING + actor-robustness problem, not "plan hurts". (3) User's lived
  experience CONFIRMED: plan-conditioning is genuinely useful (esp. platformers); it's the only thing that
  makes the markovian DiT do anything.

## Base vs A2 raw GPT-5.5 per-cycle progress (kept for reference; do NOT aggregate across mismatched lengths)
| genre | base | A2 | NOTE |
|---|---|---|---|
| racing (stk) | 0.75 | 1.00 | plan adds steering |
| platformer (thextech) | 0.17 | -0.06 | ARTIFACT: A2 has 60 cyc incl stuck tail vs base 20; by displacement plan WINS 3x |
| topdown (solarus) | 0.92 | 0.50 | same artifact (episode-length + stuck tail) |
| shmup (chromium) | 0.00 | 0.00 | both fail (DiT can't dodge/aim) |

## Verdict
0. **RL SUBSTRATE BUILT + CONTROLLED RESULT (2026-06-26 night, the definitive test):**
   - nitrogen/eval/envs/retro_rl_env.py: stable-retro Sonic2(Genesis) RL env via OUR ROM — dense
     screen_x reward, frame-exact save/load (GRPO substrate), our-own start states. Solves the user's 3
     RL blockers (start/reward/verify) on a real game, no IDM needed. 1007 integrations available.
   - planner_poc/rl_eval_plan_vs_null.py: closed-loop btn_s600 on Sonic, SAME start, GT reward, 3 seeds.
     **NULL(base DiT) reward +0.00, screen_x 0->0 every seed (base is INERT); PLAN +1.28, screen_x ->58-167.
     PLAN - NULL = +1.28 -> plan HELPS** (rigorous: same-start + ground-truth reward, NOT a judge). This is
     the proper controlled version of the base-vs-plan test and CONFIRMS the corrected finding: base DiT
     does nothing, the plan activates it. (Abs reward small -> policy OOD on Sonic -> large RL headroom.)
1. **VLM-judge feasibility (PRIMARY):** the Qwen-2B judge is NOT RL-reward-grade — it over-credits
   progress everywhere and is INVERTED from truth on the hardest genres (shmup 0.94-vs-0, puzzle
   1.0-vs-0; rated a flailing sokoban agent 8/8 success). Training against it would optimize "look busy",
   not "make progress". GPT-5.5 IS reward-grade (0.94 vs GT) but expensive. Reward options, ranked:
   (a) GROUND-TRUTH RAM state where available (exact, free, e.g. thextech x-pos / solarus pos+map) —
   STRONGLY PREFERRED for the first RL loop; (b) calibrate/distill a cheap judge toward GT/GPT-5.5
   (user's "perfect the judge"; YouTube data could help); (c) strong-model reward on a small RL budget.
2. **DiT (actor) is the first-order problem (user's flag, CONFIRMED):** actor-attributed failures
   dominate every genre except racing; on shmup/puzzle the DiT fails 100%. Plan-only RL CANNOT fix
   execution — joint DiT-LoRA training is required for the middle+hard genres. Plan-only RL is
   defensible mainly where the DiT is already competent (racing) — but there the plan is low-value.
3. **Where RL helps:** the MIDDLE (platformer/topdown) is the only band where BOTH a plan matters AND
   the system partly works — best RL target, but needs joint plan+DiT training + a trustworthy reward.
   System-1-heavy (racing/shmup): plan is noise/stagnant. Puzzles: DiT unusable.
4. **Exploration deficit measured:** consecutive plan repetition up to 15/15 (shmup) — confirms the
   user's "not curious / no corrective planning" worry. Inject exploration (diverse sampling + in-prompt
   experience memory); do NOT expect it from GRPO (which sharpens, not explores).

## Deliverables / reusable artifacts (committed-rollout + live-env bench pipeline)
- planner_poc/merge_slim_to_full.py — slim handoff ckpt → loadable full (fixes LoRA .base remap)
- planner_poc/bench_rollout.py — live-env closed-loop rollout → manifest (works on apt-installed envs)
- planner_poc/build_judge_manifest_from_rollout.py — committed play.mp4+actions.txt → judge manifest
  (+ GT anchor) WITHOUT any env install
- planner_poc/bench_judge_vlm.py — Qwen VLM-judge; planner_poc/bench_aggregate.py — agreement + GT
  anchor + taxonomy
- planner_poc/generate_grounded_plan.py — the reconstructed interleaved grounded-plan teacher prompt
- Data: docs/bench/<game>/{manifest.json,judge_vlm.json,judge_gpt55.json,frames/}, docs/bench/AGGREGATE.json
- Strong-model judging = GPT-5.5 subagents (Gemini dropped: both 3.5-flash & 3.1-pro fail the file-write
  subagent task). 10 games judged: stk, thextech, sdlpop, solarus, blobwars, castlevania, chromium_bsu,
  witchblast (committed) + sokoban, pekka_kana_2 (fresh live-env).

---

## DEMO-STATE PLANNER ABLATION (2026-06-27) — base vs plan vs learn, on GT reward

Quantitative ablation the user asked for, run from the human-gold demo START STATES with the
**corrected per-game prompts** (planner_poc/game_planner.py) + the new `<learnings>`/`<plan>` context mode.
Deterministic matched per-(state,chunk) latent seeds. btn_s600_full, cfg=8, A=2, 16 chunks, 2 seeds.
Tool: planner_poc/demo_planner_ablation.py. Reward = screen_x advance (smw/sonic), top-down step-reward (minish).

| game   | base  | plan   | learn | Δ plan vs base | Δ learn vs plan |
|--------|-------|--------|-------|----------------|-----------------|
| SMW    | +36.1 | +54.0  | +38.2 | **+17.9 (+50%)**  | −15.8 |
| Sonic  | +37.1 | +140.2 | +80.9 | **+103.1 (+278%)**| −59.4 |
| Minish | +0.7  | +1.1   | +0.9  | **+0.4 (+57%)**   | −0.2  |

Per-seed: SMW plan [37.8,70.2] base [43.0,29.2]; Sonic plan [155.0,125.5] base [18.5,55.8] (BOTH seeds
plan>>base); Minish plan [1.2,1.0] base [0.8,0.7].

### Two robust findings (consistent across all 3 games)
1. **The corrected per-game prompt makes the plan HELP** (+50% to +278%). This FLIPS the overnight-bench
   result (plan HURT platformers −0.23) — that bench used the OLD miswired global prompt (FE routed to
   "Mario platformer"; global INSTR hardcoded "jump" for every game). The bottleneck was the PROMPT, not
   the idea of planning. Directly answers the user's "the planner seems to be the problem."
2. **Carrying `<learnings>` context forward HURTS** vs stateless per-replan planning (every game). Matches
   the earlier prev-plans ablation (eval_prevplans_ablation.py: prev-plans ON Δ−7.45, induces plan
   OSCILLATION, not exploration). Naive context accumulation crowds out fresh-frame grounding / biases the
   planner toward stale, verbose plans. → Keep the planner STATELESS per replan; do NOT paste history.

CAVEAT: 2 seeds × ≤4 states = high variance (SMW plan 37.8 vs 70.2). Directions are consistent across all
3 games and both Sonic seeds; treat magnitudes as indicative. FE excluded (reward stub; qualitative video only).

### Implication for the architecture
- Planner value is REAL once the prompt is genre-correct → invest in per-game/genre prompt correctness
  (game_planner.GAMES), not in context-carry.
- For RL/deferral: the plan's marginal value is now clearly POSITIVE on these demo states (unlike the
  pure-forward Sonic-only earlier finding) → these demo start states are a better deferral/plan-value
  substrate than the auto-booted forward-only states.
