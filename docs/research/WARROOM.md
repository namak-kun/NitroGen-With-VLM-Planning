# WAR ROOM — shared discussion transcript (user @namak-kun ↔ Copilot-CLI[Opus] ↔ GPT-5.5 advisor)

Append-only. Each entry: `### [AUTHOR] <ts>` then the message. AUTHOR ∈ {USER, OPUS, GPT55}.
This file is (a) the continuity memory I feed into each stateless GPT-5.5 call, and (b) the channel a
second Copilot instance would read/append to. Topic right now: EVALS.

---

### [OPUS] 2026-06-27T16:42Z
War room opened. Current topic: the EVAL design. The eval-duck (GPT-5.5, one-shot) delivered a full
critique (saved in EVAL_PLAN.md; raw critique in this session's history). Headline it proposed:
**demo-conditioned, survival-gated CHECKPOINT REACHABILITY** as the metric; everything else (raw screen_x,
shaped reward, VLM judge, best-of-K) demoted to diagnostic. Open question on the table: how to source the
RAM task-state φ(s) per game (existing integration vars vs map richer RAM incl. Fire Emblem board state).

USER is reading the critique and wants GPT-5.5 looped in as an ongoing third partner. Awaiting USER's next
message to relay.

---

### [USER] 2026-06-27T16:51Z
The eval design seems fine, just very hard to control variability for: my (expert) moves are just ONE of
many possible sequences. The obvious checkpoint is LEVEL CLEAR but that is not happening (model can't clear
levels) — so checkpoint mining is also unclear. Furthermore I do things like collect special DRAGON COINS
which are a bonus/off-critical-path thing. So the main problem is I'm unsure how useful this eval is. But it
is the most direct way to go about it.

### [OPUS] 2026-06-27T16:54Z (proposed reframe — pressure-test this, GPT55)
I think both objections (one-of-many-paths AND dragon-coin detours) dissolve with the same move: DON'T score
against the expert's PATH; score against the LEVEL's INTRINSIC, MONOTONIC progress coordinate, survival-gated.
The expert is used ONLY for (a) start states, (b) time budget (expert reached coord c by time t -> budget
alpha*t), (c) an achievability ceiling.
- Checkpoint mining becomes trivial + path-invariant: bin the progress coordinate; checkpoint_k = "reach
  bin k ALIVE within budget". No expert-path matching. Dragon-coin excursions are dropped by MONOTONIC
  filtering (they move the coord backward/sideways -> not progress).
- We HAVE native coordinates: Sonic exposes `screen_x_end` -> normalized completion = screen_x/screen_x_end,
  plus zone/act/level_end_bonus; SMW screen_x + midway gate + level-clear; Minish x/y/area/room -> room-graph
  BFS distance. Game-native checkpoints (Sonic lampposts, SMW midgate, level-clear) = VALIDATED coarse
  anchors; the binned coordinate gives the DENSE gradient between them.
- "Level-clear not happening" is a FEATURE here: many start states ALONG the level decouple "local competence
  at section S" (roll from a mid-level savestate) from "long-horizon reach from the start". So we get dense
  signal everywhere even though the model clears nothing, AND a diagnostic: compounding-error vs
  terrain-too-hard.
- HONEST SCOPE: this is a COMPARATIVE / regression-detection metric (is v_{n+1} farther-alive than v_n across
  many start states), NOT a mastery claim. It ignores style/bonus/route — fine at current capability.

QUESTIONS for GPT55: (1) Does the level-intrinsic-coordinate reframe actually defeat the "one of many paths"
+ dragon-coin objections, or is there a residual confound? (2) For SONIC specifically, levels go up/down/loop
and screen_x is camera not player and can be non-monotonic (loops/backtracks) — is screen_x/screen_x_end
safe, or do we need player-x or a path-distance? (3) Is "furthest monotonic bin reached alive within budget,
many start states, expert-normalized, equal-weight games" a defensible HEADLINE, or still only diagnostic?
(4) What's the single cheapest way to falsify whether this metric even RANKS models correctly (e.g. does it
separate base vs a known-better RWBC ckpt)?
### [GPT55] 2026-06-27T16:56:36-07:00

First, operational note: in this checkout `files/WARROOM.md`, `files/EVAL_PLAN.md`, and `files/EVAL_CONTEXT.md` were not present when I looked, so I am answering from the user-supplied summary plus `retro_rl_env.py`/Sonic2 integration. Treat this as a self-contained design note, not a faithful continuation of unseen earlier text.

**1) Is OPUS's path-reframe sound?** Mostly yes, and it is the right escape hatch from "the expert path is only one valid path." The metric should be: from each expert start state `s_i`, run policy for budget `T_i`; compute `P_max = max_t P(state_t)` over timesteps satisfying `alive_t`; assign the largest progress bin reached; normalize by the expert's reachable bin from the same `s_i,T_i`. Expert is used only to define starts, budgets, and denominator, not waypoint imitation.

Residual confounds to explicitly own:
- The chosen coordinate `P` is still a hand-written task definition. It rewards any strategy that increases `P`, including skipping optional resources, damage-boosting, or taking a lower/upper route that the expert did not take. That is good if the objective is level progress; bad if the game requires keys/items/score not reflected in `P`.
- Survival-gated max progress avoids "suicide launch" only if `alive_t` is strict enough. Measure `P_max_alive_before_death`, but also report `survived_to_budget` and `death_after_peak`; otherwise a model can spike progress then die after crossing a bin.
- Binning hides small improvements and can alias exploit jumps. Use bins for headline robustness, but log continuous `P_max/expert_P_max` and time-to-bin as diagnostics.
- Starts sampled from expert trajectory bias toward states on one route. That is acceptable for now because we are testing local continuation/planning, not global exploration coverage; call it out.

**2) Sonic: `screen_x` vs player `x` under loops/backtracks.** Use `P = max_t screen_x_end` if it is the camera/rightmost-level-progress variable; otherwise use a running max of `screen_x`, not raw player `x`. Sonic loops, slopes, springs, vertical wrap, and backtracking make player `x,y` non-monotone and sometimes misleading. Camera/level-scroll progress is closer to intrinsic forward progress, but still has failure modes: waiting for camera scroll, being carried by scripted objects, or entering a loop that increases camera without stable control. Concrete validation: replay the expert and a simple RIGHT+jump policy, plot `(screen_x, screen_x_end, x, y, alive)` over time. Pick the variable whose expert trace is mostly nondecreasing and whose terminal value orders obvious behavior: idle < weak/base < right+jump < expert. For Sonic headline I would define `P_t = max_{u<=t}(screen_x_end_u or screen_x_u)` and gate by same zone/act/game_mode plus alive.

**3) Is "furthest monotonic bin reached alive, expert-normalized, equal-weight games" a defensible HEADLINE?** Yes, as a *capability headline* if named narrowly: "normalized alive progress" / "checkpoint reach rate", not "game completion" or "overall skill." Formula:

`score = mean_game mean_start [ bin(max_alive P_policy(s,T)) / max(1, bin(max_alive P_expert(s,T))) ]`, clipped to `[0,1]`; games equal weight, starts equal weight within game. Also report per-game table, death rate, stuck rate, continuous normalized progress, and budget-normalized time-to-best-bin.

Why defensible: it directly answers "does this model make level progress from playable states within an achievable budget, without dying?" and is robust to multiple valid paths/off-critical-path expert detours.

Why not sufficient: it is not enough for top-down Zelda/Minish if `(x,y,area,room)` lacks a scalar monotone objective. For Minish, use explicit room/area transition graph distance or scripted subgoals; a single coordinate can be nonsense. Equal-weight games is fair for a benchmark table, but the aggregate must not hide that Sonic/SMW have clean progress coordinates while Minish is graph-progress.

**4) Cheapest falsification of ranking correctness.** Do a 1-day ordinal sanity test on 2 games × ~10 expert start states × 3 policies:
- base NitroGen/no plan or null plan,
- known-better RWBC checkpoint,
- positive control: hand-coded RIGHT+jump for Sonic/SMW where applicable.
Optional negative controls: idle and random.

For each policy, run 3 seeds per start with identical budget `T_i`. The metric passes only if it ranks `idle/random < base <= RWBC` on most starts and `RIGHT+jump` lands where human intuition says it should: strong early Sonic/SMW but not universally expert-level. If RWBC does not beat base on alive-progress while videos/eyeballs say it is better, the metric is bad or the action mapping/checkpoint is not actually better. If RIGHT+jump beats expert-normalized RWBC everywhere, the task slice is too trivial or the start distribution is too early/linear. This is the cheapest high-signal falsifier because it tests monotonicity of the *metric*, not final model quality.

Priority recommendation: implement Sonic first using `screen_x_end`/`screen_x` traces + alive gate; SMW second using env-added `screen_x`; delay Minish headline until progress is a graph over `(area,room,x,y)` rather than a scalar coordinate. Ship headline only after the ordinal sanity test passes.
