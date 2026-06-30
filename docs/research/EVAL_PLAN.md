# EVAL PLAN — synthesis of the GPT-5.5 eval rubber-duck (2026-06-27)

## The verdict (one line)
Replace raw `screen_x` / shaped-reward as the headline with:
> **Demo-conditioned, survival-gated CHECKPOINT REACHABILITY**: from many frame-exact expert savestates,
> can the agent reach the expert's FUTURE task-state checkpoints (within a time budget, alive & non-stuck),
> scored as expert-normalized [0,1] segment progress, **averaged EQUALLY across games** (+ report worst-game).

Everything else — raw screen_x, current shaped reward, the VLM judge, best-of-K-by-progress — is
**diagnostic only** until this exists.

## Why (the 5 blocking issues)
1. Raw progress is invalid as headline — rewards run-into-death / off-task motion (the +194 raw vs ≈0 shaped
   disagreement IS this failure: "visible motion, not robust task progress").
2. Shaped reward isn't cross-game (platformer-centric, arbitrary weights, no FE/top-down fit).
3. Fire Emblem has NO valid reward — cursor movement/turn-increment is an exploit target, not progress.
4. The VLM judge is unvalidated + contamination-prone (few-hundred motion pairs, popular-game prior).
5. We UNDERUSE the demos — 15 multi-minute `.bk2` trajectories = thousands of replayable start states, not
   "15 clips." [VERIFIED tonight: expert npz replay is deterministic; screen_x advances; can snapshot
   savestates at any point + read the expert progress curve + the npz ships per-frame `rewards`.]

## The metric, precisely
- **Task-state feature φ(s) from RAM** (NOT pixels, NOT full-RAM distance, NOT screen_x):
  - platformer: level/act/room/mode, player x/y, camera, alive/lives (score/rings secondary).
  - top-down (Minish): area/room/map-id, x/y, transition flags, menu/dialogue mode, alive.
  - Fire Emblem: chapter/phase/turn, unit positions, selected/moved/waited, enemy HP/deaths, objective flags
    — NOT cursor.
- **Checkpoint k reached** iff: same discrete context (level/room/phase) AND `dist(φ_agent, φ_expert_k) <
  radius` AND alive/not-softlocked/valid-mode. Restrict projection to a FUTURE window `[i, i+T+slack]`;
  enforce MONOTONIC checkpoint index (no backward jumps); report off-path fraction separately.
- **Segment score** = furthest future checkpoint reached / target, clipped [0,1]; death/invalid → 0.
- **Rollup**: GameScore = mean segment score; GeneralistScore = mean over games (EQUAL weight, so SMW's 8
  demos don't dominate); also report worst-game / bottom-quartile.
- **Stats**: hierarchical bootstrap over demos/segments (nearby starts are correlated — NOT frame-level CIs).
- **Mandatory sanity**: expert `.bk2` replay from s_i must score 1.0; no-op/random/null must be ~0.

## Dense-signal harness (build FIRST, SMW+Sonic+Minish; FE after board-state)
1. Replay each demo deterministically (npz actions; VERIFIED) → snapshot savestates every 1–2 s, skipping
   menus/cutscenes/death-junk.
2. Per start i, define future checkpoints at +2 s / +5 s / +10 s.
3. Run policy from s_i, budget 1.5–2× target, matched stochastic seeds across conditions.
4. Emit per segment: reach@{2,5,10}s ∈{0,1}, progress_fraction∈[0,1], death, stuck, off_path_fraction.
5. Baselines every run: expert-replay (=1.0 sanity), no-op, action-noise, null/base ckpt, fixed-human-plan,
   planner-plan.

## Contamination + judge validation (do NOT use judge as headline until this passes)
- GT-RAM (or demo-checkpoint success) is the ONLY trustworthy anchor for cross-game/generalist claims.
- Validate the VLM judge (and any LLM/rubric judge) on OBSCURE/homebrew GT-labeled games (Witchblast, a
  Solarus custom non-Zelda quest, a reskinned TheXTech level, an obscure platformer, ideally a homebrew
  tactics game). Controls for: sprite recognition, memorized directionality, "rightward=progress" prior,
  game-name leakage.
- Report: AUROC, Brier/calibration, FP-rate on death/stuck/wall/menu, recall on real progress,
  leave-one-game-out, train-popular→test-obscure, train-platformer→test-topdown, known→reskinned sprites.
- Adversarial controls: no-op pairs, death-after-moving-right, shuffled frames, wrong/ inverted objective
  text, big-motion-no-progress, small-motion-real-progress (menu/tactical). Make judge BLIND to game name.
- FLAG: the planner bridge (btn_s600) is SYNTHETIC-plan-trained → any eval using the model's own plans is
  partly circular; the judge is GT-pair-trained but on a narrow motion-correlated distribution.

## Capacity-vs-distribution probes (the experiments to settle the architecture debate)
- ID-vs-OOD game split (Genesis Sonic / GBA = likely OOD) under the SAME checkpoint eval.
- BC-fit headroom: can it fit train chunks? roll out from train starts? held-out starts? held-out demos?
  (cannot-fit → capacity/action-head; fits-but-rollout-fails → compounding/horizon; fits-train-fails-heldout
  → distribution.)
- Oracle plan-token ceiling: search latent plan tokens per segment to max checkpoint reachability (oracle
  succeeds & planner fails → planner/bridge/text; oracle fails & BC succeeds → plan-interface; both fail →
  actor capacity).
- Expert-replay sanity from every savestate.

## Explicitly DO NOT build yet
VLM/LLM judge as headline; more raw-screen_x best-of-K; FE cursor reward; image-embedding-only projection;
elaborate global game-completion eval; per-game scalar rewards that don't normalize.

## Open design choices to settle with the user before building
- φ(s) RAM addresses per game — which we actually have mapped (platformer x/y/room exist; FE board-state
  likely NOT mapped → FE may need hand-annotation or deferral).
- checkpoint radius + discrete-context gating per game.
- the "expert path is one of many valid routes" caveat: this is a LOWER-BOUND, demo-corridor competence
  metric — good for ranking model versions, not a claim of open-ended mastery.

---

## ORDINAL FALSIFIER RESULT (2026-06-27) — the ruler caught its own flaw [docs/graded/ordinal_*.json]
`planner_poc/ordinal_sanity.py`: 8 mined mid-level start states/game, 3 seeds, 900-frame budget, controls =
idle / random / RIGHT+jump (hand-coded) / base(null-plan DiT) / rwbc(adapted delta).

RAW metric P_max_alive (peak screen_x before first death — GPT-55's proposed headline) IS EXPLOITABLE:
  SMW right_jump reach=0.925 (HIGHEST) but survived=0.00 -> a sprint-right-and-DIE policy "wins". Suicide
  exploit confirmed. base also dies 38%, rwbc 79% on mid-level states.

FIX = survival-weighted headline **reach × survived_frac**:
  SONIC: idle .194 < random .307 < right_jump .659 < base .670 < rwbc .945   (clean; rwbc best, survives 100%)
  SMW:   right_jump .000 < idle .072 < rwbc .206 < random .250 < base .522    (suicide->0; base best)

TWO findings:
1. The falsifier WORKED — it broke the naive metric (suicide-sprint) and the fix (reach×survival) restores
   sensible ordering. Adopt reach×survival (or survival-gated end-progress) as the headline; report reach,
   survived_frac, death_after_peak separately as diagnostics.
2. SMW base > rwbc under survival-weighting: the SMW RWBC delta (trained from LEVEL-START states, +0.085
   shaped there) does NOT generalize to MID-LEVEL survival — it dies more (survived 0.62->0.21). The mid-level
   ordinal eval is MORE discriminative than the level-start held_out_eval and caught a generalization failure
   the per-start eval missed. This validates the harder eval's worth. (Sonic RWBC genuinely generalizes:
   .945 @ 100% survival.)

NEXT for the eval: (a) make reach×survival the default headline (DONE in ordinal_sanity.py); (b) extend to
SMW player-progress var if camera screen_x proves campy; (c) Minish needs room-graph progress (no scalar);
(d) scale start states via .bk2 mining; (e) the YT multi-path envelope (see YT_IDM_FINDINGS.md) as an
aggregate human reference later.

---

## REACH_EVAL GENERALIST HARNESS + the multigenre METRIC-FLIP (2026-06-27) [docs/graded/reach_eval.json]
planner_poc/reach_eval.py: reusable, survival-weighted (reach×survived), multi-horizon, per-game->generalist.
| policy | GENERALIST | sonic | smw | worst |
|---|---|---|---|---|
| idle | 0.173 | 0.33 | 0.015 | smw |
| random | 0.283 | 0.30 | 0.26 | smw |
| right_jump | 0.566 | 0.88 | 0.25 | smw |
| base | 0.593 | 0.68 | 0.51 | 0.51 |
| multigenre (ONE shared LoRA) | 0.671 | 0.72 | 0.62 | **0.62** |
| rwbc (per-game deltas) | 0.723 | 0.93 | 0.52 | 0.52 |

BIG FINDING: on the survival-weighted metric the ONE shared multigenre LoRA IMPROVES BOTH games over base
(no interference) — OVERTURNING last night's raw-screen_x "cross-genre interference" claim (that was the
suicide-sprint artifact: raw progress rewarded the reckless per-game policy). Per-game rwbc tops the average
but ONLY via Sonic (smw~base) and is per-game-models (rejected). Shared LoRA has the best WORST-game.
Notes: SonicScreen_x has a camera-scroll artifact (idle=0.40) -> Sonic abs scores inflated; SMW is clean
(idle=0.015). SMW start-states came from the junk demo -> re-run with a good SMW demo. RE-JUDGE all prior
capacity/interference claims on THIS metric, not raw progress.

## IDM GATE = FAIL [files/IDM_GATE_FINDINGS.md, docs/idm_gate/]
Small non-causal IDM, 6 SMW demos, held-out playthrough: jump F1 0.37, LEFT 0.29, RIGHT 0.58 (~constant-on),
over-calls jump. Data-blocked. YouTube pseudo-labeling premature until more demos + stronger temporal model
(>0.6-0.7 held-out F1 on jump/left/right). yt_farm.py grab FIXED (progressive -f fallback; verified 16 frames).

---

## CORRECTED 3-GAME RESULT on GOOD demos (2026-06-27) [docs/graded/reach_eval_3game.json]
Survival-weighted reach, LONGEST demos (fixes the junk-demo confound), 6 starts, 2 seeds, horizons 450/900:
| policy | GENERALIST | sonic (camera-inflated) | smw (CLEAN) |
|---|---|---|---|
| idle | 0.20 | 0.33 | 0.074 |
| base | 0.49 | 0.68 | **0.308** |
| multigenre (shared) | 0.47 | 0.72 | 0.228 |
| rwbc (per-game) | 0.61 | 0.93 | 0.298 |

CORRECTIONS (retractions of earlier overclaims):
1. multigenre does NOT "strictly improve both games" — that was the JUNK-demo artifact. On good demos it
   HELPS Sonic, HURTS SMW (0.228<0.308) -> ~neutral generalist (Sonic-for-SMW trade). Interference is real
   but milder than raw-screen_x suggested.
2. On the CLEAN game (SMW) NOTHING beats base: base 0.308 >= rwbc 0.298 >= multigenre 0.228. The only clear
   win (rwbc Sonic 0.93) is camera-inflated (idle Sonic=0.40 vs SMW=0.074) AND a per-game model. So on the
   trustworthy metric we have NO model change that robustly beats base yet. Prior "+0.98/+194 generalizes"
   was largely raw-screen_x + camera artifact.
3. Minish x-coordinate FAILS (usable=0 starts: expert x not monotone-forward enough) -> top-down needs
   path-relative or room-graph (neither works on this demo). FE: no dense signal. EVAL SOLID ONLY for
   side-scrollers (sonic w/ camera caveat, smw clean).

IMPLICATIONS: (a) Sonic needs a non-camera progress var (player x / level-progress) to de-inflate. (b) the
generalist eval currently = 2 side-scrollers; top-down/turn-based need richer RAM or a judge. (c) the model
side has no robust win on the good metric -> this is the real target (the Dual-rate Modulated Bridge from
ARCH_RESEARCH.md, trained with emulator recovery data, is the proposed path). (d) n=6 starts/2 seeds still
small; scale before strong claims.

## ===== EXPERT-DENOMINATOR BUG FIXED (2026-06-28, user-caught) =====
USER CAUGHT IT: "Sonic base=0 contradicts the videos — the base model does stuff regardless." Checked the
actual actions in demo_videos/state0_base.mp4: base GENUINELY moves Sonic (mid-start screen_x gains
1055/868/988). The eval said base=0.000 → that was a METRIC BUG, not the model.

ROOT CAUSE: reach_eval normalized by the EXPERT's short-horizon forward-gain as the denominator. Sonic
loops/backtracks, so the expert's LOCAL forward-gain at many mid starts is tiny or zero (logged
expert_ref/ceil pairs: 1085, 195, 299, **0**). Dividing a legitimate base gain by ~0 → score clips to >1 or
produces garbage (one start computed 988). The `usable` filter (expert-gain ≥ 80) also wrongly excluded the
backtracking starts. Net: base appeared as 0.000 — false.

FIX (planner_poc/reach_eval.py): denominator is now a scripted-RIGHT(+jump) ALWAYS-FORWARD ceiling and an
idle floor — `skill = clip((gain − idle_gain) / (rightjump_gain − idle_gain), 0, 1)`. The expert gain is now
only a logged REFERENCE line (`expert_ref/ceil`), never the denominator. Headline still = reach × survived.

FIXED RESULTS (4 mined starts/game, H=600f):
  | policy      | Sonic | SMW   |
  | idle        | 0.000 | 0.000 |
  | right_jump  | 1.000 | 0.500 |  ← SMW: blind right+jump SUICIDES (survival-weighting docks it to 0.5)
  | base(null)  | 0.250 | 0.408 |  ← base is NO LONGER 0 on Sonic (bug fixed); genuinely moves both
  | rwbc(32t)   | 0.584 | 0.529 |  ← RWBC BEATS base on BOTH games

OVERTURNS the earlier "no model robustly beats base" — that conclusion was an ARTIFACT of the broken
expert-denominator. On the fixed metric, RWBC clearly beats base on both side-scrollers (Sonic 0.58>0.25;
SMW 0.53>0.41), and the control ordering is sane (idle < base < rwbc; right_jump=ceiling on open Sonic but
survival-penalized on obstacle-heavy SMW). JSONs: docs/graded/reach_fix_{sonic,smw}.json.

CAVEAT carried forward: right_jump is a valid ceiling for side-scrollers only; top-down/turn-based games
still need a room-graph / RAM-semantic ceiling. n=4 starts/game — scale before final claims.
