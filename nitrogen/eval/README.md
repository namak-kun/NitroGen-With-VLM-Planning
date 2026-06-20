# Closed-loop evaluation harness (`nitrogen.eval`)

Game-agnostic infrastructure to evaluate **plan-conditioned NitroGen in real games** — the
ground-truth test that our proxy metrics (velocity-MSE, stick-delta, override flip-rate) cannot
provide. See `EXPERIMENTS.md` (the VERDICT section) for *why* this exists: env-free proxies kept
misleading us (95% → 62% → 22% as eval bias was removed), so we need closed-loop validation.

## The four decoupled pieces

| piece | interface | responsibility |
|---|---|---|
| **Scenario** | `core.Scenario` | initial state + the **plan** given to the policy + the success criterion + max steps + CFG scale |
| **GameEnv** | `core.GameEnv` | renders RGB frames the policy sees; exposes **privileged state** (pos, map, items, flags); maps the 25-dim NitroGen gamepad onto the game |
| **Policy** | `core.Policy` | frame (+ plan set at reset) → `(H,25)` action chunk |
| **SuccessDetector** | `core.SuccessDetector` | reads state/frame → did the objective complete? |

`EpisodeRunner` ties them in a loop; `EvalProtocol` runs suites and computes metrics.

## The headline metric: plan-selection (same start, different goals)

`EvalProtocol.plan_selection_matrix` runs a group of scenarios that **share an initial state**
but carry **different plans+objectives**, and checks each plan against every objective:

```
              objective→
            west  east  north south
plan west [  OK    .     .     .  ]
plan east [   .   OK     .     .  ]
plan north[   .    .    OK     .  ]   diagonal = own plan reaches own goal
plan south[   .    .     .    OK  ]   off-diag = own plan reaches a DIFFERENT goal
```

`selectivity = diag_rate − off_rate`. High selectivity from an **identical start** means the
plan causally selects among valid futures — the planning capability is real, measured in a game.
This is the legitimate counterfactual test: both outcomes are valid; the plan is *selecting*,
not fighting the frame prior (the thing that hit a ceiling env-free).

## Status

- ✅ Core abstractions, detectors (state-predicate + VLM-judge), runner, protocol.
- ✅ `envs/dummy.DummyGridEnv` + `planner_poc/eval_harness_smoke.py` — validates the whole
  harness end-to-end (selectivity +1.00) with no game/model.
- ✅ `planner_poc/eval_policy.NitroGenPolicy` — loads any EXP-04x checkpoint (auto-detects
  LoRA / plan-adaLN), produces chunks via plan-CFG. The model→harness bridge.
- 🚧 `envs/cavestory.CaveStoryEnv` — scaffold + protocol; needs the doukutsu-rs agent-socket
  patch (inject inputs / read state / grab framebuffer). Documented in that file.
- ⏳ Flagship `zelda3` (ALttP) env — pending a ROM (best demonstrator; the "go to Hyrule castle"
  example). Plugs in as another `GameEnv`.

## Quick start

```bash
# validate the harness (no game, no GPU):
python planner_poc/eval_harness_smoke.py

# load the best override model into the harness (GPU):
CFG=8 python planner_poc/eval_policy.py runs/stage2_student/plan_stage1_2500.pt
```

## Adding a game

Implement `GameEnv.reset/step` (+ map the 25-dim gamepad to the game's input, expose state),
then write Scenarios with a `success_spec` the `StatePredicateDetector` understands
(`reach_region`, `reach_map`, `obtain`, `flag_true`, `predicate`, `fail_if`). Nothing else
changes — policy, detector, runner, and metrics are game-agnostic.
