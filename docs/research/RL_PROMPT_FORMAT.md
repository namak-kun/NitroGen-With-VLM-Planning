# RL Planner Prompt Format (the "input format, revisited")

Implemented in `planner_poc/rl_planner_prompt.py`. This is the **closed-loop observation the System-2
planner sees at RL time** (invoked every A=2 chunks). It is distinct from the two existing prompt paths:

| path | when | granularity | actions shown? | think mode |
|---|---|---|---|---|
| `PlanEncoder.generate_plan` | closed-loop eval | flat frame window | no | no |
| `generate_grounded_plan.py` | offline teacher/distill | per-CHUNK frames+actions | yes (coarse) | no |
| **`rl_planner_prompt.py`** (this) | **RL rollout** | **per-SUBCHUNK interleave** | **yes (fine)** | **yes (budget-forced)** |

## Exact layout (your spec)

```
[SYSTEM]
  intro + task            role of the System-2 planner (frozen VLM) above a System-1 controller
  game info               which game / objective
  controls                grounded control vocabulary (per genre; reused from generate_grounded_plan.CONTROLS)
[USER]
  frame f0                                          state BEFORE this 2-chunk window
  Chunk 1, inputs s1_1: <action summary>            subchunk 1 = first 6 of the 18 steps
  Resulting frame f1_1                              frame AFTER s1_1
  s1_2 / f1_2
  s1_3 / f1_3 (after the full chunk elapsed)
  Chunk 2, inputs s2_1: ... / f2_1
  s2_2 / f2_2
  s2_3 / f2_3 (after the full chunk elapsed)        <- LAST frame; planner invoked here (A=2)
  [optional] your recent plans (last <=2)           exploration prior — ABLATABLE
  OUTPUT INSTRUCTIONS                                broad, control-grounded, one sentence
                                                     [+ optional DEFER sentinel]
=> model THINKS (<think>…</think>), then emits the plan that becomes K=8 plan tokens.
```

`images` are passed in order: f0, then f{c}_{s} after each subchunk. Demo prompt has 1 + A·S = 7 images.

## Config (the knobs you fixed)
`RLPlannerConfig`: `num_plan_tokens=8` (K), `num_chunks=2` (A — replan cadence & prompt span),
`intra_chunk_rate=3` (S — subchunks/chunk → `subchunk_len = 18 // 3 = 6`), `action_horizon=18`.

## Think mode — why budget forcing
Your requirement: *"we work in think mode … lest model analysis falls into the actual output and makes
the plan diluted."* Qwen3.5-2B's chat template supports `enable_thinking=True` (injects `<think>\n` after
the generation prompt). Verified behavior:
- The model **does** reason well (per-subchunk cause/effect + progress evaluation) inside the think block.
- But the 2B is **verbose** and frequently does **not** close `</think>` within a small token budget
  (>512 tokens), so a naive single `generate` leaks raw analysis into the plan.

So `generate_rl_plan` does **budget forcing**:
1. Generate up to `think_budget` (default 256) tokens of reasoning.
2. If the model already closed `</think>` and produced a plan → use it (`forced=False`).
3. Otherwise **force-close**: append `</think>\n\nPlan:` (the `Plan:` seed is neutral — no direction
   words — so the model emits a fresh plan instead of finishing the truncated thought) and generate up to
   `plan_budget` (default 48) tokens. Re-passes `pixel_values`/`image_grid_thw`/`mm_token_type_ids` so the
   image tokens re-embed correctly in the second pass. `forced=True`.

This **guarantees the analysis stays in `<think>`** and **bounds planner compute per RL call** (important
when the planner is invoked every 2 chunks across long rollouts). Returns
`{think, plan, raw, forced}`.

Verified (synthetic platformer demo):
- THINK: full structured analysis ("Frame f0: … Chunk 1: s1_1 Left stick RIGHT … Evaluate progress: …").
- PLAN: `Move RIGHT and JUMP briefly to advance forward.`  (clean, grounded, undiluted)
- `--no-think`: empty think block, straight to `Move RIGHT and JUMP to reach the next ledge.`

### Think-mode A/B — a SECOND reason to keep think ON (real Sonic, identical observations)
WITHOUT think, the 2B frequently **degenerates into echoing the interleaved action-summary format** —
plan = `RIGHT 6, DOWN 6, ACCELERATE/FIRE 6` or `RIGHT 1, UP 5, ATTACK 4, ACCELERATE/FIRE 6` (it copies the
s{c}_{s} input style). WITH think, it reliably emits a clean grounded plan (`Move RIGHT to avoid the
floating orange ball and continue advancing`). The think block gives the model room to process the action
trace so the final plan doesn't collapse into action-token echoing. So think mode here is not only about
keeping analysis out of the output (your stated reason) — it also **prevents format-echo** when the prompt
interleaves action summaries. Keep think ON for the RL planner.

## Two ablations you flagged
- **prev-plans** (`include_prev_plans=True`): appends the last ≤2 plans ("avoid repeating one that did not
  make progress") to incentivize exploration. OFF by default — you said *"maybe ablate?"*.
- **deferral** (`allow_defer=True`): adds to the output instructions that the planner may output exactly
  `NO GUIDANCE NEEDED` when the moment is purely reactive (System-1 can handle it). The caller maps that
  sentinel → a null/masked plan (base-DiT-exact). This is your **deferral-via-output** idea (preferred over
  a model-controlled CFG weight). **Scaffolded only**: deciding *when* to defer needs an RL/training signal
  we do not yet have (you noted "we lack signal"). **PROBED: 0/5 zero-shot defer rate on real Sonic — the
  frozen 2B NEVER defers on its own, it always plans even in smooth System-1 moments.** So the mechanism is
  in place but the *policy* must be trained. The signal already exists: `rl_eval_plan_vs_null.py` gives
  per-situation plan-vs-null GT reward → "defer is correct ⇔ null ≈ plan reward". Reward the defer token on
  exactly those situations (after the actor+reward loop is solid).

## API (for the RL loop)
```python
from rl_planner_prompt import build_rl_messages, generate_rl_plan, RLPlannerConfig, DEFER_PHRASE
cfg = RLPlannerConfig()                                   # K=8, A=2, S=3
sys, content, images, render = build_rl_messages(
    f0, chunks,                                           # chunks: A records, each {"subs":[(action,frame)*S]}
    genre="platformer", game_info="Game: …. Objective: …", cfg=cfg,
    include_prev_plans=False, allow_defer=False)          # ablations
res = generate_rl_plan(planner, sys, content, images, device,
    enable_thinking=True, think_budget=256, plan_budget=48, temperature=0.0)
plan = res["plan"]                                        # clean; res["think"] = analysis (for logging/credit)
```
A `chunk` record may also be `{"chunk": <load_chunk_actions dict>, "frames":[f_sub1..f_subS]}` and the
builder will summarize the subchunks itself (via `nitrogen.training.actions.summarize_chunk` on slices).

## Run
```bash
RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
$RUN .venv/bin/python planner_poc/rl_planner_prompt.py --demo --genre platformer            # think demo
$RUN .venv/bin/python planner_poc/rl_planner_prompt.py --demo --print-prompt-only           # assemble only
$RUN .venv/bin/python planner_poc/rl_planner_prompt.py --demo --no-think                     # ablation
$RUN .venv/bin/python planner_poc/rl_planner_prompt.py --demo --include-prev-plans --allow-defer
$RUN .venv/bin/python planner_poc/rl_planner_prompt.py --chunk-dir <c1> <c2> --frames-dir <f> --genre platformer
```

## Open (to wire up in the morning)
1. **Integrate into the rollout/RWBC loop**: `planner_poc/rl_rollout_demo.py` ALREADY runs this format
   end-to-end on a live env (real actor → sub-chunk frames → think-mode planner). VERIFIED on Sonic: the
   planner grounds on real pixels (reads "Sonic … ground hazards … hanging enemy … floating rings"), think
   stays in the trace, plans are sensible, GT reward climbs +4.1 → +10.6 over 2 cycles. Remaining: fold
   this sub-chunk rollout into `rwbc_actor_adapt.py`/the RL trainer (which currently uses the simpler
   `generate_plan` frame window) so training consumes the same observation eval will use.
   - **GOTCHA for sub-chunk capture**: `ProcRLEnv.step()` applies only the first `self.A` rows of the chunk
     it is given (`rows[:self.A]`), so the S=3 sub-chunks of 6 are only FAITHFULLY applied on the emulator
     envs (retro/mgba/snes, which apply every row). On proc envs (TheXTech/Solarus) set `env.A = subchunk_len`
     (or step row-by-row) before stepping, else only 2 of each 6 rows take effect.
2. **Think-mode quality A/B**: DONE tonight — without think the 2B echoes the action-summary format
   ("RIGHT 6, DOWN 6, …"); with think it plans cleanly. Keep think ON. (Whether think changes the plan vs
   GPT-5.5 quality is still open.)
3. **Ablate prev-plans + defer** once the actor+reward loop is solid (the duck: exploration before memory).
   Defer probed: 0/5 zero-shot — needs a training signal; use `rl_eval_plan_vs_null.py` null-vs-plan reward
   as the "defer-correct" label.
