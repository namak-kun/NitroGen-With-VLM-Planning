# ARCH WARROOM — Round 4 (OPUS-4.8, ROLE A: the PRESCRIPTION)
2026-06-28 · the concrete actor-OOD cure recipe, grounded in the repo. Latency OFF the table.

**Thesis:** the cure is **save-state GRPO implemented as advantage-weighted flow-BC**, with the Q3 gate
re-used as an **active off-manifold sampler** (DAgger without an oracle — the emulator+reward IS the
labeler), one shared generalist, frozen-VLM-first, exact null-invariance preserved by construction.

---

## 1. No queryable expert → save-state GRPO (critic-free), warm-started by RWBC

Classic DAgger (Ross-Gordon-Bagnell, arXiv:1011.0686) needs an oracle to **label** the actor's visited
states; we have only fixed human demos + emulator reward/reset. The realistic spectrum already half-lives
in `planner_poc/rwbc_actor_adapt.py`:

- **RWBC** (current): `collect()` keeps `--top-frac` highest-return chunks, BC's them with the model's own
  flow loss (`build_batch`→`m(batch)["loss"]`), reward-weighted (`rw` line 213). Critic-free. **Weakness:**
  hard-quantile, positive-only — imitates the best of what it *already* samples, discards the signed
  gradient in low-reward samples.
- **AWR** (Peng 2019, arXiv:1910.00177): weight BC by `exp(Â/β)`. Right *form*, but needs a value `V` →
  the long frame→plan→18-action→sparse-reward credit path makes `V` hard.
- **Save-state GRPO** (DeepSeek, arXiv:2402.03300): from a frame-exact σ, sample a **group** of G chunks,
  score by emulator return, use the **group mean as the baseline** — no critic net.

**Primary = GRPO, expressed as AWR-form weighting inside the existing loop.** It is the cleanest because
(a) it kills the critic (GRPO's whole point), (b) `emulator_env.save_state/load_state` (lines 199/203;
`snes_env.py:189/193`) make the group baseline **frame-exact** — all G branches share an identical start,
so the mean is a zero-variance baseline, and (c) it is a *one-function* change to the repo: replace
RWBC's quantile `rw` with a group-relative advantage.

**Exact data unit — a "branch group":** `(σ, plan p, {chunk_1..chunk_G})`, each chunk sampled with
`_sample_chunk(noise_sigma=--explore-sigma, noise_seed=…)` (eval_policy.py:195, SDE noise on the last
`noise_last_n` steps), then rolled `h` rows via `env.step` for return `R_i = Σ` shaped reward
(retro_rl_env.py:283–320: `scale·Δscreen_x + score − death − stuck`; `info{died,stuck,dprog}`).
**Advantage:** `Â_i = (R_i − mean_g R)/(std_g R + ε)`. **Loss:** the repo flow-BC loss weighted by `Â_i`
(positive-clamped AWR form `max(Â,0)` to start; signed GRPO once stable), base DiT **frozen**, only
LoRA(+plan-head) train (lines 187–189), with `--anchor` L2 as the trust-region/KL proxy (line 166).
Keep **RWBC as warm-start** (it already lifts reward in-repo) then switch the weight to `Â`.

---

## 2. The active loop — the Q3 gate IS the data sampler (DAgger w/o oracle)

The detector becomes the labeler-selector. During `collect()`:

- **Trigger:** `D_t > τ_D` (K=4-sample disagreement, conformal p95) **OR** (`r_t≈0` ∧ `dprog≈0` from
  `info`) for **2 consecutive chunks** (debounce). `r_t=‖v_c−v_u‖/(‖v_u‖+ε)` and `D_t` are **already
  computed** in `_sample_chunk` (eval_policy.py:237–242) — free at deploy.
- **Snapshot:** `σ = env.save_state()` at the drifted state — *exactly* the high-uncertainty off-manifold
  state BC never covered (the DAgger insight; the emulator+reward is the oracle, not a human policy).
- **Branch-search budget:** K=8 plan branches × G≈4 chunks × horizon h=6 rows ≈ 48 chunk-rollouts/trigger.
  Cheap — we **pause** (latency off the table) and stepping is in-process. Re-`load_state(σ)` between
  branches so all share the exact start.
- **What's added:** the best branch's first chunk `(σ-frame, plan, chunk, Â)` → the GRPO buffer, tagged
  *priority* (OOD states the demo set lacks). `apply_chunk_capture` (emulator_env.py:186) gives the exact
  per-row `(action, frame, state)` if we want dense rows.
- **Interleave:** DAgger-style aggregate→retrain: `collect (with triggers) → GRPO update on mixed buffer
  (on-trajectory demos + triggered recoveries) → re-collect`. Up-weight triggered states.
- **Null-invariance — by construction:** every added sample is **conditional** (plan present); we *never*
  write a null/dropped-plan recovery. `compute_plan_tokens`/`apply_null_mask` (nitrogen.py:594) still zero
  the K `_PLAN_TOKEN` positions for `plan_dropped` rows → base DiT **bit-identical** under null. The
  gate's actions (replan, raise `noise_sigma`, branch-search) only read/condition; under null `v_c==v_u`→
  `r_t=0`→gate inert.

---

## 3. One generalist, no collapse

**One model, shared LoRA + plan-head across games** (the constraint). The earlier multigenre
"interference" was **partly a `screen_x` metric artifact** — cumulative `screen_x` has different units/
scales per game, so a shared metric *looked* like interference.

- **Per-game reward normalization — for free:** GRPO's `(R−mean)/std` is computed **within a game's
  save-state group**, so advantages are unitless and per-game standardized. The artifact dissolves; no
  cross-game scale leaks into the gradient.
- **Batch balancing:** temperature-sample equal branch-groups per game per update; thread real
  `game_id`/`embodiment_id` (currently 0 in `build_batch` lines 117–118) so no game dominates.
- **Curriculum:** start where the actor has support (SMW/Sonic, RWBC>base shown), add minish/gba_* once
  stable.
- **Shared-vs-per-game LoRA:** **shared LoRA primary.** Only if a game *still* regresses **after**
  reward-normalization (real interference, not artifact) allow a tiny per-game FiLM/bias — flagged as
  against the one-generalist spirit; try balancing+norm first.
- **Frozen-VLM-first:** objectives for popular games are in Qwen's pretraining → keep it frozen.
  **VLM-LoRA cost-flagged:** only if frozen Qwen can't represent an objective; high cost (moves the
  bridge-aligned latents → forces re-fitting the adapter). Defer.

---

## 4. Confront the hole (off-manifold the plan may re-acquire authority)

The near-zero plan gradient was measured **on-trajectory** (expert demos, buttons AUC 0.82–0.96). Drift is
**off-trajectory** — unmeasured. Off-manifold the frame may be genuinely ambiguous → the plan gradient
could be **non-zero**, and a stale plan can itself **cause** drift (entanglement). My recipe does **not**
extrapolate the on-manifold gradient; it **measures** the hole at every drifted state:

- The branch-search in §2 **varies the plan** (`--plan-temp`>0, K plan branches), so at each off-manifold
  σ it empirically compares plan-varied vs fixed-plan recoveries. **Log `Δ_plan-off = R(best
  plan-varied) − R(best fixed-plan)`.** This is exactly GPT-5.5's proposed 4th rung (R_D = oracle-plan +
  actor-adapt + **fresh-replan-at-drift**) — baked into the loop as a logged quantity, not an assumption.
- **Replan-on-drift** (gate rung-1, `generate_plan` at σ; planner.py:238) routes a **fresh objective to
  exactly the drifted state** — the one place a fresh plan could matter (π0.5 text-subtask→flow chunk;
  pi.website/blog/pi05). The GRPO return tells us if it helped.
- **Decision rule:** if `Δ_plan-off ≫ 0` *systematically* → the consensus's "plan not load-bearing" is
  **refuted off-manifold** → keep plan-head trainable (NOT `--lora-only`) and up-weight replan-on-drift.
  If `Δ_plan-off ≈ 0` → actor coverage dominates as predicted, lock `--lora-only`. The recipe is
  **self-correcting** w.r.t. the hole; it instruments it rather than dodging it.

---

## 5. Build order (1–2 weeks, existing repo)

0. **Day 0 — confirm the premise (already running):** R3 3-rung discriminator
   (`--use-correct-plan --lora-only --save-delta`) on SMW+Sonic → check Δ_actor≫Δ_plan. De-risks: *is it
   actor-OOD at all.*
1. **Days 1–3 — GRPO weight:** swap `rwbc_actor_adapt.py` quantile `rw` (line 213) for save-state
   group-relative `Â`; validate on SMW from save-states (reward↑ vs RWBC, no collapse, `--anchor` on).
   De-risks: *critic-free save-state GRPO improves the actor.*
2. **Days 3–6 — active gate sampler:** wire `r_t`/`D_t` from `_sample_chunk` into `collect()`;
   trigger→`save_state`→branch-search→buffer; **log `Δ_plan-off`** (the hole probe). De-risks: *targeting
   OOD beats uniform collection* + measures the hole.
3. **Days 6–9 — generalist:** real `game_id` in `build_batch`, per-game GRPO normalization, balanced
   multi-game batches; joint SMW+Sonic+minish; check no per-game regression. De-risks: *one model, no
   collapse.*
4. **Days 9–12 — deploy gate + invariance:** Q3 ladder with conformal τ at eval; assert **bit-identical
   null-plan output**; eval on **held-out obscure/custom levels** (dodge VLM memorization). De-risks:
   *online detection + generalization, invariance intact.*
5. **Days 12–14 — resolve the hole:** ablate plan-head-trainable vs `--lora-only` **conditioned on
   `Δ_plan-off`** from Step 2. De-risks: *the plan-vs-actor authority question, with off-manifold data.*

---

### Citations (primary)
- **GRPO** — DeepSeek, arXiv:2402.03300. **AWR** — Peng et al. 2019, arXiv:1910.00177. **DAgger** — Ross,
  Gordon, Bagnell, AISTATS 2011, arXiv:1011.0686. **π0.5** (subgoal→flow chunk) — pi.website/blog/pi05.
- **Repo:** `rwbc_actor_adapt.py` (collect/build_batch/eval_reward; `--lora-only`:165,
  `--use-correct-plan`:170, `--save-delta`:171, `--explore-sigma`:168, `--plan-temp`:167, `--anchor`:166,
  `--top-frac`:160, `rw`:213); `eval_policy.py:195` `_sample_chunk` (`noise_sigma`/`v_c`:237/`v_u`:238);
  `emulator_env.py` `save_state`:199/`load_state`:203/`apply_chunk_capture`:186; `snes_env.py`:189/193;
  `retro_rl_env.py`:283–320 (`info{died,stuck,dprog}`); `nitrogen.py` `apply_null_mask`:594; `planner.py`
  `generate_plan`:238.
