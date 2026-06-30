# Feasibility: joint training, OPSD judge, deferral, save-state authoring, format-training
Written 2026-06-26 PM while @namak-kun is away. Answers the "check feasibility of the rest" ask.
Cross-refs: RL_DESIGN.md (staged RL plan), RL_PROMPT_FORMAT.md (interleaved obs), MORNING_BRIEF.md.

---

## 1. JOINT TRAINING (planner plan-head + actor DiT-LoRA together) — the explicit ask

### What "joint" actually means here (and the hard constraint)
Trainable params in the whole stack: **plan-head** (resampler+adapter, few M) and **DiT-LoRA** (0.52M).
The **VLM is frozen** and the **plan TEXT is sampled by the frozen VLM** — so we can train how
(frame,text)→K tokens are *projected/used*, but we **cannot change which plan text is produced** without
VLM-text-RL (user: VLM stays frozen, no traces). So "planner training" = re-tune the token projection +
deferral, NOT rewrite plan content. (Consistent with the OPSD diagnostic: fixed-good-plan ≈ Qwen-plan →
plan *content* isn't the bottleneck, plan *use*/actor is.)

### Verdict: simultaneous joint = UNSTABLE; alternating-RWBC also DEGRADES; alternating with an OPSD Phase-B + adaln_cond-freeze is the remaining feasible route
- **Simultaneous (train plan-head + LoRA together on reward) COLLAPSES** — measured: naive LoRA+plan-head
  lr1e-4 went +15.77→+1.58. Two coupled modules on a noisy reward-weighted-BC signal drift off the base
  manifold together (the plan-head can "explain away" reward with degenerate token shifts; LoRA follows).
  LoRA-ONLY (freeze plan-head) was stable. → do NOT co-train simultaneously.
- **Alternating / two-timescale IS the feasible joint recipe:**
  - Phase A — freeze plan-head, adapt **DiT-LoRA** via RWBC until reward plateaus. (PROVEN to work, 3 genres.)
  - Phase B — freeze LoRA, adapt **plan-head** to re-optimize the token projection for the *new* actor
    (RWBC, or OPSD-privileged distillation §2). Low LR + anchor.
  - Alternate A/B (two-timescale: actor fast, planner slow). True simultaneous joint only as a final,
    optional polish once both are individually stable.
- **THE key regularizer for Phase B: preserve masked-null invariance.** The whole CFG mechanism relies on
  `null-plan output == base-DiT output`. **MEASURED 2026-06-26 (joint_alternating.py): this does NOT hold
  automatically.** The null/plan-dropped path routes through `plan_head.adaln_cond(plan_tokens,
  dropped=True)` (eval_policy.py:209), so ANY plan-head update shifts the null output (masked-null drift
  0.0→0.085 across rounds). Worse, **Phase-B plan-head RWBC actively HURT reward** (sonic alternating
  trajectory: base +1.37 → r0A +2.42 [LoRA helps] → r0B +1.00 [plan-head hurts] → … → final −0.53, fully
  degraded). CONCLUSION (corrected): naive alternating-RWBC does NOT work. For a real planner phase:
  (a) use OPSD-privileged distillation, NOT RWBC, as the Phase-B objective; (b) FREEZE adaln_cond's
  dropped path (or add an explicit null-output L2 regularizer) to keep CFG intact; (c) honestly, given the
  actor is the lever and OPSD-diag showed plan CONTENT isn't the bottleneck, the simplest path is to keep
  Phase A (LoRA) only and invest the planner side in OPSD/deferral, not plan-head RL. joint_alternating.py
  now stands as the NEGATIVE control proving this.

### Why co-train at all (since the actor is the bottleneck)?
Three legitimate planner-side reasons — none of which is "discover new plan text":
1. **Token-projection re-tuning**: as DiT-LoRA shifts the actor, the optimal K-token projection shifts.
2. **Deferral** (§3) — a planner-side decision that genuinely needs training.
3. **Plan stability** (the user's ~50–60% replan flip-rate bottleneck) — a planner-side regularizer.
So Phase B is modest-headroom but real. Budget most compute to Phase A (actor).

### Compute/feasibility note
Both phases are reward-weighted BC (supervised flow-matching loss on elite rollouts) → cheap, single-GPU,
no critic, no rollout-replay buffer. The expensive part is ENV ROLLOUTS (collection), not the gradient.
Alternating just re-uses the same collect→filter→BC loop with a different param group frozen. So joint via
alternation costs ≈ 2× the single-module RWBC, well within the 4×A6000 budget. **Feasible now.**

---

## 2. JUDGE via OPSD + privileged info (user's stated method; NOT trace-distillation)

User ruled out SFT on bigger-model thinking traces (don't have real GPT-5.5 traces; don't want them).
Method = **OPSD-style self-distillation with GT/GPT predictions as PRIVILEGED INFO**:
- **Privileged student**: the SAME judge VLM, but its prompt INCLUDES the privileged signal — e.g. the GT
  emulator Δprogress, or GPT's verdict, stated in-context ("Given that the player advanced +213px, …") OR
  appended as a privileged token. With the answer in-context it produces a correct/confident verdict.
- **Unprivileged student**: the judge VLM seeing ONLY the frames (+ objective).
- **Distill privileged → unprivileged** (KL on the verdict distribution, or on a scalar progress head) so
  the unprivileged judge learns to INFER what the privileged one knew — no external traces, GT is free
  from the emulator at collection time.
- **Keep it DISCRIMINATIVE** (a scalar progress/verdict head on the frozen VLM hidden), NOT generative →
  avoids the hybrid-reasoning problem entirely (no think traces needed, LM head untouched, think mode not
  degraded). This is the clean answer to "training the VLM directly is problematic."
- Train only a small head (+ optional LoRA, low LR) on the frozen Qwen → the VLM's game-aware multimodal
  features become the judge, which is exactly the win over the SigLIP probe, without breaking generation.
- DATA: we already collect GT-labeled before/after frame pairs at rollout time (docs/rl_data/*). For
  no-GT domains (YouTube via cookies.txt), use GPT predictions as the privileged signal on a subset.
FEASIBLE and matches the constraint set. The SigLIP probe stays as the cheap baseline to beat.

**MEASURED (planner_poc/judge_vlm_opsd.py):**
- 253 pairs/3 games: VLM-unpriv 0.719 > SigLIP 0.688 (fp 0.12 vs 0.25). 413 pairs/4 games (+SMW):
  VLM-unpriv **0.761 > SigLIP 0.716**, recall 0.79 → the VLM-as-judge advantage is robust + strengthens
  with more/diverse data.
- OPSD privileged→unprivileged distillation: with the TRIVIAL hint-teacher (hint=GT-sign, teacher=1.000)
  distillation was FLAT (0.761). With a NON-TRIVIAL teacher (`--priv-mode future`: the teacher sees a
  future frame = outcome hindsight, NOT the label → teacher 0.734, genuinely not 1.0), **OPSD distilled
  0.752 > unpriv 0.734** → privileged self-distillation DOES help when the signal is non-trivial. Small-
  data-noisy but the mechanism is demonstrated. Your exact method (OPSD + privileged info, no traces,
  think untouched) works end-to-end. Next: more data, bigger look-ahead, GPT-verdict as an alt privileged signal.

---

## 3. DEFERRAL — needs active training (user agreed); two feasible paths
Probed: **0/5 zero-shot** — the frozen 2B never emits "NO GUIDANCE NEEDED" on its own. So the mechanism
(scaffolded in rl_planner_prompt.allow_defer) needs a trained policy. Signal already exists:
`rl_eval_plan_vs_null.py` gives per-situation (null reward, plan reward) → **label: defer-correct ⇔
null ≈ plan** (planning doesn't beat the inert/base actor).
- **Path A — output-token deferral (what the user literally wanted)**: RL/bandit on the single sentinel
  token. Tractable (one binary decision, dense-ish label) but IS VLM-token-RL → small but touches the
  frozen-VLM rule. Use a tiny LoRA + the plan-vs-null reward; keep it to the defer token only.
- **Path B — learned defer GATE (supervised, stable)**: a 1-layer head on the plan hidden → P(defer) →
  mask the plan tokens (inject null). Trained supervised on the plan-vs-null labels. No VLM-RL, no traces,
  collapse-free. NOT literally an output token, but functionally identical deferral.
RECOMMENDATION: prototype Path B first (cheap, stable, proves the signal separates), then promote to Path
A if you want the decision *in the text* for interpretability. Either way: do it AFTER the actor+reward
loop is solid (deferral on an inert actor just means "idle", which we know hurts platformers).

**MEASURED (planner_poc/deferral_gate.py, Sonic, frame-exact save/load, per-situation plan-vs-null):**
INCONCLUSIVE due to a real env limitation. On Sonic the labels are DEGENERATE — plan ≈ null per-chunk
almost everywhere (defer-rate ~100%, val majority 1.000), because the OOD Sonic actor barely moves with OR
without a plan, so the per-chunk plan-vs-null delta is ~0. The gate can't beat a single-class majority.
Oracle upside exists but tiny in absolute terms (as-is −0.036 → oracle-defer +0.048, Δ+0.084/situation, on
~0-scale rewards). LESSON: the deferral-learnability test needs an env that has BOTH (a) frame-exact
save/load AND (b) a COMPETENT actor (so plan-vs-null actually VARIES). No current env has both (TheXTech is
competent but proc/no-save-load; Sonic/SMW have save/load but OOD actors). FIX: RWBC-adapt the actor on a
save/load env FIRST (Sonic +1.5→+4.7 makes it more competent), THEN run the gate; or add save/load to a
proc env. Until then, deferral learnability is untestable on this substrate — flag for when a competent
emulator actor exists.

**FOLLOW-UP (planner_poc/deferral_after_adapt.py — RWBC-adapt Sonic THEN gate): a DEEPER correction.**
After adapting (Sonic +1.71→+3.62, more competent), the gate labels are STILL degenerate, but now in the
OPPOSITE direction: per-chunk **null_r (+0.205) > plan_r (−0.020)** — the base/null action advances MORE
per isolated chunk than the plan-conditioned one — EVEN THOUGH the plan helps at the EPISODE level (+3.62
adapted). KEY INSIGHT: **per-chunk plan-vs-null is the WRONG deferral signal.** The plan's value is
TEMPORALLY EXTENDED (sustained direction / replanning across chunks), not realizable in a single isolated
chunk; a myopic per-chunk advantage MISATTRIBUTES it (says "always defer" even when the plan helps over the
horizon). So the deferral label must be HORIZON-level (does the plan help reward over N chunks from here?),
not per-chunk save/load. This reframes the whole deferral-data design: roll plan-on vs plan-off for a
WINDOW from the same start state, compare windowed reward. Recorded as the corrected approach.

---

## 4. SAVE-STATE AUTHORING (your manual plan) — contract proven; you'll use external emulators
- **Contract VERIFIED**: a state = the raw libretro core blob from `env.save_state()`; `env.load_state()`
  round-trips FRAME-EXACT. Stored to `tmp/states/<game>/<label>.state`.
- **You'll author states in external emulators directly** (not through our env). The ONLY constraint is
  CORE/FORMAT MATCH — verified core + savestate formats on this box:

  | Console | Our core (stable-retro) | Savestate magic/version | External states load if… |
  |---|---|---|---|
  | Genesis | Genesis Plus GX **1.7.5** | `GENPLUS-GX 1.7.5` (binary, ~1.01MB) | from GPGX **1.7.5** (RetroArch core). Standalone Kega/BlastEm/Gens = different format, NO. |
  | SNES | snes9x (libretro) | `#!s9xsnp:0009` (snes9x native snapshot v9, ~430KB) | from any snes9x (standalone OR libretro) emitting **snapshot v0009**; newer Snes9x may bump the version. |
  | GBA | mGBA (libretro) | binary, version `0x01000002` (~397KB, embeds game code) | from a matching **mGBA** version (standalone .ss OR RetroArch mGBA core). |

- **RetroArch wrapping**: RetroArch wraps savestates in a `RASTATE` container (raw core block inside a
  `MEM ` chunk). `planner_poc/ingest_external_state.py` auto-detects + unwraps it, then VERIFIES the state
  loads into the matching core and copies the normalized raw blob to tmp/states/<game>/<label>.state (+
  json with the reward-var value). **Workflow: author ONE test state, run the ingest tool, confirm
  `LOADS = True` before authoring a pile.** Tool validated (Genesis state -> LOADS=True, reward 1877).
- **No recording tool needed** (per your call). `emulator_state_authoring_server.py` exists as a fallback
  (browser play + snapshot through our core, guaranteed-compatible) if external states ever mismatch.

---

## 5. TRAIN WITH THE INTERLEAVED FORMAT — cost & feasibility (the gap you spotted)
Confirmed: **no training has used the interleaved sub-chunk format** (f0,s1_1,f1_1,…). All training used
`generate_plan(last-4-frames)` + single-frame `encode_multimodal`. Stage-1 trains the resampler/adapter on
a PRECOMPUTED CACHE of VLM hiddens (cache_mm_hidden.py → train_planner --mm-plan-hidden-lookup).
- To train WITH the new format: build the plan-hidden cache from the interleaved prompt instead — i.e. run
  `encode_multimodal` over [f0, sub-frames…] + the interleaved action text (build_rl_messages already
  assembles exactly this). The resampler/adapter then learn to distill tokens from the richer obs.
- COST: mechanical — a new cache builder mirroring cache_mm_hidden.py but using build_rl_messages content;
  the trainer path is unchanged (it just reads the cache). The hidden is bigger (more frames+text) → more
  cache compute + VRAM per example, but it's precomputed once (frozen VLM).
- FEASIBILITY: straightforward; the main question is whether the richer obs actually improves token quality
  (the interleaved A/B showed think-mode plans ground better — but that's generation, not the cached
  resampler path). Worth an ablation AFTER the actor loop, since plan *content* wasn't the bottleneck.

---

## 6. PREV-PLANS — EVALUATED (the other gap you spotted) → naive impl HURTS, doesn't explore
`include_prev_plans` (append last ≤2 plans) ablated on Sonic (planner_poc/eval_prevplans_ablation.py,
3 seeds × 6 cycles, matched start/seed, interleaved think-mode obs):
- **prev-plans ON LOWERS reward** (Δ ON−OFF = **−7.45**; OFF +25.3/+34.1 vs ON +11.9/+15.8/+22.1 per seed).
- **It does NOT increase plan diversity** — flip_rate identical (1.00 both), but UNIQUE-plan fraction went
  DOWN with ON (0.50–0.83 vs 1.00 OFF). I.e. showing the last 2 plans induced plan **oscillation/cycling**
  (A,B,A,B → always "flips" but small unique set), NOT exploration.
- VERDICT: the naive "paste the last 2 plans" knob is counter-productive here (your "maybe ablate?"
  instinct was right). If we want exploration it needs a real mechanism (e.g. an explicit "try something
  DIFFERENT because X failed" instruction tied to a stuck-detector, or actor-side noise), not raw history.
  Small n (3 seeds, Sonic only) — directional, but the sign is consistent across every seed. Worth a
  re-check on a game where the planner genuinely loops (TheXTech stuck-at-obstacle) before fully discarding.

---

## Priority recommendation (for the morning)
1. **Actor RL scale-up** (Phase A) on the new graded TheXTech suite + SMW once envs land — the bottleneck.
2. **State-authoring tool** ready for you to author level start/end states (enables targeted RL + eval).
3. **OPSD discriminative judge** (privileged GT→unprivileged) — the reward path for no-GT/eval.
4. Planner-side (Phase B / deferral-gate / prev-plans / interleaved-format) — modest headroom, do after 1.
5. Joint = alternating two-timescale with masked-null regularizer — once A and B are each stable.
