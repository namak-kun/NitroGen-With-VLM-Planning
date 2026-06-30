# ARCH WARROOM — Round 10 seed: the SHORT path vs LONG path, and System-2 → System-1 consolidation

The owner re-opened the architecture debate with a new, sharper framing that may unify three open problems
(stale plans, actor-OOD, capability transfer). This round is CONCEPTUAL-but-RUNNABLE: converge on a design and
name the ONE experiment the orchestrator runs tonight on 4×A6000. Keep both agents in the loop. Goal unchanged:
ONE generalist (no per-game/genre models). VLM-LoRA now ALLOWED (owner). Null-invariance must hold.

## What's VALIDATED (build on it, do NOT re-litigate)
- **Routed recipe (one model, two MODES):** plan-OOD games (SMW/MMX/SMB1) → SUPERVISED plan-head demo-fit
  (LoRA+base frozen, fit to human actions on the correct plan, no env reward). actor-OOD games (Sonic) → plain
  local RWBC (lora). Merge = disjoint params, null-invariant.
- **R9 (just done):** demo-fit DESTROYED the bridge's plan-responsiveness for rare situational actions (duck:
  base steerable 1.2→10.8%, collapsed demo-fit 0%). Root cause = DATA (one constant terse plan → plain BC
  ignores the plan → drops rare actions). FIX = **KL-anchor** (terse plan + λ·L2 of plan-tokens to a frozen
  PRE-fit plan_head over a broad plan dist incl duck): SMW Δ_plan +45.1 (3/3 seeds, best), expressiveness
  battery R=0.83 (only PASS, preserves duck/retreat/up/wait). Situational per-chunk plans = 2nd (R=0.53, loses
  unlabeled actions = taxonomy treadmill). Tooling: demo_bc.py --kl-anchor / --situational-plans / --duck-probe;
  expressiveness_battery.py.
- **Dissociation:** SMW=plan-OOD (oracle plan helps), Sonic=actor-OOD (actor-adapt helps, plan redundant).
- **Eval = Δ_plan** from fixed demo save-states (low var); ≥3 seeds + bootstrap CIs; single-seed retro = noise.
- **Env decode (no-analog-stick consoles):** "analog stick == d-pad" is ENFORCED by the env (down-projection to
  console buttons); the LEFT STICK (JLX21/JLY22, 0.5±0.2 thr) is the load-bearing directional channel (every
  env reads it; demo map_action targets it). dpad dims (1-4) are secondary/ignored. Measure direction on the
  STICK.

## THE OWNER'S NEW FRAMING (this round's center of gravity)
"There's a SHORT path from the VLM to the DiT and a LONG, more explicit path. We're missing the part where
System-2 knowledge BECOMES System-1 relevant." Concrete examples he gave:
- **SMW state 7:** the human GRABS the fence/mesh and climbs it in the demo. The DiT (System 1) doesn't seem to
  know this maneuver exists (actor-OOD). Ideally the VLM teaches the DiT this capability so it becomes reflexive.
- **"Does the DiT know spin-jumping instantly kills a Rex?"** Unknown — runs never reach the Rex. But we have
  SAVE-STATES FOR EVERYTHING, so we can load a state AT a Rex and probe the base DiT directly.
- **"Carry over learnings from each attempt"** — accumulate experience across attempts and consolidate it.

### The orchestrator's reading (agents: validate, sharpen, or reject)
The framing splits cleanly into **capability EVOCATION vs capability ADDITION**:
- **SHORT path = EVOCATION** (bridge: resampler→adapter→K plan-tokens→cross-attn, + demo-fit/KL-anchor). Summons
  maneuvers the DiT ALREADY has in its action manifold. Ducking worked because the DiT can already duck — the
  plan just had to ASK. Fast, continuous, low-latency. "System-2 intent amortized into a System-1 reflex."
- **LONG path = ADDITION** (explicit text + DiT-LoRA, + now VLM-LoRA). EXPANDS System-1's repertoire — teaches a
  maneuver the DiT physically can't produce yet (grab-mesh), by training LoRA on demos that contain it.
- **Router = the actor-OOD per-dim probe** (does base DiT reproduce the human action from the frame alone?):
  YES → evoke (short path, cheap, plan-head only); NO → add (long path, DiT-LoRA on those chunks).
- **Consolidation loop (the missing piece the owner names):** the existing `game_planner.ClosedLoopPlanner`
  `learn` mode already does INFERENCE-TIME episodic memory (System 2 reviews System 1's execution, accumulates
  <learnings> bullets, carries them forward). The missing step = DISTILL accumulated learnings + gold narration
  back into the bridge/DiT WEIGHTS periodically (a "sleep/consolidation" phase): discover in-context (S2) →
  accumulate → consolidate into weights (S1) → reflex. This is also a stale-plan fix (continuous latent flips
  less than re-sampled text; carried learnings give cross-replan coherence; consolidated capabilities don't need
  to be re-stated every frame → shorter, more stable plans).

## CRITICAL GAP just discovered (relevant to all of the above)
The owner's GOLD NARRATION (demo_explanations.md: "spin jump on the rex to instantly kill", "duck to dodge the
bullet bill", grabbing meshes) is **NOT in training** — only 2/8 narration.json got parsed, and the TRAINER
(demo_bc.py) never reads them anyway (it uses terse BATTERY plans or action-derived SIT plans). So the exact
System-2 knowledge the owner wants in System-1 exists in text and never reaches the model. Wiring it in is the
cheapest possible test of the SHORT path.

## Questions for Round 10 (concrete; orchestrator runs your pick tonight)

**Q1 — Formalize & validate the evocation/addition split + router.** Is the actor-OOD per-dim probe (base DiT
reproduces human action from frame?) the right router between short-path (bridge) and long-path (DiT-LoRA)?
Define the EXPERIMENT: pick 2-3 SMW maneuvers from save-states — grab-mesh (state 7, owner says DiT lacks it),
spin-jump-kills-Rex (load a state at a Rex), duck (known evocable) — run the per-dim probe + a plan-conditioned
evocation test (does an explicit plan summon the maneuver?), and CLASSIFY each as evoke vs add. Name the metric
+ the threshold that says "addition required" (e.g. per-dim AUC<0.6 AND explicit-plan evocation <2× null).

**Q2 — The consolidation loop, concretely.** Is "System-2 → System-1" just (short) demo-fit-on-gold-narration +
(long) DiT-LoRA-on-OOD-maneuver-chunks, or does it need a SEPARATE consolidation objective? Is the `learn`-mode
<learnings> episodic memory the right substrate to distill from? Propose the minimal runnable version: e.g.
"collect closed-loop attempts from a save-state, keep the ones that progressed, BC the bridge(+LoRA) on them
conditioned on the learning/narration that produced them" — a self-distillation of System-2-discovered behavior
into System-1. Is this just save-state GRPO/RWBC with a narration-conditioned actor, or distinct? Pick ONE form.

**Q3 — Two-path bandwidth: add, don't overwrite.** R9 showed overwriting plan_head is destructive (needed a
KL-anchor to not collapse). Does the LONG path imply a SEPARATE module so the short path stays intact? Candidates
(all parked): zero-init FastPlanMod modulator (adaln side-path, planner.py:305-309 — adds authority without
touching the duck pathway, null-invariant by construction), K=8→32 plan-token bandwidth (note: breaks btn_s600
plan_head shape → retrain), a dedicated DiT-LoRA for added capabilities. Which is the right "long path" carrier,
and does it coexist with the KL-anchored short path? Recommend ONE.

**Q4 — Wire in the gold narration (the cheap short-path test).** The owner's narration is timestamped per-second
("00:14 duck", "00:16 spin jump on the rex"). Map it to chunks via timestamps and condition demo-fit on it
(replacing terse BATTERY), WITH the R9 KL-anchor. Predict: does gold-narration conditioning beat terse + SIT on
(a) Δ_plan, (b) the expressiveness battery, (c) a NEW grab-mesh / spin-jump evocation probe? What would FALSIFY
"richer real plans help" (e.g. Δ_plan within CI of terse AND no new maneuver evoked)? This is the orchestrator's
default tonight unless you redirect — sharpen it or propose better.

**Q5 — Super Metroid (exploration game) into the POOLED generalist, training-only.** The npz is trainable (no
env/progress-addr needed) but it's NOT a right-runner — "advance" plan conditioning is wrong for it. Does adding
an exploration game's demos to the pooled plan-OOD generalist HELP (more diverse plan→action coverage, supports
the evocation story) or HURT (off-distribution directional noise)? What plan should condition its chunks (gold
narration? VLM-generated objective? action-derived SIT)? Keep short — training-only, eval deferred.

## Orchestrator hygiene (not a war-room question)
Will also: ingest the full demo_explanations.md (only 2/8 parsed) → per-chunk narration; add the death/done
cutoff to the eval + recorders (RetroRLEnv already returns done; eval ignores it → models play through death
into menus). These are plumbing for Q4/Q1, run regardless.

## What to write
Each agent: TL;DR + Q1-Q5 (concrete, runnable, with metrics/thresholds/seed-counts) + one steelman of the other's
likely pick + the single highest-value experiment tonight. Terse, technical, numeric. Disagree with the
orchestrator's reading where warranted.
