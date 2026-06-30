# ARCH WARROOM — Round 12 seed: action-grounded VLM plans + the path to System-1 capability addition

Big new empirical result since R11, plus new EXPLORATION-genre demos. This round: turn the findings into the
concrete training plan for the next phase. Both agents: read the new results, then answer Q1-Q5 with runnable
recommendations (the orchestrator runs your pick on 4×A6000). Constraints unchanged: ONE generalist (no per-game/
genre models), null-invariance exact, VLM-frozen-first but Qwen-LoRA allowed, GPU-frugal, kill by numeric PID.

## NEW since R11 (build on these — they reshape the plan)
1. **STALENESS IS TEXT-BOUND (R11, SMW+MMX 3 seeds).** The K=8 plan tokens are remarkably STABLE chunk-to-chunk
   (token-drift ~0.001); re-grounding them on the fresh frame (forward-pass TTA) does NOT recover staleness.
   The ceiling gap is plan TEXT QUALITY (curated oracle >> live VLM by ~20 on SMW). On MMX the VLM's LIVE
   re-planning is the WORST non-null mode (a fixed good plan beats it). => the stale-plan lever is PLAN-TEXT
   QUALITY/STABILITY, not token-TTA, not actor weight-TTT. (Opus's R11 falsifier fired cleanly.)
2. **MANEUVER ROUTER (R11).** On the frozen DiT, from save-states reconstructed at the maneuver: the JUMP
   primitive is strongly evocable by an explicit plan (7.2x) BUT the spin-jump-KILL OUTCOME barely moves
   (survive+advance past the Rex 4%->7%). So 'spin-jump-kills-Rex' is an ADDITION at the OUTCOME level (route to
   DiT-LoRA on human Rex chunks), even though the jump button is evocable. Duck is STATE-DEPENDENT (reflexive
   when a bullet is actually approaching: base DiT ducks 46% there; evocable from generic starts). NOTE: SMW
   'score' RAM does NOT register kills (verified) -> use sprite-status RAM or survive+advance.
3. **GOLD-ACTION VLM NARRATION WORKS (today, the breakthrough).** Feeding the VLM the INTERLEAVED gold actions
   (frame, [action held next: Down=DUCK / Spin-jump(A) / ...], frame, ...) makes it NAME the situational
   maneuvers it otherwise misses. Gemma-4-12B on the SMW demo: maneuver-word counts went frames-only
   {duck 0, spin 0, stomp 0} -> +control-schema {duck 0, spin 2, stomp 12} -> +schema+GOLD-ACTIONS
   {duck 8, spin 5, stomp 17} (generic 'jump' 67->50). It correctly produced 'Duck to slide under the moving
   Bullet Bill' -- the duck it NEVER named from frames alone. The two levers STACK (schema = vocabulary; gold
   actions = evidence). Tools: planner_poc/vlm_narrate_demo.py (--interleave-actions, --game schema),
   PlanEncoder.generate_interleaved. enable_thinking=False needed for reasoning VLMs. Residual: enemy NOUNS
   still wrong (Rex->Koopa) but the MANEUVER is right (fine for conditioning). Bigger same-family planners load
   drop-in (Qwen3.5-9B/27B same arch as the 2B; Gemma-4-12B/26B-A4B/31B also load). DeepSeek-V4-Flash is
   text-only (no vision).
4. **VALIDATED RECIPE (R9):** plan-OOD demo-fit of the plan-head with a KL-ANCHOR (functional L2 of plan-tokens
   to a frozen pre-fit head over a broad plan dist) = the winner: SMW Δ_plan +45.1 (3/3), expressiveness battery
   R=0.83, null-invariant. Plain BC collapses rare actions (duck 10.8%->0); KL-anchor preserves them. Pooled fit
   across plan-OOD games generalizes (R8). Owner: do NOT use the raw human narration as training plans (too
   granular, mixes S1/S2) -- use VLM-ABSTRACTED plans; the gold-action narration above is exactly that source.
5. **NEW EXPLORATION-GENRE DEMOS + annotations (today).** Super Metroid (2 demos, npz recovered from bk2; 1
   human-annotated for stage-0, ~1:22, STOPPING BEFORE the Ridley boss fight + escape sequence -- owner: the
   boss fight is 'just hard to annotate', the escape 'broken narration looked weird'). Minish Cap (1 demo,
   human-annotated first ~3:24: top-down roaming, dialogue-skip, grass pick-and-throw, signboard reading -- NO
   combat/action yet; 'the initial few minutes are the most valuable'). These are NON-right-runner genres
   (top-down exploration, dialogue, item manipulation) -- a different objective space than the platformers.

## Lit anchors (from R11, reuse): ReST-EM (arXiv:2312.06585) = our consolidation loop; DAgger (arXiv:1011.0686,
O(εT²)) = covariate-shift fix for addition; KL-anchor = CoTTA stochastic-restoration (arXiv:2203.13591) in
functional form; EATA Fisher (arXiv:2204.02610) = forgetting guard; plan tokens = fast weights / implicit TTT
(Schlag arXiv:2102.11174, von Oswald arXiv:2212.07677); RT-H/LAPA 'language motion' = our VLM-abstracted plan.

## Questions for Round 12 (concrete + runnable; the orchestrator runs your pick)

**Q1 — Gold-action VLM plans as TRAINING conditioning (the immediate high-value move).** We can now generate
action-grounded VLM plans that name duck/spin-jump per chunk. Plan: generate per-chunk VLM-abstracted plans
(gold-action interleaved) for the plan-OOD demos, then re-run the KL-anchored pooled demo-fit conditioned on
THESE plans (replacing the terse BATTERY plan). Predict vs terse+KL (R9 winner) on (a) Δ_plan, (b) the
expressiveness battery R, (c) duck-on-command. Does action-grounded conditioning beat terse, and does it fix the
duck-COMMAND collapse (terse demo-fit emits 0% duck)? Define the exact run + pass bar + falsifier. Is this the
single highest-value experiment, or is the router-confirmed ADD (Rex DiT-LoRA) higher value?

**Q2 — Capability ADDITION (the long path), now actionable.** The router says spin-jump-kills-Rex is an ADDITION.
Concretize: DiT-LoRA (plan-token-gated for null-exactness) on the human's Rex-kill chunks (gold actions give us
the exact frames), conditioned on a 'spin-jump the Rex' VLM plan, with the EATA/Fisher or replay forgetting
guard. What's the eval (survive+advance past the Rex from the reconstructed save-states; need a kill-vs-dodge
signal -> sprite-status RAM)? Minimal runnable version + pass bar. Does it merge cleanly with the KL-anchored
plan-head (disjoint params, R8/R9)?

**Q3 — Plan-text quality/stability (the staleness lever).** Given staleness is text-bound: rank the levers —
(a) bigger/better planner (Qwen3.5-9B/27B or Gemma-4 as the deployment planner, not just the labeler),
(b) Qwen-LoRA distilling the gold-action plans into the 2B planner (cheaper at inference), (c) replan-less /
hold a stable abstracted plan (the R11 'fixed plan beats live re-planning' finding), (d) the gold-action
prompting at inference (but we have no gold actions at test time -- only at training). Which to test, and how to
measure 'plan quality' without the oracle (emulator advance from the live plan vs null, survival-weighted)?

**Q4 — Exploration-genre demos (Metroid, Minish): what do they ADD and how to use them?** They're NOT
right-runners. (i) Do they fold into the pooled plan-OOD generalist as POSITIVE plan-diversity (wait, climb,
talk, pick-throw, backtrack -- the situational actions platformers lack), or do they need a separate eval
coordinate (no screen_x progress)? (ii) The owner annotated only the early non-combat parts (Minish exploration,
Metroid stage-0) and EXPLICITLY left out the Ridley boss fight ('hard to annotate') + escape ('broken narration
looked weird'). Does the gold-action VLM narration (which needs no human annotation -- the action IS the label)
SOLVE the un-annotated-boss-fight problem -- i.e. can we auto-narrate the Ridley fight from its gold actions
where the human couldn't? Propose the experiment. (iii) What eval works for top-down exploration (Minish has no
'advance' axis -- reach-a-room? dialogue-advanced? item-count RAM?)?

**Q5 — The ONE coherent next-phase plan + tonight's experiment.** Synthesize Q1-Q4 into the priority order and
name the single highest-value runnable experiment tonight (metric, pass bar, falsifier, seed count). Respect:
KL-anchor, null-invariance exact, >=3 seeds + bootstrap CIs, VLM-abstracted (not raw) plans, survival-weighted
death-aware eval (eval_common.survival_advance).

## Format
Each agent: TL;DR + Q1-Q5 (concrete, runnable, cite the relevant prior result/lit) + a steelman of the other's
likely pick + the single experiment tonight. Terse, technical, numeric. Disagree with the seed where warranted.
