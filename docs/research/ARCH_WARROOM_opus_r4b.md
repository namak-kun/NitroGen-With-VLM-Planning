# ARCH WARROOM — Round 4b (OPUS-4.8): the discriminator data, reconciled

2026-06-28 · my R4 recipe vs the DISSOCIATION. Constraints unchanged: ONE generalist,
frozen-VLM-first, null-invariance.

**Verdict up front:** the clean game-dependent split (SMW plan-OOD Δ_plan **+0.20** / Sonic actor-OOD
Δ_actor **+1.90**; DISCRIMINATOR_RESULTS.md) **vindicates the self-correcting core of R4 and kills any
global `--lora-only`.** R3 ("plan never load-bearing") holds for Sonic; GPT-5.5's steelman ("plan
re-acquires authority where the frame prior is weak") holds for SMW. The dissociation is the synthesis.

## 1. Does it change the prescription? No — it confirms the routing, and hardens it
R4 §4 logged `Δ_plan-off` and kept the plan-head trainable iff the plan helps. The data proves it:
`--lora-only` **ZEROED** SMW (Δ_actor −0.01) while the oracle plan was SMW's **only** lever (+0.20). The
recipe stands, with three locks:
- **Params:** keep the generalist's **plan-head + PlanAdapter trainable globally** — drop `--lora-only`
  (rwbc_actor_adapt.py:188 default trains `lora_*` **and** `plan_head.*`). Never freeze the plan path.
- **GRPO advantage weight:** per-game `Â=(R−mean_g)/(std_g+ε)` (R4 §3). This is what lets SMW's +0.20
  plan-gradient and Sonic's +1.90 actor-gradient coexist in ONE shared LoRA+plan-head: once unitless,
  Sonic's large raw screen_x can't swamp SMW's smaller-but-real plan signal.
- **Routing = emergent, not hand-coded:** one model; the per-game normalized advantage decides where
  gradient flows (no per-game heads — stays one generalist).

## 2. SMW: the plan-head IS the move, and it smells like adapter capacity
`--lora-only` did nothing ⇒ actor coverage is NOT SMW's lever; the plan→action pathway is. Train
**plan_head + PlanAdapter** (planner.py:103) under save-state GRPO, **not** LoRA-only. This likely IS the
repo's known **adapter-collapse** pathology: loss localizes to PlanAdapter; left/right token separation
only 0.50→0.688 after label cleaning (AGENTS.md). SMW's "jump the pit/enemy" is the same kind of
under-represented decision token. Concretely: (a) run the left/right-style token-separation probe on SMW
jump/wait plan tokens; if low, clean contrastive labels (direction-consistent chunks only) and/or widen
resampler K; (b) **wire replan to the DRIFT GATE**, not cadence (eval_policy.py:125) — the authority map
shows SMW authority RISES +14% at high-D_t, so a fresh objective AT the obstacle is where it pays. Map
caveat: **authority≠benefit**, so the training signal must be emulator reward (did the jump clear the
pit), not authority magnitude.

## 3. Sonic guardrail for the 3.22 ceiling
The +1.90 is partly real (survival-weighted reach_eval 0.584>0.250) but raw screen_x **saturates** ⇒ hack
risk. Add to the GRPO loop: **multi-component reward** (screen_x + survival + level-checkpoint, not
screen_x alone); **clamp the per-branch screen_x advantage** so a ceilinged 3.22 branch can't dominate the
group baseline; **cap reuse of any save-state** (branch-search collapse); `--anchor` KL + entropy floor;
**video + held-out save-state audit** on top-ranked rollouts.

## 4. Single most important decision + next experiment
**Decision:** ship the generalist with **plan-head trainable + per-game GRPO advantage-norm**; reject
global `--lora-only`. The dissociation means one fixed knob cannot serve both genres.
**Next experiment (one flag flip, existing harness):** re-run SMW R_C with plan-head **trainable** (drop
`--lora-only`) from save-states — the missing discriminator cell. Does LoRA+plan-head GRPO beat both R_A
(+0.05) and the zeroed lora-only R_C? That settles whether SMW's oracle-plan lever is **trainable into the
actor** or only oracle-available — the largest remaining uncertainty, cheap, today. Fold in GPT-5.5's R_D
(replan-at-drift) as the second cell.

### Citations (primary)
Repo: rwbc_actor_adapt.py:165/188, planner.py:103 (PlanAdapter), eval_policy.py:125;
DISCRIMINATOR_RESULTS.md; AGENTS.md (adapter-collapse). GRPO arXiv:2402.03300; AWR arXiv:1910.00177;
DAgger arXiv:1011.0686.
