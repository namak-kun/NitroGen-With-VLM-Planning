# RL Design — plan-conditioned NitroGen (2026-06-26 night, incl. GPT-5.5 rubber-duck critique)

Synthesis for the morning brainstorm. Built on the night's findings + a sharp GPT-5.5 critique.

## What we now have (built + verified this night)
- **3 RL-ready envs with ground-truth reward:**
  - `nitrogen/eval/envs/retro_rl_env.py` — Sonic2 (Genesis, stable-retro). Dense screen_x reward,
    FRAME-EXACT save/load (→ save-state GRPO substrate), our-own start states. (1007 integrations avail.)
  - `nitrogen/eval/envs/proc_rl_env.py` — TheXTech (platformer, Δx) + Solarus (topdown, move+map). Semantic
    state reward; NO frame-exact save/load (reset=relaunch) → PPO-style, not GRPO.
- **Controlled result:** `planner_poc/rl_eval_plan_vs_null.py` — same-start GT-reward plan-vs-null on Sonic:
  NULL(base DiT) +0.00 (inert), PLAN +1.28. Plan ACTIVATES the inert DiT. Abs reward low (OOD).
- **Judge finding:** the cheap Qwen-2B judge is NOT reward-grade (over-credits, inverted on shmup/puzzle);
  GPT-5.5≈GT (0.94). Use GT state for reward; keep judge-calibration a SEPARATE track.

## THE CRITICAL REFRAME (GPT-5.5 critique — internalize this)
The decisive question is NOT "can plans improve reward?" (yes — they activate the actor). It is:
> **Can DiT-LoRA acquire new closed-loop MOTOR skills from emulator reward without collapsing/overfitting?**
The DiT (actor) is the bottleneck almost everywhere. The plan-token interface may be too low-bandwidth to
repair an OOD motor policy — if so the project becomes "adapt the diffusion actor to a game via sparse
reward", planner secondary. Everything below depends on the actor-adaptation test (Priority 1).

## Corrected RL plan (was: "joint GRPO on the plan")

### ⚠️ Technical correction: two DIFFERENT objectives, do not conflate
- **Planner (plan-head / VLM-LoRA):** GRPO/PPO over plan-token logprobs (text-like, clean logprobs). OK.
- **Actor (DiT-LoRA):** the action sampler is a FLOW/DIFFUSION process — text-GRPO will NOT update it
  correctly. Need a diffusion-policy RL method: **DDPO/DPPO-style denoising-step likelihood ratios**, OR
  **reward-weighted behavior cloning** (filter high-return rollouts → BC) if full diffusion-PG is too
  expensive. "Joint GRPO" is invalid unless the DiT sampler has a real likelihood-ratio path.

### Staged training (do NOT jump to joint) — with ablations
1. **Actor-only LoRA warmup** (THE critical test): FIX the plan distribution (fixed simple plans:
   "move right", "jump over obstacle"), train DiT-LoRA via reward-weighted BC / diffusion-PG on emulator
   reward. Eval on HELD-OUT start states. If LoRA can't improve here → joint RL won't work → pivot.
2. **Planner-only RL:** freeze DiT-LoRA, optimize plans (GRPO/PPO).
3. **Alternating** planner/actor updates on held-out start states.
4. **Joint** only after 1-3, with KL constraints on BOTH plan dist and DiT action dist.
Ablations required: plan-head-only, LoRA-only, alternating, joint. Without these we won't know what works.

### First RL game = by ACTOR LEARNABILITY, not Sonic (which is OOD)
**DECIDED (measured this night, planner_poc/rank_rl_games.py):** btn_s600+plan GT-reward over 12 chunks:
| game | reward | stuck_frac | verdict |
|---|---|---|---|
| **thextech** | **+15.75** | **0.00** | BEST first RL target (Mario platformer ~ NitroGen pretraining; actor coherent) |
| solarus_zelda | +0.76 | 0.67 | actor mostly stuck (topdown OOD) |
| sonic | +0.11 | 0.75 | actor mostly stuck (Genesis OOD — duck's warning confirmed) |
=> **Start RL on TheXTech.** Tradeoff: TheXTech is a PROC env (PPO/RWBC, reset=relaunch, NO frame-exact
save/load → no save-state GRPO); Sonic has the GRPO save-state substrate but the actor is OOD there.
Ideal-but-unavailable: a SNES/Genesis platformer that BOTH matches a stable-retro integration (frame-exact
GRPO) AND the actor handles — our ROMs (All-Stars, MegaManX) don't SHA-match integrations. So: actor-
adaptation test on TheXTech via PPO/reward-weighted-BC first; use Sonic for save-state GRPO experiments.

**METHODOLOGY REFINEMENT (found during the RWBC test):** "actor learnability" needs BOTH coherence AND
HEADROOM. TheXTech is coherent but its reward CEILING-CAPS (~+15.77: the policy already reaches the short
level's right-edge menu-strand) → NO headroom → RWBC there can only show collapse, not improvement. Sonic
has headroom (policy +0.11 ≪ a good run) but the actor is OOD. The clean first RL game needs both → either
fix TheXTech (longer level / remove menu-strand termination → headroom) or accept Sonic's OOD-but-headroom
for the adaptation test. Running RWBC on BOTH to see which gives a clean signal.

### Anti-collapse (DiT-LoRA) — from day one
small LoRA rank; KL/action-distance reg to base DiT; replay pretrained-action trajectories if available;
held-out game/state eval each ckpt; early-stop on VALIDATION return not training reward.

### Anti-overfit (save-state GRPO) — from day one
train/val split of save-states; randomized no-op/frame-offset starts; multiple level positions; held-out
sections; if gains appear only on training snapshots → treat as FAILURE.

### Reward (anti-exploit; screen_x alone trains a fast-suicide policy)
reward = +Δprogress − death_penalty − stuck/timeout_penalty + level_clear_bonus (+ ring/damage terms);
discount progress just before death. Track SUCCESS METRICS separately from reward: distance, survival
time, death rate, stuck rate, completion. (RetroRLEnv has score+done-on-life; ADD death & stuck penalties.)

### De-prioritized to ABLATIONS (not pillars)
- THINK MODE: with a frozen VLM it may add verbose non-causal traces; measure downstream RETURN, not plan
  quality. Compare {no-think short, structured {intent,dir,hazard,timing}, free-think, GPT-plans}.
- In-prompt EXPERIENCE MEMORY: changes the planner input distribution mid-RL → non-stationary, destabilizes
  credit assignment. Do CONTROLLED exploration first (temperature/top-p, entropy bonus, plan-diversity
  penalty, diffusion action noise, start-state randomization). Add memory only after the basic loop works.

### OPSD (user's GPT-5.5-plan-distillation dart) — diagnostic only
On-Policy Self-Distillation = STUDENT generates on-policy rollouts; a TEACHER (EMA/snapshot, or stronger
model) gives per-token target distributions on the STUDENT'S OWN trajectories → KL-distill (fixes exposure
bias / train-infer gap; student improves where it actually visits). Ref: "Self-Distillation Improves
Policy Iteration" (2204.02969).
- **Caveat for the GPT-5.5-teacher variant:** GPT-5.5 ≠ Qwen tokenizer/arch → CANNOT do per-token KL.
  It degenerates to SEQUENCE-LEVEL imitation (on-policy DAgger): GPT-5.5 writes target plans for the
  student's visited frames; student SFTs toward them. Fine, but it's imitation, not true OPSD.
- **Per the duck (correct) + DIAGNOSTIC RUN (planner_poc/opsd_diag.py):** helps ONLY if the PLANNER is the
  limiting factor; our actor is. The diagnostic CONFIRMED it: on Sonic a fixed hand-written "good" plan ≈
  Qwen plan (+1.25 vs +1.13, within noise), both ≫ null (−0.60) → the plan's value is BINARY (activate the
  inert actor) NOT graded → a BETTER plan does NOT raise DiT return → the actor is the wall → **OPSD/GPT-5.5
  plan-distillation is MOOT.** Do not spend on it; invest in actor-side RL.

## RL framework: verl vs prime-rl
prime-rl was already used on this box (the killed vllm serve was prime-rl). Both do GRPO/PPO for LLMs.
Caveat: both are built for TEXT-token policies → they fit the PLANNER objective, NOT the DiT diffusion
actor. The DiT needs a custom diffusion-policy loop (DDPO/DPPO or reward-weighted BC) — likely hand-rolled.
Plan: planner-RL via prime-rl/verl (logprob policy); actor-RL via a small custom diffusion-PG/RWBC loop.

## Concrete next steps (ordered)
1. Game-learnability ranking (this night): run btn_s600+plan on Sonic/TheXTech/Solarus, measure GT reward
   + behavior coherence → pick first RL game.  [DONE: TheXTech, see above]
2. Actor-adaptation test (THE critical experiment): fixed plans + DiT-LoRA reward-weighted BC on the chosen
   game; held-out start states; does return improve without collapse?
   [DONE — PRELIMINARY YES. planner_poc/rwbc_actor_adapt.py. Careful LoRA-ONLY(0.52M)+low-LR RWBC: Sonic
   reward +1.32→+2.80 (>2×, no collapse, monotonic); naive LoRA+plan-head collapsed (+15.8→+1.6); TheXTech
   ceiling-capped (flat, no headroom). => actor IS adaptable from reward w/ anti-collapse constraints.
   NEXT: (a) scale + characterize the self-imitation PLATEAU; (b) add EXPLORATION (plan temperature +
   diffusion action noise) so it discovers NEW skills not just sharpens; (c) HELD-OUT start-state eval;
   (d) fix TheXTech headroom (longer level) for the actor-coherent+headroom best case; (e) graduate to real
   diffusion-PG (DDPO/DPPO) + KL-to-base once self-imitation plateaus.]
3. Harden reward (death/stuck penalties + success metrics).  [DONE in retro_rl_env.py]
4. OPSD diagnostic (GPT-5.5 plan vs Qwen plan return on the chosen game).
5. Build train/val start-state splits.
