# RL Design V2 — training the closed-loop System-2/System-1 stack (2026-06-27 night)

Supersedes/extends `RL_DESIGN.md` (2026-06-26). Folds in: (a) tonight's **closed-loop planner redesign**
(System-2-planning-for-System-1, with `<learnings>`/`<plan>` grounded on the prior plan + executed trace),
and (b) the **corrected per-game-prompt ablation** that *qualifies* the prior "plan value is binary" claim.
Goal: enumerate every RL-able surface and give each a concrete, tractable train recipe + credit path.

---

## 0. The big update since V1 (READ FIRST)

V1's headline (GPT-5.5 critique): *the DiT actor is the bottleneck; the plan's value is BINARY (it merely
activates the inert actor), so a better plan doesn't raise return → planner-RL is moot.* That was measured
**with the OLD miswired global prompt**.

Tonight, with **corrected per-game prompts** (`game_planner.GAMES`), the plan's value is **GRADED and
LARGE** on GT reward from the demo start states (`demo_planner_ablation.py`, deterministic matched seeds):

| game | base (null DiT) | plan (corrected) | Δ |
|---|---|---|---|
| SMW | +36.1 | +54.0 | +50% |
| Sonic | +37.1 | +140.2 | +278% |
| Minish | +0.7 | +1.1 | +57% |

And the **miswired→corrected prompt swing is itself huge** (the overnight bench had plan *hurting*
platformers at −0.23). **Conclusion update:** plan value is NOT binary — *plan TEXT quality drives big
return swings*. So **planner optimization is back on the table** as a first-class, complementary lever to
actor-RL. Both matter:
- **Actor (DiT)** has a *motor skill ceiling* → raise it with weight RL (RWBC→DDPO). (V1 still holds: actor
  IS adaptable with anti-collapse; it's often the wall on OOD games.)
- **Planner (text)** has a *steering bandwidth* that, when the prompt is genre-correct, grades return
  strongly → optimize plan TEXT (by selection/distillation; Qwen stays frozen).

These are not in competition; the staged plan trains both and their interface.

---

## 1. The surfaces (outer System-2 → inner System-1)

```
[prior plan + executed trace + learnings + frames]
   │  frozen Qwen3.5-2B (System 2)            ── SURFACE 1: <plan> text   (frozen-generated)
   │                                          ── SURFACE 2: <learnings>    (frozen-generated, memory)
   ▼  <plan> text + frames → frozen Qwen encode → text-token hiddens
   ▼  resampler (K=8) + adapter               ── SURFACE 3: the BRIDGE     (TRAINABLE)
   ▼  K plan tokens → frozen NitroGen DiT cross-attn (+ LoRA)
   ▼  flow-matching (stochastic via noise)    ── SURFACE 4: DiT-LoRA       (TRAINABLE)
   ▼  18-step action chunk → emulator → reward
```

**Hard constraint:** Qwen VLM stays FROZEN → surfaces 1–2 are *not weight-trainable*; 3–4 are. This is THE
design fact that shapes everything: you optimize the plan/learnings TEXT by **search + distillation into the
trainable bridge**, not by gradient on the generator (unless the freeze is relaxed to a Qwen-LoRA — gated).

| # | Surface | Trainable? | RL handle | What it buys |
|---|---|---|---|---|
| 1 | Plan text | frozen-gen | temperature (explore), **best-of-K selection**, **distill into bridge**, *(opt) Qwen-LoRA GRPO* | steer/grade return; now shown LARGE |
| 2 | Learnings (memory) | frozen-gen | same as 1, **or replace with a trainable latent memory** | cross-time credit, game knowledge — *value TBD (measuring)* |
| 3 | Bridge (resampler+adapter) | YES | RWBC / GRPO / distillation target | turn plan+frame → good action conditioning |
| 4 | DiT-LoRA | YES | RWBC → DDPO/DPPO, save-state GRPO | raise the motor-skill ceiling |

---

## 2. Training each surface

### Surface 4 (+3): the weight-trainable core — **start here, it's the wall**
Methods, in order of cost/power:
1. **Reward-weighted BC (RWBC)** — roll the policy, keep top-return rollouts, BC on them. *Verified* (V1:
   Sonic +1.32→+2.80, LoRA-only + low-LR, no collapse; naive LoRA+plan-head collapsed). Cheap, stable, but
   **self-imitation → sharpens, doesn't discover** → plateaus.
2. **Save-state GRPO** — from a frame-exact save-state, sample **K action-chunks** (the flow sampler is
   stochastic via the noise seed → free exploration), roll each, advantage = R_i − mean(R), weight the
   denoising-likelihood. **No critic, short credit path.** Needs frame-exact save/load (Sonic/SNES/GBA
   emulators have it; Genesis subprocess does too via stable-retro).
3. **Diffusion-policy PG (DDPO/DPPO)** — proper PG through the flow denoising steps. Highest ceiling, most
   unstable. **Graduate here only after RWBC plateaus.** Text-GRPO (verl/prime-rl) will NOT update the flow
   sampler — needs a custom denoising-likelihood loop.

Anti-collapse (from day 1): LoRA-only (small rank), low LR, KL/action-distance reg to base DiT, replay base
trajectories, **held-out start-state eval**, early-stop on **validation** return.
Anti-overfit: train/val save-state split, randomized no-op/frame offsets, held-out level sections.

### Surface 1: plan text — frozen Qwen → **search + amortize**
- **Best-of-K plan selection** (RL-free, inference): sample K plans (temperature), roll each via the DiT
  from a save-state, pick max-return. Directly cashes in the now-graded plan value. Cost = K rollouts.
- **Plan→bridge distillation** (train-time amortization): collect (frame, plan\*) from best-of-K; SFT the
  **bridge** so the *greedy* plan's K tokens move toward plan\*'s tokens (token-MSE or action-match). Bakes
  search gains into trainable weights **without touching Qwen**. This is the key trick for a frozen planner.
- *(Gated)* **Qwen-LoRA plan-GRPO**: if the user relaxes the freeze to a small Qwen-LoRA, GRPO on plan-token
  logprobs with downstream return (clean text logprobs; verl/prime-rl fit). Highest plan-quality ceiling.
  **Default: do NOT; flag for user.** (The freeze is a stated hard constraint.)

### Surface 2: learnings (memory) — **measure first, then decide**
- The redesigned behavioural learnings (grounded on prior plan + executed trace) are *qualitatively* much
  better (e.g. SMW: "controller's move-right+down,JUMP landed Mario on the ledge"; the model now reasons
  about adherence/efficacy). **But marginal value on return is being measured** (`abl2_*` running). V1's
  naive learnings HURT (−15.8/−59.4/−0.2).
- **If neutral/harmful:** keep as a *diagnostic* surface (it tells US where the DiT fails), do NOT bake into
  RL — in-prompt text memory makes the planner input **non-stationary** → destabilizes credit assignment
  (V1 critique, still valid).
- **If it helps:** two paths — (a) best-of-K / distill like the plan; (b) **replace text memory with a
  trainable LATENT memory** (a small slot/GRU that summarizes the trace → feeds the bridge). This turns
  "learnings" into a *trainable* surface (3.5), avoids the frozen-Qwen text round-trip AND the
  non-stationarity. Strong candidate if memory proves valuable.

---

## 3. Credit assignment (the hierarchy's hard part)
`frame → plan (slow, every A chunks) → K tokens → 18×A actions (fast) → sparse reward.`
- **Save-state GRPO is the cleanest mechanism for BOTH surfaces**: sample K (plans OR action-chunks) from
  the SAME state, rank by realized A-chunk return → no value net, no long backprop. Use it everywhere.
- **Two-timescale**: System-2 credited on its A-chunk-horizon return; System-1 on per-chunk return.
- Avoid a learned critic initially (long, noisy credit path frame→plan→tokens→actions→sparse reward).

---

## 4. Staged plan (refined; ablations mandatory)
0. **(running) Validate new surfaces**: corrected-plan helps (✓ measured, large); redesigned-learnings
   value (measuring `abl2_*`).
1. **Re-run actor-adaptation (RWBC) with the CORRECTED planner.** V1's RWBC used the miswired prompt; the
   corrected plan is a far better conditioning signal → RWBC should go further / cleaner. *High value, cheap,
   do first.* Game choice by actor-learnability+headroom (V1: TheXTech coherent but ceiling-capped; Sonic
   headroom but OOD; pick by re-measuring with corrected plans).
2. **Best-of-K plan selection → bridge distillation.** Cash in graded plan value; amortize into weights.
3. **Save-state GRPO** on bridge+LoRA (Sonic/SNES substrate; frame-exact).
4. **Alternating** plan-distill ↔ actor-RWBC/GRPO on held-out states.
5. **DDPO/DPPO** once self-imitation plateaus (+ KL-to-base on the action dist).
6. *(gated)* **Qwen-LoRA plan-GRPO** for plan-quality ceiling.
Ablations throughout: plan-head-only, LoRA-only, alternating, joint, ±learnings, ±best-of-K.

## 5. Reward
`+Δprogress − death − stuck/timeout + clear_bonus` (+ per-game terms). Reward vars: screen_x (smw/sonic),
top-down euclidean (minish), **FE needs a real reward** (turn-based: unit-state/objective via RAM or judge).
Emulator save-states = dense, checkable, counterfactual reward (the RL substrate).

## 6. Frameworks
- **Text-token policy** (Qwen-LoRA plan-GRPO, if used): verl / prime-rl (prime-rl already ran on this box).
- **DiT diffusion actor**: custom RWBC + denoising-likelihood PG loop (text-RL frameworks can't update it).

---

## 7. Open questions for the morning
- Does the redesigned **learnings** actually raise return (or only improve diagnostics)? (`abl2_*`)
- With corrected plans, does **RWBC** reach higher than V1's +2.80, and on which game (re-rank)?
- **Best-of-K → distill**: how much of the K-search gain amortizes into the greedy bridge?
- Latent memory vs text memory — worth the scope?
- Is a small **Qwen-LoRA** worth relaxing the freeze for? (user call)

---

## 8. GPT-5.5 rubber-duck critique (2026-06-27 night) + REVISIONS

A sharp critique (`rubber-duck`, gpt-5.5) gated the whole planner-side machinery behind ONE unproven
premise. Revisions, in priority order:

### R1 — "plan value is GRADED" is NOT yet proven; it's the GATE (run before anything else)
The §0 table compares **null (no plan tokens) vs corrected plan** — that still confounds **ACTIVATION**
(any non-null tokens wake the under-conditioned DiT) with **graded SEMANTICS** (better text → more return
*within* non-null plans). If it's mostly activation, best-of-K / distillation / memory / planner-RL are ALL
low-leverage. **Decision rule:** run a within-state, matched-seed plan-quality LADDER
(`planner_poc/plan_graded_test.py`): null, dummy(nonsense, same length), wrong_game, generic, bad(opposite),
correct, hand_good. If `dummy ≈ correct` → ACTIVATION → cut planner machinery, focus actor/bridge. If
`correct > bad/wrong_game/dummy` with margin → GRADED → proceed. **Everything planner-side waits behind
this gate.** [BUILT + RUNNING tonight: docs/graded/<game>.json, crash-safe incremental.]

### R2 — "save-state GRPO on the flow sampler" is NOT a valid policy gradient (rename it)
Reward-weighting the flow-matching velocity-MSE over K noise-seed samples = **elite regression /
advantage-weighted BC**, a biased self-imitation heuristic — NOT an unbiased PG. A correct diffusion PG
(DDPO/DPPO) needs the score of the *stochastic denoising trajectory* `A_i·Σ_t ∇θ log p(x_{t-1}|x_t,cond)`
with explicit Gaussian transitions + old-policy logprobs + KL/clip. Flow ODE sampling is deterministic given
the initial noise → final-action likelihood needs the CNF change-of-variables Jacobian, not the training
MSE. **REVISION:** call stage-3 "save-state best-of-K + advantage-weighted regression (RWBC)", treat as
heuristic self-imitation (fine, but label it honestly); reserve true DDPO/DPPO for later with the correct
stochastic-transition formulation, and validate that formulation on a TOY continuous-control task with a
known optimum BEFORE spending emulator compute.

### R3 — best-of-K → bridge distillation needs an EXPLICIT target (token-MSE is likely incoherent)
"Move the greedy plan's tokens toward the best plan's tokens" is a no-op unless the target is a frozen
teacher, and chasing a moving bridge causes target drift / text-insensitive latent overfit. **Two coherent
forms:** (a) **behavioral distillation (PREFERRED):** treat best-of-K as a data generator — roll the winning
plan, RWBC/advantage-weighted-BC the bridge+LoRA on the high-return action chunks it produced (directly
return-aligned); (b) **frozen-teacher latent distillation:** `z_tgt = stopgrad(B_old(h(c, p*)))`, minimize
`‖B_θ(h(c, p0)) − z_tgt‖²` with replay + slow teacher. Until amortization is proven, **keep best-of-K as
inference-time search.**

### R4 — bridge BANDWIDTH (K=8) may be the real bottleneck — cheap tests
If `correct ≈ hand_good ≈ ORACLE`, or if text can't separate behaviorally, the K=8-token bridge may be too
low-bandwidth to carry plan nuance (esp. Zelda/FE long-horizon). **Cheap disentanglers:**
- **Oracle-latent search** (HIGH VALUE): from a save-state, directly optimize the continuous plan tokens
  `z` (CEM/random search) to maximize return, bypassing text/Qwen. If optimized-`z` ≫ correct-text-`z` →
  the DiT IS steerable and the **text→token path is the bottleneck** (planner/distillation/latent-RL worth
  it). If optimized-`z` ≈ null → the **actor is the wall** (focus DiT-LoRA). [BUILDING: `latent_plan_search.py`]
- **K-sweep** (freeze DiT, retrain bridge K∈{1,2,4,8,16,32}): returns saturate early → K enough; scale with
  K → bridge bottlenecked.
- **Semantic-perturbation:** good vs paraphrase vs wrong-state vs shuffled, equal token count — can the
  bridge separate them behaviorally?

### R5 — memory/learnings: diagnostic-only until it wins a CLEAN ablation
The redesigned behavioural learnings are better, but in-prompt NL memory still makes the (frozen) planner
input NON-STATIONARY → destabilizes credit assignment. **Keep memory as logging/diagnostic** until an
ablation clearly wins; if it wins, prefer a **trainable latent trace-encoder** over NL text memory.
[Tonight's redesigned-learn ablation: Minish learn +1.3 > plan +1.1 (flipped from −0.2); SMW/Sonic learn
runs SEGFAULTED in stable-retro before printing — re-run needed, smaller scope.]

### CUT / postpone (duck): Qwen-LoRA plan-GRPO (until gate+bandwidth proven); DDPO/DPPO (until RWBC
plateaus + correct formulation); text-`<learnings>` in the RL loop; naive token-MSE distillation.

### Revised tonight/this-week ORDER (gated)
1. **GATE: graded-plan ladder** (`plan_graded_test.py`) — decides if planner-side is real. [running]
2. **Oracle-latent search** (`latent_plan_search.py`) — actor vs bridge vs text bottleneck. [building]
3. Only if GATE=GRADED: best-of-K → **behavioral** distillation; else go straight to actor RWBC w/ corrected plans.
4. Re-run RWBC actor-adaptation **with corrected plans** (V1 used miswired) — high value either way.
5. K-sweep bandwidth test if bridge looks limiting.

---

## 9. GATE RESULT (2026-06-27 night) — graded-plan ladder [docs/graded/*.json]

`plan_graded_test.py`, matched-seed, btn_s600_full, w=8, demo start states, 12 chunks, 2 seeds:

| game | null | dummy(nonsense) | bad(opposite dir) | correct | hand_good | **correct−dummy** | **correct−bad** | verdict |
|---|---|---|---|---|---|---|---|---|
| SMW | +23.6 | +4.3 | +4.9 | +28.0 | +23.8 | **+42.8** | **+49.1** | **GRADED** |
| Sonic| +23.9 | +19.1 | ~+45 | +75.9 | +65.6 | **+56.8** | **+15.8** | **GRADED** |
| Minish| +0.6 | +1.1 | +1.1 | +1.1 | +1.0 | −0.04 | −0.07 | ACTIVATION |

(SMW/Sonic exact per-condition in JSON; deltas are the printed summary.)

**VERDICT: the §0 premise HOLDS for side-scrollers.** Plan value is GRADED, not mere activation:
- `correct ≫ dummy` (a real plan beats equal-length NONSENSE by +43/+57) — rules out pure activation.
- `correct ≫ bad` (a "go RIGHT" plan beats a "go LEFT" plan by +49/+16) — DIRECTIONAL SEMANTICS drive
  return. This is the clean, confound-free signal.
- **Minish (top-down, euclidean-movement reward) is ACTIVATION-only** — any plan token ≈ same. Plan
  optimization is genre-gated: worth it where there's a directional progress axis the text can express.
- Note `hand_good < correct` on both side-scrollers (−4 to −10): VERBOSE plans slightly hurt vs a concise
  direction-forward plan (consistent with "direction word diluted in long plans"). Implication for the
  planner: bias to SHORT, direction-first plans.
- Caveat: `wrong_game` was a poor probe (the other side-scroller's plan is ALSO "go right" → not
  semantically wrong); `bad`/`dummy` are the valid semantic probes. n_states small (1–4) — treat magnitudes
  as indicative, directions are consistent across both seeds.

**=> GATE PASSED for SMW/Sonic.** Planner optimization (best-of-K selection → behavioral distillation) is
justified there. Minish/top-down: skip planner-side, focus actor. Next: latent-plan search (actor vs
text→token bottleneck) to choose the planner-side mechanism.

---

## 10. BEST-OF-K + LATENT-SEARCH results (2026-06-27 night) — the lever is PLANNER VARIANCE

### Best-of-K plan selection [docs/graded/bestofk_*.json] (winner's-curse-proof: select on train seeds,
score on disjoint val seeds)

| game | null | greedy(correct) | mean-of-K | best-of-K(sel) | oracle-of-K | val-spread | best−greedy |
|---|---|---|---|---|---|---|---|
| SMW | +7.5 | +48.2 | +34.9 | +47.5 | +52.2 | **+33.0** | −0.8 |
| Sonic| +0.0 | +41.5 | +35.0 | +42.2 | +43.8 | **+22.5** | +0.8 |

**Interpretation (important):**
- **Plan choice matters a LOT** (val-spread +33/+22 among K sampled plans) — re-confirms graded-plan from a
  3rd angle. Sampled GOOD plans ("keep running right") vs BAD ("crawl forward to move left", "move left
  hold down jump") differ by 30+ reward.
- **But best-of-K selection ≈ a good FIXED concise plan** (greedy≈oracle; selection −0.8/+0.8). The hand
  "correct" plan is already near the ceiling. Selecting among NOISY Qwen samples doesn't beat it because
  ~30–40% of temp-0.9 samples are WRONG-DIRECTION and drag the set down.
- **=> The bottleneck is PLANNER VARIANCE, not the ceiling.** The lever is RELIABILITY: short
  direction-first prompts, LOW temperature / greedy (greedy already emits "Move right to advance…"), a cheap
  plan FILTER/RANKER (reject wrong-direction), or distilling the good plan — NOT expensive K-rollout
  selection (it doesn't beat greedy). For INFERENCE, greedy is already good; variance only bites RL
  exploration (sampled plans are noisy → a plan filter or actor-side RL is preferable to plan-sampling RL).

### Latent CEM search [docs/graded/latent_*.json] — robust part + caveat
- **Robust:** with per-chunk-varying noise + multi-state, `text_correct ≫ null` (SMW +48 vs +7.5; Sonic +41
  vs +0) and **pt_inject ≈ text_correct** (faithful direct-token injection). Good sanity + re-confirms plan
  value.
- **Caveat (inconclusive on ceiling):** CEM over the 8192-dim (K=8×1024) plan-token space with pop=12 is far
  too weak an optimizer to beat a good text plan, so "CEM didn't exceed text" is NOT evidence the actor
  lacks headroom (absence of evidence). To actually test token-space headroom, use **gradient-based** z
  optimization (the DiT is differentiable wrt plan tokens → backprop a reward proxy) or CMA-ES with a big
  population — deferred.

### Revised highest-leverage levers (post-gate, post-best-of-K)
1. **Actor-side RWBC with corrected (greedy) plans** — the greedy plan is good; train DiT-LoRA to follow it
   better / raise the motor ceiling. (V1 RWBC used miswired plans → re-run.) **Top actionable item.**
2. **Planner reliability** (cheap, inference-ready): greedy + short direction-first prompt already strong;
   add a wrong-direction FILTER for any sampling path.
3. **Behavioral distillation** (duck R3): use the GOOD plans' high-return rollouts as RWBC data for the
   bridge+LoRA — amortizes the "good plan" into weights so even mediocre plans' tokens behave.
4. **Gradient-based latent-z probe** (replaces weak CEM) to settle the token-space-headroom question.
5. Cut: best-of-K *rollout* selection (doesn't beat greedy); naive token-MSE distillation; Qwen-LoRA (gated).
