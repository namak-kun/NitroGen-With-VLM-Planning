# FORWARD REPORT — System-2 → System-1 Knowledge Transfer (NitroGen VLM-planner)

**Status:** COMPLETE (War-room R13 synthesized). The "Positive results" section is settled (multi-seed,
grounded in `files/DISCRIMINATOR_RESULTS.md`). The "Mechanism / forward path" sections synthesize War-room R13
(GPT-5.5 + Opus-4.8, who converged) + lit (`LIT_R13_S2TOS1.md`, 24 arXiv-verified papers). Companion: Task-2
RAM-free demo verification (`VERIFY_SMW.md`, `VERIFY_CROSSGAME.md`) is being produced in parallel.

The owner's north star (verbatim-ish): *"Discovery becomes instinct which drives higher abstractions of
discovery."* A maneuver the **System-2** VLM evokes in **System-1** should become **owned/instinctual** by S1
in that game, so S2 can stop naming it and compose at a higher level. Shared capabilities exist across games
but are *"subtly different enough that the broad concept holds but not the execution"* → we need online
**addition/tweaking**, not offline LoRA (*"nobody plays games like this"*). Informally: **option learning,
options discovered by System-2.** Caveat from the owner: don't overindex on the north star; a game-specific
LoRA is a fine bootstrap past the single-level plateau — *"sow seeds."*

Architecture: frozen ~500M flow-matching **DiT actor (S1)**, markov, frame→18-action chunk. Frozen **VLM (S2)**
emits a text plan. Trainable **bridge** (Perceiver resampler + adapter + plan-head) → **K=8 continuous plan
tokens** injected into DiT cross-attention. **Masked-null** = a null plan reproduces base DiT *exactly* (CFG-
style plan guidance works). Qwen-LoRA on the VLM is now allowed. Goal: **ONE generalist** across all games.

---

## 1. The positive S2→S1 transfer results we already have (settled)

These are the empirical foundation. Every number is from `files/DISCRIMINATOR_RESULTS.md` (multi-seed, bootstrap
CIs, null-invariance checked). They form a coherent story: **S2 can already transfer "concept-level" skill into
S1 for free where S1 owns the primitive; the open problem is making that ownership persistent and extending it
to skills S1 does not yet own.**

### P1 — EVOCATION WORKS: S2 can *summon* a primitive S1 already owns (the cheap, free transfer)
SMW duck-on-command (BASE bridge, no retraining), DOWN-press rate vs plan text:
| plan text | DOWN-press |
|---|---|
| terse "move right…" | 1.2% |
| "duck to dodge the bullet" | 3.4% |
| "press down to duck…" | **10.8% (9×)** |
The DiT **has** duck; the **plan token evokes it**, monotone in plan-explicitness. This is the existence proof
that S2→S1 skill *naming* works when the skill is already in S1's repertoire.

### P2 — ADDITION is needed for precise OUTCOMES: naming is not enough when S1 doesn't own the skill
SMW spin-jump-kills-Rex (maneuver router, reconstructed save-states ~30 frames pre-maneuver, n=10):
the **JUMP button is 7.2× evocable** by the plan, BUT the **kill/dispatch OUTCOME** (survive + advance past
Rex) stays flat **3.75%→7.5%** (both near chance). So the *primitive* (jump) is evocable but the *precise
maneuver outcome* is an **ADDITION** — it must be **learned** into S1 (demo-fit / DiT-LoRA on the human's
rex-kill chunks), not merely named.
> **The clean dichotomy (P1 vs P2): EVOCATION (free, S2 names a skill S1 has) vs ADDITION (costly, S1 must
> learn the skill). This is the measured backbone of the whole forward plan.**

### P3 — POSITIVE CROSS-GAME TRANSFER at the data level (skills generalize across games)
ONE pooled plan-head fit on **SMW+MMX+SMB1** (each chunk on its *own* game's plan) widens Δ_plan on **all**:
SMW +26, MMX +31 (3/3 seeds, CIs>0, null-invariance exact 0.0). **SMW *benefits* from pooling (1.3× vs solo)**
— SMB1+MMX demos *help* SMW. Transfers to a **NEW game (MMX) out of the box (+117%)**. → the "broad concept"
genuinely transfers across games at the plan-head level.

### P4 — NO-INTERFERENCE MERGE: skills compose into ONE model
Pooled plan-head (`plan_head.*`) + Sonic RWBC (`lora_*`) write **mostly-disjoint params** → merged on one
`btn_s600`, **all 4 games retained/improved**, null-invariance exact. Two training modes stack cleanly →
evidence that a single generalist *can* hold multiple games' skills without catastrophic interference.

### P5 — CONSOLIDATION WITHOUT FORGETTING is achievable: the KL-anchor (the key enabler for "instinct")
Naive demo-fit on ONE terse plan **catastrophically overwrote** the bridge's plan-responsiveness (duck → 0%
for *all* plans — it broke evocation P1). **KL-anchor** (functional L2 of plan-tokens to a frozen pre-fit head
over a broad plan distribution) **ADDS the advance skill AND preserves the whole action space**:
expressiveness-battery R=0.83 (only PASS), Δ_plan +45.1 (3/3 seeds), duck-on-command restored monotone.
= CoTTA stochastic-restoration in functional form. **This is the "add a skill without destroying the others"
primitive** — the candidate mechanism for turning a transient evocation into a persistent owned skill.

### P6 — STALENESS IS TEXT-BOUND: the continuous short path is remarkably stable
K plan tokens barely change frame-to-frame (**tok-drift ~0.001**; cached ≈ fresh ≈ null ≪ oracle). The owner's
"50-60% plan flip" lives in the **TEXT**, not the tokens. Re-grounding tokens (forward-pass TTA; von Oswald
"short path is already TTT") does **not** help — there is no token-staleness to fix. LIVE re-planning even
**HURTS** (MMX live-fresh −11) — churns the text without payoff. **Lever = plan TEXT quality (VLM-abstraction /
Qwen-LoRA), not token-TTA, not actor-weight-TTT.** (Implication: the online "tweak" surface for *concept* is
text; the open question is the surface for *execution* nuance — Q4.)

### P7 — demo-fit FIXES the live-deployment gap (robustness to imperfect S2 plans)
Real deployment uses the live VLM plan, not the oracle. On BASE the live plan **HURTS** SMW (−10.7); after
demo-fit → **+56.3** (≈ oracle ceiling). Demo-fit's real value is making the bridge **robust to the VLM's
imperfect live plans**, not just "use the oracle better." Replicated on 3 games.

### Honest caveats on all of the above
All Δ_plan evals are **oracle-or-fixed-text plan, short-horizon (≤16 chunks), from fixed save-states**. They
prove the **prerequisite** — the bridge *can* learn to use a good plan better and skills compose — but **closed-
loop long-horizon drift, real-time replan jitter, and the persistence of evocation across a full level are
UNPROVEN.** Task 2 (furthest-rollout + VLM video verification, running now) is the closed-loop reality check.

---

## 2. The forward problem, framed by the results

The owner's recursion — *discovery → instinct → higher discovery* — decomposes into four sub-problems, each now
anchored to a measured result:

1. **Routing (evocable vs addition).** P1/P2 give a *measured* dichotomy. We need a cheap online classifier
   (does S1 already own this skill, or must it learn it?) to spend effort correctly. (maneuver_router exists.)
2. **Persistent evocation → instinct.** P1 evokes transiently; P5 shows we can *add a skill without forgetting
   others*. The core mechanism question: how to **amortize** an evoked behavior into S1 so it fires *without*
   the plan token — freeing plan bandwidth — **while preserving exact null-invariance** (the crux tension:
   changing the null policy breaks masked-null).
3. **Token-dilution wall.** Can't verbose-prompt every primitive (P6: text churn hurts; image-token-overwhelm
   EXP-050/051). As games recurse, S2 must say *more* in the same K=8 budget → a compact **skill-token
   codebook** (options-as-tokens) vs widening K vs hierarchical plans.
4. **Subtly-different execution.** P3/P4: *concept* transfers across games; P2: precise *outcome/execution*
   does not. → a **shared concept channel** (generalist plan-head) + a small **per-state execution add**
   (router-gated), kept as ONE model.

---

## 3. The forward mechanism — "persistent evocation → instinct," resolved by construction

War-room R13 (GPT-5.5 + Opus-4.8) and the lit review (`LIT_R13_S2TOS1.md`, 24 arXiv-verified papers)
**converged independently** on the same forward path. The single most important result of the round is the
resolution of a **construction trap** that sits at the heart of the owner's north star.

### 3.1 The trap: "consolidate the skill into the policy" vs exact null-invariance are the SAME object
"Discovery becomes instinct" literally means: self-distill the **plan-conditioned** behavior (S2 says "duck")
into the **unconditioned** policy, so the skill fires *without* the plan token (freeing plan bandwidth for a
higher abstraction). But in our architecture the unconditioned policy **is** the masked-null base:
`apply_null_mask` zeroes only the K `_PLAN_TOKEN` positions on a **frozen** DiT → bit-identical to base
(`nitrogen.py:602`, verified). **Distilling instinct *into the null* changes the frozen image-only forward →
masked-null exactness dies.** You cannot consolidate into the null and keep null==base; they are one object.

### 3.2 The construction-safe fix: consolidate into the ALWAYS-ON game-context channel (not the null)
The frozen DiT already exposes a conditioning surface present in **both** the null and plan paths and **never
masked**: the `_GAME_ID_TOKEN` (=6) embedding — `nn.Embedding(len(games), vision_hidden, padding_idx=0)`, a
**shared table indexed by game-id** (`nitrogen.py:264-269, 477`; `apply_null_mask` touches only `_PLAN_TOKEN`=7,
verified). Put the instinct there:

- Add a **zero-init `InstinctResidual: game_id → R^{vision_hidden}`** at the always-on `_GAME_ID_TOKEN`
  position (a disjoint param group; zero-init ⇒ identity at start, so the `null==base` bit-identity check
  passes at init and migrates cleanly as it trains).
- The skill now fires **without the plan token** (instinct ✓, frees bandwidth ✓), while the K plan tokens stay
  cleanly maskable → **plan-relative null-invariance holds EXACTLY by construction** (masking the K tokens is
  a no-op w.r.t. the game-context policy; CFG plan-guidance still extrapolates along the K tokens).
- **What we deliberately relax** is *which* policy is the CFG reference: null moves from "pretrained-base" to
  "base + game-instincts." That redefinition **is the definition of instinct** (CFG only needs a stable
  unconditional anchor — Ho & Salimans 2207.12598 — never specifically the pretrained weights). For an unseen
  game, `game_id=0` is the zero/unconditional row → base-exact preserved by construction; un-consolidated games
  still reproduce the original base. This is the owner-sanctioned "single table indexed by game-context, NOT
  separate per-game weights."

> This is the load-bearing idea of the report: **instinct lives on the always-on game channel; the plan tokens
> stay the exact, maskable CFG contrast. Persistence and null-invariance stop fighting.**

### 3.3 Mechanism ranking (both agents + lit agree), construction-safety first
| rank | mechanism | what it buys | grounding | when |
|---|---|---|---|---|
| **1 (a)** | **KL-anchored self-distillation into the always-on game channel** | transient evocation → owned instinct, bandwidth freed, **null-exact by construction** | **P5** (KL-anchor adds-without-forgetting, R=0.83) + **Distral 1707.04175** + **LwF 1606.09282** + Policy-Distillation 1511.06295; = the inner loop of **ExIt 1705.08439** | **NOW** (Q5) |
| **2 (c)** | **skill-token codebook** (VQ over plan tokens; BPE-merge frequent chunks → one token) | beats the **token-dilution wall** (Q3); "chain combos c1 c2 c3 = one token"; null-safe behind the same gate | **LISA 2203.00054** + **PRISE 2402.10450** + QueST 2407.15840; LAPA/Genie/FAST for the from-video path | after ≥a few consolidated skills exist |
| **3 (b)** | **S2-written fast-weights / hypernet** | the **subtly-different-execution** nuance (Q4), router-gated per-state ADD | HyperNetworks 1609.09106; P6 (short path stable) does NOT refute it (P6 only killed *token*-TTT) | defer; least de-risked |

**The novelty, stated honestly (from the lit):** RT-H (2403.01823), SayCan (2204.01691), Hi Robot, and VOYAGER
(2305.16291) all let a high level *name* low-level skills, but **none amortize the named skill away** — the
"S2 stops naming it, freeing bandwidth" step is **ours**, and P5 + the always-on-channel construction is what
makes it possible. ExIt (1705.08439) is the cleanest skeleton for the whole recursion (propose→verify→distill→
compose); our twist is that the distillation target is a frozen cross-attention actor reached only through a
K-token bridge with exact null-invariance — so the LwF/Distral anchor (P5) is not optional but **structural**.

### 3.4 The recursion loop, concretely, with OUR tools
1. **S2 proposes** a maneuver from a save-state (gold-action/explicit plan).
2. **Emulator verifies** the *outcome* (not the button) via `maneuver_router.py` + save-states (P2's
   survive+advance = dense checkable reward). **Route:** null already succeeds → no-op; explicit plan succeeds
   → **evocation**; outcome fails despite button moving → **addition** (long path, gated DiT-LoRA, deferred).
3. **Consolidate** the verified evocation into the always-on game channel (§3.2) via KL-anchored
   self-distillation (`demo_bc.py --consolidate game_id`, §4).
4. **S2 stops naming it** — measured: the null policy now executes the skill at the relevant states; the
   plan-marginal contribution of the skill word → 0.
5. **Compose higher** — the freed K-budget carries a higher-abstraction plan ("cross the bullet corridor"
   instead of "press down to duck"); repeat. This is ExIt's outer loop / VOYAGER's growing library, but with
   the skill written into S1 instead of a text file.

## 4. Recommendation — the minimal this-week experiment: **CONSOLIDATE-DUCK**

Both agents independently picked the **same** first experiment, and **both rejected Rex-first** (P2 is near
chance, needs an unbuilt sprite-status kill-detector; it conflates "instinct" with "unlearned addition"). Duck
is the only this-week test that isolates *evoked → instinct*: it is **evocable** (P1, 9×), **verifiable**
(router bullet-states), and the **construction-safe target is identified** (the game-id row).

**One new flag, ≈30 lines, reusing existing machinery** (`demo_bc.py --consolidate game_id`): trains a zero-init
`InstinctResidual(game_id)` by self-distillation — **teacher** = the plan-conditioned chunk under the explicit
duck plan ("press down to duck") at duck-relevant frames (`--duck-probe` set + router bullet-states);
**student** = the same frames with the K plan tokens **masked**; plus the existing `--kl-anchor` pool as the
"elsewhere = base" pin. Everything else (KL-anchor, `duck_probe` dim22/dim1, `--save-delta`, `survival_advance`)
already exists. 3 seeds.

**Pass bars (POST vs PRE):** (i) **null-policy** duck-rate at held-out bullet states **+≥20pp or ≥0.65** (the
skill fires WITHOUT the plan token); (ii) plan-**marginal** duck contribution → ~0 (S2 no longer needs to name
it); (iii) expressiveness-battery **R ≥ 0.80** (no forgetting of retreat/jump/wait/up); (iv) **action-level null
bit-identity = 0.0 on every other state** (masked-null preserved by construction); (v) advance Δ_plan retains
≥80% of the R9 KL baseline; 2/3 seeds pass, no collapse seed.
**Closeout (closes the loop):** re-run `staleness_probe.py live` with a plan that **omits duck** ("advance,
jump pits"). PASS = the consolidated model survives bullet-bills (null-duck instinct fires) at ≥ the
duck-naming model's survival — *empirical proof S2 stopped naming duck and the freed budget is usable.*
**Falsifier:** null duck doesn't rise despite explicit-plan winners, or it rises only by collapsing to
always-down / hurting advance / R<0.80, or null bit-identity breaks elsewhere ⇒ consolidation-into-instinct is
not solved by this surface ⇒ fall back to the skill-token codebook (rank 2) for that skill.

**Sequencing:** CONSOLIDATE-DUCK (now) → if it passes, a 2nd skill + mint the first codebook token (rank 2) →
Rex/execution-add via gated-LoRA only after a sprite-status kill-detector exists (Q4 long path). The
game-specific delta the owner sanctioned as a "bootstrap past the single-level plateau" *is* the per-game row
of the always-on channel — one model, indexed by game-context, no separate checkpoints.

## 5. Highlight reel — lead with the full forward arc (honest)

1. **P1 → P5 is the lead** (both agents): **P1** = S2 evokes a skill S1 already owns (duck 1.2%→10.8% on
   command, the "discovery"); **P5** = KL-anchor consolidates a skill **without forgetting the rest** (Δ_plan
   +45.1, expressiveness R=0.83, only PASS — the "becomes instinct," *construction-guaranteed*, not tuned).
   Together they are the only evidence of the **full forward arc** the report is about.
2. **P3 / P4** = it stays ONE generalist: a single pooled plan-head fit transfers across SMW+MMX+SMB1 (+26/+31,
   3/3 seeds, +117% on a new game) and merges with the actor-OOD Sonic LoRA with no interference, null-exact.
3. **P7** = deployment relevance: demo-fit flips the live VLM plan from harmful (−10.7) to useful (+56.3).
4. **P2** = discipline: not everything named transfers — precise outcomes (Rex) are **additions**, routed to the
   long path. **P6** = a valuable negative: don't spend on token-staleness; the lever is plan-text quality.

**Honest caveats (carry these into any claim):** every quantitative win is **oracle-or-fixed-text plan,
short-horizon (≤16 chunks), from fixed save-states.** Closed-loop long-horizon drift, real-time replan jitter,
and the persistence of an evoked skill across a full level are **unproven** — that is what Task-2 (furthest
rollouts + RAM-free VLM video verification, running now) and the CONSOLIDATE-DUCK closeout are for. The
always-on-channel consolidation is **designed and code-verified, not yet run.**

---
### Source memos (this round)
`ARCH_WARROOM_R13_SEED.md` (framing) · `ARCH_WARROOM_gpt55_r13.md` · `ARCH_WARROOM_opus_r13.md` (the
construction resolution) · `LIT_R13_S2TOS1.md` (24 arXiv-verified papers). Results: `DISCRIMINATOR_RESULTS.md`.
Code anchors: `nitrogen.py:594-605` (masked-null), `:264-269,477,602` (always-on game-id channel = the
consolidation target), `lora.py:8-12` (plan-token-gated ADD, Q4). Tools for Q5: `demo_bc.py` (new
`--consolidate game_id`), `maneuver_router.py`, `expressiveness_battery.py`, `staleness_probe.py live`,
`eval_common.py:survival_advance`.
