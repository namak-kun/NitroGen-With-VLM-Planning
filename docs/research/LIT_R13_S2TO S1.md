# LIT_R13_S2TOS1.md — System‑2 → System‑1 Knowledge Transfer & the Recursive Skill Hierarchy

**Compiled 2026‑06‑29 | for @namak‑kun | anchors the R13 forward report (ARCH_WARROOM_R13_SEED.md).**
North star: *"discovery becomes instinct which drives higher abstractions of discovery."* A maneuver S2
evokes in S1 should become OWNED by S1 (instinct) so S2 can stop naming it and compose at a higher level.

**Setup this lit must map onto:** frozen ~500M flow‑matching DiT actor (S1, frame→18‑action chunk, markov);
frozen/LoRA Qwen VLM (S2, emits plan); trainable bridge = Perceiver resampler + adapter + plan‑head → **K=8
continuous plan tokens** injected into the DiT cross‑attention; **masked‑null = EXACT null‑invariance** (null
plan ⇒ base DiT, CFG‑style guidance); **emulator save‑states** = dense checkable outcome reward (P2). One
generalist across games (per‑game models rejected; intra‑genre interference is the problem to solve).

**Verification discipline.** Every arXiv ID tagged ✓ was fetched from `export.arxiv.org/api` THIS session and
its abstract read to confirm the claim below. IDs tagged **(R11✓)** were verified in the prior reviews
(`LIT_R11_{ttt,distill,fastweights}.md`) — referenced here, not re‑litigated. ⚠️ flags anything unverifiable.
This review **builds on** R11 (TTT/TTA, distillation, fast‑weights, dual‑process VLAs) and does not repeat it.

> **Two corrections to prior lit, established this session:**
> • **RT‑H arXiv ID = 2403.01823 ✓** (R11_fastweights left it "unconfirmed").
> • Sutton‑Precup‑Singh 1999 is **NOT on arXiv** (AIJ journal); its existence/claim is grounded via the
>   Option‑Critic abstract, which cites it explicitly. Flagged in Thread 1.

---

## THREAD 1 — OPTIONS / HRL (is "options discovered by System‑2" a real framing?)

**Sutton, Precup & Singh 1999 — "Between MDPs and semi‑MDPs: A framework for temporal abstraction" ⚠️ NOT on arXiv**
(Artificial Intelligence 112:181–211; grounded via the Option‑Critic abstract's citation "[Sutton, Precup &
Singh, 1999]").
- *Idea:* an **option** = ⟨initiation set I, intra‑option policy π, termination β⟩ — a temporally‑extended
  action an agent can plan over as a first‑class object.
- *Maps to us:* our K=8 plan‑token bundle driving an 18‑action chunk **IS an option in all but name** — the
  plan token = option identifier/selector, the DiT chunk = intra‑option policy π, the re‑plan boundary = β,
  **masked‑null = the "no‑option / base primitive" baseline**. This is the formal vocabulary for Q2c
  (options‑as‑tokens) and the whole recursion. Flag the non‑arXiv status when citing.

**Option‑Critic — Bacon, Harb, Precup 2017. arXiv:1609.05140 ✓**
- *Idea:* policy‑gradient theorems that learn intra‑option policies **and** termination β **and** the
  policy‑over‑options jointly, end‑to‑end from a scalar reward, with **no subgoals or extra rewards**.
- *Maps to us:* the gradient mechanism to *shape* a discovered option from a scalar — our **emulator
  survive+advance outcome (P2)** is exactly that scalar. But option‑critic discovers options *bottom‑up* from
  reward; we want them *proposed top‑down* by S2 (VOYAGER/ELLM, below). Known failure: options **degenerate**
  (one dominates, β collapses) without regularization — same risk as one plan token dominating; argues for the
  KL‑anchor (Thread 4) as the stabilizer.

**FeUdal Networks — Vezhnevets et al. 2017. arXiv:1703.01161 ✓**
- *Idea:* **Manager** sets abstract goal vectors at a *low* temporal resolution; **Worker** emits primitives
  every tick; decoupled training; sub‑policies emerge per Manager goal; goals are *directional* (cosine), not
  absolute.
- *Maps to us:* our exact two‑timescale split — **VLM (Manager, replan‑every‑N) = directional latent goal;
  DiT (Worker, per chunk) = realization.** FeUdal's *directional* goal ≈ our **plan‑token DIRECTION as the
  shared concept channel (P3)**; its low‑frequency manager signal ≈ **P6 (tokens barely drift frame‑to‑frame)**.
  Grounds `--replan‑every N` (Q‑stabilize) and the two‑level recursion structure.

**Is "options discovered by System‑2" real?** *Partially, and it's an active frontier.* Classic HRL discovers
options bottom‑up (Option‑Critic, FeUdal). The **S2‑proposes‑options** direction is newer and real: **ELLM —
Guiding Pretraining in RL with LLMs (arXiv:2302.06692 ✓)** has an LLM suggest goals from a state description to
shape low‑level exploration; **VOYAGER (Thread 2)** has an LLM write/store/reuse skills; **Code‑as‑Policies
(arXiv:2209.07753 ✓)** has an LLM compose primitive APIs into new policies. So the owner's hunch is a genuine,
named framing (LLM‑as‑option‑proposer) — and **masked‑null + the plan‑token is the missing differentiable
channel** to inject an S2‑proposed option into a *frozen* low‑level actor, which none of those three provide.

---

## THREAD 2 — SKILL LIBRARIES / LIFELONG (consolidation‑into‑instinct analogue)

**VOYAGER — Wang et al. 2023. arXiv:2305.16291 ✓**
- *Idea:* LLM agent in Minecraft with (1) automatic curriculum, (2) an **ever‑growing skill library of
  executable code**, (3) iterative prompting with env feedback + **self‑verification**; no fine‑tuning;
  reuses the library in a *new* world; "alleviates catastrophic forgetting" because skills live as code.
- *Maps to us:* **the template for our recursion (Q5)** — S2 proposes a maneuver → **emulator verifies
  (self‑verification = our save‑state outcome, P2)** → store reusable skill → compose at higher abstraction.
  **The gap we must close:** VOYAGER stores skills as external text/code and *re‑invokes them by name forever*
  — it never frees prompt bandwidth, which is **exactly the token‑dilution wall (Q3)**. Our north‑star adds the
  step VOYAGER lacks: *consolidate the skill into S1's weights/tokens so S2 stops naming it* (Threads 4, 6).

**DIAYN — Eysenbach et al. 2018. arXiv:1802.06070 ✓**
- *Idea:* unsupervised skill discovery — maximize MI between a skill latent z and visited states (diversity),
  max‑entropy policy, **no reward**; emergent skills compose hierarchically and seed downstream tasks.
- *Maps to us:* the bottom‑up complement to S2‑proposed skills — a label‑free way to **pre‑mint a vocabulary of
  distinguishable primitives the DiT already supports (P1: "the DiT HAS duck")**, which S2 then names/selects.
  z ≈ a skill code (pre‑codebook for Q2c). *Caveat:* DIAYN skills are diversity‑driven, not outcome‑driven —
  they won't target the owner's specific maneuver (Rex‑kill, P2); our **emulator reward is the missing task
  signal**. Good for bootstrapping the codebook, not for precise additions.

---

## THREAD 3 — FAST‑WEIGHTS / HYPERNETWORKS (per‑episode S2‑written weight delta vs offline LoRA)

**HyperNetworks — Ha, Dai, Le 2016. arXiv:1609.09106 ✓**
- *Idea:* one network (the hypernet) **generates the weights** of another (genotype→phenotype), trained
  end‑to‑end by backprop; a relaxed form of weight‑sharing.
- *Maps to us:* the architecture for **Q2b** — instead of S2 emitting K plan *tokens*, an S2‑conditioned
  hypernet emits a small **per‑game/per‑episode weight delta** (realistically → DiT‑LoRA A/B factors or
  AdaLN/FiLM gains, not full DiT weights). This is the owner's "online ADDITION/TWEAK, not offline LoRA
  ('nobody plays games like this')" and stays **ONE model** (Q4: a *shared* hypernet *indexed by game‑context*,
  not separate weights). **Null‑invariance is preserved iff the null/masked context makes the hypernet emit a
  ZERO delta** — a zero‑init output head, exactly the AdaLN‑zero trick (R11✓ DiT 2212.09748).

**Build‑on (R11✓, reference only):** Schlag/Irie/Schmidhuber FWP (2102.11174), von Oswald ICL‑as‑GD (2212.07677).
R11/**P6** already settled that the continuous short path is stable and forward‑pass TTA does **not** help (no
token staleness). **Sharpen the distinction:** P6 killed *token*‑TTT; it did **not** test *weight‑space
per‑game specialization*. So fast‑weights as a staleness‑fix = dead; fast‑weights as the **execution‑nuance
surface (Q4)** = live. *Verdict:* a hypernet→LoRA is the most principled answer to "subtly‑different execution."
*Falsifier:* a game‑id‑conditioned hypernet‑LoRA must beat a single shared LoRA on per‑game execution
(e.g. Rex‑kill timing) **without** hurting other games; else fall back to consolidation (Q2a).

---

## THREAD 4 — CONSOLIDATION WITHOUT FORGETTING (closest named precedent for our KL‑anchor, P5)

**Distral — Teh et al. 2017. arXiv:1707.04175 ✓ — CLOSEST PRECEDENT (a).**
- *Idea:* share a **distilled policy = centroid of all task policies**; each task policy is **KL‑constrained to
  stay close to it**; this damps negative gradient interference and stabilizes multitask RL.
- *Maps to us:* Distral's *"KL to a shared distilled policy"* ≈ our **KL‑anchor (P5): L2 of plan‑tokens to a
  frozen pre‑fit head over a broad plan distribution.** It also models **P3 (pooled head = centroid)** and
  **P4 (no‑interference merge)** — Distral exists *precisely* to stop the intra‑genre interference the owner
  named as "the problem to SOLVE" (Q4). This is the single best RL‑side citation for our consolidation primitive.

**Learning without Forgetting (LwF) — Li & Hoiem 2016. arXiv:1606.09282 ✓ — CLOSEST PRECEDENT (a), function‑space twin.**
- *Idea:* train on **new‑task data only** while preserving old capabilities via **knowledge distillation on the
  old model's outputs** (functional/output‑space anchor) — no old data, no weight‑importance estimate.
- *Maps to us:* our KL‑anchor is **functional** (matches plan‑conditioned *outputs* over a broad plan dist),
  which is LwF's mechanism — not weight‑space EWC. **P5's "add the advance skill while preserving the whole
  action space" = LwF's "preserve old capabilities using only new data."** Cite LwF + Distral together as the
  named precedent; our P5 is their instantiation in plan‑token function space.

**Policy Distillation — Rusu et al. 2015. arXiv:1511.06295 ✓**
- *Idea:* distill an RL policy into a smaller net; **consolidate multiple task‑specific policies into ONE** that
  can *outperform* the single‑task teachers.
- *Maps to us:* the mechanism for **Q2a / the Q5 experiment** — self‑distill the *plan‑on* (evoked) DiT into the
  *plan‑off/null* DiT on states where the skill should fire → **duck becomes instinct without the token**.
  Multi‑teacher→one‑student = our pooled generalist (P3).

**Weight‑space alternatives (cite if we move the anchor off function space):**
- **EWC — Kirkpatrick et al. 2016. arXiv:1612.00796 ✓** — slow learning on params important to old tasks
  (Fisher quadratic penalty). (R11✓ already covered **EATA 2204.02610** = EWC‑in‑TTA, and **CoTTA 2203.13591**
  = stochastic restoration, the seed's stated analogue for P5.)
- **PackNet — Mallya & Lazebnik 2017. arXiv:1711.05769 ✓** — prune → free params → "pack" the new task into
  freed capacity; **zero forgetting by construction**. **Our P4 disjoint‑param merge (`plan_head.*` vs `lora_*`)
  is empirically a PackNet‑style result** (different param groups = different capacity).
- **Progressive Neural Networks — Rusu et al. 2016. arXiv:1606.04671 ✓** — freeze old columns, add a new column
  with lateral connections; **immune to forgetting**; transfer at both sensory and control layers. The
  architectural cousin of our merge (P4) and of a per‑skill gated‑LoRA column (Q4).

*Closest precedent for the KL‑anchor (P5):* **Distral (1707.04175) + LwF (1606.09282)**, with CoTTA (R11✓) as
the TTA‑flavored restoration cousin and EWC as the weight‑space fallback.

---

## THREAD 5 — LANGUAGE→MOTION HIERARCHY (do any AMORTIZE the named skill into the low level?)

**RT‑H: Action Hierarchies Using Language — Belkhale et al. 2024. arXiv:2403.01823 ✓ (ID corrected from R11)**
- *Idea:* insert **"language motions"** ("move arm forward") as an intermediate between the high‑level task and
  low‑level action; conditioning on language‑motions exposes **shared low‑level structure across diverse tasks**;
  supports **human language‑intervention corrections that the policy LEARNS from**.
- *Maps to us:* directly models S2 task → "language motion" (≈ our plan text/token) → S1 action. Crucially,
  RT‑H shows language‑motions make low‑level structure **shareable across tasks (= P3 cross‑game transfer)**,
  and learning‑from‑intervention ≈ our **demo‑fit (P7)**. **But RT‑H keeps naming motions forever — it does NOT
  amortize them away.** Validates the hierarchy; leaves the "stop naming it" step to us.

**SayCan — Ahn et al. 2022. arXiv:2204.01691 ✓**
- *Idea:* LLM proposes *feasible* language skills; **per‑skill value functions ("affordances") ground** them;
  pick = p_LLM(skill) × value(skill). "Do as I can, not as I say."
- *Maps to us:* the affordance gate ≈ our **maneuver_router (Q1)** — SayCan multiplies LLM probability by skill
  value; our router multiplies **evoc‑ratio × emulator survive+advance outcome** to classify **evocable (P1) vs
  addition (P2)**. SayCan = precedent for "S2 only names skills S1 can actually do." *Caveat:* SayCan's skills
  are **fixed** pretrained primitives (no growth, no consolidation) — combine with VOYAGER (growth) + Thread 4
  (consolidation) for the full loop.

**Reference (R11✓):** RT‑2 (2307.15818), Hi Robot (2502.19417, closest dual‑system robot analogue), π0/π0.5.
**Amortization verdict (important for Q6 honesty + novelty):** **NONE of RT‑H / SayCan / Hi Robot amortize a
language‑named skill INTO the low level so the high level can stop naming it.** That "stop naming it" step is the
genuinely novel contribution our **P5 KL‑anchored consolidation** enables — lead with this framing.

---

## THREAD 6 — SKILL‑TOKEN / DISCRETE‑OPTION‑AS‑TOKEN  *(the most important / novel thread — searched hard)*

**LISA: Learning Interpretable Skill Abstractions from Language — Garg et al. 2022. arXiv:2203.00054 ✓ — CLOSEST PRECEDENT (b).**
- *Idea:* hierarchical IL that uses **vector quantization to learn DISCRETE skill codes** highly correlated with
  the language instruction *and* the policy's behavior; **compose codes to solve unseen long‑range instructions**;
  beats a flat Decision Transformer in the low‑data regime.
- *Maps to us:* the textbook instance of **Q2c** — a high level emits a **discrete skill CODE (not prose)** and
  the low level expands it; codes are interpretable + composable. Our K=8 plan tokens are the *continuous*
  analogue; LISA says **quantize them into a learned codebook**, so each code = an option = a compact name S2
  can re‑emit at higher abstraction — **directly beats the token‑dilution wall (Q3)**. *Falsifier:* does a VQ
  codebook over our plan‑tokens preserve Δ_plan **and** exact null‑invariance while cutting token count?

**PRISE: LLM‑Style Sequence Compression for Temporal Action Abstractions — Zheng et al. 2024. arXiv:2402.10450 ✓ — CLOSEST PRECEDENT (b), the recursion.**
- *Idea:* treat skill induction as **sequence compression**: run **byte‑pair encoding (BPE)** over quantized
  continuous actions → variable‑length **skill tokens**; boosts multitask + few‑shot IL on unseen tasks.
- *Maps to us:* the most on‑point cite for the owner's recursion — **BPE merges a frequently‑used action
  sub‑sequence into a single token**, i.e. *discovery (frequent chunk) → mint a token → reuse at higher level*
  = **"chain combos c1 c2 c3 become one token."** This is exactly Q3's compression path and Q5's "S2 stops
  saying the primitive."

**QueST: Quantized Skill Transformer — Mete et al. 2024. arXiv:2407.15840 ✓**
- *Idea:* a larger, flexible **quantized latent skill space** with **causal inductive bias** from action
  sequences → transferable low‑level skills; strong multitask + few‑shot.
- *Maps to us:* the architecture for a **shared skill‑token codebook across games** (our ONE‑generalist
  constraint) that **few‑shot transfers to a new game** (= P3, MMX out‑of‑the‑box). Its causal bias ≈ our
  within‑chunk ordering (SEQ) results.

**Director: Deep Hierarchical Planning from Pixels — Hafner et al. 2022. arXiv:2206.04114 ✓**
- *Idea:* hierarchical RL by planning in the **latent space of a learned world model**; the high level picks
  **discrete latent goal codes**, the low level achieves them; goals **decode to images** (interpretable).
- *Maps to us:* the **model‑based** version of S2(code)→S1(expand), and the codes are **world‑model/emulator
  grounded** — ties Q2c to our **emulator save‑state substrate (P2)** and to a learned forward model for RL.

**OPAL: Offline Primitive Discovery — Ajay et al. 2020. arXiv:2010.13611 ✓**
- *Idea:* extract a **continuous space of recurring temporally‑extended primitives** from offline data;
  temporal abstraction shrinks the effective horizon → better offline RL, few‑shot IL, transfer.
- *Maps to us:* justifies **mining a primitive space from our offline emulator/IDM data BEFORE RL** (lowers the
  long credit path the seed flags for RL). Continuous (like our current plan tokens) = the **pre‑quantization**
  stage feeding LISA/PRISE/QueST.

**Reference (R11✓), the unsupervised‑from‑video codebook cousins (directly relevant to our YouTube/IDM bootstrap):**
- **LAPA — Latent Action Pretraining (2410.11758, re‑verified ✓):** VQ‑VAE learns **discrete latent actions
  between frames** from action‑free video; pretrain a latent VLA to predict them; finetune to real actions.
- **Genie (2402.15391):** VQ latent‑action codebook learned unsupervised from video — the generative cousin.
- **FAST: Frequency‑space Action Tokenization (2501.09747 ✓):** **DCT/compression** action tokenizer (universal,
  black‑box) — engineering proof that *compressed* action tokens beat naïve per‑step binning for high‑frequency
  control; supports "**compact tokens expand to dense actions**" without the dilution of per‑step prose.

*Thread‑6 verdict:* rich and real. **Build on LISA (discrete language→skill codes) + PRISE (BPE compression =
the recursion)**; QueST/Director add shared‑codebook + world‑model grounding; LAPA/Genie/FAST give the
unsupervised‑from‑YouTube + compression path. This is the strongest answer to Q3 (token dilution) and the
eventual compression target for a consolidated skill (Q2c).

---

## THREAD 7 — EXPERT‑ITERATION RECURSION (verify → consolidate → compose higher)

**ExIt: Thinking Fast and Slow with Deep Learning and Tree Search — Anthony, Tian, Barber 2017. arXiv:1705.08439 ✓ (R11_distill✓; re‑verified).**
- *Idea:* decompose into **PLAN (tree search, slow)** + **GENERALISE (NN, fast)**; the NN then **guides the next
  search**, producing stronger plans to distill again; tabula‑rasa beats a champion.
- *Maps to us:* **the cleanest skeleton for the recursion (Q5).** S2 + emulator = the *slow planner* (search over
  plans, **verified by save‑states = P2 outcome reward**); the S1 bridge = the *fast generaliser* we **distill
  the verified plan into (Thread 4)**; the improved S1 makes S2's next search cheaper → frees bandwidth → higher
  abstraction. Our **save‑state GRPO** (seed §RL) = the policy‑improvement operator; **consolidation (P5)** = the
  distillation step. ExIt is the single best precedent for *"discovery → instinct → higher discovery."*

**Reference (R11_distill✓):** AlphaZero (1712.01815), STaR (2203.14465), ReST/ReST‑EM (2308.08998 / 2312.06585),
"Distilling System 2 into System 1" (2407.06023) — all the same *propose → verify → distill* loop. **Our
novelty vs all of them:** the consolidation target is a **frozen cross‑attention actor reached only through a
K‑token bridge with EXACT null‑invariance** — none carry the masked‑null constraint, so our distillation must
*preserve the null policy* (⇒ the LwF/Distral anchor, P5, is not optional but structural).

---

## SYNTHESIS — best‑grounded mechanisms for "persistent evocation → instinct → higher abstraction"

**Ranking the Q2 candidates by how well OUR results + the literature ground them:**

1. **Amortized consolidation via a FUNCTIONAL KL‑anchor (Q2a) — RANK 1, do this first.**
   Best‑grounded: **Distral (1707.04175)** + **LwF (1606.09282)** + **Policy Distillation (1511.06295)**, and it
   is the *inner distillation step of ExIt (1705.08439)*. We already have the positive result (**P5**: KL‑anchor
   adds a skill while preserving the whole action space + exact null‑invariance). This is the bridge from
   transient evocation (P1) to owned instinct. *Falsifier:* null‑invariance or another skill breaks during the
   fold‑in.

2. **Skill‑token codebook (Q2c) — RANK 2, the compression that follows consolidation.**
   Best‑grounded: **LISA (2203.00054)** + **PRISE (2402.10450)** + **QueST (2407.15840)** (+ LAPA/Genie/FAST,
   R11✓, for the unsupervised‑from‑video path). This is the only mechanism that lets S2 keep naming at *higher*
   abstraction inside K=8 (beats Q3 dilution). Sequencing matters: **consolidate the behavior first (1), then
   compress its name into a code (2)** — PRISE's "mint a token for a frequent chunk" is literally that.

3. **Expert‑iteration recursion (ExIt, 1705.08439) — RANK 1 as the OUTER loop.**
   It ties propose(S2) → verify(emulator save‑state, P2) → consolidate(P5) → compose(higher). Mechanism (1) is
   its inner loop; mechanism (2) is its compression stage. This is the scaffold for the whole report.

4. **Fast‑weights / hypernet (Q2b) — promising but LEAST‑grounded for us; keep as the Q4 hypothesis.**
   **HyperNetworks (1609.09106)** → DiT‑LoRA/AdaLN gains is the principled home for "subtly‑different execution"
   (Q4), and P6 (short path stable) does *not* refute it (P6 only killed *token*‑TTT). But it is the least
   de‑risked — pursue only if a game‑id‑conditioned hypernet‑LoRA beats a shared LoRA without collateral damage.

**Single closest precedent for each pillar:**

- **(a) Consolidation‑without‑forgetting → Distral (arXiv:1707.04175 ✓)** — KL‑constrain each task policy to a
  shared *distilled centroid* policy; anti‑interference by design. Our **KL‑anchor (P5) = Distral in plan‑token
  function space**, with **LwF (1606.09282 ✓)** as the functional‑KD twin and **CoTTA (2203.13591, R11✓)** as
  the TTA‑flavored restoration cousin the seed already named.

- **(b) Skill‑token codebook → LISA (arXiv:2203.00054 ✓)** — language → **discrete VQ skill codes** that a
  low‑level policy expands and composes. Runner‑up for the *recursion/compression* specifically:
  **PRISE (2402.10450 ✓)** (BPE‑merge a frequent action chunk into one reusable token).

- **(c) S2‑discovered options → VOYAGER (arXiv:2305.16291 ✓)** — an LLM (System‑2) **discovers, names, stores,
  reuses, and composes** an ever‑growing skill library. RL‑mechanism backstop: **Option‑Critic (1609.05140 ✓)**
  (autonomous option + termination discovery); LLM‑shapes‑low‑level precedent: **ELLM (2302.06692 ✓)**. *Our
  addition over VOYAGER:* it never consolidates skills into S1 or frees prompt bandwidth — pillars (a)+(b) supply
  the missing "stop naming it" step.

**The Q5 minimal this‑week experiment, fully lit‑grounded:** self‑distill (Policy Distillation / LwF) the
**duck‑evoked DiT** (P1: "press down to duck") into the **null / masked‑null policy** on bullet‑bill states,
**KL‑anchored over the broad plan distribution (Distral / P5)**. *Success* = null‑policy duck‑rate at bullet
states rises while Δ_plan + **exact null‑invariance on every other skill is preserved**, then S2 drops "duck"
and spends the freed K‑budget on a higher‑abstraction plan. *Falsifier* = null‑invariance breaks elsewhere, or
duck fails to fire on held‑out bullet states (overfit, not instinct). This is ExIt's inner loop on one skill.

---

### Citation index (primary papers verified by live abstract fetch THIS session)
| Thread | Paper | arXiv | Status |
|---|---|---|---|
| 1 | Sutton, Precup & Singh — Options framework | — | ⚠️ AIJ 1999, NOT on arXiv (grounded via 1609.05140) |
| 1 | Option‑Critic | 1609.05140 | ✓ |
| 1 | FeUdal Networks | 1703.01161 | ✓ |
| 1 | ELLM (LLM‑guided pretraining) | 2302.06692 | ✓ |
| 1 | Code as Policies | 2209.07753 | ✓ |
| 2 | VOYAGER | 2305.16291 | ✓ |
| 2 | DIAYN | 1802.06070 | ✓ |
| 3 | HyperNetworks | 1609.09106 | ✓ |
| 4 | Distral | 1707.04175 | ✓ (closest precedent a) |
| 4 | Learning without Forgetting | 1606.09282 | ✓ (closest precedent a, twin) |
| 4 | Policy Distillation | 1511.06295 | ✓ |
| 4 | EWC | 1612.00796 | ✓ |
| 4 | PackNet | 1711.05769 | ✓ |
| 4 | Progressive Neural Networks | 1606.04671 | ✓ |
| 5 | RT‑H (Action Hierarchies w/ Language) | 2403.01823 | ✓ (corrects R11 "unconfirmed") |
| 5 | SayCan | 2204.01691 | ✓ |
| 6 | LISA | 2203.00054 | ✓ (closest precedent b) |
| 6 | PRISE | 2402.10450 | ✓ (recursion/compression) |
| 6 | QueST | 2407.15840 | ✓ |
| 6 | Director | 2206.04114 | ✓ |
| 6 | OPAL | 2010.13611 | ✓ |
| 6 | LAPA | 2410.11758 | ✓ (re‑verified; R11) |
| 6 | FAST | 2501.09747 | ✓ |
| 7 | ExIt (Thinking Fast & Slow) | 1705.08439 | ✓ |

**Cross‑referenced from R11 (verified there, not re‑fetched):** Genie 2402.15391, Hi Robot 2502.19417,
RT‑2 2307.15818, CoTTA 2203.13591, EATA 2204.02610, von Oswald 2212.07677, Schlag FWP 2102.11174,
AlphaZero 1712.01815, STaR 2203.14465, ReST 2308.08998, ReST‑EM 2312.06585, Distilling System 2 into System 1
2407.06023, DiT/AdaLN‑zero 2212.09748.
