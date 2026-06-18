# Literature Review — Plan-Conditioned Diffusion VLAs & System-1/System-2 Agents

Compiled 2026-06-17 for @namak-kun. Focus: how our design (frozen VLM planner →
adapter → plan tokens → flow-matching DiT) relates to prior work, and what to borrow
for the hard parts (temporal/cross-chunk plan influence, system-1/2 bridging).

> Confidence note: entries marked ✓verified were cross-checked online this session;
> others are from the author model's prior knowledge of these well-known papers and
> should be double-checked against the arXiv abstracts before citing in writing.

---

## A. Most directly relevant (our architecture's lineage)

### 1. GR00T N1 / N1.5 (NVIDIA, 2025) — THE parent of NitroGen  ✓
- Dual-system VLA: **System 2** = a (frozen-ish) VLM that encodes vision+language;
  **System 1** = a **Diffusion Transformer (DiT) action head** trained with **flow
  matching**, cross-attending to the VLM tokens. Action chunks via flow matching.
- NitroGen is *explicitly* "adapted from GR00T N1 with the language and state encoders
  removed, and a single action head" (NitroGen paper §5). So NitroGen = GR00T-N1's
  System 1 alone. **Our project effectively re-adds a System 2 (the Qwen planner)** —
  i.e. we are reconstructing the GR00T dual-system, but with a game-playing System 1
  and a plan/transcript-driven System 2.
- Borrow: GR00T's recipe for connecting VLM tokens → DiT via cross-attention is the
  proven template; their later N1.5 improved the VLM↔DiT alignment (worth reading for
  how they make the language actually influence actions — our exact problem).

### 2. π0 / π0.5 (Physical Intelligence, 2024–25) — flow-matching VLA  ✓
- VLA built on a VLM (PaliGemma) with a **flow-matching action expert** producing
  continuous action chunks at high frequency. Same flow-matching objective family as
  NitroGen (v = a − ε, shifted-beta time sampling; NitroGen cites π0).
- π0.5 adds open-world generalization + a discrete "high-level subtask" prediction
  step before low-level action — an explicit hierarchy.
- Borrow: π0's evidence that a *frozen/lightly-tuned* VLM + a small flow-matching
  expert is sufficient; their action-expert conditioning details.

### 3. Hi Robot (Physical Intelligence, 2025) — explicit System 1/2 for instructions  ✓
- Two-level: a **high-level VLM** turns complex instructions + real-time feedback into
  simpler **language commands**; a **low-level VLA (π0)** executes them. "Thinking
  before acting" via language as the interface.
- Difference from us: Hi Robot's bridge is **natural-language subcommands** (discrete,
  interpretable). Ours is **continuous plan tokens** (latent). Trade-off: language is
  interpretable + composable but lossy/serial; latent is higher-bandwidth but opaque.
- Borrow: their training setup pairs (instruction, decomposed-command, execution) —
  exactly the structure our Stage-2 transcripts could take.

### 4. Helix (Figure AI, 2025) — latent-vector bridge, decoupled frequencies  ✓
- **System 2** = 7B VLM running ~7–9 Hz; **System 1** = ~80M visuomotor transformer
  running ~200 Hz. A **single continuous latent vector** from S2 conditions S1; the
  two run at **different frequencies** and are trained so S1 acts continuously while
  S2 updates the latent sparsely.
- **This is the closest analogue to our cross-chunk problem (EXPERIMENTS EXP "cross-
  chunk phase").** Helix solves "VLM fires sparsely, actor runs fast" by a latent that
  S1 consumes every step while S2 refreshes it occasionally. Their design validates
  our K×A / plan-token-persists-across-chunks direction.
- Borrow: decoupled-frequency training; conditioning S1 on a *single evolving latent*
  rather than re-running S2 each step. Key open question they answer implicitly: the
  latent must carry enough state for S1 to stay coherent between S2 updates.

### 5. LCB — "Latent Codes as Bridges" (2024)  ✓
- Bridges a high-level **LLM policy** and a low-level **diffusion policy** via a
  **learned latent code** (not language): the LLM emits an abstract latent action, the
  diffusion policy decodes it into motor commands; trained end-to-end (BC + RL).
- Directly analogous to **our adapter + plan tokens**. Validates: (a) latent > language
  for bandwidth; (b) end-to-end training of the bridge; (c) the LLM can operate
  sparsely. Closest conceptual match to our `PlanHead`.

---

## B. Foundational action-diffusion / VLA

### 6. Diffusion Policy (Chi et al., RSS 2023)
- The seminal "generate action chunks by denoising" policy (DDPM/DDIM over action
  sequences, visual conditioning). Established action chunking + receding horizon that
  NitroGen/GR00T/π0 inherit. Our 18-action chunk is this lineage.

### 7. Octo (2024, Berkeley) & OpenVLA (2024)
- Open generalist robot policies. Octo = transformer with a diffusion action head,
  conditioned on language/goal-image, trained on Open X-Embodiment. OpenVLA =
  autoregressive VLA (Llama + vision) predicting discretized actions.
- Relevance: Octo's **goal-image AND language conditioning** of a diffusion head is a
  precedent for multi-modal plan conditioning; OpenVLA shows the AR alternative we
  rejected (NitroGen is diffusion, not AR).

### 8. RT-1 / RT-2 / SayCan / PaLM-E (Google, 2022–23)
- Earlier hierarchy: LLM/VLM for high-level planning (SayCan: affordance-grounded
  skill selection; PaLM-E: embodied multimodal), discrete skills for low level. RT-2
  = VLM co-fine-tuned to emit action tokens. Historical context for "LLM plans, policy
  acts"; mostly discrete/skill-based vs our continuous flow-matching.

---

## C. Classifier-free guidance & conditioning (our null-plan / CFG machinery)

### 9. Classifier-Free Guidance (Ho & Salimans, 2021)
- The cond/uncond + extrapolation trick (`v = v_unc + s·(v_cond − v_unc)`). NitroGen's
  `get_action_with_cfg` implements it; our null-plan = the "uncond". Our masked-null
  (exact base under null) is a *clean* uncond. Relevant when we (re)enable plan CFG to
  amplify steering — note our dead-dropout bug must be fixed first.

### 10. Goal/skill-conditioned diffusion & compositionality
- Works on conditioning diffusion policies on goals, skills, or latent plans; and on
  **composing** diffusion models (e.g. composable diffusion) — relevant to our SEQ
  ("left then right") within-chunk composition, which currently FAILS (EXP-011). The
  composition literature suggests temporal composition needs either explicit temporal
  conditioning or a model that can route sub-goals across the horizon.

---

## D. For the cross-chunk / long-horizon & "simulate inter-chunk" ideas

### 11. World models / Genie (DeepMind, 2024) & action-conditioned video prediction
- Genie/Genie-2: learn latent-action world models from video; action-conditioned
  future prediction. Relevant to @namak-kun's hypothesis that NitroGen *implicitly*
  predicts state across its 18-action chunk and that this could be exploited to
  *synthesize* multi-chunk supervision without a live env. A world model could provide
  the "next frame after chunk i" needed to chain chunks offline.

### 12. Hierarchical RL / options & subgoal latent policies
- Options framework, HIRO/HRL, latent-subgoal methods: high-level proposes subgoals at
  coarse cadence, low-level executes. Theoretical backing for the K×A "one plan over A
  chunks" cadence and for learning a phase/progress signal.

---

## E. Takeaways for our project (what to actually borrow)

1. **We are rebuilding GR00T's dual system** for games (NitroGen = System 1). Read
   GR00T N1.5's VLM↔DiT alignment improvements — same problem we're solving (make
   language actually move actions). [highest priority read]
2. **Helix is the template for cross-chunk:** single evolving latent, decoupled S1/S2
   frequencies, S1 consumes latent every step while S2 refreshes sparsely. Validates
   K×A blocks + persistence; suggests the latent must carry enough "phase/state."
3. **LCB validates latent (not language) bridging** + end-to-end training of the
   adapter — exactly our PlanHead. Supports unfreezing/jointly training the bridge.
4. **Hi Robot / π0.5 validate language-plan hierarchies** and give the (instruction →
   decomposed command → execution) data structure for our Stage-2 transcripts.
5. **Within-chunk temporal composition (SEQ) is known-hard** (composable diffusion):
   our EXP-011 failure is consistent; likely needs DiT capacity (unfreeze/LoRA) +
   possibly explicit temporal conditioning of plan tokens, not just static tokens.
6. **World models (Genie)** are the principled route to the "simulate inter-chunk"
   idea if we want offline multi-chunk supervision without a live game env.

### Concrete reading list (verify arXiv ids before citing)
- GR00T N1 / N1.5 (NVIDIA 2025) — dual system, VLM+DiT flow matching.
- π0 (2410.24164) & π0.5 — flow-matching VLA, hierarchy.
- Hi Robot (2502.19417?) — system 1/2 with language commands.
- Figure Helix (blog/tech report 2025) — latent bridge, dual frequency.
- LCB "Latent Codes as Bridges" (2405.04798?) — latent high/low bridge.
- Diffusion Policy (Chi 2023, 2303.04137).
- Octo (2405.12213), OpenVLA (2406.09246).
- Genie / Genie-2 (DeepMind 2024) — latent-action world models.
