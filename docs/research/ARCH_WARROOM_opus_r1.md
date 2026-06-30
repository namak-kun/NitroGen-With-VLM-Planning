## [OPUS48 — Round 1] 2026-06-27T23:38:00-07:00

**Stance in one sentence:** The owner's fast/slow split should be drawn along **prefill-vs-decode**, not **shallow-vs-deep**. The fast channel is the *same frozen VLM* run as a single **full-depth, decode-free PREFILL** over `[frames + cached plan-text + K learned plan-query register tokens]`, read **every chunk** through the **existing bridge**; the slow channel adds the autoregressive plan-text decode **every A chunks, asynchronously**. One VLM, one bridge, one representation space — so there is *no* shallow-fast↔deep-slow distribution shift to fix, and exact null-invariance is preserved untouched.

I reject the owner's early-exit-by-depth idea (it attacks the wrong cost and creates an ungrounded second representation) **and** the prior research note's "separate tiny semantic fast module" (it re-derives grounding from scratch and forks the representation). I keep only a *tiny* zero-init DiT-side modulator for **localization**, not as the semantic source.

---

### 0. The reframe that drives everything (grounded in THIS repo's code)

The corrected facts list four gaps — bandwidth, rate, staleness, **blocking latency**. The owner proposes to cut blocking latency by tapping *early VLM layers*. But look at where the latency actually is:

- The bridge consumes **last-layer** hidden states from a **single forward** (`nitrogen/planner.py:173`, `:214–215`: `out.hidden_states[-1]`, `use_cache=False`). The resampler cross-attends over those → K plan tokens. That forward is a **prefill**: one parallel pass over `[~600 image tokens + a few text tokens]`.
- The blocking op in the live loop is **not** that prefill — it is `PlanEncoder.generate_plan(...)`, an **autoregressive decode** of `max_new_tokens=24` (`nitrogen/planner.py:238–275`), called every `replan`/A chunks (`planner_poc/eval_policy.py:124–126`; `planner_poc/cavestory_closed_loop.py:94`). The decoded text is then re-encoded by `encode_text`.

So per replan the VLM does **24 sequential, memory-bandwidth-bound token-forwards** (the decode) **plus** one prefill. On a 2B model on an A6000 (~768 GB/s, weights ~4 GB bf16), a decode step is ≥ ~5 ms just to stream weights, realistically ~15–25 ms/token → **~0.4–0.6 s for 24 tokens ≈ one whole chunk**. The prefill is one *compute-bound* pass (~2–3 TFLOP) → **~20–60 ms ≪ a chunk**. (Exact numbers must be measured — that is Experiment-0 — but the asymmetry is *structural*: sequential+memory-bound decode ≫ parallel+compute-bound prefill.)

**Conclusion:** Early-exit-by-depth shaves a fraction off the *already-cheap* prefill and does **nothing** about the decode (the real blocker). The right move is to take the decode **off the per-chunk critical path** and read a **full-depth** latent from the cheap prefill every chunk. Depth is not the axis; *autoregression* is.

This reframe is exactly what the owner's cited inspiration actually does. **iSHIFT (arXiv:2512.22009, Mehrotra et al., 2025-12-26)** — verified abstract — is "**Implicit Slow-fast Hybrid Inference with Flexible Tokens**": its fast/slow is (i) **latent thinking / implicit chain-of-thought** (reason in latent space instead of *decoding* text) and (ii) **special perception tokens that guide attention** (slow = detailed visual grounding, fast = global cues). iSHIFT is **not** a depth-wise early-exit; it is *decode-free latent readout + learned attention-routing tokens* — precisely my proposal. The owner's own source argues *against* the shallow-layer reading of it.

---

### 1. Early-exit shallow-latent fast path vs separate tiny module — I defend a THIRD option

**Verdict: neither.** The fast semantic channel is the frozen VLM's own **full-depth prefill** (no decode), reusing the existing resampler/adapter. Why both alternatives lose:

**Against the owner's early-exit (depth tap):**
1. **Wrong lever** (see §0): it cannot remove the autoregressive decode, which is the actual blocking cost.
2. **Ungrounded representation.** Decoder-only LLM early layers carry mostly lexical/positional structure; task-relevant and "semantic-direction" features emerge mid-to-late (logit-lens / early-exit & layer-probing literature; *DeeBERT/early-exit* shows confident task decisions require later layers for hard inputs). The bridge today is trained on **last-layer** states. A shallow tap is a *different vector space* → needs a **new trained readout head**, which is **exactly** the shallow-fast↔deep-slow **distribution shift** the debate prompt flags.
3. **Subtle conflation.** "Early-exit" mixes *depth* (early layers of one forward) with *autoregression* (early tokens of the decode). The plan text is the **output** of decoding; it does not exist at shallow depth of the prefill. So a shallow tap gives you neither the decoded plan nor a grounded latent — just an early-layer prompt encoding.

**Against a fully separate tiny semantic module (the research note's standalone "FastPlanMod" as the semantic source):**
1. It must **re-learn pixel→semantics grounding from scratch** → needs its own data and training, and yields a **second representation** that can drift from the VLM's (OpenHelix's documented failure: dual-system latents collapse to static instruction semantics).
2. **Redundant:** the DiT **already** sees the live frame every chunk (corrected fact) — the fast *visual reflex* exists. We don't need a second visual encoder to invent semantics; we need the *existing* semantic encoder run cheaply.

**Why full-depth prefill wins:** grounded **by construction** (same last-layer space the bridge already aligns to), **shares the bridge** (no new representation), and removes the decode (the real latency). Precedent: **Helix** (verified) — S2 (7B VLM, 7–9 Hz) distills *all* task-relevant info into a **single continuous latent**; S1 (80M, 200 Hz) consumes the **latest** latent + its own high-rate frames. Helix's fast path is **not** a shallow VLM tap and **not** a separate semantic net — it is the **cached latent of the same VLM**, re-grounded by S1's fast frame loop. I am proposing the literal Helix pattern, adapted so the "latent" is recomputed at **prefill-rate every chunk** rather than held fixed between 7–9 Hz updates.

I retain a **tiny zero-init DiT-side modulator** (FiLM/AdaLN gate + dynamic CFG) — but only for **localizing** the cached intent to the current frame, *not* as the semantic channel. That is the legitimate, cheap part of the research note's FastPlanMod.

---

### 2. What flows on each path, at what rate, and how null-invariance is preserved

**SLOW path — every A=2 chunks (~1.2 s), async, off critical path.**
- Full VLM **autoregressive decode** → human-readable plan text (+ optional structured slots: `{verb, object, direction, horizon}`). Used for interpretability/logging, long-horizon subtasking (π0.5 pattern), and to **refresh the prompt cache**. Optionally a richer pooled "deep latent."

**FAST path — every chunk (~prefill latency).**
- One **full-depth prefill** over `[recent frames + latest cached plan text + K learned "plan-query" register tokens]`. Read the **register tokens' last-layer hidden states** → existing `PlanResampler`→`PlanAdapter` (`nitrogen/planner.py:53–120`) → K plan tokens → injected at the `_PLAN_TOKEN` slots (**unchanged** injection, `nitrogen/flow_matching_transformer/nitrogen.py:482–495`).
- This is **frame-fresh every chunk** → kills **staleness (gap c)**. The intent (text) varies slowly; the *grounding/localization* of that intent against the live frame is recomputed every chunk.
- **Bandwidth (gap a):** widen `K` 8→32 (config `num_plan_tokens`, `planner.py:32`). 32 ≪ ~256 vision tokens, so it doesn't structurally out-vote vision; pair with the existing zero-init **plan-adaLN** global authority (`planner.py:47`, `adaln_cond` `nitrogen.py:338–348`) which already exists for exactly this "8 tokens outvoted by vision" problem (EXP-048).
- **Rate (gap b):** semantics now refresh at **prefill-rate (every chunk)**, not every A chunks.

**Null-invariance — preserved by construction, mechanism unchanged.**
- The register tokens live **only inside the VLM prompt**; they never touch the DiT directly. Only their distilled **K outputs** reach the DiT, through the **same masked `_PLAN_TOKEN` slots**.
- Under null/masked plan: `apply_null_mask` zeros those K positions out of DiT cross-attention (`nitrogen.py:594–605`) and `adaln_cond` forces the temb offset to 0 for dropped rows (`nitrogen.py:338–348`, zero-init proj). → base NitroGen **bit-for-bit**. Widening K or adding register tokens does **not** alter this: more masked slots are still fully masked. CFG-style plan guidance still amplifies only the real-plan delta.

---

### 3. Concrete near-real-time scheme (what runs concurrently, how stalls hide)

Three async processes across 4× A6000 (shared-memory latent, Helix-style):

- **GPU0 — VLM SLOW worker:** continuous autoregressive decode of plan text; writes `{plan_id, text, slots, ts}` to shared memory. **Never on the critical path.** Triggered every A chunks *or* on a replan-trigger event (§4).
- **GPU1 — VLM FAST prefill:** one full-depth prefill **every chunk** over `[frames + latest cached plan text + register tokens]` → K plan tokens → **versioned latent buffer** `{plan_tokens, plan_id, age_chunks, ts}`. **KV-cache trick:** the cached plan-**text** prefix KV is reused across fast prefills; only the changing image tokens + register tokens are recomputed → cheaper fast prefill. If still heavy, downsample the fast-path image (Qwen3-VL dynamic resolution) — a knob orthogonal to depth that does **not** fork the representation.
- **GPU2/3 — DiT actor:** the existing 16-step flow over the 18-action chunk; consumes the **latest** latent buffer + the **live frame**; runs the control loop. Already frame-reactive.

**Stall-hiding = Real-Time Chunking (RTC, arXiv:2506.07339, Physical Intelligence)** — verified: an **inference-time** algorithm, **no retraining**, **applicable to any diffusion/flow VLA out of the box**. It "generates the next action chunk while executing the current one, **freezing** actions guaranteed to execute and **inpainting** the rest." This is a *direct drop-in* for NitroGen (a flow DiT): generate chunk *t+1* while executing *t*, freeze the guaranteed prefix, inpaint the remainder → smooth across (i) DiT inference latency and (ii) a **plan-version switch** (apply a new `plan_id` only at the inpaint boundary, never mid-frozen-prefix → no jitter).

**Train-time staleness offset (Helix):** during training, feed the DiT plan tokens **delayed by 0–A chunks** so deployment staleness is **in-distribution**. Helix explicitly calibrates this offset to the S1/S2 latency gap; it is the single most important "free" robustness trick here.

Net: **text intent at decode-rate (slow), grounded semantics at prefill-rate (every chunk), motor reflex at DiT-rate** — three clean timescales, no process ever blocks on the decode.

---

### 4. Training objective + data for the fast path (honest about the data we lack)

**Stage 0 — reuse what already works (zero new data).** The fast prefill reads the **same last-layer space** the current bridge is trained on, so existing Stage-1 synthetic-plan alignment + SupCon (`contrastive_*`, `planner.py:40–42`) transfer directly. Add register tokens, widen K, fine-tune **bridge only** (VLM frozen). The EXP-052 `text_only` path (`planner.py:217–225`) already extracts frame-contextualized text tokens — the register-token readout is a small, in-family extension.

**The fast path's training signal = self-distillation slow→fast (this is the answer to "what supervises the shallow/fast tap").**
- **Teacher** = the SLOW path output (full decode → `encode_text` → K tokens — i.e., today's exact pipeline).
- **Student** = the FAST path (decode-free prefill register-token readout).
- Loss = per-token **MSE + InfoNCE**, which **already exists** as the privileged-distillation machinery (`distill_weight`, `planner.py:43`, EXP-045). The student is trained to **reproduce the decoded-plan latent without decoding**, while seeing the **current** frame. This *eliminates* the fast↔slow distribution shift instead of merely tolerating it — fast is supervised to track slow. (Same training budget the owner would spend grounding a shallow tap, but it yields **one** grounded full-depth representation and **removes** the decode.)

**The genuine gap — no mid-mistake recovery labels — is filled by the frame-exact emulator, not by demos.** This is where 2D-games-from-pixels actually **beats** robotics (dense checkable reward + reset; the project's own conclusion "we can't escape envs"):
- From save-states: inject a **forced perturbation** (wrong-action prefix), branch **K plan/latent variants**, roll out, **rank by emulator reward** → reward-weighted regression / save-state **GRPO** on bridge+LoRA (no critic net needed). This supervises *recover-after-mistake* and *re-grounding under a stale plan*.
- **Staleness augmentation** (delay latent 0–A chunks) is a **free** label from any rollout.
- **Dynamic-CFG / replan-trigger head:** supervise `w_plan(t)` and an "early-replan" gate from cheap emulator telemetry (stuck / progress / death). Trained offline; at deploy it both modulates guidance and **triggers the slow worker** when grounding collapses.

**Honest ROI:** do **not** expect counterfactual recovery to emerge from synthetic plan-following alone (the research note's explicit warning, and the project's prior left/right finding). The latency/staleness wins (§3) are cheap and high-ROI and *independent* of any representation breakthrough; the recovery wins require the emulator loop and are the expensive part — gate them behind the decisive cheap experiment below.

---

### 5. The single highest-leverage FIRST experiment — and what NOT to build

**EXPERIMENT-0: "Decode-free prefill latent" A/B — on the EXISTING best checkpoint, near-zero build.**
Take `btn_s600` (the main eval ckpt). **Without any new training**, at eval, replace the closed-loop `generate_plan → encode_text` with a **single prefill** that reads K plan tokens directly from (a) the frame-contextualized text tokens — the `text_only=True` path **already exists** (`planner.py:217–225`) — and/or (b) appended register tokens. Measure:
1. **Wall-clock per chunk: decode-path vs prefill-path** (this *quantifies the §0 asymmetry* — the linchpin of the whole thesis).
2. **Behavioral parity:** direction/button selectivity, plan-flip rate, and **exact null-invariance** (assert base-equality) of decode-free vs decoded plan, using the existing `eval_buttons.py` / `eval_balance.py` / probes.

**Decision rule:**
- If decode-free prefill ≈ decoded plan **behaviorally** AND is ~10× cheaper/chunk → the thesis is validated **essentially for free**, and the blocking latency is *already gone*. Then, in order: (i) register-token self-distillation + widen K=32; (ii) async + RTC + staleness-offset training; (iii) emulator recovery/GRPO.
- If decode-free **loses** behavioral quality → the decoded *text* genuinely adds information the prompt-conditioned readout can't recover. The fix is **not** shallow taps; it is keep the decode **async** (slow path) and lean harder on RTC + staleness training. Either outcome is decisive.

Why first: it tests the core claim (the cost is the decode, not the depth; full-depth prefill latents suffice) on **existing weights** with **near-zero engineering**, and the winning branch *is* the foundation for everything else.

**Do NOT build yet:**
- **Early-exit shallow taps / a separate shallow readout head** — wrong lever (§0), adds a 2nd ungrounded representation, introduces the very distribution shift we're told to avoid.
- **VLM-LoRA** *(flag cost)* — only after frozen prefill shows headroom. It moves the **very latents the bridge is aligned to** and risks the null-invariance + hard-won contrastive/left-right gains. High-cost, defer.
- **Online/test-time gradient TTT in the live loop** — latency-incompatible; do save-state GRPO **offline/between attempts** instead.
- **Hypernetwork-generated full DiT weights**; **a large video world model** — glamorous, low first-pass ROI for 2D pixels.

---

### 6. Pre-empting the opponent's strongest counters

- **"A full prefill every chunk is too expensive; early-exit is genuinely cheaper."** The **decode** dominates, not the prefill (§0); the prefill is already < 1 chunk and is the part you keep. Early-exit's saving is a fraction of an already-cheap pass, bought with an ungrounded representation + a new head + train/deploy shift. If the prefill is still heavy, the knobs are **orthogonal to depth** (fewer vision tokens, KV-cache the plan-text prefix, dedicated A6000) and don't fork the representation. And we **measure** it in Experiment-0 rather than asserting.
- **"Train the shallow latent to be grounded (distill deep→shallow); then it's cheap AND grounded."** Strictly more machinery for strictly less benefit: you still pay a partial prefill, you now maintain **two** representations kept aligned by an extra loss, and the saving over a full prefill is marginal because the prefill isn't the bottleneck. My self-distillation spends the *same* effort making a **decode-free full-depth** readout match the decoded plan — **one** representation, grounded by construction, and it removes the decode.
- **"Your fast path adds no NEW info between replans — it just re-grounds a stale plan."** That is **the point**, and it is the Helix pattern: high-level intent is slowly-varying; what must be frame-fresh is the **localization** of that intent (where is the gap/enemy/ledge *now*), which the full-depth prefill recomputes against the live frame every chunk. Reflex is handled by the DiT's own frame input + the tiny FiLM modulator. Three timescales, cleanly separated.
- **"iSHIFT supports early-exit / adaptive depth — you're cherry-picking."** Its verified abstract (arXiv:2512.22009) says **Implicit** slow-fast with **latent thinking** (skip the decode) + **perception tokens** (attention routing), slow=detailed grounding / fast=global cues. That is decode-free latent readout + learned routing tokens — **my** design — **not** depth-wise early-exit. The owner's own inspiration argues against the shallow-layer interpretation.

---

### 7. Citations (verified against primary sources this session)

- **Figure Helix** — figure.ai/news/helix (fetched). S2 = onboard 7B VLM @ **7–9 Hz**, distills task-relevant info into a **single continuous latent**; S1 = **80M** cross-attn enc-dec @ **200 Hz**, projects the latent into its token space and **concatenates with S1's own high-rate visual features**; **asynchronous shared-memory latent**, dual-GPU split; trained **end-to-end** (grads S1→S2); **temporal offset** in training calibrated to deploy latency.
- **iSHIFT** — arXiv:2512.22009, Mehrotra, Rebbapragada, Bonthu, Balasubramanian (2025-12-26), abstract fetched. "**Implicit Slow-fast Hybrid Inference with Flexible Tokens**"; latent thinking (implicit CoT); **perception tokens** guide attention; slow=detailed visual grounding / fast=global cues; 2.5B. **Not** depth early-exit. *(The generic web-search summary of this paper was hallucinated — primary abstract used.)*
- **π0.5** — pi.website/blog/pi05 (fetched). First **decodes a high-level action as text** (auto-regressive), then emits a **50-step / 1 s continuous action chunk via flow matching**; one model, **discrete AR decode + continuous flow** pathways. = the "decode-then-act" pattern I move **off** the per-chunk critical path.
- **Real-Time Chunking (RTC)** — arXiv:2506.07339, Physical Intelligence (abstract fetched). Inference-time, **no retraining**, **any diffusion/flow VLA**; generate next chunk while executing current, **freeze** guaranteed actions + **inpaint** the rest; robust to inference delay (Kinetix + real bimanual). Drop-in for NitroGen chunk-stitching.
- **GR00T N1.5** (NitroGen's lineage, per seed + repo) — V/L embeddings → connector → flow DiT with **cross-attention** to V/L + **AdaLN** diffusion-step conditioning. Mirrored in this repo: cross-attn plan-token injection (`nitrogen.py:482–495`) + zero-init plan-adaLN temb offset (`planner.py:47`, `nitrogen.py:338–348`).
- **LCB: Latent Codes as Bridges** — arXiv:2405.04798. Learned latent/`<ACT>` bridge tokens between LLM and low-level policy → precedent for the **plan-query register tokens**.
- **FiLM** — arXiv:1709.07871; **DiT/AdaLN** — Peebles & Xie, arXiv:2212.09748. Mechanism for the zero-init, null-safe global modulation path.

### 8. Exact attach points in THIS repo (for the referee/opponent)
- Register tokens: append to the VLM prompt in `PlanEncoder.encode_multimodal` (`planner.py:181–226`); read their last-layer positions exactly like the `text_only` keep-mask (`:217–225`). Feed the **same** `PlanResampler`/`PlanAdapter`/`PlanHead` (`planner.py:53–333`). **No DiT change.**
- Widen K: `PlannerConfig.num_plan_tokens` 8→32 (`planner.py:32`) + matching `_PLAN_TOKEN` slot count.
- Global authority: enable existing zero-init `plan_adaln` (`planner.py:47`; `adaln_cond` `nitrogen.py:338–348`).
- Self-distillation: existing `distill_weight` MSE+InfoNCE (`planner.py:43`); teacher = decoded-plan tokens, student = prefill register-token tokens.
- Null-invariance test after every change: assert `apply_null_mask` (`nitrogen.py:594–605`) base-equality.
