# ARCH WARROOM R9 — Opus-4.8: the collapse is a DATA-design artifact, fix it with the action AS the label

**TL;DR.** Root cause is **DATA, not objective and definitely not capacity**: a single terse plan held
constant across all training chunks makes the plan input ~constant, so plain BC's loss-optimal solution is
"ignore the plan, emit the marginal advance policy" — it *cannot* preserve a plan→action map it never sees
vary. The base bridge already does duck-on-command at K=8 (1.2%→10.8%), so K is sufficient (option 4 dead).
**Run ONE thing tonight: re-fit SMW demo-fit with PER-CHUNK situational plans where the plan label is
DERIVED FROM THE CHUNK'S OWN ACTION** (DOWN-held→duck plan, LEFT→retreat, idle→wait, airborne→jump,
else→advance) — this is simultaneously the Q3 diagnostic AND Q2-fix(1), needs ~15 lines in
`load_demo_chunks`, and the VLM can mint these labels for free at scale (Q5). **Pass bar:** under "press
down to duck under it" the demo-fit emits **DOWN-dpad ≥ 5%** (base 10.8%, collapse 0.0%) AND monotone in
plan-explicitness, WHILE SMW Δ_plan POST ≥ +20 (≥0.8× pooled +26, bootstrap CI>0, 3/3 seeds), null
max|diff|=0.0. Run the **KL-anchor arm in parallel on a 2nd GPU** so if data alone fails we have the
objective-fix in the same night. 3 seeds each.

---

## Q1 — Root cause: rank DATA > OBJECTIVE >> CAPACITY/ARCH (c is refuted)

**(c) CAPACITY/ARCH — REFUTED outright, do not spend a GPU-hour on it.** The *base* bridge expresses
duck-on-command at K=8: 1.2%→3.4%→10.8% as the plan goes terse→"dodge"→"press down" (a clean monotone
plan-response, 9×). K=8 demonstrably has the bandwidth and the architecture to route a rare situational
action from plan text to the DOWN dim. The collapse is therefore NOT a bandwidth ceiling. K=8→32 (option 4)
cannot be the cause and would also (i) change `resampler.queries`/`null_plan` shapes →
btn_s600 `plan_head` no longer loads → forces a full Stage-1 retrain (not a one-night run), (ii) leave the
*overwriting* mechanism unaddressed. Dead.

**(a) DATA — PRIMARY.** demo-fit conditions every chunk on ONE fixed string
(`BATTERY['smw']['correct']`, never says "duck"). Across the whole training set the plan embedding is
**~constant**, so the (frame,plan)→action map has zero gradient pressure to stay plan-conditional. The
BC-optimal map given a constant plan is the **marginal action distribution** of the demos = advance+jump
(duck is 1.7% → rounds to 0). The plan_head doesn't "forget duck on purpose"; it collapses to a
plan-*independent* advance policy because nothing in the data rewards plan-dependence. This is the same
mechanism that made the LIVE plan match the oracle after fitting (results §STALENESS): the bridge learned
to *ignore* plan variation. Ignoring is great for staleness, catastrophic for expressiveness.

**(b) OBJECTIVE — SECONDARY / DOWNSTREAM.** Plain BC mode-collapses rare actions and has no trust region
to base (the war-room already learned this: R7 noted the ones-mask's *implicit* KL was the only thing
that ever limited drift). But (b) is **enabled by (a)**: with a *constant* plan, the collapse is
loss-optimal, so even a perfect anchor-free BC would do it. With *varied* plans (each situational plan tied
to its action), the BC loss can no longer satisfy the duck chunks by emitting advance — it is *forced* to
keep the duck→DOWN map. So (a) is upstream; (b) is the knife (a) hands it.

**The single discriminating measurement = the Q3 diagnostic** (re-fit with per-chunk action-derived
plans, hold objective fixed at plain BC):
- DOWN-on-command RETURNS (≥5%) ⇒ **(a) DATA** confirmed; richer plans is the fix; scale via VLM labels.
- STILL 0% even on duck-labeled chunks ⇒ **(b)/(c)**: BC mode-collapses regardless of labels ⇒ ship the
  KL-anchor (preserves ALL actions, taxonomy-free) or the zero-init modulator (preserves by construction).

That one re-fit cleanly forks the tree. It is also *itself* the leading fix, so it is never a wasted run.

---

## Q2 — The fix to test tonight: (1) RICHER PER-SITUATION PLANS, with the action as the label

**Pick: option (1).** It is the cheapest, the most directly implied by Q1(a), and the only one that is
*generalist by construction* — it teaches the bridge the full plan→action map (duck/wait/jump/retreat/
advance) rather than band-aiding one axis, and the labels are free (Q5). Crucially the infra already
supports it: `build_batch` encodes `s["plan"]` **per sample** (rwbc_actor_adapt.py:156), so per-chunk
plans need NO new batch code — only a per-chunk labeler in `load_demo_chunks` (demo_bc.py:76).

**Exact implementation (≈15 lines, additive `--situational-plans` flag in demo_bc.py):**

```python
# action dims (map_action): DOWN=a[:,22]>0.5  LEFT=a[:,21]<0.5  RIGHT=a[:,21]>0.5
#                           UP=a[:,22]<0.5     JUMP(B)=a[:,18]>0.5  RUN(Y)=a[:,20]>0.5
SIT = {
 "duck":    "Press down to duck under the enemy or hazard, then keep moving right.",
 "retreat": "Turn around and move left, back away from the hazard.",
 "wait":    "Stop and wait, hold still until the path ahead is clear.",
 "jump":    "Jump now to clear the gap or enemy ahead, then keep moving right.",
 "advance": BATTERY[game]["correct"],          # the original terse plan
}
def label_chunk(a):                              # a:(18,25)
    down=(a[:,22]>0.5).mean(); left=(a[:,21]<0.5).mean()
    right=(a[:,21]>0.5).mean(); jump=(a[:,18]>0.5).mean()
    if down>=3/18:                 return "duck"
    if left>right and left>0.30:   return "retreat"
    if right<0.15 and jump<0.15:   return "wait"
    if jump>=0.40:                 return "jump"
    return "advance"
```

Attach `SIT[label_chunk(chunk)]` as each chunk's `plan` in `load_demo_chunks`. **No human narration, no
VLM needed for tonight** — the demo action IS the supervision for which situational plan to attach. This is
the load-bearing insight: we don't need the gold narration (it only exists for SMW); the action labels
itself, so it generalizes to MMX/SMB1/any game out of the box (Q5).

**Guard against frame→action shortcut (the one real risk of (1)):** giving the duck-frame both the
duck-plan and the duck-action lets the model cheat via the frame. Two existing defenses, both free:
(i) `plan_dropout=0.15` already nulls the plan on 15% of chunks → on those the duck must come from the
*frame* alone, so the model is penalized if it leans only on the frame for the non-dropped duck chunks
(the plan must carry marginal signal); (ii) the **expressiveness battery (Q4) tests duck-on-command via an
EXPLICIT duck plan on the SAME states regardless of frame** — if it's a frame shortcut, the explicit duck
plan won't move DOWN and the battery catches it. Report both.

**Optional free amplifier on this arm:** add `--residual-mask` (already in demo_bc.py:156, the *supervised
target-weight* use the R7 pivot endorsed — NOT the refuted RWBC gate). On duck chunks DOWN deviates from
base-null → mask→1 → the rare duck dims get full gradient instead of being washed out by the dominant
advance dims. Cheap insurance for rare-action retention; run it as a 4th seed-set if GPUs are free.

**EXACT pass criterion (SMW, 3 seeds, mean over ≥4 duck-relevant demo states, 12 chunks):**
1. **Expressiveness restored (primary):** under `"press down to duck under it"`, demo-fit
   **DOWN-dpad ≥ 5.0%** (base 10.8%, collapse 0.0%) — recover ≥ ~½ of base's commanded duck — AND the
   rate is **monotone non-decreasing** in plan explicitness (terse ≤ "dodge" ≤ "press down"), slope > 0
   ⇒ it is *plan-responsive*, not a constant DOWN floor.
2. **Retention (advance not sacrificed):** SMW Δ_plan POST **≥ +20** (≥0.8× pooled +26; within pooled CI
   [+18.9,+30]), 3/3 seeds, bootstrap 95% CI on the change **> 0**.
3. **Null-invariance:** max|diff| (null path vs base) = **0.0** exact (plan-head-only + masked null).
4. **Battery (Q4):** obeys ≥4/5 contrasts in base's sign, responsiveness index ≥ 0.5 (below).

**FALSIFIER:** if the richer-plan re-fit gives press-down **DOWN < 2%** (still collapsed) ⇒ DATA is NOT
the binding cause ⇒ Q1(a) refuted ⇒ commit to the **KL-anchor** (run in parallel tonight, see Q3). If
richer plans restore duck but **Δ_plan POST < +20** (retention lost) ⇒ the plan budget is being split
advance-vs-duck at this capacity ⇒ escalate to the **zero-init additive modulator** (option 3, which adds
advance authority *without* touching the duck pathway; already built, planner.py:305-309).

---

## Q3 — Diagnostic-first? YES — but run it CROSSED so it settles Q1 *and* ships Q2 in one night

I agree with the orchestrator: the diagnostic goes first. But "diagnostic then maybe a fix" risks burning
a night if data alone half-works. We have 4×A6000 and these fits are short (~600 steps). So run a **2-cell
cross tonight, in parallel**, not sequentially:

| cell | plan data | objective | what it tells us |
|------|-----------|-----------|------------------|
| **P0A0** (control) | single terse | plain BC | reproduce the 0.0% collapse (sanity) — 1 seed |
| **P1A0** (the diagnostic = fix-1) | per-chunk situational (above) | plain BC | DATA? duck returns ⇒ Q1(a) — **3 seeds** |
| **P0A1** (objective fix) | single terse | + KL-anchor to base | OBJECTIVE? anchor alone restores duck ⇒ Q1(b) — **3 seeds** |

P1A0 on GPU0-1, P0A1 on GPU2-3, P0A0 a quick sanity on any. **Decision rule next morning:**
- P1A0 passes (duck≥5% + Δ_plan≥+20) ⇒ **DATA**; ship richer-plan demo-fit; Q5 scales it. (P0A1 is a bonus
  data point on whether anchor *also* helps — keep if it stacks.)
- P1A0 fails, P0A1 passes ⇒ **OBJECTIVE**; ship KL-anchor (keeps the cheap single plan).
- both fail ⇒ **ARCH**; escalate to the zero-init modulator (option 3) — guaranteed-safe fallback.

**Per-chunk label definition (exact, so it's codeable):** a chunk (18×25) is **duck** iff DOWN-dpad
(`a[:,22]>0.5`) is held in **≥3 of 18** frames; else **retreat** iff LEFT-rate>RIGHT-rate and LEFT>0.30;
else **wait** iff RIGHT<0.15 and JUMP<0.15 (≈idle); else **jump** iff JUMP(`a[:,18]>0.5`)≥0.40; else
**advance**. Deterministic, threshold-tunable, no VLM. (These thresholds matter: duck@3/18≈human 1.7%
floor is generous on purpose — we WANT the rare chunks captured.)

**KL-anchor (P0A1) definition, exact:** snapshot the PRE-fit `plan_head` (frozen reference `g0`). Each
step, in addition to the demo BC loss, draw a small anchor batch of (demo-frame × a plan sampled from the
5 SIT plans incl. duck) and add `λ·mean((v_θ − v_g0)²)` where v is the DiT velocity at matched (x_t,t,
frame,plan) (same noise seed for θ and g0). This is a functional L2-to-base trust region on a BROAD plan
distribution → it preserves duck-on-command (and every other base behavior) *without a label taxonomy*.
Start `λ=0.3`; if Δ_plan retention fails, anneal λ down; if duck not preserved, anneal up. One frozen
PlanHead deep-copy, one extra forward/step — fine at 600 steps.

---

## Q4 — Expressiveness battery (catch whole-action-space collapse, not just DOWN)

New `planner_poc/expressiveness_battery.py`, **reusing `plan_graded_test.py`'s harness** (feed FIXED plan
strings through `_sample_chunk` from demo start states with **matched per-(state,chunk) noise seeds** →
low variance), tally per-action press rates. 5 contrasts, each a (probe plan − neutral plan) rate delta
the BASE bridge already obeys:

| # | contrast (probe vs neutral) | action dim measured | base obeys (expect) |
|---|------------------------------|---------------------|---------------------|
| 1 | **DUCK**: "press down to duck under it" vs terse advance | DOWN `a[:,22]>0.5` | +9.6pp (1.2→10.8) ✓ |
| 2 | **RETREAT**: "turn around, go back left" vs terse advance | LEFT `a[:,21]<0.5` | +rate (BATTERY 'bad' flips dir) ✓ |
| 3 | **JUMP**: "jump now over the gap" vs "walk, don't jump" | JUMP `a[:,18]>0.5` | + jump-rate ✓ |
| 4 | **WAIT**: "stop and wait, hold still" vs terse advance | RIGHT `a[:,21]>0.5` | − right-rate (motion drops) ✓ |
| 5 | **CLIMB/UP**: "climb up the vine" vs terse advance | UP `a[:,22]<0.5` | + up-rate (keep only if base obeys; else drop to 4) |

**First, characterize BASE** on all 5 (one run) — keep only contrasts where base shows |Δ| with the right
sign at a usable magnitude; that defines the reference vector `b = [b1..b5]`. Then for any demo-fit
`d = [d1..d5]`:

**Aggregate pass bar:**
- **Sign-match ≥ 4/5** contrasts agree with base's sign (no inversion). A *sign-flip* (e.g. duck plan
  *reduces* DOWN below terse) = automatic FAIL — that's active anti-expressiveness.
- **Responsiveness index** `R = mean_i clip(d_i / b_i, 0, 1) ≥ 0.50` — the demo-fit must recover ≥ half of
  base's plan-conditioned action shift, averaged across the action space (not just DOWN).
- **DUCK mandatory:** contrast #1 absolute DOWN ≥ 5% under the explicit duck plan (the known failure;
  cannot be averaged away).

Score base, pooled-collapse (expect R≈0), and every demo-fit cell on this battery. This is the
generalization of "DOWN-on-command" the seed asks for — it converts "preserve expressiveness" from a slogan
into a 5-number vector with a pass threshold, and it'll catch a taxonomy-treadmill failure (richer plans
restoring only the labeled actions while silently dropping #5).

---

## Q5 — Scaling: the fix makes the generalist story BETTER, and the VLM labels for free

**Does it change the pooled-generalist story? Yes — improves it.** The R8 pooled fit (one plan_head over
SMW+MMX+SMB1, one plan per game) is what *caused* the collapse: one constant plan per game ⇒ plan-
independent advance policy ⇒ 0% duck. Re-pool with **per-chunk situational plans** and the SAME pooled
machinery (`pooled_demofit.py`, unchanged except the per-chunk labeler) now learns one generalist that
both advances AND retains the full situational action space. Same params, same null-invariance, same
disjoint merge with the Sonic LoRA.

**Can the VLM mint the per-situation labels for free? Yes, two tiers — and tonight needs neither:**
- **Tier 0 (tonight, free, deterministic): the action IS the label.** The action-derived labeler above
  needs no narration and no VLM. It generalizes to MMX/SMB1/every game immediately. This is the key result
  for the one-generalist goal: the gold SMW narration is a nice-to-have, not a dependency.
- **Tier 1 (scale, the bootstrap path): VLM-generated grounded objectives.** `objective_label_demo.py`
  already shows the frozen Qwen emits grounded per-frame objectives; the now-allowed Qwen-LoRA sharpens
  them. Use it to caption each chunk's *situation* ("enemy overhead → duck") for richer, less templated
  plans. Noisier than Tier 0 → use it as augmentation, not the sole source. Frozen-VLM-first (variance)
  still holds; the LoRA is upside, not a requirement.

So the path is: prove DATA(a) on SMW tonight → swap `pooled_demofit.py` to per-chunk action-labels →
re-fit the pooled generalist → re-merge with Sonic LoRA. No human-narration bottleneck, no per-game model,
null-invariance preserved. The expressiveness battery becomes a standing regression gate on every future
pooled fit.

---

## Steelman the OTHER pick — the KL-anchor (option 2), and what would flip me

**Strongest case for the KL-anchor over richer plans:** my fix is a *label taxonomy* — it can only
preserve the situational actions I enumerate (duck/retreat/wait/jump). Real games have an open-ended set
(climb, swim, charge-shot, brake, look-up-to-scroll, pogo) that no finite SIT dict captures, and a 1.7%-
rare action that I *don't* label will still be mode-collapsed exactly as duck was. The KL-anchor is
**taxonomy-free**: a single L2-to-base on a broad plan distribution preserves *every* base behavior the
bridge already has — the whole action space at once — while still letting the advance demos pull the
trained axis. It keeps the cheap single-plan data (no labeler to maintain, no threshold to tune, no
frame→action shortcut risk because it never co-labels action with plan). It is also exactly the
"trust region to base" this war-room concluded was the missing ingredient — RWBC failed *because* it
lacked one (R7), and the ones-mask "worked" only because it was an *implicit* KL. The anchor makes that
explicit and principled. (Its cousin, option 3's **zero-init additive modulator**, is the even-stronger
*by-construction* version: freeze resampler+adapter+null_plan so the duck pathway is bit-identical to
base, train only the zero-init `adaln_proj` advance side-path — already built, planner.py:305-309,
nitrogen.py:673/833 — duck literally cannot collapse because its weights never move. Trades some
plan-richness for a hard guarantee.)

**What flips me to the anchor (decisive, measured tonight):** (i) if P1A0 restores DUCK but the Q4
battery shows the recovery is **narrow** — only the 4 labeled actions move and contrast #5 (CLIMB/UP,
unlabeled) stays at R<0.3 — that's the taxonomy treadmill, and the taxonomy-free anchor wins. (ii) If the
per-chunk labeler induces a **frame→action shortcut** (explicit duck plan fails to raise DOWN on a
*non-duck* frame in the battery) — the anchor, which never co-labels, avoids it. (iii) If P0A1 matches
P1A0 on duck recovery AND retention with strictly less machinery — prefer it for simplicity. The crossed
P1A0‖P0A1 design tonight measures exactly (i)–(iii), so we don't have to guess: the data picks the winner
by morning.

---

## Runnable (tonight, 4×A6000)

```bash
RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
# --- P1A0: richer per-chunk situational plans (Q3 diagnostic == Q2 fix-1), GPU0-1, 3 seeds ---
#   add --situational-plans to demo_bc.py: in load_demo_chunks, plan = SIT[label_chunk(chunk)]
for s in 0 1 2; do $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/demo_bc.py \
  --game smw --train plan_head --situational-plans --steps 600 --seed-offset $s \
  --save-delta files/r9_sitplan_s$s.pt; done
# --- P0A1: KL-anchor, terse plan, GPU2-3, 3 seeds (parallel) ---
#   add --kl-anchor 0.3 to demo_bc.py: frozen plan_head deep-copy g0; +lambda*MSE(v_theta,v_g0)
#     over (demo-frame x SIT-plan) anchor batch at matched (x_t,t,seed)
for s in 0 1 2; do $RUN CUDA_VISIBLE_DEVICES=2 .venv/bin/python planner_poc/demo_bc.py \
  --game smw --train plan_head --use-correct-plan --kl-anchor 0.3 --steps 600 --seed-offset $s \
  --save-delta files/r9_klanchor_s$s.pt; done
# --- P0A0 control (reproduce 0% collapse), 1 seed ---
$RUN CUDA_VISIBLE_DEVICES=1 .venv/bin/python planner_poc/demo_bc.py \
  --game smw --train plan_head --use-correct-plan --steps 600 --seed-offset 0 --save-delta files/r9_ctrl.pt
# --- Expressiveness battery (Q4): base + each delta, matched seeds, per-action rates ---
#   new planner_poc/expressiveness_battery.py reusing plan_graded_test _sample_chunk harness
for d in BASE r9_sitplan_s0 r9_sitplan_s1 r9_sitplan_s2 r9_klanchor_s0 r9_ctrl; do \
  $RUN CUDA_VISIBLE_DEVICES=3 .venv/bin/python planner_poc/expressiveness_battery.py \
  --game smw --delta files/$d.pt --states 4 --chunks 12; done
# Headlines: DOWN-on-command(press-down) >=5% & monotone; SMW Δ_plan >=+20 (CI>0, 3/3); null max|diff|=0.0;
#            battery sign-match >=4/5 & R>=0.5; duck contrast mandatory >=5%.
```

**Falsifier recap:** P1A0 press-down DOWN < 2% ⇒ not DATA ⇒ ship P0A1 if it passes, else escalate to the
zero-init modulator (option 3, already built). Any cell that loses Δ_plan (POST < +20) fails retention.

## Grounding (repo)
demo_bc.py:76 `load_demo_chunks` (attach per-chunk plan here), :156 `build_batch` reads `s["plan"]`
per-sample (per-chunk plans need NO batch change), :166 `plan_dropped=False` (null untrained ⇒
null-invariance), :156 `--residual-mask` supervised target-weight (rare-action amplifier). planner.py:32
`num_plan_tokens=8` (base does duck@K=8 ⇒ option 4 dead), :47 + :305-309 `adaln_proj` zero-init null-masked
modulator (option 3, already built), :301 `null_plan` (masked-null path). nitrogen.py:282 `dit_temb_dim`,
:673/:833 `adaln_cond` applied (additive). plan_graded_test.py BATTERY (fixed-plan harness to extend for
the battery). pooled_demofit.py (swap to per-chunk labels for the Q5 generalist). Results: DUCK-ON-COMMAND
§ (base 1.2→10.8%, pooled 0.0%), STALENESS § (fit learns to ignore plan variation = same mechanism).
Constraints honored: ONE generalist (per-chunk labels, not per-game weights), frozen-VLM-first (Tier-0
labels need no VLM), exact null-invariance (plan-head-only + masked null), additive/recoverable deltas,
GPU-frugal (600-step fits), no commits.
