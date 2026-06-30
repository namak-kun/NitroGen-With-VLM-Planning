# SMW Verification — RAM-independent (VLM-watched) evidence: trained models vs base

**Generated:** 2026-06-29 (overnight grind subagent)
**Game:** Super Mario World (SMW), 8 demo start states (0–7).
**Policy rollouts:** `Qwen/Qwen3.5-2B` planner + 600M flow-matching DiT. **VLM judge:** `google/gemma-4-12B-it` (watches frames only, RAM-free).
**Rollout cap:** 90 s game-time each, closed-loop, stop on death/level-reset. 32 rollouts (4 models × 8 states), all on GPUs 0–1.
**Models:** `base` = `ckpts/btn_s600_full.pt` (no delta) · `pooled` = pooled generalist · `kl` = R9 KL-anchor demo-fit · `situ` = R9 situational demo-fit.
**Artifacts:** mp4 + frames + RAM json + VLM verdict per cell in `docs/furthest/smw/`. Raw judge data: `files/judge_summary.json`.

---

## TL;DR — the honest bottom line

1. **No model actually plays SMW well.** RAM-free VLM progress ratings are **0–4 / 10 across the board**; every model either dies or gets stuck in the first section of every level. None completes a level.
2. **Trained models are only MODESTLY / PARTIALLY better than base — not "significantly" better.** In order-controlled head-to-head VLM comparisons, a trained model beats base on **3 / 8** starts (states **1, 3, 7**), base beats the best trained model on **1 / 8** (state **6**), and the rest are ties or position-bias-ambiguous. The single highest legitimate VLM progress rating of all 32 runs belongs to **base** (state 6 = 4/10); no trained run exceeds 3/10.
3. **RAM `screen_x` OVERSTATES the trained-vs-base gap — the owner's worry is confirmed.** The big trained-model `screen_x` numbers (e.g. `pooled_state1` +379, `kl_state7` +946) are largely **inflated artifacts** (pipe-warps, underground-level scaling) and in one case RAM **missed a death the VLM caught** (`kl_state7`: RAM `died=False`/survived 90 s, but the VLM sees Mario die to a fireball at ~70 s). If you trusted RAM alone you'd over-credit the trained models.
4. **State 2 is a degenerate start** — it is a static **"Course Clear!" results screen**, not gameplay. All four models get a bogus VLM 10/10 there. **Exclude state 2** from any comparison.
5. **Where trained genuinely helps:** on starts where **base instantly dies** (state 1: base dies at 3.5 s; state 7: base falls in lava at 13.6 s), the trained models survive longer and reach later areas — a real, VLM-confirmed recovery. The cleanest single demo is **`situ_state3`** (furthest of all runs *and* VLM-confirmed real landmarks *and* robust head-to-head win over base).

---

## Master table (state × model)

`RAM x` = `screen_x` gained (engine var, scale differs per level). `surv` = survived seconds. `died` = RAM death flag.
`VLM` = Gemma rating 0–10 from frames only. `R/V` = RAM-vs-VLM agreement: ✓ agree · ~ partial · ✗ disagree (RAM sketchy).

| State | Model | RAM x | surv (s) | died | VLM | R/V | VLM outcome (RAM-free, one-line) |
|---|---|---|---|---|---|---|---|
| 0 | base   | 0    | 3.5  | Y | 0  | ✓ | falls off bottom, dies ~1.8 s |
| 0 | pooled | 8    | 3.4  | Y | 0  | ✓ | falls off bottom, idle ~1.0 s |
| 0 | **kl** | 47   | 5.5  | Y | 1  | ✓ | stuck/erratic into ground/bush |
| 0 | situ   | 8    | 3.4  | Y | 1  | ✓ | loops jump-onto-pillar-and-fall |
| 1 | base   | 0    | 6.3  | Y | 0  | ✓ | dies in a pit at 3.5 s |
| 1 | **pooled** | 379 | 8.4 | Y | 1 | ✗ | **RAM +379 but** swallowed by a pipe, dies 4.7 s |
| 1 | kl     | 69   | 4.2  | Y | 1  | ✓ | stuck after Koopa collision |
| 1 | situ   | 69   | 4.2  | Y | 1  | ✓ | loops jump-and-land near a bush |
| 2 | base   | 67   | 2.9  | Y | 10 | ✗ | **"Course Clear!" results screen** (degenerate) |
| 2 | pooled | 67   | 2.9  | Y | 10 | ✗ | **"Course Clear!" results screen** (degenerate) |
| 2 | kl     | 67   | 2.9  | Y | 10 | ✗ | **"Course Clear!" results screen** (degenerate) |
| 2 | situ   | 67   | 2.9  | Y | 10 | ✗ | **"Course Clear!" results screen** (degenerate) |
| 3 | base   | 449  | 48.1 | Y | 2  | ~ | small progress; "still active at end" (RAM says died) |
| 3 | pooled | 1270 | 23.0 | Y | 3  | ✓ | progresses, dies to a bird enemy 19 s |
| 3 | kl     | 374  | 23.2 | Y | 2  | ✓ | killed by enemy ~19.2 s |
| 3 | **situ** | 1271 | 23.9 | Y | 3 | ✓ | **furthest run:** jungle, Lakitus, ?-block, coin; hit 23.9 s |
| 4 | base   | 320  | 22.0 | Y | 1  | ✓ | stuck in a vertical corridor |
| 4 | pooled | 304  | 20.7 | Y | 1  | ✓ | stuck near start |
| 4 | kl     | 298  | 13.2 | Y | 1  | ✓ | idle under a platform 10.2 s |
| 4 | **situ** | 320 | 12.2 | Y | 2 | ✓ | stuck at platform edge 8.5 s |
| 5 | base   | 321  | 19.0 | Y | 2  | ✓ | stuck at bottom of a pit 14.8 s |
| 5 | pooled | 341  | 12.7 | Y | 2  | ✓ | stuck at bottom of a pit 12.7 s |
| 5 | **kl** | 342  | 9.2  | Y | 3  | ✓ | stuck in a vertical pipe/tunnel 4.8 s |
| 5 | situ   | 498  | **90.0** | **N** | 2 | ~ | **survives 90 s but STUCK** oscillating on a block (no progress) |
| 6 | **base** | 471 | 15.9 | Y | **4** | ✓ | **best legit rating:** reaches water section, then stuck 13.1 s |
| 6 | pooled | 370  | 13.0 | Y | 2  | ✓ | falls in water, dies 7.3 s |
| 6 | kl     | 470  | 21.5 | Y | 2  | ~ | falls in water ~16.8 s (RAM x high, VLM low) |
| 6 | situ   | 406  | 14.2 | Y | 3  | ✓ | falls in water, dies 7.4 s |
| 7 | base   | 220  | 19.7 | Y | 1  | ✓ | falls into lava 13.6 s |
| 7 | pooled | 213  | 6.9  | Y | 0  | ✓ | falls into lava 3.0 s |
| 7 | **kl** | 946  | **90.0** | **N** | 2 | ✗ | **RAM +946 & "survived" but** VLM sees death to fireball ~70 s; bg static |
| 7 | situ   | 863  | 15.7 | Y | 2  | ✗ | **RAM +863 but** falls in lava 12.2 s (RAM overstates) |

**Per-state RAM leader vs per-state VLM leader** (note they often disagree):

| State | RAM leader (screen_x) | VLM leader (rating) | Area (from VLM) |
|---|---|---|---|
| 0 | kl (+47) | kl/situ (1) | start, all die instantly |
| 1 | pooled (+379) | tie trained (1) | level start; base dies instantly |
| 2 | tie (+67) | tie (10, bogus) | **Course Clear screen (degenerate)** |
| 3 | situ (+1271) | pooled/situ (3) | jungle |
| 4 | base/situ (+320) | situ (2) | overworld/vertical |
| 5 | situ (+498, 90 s) | kl (3) | hills/pit |
| 6 | base (+471) | **base (4)** | Yoshi/water |
| 7 | kl (+946, 90 s) | kl/situ (2) | underground lava cavern |

---

## Head-to-head: base vs best-trained per state (VLM compare, BOTH A/B orders)

Each comparison was run twice with the runs swapped (A↔B) to control for the VLM's well-known **position bias**. "Consensus" only counts when both orders agree on the *content* winner.

| State | Best trained | Order 1 (base=A) | Order 2 (swapped) | Consensus | Read |
|---|---|---|---|---|---|
| 0 | kl     | tie  | tie  | **tie** | both fail at the hard start |
| 1 | pooled | **pooled** | **pooled** | **pooled wins** | base killed by Koopa immediately; pooled navigates a section first |
| 2 | pooled | tie  | tie  | **tie** | degenerate results screen — ignore |
| 3 | situ   | **situ** | **situ** | **situ wins** | situ "progresses significantly further right… void/gap, coin, new area"; base "stuck in a loop" |
| 4 | situ   | tie  | situ | **ambiguous** | order-dependent; ~tie |
| 5 | kl     | base | kl   | **ambiguous (pure position bias)** | VLM picked "RUN A" in *both* orders → A-bias artifact, inconclusive |
| 6 | situ   | base | base | **base wins** | trained drowns early (1–6 s); base reaches mid-level between pillars |
| 7 | kl     | **kl** | **kl** | **kl wins** | base falls in lava 15 s; kl "moves past the lava pit, reaches ?-block & new pillars" |

**Score:** trained wins **3** (states 1, 3, 7) · base wins **1** (state 6) · ties **2** (0, 2) · ambiguous **2** (4, 5).
Trained beats base on more starts than the reverse (3–1), but it is **not a clean sweep**, and state 5's "win" for either side is a pure position-bias artifact (the judge picked whichever run was shown first).

---

## RAM-vs-VLM agreement analysis (the trust question)

**Agreement is high on FAILURE (✓):** when a model dies fast or gets stuck, RAM (low `screen_x`, short survival) and the VLM (low rating, "stuck/died") agree. ~20/32 cells agree cleanly.

**The disagreements (✗) all point the SAME way — RAM OVERSTATES progress for the trained models:**

| Cell | RAM says | VLM (watching) says | Why RAM is sketchy |
|---|---|---|---|
| `pooled_state1` | +379 "progress" | dies in a pipe at 4.7 s, rating 1 | **pipe-warp** bumps `screen_x` without real forward play |
| `kl_state7` | +946, **survived 90 s, died=False** | **dies to a fireball ~70 s**, "background largely static", rating 2 | **underground-level `screen_x` scaling** inflates the number; **death heuristic missed the death** |
| `situ_state7` | +863 | falls in lava at 12.2 s, rating 2 | same underground inflation; died early |
| `state2` (all 4) | +67, died 2.9 s | static **"Course Clear!"** screen, bogus 10 | start state is a results screen, not gameplay |
| `situ_state5` (~) | +498, **survived 90 s** | **stuck** oscillating on a block, no progress, rating 2 | "survived" ≠ skill; it never died because it idles against a wall |
| `kl_state6`, `base_state3` (~) | high `screen_x` / long survival | modest visual progress / death-flag mismatch | screen_x ≠ visual advance |

**Takeaway:** RAM `screen_x` is **directionally useful but unreliable in magnitude**, and it **missed at least one death** (`kl_state7`). The cases where RAM most flatters the trained models (states 1, 7) are exactly the cases the VLM downgrades. **Trust the VLM for "how far did it really get."** The owner's instinct that RAM signals can be sketchy is **confirmed and material** — RAM alone would have reported a much larger trained-vs-base advantage than actually exists.

---

## BEST DEMOS (mp4 paths + RAM-free verdicts)

### ★ SINGLE BEST DEMO — `situ`, state 3 → furthest + both signals agree it beats base
- **mp4:** `docs/furthest/smw/situ__state3.mp4`
- **RAM:** `screen_x` **+1271** (largest of all 32 runs), survived 23.9 s.
- **VLM describe (rating 3):** *"moved forward through a jungle area, passing a small hill, a group of Lakitus/enemies, a ?-block, and a small platform with a coin… died after being hit by an enemy at 23.9 s."*
- **VLM head-to-head vs base (both orders → situ):** *"situ progresses significantly further to the right, navigating a large gap, collecting a coin, reaching a different area; base is stuck in a loop, moving back and forth without meaningful progress."*
- **Why best:** the only run that is simultaneously (a) furthest by RAM, (b) highest-tier VLM progress with named real landmarks, and (c) a robust, order-independent head-to-head win over base. Compare baseline: `docs/furthest/smw/base__state3.mp4`.

### ★ BEST "recovers where base dies" demo — `kl`, state 7 (underground lava)
- **mp4:** `docs/furthest/smw/kl__state7.mp4` (baseline `docs/furthest/smw/base__state7.mp4`)
- **VLM head-to-head (both orders → kl):** *"base falls into the lava at 15 s and the run ends; kl navigates the platforming, moves past the lava pit, reaches a ?-block and new pillars — significant progression beyond where base failed."*
- **Caveat (honesty):** RAM over-credits this run (+946, claims survived 90 s) but the **VLM sees kl die to a fireball ~70 s**. Still a clear, order-robust win over base; just don't quote the +946/"survived" RAM figures.

### ★ BEST "recovers where base instantly dies" demo — `pooled`, state 1
- **mp4:** `docs/furthest/smw/pooled__state1.mp4` (baseline `docs/furthest/smw/base__state1.mp4`)
- **VLM head-to-head (both orders → pooled):** base is *"killed by a Koopa almost immediately"*; pooled *"navigates a large section, jumps a slope, collects a coin, survives an enemy encounter."*
- **Caveat:** low absolute progress — pooled itself dies in a pipe at 4.7 s (RAM's +379 is a pipe-warp artifact). The value is the **relative** recovery vs a base that does nothing.

### For contrast — BEST BASE run (trained did NOT beat it): `base`, state 6
- **mp4:** `docs/furthest/smw/base__state6.mp4`
- **VLM describe — rating 4 (highest legitimate rating of all 32 runs):** *"moves from a checkered platform to a green hill, between two large pillars, into a water section with small platforms"* then gets stuck at 13.1 s. Both compare orders favor base over the best trained model here.

---

## Caveats & method notes
- **State 2 excluded** as a degenerate "Course Clear!" results-screen start (bogus 10/10 for every model). State 0 is a near-impossible spot (everyone dies < 6 s) — low signal.
- The 8 demo states span **several different SMW areas** (overworld/hills, jungle, Yoshi/water, underground-lava, + the results screen), not one "level 1-1 start". There is **no clean, well-progressing 1-1-start** among them; the most-progressing genuine gameplay start is **state 3 (jungle)**.
- **VLM position bias is real** (state 5): always running compares in both A/B orders is necessary — single-order compares would have produced false winners.
- **VLM is conservative on magnitude** but reliable on direction/death; **RAM is unreliable on magnitude** and missed a death. Use the VLM as ground truth for "how far," RAM only as a coarse, corroborating signal.
- All numbers reproduce from `docs/furthest/smw/*.json` (RAM) and `*.judge.json` / `files/judge_summary.json` (VLM).
- **Visual proof frames** (rendered & inspected to confirm the two headline claims):
  - `files/verify_base_s2_t0.png` — state 2 is literally a **"COURSE CLEAR!" results screen** (Mario on a black screen, "BONUS! ★×0"), proving the start state is degenerate and the 10/10 ratings are artifacts.
  - `files/verify_situ_s3_t12.png` and `files/verify_situ_s3_t20.png` — `situ_state3` at 12 s (jungle, ?-block, mushroom) vs 20 s (a clearly different section: coin row, fruit, an enemy), with the TIME counter ticking down — confirming **real forward progress**, not a static/looped scene.

## Verdict on the headline question
**"Are the trained models significantly better than base?"** — **No, not *significantly*; they are *modestly and selectively* better.** When judged RAM-free by a VLM watching the video, trained models win 3 of 8 head-to-head starts (vs base's 1), chiefly by **recovering on starts where base dies immediately** (states 1, 7) and **reaching genuinely further on the jungle start** (state 3). But base wins state 6 outright, ties/leads on 4–5, and holds the single best absolute progress rating (4/10). **All models remain weak** (≤ 4/10; none beats a level), and the **large RAM `screen_x` advantages for trained models are substantially inflated** — so the apparent gap shrinks under honest, RAM-independent video inspection. This is consistent with the project's standing conclusion that real long-horizon capability needs environments/RL, not just plan-conditioning alignment.

---

## ★ ORCHESTRATOR CORRECTION (Opus-4.8 direct HUD-counter audit, post-hoc)

I (the orchestrator) re-examined `kl_state7` frame-by-frame using the in-game **HUD TIME counter** as a
death-arbiter, and the report's central `kl_state7` claim is **WRONG in the trained model's favor**:

- **CLAIM (this report, 6 places):** "RAM missed a death the VLM caught — `kl_state7` Mario dies to a fireball
  ~70 s; trust the VLM."
- **FACT (HUD evidence):** across all 24 frames the TIME counter **decreases monotonically with no reset**:
  `300(t0) → 255(31s) → 221(55s) → 203(66s) → 198(70s) → 192(74s) → 180(82s) → 169(90s)`, lives steady at 9,
  scene scrolling continuously start→deep-fortress. In SMW a death **reloads the level and RESETS TIME** — a
  continuous countdown is decisive proof of **no death**. **Mario SURVIVES the full 90 s.** RAM `died=False`
  was **correct**; the **Gemma VLM FALSE-POSITIVED a death** from a nearby lava Podoboo/fireball.

**Why this matters (balances the report's meta-conclusion):** the report concludes "trust the VLM as ground
truth for how far / death." That is **too strong**. The honest lesson is symmetric: **neither signal is
individually reliable** —
- **RAM** misses some deaths and its `screen_x` **magnitude is not cross-level comparable** (underground/warp
  scaling) — the report is right about this.
- **the VLM (Gemma-4-12B) FALSE-POSITIVES deaths**, calling "died" when a hazard is merely *near* the character
  (proven here) — the report missed this failure mode.
- The reliable arbiter is a **game-intrinsic HUD counter (TIME resets on death; lives decrement)** read from
  the pixels, or `survived_advance`'s lives/progress logic — NOT either model's gestalt judgment alone.

**Consequence for the headline:** the VLM-false-positive bias means trained models were in places **UNDER-credited**,
not only over-credited via RAM. Specifically, **`kl_state7` is a GENUINE strong demo**: the trained model
**survives the full 90 s navigating a hard underground lava-fortress** (continuous, real progress) where **base
dies in 13.6 s** — a clean survives-where-base-dies win. Quote it as "survives 90 s, reaches deep into the lava
fortress" (the `screen_x +946` magnitude is still not worth quoting cross-level, that part of the report holds).

**Unchanged / confirmed:** `situ_state3` (the report's #1 pick) is **legit** — I confirmed real jungle progress
(start bushes → ?-block/mushroom platform → berry hills, SCORE 32400→33800, TIME 400→378), ~3× past base; it
genuinely dies at ~24 s (no FP issue, it's a "furthest" claim). The state-2 "COURSE CLEAR!" artifact and the
Gemma position-bias finding are also confirmed-valuable. The modest-overall verdict ("no model beats a level;
trained selectively better") still stands; only the `kl_state7` death-status and the "trust-VLM-as-ground-truth"
framing are corrected.

**Best SMW demos, corrected ranking:** (1) **`kl__state7.mp4`** — survives 90 s in a lava fortress where base
dies at 13.6 s (now the cleanest survives-where-base-dies win, death-FP corrected); (2) **`situ__state3.mp4`** —
furthest absolute (+1271, real jungle landmarks), dies ~24 s. Both VLM-confirmed-direction + HUD-audited.
