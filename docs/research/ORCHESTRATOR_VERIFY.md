# ORCHESTRATOR DIRECT-VISION VERIFICATION (Opus-4.8, RAM-independent spot-checks)

**Why this file:** the owner asked us to VLM-verify the rollout videos because RAM can be sketchy and they
can't watch them. The Task-2 grinders run the automated pipeline (RAM + local Gemma-4-12B judge + A/B compare).
THIS note is a **third, higher-trust line**: I (the orchestrator, Opus-4.8) directly viewed sampled frames from
representative rollouts and cross-checked them against the RAM `.json`. It exists to validate that the automated
pipeline's conclusions are sound and that RAM is (or isn't) trustworthy per game.

Method: `np.load(<tag>__state<N>.frames.npz)` (24 evenly-sampled frames over the whole rollout) → dumped
frames {0, n/3, 2n/3, n-1} → viewed directly. Compared the visible game-state to the RAM screen_x/survived/died.

## Spot-check 1 — MMX state0, base vs pooled (plan) — RAM TRUSTWORTHY ✓
| run | RAM | what I SAW (direct frames) |
|---|---|---|
| base | x=144, survived 90s, alive | f008: X near the START firing its buster at the hanging structure; barely moved. f023: crawled to the lamppost section, **still alive**. = modest progress, survives. |
| pooled (plan) | x=457, died 21.9s | f008: X well past the start, beside a large blue cannon-enemy. f016: **X on the elevated highway riding beside the big truck** — a recognizable DEEP section of the MMX intro stage, far past anything base reached. f023: blank-white **death/reset flash**. = much further, then dies. |
**Verdict:** RAM ordering (457 ≫ 144) is **visually corroborated** — the plan-conditioned model genuinely
reaches a later, recognizable stage section. The plan model is markedly more progress-seeking but **trades
survival** (dies at 21.9s). Death-awareness in the eval is load-bearing. RAM is trustworthy on MMX (xpos is
cleanly monotone; matches the visible scroll).

## Spot-check 2 — SMW state7, base vs KL-anchor — RAM TRUSTWORTHY ✓, STRONG DEMO
| run | RAM | what I SAW (direct frames) |
|---|---|---|
| base | x=220, died 19.7s | f000: Mario at the entrance of a fortress level (pillars over lava). f023: scene **barely scrolled**, Mario still near the entrance, TIME 300→275 (~25 ticks). = stuck, dies fast. |
| KL-anchor | x=946, **survived 90s (no death)** | f008: Mario **deep over the lava pits**, climbing the pink-pipe platform structure, dodging spinning enemies, TIME 255. f023: **deeper still**, different platform layout, TIME 300→169 (~131 ticks elapsed). = navigates a hard lava-fortress level and **survives the full duration**. |
**Verdict:** the KL-anchor demo-fit model goes **~4× further (946 vs 220) AND survives the full 90s where base
dies in 19.7s**, on the *same* start state. Visually unambiguous; RAM is trustworthy (the TIME-tick delta and
visible scroll both track screen_x). **This is the best presentable single demo so far** (a clean,
same-start, survives-where-base-dies win on a hard level). Files: `docs/furthest/smw/{base,kl}__state7.{mp4,
frames.npz,json}`.
> **NOTE (important, added post-hoc):** the Task-2 SMW grinder's Gemma VLM judge **disagreed** here — it claimed
> Mario "dies to a fireball ~70 s." I re-audited all 24 frames via the **in-game HUD TIME counter**: it counts
> down monotonically `300→255→221→203→198→192→180→169` with **no reset** and lives steady at 9. In SMW a death
> reloads the level and **resets TIME**, so the continuous countdown is decisive: **Mario does NOT die; RAM
> `died=False` is correct; the VLM FALSE-POSITIVED the death** (saw a lava Podoboo near Mario). My original
> survival call stands. This is the night's key verification lesson — see conclusion #4. (Correction also
> written into `VERIFY_SMW.md`.)

## Spot-check 3 — Sonic state0, base vs pooled (plan) — RAM TRUSTWORTHY ✓ (camera-independent), HEADLINE DEMO
This is the cross-game grinder's #1 best demo AND the one place RAM is most suspect (Sonic screen_x is
camera-inflated). I blessed it directly:
| run | RAM | what I SAW (direct frames) |
|---|---|---|
| base | screen_x=1038, died 29s | f000: Sonic at the Emerald Hill Zone START (TIME 0:00, RINGS 0). f023: barely past the start by the palm trees, **TIME 0:29, RINGS 4**, then dies. = minimal progress. |
| pooled (plan) | screen_x=10165, survived 90s | f008: Sonic at the **waterfall/checkered-wall section** (TIME 0:32, RINGS 9, Tails trailing) — distinctly later EHZ. f023: **deep in the zone on loops/checkered terrain, TIME 1:30 (full 90s, survived), RINGS 40.** |
**Verdict:** dramatic, unambiguous win. CRUCIAL: the **RINGS counter (40 vs 4) and TIME-survived (1:30 vs death
at 0:29) are camera-INDEPENDENT**, so the Sonic screen_x camera caveat does NOT undermine this demo — it is
corroborated by signals that don't depend on the scroll metric. All three lines agree (RAM 10165≫1038; Gemma
B_progress 8 vs 2; my direct vision). **This is the single strongest headline demo across all games.** Files:
`docs/furthest/sonic/{base,pooled}__state0.{mp4,frames.npz,json}`. (NB: the Sonic camera caveat IS real on OTHER
states — e.g. state1 RAM favors pooled but VLM+I would trust base; see VERIFY_CROSSGAME.md disagreement notes.)

## Cross-cutting conclusions (orchestrator)
1. **RAM is directionally trustworthy on the side-scrollers (MMX xpos, SMW screen_x) for ORDERING, but its
   MAGNITUDE is not cross-level comparable** (SMW underground/warp scaling inflates screen_x; the SMW grinder is
   right about this). Direct vision matched the RAM *ordering* in every pair I checked. Sonic screen_x is
   separately camera-inflated (use ring/time counters there).
2. **The trained models demonstrably out-progress base**, often dramatically (MMX pooled 3×, SMW KL 4×, Sonic
   pooled ~10× by rings), but the far-reaching runs frequently **die** — so the honest headline is "further +
   often riskier," and the standouts are the runs that are **both** furthest and survive (SMW kl state7, Sonic
   pooled state0).
3. This direct-vision line broadly **agrees with the automated pipeline** on the WINS, so the grinders'
   `VERIFY_SMW.md` / `VERIFY_CROSSGAME.md` matrices are trustworthy as the bulk evidence — **with one correction
   (#4).**
4. **★ NEITHER RAM NOR THE VLM IS INDIVIDUALLY RELIABLE — the night's key verification lesson.** The owner
   worried RAM is sketchy; that's true (misses some deaths, non-comparable magnitude). **But the local VLM judge
   (Gemma-4-12B) ALSO errs in the opposite direction: it FALSE-POSITIVES deaths**, calling "died" when a hazard
   is merely near the character. Proven on `kl_state7`: the VLM said "dies to a fireball ~70 s," but the in-game
   **HUD TIME counter counts down continuously with no reset** (a SMW death resets TIME) ⇒ Mario **survived**;
   RAM was right, the VLM was wrong. **Implication:** the VLM-false-positive can make us UNDER-credit trained
   models on hazard-dense states, just as RAM can OVER-credit via magnitude. The reliable arbiter is a
   **game-intrinsic HUD counter (TIME/lives/rings) read from the pixels**, cross-checked against BOTH RAM and the
   VLM — not either model's gestalt alone. (Future: add a tiny HUD-OCR death/again check to the judge pipeline.)

(See the grinders' `VERIFY_SMW.md` — incl. my appended ★ORCHESTRATOR CORRECTION — and `VERIFY_CROSSGAME.md` for
the full state×model matrix + per-rollout Gemma judgments + base-vs-model A/B compares.)
