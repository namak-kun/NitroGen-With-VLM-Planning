# Example Generated Plans (Stage-2 VLM)

Curated examples from the Stage-2 plan-generation experiments (EXP-038/039/040), showing
the progression from directive → tactical → reaction-aware. Each plan is generated from a
real gameplay window (frames + the player's actual gamepad inputs as grounding context).

**Design target (user principle):** plans are *informative tactical intent*, NOT per-action
directive — no button/stick prescriptions; System 1 (the DiT) handles fast reactions.

---

## 1. Baseline prompt (TOO directive — leaks controls) — Qwen3.5-4B

These were the first attempt; note the button/stick leakage we DON'T want:

| game | plan | real actions (context) |
|---|---|---|
| other | Climb the wall, then **press B** to interact with the enemy | stick up-right; B throughout |
| rocket_league | Hold the ball, **accelerate right, and jump** to maintain possession | RT/accel + A/jump |
| other | Move left and shoot the ball into the open net | stick down-left; LB throughout |
| other | **Hold B** and move up-right to dodge the attack | stick up-right; B throughout |

## 2. Tactical prompt (goal-level, no controls) — Qwen3.5-4B

Same windows, "describe intent, never controls":

| game | plan |
|---|---|
| other | Intercept the enemy and secure the kill |
| rocket_league | Maintain offensive pressure and intercept the ball before it reaches the net |
| rocket_league | Intercept the ball before it reaches the goal |
| other | Intercept the enemy's attack and break its guard |
| other (cornered) | Position behind enemy to avoid frontal attacks |

## 3. Tactical prompt — Gemma-4-12B-it (chosen model: cleanest)

| game | plan | real actions (context) |
|---|---|---|
| rocket_league | Position the car to intercept the ball and defend the goal | mostly neutral |
| rocket_league | Maintain position and prepare to intercept the ball | up-right; RT/accel + A/jump |
| rocket_league | Position the car to intercept the ball and clear the net | neutral |
| other (souls-like) | Maintain distance and dodge the boss's incoming melee attacks | down-right; LB briefly |
| other (souls-like) | Position yourself in the corner to prepare for the next engagement | down-left; LB throughout |
| other (platformer) | Navigate the platforming section and reach the next ledge | B + RT/accel |
| other | Navigate the rocky terrain to reach the next area | up-right |
| other | Observe the surroundings and scan for nearby enemies | neutral (idle) |

## 4. Reaction-aware (plan → outcome → adjustment) — Gemma-4-12B-it

The VLM sees its OWN prior plan + what the player actually did, and adjusts:

**rocket_league sequence:**
1. `Maintain momentum and prepare to challenge for the ball`
2. (player went aerial: RT/accel + A/jump) → `Adjust: land the car and reposition to intercept the ball`
3. (idle) → `Adjust: land the car and reposition to intercept the ball`

**combat (souls-like) sequence:**
1. `Maintain defensive stance and wait for the enemy to commit`
2. (player moved down-right, LB) → `Adjust: circle the enemy and look for an opening to strike`
3. (player moved up-right, B) → `Adjust: keep circling to maintain distance and wait for a clear opening`

---

## Observations

- **Grounding works:** plans track the real actions (down-left+LB → "position in the
  corner"; aerial inputs → "land the car and reposition").
- **Abstraction is right:** the tactical/Gemma plans never prescribe a button or stick.
- **Idle windows** → generic "observe/hold position" (flag/skip these in the dataset).
- **The hard part (alignment):** plans name entities ("the ball", "the boss", "the ledge",
  "the net"). The DiT sees those pixels via SigLIP but has NO language prior — it doesn't
  know "net" means the goal. Whether the plan token can be aligned to the DiT's *visual*
  entity (frame-grounded), without the DiT understanding the word, is the open Stage-2
  alignment question (see EXPERIMENTS.md, morning handoff in plan.md).

Recipe: Gemma-4-12B-it + tactical/reaction prompt; A=4 (2.4s) window; frames +
action-summary-as-context + prior plans + windowed transcript → tactical GOAL plan.
