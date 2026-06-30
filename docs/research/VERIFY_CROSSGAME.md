# Cross-game furthest-rollout verification (RAM + VLM video judge)

Generated: 2026-06-29T16:27:03

Scope: non-SMW games only: `mmx`, `smbas`, `sonic`. Each state has a 90s base rollout (`ckpts/btn_s600_full.pt`, no delta) and a pooled-planfit rollout (`--delta pooled_planfit_s0.pt --mode plan`). Rollouts used GPUs 2/3 only. VLM judging used `QWEN=google/gemma-4-12B-it` and watched `.frames.npz` files, not RAM.

Important caveat: all RAM progress numbers below are the tool metric named `screen_x_reached` in JSON. It is `xpos` for mmx/smbas and `screen_x` for sonic. Treat the VLM visual verdict as the RAM-independent check, especially for Sonic.

## Executive summary

- **MMX:** pooled clearly beats base on state 0 by RAM and VLM. State 1 is low-confidence/contradictory: pairwise VLM says pooled, but individual VLM ratings/outcomes say pooled dies sooner and scores lower.
- **SMBAS:** pooled helps **state 0 only**. Base is better or tied on states 5/10/15/20/25 by VLM; several pooled runs die early despite larger or similar RAM progress.
- **Sonic:** pooled is visually better on states 0 and 3. State 1 is a RAM-vs-VLM disagreement (pooled has higher `screen_x`, VLM says base progressed further). State 2 is also inconsistent (pairwise VLM favors pooled, but RAM and individual describe are both low); do not use it as a best demo.

## Mega Man X (`mmx`)

| state | model | RAM progress | survived_sec | died | VLM rating | VLM outcome | mp4 |
|---:|---|---:|---:|---|---:|---|---|
| 0 | base | 144 | 90.0 | False | 1 | The character is stuck/idle, moving only slightly back and forth in a small area without making forward progress. | `docs/furthest/mmx/base__state0.mp4` |
| 0 | pooled | 457 | 21.9 | True | 3 | The character dies (game over) at approximately t=20.0s. | `docs/furthest/mmx/pooled__state0.mp4` |
| 1 | base | 46 | 17.5 | True | 2 | The character falls off the platform and dies at approximately t=13.7s. | `docs/furthest/mmx/base__state1.mp4` |
| 1 | pooled | 60 | 4.5 | True | 1 | The character falls off the platform and out of view at approximately t=1.3s. | `docs/furthest/mmx/pooled__state1.mp4` |

### Base vs pooled VLM comparisons

| state | RAM winner | VLM winner (24-frame compare) | agreement flag | VLM compare evidence |
|---:|---|---|---|---|
| 0 | pooled (144 vs 457) | pooled | **AGREE** | The provided frames for RUN A and RUN B show the following: **RUN A:** The player character (X) is moving along a highway platform. He encounters a large truck, fires a projectile… |
| 1 | pooled (46 vs 60) | pooled | **AGREE** | WINNER: B | WHY: Run A ends with the character falling into a pit and the screen going black. Run B shows the character successfully navigating a complex platforming section, jump… |

## Super Mario All-Stars / SMB (`smbas`)

| state | model | RAM progress | survived_sec | died | VLM rating | VLM outcome | mp4 |
|---:|---|---:|---:|---|---:|---|---|
| 0 | base | 139 | 8.9 | True | 1 | Mario is killed by a Goomba at approximately t=5.0s. | `docs/furthest/smbas/base__state0.mp4` |
| 0 | pooled | 561 | 12.9 | True | 3 | The character died (fell into a pit or was hit) at t=11.8s, resulting in a black screen. | `docs/furthest/smbas/pooled__state0.mp4` |
| 5 | base | 45 | 13.6 | True | 3 | Mario got stuck/idle on the large pillar at t=12.4s. | `docs/furthest/smbas/base__state5.mp4` |
| 5 | pooled | 46 | 6.1 | True | 1 | Mario died by falling into the water at approximately t=1.1s. | `docs/furthest/smbas/pooled__state5.mp4` |
| 10 | base | 159 | 90.0 | False | 1 | Mario got stuck/idle, repeatedly jumping in place near a lava pit and a fire flower. | `docs/furthest/smbas/base__state10.mp4` |
| 10 | pooled | 136 | 6.9 | True | 2 | Mario gets stuck/idle, hovering in place near a fire flower and a block. | `docs/furthest/smbas/pooled__state10.mp4` |
| 15 | base | 335 | 27.8 | True | 3 | The character died by colliding with a Spiny at [t=22.9s]. | `docs/furthest/smbas/base__state15.mp4` |
| 15 | pooled | 435 | 11.8 | True | 3 | The character gets stuck/idle at the end of the sequence, appearing to be waiting or moving very slowly near a cluster … | `docs/furthest/smbas/pooled__state15.mp4` |
| 20 | base | 209 | 90.0 | False | 2 | Mario got STUCK/idle on a platform near a black enemy at t=82.1s. | `docs/furthest/smbas/base__state20.mp4` |
| 20 | pooled | 355 | 13.5 | True | 3 | Mario got stuck/idle on a platform at t=12.3s as the screen faded. | `docs/furthest/smbas/pooled__state20.mp4` |
| 25 | base | 122 | 73.4 | True | 1 | The character gets stuck/idle, moving back and forth in a small area until the footage ends at t=67.0s. | `docs/furthest/smbas/base__state25.mp4` |
| 25 | pooled | 191 | 5.7 | True | 1 | Mario died by falling into lava at approximately t=3.2s. | `docs/furthest/smbas/pooled__state25.mp4` |

### Base vs pooled VLM comparisons

| state | RAM winner | VLM winner (24-frame compare) | agreement flag | VLM compare evidence |
|---:|---|---|---|---|
| 0 | pooled (139 vs 561) | pooled | **AGREE** | WINNER: B | WHY: In Run A, Mario is killed by a Goomba at the very beginning of the level (around t=3.1s). In Run B, Mario successfully navigates past the first set of blocks, col… |
| 5 | tie (45 vs 46) | tie | **AGREE** | The provided frames for RUN A and RUN B are nearly identical in terms of the level progress shown. In **RUN A**, Mario starts on a small platform, jumps into the air, and then lan… |
| 10 | base (159 vs 136) | base | **AGREE** | WINNER: A | WHY: Run A shows Mario moving through a significant portion of the level, navigating past several pillars and reaching a section with a fire-breathing enemy and a ques… |
| 15 | pooled (335 vs 435) | base | **DISAGREE** | WINNER: A | WHY: Run A shows Mario navigating a castle interior and moving past a large white cloud and green hills. Run B shows Mario in a different area with a green pipe and qu… |
| 20 | pooled (209 vs 355) | base | **DISAGREE** | WINNER: A | WHY: Run A shows Mario navigating a complex area with a large cloud, ice formations, and a brick structure, eventually reaching a section with a moving platform and a … |
| 25 | pooled (122 vs 191) | base | **DISAGREE** | WINNER: A | WHY: In RUN A, Mario successfully navigates a series of jumps and moves significantly to the right, eventually reaching a point where he is interacting with a fire-bre… |

## Sonic (`sonic`)

| state | model | RAM progress | survived_sec | died | VLM rating | VLM outcome | mp4 |
|---:|---|---:|---:|---|---:|---|---|
| 0 | base | 1038 | 29.1 | True | 3 | The character is seen running and jumping through the level without dying or getting stuck within the 29-second window. | `docs/furthest/sonic/base__state0.mp4` |
| 0 | pooled | 10165 | 90.0 | False | 4 | The character is still moving and active at the end of the clip (t=90.0s). | `docs/furthest/sonic/pooled__state0.mp4` |
| 1 | base | 2225 | 90.0 | False | 3 | The character is still moving and active at the final timestamp (t=90.0s), having successfully navigated several obstac… | `docs/furthest/sonic/base__state1.mp4` |
| 1 | pooled | 3398 | 41.3 | True | 4 | Sonic is hit by a Badnik and dies at t=34.1s. | `docs/furthest/sonic/pooled__state1.mp4` |
| 2 | base | 154 | 5.5 | True | 2 | The character is hit by a projectile/enemy and dies at approximately t=4.3s. | `docs/furthest/sonic/base__state2.mp4` |
| 2 | pooled | 147 | 2.8 | True | 1 | The character gets stuck/idle, performing a repetitive jump/spin sequence in the same spot without moving forward. | `docs/furthest/sonic/pooled__state2.mp4` |
| 3 | base | 286 | 90.0 | False | 3 | The character is still moving/active at the end of the clip. | `docs/furthest/sonic/base__state3.mp4` |
| 3 | pooled | 3341 | 31.4 | True | 4 | The character gets STUCK/idle inside a metallic structure at approximately t=31.4s. | `docs/furthest/sonic/pooled__state3.mp4` |

### Base vs pooled VLM comparisons

| state | RAM winner | VLM winner (24-frame compare) | agreement flag | VLM compare evidence |
|---:|---|---|---|---|
| 0 | pooled (1038 vs 10165) | pooled | **AGREE** | WINNER: B | WHY: Run A ends with Sonic in the same general area of the first beach zone, having only moved a short distance from the starting point. Run B progresses significantly… |
| 1 | pooled (2225 vs 3398) | base | **DISAGREE (Sonic screen_x/camera caveat)** | WINNER: A | WHY: Run A successfully navigates through a complex section involving a waterfall and a series of rings, eventually reaching a new area with palm trees and a different… |
| 2 | tie (154 vs 147) | pooled | **DISAGREE (Sonic screen_x/camera caveat)** | The provided frames show two different gameplay attempts in a tropical-themed level of Sonic the Hedgehog 2. In **RUN A**, Sonic is seen interacting with a "Special Stage" (indica… |
| 3 | pooled (286 vs 3341) | pooled | **AGREE** | WINNER: B | WHY: In RUN A, Sonic remains in the same general area of the level for the entire duration, moving back and forth between a few platforms and a small loop. In RUN B, S… |

## RAM-vs-VLM disagreement notes

- **MMX state 1:** RAM slightly favors pooled (+60 vs +46) and pairwise VLM favors pooled, but individual VLM ratings are base=2, pooled=1 and pooled dies at 4.5s. Mark as not trustworthy despite compare winner.
- **SMBAS states 15/20/25:** RAM progress favors pooled, but VLM compare says base reached more meaningful landmarks and survived longer. Trust VLM/base for demo selection; pooled dies quickly.
- **SMBAS state 5:** RAM progress is effectively tied (45 vs 46). The 24-frame VLM compare calls a tie; the 12-frame compare favored base. Treat as no pooled improvement.
- **Sonic state 1:** RAM `screen_x` favors pooled (3398 vs 2225), but VLM says base progressed further while pooled looped/fell around waterfall/pit. This is exactly the Sonic camera/metric caveat: do not claim pooled wins from RAM alone.
- **Sonic state 2:** RAM slightly favors base and individual describe ratings are base=2, pooled=1, but pairwise VLM compare favors pooled. This is VLM/RAM/self-consistency conflict; exclude from best demos.

## Best demos (RAM + VLM-supported, honest shortlist)

| rank | game/state/model | why trustworthy | mp4 |
|---:|---|---|---|
| 1 | Sonic state 0 pooled | RAM huge advantage (+10165 vs +1038), survived full 90s, VLM compare winner B (24-frame: B_progress 8 vs A 2). | `docs/furthest/sonic/pooled__state0.mp4` |
| 2 | Sonic state 3 pooled | RAM advantage (+3341 vs +286), VLM compare winner B; reaches rails/loop/industrial structures not seen in base. Dies at 31.4s but clearly further. | `docs/furthest/sonic/pooled__state3.mp4` |
| 3 | MMX state 0 pooled | RAM advantage (+457 vs +144), VLM compare winner B; reaches multi-lane highway/green structure/hover-pod area beyond base. Dies at 21.9s but clearly further. | `docs/furthest/mmx/pooled__state0.mp4` |
| 4 | SMBAS state 0 pooled | RAM advantage (+561 vs +139), VLM compare winner B; passes early blocks/pipes/Goomba where base dies quickly. Dies at 12.9s but reaches further landmarks. | `docs/furthest/smbas/pooled__state0.mp4` |

## Not-better-than-base cases

- SMBAS states 5, 10, 15, 20, 25: pooled is not visually better than base; usually worse or tie.
- Sonic state 1: pooled has larger RAM `screen_x` but VLM says base is better; do not cite pooled as an improvement.
- Sonic state 2 and MMX state 1: conflicting evidence; not suitable for claims.

## Artifacts

- Rollout logs: `docs/furthest/_logs/rollout_*.log`
- Judge logs: `docs/furthest/_logs/judge_*.log`, `compare_*.log`, `compare24_*.log`
- Per-rollout metadata: `docs/furthest/<game>/<model>__state<N>.json`
- Per-rollout VLM describe: `docs/furthest/<game>/<model>__state<N>.judge.json`
- Pairwise VLM compare: `docs/furthest/<game>/compare24_base_vs_pooled_state<N>.judge.json` (24-frame second pass used for final compare table)
