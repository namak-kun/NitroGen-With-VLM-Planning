# IDM Gate Findings

## Verdict

**FAIL.** The SMW held-out-demo IDM shows some coarse visual/action signal, but it does not meet the gate: B/jump F1=0.372, LEFT F1=0.291, RIGHT F1=0.584 (<0.6), and Y/run F1=0.791 is below a constant-on/validation-majority baseline (0.810). RIGHT is essentially tied with constant-on (0.580), and B/jump is worse than constant-on (0.385). This is not strong enough to justify YouTube pseudo-labeling yet.

## Exact setup

- Script: `planner_poc/idm_gate.py`
- Command: `CUDA_VISIBLE_DEVICES=2 .venv/bin/python planner_poc/idm_gate.py --epochs 35 --batch-size 256 --workers 2 --out-dir docs/idm_gate --holdout 20260627-110553`
- Model: TinyIDM small CNN, stacked non-causal grayscale frames, K=5 (t-2..t+2), resized to 84x84, width=32, BCE-with-logits, train-tuned per-button thresholds.
- Data: `SuperMarioWorld-Snes`; held out entire demo `20260627-110553` (7456 center frames). Train demos: 20260627-105913, 20260627-105939, 20260627-110315, 20260627-110517, 20260627-110822, 20260627-111035 (28468 sampled center frames). The all-zero/too-short SMW demo was excluded from training by `--min-demo-actions 300` and non-idle filtering.
- Artifacts: model `docs/idm_gate/idm_SuperMarioWorld-Snes_20260627-110553.pt`, metrics `docs/idm_gate/metrics_SuperMarioWorld-Snes_20260627-110553.json`, trace CSV `docs/idm_gate/trace_SuperMarioWorld-Snes_20260627-110553.csv`. Best checkpoint was epoch 1; later epochs overfit and reduced held-out F1.

## Held-out per-button F1 vs baselines

| idx | button | val prevalence | IDM F1 | precision | recall | no-op F1 | train-majority F1 | constant-on F1 | val-majority F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | B **critical** | 0.239 | 0.372 | 0.240 | 0.828 | 0.000 | 0.000 | 0.385 | 0.000 |
| 1 | Y **critical** | 0.681 | 0.791 | 0.694 | 0.919 | 0.000 | 0.000 | 0.810 | 0.810 |
| 2 | SELECT | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | START | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | UP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | DOWN | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.013 | 0.000 |
| 6 | LEFT **critical** | 0.147 | 0.291 | 0.230 | 0.398 | 0.000 | 0.000 | 0.256 | 0.000 |
| 7 | RIGHT **critical** | 0.409 | 0.584 | 0.445 | 0.848 | 0.000 | 0.000 | 0.580 | 0.000 |
| 8 | A | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 9 | X | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 |
| 10 | L | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 11 | R | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Critical-button mean F1 = 0.509. The deployable train-majority baseline is all-zero for the critical buttons because the training split has each button under 50% prevalence except RIGHT at 49.7%. Constant-on is included as a sanity baseline for held buttons; the IDM barely beats it on RIGHT, loses to it on B and Y, and only modestly beats it on LEFT.

## Temporal trace inspection

Trace inspected: frames 1131-1490 of the held-out demo (`docs/idm_gate/trace_SuperMarioWorld-Snes_20260627-110553.csv`). In this slice Y/run is truly held for all 360 frames; the IDM predicts it for 281/360 frames, missing many early frames but recovering later. LEFT is truly one long 165-frame hold; predictions cover the middle of it but are fragmented and leak after release. RIGHT has true segments at 0-19, 195-234, and 251-359; the IDM catches the late long rightward segment but overextends it from 198-359. B/jump is six bursts (14-27 frames each); the IDM predicts 292/360 positives versus 129 true positives, so it is mostly over-calling jump rather than detecting press timing.

30-frame bin summary from the trace showed coarse left/right phase awareness (left bins around 1191-1310 and right bins after 1341), but not button-accurate timing. This fails the “sensible temporal traces” requirement for pseudo-labeling.

## Reasoning against the gate criterion

- Required: materially above baseline and ideally >0.6 on dominant direction+jump buttons.
- Observed: RIGHT is close but still sub-threshold (0.584); LEFT and B/jump are far below useful levels; Y/run is not convincing because constant-on on this holdout scores 0.810.
- The model learns demo/style priors and coarse motion phases, not reliable frame-level button inversion on an unseen playthrough.

## Most likely reason and what would be needed

Most likely reason: 6 non-idle training demos are too few and distribution-shifted across playthroughs; many actions are held/autorepeated and visually ambiguous at 84x84 over only 5 frames, so the tiny CNN memorizes trajectory context and overpredicts common buttons. Needed before revisiting YouTube: collect substantially more ROM-matched human demos, evaluate leave-one-demo-out, add stronger temporal models/augmentations (RGB or frame-delta stacks, longer windows, video CNN/transformer), and require >0.6-0.7 F1 on jump/left/right across multiple held-out demos before pseudo-labeling YouTube.
