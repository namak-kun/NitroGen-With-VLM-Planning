# `planner_poc/` — script index

This directory is the research workbench: behavioral evals, probes, demo-fit trainers, the record server, and
data tooling. It accreted ~160 scripts over the project; this index categorizes the **current** ones and the
one-off probes/tests have been moved to [`attic/`](attic/) (still in git — `git mv attic/<x>.py .` to restore).

> **Conventions:** scripts run from the repo root with `PYTHONPATH=.:planner_poc` and `QWEN=<hf-id>`. They
> import each other by bare module name (e.g. `from eval_policy import NitroGenPolicy`). Trained artifacts are
> saved as "deltas" (the trainable tensors only) under the ephemeral session `files/` dir. **No script commits.**

## ⭐ Core pipeline (load-bearing — imported by many others)
| script | what |
|---|---|
| `eval_policy.py` | `NitroGenPolicy` — load base ckpt + Qwen, sample action chunks, plan/null. The eval workhorse. |
| `eval_common.py` | `survival_advance()` death-aware advance metric; `_lives`, `_step_done_info`. |
| `plan_graded_test.py` | `BATTERY` — the per-game `correct`/`hand_good`/`bad` text plans + graded-plan eval. |
| `rwbc_actor_adapt.py` | `make_env`, `build_batch` + RWBC actor-adaptation (the actor-OOD lever, e.g. Sonic). |
| `run_poc.py` | env factory + `list_envs()` + `BROKEN_ENVS`. |
| `game_planner.py`, `play_annotated_horizon.py` | closed-loop planner driving + prompt plumbing. |
| `demo_train_stack.py`, `action_summary.py`, `vtt_align.py` | gold-triple bootstrap + action/text utilities. |

## 🎯 Demo-fit & one-generalist training (Stage-3)
| script | what |
|---|---|
| `demo_bc.py` | **the plan-OOD demo-fit trainer** (`kl` / `situ` variants); `--kl-anchor`, `--situational-plans`, `--use-correct-plan`, `--duck-probe`. `eval_from_states`. |
| `pooled_demofit.py` | **the pooled generalist** — one PlanHead fit over SMW+MMX+SMB1. |
| `combine_eval.py`, `full_merge_eval.py` | merge plan-head (plan-OOD) + lora (actor-OOD) deltas → one model. |
| `pooled_eval_rigorous.py` | paired-bootstrap Δ_plan eval across games. |
| `extract_plan_head.py`, `merge_slim_to_full.py` | full ckpt ↔ slim trainable-delta conversion. |

## 🎬 Rollout demos & RAM-free verification
| script | what |
|---|---|
| `furthest_rollout.py` | longest closed-loop rollout until death → mp4 + frames.npz + RAM json. `--mode plan --delta`, `--log-plans` (overlay+log the live System-2 plan). |
| `vlm_video_judge.py` | a VLM watches frames → RAM-free `describe` / `compare` verdict. |
| `reach_eval.py`, `ordinal_sanity.py` | survival-weighted reach + the idle/random/base/right-jump falsifier controls. |

## 🔬 Plan-conditioning analysis (the research probes)
| script | what |
|---|---|
| `maneuver_router.py` | evocation-vs-addition router (null-AUC reflex + evoc-ratio + emulator survive+advance). |
| `plan_authority_map.py` | CFG ‖v_plan − v_null‖ — **how much the DiT actually moves under the plan** (plan-quality vs DiT-adherence). |
| `staleness_probe.py` | staleness/planner-variance probe (null/oracle/live/cached/fresh modes). |
| `expressiveness_battery.py` | 5-contrast plan→action responsiveness R (duck/retreat/jump/wait/up). |
| `base_dit_perdim.py`, `narration_residual*.py` | per-dim frame-counterfactual residual (does the frame alone determine the action?). |
| `eval_buttons.py`, `eval_balance.py`, `eval_plan_sensitivity.py`, `probe_*.py` | button/direction steering + separation probes. |
| `eval_seq*.py`, `eval_crosschunk*.py`, `eval_stage2*.py`, `eval_cfg_override.py` | Stage-1/2 capability evals (ordering, cross-chunk, override). |

## 📥 Data, envs & infra
| script | what |
|---|---|
| `record_play_server.py` | browser play-and-record server (human gold trajectories on a headless box). |
| `emulator_state_authoring_server.py` | emulator save-state authoring UI. |
| `env_healthcheck.py` | boot-test all envs. |
| `bk2_to_npz.py` | replay a stable-retro movie → npz (bit-exact). |
| `yt_farm.py`, `objective_label_demo.py` | YouTube frames + free VLM objective labels (data bootstrap). |
| `idm_gen_emulator_data.py`, `idm_gate.py` | emulator ground-truth (frame,action) + the IDM feasibility gate. |
| `vlm_narrate_demo.py`, `demo_narration.py` | gold-action VLM narration of demos (+ `CONTROL_SCHEMA`). |
| `gen_stage2_*.py`, `cache_mm_*.py`, `cache_s2_windows.py` | Stage-2 mm cache builders. |
| `ingest_external_state.py`, `smbas_find_progress2.py`, `verify_button_order.py` | RAM/state plumbing. |

## ⚗️ RL scaffolding (designed, mostly un-run)
`collect_rl_trajectories.py`, `rl_eval_plan_vs_null.py`, `rl_rollout_demo.py`, `rl_planner_prompt.py`,
`rank_rl_games.py`, `ddpo_explore.py`, `latent_plan_search.py`, `best_of_k.py`, `deferral_gate.py`,
`deferral_after_adapt.py`, `joint_alternating.py`.

## 🧪 OPSD judge / benchmarks
`judge_vlm_opsd.py`, `judge_calibrate.py`, `opsd_diag.py`, `bench_rollout.py`, `bench_judge_vlm.py`,
`bench_aggregate.py`, `build_judge_manifest_from_rollout.py`, `benchmark_prompts.py`, `benchmark_think.py`,
`s2_prompt_sweep.py`, `generate_grounded_plan.py`, `vlm_plan_alignment*.py`, `vlm_plan_from_frame.py`.

## 🗄️ `attic/` — archived one-off probes/tests
52 superseded one-shot scripts (cavestory POCs, `*_test.py` button/shoot/jump smokes, `probe_*_separation.py`,
`smbas_find_progress{,3}.py`, `record_all_videos.py`, etc.). Kept in git for provenance; restore with
`git mv planner_poc/attic/<name>.py planner_poc/`.
