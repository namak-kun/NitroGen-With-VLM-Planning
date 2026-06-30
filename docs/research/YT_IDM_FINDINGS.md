# YouTube gameplay leverage for IDM / plans / eval

Date: 2026-06-27
Repo: `/home/t-nagupta/NitroGen-With-VLM-Planning`

## Bottom line

YouTube is real leverage, but mostly **not** as direct policy data until an IDM passes a small, harsh feasibility gate. The highest-value path is still VPT-style: train a non-causal IDM on ROM-matched action-labeled data, then pseudo-label ordinary YouTube. The near-term surprise is that YouTube is also useful for **evaluation**: many playthroughs of the same level can define a multi-path progress envelope, even though video alone cannot recover RAM state or savestates.

## 1. Infra status table

| Component | Status | Evidence from repo / probe | Blocked or stubbed pieces |
|---|---:|---|---|
| `planner_poc/yt_farm.py search` | Works | Command below returned 3 results. It uses `.venv/bin/python -m yt_dlp`, cookies if `cookies.txt` exists, and Deno/EJS if `/home/t-nagupta/.deno/bin/deno` exists. | Search can return broad/incorrect matches; query for “SMW 1-1” returned one SMB result plus SMW longplays. |
| `planner_poc/yt_farm.py meta` | Works | `meta` on `3Tc_Ek0ASSA` returned title, duration, uploader, and useful chapters (`Yoshi's Island` from 51-584s). | Chapters are coarse objectives only; not action labels. |
| `planner_poc/yt_farm.py grab` | Partially blocked | With Deno installed, it solved the JS challenge and listed real formats, but the script selected video-only format `396`; yt-dlp/ffmpeg section cutting hit `HTTP error 403 Forbidden`. | Script needs a format fallback/option. Progressive format `18` works for the same video and section; see exact command below. |
| YouTube auth / n-challenge | Works with caveat | `cookies.txt` exists and was used. Deno was missing at start; I installed Deno to the documented path. `yt-dlp -F` showed real HTTPS formats after `[jsc:deno] Solving JS challenges using deno`. | Docs contain two recipes: `docs/SETUP_YOUTUBE.md` emphasizes bgutil PO-token + Deno + `ejs:github`; `AGENTS.md` says Deno + `ejs:npm` was the confirmed recipe. Current direct progressive-format fetch works without bgutil. |
| Existing `docs/yt_farm/` artifacts | None before probe | `find docs ...` found no `docs/yt_farm` or `docs/idm_data` artifacts before I ran probes. | Probe artifacts now exist and are gitignored: `docs/yt_farm/3Tc_Ek0ASSA_direct/` (~7.4 MB) and `docs/idm_data/smw_probe/` (~100 KB). |
| `planner_poc/idm_gen_emulator_data.py` | Runs, but data quality is poor from cold start | Small probe: 200 SMW frames generated successfully from `Game data/Super Mario World.sfc`. Diagnostic: mean frame-motion `1.10`, below the script's own “want >2” threshold; 166/199 frame-to-frame motions were `<0.5`; 33/200 frames were dark. | This confirms the documented failure: cold-start random macro sampler often stays static or dies/black-screens. Need mid-game savestates or smarter macro/state sampler before IDM training data is useful. |
| IDM model/training | Not implemented in inspected infra | `idm_gen_emulator_data.py` only emits `.npz` with `frames`, NitroGen 25-d actions, compact button labels, and `meta.json`. | Need a non-causal window→button/action model, held-out validation, and a YouTube pseudo-labeler. |
| `planner_poc/objective_label_demo.py` | Implemented demo, not re-run here | Loads frozen `PlanEncoder`/Qwen, takes frame windows, emits one-sentence grounded objectives. This is System-2 label generation. | It gives objectives/plans, not actions. Without ground-truth actions the labeler is non-privileged, unlike GPT-5.5 demo relabeling that saw actions. |
| TASVideos / movie replay | Intentionally rejected | `AGENTS.md` says “TAS” means infer actions from ordinary videos via IDM, **not** TASVideos movie files. | Do not pursue `retro.Movie` replay; it is too glitchy/precise and not representative human gameplay. |

## 2. YouTube fetch test: exact commands and outcomes

### Search: works

```bash
cd /home/t-nagupta/NitroGen-With-VLM-Planning
RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
$RUN .venv/bin/python planner_poc/yt_farm.py search "Super Mario World 1-1 playthrough" -n 3
```

Output:

```text
zLaF9fOqDaQ  2m  Super Mario Bros (Nes) - World 1-1  [Nintendius]
3Tc_Ek0ASSA  160m  Super Mario World - Complete Walkthrough  [Typhlosion4President]
Be4LB4qTiTI  159m  Super Mario World - Complete 100% Walkthrough - All Secret Exits - No Damage (Longplay)  [ModernXP]
```

### `yt_farm.py grab`: blocked by script format choice

Before Deno existed, the same script omitted `--js-runtimes` and failed with:

```text
ERROR: [youtube] 3Tc_Ek0ASSA: Requested format is not available. Use --list-formats for a list of available formats
```

After installing Deno at `/home/t-nagupta/.deno/bin/deno`, this command:

```bash
$RUN .venv/bin/python planner_poc/yt_farm.py grab "https://www.youtube.com/watch?v=3Tc_Ek0ASSA" --sections 60-75 --fps 2 --height 360
```

solved the JS challenge, but failed while downloading the selected format:

```text
[youtube] [jsc:deno] Solving JS challenges using deno
[info] 3Tc_Ek0ASSA: Downloading 1 format(s): 396
[info] 3Tc_Ek0ASSA: Downloading 1 time ranges: 60.0-75.0
[https ...] HTTP error 403 Forbidden
ERROR: ffmpeg exited with code 8
subprocess.CalledProcessError: Command ... '-f', 'bestvideo[height<=360][ext=mp4]/best[height<=360]' ... returned non-zero exit status 1.
```

Interpretation: auth/challenge solving is good enough to list real formats, but `yt_farm.py`'s hard-coded `bestvideo[height<=360][ext=mp4]` selected video-only `396`, and section cutting via ffmpeg hit a 403. This is probably fixable by trying progressive `18` before video-only formats, or exposing a `--format` argument.

### Direct yt-dlp + ffmpeg progressive format: works

This command fetched only a 15s slice and extracted frames:

```bash
cd /home/t-nagupta/NitroGen-With-VLM-Planning
mkdir -p docs/yt_farm/3Tc_Ek0ASSA_direct/frames
.venv/bin/python -m yt_dlp --no-warnings \
  --cookies cookies.txt \
  --js-runtimes deno:/home/t-nagupta/.deno/bin/deno \
  --remote-components ejs:npm \
  -f 18 \
  -o 'docs/yt_farm/3Tc_Ek0ASSA_direct/clip_%(section_start)s.%(ext)s' \
  --download-sections '*60-75' \
  'https://www.youtube.com/watch?v=3Tc_Ek0ASSA' && \
ffmpeg -loglevel error \
  -i docs/yt_farm/3Tc_Ek0ASSA_direct/clip_60.0.mp4 \
  -vf fps=2 \
  docs/yt_farm/3Tc_Ek0ASSA_direct/frames/60.0_%04d.png
```

Result: 30 extracted 640×360 frames. Sanity stats for a middle frame: mean `139.41`, std `83.11`, so frames are not blank.

Metadata command:

```bash
$RUN .venv/bin/python planner_poc/yt_farm.py meta "https://www.youtube.com/watch?v=3Tc_Ek0ASSA"
```

Key returned chapter:

```json
{"start": 51.0, "end": 584.0, "title": "Yoshi's Island"}
```

## 3. Leverage-path assessment

### A. IDM for System-1 grounding: emulator/demo GT → IDM → pseudo-label YouTube → train DiT

**Value:** Very high. This is the only path that converts abundant online video into the missing System-1 supervision: pixel/window → buttons/actions. It directly attacks the owner’s repeated blocker: NitroGen was trained from scratch and needs button grounding; the VLM/plans alone do not provide actions.

**Tractability:** Medium, with a good cheap gate. VPT works because IDM is easier than policy: it sees future frames. But the current emulator sampler is not yet producing learnable data by default. My 200-frame SMW probe reproduced the known issue: mean motion `1.10`, 83% near-static frame transitions, 16.5% dark frames. Training on that would teach “nothing happened” more than “button caused motion.”

**Main blockers:**

1. **Cold-start emulator data quality.** Random macros from title/start can get stuck, remain static, or die. Need mid-level savestates, scripted resets, or a state sampler that rejects low-motion/death segments.
2. **Domain gap.** Emulator frames are clean/pixel-perfect; YouTube frames are 360p, compressed, possibly scaled/cropped, with overlays and uploader-specific presentation. IDM training needs augmentation toward YouTube: resize/crop jitter, H.264/WebP-style compression, blur, color shifts, frame-rate jitter, HUD/black border robustness.
3. **Action non-identifiability.** Some buttons leave weak or delayed evidence; e.g. holding run vs not, jump press timing, buffering, simultaneous buttons, and same visual future from different held-button histories. Non-causal windows help, but the target may need button-level probabilities, temporal smoothing, and “unknown/low-confidence” filtering.
4. **Button-set differences across games.** A single generalist IDM must learn game-conditioned controls. Use shared physical controller vocabulary, but validate per-system/per-game calibration. Do not train per-genre models if the goal is one generalist.
5. **OOD games/systems.** Genesis/GBA OOD for NitroGen means IDM pseudo-labels alone may not fix the actor; the DiT/LoRA still needs enough visual/action diversity.

**Cheapest de-risking experiment:**

Do not scale YouTube first. Train a tiny non-causal IDM on **known-good action-labeled data**: the 15 expert demos plus short high-motion emulator snippets generated from mid-game savestates or filtered by frame-motion. Use 5-9 frame windows and predict compact buttons (`BTN_VOCAB`) plus NitroGen 25-d actions. Validate on held-out human demos. Success criterion before pseudo-labeling YouTube: high F1 on LEFT/RIGHT/JUMP/RUN, calibrated confidence, and plausible temporal button traces. If it fails on ROM-matched held-out demos, YouTube pseudo-labeling is illusory.

### B. A-posteriori plans on YouTube frames: System-2 labels without actions

**Value:** Medium for planner/critic/eval metadata; low as direct action-learning data.

**Tractability:** High technically. `objective_label_demo.py` already generates grounded objectives from frame windows, and YouTube frames can now be fetched with a small command workaround.

**Crucial distinction from the 15 demos:** In the demo relabeling setup, GPT-5.5 saw the expert actions. That makes labels **privileged hindsight plans**: “what the human actually did and why.” On YouTube frames alone, the labeler sees only pixels. It can say “go right and jump over the enemy,” but it does not know the actual held buttons, sub-frame timing, or whether the player is about to wait, take a secret route, farm a power-up, or intentionally die/reset. It is an observation-conditioned intent guess, not an action-conditioned explanation.

**Where non-privileged plans are still useful:**

- Coarse System-2 objectives: “enter the pipe,” “clear the gap,” “avoid the Koopa,” “continue right.”
- Segment/chapter labeling: uploader chapters + VLM labels can cheaply index large video corpora by objective/milestone.
- Training VLM/plan LoRA for grounded language if paired with action labels from an IDM later.
- Reward/model-eval prompts: compare whether the agent is making progress toward visible objectives.

**Where leverage is illusory:** Plans alone do not solve System-1 grounding. If a model gets frame + plan but no action labels/rewards, it still does not learn which buttons produce the next observed motion. Non-privileged plans can be wrong or underspecified, and may reinforce VLM memorization on popular games rather than true control.

**Cheapest de-risking experiment:** On the 30 fetched frames, run `objective_label_demo.py` and compare labels to visible gameplay; then run the same prompt on an obscure/homebrew game. If SMW labels are good but obscure labels degrade, use YouTube plan labels only as weak metadata, not as proof of generalist capability.

### C. Eval angle: many YouTube playthroughs as a distribution of valid paths

**Value:** High for the current eval headache, but it is not a substitute for environment rollouts.

The owner’s issue is that an expert demo is only one valid sequence. Many YouTube playthroughs of the same level can define a **distribution of human-valid progress**, not a single path. This is valuable because model rollouts can be scored against an envelope: “is the agent within the human progress/time band and reaching common milestones?” instead of “did it match one demonstration frame/action sequence?”

**What can be recovered from video alone:**

- Pixels/frames, timestamps, rough scene segments, chapters, and visible milestones.
- Approximate on-screen player position and motion via template matching / object detection / optical flow.
- Approximate progress-vs-time bands for side-scrollers if camera scroll or background position is inferable.
- Some visible state: lives, timer, score, power-up form, coins, HUD text, map/level name, if OCR/template extraction works.
- A union/envelope of common routes and timing: e.g. fast route, cautious route, secret path, power-up detour.

**What cannot be recovered from video alone:**

- Frame-exact emulator RAM state.
- Savestates or resettable start states.
- Ground-truth controller buttons.
- Hidden variables, RNG, subpixel position/velocity, collision state, inventory flags not visible on HUD.
- Exact alignment to this repo’s emulator unless the video was generated by our recorder with saved state/action logs.

**Feasibility given no frame-exact alignment:** Medium. For SMW-like side-scrollers, a useful approximate eval is feasible: segment videos to a level, normalize crop/scale, detect Mario/camera progress, build a progress-vs-time percentile band, then compare our emulator rollout video/RAM x-position to that band. But this cannot become a save-state evaluator. It is an aggregate reference, not a source of replayable states.

**Cheapest de-risking experiment:** Use 5-10 short SMW Yoshi’s Island 1 clips, manually or heuristically segment level entry/exit, and build a simple visible-progress curve: player screen x plus inferred camera scroll / level section template. Compare one existing expert demo and one model rollout to the human percentile envelope. If the envelope separates “valid but different route” from “stuck/dead,” it immediately improves eval.

## 4. Ranking by value × tractability

1. **IDM feasibility gate, then YouTube pseudo-labeling** — highest value, medium tractability. This is the only route that turns YouTube into action data for System 1. It must be gated on held-out gold-demo performance and high-motion emulator data; otherwise it will generate confident garbage.
2. **Multi-path YouTube eval envelope** — high practical value, medium-high tractability. It addresses the “one expert path” eval problem sooner than full IDM and does not require action recovery. It cannot provide savestates, but it can provide a human progress distribution.
3. **Non-privileged VLM plans on YouTube** — technically easy, medium/low standalone value. Useful as weak objective metadata and later as plan text paired with IDM pseudo-actions; not sufficient by itself for action learning.
4. **TAS/movie replay** — reject. It is not the desired problem and is likely harmful distributionally.

## 5. Single prioritized recommendation

Run a **one-day IDM gate** before building any large pipeline:

1. Patch or invoke YouTube fetching with a progressive fallback (`-f 18` worked) so short clips are reliable.
2. Train a small non-causal IDM on the 15 expert demos plus only high-motion emulator snippets; hold out at least one SMW/Sonic/Minish/FE demo per family if possible.
3. Validate button F1/calibration on held-out gold demos, especially LEFT/RIGHT/JUMP/RUN and simultaneous buttons.
4. Only if it passes, pseudo-label 1-2 minutes of SMW YouTube and inspect temporal button traces plus confidence filtering.

If this gate fails, prioritize the eval-envelope path and better human/emulator data collection over more YouTube downloads. If it passes, YouTube becomes genuinely valuable scale data for the generalist System-1 actor, and the VLM plan labels can be layered on top as System-2 weak supervision.
