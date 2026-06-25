# Setup — eval harness, record server & data bootstrapping (headless Linux)

How to stand up the **eval environments**, the **record/play server**, and the **data-bootstrapping**
tooling on a fresh headless Linux box. (For the core plan-conditioning train/eval, see the main README;
for YouTube ingestion specifics, `docs/SETUP_YOUTUBE.md`.)

> The dev box used a uv-managed `.venv`. Substitute your own venv/conda as needed. The repo shells out
> to system tools (Xvfb/ffmpeg/xdotool) — those are apt packages, not pip.

## 1. Python deps

```bash
pip install -e ".[eval]"          # adds mss, yt-dlp, stable-retro, pygba, opencv-python
# (or: uv pip install -e ".[eval]" --python .venv/bin/python)
hf download nvidia/NitroGen ng.pt   # base DiT  -> ckpts/nitrogen/ng.pt
hf download Qwen/Qwen3.5-2B         # planner backbone (0.8B also works)
```

To reload a released plan-conditioning checkpoint, unzip one from `ckpts/handoff_zips/` and:
```python
slim = torch.load("btn_s600.pt", map_location="cpu", weights_only=False)
model.load_state_dict(slim["trainable_ema"], strict=False)   # after loading base ng.pt
```

## 2. System packages (apt)

The process-backed game envs run under a headless X server and drive games via synthetic input:
```bash
sudo apt-get install -y \
  xvfb ffmpeg xdotool x11-xserver-utils matchbox-window-manager \
  gcc cmake build-essential          # to build the speedhack shim + native game envs
```
- **Xvfb** — headless X display per env.
- **ffmpeg** — fallback frame grab (the record server prefers `mss`, ~280× faster).
- **xdotool** — synthetic keyboard input + window search/focus.
- **x11-xserver-utils** — provides `xset` (record server disables key autorepeat to avoid over-input).
- **matchbox-window-manager** — needed by focus-gated games (SFML etc.).
- The **speedhack** time-shim (`nitrogen/eval/speedhack/libspeedhack.c`) auto-compiles via gcc on first
  use (freeze/slow-mo during inference + the record server's lag-free stepping).

## 3. JS runtime for YouTube (deno)

YouTube enforces a JS "n-challenge"; yt-dlp needs an external JS runtime + the EJS solver. **Deno** is
the verified path (node alone did not work on this host):
```bash
curl -fsSL https://deno.land/install.sh | sh -s -- -y     # -> ~/.deno/bin/deno (NOT on PATH)
```
`planner_poc/yt_farm.py` passes `--js-runtimes deno:/home/<user>/.deno/bin/deno --remote-components
ejs:npm` automatically (override the path via the `DENO_BIN` env var). You also need a logged-in
`cookies.txt` (Netscape format) at repo root — **gitignored, never commit it**. See
`docs/SETUP_YOUTUBE.md` for the cookies + (alternate bgutil PO-token) details.

Verify:
```bash
python planner_poc/yt_farm.py search "Super Mario World longplay no commentary" -n 3
```

## 4. Emulator envs (in-process, frame-exact save/load — the RL substrate)

```bash
pip install -e ".[eval]"        # pulls stable-retro (SNES/Genesis/NES) + pygba (GB/GBC/GBA)
```
- **mGBA** (GB/GBC/GBA): `pygba==0.2.4` wrapping `mgba==0.10.5`. If the wheel's core is missing, build a
  local mgba 0.10.5 and point pygba at it (see `nitrogen/eval/envs/_emulator_notes.md`).
- **stable-retro** (SNES via snes9x, Genesis via genesis_plus_gx, NES via fceumm): ships the cores.
- ROMs live in `tmp/roms/` (homebrew, in repo workflow) or `Game data/` (user's commercial ROMs) —
  both **gitignored**. Genesis has NO in-process save/load (Mednafen subprocess only).

Verify all envs boot:
```bash
python planner_poc/env_healthcheck.py        # ~30/38 OK; BROKEN_ENVS (lost artifacts) are hidden
```

## 5. Record / play server (collect gold trajectories)

```bash
env -u VIRTUAL_ENV -u PYTHONPATH \
  PYTHONPATH=$PWD:$PWD/planner_poc \
  nohup .venv/bin/python planner_poc/record_play_server.py --env thextech_get_flower --port 8123 \
  > /tmp/record_server.log 2>&1 &
```
- Forward port 8123 to your laptop (VS Code PORTS tab → change Local Port, or `ssh -L 9123:localhost:8123`).
- **STEP mode** (default, lag-immune): hold a key + tap **Tab** = one deterministic move. Toggle
  **auto-step** (hold-to-walk) or **REALTIME** (0.06–2× speed) from the page.
- Recordings → `docs/recordings/<env>_<ts>/{frames/*.png, actions.jsonl, meta.json}` (gitignored).
- **Stop with `kill <pid>`** (graceful — cleans up the child game + Xvfb). NOT `kill -9` (orphans them).

## 6. Data bootstrapping pipeline

```bash
# YouTube frames (+ chapters as coarse objectives):
python planner_poc/yt_farm.py grab "<url>" --sections 600-660 1200-1260 --fps 4
# Free System-2 objective labels from the frozen VLM on farmed frames:
python planner_poc/objective_label_demo.py --frames "docs/yt_farm/<id>/frames/*.png"
# Ground-truth (frame, action) from an emulator, for IDM training:
python planner_poc/idm_gen_emulator_data.py --rom "Game data/Super Mario World.sfc" \
  --system snes --steps 4000 --out docs/idm_data/smw
```
See `AGENTS.md §3C` and session `files/BOOTSTRAP_DATA_STRATEGY.md` for the full strategy (emulator
ground truth → IDM/VPT → pseudo-label YouTube → plan-conditioned training + save-state RL verification).

## 7. Native game envs that need building (optional)

Most envs are apt games or emulators. A few (TheXTech, SDLPoP, Solarus/ZSDX, some Godot/Bevy titles)
build from source into `/tmp/<game>` or `.nitrogen-env-build/`. The 5 envs hidden by `BROKEN_ENVS`
(daemon_vs_demon, megaman_maverick, openmw, theseeker, zelda_classic) have lost build artifacts and need
re-fetching — low priority (30/38 work). See `docs/FOSS_GAMES.md` for per-game build notes.

**State export (TheXTech / Solarus) — de-fork (recommended).** Today these use a source/quest patch.
No-fork alternatives are documented in `AGENTS.md §6`: TheXTech via a stock debug-symbol build + a
`/proc/<pid>/mem` reader (non-PIE fixed addresses); Solarus via the engine's `-s=` startup-Lua flag
instead of editing the quest's `main.lua`.
