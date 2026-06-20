# Cave Story (doukutsu-rs) headless eval setup

Reproducible setup for the first FOSS closed-loop eval env. **Feasibility is proven**: the
entire I/O loop runs headless with **zero engine modifications** — build → software-GL render
under Xvfb → freeware data → frame capture (observation) → key injection (action) → navigated
menus → reached gameplay. `nitrogen/eval/envs/cavestory.py` implements `CaveStoryEnv` on this
backend; `planner_poc/cavestory_demo.py` validates the Python class end-to-end.

## Why this works (faithful GAMEPAD interface, no engine integration)
- **Observe**: doukutsu-rs renders with OpenGL; under `Xvfb` + Mesa `llvmpipe` (software GL)
  it produces a real framebuffer headlessly. We grab it with `ffmpeg -f x11grab` → RGB frame.
- **Act**: NitroGen outputs a **gamepad** action (analog sticks + buttons), so we present a
  **virtual Xbox-360 gamepad** via Linux `uinput` (VID/PID `0x045e/0x028e` so SDL2 auto-maps
  it). NitroGen's 25-dim action → analog axes `ABS_X/Y/RX/RY` + buttons + d-pad. SDL reads the
  device directly (evdev), so **no window focus / xdotool is needed** — even menus.
- **One required engine tweak**: on desktop Linux doukutsu-rs defaults Player 1 to the
  **keyboard**, so the gamepad is detected but unused. Patch `default_p1_controller_type()` in
  `src/game/settings.rs` to `ControllerType::Gamepad(0)` and rebuild. (This is a config default,
  not the deeper v2 socket integration.)

## One-time setup (already done on this machine)

```bash
# 1. Rust toolchain (system cargo 1.75 is too old for edition2024 deps):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"        # rustc 1.96

# 2. Build deps + headless display + capture tool + virtual-gamepad lib:
sudo apt-get install -y cmake libasound2-dev libudev-dev libx11-dev libxext-dev \
    libgl1-mesa-dev libegl1-mesa-dev xvfb mesa-utils libgl1-mesa-dri ffmpeg xdotool
uv pip install --python .venv/bin/python evdev      # virtual gamepad (uinput)
sudo modprobe uinput && sudo chmod 666 /dev/uinput  # allow creating the virtual pad

# 3. Get doukutsu-rs + patch P1 to use the gamepad, then build:
cd /tmp && git clone --depth 1 https://github.com/doukutsu-rs/doukutsu-rs.git
cd doukutsu-rs
#   edit src/game/settings.rs: default_p1_controller_type() -> ControllerType::Gamepad(0)
cargo build --release                          # -> target/release/doukutsu-rs

# 4. Freeware Cave Story data (free, no ROM) next to the binary:
curl -sSL -o /tmp/cs.zip https://www.cavestory.org/downloads/cavestoryen.zip
cd /tmp && unzip -q cs.zip
cp -r /tmp/CaveStory/data /tmp/doukutsu-rs/target/release/
cp /tmp/CaveStory/Doukutsu.exe /tmp/doukutsu-rs/target/release/   # engine extracts assets
```

## Run

```bash
# validate the env class (boots headless, navigates menus, steps, saves frames):
python planner_poc/cavestory_demo.py        # -> /tmp/cs_env_*.png

# drive it with the real model:
#   build scenarios + run EvalProtocol with planner_poc/eval_policy.NitroGenPolicy
```

## Status & limitations (v1)
- ✅ Headless build + run, frame capture, **virtual gamepad detected AND controlling the game**
  (drove the title menu, reached free-roam gameplay "Start Point", and **Quote walks** on
  left-stick input — verified by frame diffs + visible position change).
- ✅ `CaveStoryEnv` (GameEnv): boot, gamepad reset-macro (`btn`/`dir`/`wait`), `step(chunk)` →
  per-step analog gamepad replay → frame observation.
- ⚠️ **Real-time + async**: the game runs while the GPU policy thinks (matches the intended
  System-2 async deployment, but adds timing noise).
- ⚠️ **No privileged state yet**: `obs.state` only has `step`. Use `VLMJudgeDetector` (frames)
  for success until the v2 engine patch.
- ⚠️ **Reset determinism**: v1 reset replays a menu+intro macro to reach the First Cave. For
  exact "same start, different goals", make a **save slot** at a chosen spot and reset = load it
  (doukutsu-rs save profiles) — sidesteps the long opening cutscene entirely.

## Generality + other games
The **virtual gamepad is game-agnostic**: any SDL2 game (most of the FOSS list in
`docs/FOSS_GAMES.md`) will see the same Xbox-360 controller, so this backend is not Cave-Story-
specific. Free **roguelikes** (many on itch.io, often Linux-native) and other open-source titles
plug in by subclassing `GameEnv` (or reusing `CaveStoryEnv`'s process+pad+grab machinery) and
writing a reset macro + success spec. Windows-only titles can run in a VM as a fallback.

## v2 (deterministic, exact-state) — engine patch
A doukutsu-rs fork with a `--agent-socket` flag (a socket `PlayerController` + per-tick state
export + framebuffer send + frame-stepping) removes real-time noise and exposes player
x/y/map/life/flags for exact `StatePredicateDetector`. Protocol sketch at the bottom of
`nitrogen/eval/envs/cavestory.py`. Templates: `src/input/{dummy,replay}_player_controller.rs`.
