# Cave Story (doukutsu-rs) headless eval setup

Reproducible setup for the first FOSS closed-loop eval env. **Feasibility is proven**: the
entire I/O loop runs headless with **zero engine modifications** — build → software-GL render
under Xvfb → freeware data → frame capture (observation) → key injection (action) → navigated
menus → reached gameplay. `nitrogen/eval/envs/cavestory.py` implements `CaveStoryEnv` on this
backend; `planner_poc/cavestory_demo.py` validates the Python class end-to-end.

## Why this works without patching the engine
- **Observe**: doukutsu-rs renders with OpenGL; under `Xvfb` + Mesa `llvmpipe` (software GL)
  it produces a real framebuffer headlessly. We grab it with `ffmpeg -f x11grab` → RGB frame.
- **Act**: doukutsu-rs reads the keyboard (default map: arrows=move, `Z`=jump/confirm,
  `X`=shoot/back, `A`/`S`=weapon). We inject `xdotool` XTEST key events to the focused window.
- **Map** the 25-dim NitroGen gamepad → these keys in `CaveStoryEnv._chunk_keys`.

## One-time setup (already done on this machine)

```bash
# 1. Rust toolchain (system cargo 1.75 is too old for edition2024 deps):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"        # rustc 1.96

# 2. Build deps + headless display + capture/inject tools:
sudo apt-get install -y cmake libasound2-dev libudev-dev libx11-dev libxext-dev \
    libgl1-mesa-dev libegl1-mesa-dev xvfb mesa-utils libgl1-mesa-dri ffmpeg xdotool

# 3. Build doukutsu-rs (open-source Cave Story engine):
cd /tmp && git clone --depth 1 https://github.com/doukutsu-rs/doukutsu-rs.git
cd doukutsu-rs && cargo build --release        # -> target/release/doukutsu-rs

# 4. Freeware Cave Story data (free, no ROM) next to the binary:
curl -sSL -o /tmp/cs.zip https://www.cavestory.org/downloads/cavestoryen.zip
cd /tmp && unzip -q cs.zip
cp -r /tmp/CaveStory/data /tmp/doukutsu-rs/target/release/
cp /tmp/CaveStory/Doukutsu.exe /tmp/doukutsu-rs/target/release/   # engine extracts embedded assets
```

## Run

```bash
# validate the env class (boots headless, navigates menus, steps, saves frames):
python planner_poc/cavestory_demo.py        # -> /tmp/cs_env_*.png

# drive it with the real model:
#   build scenarios + run EvalProtocol with planner_poc/eval_policy.NitroGenPolicy
```

## Status & limitations (v1)
- ✅ Headless build + run, frame capture, key injection, menu navigation, reach gameplay.
- ✅ `CaveStoryEnv` (GameEnv): boot, reset-macro, `step(chunk)` → frame observation.
- ⚠️ **Real-time + async**: the game runs while the GPU policy thinks (matches the intended
  System-2 async deployment, but adds timing noise). Each `step` holds the chunk's dominant
  keys for `chunk_seconds` (intra-chunk timing is collapsed in v1).
- ⚠️ **No privileged state yet**: `obs.state` only has `step`. Use `VLMJudgeDetector` (frames)
  for success until the v2 engine patch.
- ⚠️ **Reset determinism**: v1 reset replays a menu macro; for exact "same start, different
  goals", point the macro at a save-slot load (doukutsu-rs save profiles).

## v2 (deterministic, exact-state) — engine patch
A doukutsu-rs fork with a `--agent-socket` flag (a socket `PlayerController` + per-tick state
export + framebuffer send + frame-stepping) removes real-time noise and exposes player
x/y/map/life/flags for exact `StatePredicateDetector`. Protocol sketch at the bottom of
`nitrogen/eval/envs/cavestory.py`. Templates: `src/input/{dummy,replay}_player_controller.rs`.
