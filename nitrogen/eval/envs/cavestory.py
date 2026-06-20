"""CaveStoryEnv — closed-loop GameEnv backed by doukutsu-rs (open-source Rust reimplementation
of the freeware Metroidvania Cave Story). FEASIBILITY-PROVEN backend: the whole I/O loop runs
HEADLESS with ZERO engine modifications —
  * render/observe: the game runs under Xvfb with software OpenGL (Mesa llvmpipe); we grab the
    framebuffer with ffmpeg x11grab -> the RGB frame the policy sees.
  * act: we inject the 25-dim NitroGen gamepad as X11 key events (xdotool XTEST) to the focused
    game window, using doukutsu-rs's default keyboard map (arrows=move, Z=jump/confirm,
    X=shoot, A/S=weapon switch).
Validated end-to-end: launched the game, navigated the menus, started a new game, reached
gameplay (stage 13) — all programmatically, headless.

This v1 is REAL-TIME and asynchronous (the game keeps running while the GPU policy thinks),
which actually matches the intended System-2(async)/System-1(real-time) deployment. A future
v2 engine patch (a socket PlayerController + state export + frame-stepping) would add exact
determinism + privileged state (player x/y/map/life/flags) for StatePredicateDetector; until
then use VLMJudgeDetector on frames. The engine-patch protocol is documented at the bottom.

Setup (once): build doukutsu-rs (cargo), drop the freeware Cave Story `data/` + `Doukutsu.exe`
next to the binary, install Xvfb + ffmpeg + xdotool. See scripts/setup_cavestory.md.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

import numpy as np

from ..core import JLX, JLY, N_BUTTONS, GameEnv, Observation, Scenario

# Cave Story key bindings (doukutsu-rs p1_default_keymap) as X11 keysyms for xdotool.
KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = "Left", "Right", "Up", "Down"
KEY_JUMP, KEY_SHOOT = "z", "x"           # jump / confirm, shoot / back
KEY_WPREV, KEY_WNEXT = "a", "s"          # prev / next weapon
# NitroGen button indices (xbox-ish, see nitrogen/training/actions.py BUTTON_ORDER):
BTN_JUMP, BTN_SHOOT, BTN_WPREV, BTN_WNEXT = 0, 1, 4, 5
MOVE_THRESH = 0.4                         # stick deflection that counts as a held direction
BTN_FRAC = 0.30                           # press a button if active in >= this fraction of a chunk


def _free_display() -> int:
    for d in range(101, 200):
        if not os.path.exists(f"/tmp/.X{d}-lock"):
            return d
    raise RuntimeError("no free X display")


class CaveStoryEnv(GameEnv):
    name = "cave_story"
    action_hz = 30.0

    def __init__(self, drs_dir: str = "/tmp/doukutsu-rs/target/release",
                 width: int = 640, height: int = 480, display: int | None = None,
                 reset_macro: list | None = None, boot_wait: float = 10.0,
                 chunk_seconds: float = 0.6, launch: bool = True):
        self.drs_dir = drs_dir
        self.width, self.height = width, height
        self.display = display if display is not None else _free_display()
        self.boot_wait = boot_wait
        self.chunk_seconds = chunk_seconds
        # reset_macro: list of ("key", keysym) | ("wait", seconds) to drive menus to a known
        # start (e.g. New Save -> Normal -> Single Player -> gameplay). Game-flow-specific, so
        # the caller/scenario supplies it. Default = boot only.
        self.reset_macro = reset_macro or []
        self._xvfb = None
        self._game = None
        self._wid = None
        self._step = 0
        for tool in ("Xvfb", "ffmpeg", "xdotool"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"{tool} not found; see scripts/setup_cavestory.md")
        if launch:
            self._boot()

    # ---- process / display management --------------------------------------------------
    def _env(self):
        return dict(os.environ, DISPLAY=f":{self.display}")

    def _boot(self):
        self._xvfb = subprocess.Popen(
            ["Xvfb", f":{self.display}", "-screen", "0", f"{self.width}x{self.height}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        binp = os.path.join(self.drs_dir, "doukutsu-rs")
        self._game = subprocess.Popen(
            [binp, "--window-width", str(self.width), "--window-height", str(self.height)],
            cwd=self.drs_dir, env=self._env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(self.boot_wait)
        self._wid = self._find_window()

    def _find_window(self) -> str:
        out = subprocess.run(["xdotool", "search", "--name", "Cave Story"],
                             env=self._env(), capture_output=True, text=True)
        ids = [x for x in out.stdout.split() if x.strip()]
        if not ids:
            raise RuntimeError("doukutsu-rs window not found on the display")
        return ids[-1]

    def _xdo(self, *args):
        subprocess.run(["xdotool", *args], env=self._env(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _press(self, key: str, hold: float = 0.12):
        self._xdo("keydown", "--window", self._wid, key)
        time.sleep(hold)
        self._xdo("keyup", "--window", self._wid, key)

    # ---- observation: grab the framebuffer --------------------------------------------
    def _grab(self) -> np.ndarray:
        p = subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-f", "x11grab",
             "-video_size", f"{self.width}x{self.height}", "-i", f":{self.display}.0",
             "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            capture_output=True)
        buf = p.stdout
        n = self.width * self.height * 3
        if len(buf) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(buf[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    # ---- action: chunk -> held keys ----------------------------------------------------
    def _chunk_keys(self, action_chunk: np.ndarray) -> list[str]:
        """Map an (H,25) chunk to the keys to HOLD this control step. v1 collapses intra-chunk
        timing to a dominant hold (mean stick -> direction; button if pressed in a fraction of
        steps). A v2 engine patch would honor per-step timing."""
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys: list[str] = []
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < -MOVE_THRESH: keys.append(KEY_LEFT)
        if mx > MOVE_THRESH: keys.append(KEY_RIGHT)
        if my < -MOVE_THRESH: keys.append(KEY_UP)
        if my > MOVE_THRESH: keys.append(KEY_DOWN)
        btn = a[:, :N_BUTTONS]
        if (btn[:, BTN_JUMP] > 0.5).mean() >= BTN_FRAC: keys.append(KEY_JUMP)
        if (btn[:, BTN_SHOOT] > 0.5).mean() >= BTN_FRAC: keys.append(KEY_SHOOT)
        if (btn[:, BTN_WPREV] > 0.5).mean() >= BTN_FRAC: keys.append(KEY_WPREV)
        if (btn[:, BTN_WNEXT] > 0.5).mean() >= BTN_FRAC: keys.append(KEY_WNEXT)
        return keys

    # ---- GameEnv interface -------------------------------------------------------------
    def reset(self, scenario: Scenario) -> Observation:
        macro = (scenario.success_spec or {}).get("reset_macro", self.reset_macro)
        for kind, val in macro:
            if kind == "key":
                self._press(val)
            elif kind == "wait":
                time.sleep(float(val))
        self._step = 0
        return Observation(frame=self._grab(), state={"step": 0}, step_idx=0, done=False)

    def step(self, action_chunk: np.ndarray) -> Observation:
        keys = self._chunk_keys(action_chunk)
        for k in keys:
            self._xdo("keydown", "--window", self._wid, k)
        time.sleep(self.chunk_seconds)
        for k in keys:
            self._xdo("keyup", "--window", self._wid, k)
        self._step += 1
        return Observation(frame=self._grab(), state={"step": self._step},
                           step_idx=self._step, done=False)

    def save_frame(self, path: str) -> None:
        from PIL import Image
        Image.fromarray(self._grab()).save(path)

    def close(self) -> None:
        for proc in (self._game, self._xvfb):
            if proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        lock = f"/tmp/.X{self.display}-lock"
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except OSError:
                pass


# ---- v2 (deterministic) engine-patch protocol, for reference --------------------------
# A doukutsu-rs fork with `--agent-socket <addr>` would enable a synchronous, exact env:
#   - a socket-driven PlayerController (template: src/input/dummy_player_controller.rs or
#     replay_player_controller.rs) that blocks each tick reading one input message;
#   - export player state each tick (src/game/player Player {x,y,life,...} + stage id + flags);
#   - send the rendered framebuffer (already in the GL backend) as PNG/raw.
#   Protocol (one JSON msg per control tick):
#     py->engine: {"buttons":{...},"move":{"left":bool,...},"frames":N} | {"reset":{"save":k}}
#     engine->py: {"frame_b64":..,"state":{"x":..,"y":..,"map":..,"life":..,"flags":[..]},"done":bool}
# This removes real-time noise and gives exact StatePredicateDetector success detection.
