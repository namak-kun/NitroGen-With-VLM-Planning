"""ProcGameEnv: a reusable base for adding a FOSS game to the NitroGen eval harness in ~30 lines.

A new game subclass declares only the GAME-SPECIFIC bits; this base owns ALL the shared machinery:
  * Xvfb headless display (auto free display number)
  * LD_PRELOAD speedhack (freeze-during-inference) — works on any dynamically-linked SDL/GL game
  * ffmpeg x11grab frame capture
  * xdotool keyboard control (window-targeted) with held-key state tracking
  * the GameEnv reset()/step()/close() loop with per-chunk timing + freeze handling

To add a game, subclass and set/override:
  REQUIRED:
    name: str                         # env id
    def launch_cmd(self) -> list[str] # argv to start the game (under the env's DISPLAY+speedhack)
    window_name: str                  # xdotool --name substring to find the game window
    def action_to_keys(self, chunk) -> set[str]
        # map a NitroGen (H,25) action chunk -> the set of keyboard keys to HOLD this step
        # (use core.JLX/JLY for sticks (dims 21/22), the BUTTON_* indices for buttons; threshold
        #  continuous stick to discrete keys). See KEYS helper + SuperTuxKart/CaveStory for examples.
  OPTIONAL:
    control: "keyboard" | "gamepad"   # default "keyboard" (xdotool). "gamepad" uses VirtualGamepad.
    boot_wait: float                  # seconds to wait after launch before the first window grab
    def read_state(self) -> dict      # privileged state for the detector (default {})
    def reset_macro(self, scenario)   # list of ("key",k)|("wait",s)|("hold",k,s) to reach a start
    needs_input_perms: bool           # chmod /dev/input for gamepad mode (default True if gamepad)

Then it's: env = MyGameEnv(); runner runs scenarios. NO boot/capture/freeze/control boilerplate.

Reward/state is the ONLY part that may need game-specific work (read_state + a detector); control,
capture, and freeze are free. For games where state isn't exposed, use a frame-based detector
(VLMJudgeDetector) — no game code needed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Optional

import numpy as np

from ..core import GameEnv, Observation, Scenario


def free_display() -> int:
    for d in range(80, 120):
        if not os.path.exists(f"/tmp/.X11-unix/X{d}"):
            return d
    return 99


class ProcGameEnv(GameEnv):
    """Process-backed game behind Xvfb + speedhack + x11grab + xdotool. Subclass per game."""

    name: str = "proc_game"
    window_name: str = ""
    control: str = "keyboard"            # "keyboard" | "gamepad"
    action_hz: float = 30.0

    def __init__(self, width: int = 800, height: int = 600, display: int | None = None,
                 boot_wait: float = 15.0, chunk_seconds: float = 0.6,
                 freeze_during_inference: bool = True, launch: bool = True):
        self.width, self.height = width, height
        self.display = display if display is not None else free_display()
        self.boot_wait = boot_wait
        self.chunk_seconds = chunk_seconds
        self.freeze_during_inference = freeze_during_inference
        self._xvfb = None
        self._game = None
        self._sh = None
        self._wid = None
        self._pad = None
        self._held: set[str] = set()
        self._step = 0
        req = ["Xvfb", "ffmpeg"] + (["xdotool"] if self.control == "keyboard" else [])
        for tool in req:
            if shutil.which(tool) is None:
                raise RuntimeError(f"{tool} not found (needed for ProcGameEnv)")
        if launch:
            self.boot()

    # ---- subclass hooks ----------------------------------------------------------------
    def launch_cmd(self) -> list[str]:
        raise NotImplementedError("subclass must return the game's argv")

    def action_to_keys(self, action_chunk: np.ndarray) -> set[str]:
        """Map (H,25) NitroGen action -> set of keyboard keys to hold (keyboard control)."""
        raise NotImplementedError

    def action_to_gamepad(self, action_chunk: np.ndarray) -> np.ndarray:
        """Map (H,25) chunk -> a single (25,) row for the VirtualGamepad (gamepad control).
        Default: mean over the chunk (analog passthrough)."""
        a = np.asarray(action_chunk, dtype=np.float32)
        return a.mean(0) if a.ndim == 2 else a

    def read_state(self) -> dict:
        return {}

    def reset_macro(self, scenario: Optional[Scenario]) -> list:
        return []

    # ---- shared machinery --------------------------------------------------------------
    def _env(self) -> dict:
        e = dict(os.environ, DISPLAY=f":{self.display}")
        if self._sh is not None:
            e = {**e, **self._sh.env()}
        return e

    def boot(self):
        self._xvfb = subprocess.Popen(
            ["Xvfb", f":{self.display}", "-screen", "0", f"{self.width}x{self.height}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        if self.control == "gamepad":
            from .virtual_gamepad import VirtualGamepad
            self._pad = VirtualGamepad()
            time.sleep(0.3)
            subprocess.run("sudo chmod 666 /dev/input/event* /dev/input/js* 2>/dev/null",
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if self.freeze_during_inference:
            from ..speedhack import SpeedHack
            self._sh = SpeedHack()
        self._game = subprocess.Popen(self.launch_cmd(), env=self._env(),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(self.boot_wait)
        if self.control == "keyboard":
            self._wid = self._find_window()

    def _find_window(self) -> str:
        if not self.window_name:
            return ""
        out = subprocess.run(["xdotool", "search", "--name", self.window_name],
                             env=self._env(), capture_output=True, text=True)
        ids = [l for l in out.stdout.split() if l.strip()]
        return ids[-1] if ids else ""

    def _xdo(self, *args):
        subprocess.run(["xdotool", *args], env=self._env(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _set_keys(self, keys: set[str]):
        for k in keys - self._held:
            self._xdo("keydown", *(["--window", self._wid] if self._wid else []), k)
        for k in self._held - keys:
            self._xdo("keyup", *(["--window", self._wid] if self._wid else []), k)
        self._held = set(keys)

    def _grab(self) -> np.ndarray:
        p = subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-f", "x11grab",
             "-video_size", f"{self.width}x{self.height}", "-i", f":{self.display}.0",
             "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            env=self._env(), capture_output=True)
        buf = p.stdout
        n = self.width * self.height * 3
        if len(buf) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(buf[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    def _apply(self, action_chunk: np.ndarray):
        """Apply one chunk: unpause, drive input for chunk_seconds (busy-wait), re-freeze."""
        if self._sh is not None:
            self._sh.unpause()
        if self.control == "keyboard":
            self._set_keys(self.action_to_keys(action_chunk))
            t = time.perf_counter()
            while time.perf_counter() - t < self.chunk_seconds:
                pass
        else:  # gamepad: replay rows for analog fidelity
            a = np.asarray(action_chunk, dtype=np.float32)
            if a.ndim == 1:
                a = a[None]
            per = self.chunk_seconds / max(a.shape[0], 1)
            for row in a:
                self._pad.set_action(row)
                t = time.perf_counter()
                while time.perf_counter() - t < per:
                    pass
        if self._sh is not None:
            self._sh.pause()

    # ---- GameEnv interface -------------------------------------------------------------
    def reset(self, scenario: Optional[Scenario] = None) -> Observation:
        if self._sh is not None:
            self._sh.unpause()
        macro = self.reset_macro(scenario)
        for entry in macro:
            kind = entry[0]
            if kind == "key":
                self._xdo("key", *(["--window", self._wid] if self._wid else []), entry[1])
            elif kind == "hold":
                self._xdo("keydown", *(["--window", self._wid] if self._wid else []), entry[1])
                time.sleep(float(entry[2]))
                self._xdo("keyup", *(["--window", self._wid] if self._wid else []), entry[1])
            elif kind == "wait":
                time.sleep(float(entry[1]))
        if self.control == "keyboard":
            self._set_keys(set())
        elif self._pad is not None:
            self._pad.neutral()
        if self._sh is not None:
            self._sh.pause()
        self._step = 0
        return Observation(frame=self._grab(), state={"step": 0, **self.read_state()},
                           step_idx=0, done=False)

    def step(self, action_chunk: np.ndarray) -> Observation:
        self._apply(action_chunk)
        self._step += 1
        return Observation(frame=self._grab(), state={"step": self._step, **self.read_state()},
                           step_idx=self._step, done=False)

    def save_frame(self, path: str) -> None:
        from PIL import Image
        Image.fromarray(self._grab()).save(path)

    def close(self) -> None:
        try:
            if self.control == "keyboard":
                self._set_keys(set())
            elif self._pad is not None:
                self._pad.neutral(); self._pad.close()
        except Exception:
            pass
        for proc in (self._game, self._xvfb):
            if proc is not None:
                try:
                    proc.terminate(); proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._game = self._xvfb = None


# ---- keyboard mapping helper (use in action_to_keys) ----------------------------------
def keys_from_dirs_and_buttons(action_chunk, *, steer_thresh=0.25, vert_thresh=0.25,
                               button_map=None, button_frac=0.3):
    """Convenience: turn a NitroGen chunk into a key set using a {button_index: keysym} map plus
    standard arrow-key steering. Returns a set of keysyms. `button_map` e.g. {18:'z', 20:'x'}
    (SOUTH->z jump, WEST->x attack)."""
    from ..core import JLX, JLY
    a = np.asarray(action_chunk, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    keys = set()
    mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
    if mx < -steer_thresh:
        keys.add("Left")
    elif mx > steer_thresh:
        keys.add("Right")
    if my < -vert_thresh:
        keys.add("Up")
    elif my > vert_thresh:
        keys.add("Down")
    for idx, key in (button_map or {}).items():
        if (a[:, idx] > 0.5).mean() >= button_frac:
            keys.add(key)
    return keys
