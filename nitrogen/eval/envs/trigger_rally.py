"""Trigger Rally env as a thin ProcGameEnv subclass (3D rally racing).

Trigger Rally's menus are mouse-driven, then the race itself uses the default
keyboard map: accel=Up, brake=Down, steer=Left/Right, handbrake=Space.
"""
from __future__ import annotations

import shutil
import subprocess
import time

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, Observation

# NitroGen button indices used for racing
I_RTRIG, I_LTRIG, I_WEST = 16, 9, 20
STEER_THRESH = 0.2   # stick is [0,1] w/ 0.5 neutral


class TriggerRallyEnv(ProcGameEnv):
    name = "trigger_rally"
    window_name = "Trigger Rally"
    control = "keyboard"
    window_manager = "matchbox-window-manager"
    target_keys_to_window = False

    def __init__(self, always_accel: bool = True, width: int = 800, height: int = 600,
                 boot_wait: float = 14.0, **kw):
        self.always_accel = always_accel
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def _env(self) -> dict:
        e = super()._env()
        e.setdefault("SDL_VIDEODRIVER", "x11")
        e.setdefault("SDL_AUDIODRIVER", "dummy")
        e.setdefault("ALSOFT_DRIVERS", "null")
        e.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        return e

    def launch_cmd(self):
        return [shutil.which("trigger-rally") or shutil.which("trigger") or "/usr/games/trigger-rally"]

    def _helper_env(self) -> dict:
        e = super()._env()
        e.pop("LD_PRELOAD", None)
        e.pop("SPEEDHACK_CTRL", None)
        return e

    def _xdo(self, *args):
        subprocess.run(["xdotool", *args], env=self._helper_env(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _grab(self) -> np.ndarray:
        p = subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-f", "x11grab",
             "-video_size", f"{self.width}x{self.height}", "-i", f":{self.display}.0",
             "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            env=self._helper_env(), capture_output=True)
        buf = p.stdout
        n = self.width * self.height * 3
        if len(buf) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(buf[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys = set()
        mx = float(a[:, JLX].mean())          # [0,1], 0.5 = straight
        if mx < 0.5 - STEER_THRESH:
            keys.add("Left")
        elif mx > 0.5 + STEER_THRESH:
            keys.add("Right")
        if self.always_accel or (a[:, I_RTRIG] > 0.5).mean() >= 0.3:
            keys.add("Up")
        if (a[:, I_LTRIG] > 0.5).mean() >= 0.3:
            keys.discard("Up"); keys.add("Down")
        if (a[:, I_WEST] > 0.5).mean() >= 0.3:
            keys.add("space")
        return keys

    def reset_macro(self, scenario):
        # Coordinates are fractions of the 800x600-style game window:
        # main menu Practice -> first event -> first race -> Race -> default car.
        return [
            ("click", 0.50, 0.50), ("wait", 1.2),
            ("click", 0.26, 0.28), ("wait", 1.2),
            ("click", 0.23, 0.28), ("wait", 1.7),
            ("click", 0.92, 0.94), ("wait", 1.5),
            ("key", "Return"), ("wait", 4.0),
        ]

    def reset(self, scenario=None) -> Observation:
        if self._sh is not None:
            self._sh.unpause()
        if self.control == "keyboard":
            self._set_keys(set())
            self._focus_window()
        for entry in self.reset_macro(scenario):
            kind = entry[0]
            if kind == "click":
                x = str(int(round(float(entry[1]) * self.width)))
                y = str(int(round(float(entry[2]) * self.height)))
                self._xdo("mousemove", x, y)
                time.sleep(0.1)
                self._xdo("click", "1")
            elif kind == "key":
                self._xdo("key", *self._key_target_args(), entry[1])
            elif kind == "hold":
                self._xdo("keydown", *self._key_target_args(), entry[1])
                time.sleep(float(entry[2]))
                self._xdo("keyup", *self._key_target_args(), entry[1])
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
