"""Dust Racing 2D env: top-down kart/tile racer with keyboard steering."""
from __future__ import annotations

import os
import subprocess

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX


I_RTRIG, I_LTRIG = 16, 9
STEER_THRESH = 0.2


class DustRacingEnv(ProcGameEnv):
    name = "dustracing"
    window_name = "Dust Racing"
    control = "keyboard"
    window_manager = "matchbox-window-manager"
    reset_by_relaunch = True             # menu-driven race setup: respawn for a clean reset

    def __init__(self, always_accel: bool = True, width: int = 800, height: int = 600,
                 boot_wait: float = 14.0, **kw):
        self.always_accel = always_accel
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def launch_cmd(self):
        return ["/usr/bin/env", "LIBGL_ALWAYS_SOFTWARE=1", "ALSOFT_DRIVERS=null",
                "/usr/games/dustrac-game", "--no-vsync"]

    def reset_macro(self, scenario):
        return [
            ("wait", 0.5),
            ("key", "Return"),  # main menu PLAY -> difficulty
            ("wait", 0.5),
            ("key", "Return"),  # Easy -> lap count
            ("wait", 0.5),
            ("key", "Return"),  # default 5 laps -> track select
            ("wait", 0.8),
            ("key", "Return"),  # selected Ring track -> race
            ("wait", 6.0),       # start-light countdown
        ]

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys = set()
        mx = float(a[:, JLX].mean())
        if mx < 0.5 - STEER_THRESH:
            keys.add("Left")
        elif mx > 0.5 + STEER_THRESH:
            keys.add("Right")
        if self.always_accel or (a[:, I_RTRIG] > 0.5).mean() >= 0.3:
            keys.add("Up")
        if (a[:, I_LTRIG] > 0.5).mean() >= 0.3:
            keys.discard("Up")
            keys.add("Down")
        return keys

    def _tool_env(self):
        env = {k: v for k, v in os.environ.items()
               if k != "LD_PRELOAD" and not k.startswith("SPEEDHACK_")}
        env["DISPLAY"] = f":{self.display}"
        return env

    def _xdo(self, *args):
        subprocess.run(["xdotool", *args], env=self._tool_env(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _grab(self) -> np.ndarray:
        p = subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-f", "x11grab",
             "-video_size", f"{self.width}x{self.height}", "-i", f":{self.display}.0",
             "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            env=self._tool_env(), capture_output=True)
        buf = p.stdout
        n = self.width * self.height * 3
        if len(buf) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(buf[:n], np.uint8).reshape(self.height, self.width, 3).copy()
