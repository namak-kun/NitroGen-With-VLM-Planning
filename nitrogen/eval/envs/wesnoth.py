"""The Battle for Wesnoth env.

Launches the built-in GUI test scenario directly into a playable hex-grid battle. Wesnoth supports
keyboard map scrolling and unit cycling, but actual unit movement/attacks are mouse-driven; this env
therefore exposes arrow-key scrolling plus optional xdotool mouse cursor/click actions.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from nitrogen.shared import BUTTON_ACTION_TOKENS
from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY, JRX, JRY

I_RIGHT_TRIGGER = BUTTON_ACTION_TOKENS.index("RIGHT_TRIGGER")
I_SOUTH = BUTTON_ACTION_TOKENS.index("SOUTH")
I_START = BUTTON_ACTION_TOKENS.index("START")
I_WEST = BUTTON_ACTION_TOKENS.index("WEST")

SCROLL_THRESH = 0.2
MOUSE_THRESH = 0.15
BUTTON_FRAC = 0.3


class WesnothEnv(ProcGameEnv):
    name = "wesnoth"
    window_name = "The Battle for Wesnoth"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 18.0, **kw):
        self._mouse_x = width // 2
        self._mouse_y = height // 2
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        exe = shutil.which("wesnoth") or shutil.which("wesnoth-1.16") or "/usr/games/wesnoth"
        return [
            "/usr/bin/env", "LIBGL_ALWAYS_SOFTWARE=1",
            exe, "--windowed", "--resolution", f"{self.width}x{self.height}",
            "--max-fps", "30", "--test",
        ]

    def reset_macro(self, scenario):
        self._mouse_x = self.width // 2
        self._mouse_y = self.height // 2
        return [("key", "Escape"), ("wait", 0.5)]

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]

        keys = set()
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < 0.5 - SCROLL_THRESH:
            keys.add("Left")
        elif mx > 0.5 + SCROLL_THRESH:
            keys.add("Right")
        if my < 0.5 - SCROLL_THRESH:
            keys.add("Up")
        elif my > 0.5 + SCROLL_THRESH:
            keys.add("Down")

        if (a[:, I_RIGHT_TRIGGER] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("n")
        if (a[:, I_START] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("Return")

        self._apply_mouse(a)
        return keys

    def _apply_mouse(self, a: np.ndarray) -> None:
        rx, ry = float(a[:, JRX].mean()), float(a[:, JRY].mean())
        dx, dy = rx - 0.5, ry - 0.5
        moved = abs(dx) > MOUSE_THRESH or abs(dy) > MOUSE_THRESH
        if moved:
            self._mouse_x = int(np.clip(self._mouse_x + dx * 120, 8, self.width - 8))
            self._mouse_y = int(np.clip(self._mouse_y + dy * 120, 8, self.height - 8))
            self._xdo("mousemove", "--sync", str(self._mouse_x), str(self._mouse_y))

        left_click = (a[:, I_SOUTH] > 0.5).mean() >= BUTTON_FRAC
        right_click = (a[:, I_WEST] > 0.5).mean() >= BUTTON_FRAC
        if left_click or right_click:
            if not moved:
                self._xdo("mousemove", "--sync", str(self._mouse_x), str(self._mouse_y))
            self._xdo("click", "1" if left_click else "3")

    def save_frame(self, path: str) -> None:
        if path.startswith(("/tmp/", "/var/tmp/")):
            path = os.path.join(os.getcwd(), "vet_frame_wesnoth.png")
        super().save_frame(path)
