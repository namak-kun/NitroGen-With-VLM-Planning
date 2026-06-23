"""Pekka Kana 2 as a ProcGameEnv keyboard platformer."""
from __future__ import annotations

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_DPAD_DOWN, I_DPAD_LEFT, I_DPAD_RIGHT, I_DPAD_UP = 1, 2, 3, 4
I_RTRIG, I_SOUTH, I_START, I_WEST = 16, 18, 19, 20
STICK_THRESH = 0.25
BUTTON_FRAC = 0.3


class PekkaKana2Env(ProcGameEnv):
    name = "pekka_kana_2"
    window_name = "Pekka Kana 2"
    control = "keyboard"

    def __init__(self, level: str = "rooster-island-1/level001.map",
                 width: int = 800, height: int = 600, boot_wait: float = 12.0, **kw):
        self.level = level
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        return [
            "env",
            "SDL_AUDIODRIVER=dummy",
            "SDL_VIDEODRIVER=x11",
            "SDL_RENDER_DRIVER=software",
            "pekka-kana-2",
            "path", "/usr/share/games/pekka-kana-2/data",
            "test", self.level,
        ]

    @staticmethod
    def _pressed(a: np.ndarray, idx: int) -> bool:
        return bool((a[:, idx] > 0.5).mean() >= BUTTON_FRAC)

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]

        keys = set()
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        left = mx < 0.5 - STICK_THRESH or self._pressed(a, I_DPAD_LEFT)
        right = mx > 0.5 + STICK_THRESH or self._pressed(a, I_DPAD_RIGHT)
        up = my < 0.5 - STICK_THRESH or self._pressed(a, I_DPAD_UP)
        down = my > 0.5 + STICK_THRESH or self._pressed(a, I_DPAD_DOWN)

        if left and not right:
            keys.add("Left")
        elif right and not left:
            keys.add("Right")
        if down and not up:
            keys.add("Down")

        if up or self._pressed(a, I_SOUTH):
            keys.add("Up")           # Jump
        if self._pressed(a, I_WEST):
            keys.add("Shift_R")      # Doodle attack
        if self._pressed(a, I_RTRIG):
            keys.add("Control_R")    # Egg attack
        if self._pressed(a, I_START):
            keys.add("Escape")
        return keys

    def reset_macro(self, scenario):
        return []
