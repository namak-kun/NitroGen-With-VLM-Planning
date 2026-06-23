"""SDLPoP (Prince of Persia) env as a thin ProcGameEnv subclass.

Keyboard controls (README.md): arrows move/jump/climb/crouch, Shift grabs/steps/attacks.
"""
from __future__ import annotations

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_DPAD_DOWN, I_DPAD_LEFT, I_DPAD_RIGHT, I_DPAD_UP = 1, 2, 3, 4
I_LSHLD, I_RSHLD = 7, 14
I_SOUTH, I_WEST = 18, 20
STICK_THRESH = 0.25


class SDLPoPEnv(ProcGameEnv):
    name = "sdlpop"
    window_name = "Prince of Persia"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 15.0, **kw):
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        return ["sh", "-c", "cd /tmp/SDLPoP && exec /tmp/SDLPoP/prince megahit 0 mute"]

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]

        keys = set()
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        stick_vals = a[:, 21:25]
        centered_01 = bool(np.any((stick_vals > 0.25) & (stick_vals < 0.75)))

        if centered_01:
            if mx < 0.5 - STICK_THRESH:
                keys.add("Left")
            elif mx > 0.5 + STICK_THRESH:
                keys.add("Right")
            if my < 0.5 - STICK_THRESH:
                keys.add("Up")
            elif my > 0.5 + STICK_THRESH:
                keys.add("Down")
        else:
            if mx < -STICK_THRESH:
                keys.add("Left")
            elif mx > STICK_THRESH:
                keys.add("Right")
            if my < -STICK_THRESH:
                keys.add("Up")
            elif my > STICK_THRESH:
                keys.add("Down")

        if (a[:, I_DPAD_LEFT] > 0.5).mean() >= 0.3:
            keys.add("Left")
        if (a[:, I_DPAD_RIGHT] > 0.5).mean() >= 0.3:
            keys.add("Right")
        if (a[:, I_DPAD_UP] > 0.5).mean() >= 0.3:
            keys.add("Up")
        if (a[:, I_DPAD_DOWN] > 0.5).mean() >= 0.3:
            keys.add("Down")
        if (a[:, I_SOUTH] > 0.5).mean() >= 0.3:
            keys.add("Up")
        if any((a[:, i] > 0.5).mean() >= 0.3 for i in (I_WEST, I_LSHLD, I_RSHLD)):
            keys.add("Shift_L")
        return keys
