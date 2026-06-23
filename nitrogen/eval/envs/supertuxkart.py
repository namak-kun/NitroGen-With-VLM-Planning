"""SuperTuxKart env as a thin ProcGameEnv subclass (racing — NitroGen's markov strength).

Demonstrates the base-class pattern: ~30 lines of game-specific declarations, everything else
(boot/Xvfb/speedhack/capture/keyboard) inherited. Control VERIFIED (kart accelerates+steers).

STK keyboard map (input.xml): accel=Up, brake=Down, steer=Left/Right, nitro=n, drift=v, fire=space.
Reward/state: --profile-laps stdout is AI-only; policy-mode needs frame-based (HUD/finish) — TODO.
"""
from __future__ import annotations

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX

# NitroGen button indices used for racing
I_RTRIG, I_LTRIG, I_RSHLD, I_LSHLD, I_WEST = 16, 9, 14, 7, 20
STEER_THRESH = 0.2   # stick is [0,1] w/ 0.5 neutral; 'right' = JLX > 0.5+thresh, 'left' = < 0.5-thresh


class SuperTuxKartEnv(ProcGameEnv):
    name = "supertuxkart"
    window_name = "SuperTuxKart"
    control = "keyboard"

    def __init__(self, track: str = "hacienda", ai: int = 2, laps: int = 3,
                 always_accel: bool = True, width: int = 800, height: int = 600,
                 boot_wait: float = 20.0, **kw):
        self.track, self.ai, self.laps, self.always_accel = track, ai, laps, always_accel
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        return ["supertuxkart", "-R", f"--track={self.track}", f"--ai={self.ai}",
                f"--laps={self.laps}", f"--width={self.width}", f"--height={self.height}"]

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
        if (a[:, I_RSHLD] > 0.5).mean() >= 0.3:
            keys.add("n")
        if (a[:, I_LSHLD] > 0.5).mean() >= 0.3:
            keys.add("v")
        if (a[:, I_WEST] > 0.5).mean() >= 0.3:
            keys.add("space")
        return keys
