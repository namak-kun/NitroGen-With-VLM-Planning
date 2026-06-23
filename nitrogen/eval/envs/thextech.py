"""TheXTech (Adventures of Demo asset pack) as a ProcGameEnv keyboard platformer."""
from __future__ import annotations

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_RTRIG, I_SOUTH, I_START, I_WEST = 16, 18, 19, 20
STICK_THRESH = 0.25


class TheXTechEnv(ProcGameEnv):
    name = "thextech"
    window_name = "Adventures of Demo"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 15.0,
                 binary: str = "/tmp/TheXTech/build/output/bin/thextech",
                 asset_dir: str = "/tmp/TheXTech/aod-assets/usr/share/games/TheXTech/aod",
                 user_dir: str = "/tmp/TheXTech/nitrogen-user-aod",
                 level: str = "worlds/the first adventure/bonus1.lvlx", **kw):
        self.binary = binary
        self.asset_dir = asset_dir
        self.user_dir = user_dir
        self.level = level
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        level_path = self.level if self.level.startswith("/") else f"{self.asset_dir}/{self.level}"
        return [
            "env",
            "SDL_AUDIODRIVER=dummy",
            "SDL_VIDEODRIVER=x11",
            "LD_LIBRARY_PATH=/tmp/TheXTech/build/output/lib",
            self.binary,
            "-s",                 # no sound
            "-p",                 # keep running if focus changes
            "-r", "sw",           # robust under Xvfb
            "-u", self.user_dir,
            "-c", self.asset_dir,
            "-l", level_path,
        ]

    @staticmethod
    def _axis(values: np.ndarray, raw_sticks: bool) -> float:
        mean = float(values.mean())
        if raw_sticks:
            return mean
        if 0.0 <= mean <= 1.0:
            return (mean - 0.5) * 2.0
        return mean

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]

        sticks = a[:, [JLX, JLY]]
        raw_sticks = bool(np.any(sticks < 0.0) or np.any(sticks > 1.0) or
                          np.allclose(sticks, 0.0))
        mx = self._axis(a[:, JLX], raw_sticks)
        my = self._axis(a[:, JLY], raw_sticks)

        keys = set()
        if mx < -STICK_THRESH:
            keys.add("Left")
        elif mx > STICK_THRESH:
            keys.add("Right")
        if my < -STICK_THRESH:
            keys.add("Up")
        elif my > STICK_THRESH:
            keys.add("Down")

        if (a[:, I_SOUTH] > 0.5).mean() >= 0.3:
            keys.add("z")          # Jump
        if (a[:, I_RTRIG] > 0.5).mean() >= 0.3 or (a[:, I_WEST] > 0.5).mean() >= 0.3:
            keys.add("x")          # Run / hold item
        if (a[:, I_START] > 0.5).mean() >= 0.3:
            keys.add("Return")
        return keys
