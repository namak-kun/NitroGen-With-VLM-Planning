"""TheXTech (Adventures of Demo asset pack) as a ProcGameEnv keyboard platformer.

Ground-truth STATE TRACKING: TheXTech is open source, so we patched src/graphics/gfx_update.cpp
(UpdateGraphics) to export Player[1] position/velocity, Lives, Dead, GameMenu/LevelSelect/GamePaused
to the file in env var THEXTECH_STATE_EXPORT every frame. read_state() reads it, so the eval harness
annotates VERIFIED game state (position, lives, death, menu) instead of guessing from pixels. The
patch is saved at docs/env_candidates/thextech_state_export.patch; apply it to the TheXTech source and
`make thextech` to rebuild the instrumented binary.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY, Observation

I_RTRIG, I_SOUTH, I_START, I_WEST = 16, 18, 19, 20
STICK_THRESH = 0.25
REPO = Path(__file__).resolve().parents[3]
GET_FLOWER_GO_RIGHT_LEVEL = str(REPO / "docs" / "env_candidates" / "thextech_minimal_get_flower_go_right.lvlx")


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
        # ground-truth state export (THEXTECH_STATE_EXPORT, written each frame by the patched
        # UpdateGraphics in src/graphics/gfx_update.cpp): X Y SpeedX SpeedY Lives Dead GameMenu
        # LevelSelect numPlayers.
        self._state_path = f"/tmp/thextech_state_{id(self)}.txt"
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        level_path = self.level if self.level.startswith("/") else f"{self.asset_dir}/{self.level}"
        return [
            "env",
            "SDL_AUDIODRIVER=dummy",
            "SDL_VIDEODRIVER=x11",
            "LD_LIBRARY_PATH=/tmp/TheXTech/build/output/lib",
            f"THEXTECH_STATE_EXPORT={self._state_path}",
            self.binary,
            "-s",                 # no sound
            "-p",                 # keep running if focus changes
            "-r", "sw",           # robust under Xvfb
            "-u", self.user_dir,
            "-c", self.asset_dir,
            "-l", level_path,
        ]

    # max time to wait for a warm relaunch to reach the loaded level before giving up (we poll and
    # return as soon as the state export shows the level is live, so this is just an upper bound)
    restart_wait: float = 5.0

    def reset(self, scenario=None) -> Observation:
        """Robustly restart the level by RELAUNCHING the game process (keeping Xvfb/WM up).

        TheXTech in level-test mode shows a 6-item menu that WRAPS and starts at an unknown cursor
        position, so menu-navigation restart is unreliable. Respawning thextech always lands at a clean
        level start. We poll the state export and return as soon as the level is live (warm relaunch is
        typically ~1.5-2.5s), so reset is as fast as the engine allows."""
        import os
        try:
            if self._game is not None:
                self._game.terminate()
                try:
                    self._game.wait(timeout=3)
                except Exception:
                    self._game.kill()
        except Exception:
            pass
        try:
            if os.path.exists(self._state_path):
                os.remove(self._state_path)              # so we can detect the FRESH level coming up
        except Exception:
            pass
        if self._sh is not None:
            self._sh.unpause()                           # run at normal speed while the level loads
        self._game = subprocess.Popen(self.launch_cmd(), env=self._env(),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + self.restart_wait
        while time.time() < deadline:                    # return as soon as the level is live
            time.sleep(0.1)
            st = self.read_state()
            if st and not st.get("in_menu") and st.get("y", 0) > 0:
                break
        if self.control == "keyboard":
            self._wid = self._find_window()
            if self.window_manager and self._wid:
                self._focus_window()
            self._set_keys(set())
        if self._sh is not None:
            self._sh.pause()
        self._step = 0
        return Observation(frame=self._grab(), state={"step": 0, **self.read_state()},
                           step_idx=0, done=False)

    def read_state(self) -> dict:
        """Ground-truth player state from the source-patched export file (y increases DOWNWARD).
        in_menu/dead let the harness label deaths + menu-stranding without guessing from pixels."""
        try:
            with open(self._state_path) as f:
                p = f.read().split()
            x, y, sx, sy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            lives, dead, game_menu, level_select, nplayers = (int(p[4]), int(p[5]), int(p[6]),
                                                              int(p[7]), int(p[8]))
            paused = int(p[9]) if len(p) > 9 else 0    # GamePaused (!=0 => pause/test menu)
            end_level = int(p[10]) if len(p) > 10 else 0   # EndLevel: the level-end transition is running
            beat_code = int(p[11]) if len(p) > 11 else 0   # LevelBeatCode: POSITIVE = beaten (3=offscreen,
            # 7=star, 8=goal-tape, 9=flag, ...); 0=none; NEGATIVE (-1 quit/-2 restart/-3 setup) = a menu
            # selection, NOT a win.
            return {"x": round(x, 1), "y": round(y, 1), "vx": round(sx, 2), "vy": round(sy, 2),
                    "lives": lives, "dead": dead,
                    "in_menu": int(bool(game_menu or level_select or paused)),
                    "won": int(end_level == 1 and beat_code > 0), "beat_code": beat_code,
                    "nplayers": nplayers}
        except Exception:
            return {}

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


class TheXTechGetFlowerEnv(TheXTechEnv):
    """Tiny authored RL task: grab the flower/powerup and run right to the offscreen exit."""

    name = "thextech_get_flower"

    def __init__(self, level: str = GET_FLOWER_GO_RIGHT_LEVEL, **kw):
        super().__init__(level=level, **kw)


__all__ = ["TheXTechEnv", "TheXTechGetFlowerEnv", "GET_FLOWER_GO_RIGHT_LEVEL"]
