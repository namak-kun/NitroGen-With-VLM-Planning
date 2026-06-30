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
                 binary: str | None = None,
                 asset_dir: str | None = None,
                 user_dir: str | None = None,
                 level: str = "worlds/the first adventure/bonus1.lvlx", **kw):
        build_root = REPO / ".nitrogen-env-build" / "TheXTech"
        self.binary = binary or str(build_root / "build" / "output" / "bin" / "thextech")
        self.asset_dir = asset_dir or str(
            build_root / "aod-assets" / "thextech-adventure-of-demo-assets-full-v1.3.7.2"
        )
        self.user_dir = user_dir or str(build_root / "nitrogen-user-aod")
        self.level = level
        # Ground-truth state is read from the live process memory (no source fork) — see
        # thextech_memread.TheXTechMemReader. Requires a NON-STRIPPED build (RelWithDebInfo). The reader
        # resolves the global symbol addresses from `binary` at construction; we (re)attach it to the
        # game PID after each (re)launch. Falls back to {} if symbols/permissions are unavailable.
        self._mem_reader = None
        try:
            from .thextech_memread import TheXTechMemReader
            self._mem_reader = TheXTechMemReader(self.binary)
        except Exception:
            self._mem_reader = None
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        level_path = self.level if self.level.startswith("/") else f"{self.asset_dir}/{self.level}"
        return [
            "env",
            "SDL_AUDIODRIVER=dummy",
            "SDL_VIDEODRIVER=x11",
            f"LD_LIBRARY_PATH={REPO / '.nitrogen-env-build' / 'TheXTech' / 'build' / 'output' / 'lib'}",
            self.binary,
            "-s",                 # no sound
            "-p",                 # keep running if focus changes
            "-r", "sw",           # robust under Xvfb
            "-u", self.user_dir,
            "-c", self.asset_dir,
            "-l", level_path,
        ]

    def _attach_mem_reader(self):
        """(Re)point the memory reader at the current game process. Called after boot + each relaunch."""
        if self._mem_reader is not None and self._game is not None:
            try:
                self._mem_reader.attach(self._game.pid)
            except Exception:
                pass

    def boot(self):
        super().boot()
        self._attach_mem_reader()

    def close(self) -> None:
        if self._mem_reader is not None:
            self._mem_reader.detach()
        super().close()

    # max time to wait for a warm relaunch to reach the loaded level before giving up (we poll and
    # return as soon as the state export shows the level is live, so this is just an upper bound)
    restart_wait: float = 5.0

    def reset(self, scenario=None) -> Observation:
        """Robustly restart the level by RELAUNCHING the game process (keeping Xvfb/WM up).

        TheXTech in level-test mode shows a 6-item menu that WRAPS and starts at an unknown cursor
        position, so menu-navigation restart is unreliable. Respawning thextech always lands at a clean
        level start. We poll read_state() and return as soon as the level is live (warm relaunch is
        typically ~1.5-2.5s), so reset is as fast as the engine allows."""
        try:
            if self._game is not None:
                self._game.terminate()
                try:
                    self._game.wait(timeout=3)
                except Exception:
                    self._game.kill()
        except Exception:
            pass
        if self._mem_reader is not None:
            self._mem_reader.detach()                    # old PID is gone
        if self._sh is not None:
            self._sh.unpause()                           # run at normal speed while the level loads
        self._game = subprocess.Popen(self.launch_cmd(), env=self._env(),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._attach_mem_reader()                        # point the reader at the fresh PID
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
        """Ground-truth player state read from the live process memory (no source fork; y increases
        DOWNWARD). in_menu/dead let the harness label deaths + menu-stranding without guessing from
        pixels. beat_code: POSITIVE = beaten (3=offscreen exit, 7=star, 8=goal-tape, 9=flag); 0=none;
        NEGATIVE (-1 quit/-2 restart/-3 setup) = a menu selection, NOT a win. Returns {} if the reader
        is unavailable (e.g. stripped binary or process not yet mapped)."""
        if self._mem_reader is None:
            return {}
        return self._mem_reader.read()

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
