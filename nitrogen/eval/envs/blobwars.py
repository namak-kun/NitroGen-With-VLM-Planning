"""Blobwars: Metal Blob Solid as a ProcGameEnv keyboard platformer."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_RTRIG, I_SOUTH, I_WEST = 16, 18, 20
STICK_THRESH = 0.25
BUTTON_FRAC = 0.3

REPO = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO / ".nitrogen-env-build" / "blobwars_runtime"


class BlobwarsEnv(ProcGameEnv):
    name = "blobwars"
    window_name = "Blobwars: Metal Blob Solid"
    control = "keyboard"

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 12.0,
        binary: str = "blobwars",
        **kw,
    ):
        self.binary = binary

        self._old_tmpdir_env = os.environ.get("TMPDIR")
        self._old_tempfile_tempdir = tempfile.tempdir
        self.run_dir = RUNTIME_ROOT / str(os.getpid())
        self.home_dir = self.run_dir / "home"
        self.cache_dir = self.run_dir / "cache"
        self.tmp_dir = self.run_dir / "tmp"
        for path in (self.home_dir, self.cache_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(self.tmp_dir)
        tempfile.tempdir = str(self.tmp_dir)

        self._stopped = False
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)
        if getattr(self, "_sh", None) is not None:
            self._sh.pause_scale = 0.0

    def launch_cmd(self):
        return [
            "env",
            "SDL_AUDIODRIVER=dummy",
            "SDL_VIDEODRIVER=x11",
            "LIBGL_ALWAYS_SOFTWARE=1",
            f"HOME={self.home_dir}",
            f"XDG_CACHE_HOME={self.cache_dir}",
            f"MESA_SHADER_CACHE_DIR={self.cache_dir / 'mesa_shader_cache'}",
            f"TMPDIR={self.tmp_dir}",
            self.binary,
            "-window",
            "-noaudio",
        ]

    def _tool_env(self) -> dict:
        env = dict(os.environ, DISPLAY=f":{self.display}")
        env.pop("LD_PRELOAD", None)
        env.pop("SPEEDHACK_CTRL", None)
        return env

    def _xdo(self, *args):
        subprocess.run(
            ["xdotool", *args],
            env=self._tool_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _grab(self) -> np.ndarray:
        p = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-f",
                "x11grab",
                "-video_size",
                f"{self.width}x{self.height}",
                "-i",
                f":{self.display}.0",
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            env=self._tool_env(),
            capture_output=True,
        )
        n = self.width * self.height * 3
        if len(p.stdout) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(p.stdout[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    def _freeze_game(self) -> None:
        if self.freeze_during_inference and self._game is not None and self._game.poll() is None:
            os.kill(self._game.pid, signal.SIGSTOP)
            self._stopped = True

    def _unfreeze_game(self) -> None:
        if self._stopped and self._game is not None and self._game.poll() is None:
            os.kill(self._game.pid, signal.SIGCONT)
        self._stopped = False

    def reset(self, scenario=None):
        self._unfreeze_game()
        obs = super().reset(scenario)
        self._freeze_game()
        return obs

    def _unpause_world(self) -> None:
        self._unfreeze_game()

    def _pause_world(self) -> None:
        self._freeze_game()

    def _apply(self, action_chunk):
        self._unfreeze_game()
        try:
            super()._apply(action_chunk)
        finally:
            self._freeze_game()

    def _find_window(self) -> str:
        out = subprocess.run(
            ["xdotool", "search", "--name", self.window_name],
            env=self._tool_env(),
            capture_output=True,
            text=True,
        )
        ids = [line for line in out.stdout.split() if line.strip()]
        if ids:
            return ids[-1]
        out = subprocess.run(
            ["xdotool", "search", "--class", "blobwars"],
            env=self._tool_env(),
            capture_output=True,
            text=True,
        )
        ids = [line for line in out.stdout.split() if line.strip()]
        return ids[-1] if ids else ""

    def reset_macro(self, scenario):
        return [
            ("wait", 0.5),
            ("hold", "space", 0.35),    # Start New Game
            ("wait", 2.0),
            ("hold", "space", 0.35),    # Practice
            ("wait", 3.0),
            ("hold", "space", 0.35),    # dismiss brief/loading screen into gameplay
            ("wait", 4.0),
        ]

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]

        keys: set[str] = set()
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < 0.5 - STICK_THRESH:
            keys.add("Left")
        elif mx > 0.5 + STICK_THRESH:
            keys.add("Right")
        if my > 0.5 + STICK_THRESH:
            keys.add("Down")

        def pressed(idx: int) -> bool:
            return bool((a[:, idx] > 0.5).mean() >= BUTTON_FRAC)

        if pressed(I_SOUTH):
            keys.add("Up")
        if pressed(I_WEST) or pressed(I_RTRIG):
            keys.add("Control_L")
        return keys

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            path = str(self.run_dir / Path(path).name)
        super().save_frame(path)

    def close(self) -> None:
        try:
            self._unfreeze_game()
            super().close()
        finally:
            if getattr(self, "_sh", None) is not None:
                self._sh.close()
            if self._old_tmpdir_env is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = self._old_tmpdir_env
            tempfile.tempdir = self._old_tempfile_tempdir
            shutil.rmtree(self.run_dir, ignore_errors=True)
