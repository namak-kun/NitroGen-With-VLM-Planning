"""CavePacker Sokoban as a ProcGameEnv keyboard puzzle env."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_BACK, I_NORTH = 0, 10
STICK_THRESH = 0.25

REPO = Path(__file__).resolve().parents[3]
BUILD_ROOT = REPO / ".nitrogen-env-build" / "sokoban"
CAVEPACKER = Path("/usr/games/cavepacker")


class SokobanEnv(ProcGameEnv):
    name = "sokoban"
    window_name = "cavepacker"
    control = "keyboard"
    window_manager = "matchbox-window-manager"
    target_keys_to_window = False

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 12.0,
        chunk_seconds: float = 1.0,
        map_name: str = "sasquatch08_0036",
        **kw,
    ):
        self.map_name = map_name
        self.run_dir = BUILD_ROOT / str(os.getpid())
        self.home_dir = self.run_dir / "home"
        self.data_dir = self.run_dir / "xdg"
        self.tmp_dir = self.run_dir / "tmp"
        for path in (self.home_dir, self.data_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._old_tmpdir_env = os.environ.get("TMPDIR")
        self._old_tempfile_tempdir = tempfile.tempdir
        os.environ["TMPDIR"] = str(self.tmp_dir)
        tempfile.tempdir = str(self.tmp_dir)
        super().__init__(
            width=width,
            height=height,
            boot_wait=boot_wait,
            chunk_seconds=chunk_seconds,
            **kw,
        )
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def launch_cmd(self):
        if not CAVEPACKER.exists():
            raise FileNotFoundError(f"CavePacker binary not found: {CAVEPACKER}")
        return [
            "env",
            "SDL_VIDEODRIVER=x11",
            "SDL_AUDIODRIVER=dummy",
            "LIBGL_ALWAYS_SOFTWARE=1",
            f"HOME={self.home_dir}",
            f"XDG_DATA_HOME={self.data_dir}",
            f"TMPDIR={self.tmp_dir}",
            str(CAVEPACKER),
            "-set",
            "fullscreen",
            "false",
            "-set",
            "width",
            str(self.width),
            "-set",
            "height",
            str(self.height),
            "-set",
            "sound",
            "false",
            "-set",
            "grabmouse",
            "false",
            "-set",
            "showfps",
            "false",
            "-set",
            "frontend",
            "sdl",
            "-set",
            "renderer",
            "software",
            "-map",
            self.map_name,
        ]

    def _focus_window(self):
        if self._wid:
            self._xdo("windowactivate", self._wid)
            self._xdo("windowfocus", self._wid)

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_NORTH: "u",
                I_BACK: "BackSpace",
            },
            button_frac=0.3,
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
            env=dict(os.environ, DISPLAY=f":{self.display}"),
            capture_output=True,
        )
        n = self.width * self.height * 3
        if len(p.stdout) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(p.stdout[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            BUILD_ROOT.mkdir(parents=True, exist_ok=True)
            path = str(BUILD_ROOT / Path(path).name)
        super().save_frame(path)

    def close(self) -> None:
        try:
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
