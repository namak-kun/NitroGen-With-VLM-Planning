"""Mr. Rescue (LÖVE/Love2D) as a ProcGameEnv keyboard action platformer."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_EAST, I_RTRIG, I_SOUTH, I_WEST = 5, 16, 18, 20
STICK_THRESH = 0.25

REPO = Path(__file__).resolve().parents[3]
BUILD_ROOT = REPO / ".nitrogen-env-build" / "mrrescue"
MRRESCUE_LOVE = Path("/usr/share/games/mrrescue/mrrescue.love")


class MrRescueEnv(ProcGameEnv):
    name = "mrrescue"
    window_name = "Mr. Rescue"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 12.0, **kw):
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
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def launch_cmd(self):
        if not MRRESCUE_LOVE.exists():
            raise FileNotFoundError(f"Mr. Rescue .love file not found: {MRRESCUE_LOVE}")
        return [
            "env",
            "SDL_AUDIODRIVER=dummy",
            "LIBGL_ALWAYS_SOFTWARE=1",
            f"HOME={self.home_dir}",
            f"XDG_DATA_HOME={self.data_dir}",
            f"TMPDIR={self.tmp_dir}",
            "love",
            str(MRRESCUE_LOVE),
        ]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "s",   # jump
                I_WEST: "d",    # water gun
                I_RTRIG: "d",   # water gun, matching RT-heavy base-DiT actions
                I_EAST: "a",    # rescue/throw/action
            },
        )

    def reset_macro(self, scenario):
        return [
            ("wait", 0.2),
            ("key", "Return"),  # splash -> main menu
            ("wait", 0.35),
            ("key", "Return"),  # start game -> level select
            ("wait", 0.35),
            ("key", "Return"),  # select first building
            ("wait", 5.5),
        ]

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
