"""bab-be-u (LÖVE/Love2D Baba-Is-You-style puzzle game) as a ProcGameEnv."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nitrogen.shared import BUTTON_ACTION_TOKENS

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

REPO = Path(__file__).resolve().parents[3]
BUILD_ROOT = REPO / ".nitrogen-env-build" / "baba"
LOCAL_BABBEU_DIR = REPO / ".nitrogen-env-build" / "bab-be-u" / "source"
FALLBACK_BABBEU_DIR = Path("/tmp/bab-be-u")

I_BACK = BUTTON_ACTION_TOKENS.index("BACK")
I_EAST = BUTTON_ACTION_TOKENS.index("EAST")
I_NORTH = BUTTON_ACTION_TOKENS.index("NORTH")
I_SOUTH = BUTTON_ACTION_TOKENS.index("SOUTH")
I_START = BUTTON_ACTION_TOKENS.index("START")

STICK_THRESH = 0.25


class BabaEnv(ProcGameEnv):
    name = "baba"
    window_name = "bab be u"
    control = "keyboard"
    window_manager = "matchbox-window-manager"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 14.0, **kw):
        self.game_dir = Path(os.environ.get("BABA_GAME_DIR", self._default_game_dir()))
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

    @staticmethod
    def _default_game_dir() -> Path:
        if LOCAL_BABBEU_DIR.exists():
            return LOCAL_BABBEU_DIR
        return FALLBACK_BABBEU_DIR

    def launch_cmd(self):
        if not (self.game_dir / "main.lua").exists():
            raise FileNotFoundError(
                f"bab-be-u main.lua not found in {self.game_dir}; clone bab-be-u or set BABA_GAME_DIR"
            )
        return [
            "env",
            "SDL_AUDIODRIVER=dummy",
            "LIBGL_ALWAYS_SOFTWARE=1",
            f"HOME={self.home_dir}",
            f"XDG_DATA_HOME={self.data_dir}",
            f"TMPDIR={self.tmp_dir}",
            "love",
            str(self.game_dir),
        ]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_NORTH: "z",   # undo
                I_START: "z",   # legacy/index-19 undo mapping used by older action docs
                I_BACK: "r",    # restart current puzzle
                I_SOUTH: "space",
                I_EAST: "Return",
            },
        )

    def reset_macro(self, scenario):
        return [
            ("wait", 0.25),
            ("key", "Down"),    # main menu: select Play
            ("wait", 0.15),
            ("key", "Return"),
            ("wait", 1.25),
            ("key", "Down"),    # level selector: select first official world
            ("wait", 0.15),
            ("key", "Return"),  # opens its overworld/first puzzle
            ("wait", 4.0),
            ("key", "r"),       # normalize the level after load
            ("wait", 0.4),
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
