"""X-Moto env — 2D side-view motocross physics via SDL/OpenGL.

Launches directly into a built-in tutorial level. Controls are keyboard:
Up=drive, Down=brake, Left/Right=lean, Space=flip/change direction.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_RTRIG, I_LTRIG, I_SOUTH = 16, 9, 18
LEAN_THRESH = 0.2
DRIVE_THRESH = 0.25
BUTTON_FRAC = 0.3

REPO = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO / ".nitrogen-env-build" / "xmoto_runtime"


class XMotoEnv(ProcGameEnv):
    name = "xmoto"
    window_name = "X-Moto"
    control = "keyboard"

    def __init__(
        self,
        level: str = "tut1",
        always_accel: bool = True,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 12.0,
        binary: str = "/usr/games/xmoto",
        **kw,
    ):
        self.level = level
        self.always_accel = always_accel
        self.binary = binary

        self._old_tmpdir_env = os.environ.get("TMPDIR")
        self._old_tempfile_tempdir = tempfile.tempdir
        self.run_dir = RUNTIME_ROOT / str(os.getpid())
        self.config_dir = self.run_dir / "config"
        self.home_dir = self.run_dir / "home"
        self.tmp_dir = self.run_dir / "tmp"
        for path in (self.config_dir, self.home_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(self.tmp_dir)
        tempfile.tempdir = str(self.tmp_dir)

        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        return [
            "env",
            "SDL_AUDIODRIVER=dummy",
            "LIBGL_ALWAYS_SOFTWARE=1",
            f"HOME={self.home_dir}",
            f"TMPDIR={self.tmp_dir}",
            self.binary,
            "--nowww",
            "--nosound",
            "--noLog",
            "--noDBDirsCheck",
            "--configpath",
            str(self.config_dir),
            "-res",
            f"{int(self.width)}x{int(self.height)}",
            "-win",
            "--level",
            self.level,
        ]

    def _find_window(self) -> str:
        wid = super()._find_window()
        if wid:
            return wid
        out = subprocess.run(
            ["xdotool", "search", "--class", "xmoto"],
            env=self._env(),
            capture_output=True,
            text=True,
        )
        ids = [line for line in out.stdout.split() if line.strip()]
        return ids[-1] if ids else ""

    def reset_macro(self, scenario):
        return [("key", "Return"), ("wait", 0.5)]

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]

        keys = set()
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < 0.5 - LEAN_THRESH:
            keys.add("Left")
        elif mx > 0.5 + LEAN_THRESH:
            keys.add("Right")

        drive = (
            self.always_accel
            or my < 0.5 - DRIVE_THRESH
            or (a[:, I_RTRIG] > 0.5).mean() >= BUTTON_FRAC
        )
        brake = my > 0.5 + DRIVE_THRESH or (a[:, I_LTRIG] > 0.5).mean() >= BUTTON_FRAC
        if drive:
            keys.add("Up")
        if brake:
            keys.discard("Up")
            keys.add("Down")
        if (a[:, I_SOUTH] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("space")
        return keys

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            path = str(self.run_dir / Path(path).name)
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
