"""Megaman Maverick desktop/libGDX env.

The env launches the source-built LWJGL3 jar in the game's test mode, which
boots directly into a playable room and avoids title/story menu navigation.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_EAST, I_LSHLD, I_LTRIG, I_NORTH = 5, 7, 9, 10
I_RSHLD, I_RTRIG, I_SOUTH, I_START, I_WEST = 14, 16, 18, 19, 20
STICK_THRESH = 0.25
BUTTON_FRAC = 0.3

REPO = Path(__file__).resolve().parents[3]
GAME_DIR = REPO / ".nitrogen-env-build" / "megaman-maverick"
RUNTIME_ROOT = REPO / ".nitrogen-env-build" / "megaman_maverick_runtime"


class MegamanMaverickEnv(ProcGameEnv):
    name = "megaman_maverick"
    window_name = "Megaman Maverick"
    control = "keyboard"

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 16.0,
        game_dir: str | None = None,
        jar_path: str | None = None,
        **kw,
    ):
        self.game_dir = Path(game_dir) if game_dir else GAME_DIR
        self.assets_dir = self.game_dir / "assets"
        self.jar_path = Path(jar_path) if jar_path else self._find_jar()

        self._old_tmpdir_env = os.environ.get("TMPDIR")
        self._old_tempfile_tempdir = tempfile.tempdir
        self.run_dir = RUNTIME_ROOT / str(os.getpid())
        self.home_dir = self.run_dir / "home"
        self.data_dir = self.run_dir / "xdg"
        self.tmp_dir = self.run_dir / "tmp"
        for path in (self.home_dir, self.data_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(self.tmp_dir)
        tempfile.tempdir = str(self.tmp_dir)

        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def _find_jar(self) -> Path:
        libs = self.game_dir / "lwjgl3" / "build" / "libs"
        candidates = sorted(libs.glob("Megaman-Maverick-*.jar"))
        if candidates:
            return candidates[-1]
        return libs / "Megaman-Maverick-alpha-1.14.2.jar"

    def launch_cmd(self):
        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"Megaman Maverick jar not found: {self.jar_path}. "
                f"Build it with: cd {self.game_dir} && ./gradlew lwjgl3:dist"
            )
        if not self.assets_dir.exists():
            raise FileNotFoundError(f"Megaman Maverick assets dir not found: {self.assets_dir}")

        env = [
            "LIBGL_ALWAYS_SOFTWARE=1",
            "ALSOFT_DRIVERS=null",
            f"HOME={shlex.quote(str(self.home_dir))}",
            f"XDG_DATA_HOME={shlex.quote(str(self.data_dir))}",
            f"TMPDIR={shlex.quote(str(self.tmp_dir))}",
            f"JAVA_TOOL_OPTIONS={shlex.quote('-Djava.io.tmpdir=' + str(self.tmp_dir))}",
        ]
        cmd = (
            f"cd {shlex.quote(str(self.assets_dir))} && "
            f"exec env {' '.join(env)} java -jar {shlex.quote(str(self.jar_path))} "
            f"--runType test --width {int(self.width)} --height {int(self.height)} "
            "--performance low --musicVolume 0 --soundVolume 0"
        )
        return ["bash", "-lc", cmd]

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys = set()

        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < 0.5 - STICK_THRESH:
            keys.add("a")
        elif mx > 0.5 + STICK_THRESH:
            keys.add("d")
        if my < 0.5 - STICK_THRESH:
            keys.add("w")
        elif my > 0.5 + STICK_THRESH:
            keys.add("s")

        def pressed(idx: int) -> bool:
            return (a[:, idx] > 0.5).mean() >= BUTTON_FRAC

        if pressed(I_SOUTH) or pressed(I_EAST) or pressed(I_LTRIG):
            keys.add("k")       # in-game A: jump / slide / air-dash
        if pressed(I_WEST) or pressed(I_RTRIG):
            keys.add("j")       # in-game B: buster / weapon fire
        if pressed(I_NORTH) or pressed(I_RSHLD) or pressed(I_LSHLD):
            keys.add("l")       # in-game SELECT: next weapon
        if pressed(I_START):
            keys.add("Return")
        return keys

    def _tool_env(self) -> dict:
        return dict(os.environ, DISPLAY=f":{self.display}")

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
        frame = np.frombuffer(p.stdout[:n], np.uint8).reshape(self.height, self.width, 3)
        y0, y1 = int(self.height * 0.58), int(self.height * 0.92)
        x0, x1 = int(self.width * 0.19), int(self.width * 0.50)
        return frame[y0:y1, x0:x1].copy()

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
            path = str(RUNTIME_ROOT / Path(path).name)
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
