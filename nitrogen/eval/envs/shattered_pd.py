"""Shattered Pixel Dungeon desktop/libGDX env.

Controls: left stick (JLX/JLY, [0,1] with 0.5 neutral) maps to the game's
default arrow-key movement bindings: Left/Right/Up/Down. SOUTH (index 18)
maps to Space, the default wait/pickup/interact binding. The reset path uses
mouse clicks for the mouse-only title/hero/start menus, then keyboard control
for gameplay.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons
from ..core import Observation, Scenario

I_SOUTH = 18
STICK_THRESH = 0.25

REPO = Path(__file__).resolve().parents[3]
GAME_DIR = REPO / "tmp" / "shattered-pixel-dungeon"
JAR_PATH = GAME_DIR / "desktop" / "build" / "libs" / "desktop-3.3.8.jar"
SCRATCH = REPO / "tmp" / "shattered_pd_runtime"


class ShatteredPDEnv(ProcGameEnv):
    name = "shattered_pd"
    window_name = "Shattered Pixel Dungeon"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 15.0, **kw):
        self._old_tmpdir_env = os.environ.get("TMPDIR")
        self._old_tempfile_tempdir = tempfile.tempdir
        self.run_dir = SCRATCH / str(os.getpid())
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

    def launch_cmd(self):
        return [
            "env",
            "LIBGL_ALWAYS_SOFTWARE=1",
            "ALSOFT_DRIVERS=null",
            f"HOME={self.home_dir}",
            f"XDG_DATA_HOME={self.data_dir}",
            f"TMPDIR={self.tmp_dir}",
            f"JAVA_TOOL_OPTIONS=-Djava.io.tmpdir={self.tmp_dir}",
            "java",
            "-jar",
            str(JAR_PATH),
        ]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={I_SOUTH: "space"},
        )

    def reset_macro(self, scenario: Optional[Scenario]):
        return [
            ("wait", 0.5),
            ("click", 0.50, 0.80),  # welcome/enter
            ("wait", 1.5),
            ("click", 0.07, 0.28),  # Warrior hero button
            ("wait", 0.5),
            ("click", 0.19, 0.90),  # start
            ("wait", 8.0),
            ("key", "space"),
            ("wait", 0.5),
            ("key", "Return"),
            ("wait", 1.0),
        ]

    def _run_reset_entry(self, entry):
        kind = entry[0]
        if kind == "click":
            x = int(float(entry[1]) * self.width)
            y = int(float(entry[2]) * self.height)
            self._xdo("mousemove", str(x), str(y))
            self._xdo("click", "1")
        elif kind == "key":
            self._xdo("key", *(["--window", self._wid] if self._wid else []), entry[1])
        elif kind == "hold":
            self._xdo("keydown", *(["--window", self._wid] if self._wid else []), entry[1])
            time.sleep(float(entry[2]))
            self._xdo("keyup", *(["--window", self._wid] if self._wid else []), entry[1])
        elif kind == "wait":
            time.sleep(float(entry[1]))

    def _wait_for_ready_frame(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._wid:
                self._wid = self._find_window()
            if self._wid and self._grab().std() > 8:
                return
            time.sleep(0.5)

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
        return np.frombuffer(p.stdout[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    def reset(self, scenario: Optional[Scenario] = None) -> Observation:
        if self._sh is not None:
            self._sh.unpause()
        self._wait_for_ready_frame()
        if self._wid:
            self._xdo("windowfocus", "--sync", self._wid)
        for entry in self.reset_macro(scenario):
            self._run_reset_entry(entry)
        self._set_keys(set())
        if self._sh is not None:
            self._sh.pause()
        self._step = 0
        return Observation(
            frame=self._grab(),
            state={"step": 0, **self.read_state()},
            step_idx=0,
            done=False,
        )

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
