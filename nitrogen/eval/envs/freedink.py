"""FreeDink / Dink Smallwood SDL2 top-down adventure env."""
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

I_EAST, I_LTRIG, I_RTRIG, I_SOUTH, I_START, I_WEST = 5, 9, 16, 18, 19, 20
STICK_THRESH = 0.25
BUTTON_FRAC = 0.3

REPO = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO / ".nitrogen-env-build" / "freedink" / "runs"


class FreeDinkEnv(ProcGameEnv):
    name = "freedink"
    window_name = "GNU FreeDink"
    control = "keyboard"
    window_manager = "matchbox-window-manager"
    target_keys_to_window = False

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 14.0, **kw):
        self.binary = shutil.which("freedink") or "/usr/games/freedink"
        if not Path(self.binary).exists():
            raise FileNotFoundError("freedink not found; install the freedink package")

        self._old_tmpdir_env = os.environ.get("TMPDIR")
        self._old_tempfile_tempdir = tempfile.tempdir
        self.run_dir = RUNS_ROOT / str(os.getpid())
        self.home_dir = self.run_dir / "home"
        self.xdg_config_dir = self.run_dir / "xdg_config"
        self.xdg_data_dir = self.run_dir / "xdg_data"
        self.tmp_dir = self.run_dir / "tmp"
        for path in (self.home_dir, self.xdg_config_dir, self.xdg_data_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(self.tmp_dir)
        tempfile.tempdir = str(self.tmp_dir)

        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def _env(self) -> dict:
        env = super()._env()
        env.update(
            {
                "SDL_VIDEODRIVER": "x11",
                "SDL_AUDIODRIVER": "dummy",
                "LIBGL_ALWAYS_SOFTWARE": "1",
                "HOME": str(self.home_dir),
                "XDG_CONFIG_HOME": str(self.xdg_config_dir),
                "XDG_DATA_HOME": str(self.xdg_data_dir),
                "TMPDIR": str(self.tmp_dir),
            }
        )
        return env

    def launch_cmd(self):
        return [self.binary, "--window", "--nojoy", "--nosound", "--software-rendering"]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "ctrl",    # attack / use equipped item
                I_WEST: "space",    # talk / examine / manipulate
                I_LTRIG: "space",
                I_EAST: "Return",   # inventory / equip screen
                I_RTRIG: "Return",  # vet_env strong action visibly opens inventory
                I_START: "Return",
            },
            button_frac=BUTTON_FRAC,
        )

    def reset_macro(self, scenario: Optional[Scenario]):
        return [
            ("wait", 0.3),
            ("click", 0.0625, 0.083),  # title screen Start
            ("wait", 6.5),
            ("key", "Return"),
            ("wait", 4.0),
        ]

    def reset(self, scenario: Optional[Scenario] = None) -> Observation:
        if self._sh is not None:
            self._sh.unpause()
        self._wid = self._wid or self._find_window()
        if self._wid:
            self._focus_window()
        self._set_keys(set())
        for entry in self.reset_macro(scenario):
            kind = entry[0]
            if kind == "click":
                x = str(int(round(float(entry[1]) * self.width)))
                y = str(int(round(float(entry[2]) * self.height)))
                self._xdo("mousemove", x, y)
                time.sleep(0.1)
                self._xdo("click", "1")
            elif kind == "key":
                self._xdo("key", *self._key_target_args(), entry[1])
            elif kind == "hold":
                self._xdo("keydown", *self._key_target_args(), entry[1])
                time.sleep(float(entry[2]))
                self._xdo("keyup", *self._key_target_args(), entry[1])
            elif kind == "wait":
                time.sleep(float(entry[1]))
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

    def _tool_env(self) -> dict:
        env = self._env()
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
