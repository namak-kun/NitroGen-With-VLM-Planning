"""Chromium B.S.U. env — fast vertical shmup control via keyboard.

Chromium B.S.U. controls (man page): arrows move the ship, Space fires.
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_RTRIG, I_SOUTH = 16, 18
STICK_THRESH = 0.2
VIDEO_MODES = {
    (512, 384): 0,
    (640, 480): 1,
    (800, 600): 2,
    (1024, 768): 3,
    (1280, 960): 4,
}


class ChromiumBSUEnv(ProcGameEnv):
    name = "chromium_bsu"
    window_name = "Chromium B.S.U."
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 12.0, **kw):
        self.run_dir = Path.cwd() / ".nitrogen-env-build" / "chromium_bsu"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(self.run_dir)
        tempfile.tempdir = str(self.run_dir)
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def launch_cmd(self):
        mode = VIDEO_MODES.get((self.width, self.height), VIDEO_MODES[(800, 600)])
        return [
            "sh",
            "-c",
            "SDL_AUDIODRIVER=dummy LIBGL_ALWAYS_SOFTWARE=1 "
            f"exec /usr/games/chromium-bsu -w -v {mode} -na",
        ]

    def _find_window(self) -> str:
        wid = super()._find_window()
        if wid:
            return wid
        out = subprocess.run(
            ["xdotool", "search", "--class", "chromium-bsu"],
            env=self._env(),
            capture_output=True,
            text=True,
        )
        ids = [line for line in out.stdout.split() if line.strip()]
        return ids[-1] if ids else ""

    def _pause_game_process(self) -> None:
        if self.freeze_during_inference and self._game is not None and self._game.poll() is None:
            os.kill(self._game.pid, signal.SIGSTOP)

    def _resume_game_process(self) -> None:
        if self._game is not None and self._game.poll() is None:
            os.kill(self._game.pid, signal.SIGCONT)

    def reset(self, scenario=None):
        self._resume_game_process()
        obs = super().reset(scenario)
        self._pause_game_process()
        return obs

    def _apply(self, action_chunk: np.ndarray):
        self._resume_game_process()
        super()._apply(action_chunk)
        self._pause_game_process()

    def _grab(self) -> np.ndarray:
        env = dict(os.environ, DISPLAY=f":{self.display}")
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
            env=env,
            capture_output=True,
        )
        buf = p.stdout
        n = self.width * self.height * 3
        if len(buf) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(buf[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    def action_to_keys(self, action_chunk: np.ndarray):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "space",
                I_RTRIG: "space",
            },
            button_frac=0.3,
        )

    def reset_macro(self, scenario):
        return [
            ("wait", 0.3),
            ("key", "Return"),
            ("wait", 1.5),
        ]

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            path = str(self.run_dir / Path(path).name)
        super().save_frame(path)

    def close(self) -> None:
        try:
            self._resume_game_process()
            super().close()
        finally:
            if getattr(self, "_sh", None) is not None:
                self._sh.close()
