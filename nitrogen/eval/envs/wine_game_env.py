"""Reusable Wine-backed Windows game envs."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from nitrogen.shared import BUTTON_ACTION_TOKENS

from ..core import JLX, JLY
from .proc_game_env import ProcGameEnv

I_RIGHT_TRIGGER = BUTTON_ACTION_TOKENS.index("RIGHT_TRIGGER")
I_SOUTH = BUTTON_ACTION_TOKENS.index("SOUTH")
I_START = BUTTON_ACTION_TOKENS.index("START")
I_WEST = BUTTON_ACTION_TOKENS.index("WEST")

STICK_THRESH = 0.2
BUTTON_FRAC = 0.3
DEFAULT_BUTTON_KEY_MAP = {
    I_SOUTH: "z",          # jump
    I_WEST: "x",           # shoot
    I_RIGHT_TRIGGER: "x",  # shoot
    I_START: "Return",
}


class WineGameEnv(ProcGameEnv):
    """Generic process env for Windows games launched under system Wine."""

    name = "wine_game"
    control = "keyboard"
    target_keys_to_window = True

    def __init__(
        self,
        exe_path: str | os.PathLike[str],
        wineprefix: str | os.PathLike[str] | None = None,
        *,
        winearch: str = "win64",
        wine_binary: str = "wine",
        wineboot_binary: str = "wineboot",
        button_key_map: Mapping[int, str] | None = None,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 15.0,
        **kw,
    ):
        self.exe_path = str(Path(exe_path).expanduser())
        self.wineprefix = str(
            Path(wineprefix or f"/tmp/nitrogen-wine-game-prefix-{os.getpid()}").expanduser()
        )
        self.winearch = winearch
        self.wine_binary = wine_binary
        self.wineboot_binary = wineboot_binary
        self.button_key_map = dict(button_key_map or DEFAULT_BUTTON_KEY_MAP)
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def _wine_vars(self) -> dict[str, str]:
        return {
            "WINEPREFIX": self.wineprefix,
            "WINEARCH": self.winearch,
            "WINEDLLOVERRIDES": "mscoree,mshtml=",
            "WINEDEBUG": "-all",
        }

    def _env(self) -> dict:
        env = super()._env()
        env.update(self._wine_vars())
        return env

    def _tool_env(self) -> dict:
        env = dict(os.environ, DISPLAY=f":{self.display}")
        env.update(self._wine_vars())
        env.pop("LD_PRELOAD", None)
        env.pop("SPEEDHACK_CTRL", None)
        return env

    def _wineprefix_initialized(self) -> bool:
        prefix = Path(self.wineprefix)
        return (prefix / "system.reg").exists() and (prefix / "user.reg").exists()

    def _ensure_wineprefix(self) -> None:
        Path(self.wineprefix).mkdir(parents=True, exist_ok=True)
        if self._wineprefix_initialized():
            return
        subprocess.run(
            [self.wineboot_binary, "-i"],
            env=self._tool_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )

    def boot(self):
        self._xvfb = subprocess.Popen(
            ["Xvfb", f":{self.display}", "-screen", "0", f"{self.width}x{self.height}x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)
        self._ensure_wineprefix()
        if self.window_manager:
            self._wm = subprocess.Popen(
                self.window_manager.split(),
                env=self._tool_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.5)
        if self.control == "gamepad":
            from .virtual_gamepad import VirtualGamepad

            self._pad = VirtualGamepad()
            time.sleep(0.3)
            subprocess.run(
                "sudo chmod 666 /dev/input/event* /dev/input/js* 2>/dev/null",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if self.freeze_during_inference:
            from ..speedhack import SpeedHack

            self._sh = SpeedHack()
        self._game = subprocess.Popen(
            self.launch_cmd(),
            env=self._env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(self.boot_wait)
        if self.control == "keyboard":
            self._wid = self._find_window()
            if self.window_manager and self._wid:
                self._focus_window()
        if self._sh is not None:
            self._sh.pause_scale = 0.0
            self._sh.pause()

    def launch_cmd(self) -> list[str]:
        return [self.wine_binary, self.exe_path]

    def _find_window(self) -> str:
        if not self.window_name:
            return ""
        for flag in ("--name", "--class"):
            out = subprocess.run(
                ["xdotool", "search", flag, self.window_name],
                env=self._tool_env(),
                capture_output=True,
                text=True,
            )
            ids = [line for line in out.stdout.split() if line.strip()]
            if ids:
                return ids[-1]
        return ""

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
        buf = p.stdout
        n = self.width * self.height * 3
        if len(buf) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(buf[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    def action_to_keys(self, action_chunk) -> set[str]:
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys = set()
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < 0.5 - STICK_THRESH:
            keys.add("Left")
        elif mx > 0.5 + STICK_THRESH:
            keys.add("Right")
        if my < 0.5 - STICK_THRESH:
            keys.add("Up")
        elif my > 0.5 + STICK_THRESH:
            keys.add("Down")
        for idx, key in self.button_key_map.items():
            if (a[:, idx] > 0.5).mean() >= BUTTON_FRAC:
                keys.add(key)
        return keys

    def close(self) -> None:
        try:
            subprocess.run(
                [self.wineboot_binary, "-k"],
                env=self._tool_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except Exception:
            pass
        try:
            super().close()
        finally:
            if getattr(self, "_sh", None) is not None:
                self._sh.close()

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            out_dir = Path.cwd() / ".nitrogen-env-build" / "wine_frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = str(out_dir / Path(path).name)
        super().save_frame(path)
