"""Reusable Mednafen-backed console emulator envs."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping

import numpy as np

from nitrogen.shared import BUTTON_ACTION_TOKENS

from ..core import JLX, JLY
from .proc_game_env import ProcGameEnv

I_SOUTH = BUTTON_ACTION_TOKENS.index("SOUTH")
I_START = BUTTON_ACTION_TOKENS.index("START")
I_WEST = BUTTON_ACTION_TOKENS.index("WEST")

STICK_THRESH = 0.2
BUTTON_FRAC = 0.3

SYSTEM_ALIASES = {
    "nes": "nes",
    "famicom": "nes",
    "snes": "snes",
    "sfc": "snes",
    "snes_faust": "snes_faust",
    "md": "md",
    "genesis": "md",
    "megadrive": "md",
    "mega_drive": "md",
    "gb": "gb",
    "gbc": "gb",
    "gameboy": "gb",
    "gameboy_color": "gb",
    "gba": "gba",
    "pce": "pce",
    "pc_engine": "pce",
    "pce_fast": "pce_fast",
}

DEFAULT_BUTTON_KEY_MAP = {
    I_SOUTH: "bracketleft",  # Mednafen default NES A button.
    I_WEST: "z",            # Mednafen default NES B button.
    I_START: "Return",
}


class MednafenConsoleEnv(ProcGameEnv):
    """Generic Mednafen console env for legal ROMs supplied by callers."""

    name = "mednafen_console"
    control = "keyboard"
    target_keys_to_window = True

    def __init__(
        self,
        rom_path: str,
        system: str,
        mednafen_home: str,
        *,
        button_key_map: Mapping[int, str] | None = None,
        mednafen_binary: str = "mednafen",
        width: int = 800,
        height: int = 600,
        boot_wait: float = 4.0,
        **kw,
    ):
        self.rom_path = str(Path(rom_path).expanduser())
        self.system = self._normalize_system(system)
        self.mednafen_home = Path(mednafen_home).expanduser()
        self.mednafen_home.mkdir(parents=True, exist_ok=True)
        self._speedhack_tmpdir = self.mednafen_home / "tmp"
        self._speedhack_tmpdir.mkdir(parents=True, exist_ok=True)
        self.mednafen_binary = mednafen_binary
        self.button_key_map = dict(button_key_map or DEFAULT_BUTTON_KEY_MAP)
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    @staticmethod
    def _normalize_system(system: str) -> str:
        key = system.lower()
        try:
            return SYSTEM_ALIASES[key]
        except KeyError as exc:
            supported = ", ".join(sorted(SYSTEM_ALIASES))
            raise ValueError(
                f"Unsupported Mednafen system {system!r}; expected one of: {supported}"
            ) from exc

    def _env(self) -> dict:
        env = super()._env()
        env.update(
            {
                "SDL_VIDEODRIVER": "x11",
                "SDL_AUDIODRIVER": "dummy",
                "MEDNAFEN_HOME": str(self.mednafen_home),
                "MEDNAFEN_ALLOWMULTI": "1",
            }
        )
        return env

    def _helper_env(self) -> dict:
        # A 0.0 speedhack freeze also freezes helper subprocess clocks; keep helpers unpreloaded.
        env = dict(os.environ, DISPLAY=f":{self.display}")
        env.update({"SDL_VIDEODRIVER": "x11", "SDL_AUDIODRIVER": "dummy"})
        return env

    def launch_cmd(self) -> list[str]:
        return [
            self.mednafen_binary,
            "-force_module",
            self.system,
            "-sound",
            "0",
            "-video.driver",
            "softfb",
            "-video.fs",
            "0",
            f"-{self.system}.xscale",
            "2",
            f"-{self.system}.yscale",
            "2",
            f"-{self.system}.stretch",
            "0",
            "-nothrottle",
            "0",
            self.rom_path,
        ]

    def boot(self):
        old_tmpdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = str(self._speedhack_tmpdir)
        try:
            super().boot()
        finally:
            if old_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = old_tmpdir
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def _find_window(self) -> str:
        out = subprocess.run(
            ["xdotool", "search", "--class", "mednafen"],
            env=self._helper_env(),
            capture_output=True,
            text=True,
        )
        ids = [line for line in out.stdout.split() if line.strip()]
        return ids[-1] if ids else ""

    def _xdo(self, *args):
        subprocess.run(
            ["xdotool", *args],
            env=self._helper_env(),
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
            env=self._helper_env(),
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
            keys.add("a")
        elif mx > 0.5 + STICK_THRESH:
            keys.add("d")
        if my < 0.5 - STICK_THRESH:
            keys.add("w")
        elif my > 0.5 + STICK_THRESH:
            keys.add("s")

        for idx, key in self.button_key_map.items():
            if (a[:, idx] > 0.5).mean() >= BUTTON_FRAC:
                keys.add(key)
        return keys

    def close(self) -> None:
        try:
            super().close()
        finally:
            if getattr(self, "_sh", None) is not None:
                self._sh.close()

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            path = str(self.mednafen_home / Path(path).name)
        super().save_frame(path)


class MednafenNesEnv(MednafenConsoleEnv):
    """Tiny concrete NES wrapper for smoke tests and quick env vetting."""

    name = "mednafen_nes"

    def __init__(self, rom_path: str, mednafen_home: str, **kw):
        super().__init__(rom_path=rom_path, system="nes", mednafen_home=mednafen_home, **kw)
