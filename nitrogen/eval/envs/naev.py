"""Naev env (top-down 2D space combat / exploration).

Ubuntu's Naev 0.8.2 boots headless under Xvfb with llvmpipe. The reset macro creates a
throwaway pilot, skips the intro, declines the tutorial, and leaves the ship flying in space.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_LTRIG, I_RSHLD, I_RTRIG, I_SOUTH, I_WEST = 9, 14, 16, 18, 20
STEER_THRESH = 0.2
THRUST_THRESH = 0.2
BUTTON_FRAC = 0.3


class NaevEnv(ProcGameEnv):
    name = "naev"
    window_name = "Naev"
    control = "keyboard"
    window_manager = "matchbox-window-manager"
    target_keys_to_window = False

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 16.0,
                 chunk_seconds: float = 3.0, always_accel: bool = False, **kw):
        self.always_accel = always_accel
        self._repo_root = Path(__file__).resolve().parents[3]
        self._runtime_root = self._repo_root / ".nitrogen-runtime"
        self._runtime_dir = self._runtime_root / f"naev_{os.getpid()}_{int(time.time() * 1000)}"
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(width=width, height=height, boot_wait=boot_wait,
                         chunk_seconds=chunk_seconds, **kw)

    def launch_cmd(self):
        return ["env", "LIBGL_ALWAYS_SOFTWARE=1", "naev", "-d", str(self._runtime_dir),
                "-W", str(self.width), "-H", str(self.height), "-M"]

    def boot(self):
        self._xvfb = subprocess.Popen(
            ["Xvfb", f":{self.display}", "-screen", "0", f"{self.width}x{self.height}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        self._wm = subprocess.Popen(self.window_manager.split(), env=self._env(),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        if self.freeze_during_inference:
            from ..speedhack import SpeedHack
            ctrl = self._runtime_dir / f"speedhack_{os.getpid()}_{self.display}.ctl"
            self._sh = SpeedHack(ctrl_path=str(ctrl))
        self._game = subprocess.Popen(self.launch_cmd(), env=self._env(),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(self.boot_wait)
        self._wid = self._find_window()
        if self._wid:
            self._focus_window()

    def _find_window(self) -> str:
        out = subprocess.run(["xdotool", "search", "--class", "Naev"],
                             env=self._env(), capture_output=True, text=True)
        ids = [line for line in out.stdout.split() if line.strip()]
        return ids[-1] if ids else super()._find_window()

    def reset_macro(self, scenario):
        pilot = "nitrogen"
        macro = [
            ("wait", 0.5),
            ("key", "Return"),  # main menu: New Game
            ("wait", 0.5),
        ]
        macro += [("key", ch) for ch in pilot]
        macro += [
            ("wait", 0.2),
            ("key", "Return"),  # create pilot
            ("wait", 1.0),
            ("key", "Return"),  # accept hyperspace speed prompt
            ("wait", 6.0),
            ("key", "Escape"),  # skip opening crawl to tutorial prompt
            ("wait", 1.0),
            ("key", "Tab"),     # select "No" on tutorial yes/no
            ("wait", 0.2),
            ("key", "Return"),
            ("wait", 1.0),
            ("key", "Return"),  # dismiss "No, thanks"
            ("wait", 1.5),
        ]
        return macro

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys = set()
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())

        if mx < 0.5 - STEER_THRESH:
            keys.add("a")
        elif mx > 0.5 + STEER_THRESH:
            keys.add("d")

        rt = (a[:, I_RTRIG] > 0.5).mean() >= BUTTON_FRAC
        if self.always_accel or rt or my < 0.5 - THRUST_THRESH:
            keys.add("w")
        if (a[:, I_LTRIG] > 0.5).mean() >= BUTTON_FRAC or my > 0.5 + THRUST_THRESH:
            keys.discard("w")
            keys.add("s")

        if rt or (a[:, I_SOUTH] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("space")
        if (a[:, I_WEST] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("Shift_L")
        if (a[:, I_WEST] > 0.5).mean() >= BUTTON_FRAC or (a[:, I_RSHLD] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("t")
        return keys

    def save_frame(self, path: str) -> None:
        if os.path.abspath(path).startswith("/tmp/"):
            path = str(self._runtime_dir / "naev_vet_frame.png")
        super().save_frame(path)

    def close(self) -> None:
        sh = getattr(self, "_sh", None)
        try:
            super().close()
        finally:
            if sh is not None:
                sh.close()
            shutil.rmtree(self._runtime_dir, ignore_errors=True)
            try:
                self._runtime_root.rmdir()
            except OSError:
                pass
