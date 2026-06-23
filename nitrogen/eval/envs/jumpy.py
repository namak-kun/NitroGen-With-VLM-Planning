"""Fish Folk: Jumpy (Rust/Bevy brawler) as a ProcGameEnv keyboard platformer."""
from __future__ import annotations

import os
import shlex
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_EAST, I_LSHLD, I_NORTH, I_RTRIG, I_SOUTH, I_WEST = 5, 7, 10, 16, 18, 20
STICK_THRESH = 0.25
BUTTON_FRAC = 0.3


class JumpyEnv(ProcGameEnv):
    name = "jumpy"
    window_name = "App"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 22.0, **kw):
        repo = Path(__file__).resolve().parents[3]
        self.game_dir = Path(os.environ.get("JUMPY_DIR", repo / ".nitrogen-env-build/jumpy"))
        self.binary = Path(
            os.environ.get(
                "JUMPY_BINARY",
                repo / ".nitrogen-env-build/jumpy-target/release/jumpy",
            )
        )
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        if not self.binary.exists():
            raise RuntimeError(f"Jumpy binary not found: {self.binary}")
        quoted_dir = shlex.quote(str(self.game_dir))
        quoted_bin = shlex.quote(str(self.binary))
        return [
            "sh",
            "-c",
            "cd {dir} && "
            "WINIT_UNIX_BACKEND=x11 "
            "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json "
            "WGPU_BACKEND=vulkan "
            "exec {bin}".format(dir=quoted_dir, bin=quoted_bin),
        ]

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys: set[str] = set()

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
            return bool((a[:, idx] > 0.5).mean() >= BUTTON_FRAC)

        if pressed(I_SOUTH):
            keys.add("space")  # jump / menu confirm
        if pressed(I_WEST) or pressed(I_RTRIG):
            keys.add("c")      # shoot / use held weapon
        if pressed(I_EAST):
            keys.add("v")      # grab
        if pressed(I_NORTH):
            keys.add("b")      # slide
        if pressed(I_LSHLD):
            keys.add("f")      # ragdoll / drop
        return keys

    def reset_macro(self, scenario):
        return [
            ("wait", 0.5),
            ("hold", "space", 0.25),   # Local Game
            ("wait", 0.7),
            ("hold", "space", 0.25),   # join as Keyboard 1
            ("wait", 0.7),
            ("hold", "space", 0.25),   # choose fish
            ("wait", 0.7),
            ("hold", "space", 0.25),   # choose hat / ready
            ("wait", 0.7),
            ("hold", "Return", 0.4),   # continue to map select
            ("wait", 0.7),
            ("hold", "space", 0.25),   # select All Maps
            ("wait", 4.0),
        ]
