"""The Seeker (Rust/Bevy metroidvania) as a ProcGameEnv keyboard platformer."""
from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons
from ..core import Observation, Scenario

I_EAST, I_LSHLD, I_RTRIG, I_SOUTH, I_WEST = 5, 7, 16, 18, 20
STICK_THRESH = 0.25


class TheSeekerEnv(ProcGameEnv):
    name = "theseeker"
    window_name = "The Seeker"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 22.0, **kw):
        repo = Path(__file__).resolve().parents[3]
        self.game_dir = Path(os.environ.get("THESEEKER_DIR", repo / ".nitrogen-env-build/TheSeeker"))
        self.binary = self.game_dir / "target/release/theseeker_game"
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        self._ensure_runtime_assets()
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

    def reset(self, scenario: Scenario | None = None) -> Observation:
        if self._sh is not None:
            self._sh.unpause()
        self._enter_gameplay()
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

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "space",  # jump
                I_WEST: "j",       # attack
                I_RTRIG: "j",      # alternate attack for base-DiT RT-heavy actions
                I_EAST: "k",       # dash
                I_LSHLD: "k",      # alternate dash
            },
        )

    def _enter_gameplay(self) -> None:
        if not self._wid:
            return
        self._focus_window()
        target = self._key_target_args()
        self._xdo("key", *target, "grave")
        time.sleep(0.4)
        self._xdo("type", *target, "AppState Restart")
        time.sleep(0.2)
        self._xdo("key", *target, "Return")
        time.sleep(8.0)

    def _ensure_runtime_assets(self) -> None:
        if not self.binary.exists():
            raise RuntimeError(f"The Seeker binary not found: {self.binary}")
        assets_link = self.binary.parent / "assets"
        if not assets_link.exists():
            assets_link.symlink_to("../../assets")
        music_dir = self.game_dir / "assets/audio/music"
        ambience = music_dir / "Ambience1.flac"
        for name in ("Music2.flac", "Music3.flac"):
            path = music_dir / name
            if not path.exists() and ambience.exists():
                path.symlink_to("Ambience1.flac")
