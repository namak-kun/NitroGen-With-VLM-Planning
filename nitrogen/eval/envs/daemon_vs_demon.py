"""Daemon vs Demon (Godot 2.x) as a ProcGameEnv keyboard action game."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_RTRIG, I_SOUTH, I_START, I_WEST = 16, 18, 19, 20
STICK_THRESH = 0.25

REPO = Path(__file__).resolve().parents[3]
BUILD_ROOT = REPO / ".nitrogen-env-build"
DEFAULT_GODOT = BUILD_ROOT / "godot-2.1.6" / "Godot_v2.1.6-stable_x11.64"
DEFAULT_PROJECT = BUILD_ROOT / "daemon-vs-demon"
RUNTIME_ROOT = BUILD_ROOT / "daemon_vs_demon_runtime"


class DaemonVsDemonEnv(ProcGameEnv):
    name = "daemon_vs_demon"
    window_name = "Daemon vs Demon"
    control = "keyboard"
    window_manager = "matchbox-window-manager"
    target_keys_to_window = False

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 14.0,
        godot_binary: str | None = None,
        project_dir: str | None = None,
        **kw,
    ):
        self.godot_binary = Path(godot_binary) if godot_binary else DEFAULT_GODOT
        self.project_dir = Path(project_dir) if project_dir else DEFAULT_PROJECT

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

    def launch_cmd(self):
        if not self.godot_binary.exists():
            raise FileNotFoundError(f"Godot 2.1.6 binary not found: {self.godot_binary}")
        if not self.project_dir.exists():
            raise FileNotFoundError(f"Daemon vs Demon project not found: {self.project_dir}")
        return [
            "env",
            "LIBGL_ALWAYS_SOFTWARE=1",
            f"HOME={self.home_dir}",
            f"XDG_DATA_HOME={self.data_dir}",
            f"TMPDIR={self.tmp_dir}",
            str(self.godot_binary),
            "-w",
            "-r",
            f"{int(self.width)}x{int(self.height)}",
            "-path",
            str(self.project_dir),
        ]

    def reset_macro(self, scenario):
        return [
            ("wait", 0.2),
            ("hold", "Return", 0.7),
            ("wait", 2.0),
            ("hold", "Down", 0.8),
            ("wait", 5.0),
            ("hold", "Right", 1.0),
            ("wait", 0.2),
        ]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "k",       # pick / interact
                I_WEST: "j",        # attack
                I_RTRIG: "j",       # alternate attack
                I_START: "Return",  # menu confirm
            },
        )

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
