"""Castlevania on Godot (Godot 3.x) as a ProcGameEnv keyboard platformer."""
from __future__ import annotations

from pathlib import Path

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_RTRIG, I_SOUTH, I_START, I_WEST = 16, 18, 19, 20
STICK_THRESH = 0.25


class CastlevaniaGodotEnv(ProcGameEnv):
    name = "castlevania_godot"
    window_name = "Castlevania on Godot"
    control = "keyboard"
    reset_by_relaunch = True             # title/progression game: respawn for a clean reset

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 15.0,
        godot_binary: str | None = None,
        project_dir: str | None = None,
        **kw,
    ):
        repo = Path(__file__).resolve().parents[3]
        self.godot_binary = godot_binary or str(
            repo / "tmp/godot-3.5.3/Godot_v3.5.3-stable_x11.64"
        )
        self.project_dir = project_dir or str(repo / "tmp/castlevania-runtime")
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        return [
            "env",
            "LIBGL_ALWAYS_SOFTWARE=1",
            self.godot_binary,
            "--video-driver",
            "GLES2",
            "--audio-driver",
            "Dummy",
            "--path",
            self.project_dir,
        ]

    def reset_macro(self, scenario):
        return [("key", "z"), ("wait", 9.0), ("hold", "Right", 1.5), ("wait", 0.2)]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "x",       # jump
                I_WEST: "z",        # whip attack
                I_RTRIG: "z",       # alternate attack
                I_START: "Return",  # menu accept
            },
        )
