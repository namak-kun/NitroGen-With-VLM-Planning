"""Project: Starfighter env — mission-based space shmup control via keyboard."""
from __future__ import annotations

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_RTRIG, I_SOUTH, I_WEST = 16, 18, 20
STICK_THRESH = 0.2


class StarfighterEnv(ProcGameEnv):
    name = "starfighter"
    window_name = "Project: Starfighter"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 12.0, **kw):
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        return ["starfighter", "-noaudio"]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_RTRIG: "Control_L",
                I_SOUTH: "Control_L",
                I_WEST: "space",
            },
            button_frac=0.3,
        )

    def reset_macro(self, scenario):
        return [
            ("wait", 0.5),
            ("key", "Return"),
            ("wait", 0.6),
            ("key", "Return"),
            ("wait", 0.6),
            ("key", "Return"),
            ("wait", 1.0),
            ("key", "Escape"),
            ("wait", 0.8),
            ("key", "Escape"),
            ("wait", 0.5),
            ("key", "Return"),
            ("wait", 1.5),
        ]
