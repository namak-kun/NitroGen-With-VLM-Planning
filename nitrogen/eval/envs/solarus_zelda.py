"""Solarus/ZSDX top-down Zelda-like env."""
from __future__ import annotations

import shutil
from pathlib import Path

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_RTRIG, I_LTRIG, I_NORTH, I_EAST, I_SOUTH, I_START, I_WEST = 16, 9, 10, 5, 18, 19, 20
STICK_THRESH = 0.25


class SolarusZeldaEnv(ProcGameEnv):
    name = "solarus_zelda"
    window_name = "Zelda Mystery of Solarus DX"
    control = "keyboard"
    window_manager = "matchbox-window-manager"
    target_keys_to_window = False

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 12.0,
        solarus_binary: str | None = None,
        quest_dir: str | None = None,
        **kw,
    ):
        repo = Path(__file__).resolve().parents[3]
        self.solarus_binary = solarus_binary or str(
            repo / "tmp/solarus/build/cli/solarus-run"
        )
        if not Path(self.solarus_binary).exists():
            self.solarus_binary = shutil.which("solarus-run") or self.solarus_binary
        self.quest_dir = quest_dir or str(repo / "tmp/zsdx")
        self.solarus_home = repo / "tmp/solarus-home"
        self._ensure_runtime_files()
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def _ensure_runtime_files(self) -> None:
        write_dir = self.solarus_home / ".solarus/zsdx"
        write_dir.mkdir(parents=True, exist_ok=True)
        (write_dir / "debug").touch()
        (write_dir / "settings.dat").write_text(
            'language = "en"\n'
            "joypad_enabled = false\n"
            "sound_volume = 0\n"
            "music_volume = 0\n"
            "fullscreen = false\n",
            encoding="utf-8",
        )
        (write_dir / "save1.dat").write_text(
            '_version = 2\n'
            '_starting_map = "3"\n'
            '_starting_point = "out_link_house"\n'
            '_keyboard_action = "space"\n'
            '_keyboard_attack = "c"\n'
            '_keyboard_item_1 = "x"\n'
            '_keyboard_item_2 = "v"\n'
            '_keyboard_pause = "d"\n'
            '_keyboard_right = "right"\n'
            '_keyboard_up = "up"\n'
            '_keyboard_left = "left"\n'
            '_keyboard_down = "down"\n'
            '_joypad_action = "b"\n'
            '_joypad_attack = "a"\n'
            '_joypad_item_1 = "x"\n'
            '_joypad_item_2 = "y"\n'
            '_joypad_pause = "start"\n'
            '_joypad_right = "left_x +"\n'
            '_joypad_up = "left_y -"\n'
            '_joypad_left = "left_x -"\n'
            '_joypad_down = "left_y +"\n'
            '_current_life = 12\n'
            '_max_life = 12\n'
            '_current_money = 0\n'
            '_max_money = 100\n'
            '_current_magic = 0\n'
            '_max_magic = 0\n'
            '_ability_tunic = 1\n'
            '_ability_sword = 1\n'
            '_ability_sword_spin_attack = 1\n'
            '_ability_push = 1\n'
            '_ability_grab = 1\n'
            '_ability_pull = 1\n'
            'i1034 = 1\n'
            'i1128 = 1\n'
            'i1129 = 1\n'
            'player_name = "Nitro"\n',
            encoding="utf-8",
        )

    def launch_cmd(self):
        return [
            "env",
            f"HOME={self.solarus_home}",
            "SDL_AUDIODRIVER=dummy",
            "LIBGL_ALWAYS_SOFTWARE=1",
            self.solarus_binary,
            "-no-audio",
            "-suspend-unfocused=no",
            "-fullscreen=no",
            "-cursor-visible=no",
            "-force-software-rendering",
            '-s=sol.language.set_language("en")',
            self.quest_dir,
        ]

    def reset_macro(self, scenario):
        return [
            ("key", "F1"),
            ("wait", 0.5),
            ("key", "Return"),
            ("wait", 1.0),
            ("key", "r"),  # debug speed toggle, available because we seed the quest debug flag.
            ("wait", 0.3),
        ]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "c",       # sword/attack
                I_EAST: "space",    # action/interact
                I_LTRIG: "space",   # alternate action/interact
                I_WEST: "x",        # item slot 1
                I_NORTH: "v",       # item slot 2
                I_RTRIG: "d",       # pause/menu; vet_env exercises right trigger
                I_START: "d",       # pause
            },
        )
