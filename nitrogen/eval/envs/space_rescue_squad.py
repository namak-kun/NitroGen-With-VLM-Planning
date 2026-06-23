"""Space Rescue Squad SNES homebrew environment backed by Mednafen."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from nitrogen.shared import BUTTON_ACTION_TOKENS

from ..core import Scenario
from .mednafen_console import MednafenConsoleEnv

I_RIGHT_TRIGGER = BUTTON_ACTION_TOKENS.index("RIGHT_TRIGGER")
I_SOUTH = BUTTON_ACTION_TOKENS.index("SOUTH")
I_WEST = BUTTON_ACTION_TOKENS.index("WEST")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROM_PATH = str(REPO / "tmp" / "roms" / "space_rescue_squad.sfc")
DEFAULT_MEDNAFEN_HOME = str(REPO / "tmp" / "nitrogen-mednafen-space-rescue-squad")


class SpaceRescueSquadEnv(MednafenConsoleEnv):
    """zlib-licensed SNESDEV game-jam platformer homebrew running through Mednafen."""

    name = "space_rescue_squad"
    system = "snes"

    def __init__(
        self,
        rom_path: str = DEFAULT_ROM_PATH,
        mednafen_home: str = DEFAULT_MEDNAFEN_HOME,
        **kw,
    ):
        kw.setdefault("chunk_seconds", 1.2)
        kw.setdefault(
            "button_key_map",
            {
                I_SOUTH: "bracketleft",
                I_RIGHT_TRIGGER: "bracketleft",
                I_WEST: "x",
            },
        )
        super().__init__(
            rom_path=rom_path,
            system="snes",
            mednafen_home=mednafen_home,
            mednafen_binary="/usr/games/mednafen",
            **kw,
        )

    def launch_cmd(self) -> list[str]:
        cmd = super().launch_cmd()
        return cmd[:-1] + [
            "-snes.input.port1.gamepad.b",
            "keyboard 0x0 47",
            "-snes.input.port1.gamepad.y",
            "keyboard 0x0 48",
            "-snes.input.port1.gamepad.a",
            "keyboard 0x0 29",
            "-snes.input.port1.gamepad.x",
            "keyboard 0x0 27",
            "-snes.input.port1.gamepad.start",
            "keyboard 0x0 40",
        ] + cmd[-1:]

    def reset_macro(self, scenario: Optional[Scenario]) -> list:
        return [
            ("key", "Return"),
            ("wait", 0.5),
            ("key", "Return"),
            ("wait", 1.5),
            ("hold", "d", 1.0),
            ("wait", 0.5),
        ]
