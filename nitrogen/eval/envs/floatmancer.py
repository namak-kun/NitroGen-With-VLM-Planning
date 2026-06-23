"""Floatmancer NES homebrew environment backed by Mednafen."""
from __future__ import annotations

from typing import Optional

from nitrogen.shared import BUTTON_ACTION_TOKENS

from ..core import Scenario
from .mednafen_console import MednafenNesEnv

I_RIGHT_TRIGGER = BUTTON_ACTION_TOKENS.index("RIGHT_TRIGGER")
I_SOUTH = BUTTON_ACTION_TOKENS.index("SOUTH")
I_START = BUTTON_ACTION_TOKENS.index("START")
I_WEST = BUTTON_ACTION_TOKENS.index("WEST")


class FloatmancerEnv(MednafenNesEnv):
    """CC0 NES precision-platformer homebrew running through Mednafen."""

    name = "floatmancer"
    system = "nes"

    def __init__(
        self,
        rom_path: str = "/tmp/roms/floatmancer.nes",
        mednafen_home: str = "/tmp/nitrogen-mednafen-floatmancer",
        **kw,
    ):
        kw.setdefault("chunk_seconds", 1.2)
        kw.setdefault(
            "button_key_map",
            {
                I_SOUTH: "Return",
                I_RIGHT_TRIGGER: "Return",
                I_WEST: "z",
                I_START: "space",
            },
        )
        super().__init__(
            rom_path=rom_path,
            mednafen_home=mednafen_home,
            mednafen_binary="/usr/games/mednafen",
            **kw,
        )

    def launch_cmd(self) -> list[str]:
        cmd = super().launch_cmd()
        # The title menu enters gameplay with NES A; map A to Return so SOUTH/jump and the
        # generic RIGHT_TRIGGER vet action can exercise gameplay without colliding with START.
        return cmd[:-1] + [
            "-nes.input.port1.gamepad.a",
            "keyboard 0x0 40",
            "-nes.input.port1.gamepad.start",
            "keyboard 0x0 44",
        ] + cmd[-1:]

    def reset_macro(self, scenario: Optional[Scenario]) -> list:
        return [("key", "Return"), ("wait", 4.0)]
