"""Platformer Challenge Mega Drive homebrew environment backed by Mednafen."""
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
DEFAULT_ROM_PATH = str(REPO / "tmp" / "roms" / "platformer_challenge.bin")
DEFAULT_MEDNAFEN_HOME = str(REPO / "tmp" / "nitrogen-mednafen-platformer-challenge")


class PlatformerChallengeEnv(MednafenConsoleEnv):
    """MIT-licensed SGDK platformer homebrew running through Mednafen."""

    name = "platformer_challenge"
    system = "md"

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
                I_RIGHT_TRIGGER: "bracketright",
                I_WEST: "bracketright",
            },
        )
        super().__init__(
            rom_path=rom_path,
            system="md",
            mednafen_home=mednafen_home,
            mednafen_binary="/usr/games/mednafen",
            **kw,
        )

    def reset_macro(self, scenario: Optional[Scenario]) -> list:
        return [("wait", 1.0)]
