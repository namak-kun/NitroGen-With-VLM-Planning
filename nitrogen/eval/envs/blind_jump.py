"""Blind Jump GBA homebrew environment backed by in-process mGBA.

ROM source: https://github.com/evanbowman/blind-jump-portable/releases/tag/2021.11.20.0
License: MIT (https://github.com/evanbowman/blind-jump-portable/blob/master/LICENSE)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from nitrogen.shared import BUTTON_ACTION_TOKENS

from ..core import JLX, JLY, Observation, Scenario
from .mgba_env import MgbaEnv

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROM_PATH = str(REPO / "tmp" / "roms" / "BlindJump.gba")

I_SOUTH = BUTTON_ACTION_TOKENS.index("SOUTH")


def _chunk(n_rows: int, *, lx: float = 0.5, ly: float = 0.5, buttons: tuple[int, ...] = ()) -> np.ndarray:
    a = np.zeros((n_rows, 25), dtype=np.float32)
    a[:, JLX] = lx
    a[:, JLY] = ly
    for idx in buttons:
        a[:, idx] = 1.0
    return a


class BlindJumpEnv(MgbaEnv):
    """MIT GBA top-down action/adventure; reset macro reaches Part 1 gameplay."""

    name = "blind_jump"

    def __init__(self, rom_path: str = DEFAULT_ROM_PATH, **kw):
        super().__init__(rom_path=rom_path, **kw)

    def reset(self, scenario: Optional[Scenario] = None) -> Observation:
        obs = super().reset(scenario)
        # Language confirm, select "new game", wait through intro, then advance into gameplay.
        for action in (
            _chunk(60, buttons=(I_SOUTH,)),
            _chunk(120),
            _chunk(60, buttons=(I_SOUTH,)),
            _chunk(900),
            _chunk(60, buttons=(I_SOUTH,)),
            _chunk(180),
        ):
            obs = super().step(action)
        self._step = 0
        return Observation(frame=obs.frame, state=self._state_with_step(), step_idx=0, done=False)


__all__ = ["BlindJumpEnv"]
