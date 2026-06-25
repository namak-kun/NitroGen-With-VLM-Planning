"""Shared in-process emulator base for frame-exact NitroGen eval/RL rollouts.

This module factors out the contract that the SNES backend's standalone base
already exposed: frame-stepped reset/step, savestate bytes, RGB frames, direct
state reads, and per-row capture.  It is backend-agnostic; concrete cores only
implement the small `_emu_*` hook surface plus `button_map`.

Frame cadence: the default is `frames_per_row=2`, matching the existing SNES
backend convention where one NitroGen 30 Hz control row advances two ~60 Hz
emulator frames.  Subclasses may override it if a core exposes a different
native rate, but GB/GBC/GBA and SNES normally use 2.

Buttons: action rows use NitroGen's 25-dim layout.  Buttons 0..20 are named by
`nitrogen.shared.BUTTON_ACTION_TOKENS`; indices 21/22 are the left stick with
the established 0.5-neutral convention and a +/-0.2 threshold.  Digital button
chunks use `BUTTON_FRAC=0.3` aggregation in `action_chunk_to_buttons`.
"""
from __future__ import annotations

import gc
import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nitrogen.shared import BUTTON_ACTION_TOKENS

from ..core import GameEnv, JLX, JLY, Observation, Scenario


STICK_THRESH = 0.2
BUTTON_FRAC = 0.3
DEFAULT_FRAMES_PER_ROW = 2
DEFAULT_CHUNK_SECONDS = 0.6


def _idx(token: str) -> int | None:
    try:
        return BUTTON_ACTION_TOKENS.index(token)
    except ValueError:
        return None


def _button_index_map(mapping: Mapping[str, str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for token, button in mapping.items():
        idx = _idx(token)
        if idx is not None:
            out[idx] = button
    return out


class EmulatorEnv(GameEnv):
    """Reusable in-process emulator environment base.

    Subclasses implement `_make_emulator`, `_emu_step`, `_emu_screen_rgb`,
    `_emu_set_buttons`, `_emu_save`, `_emu_load`, and `button_map`.
    `_make_emulator(rom_path, system)` is the preferred hook signature; the
    legacy one-argument `_make_emulator(rom_path)` used by `snes_env.py` is also
    accepted so that SNES can reconcile onto this base with minimal edits.
    """

    name = "emulator"
    action_hz = 30.0
    button_map: Mapping[str, str] = {}

    def __init__(
        self,
        rom_path: str,
        *,
        system: str | None = None,
        frames_per_row: int = DEFAULT_FRAMES_PER_ROW,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        launch: bool = True,
        **_: Any,
    ):
        self.rom_path = str(Path(rom_path).expanduser())
        if system is not None:
            self.system = system
        elif not hasattr(self, "system"):
            self.system = None
        self.frames_per_row = int(frames_per_row)
        self.chunk_seconds = float(chunk_seconds)
        self._step = 0
        self._emulator = None
        self._button_indices = _button_index_map(self.button_map)
        if launch:
            self._ensure_emulator()

    # ---- hooks supplied by concrete emulator backends ---------------------------------
    def _make_emulator(self, rom_path: str, system: str | None = None):
        raise NotImplementedError

    def _emu_step(self, n_frames: int) -> None:
        raise NotImplementedError

    def _emu_screen_rgb(self) -> np.ndarray:
        raise NotImplementedError

    def _emu_set_buttons(self, buttons: set[str]) -> None:
        raise NotImplementedError

    def _emu_save(self) -> bytes:
        raise NotImplementedError

    def _emu_load(self, state: bytes) -> None:
        raise NotImplementedError

    # ---- public contract ---------------------------------------------------------------
    def _call_make_emulator(self):
        make = self._make_emulator
        system = getattr(self, "system", None)
        try:
            sig = inspect.signature(make)
        except (TypeError, ValueError):
            return make(self.rom_path, system)

        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return make(self.rom_path, system)
        positional = [
            p
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            return make(self.rom_path, system)
        if "system" in sig.parameters:
            return make(self.rom_path, system=system)
        return make(self.rom_path)

    def _ensure_emulator(self) -> None:
        if self._emulator is None:
            self._emulator = self._call_make_emulator()
            self._emu_set_buttons(set())

    def _destroy_emulator(self) -> None:
        self._emulator = None
        gc.collect()

    def _rows(self, action_chunk: np.ndarray) -> np.ndarray:
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        if a.ndim != 2 or a.shape[1] < 25:
            raise ValueError(f"Expected NitroGen action chunk with shape (H,25); got {a.shape}")
        return a

    def _state_with_step(self) -> dict[str, Any]:
        state = self.read_state()
        return {"step": self._step, **state}

    def _frames_from_per_row(self, per_row: float | int | None) -> int:
        if per_row is None:
            return max(1, int(self.frames_per_row))
        value = float(per_row)
        if value <= 1.0:
            fps = float(getattr(self, "emulator_fps", 60.0) or 60.0)
            return max(1, int(round(value * fps)))
        return max(1, int(round(value)))

    def reset(self, scenario: Scenario | None = None) -> Observation:
        self._destroy_emulator()
        self._ensure_emulator()
        self._step = 0
        init = getattr(scenario, "init", None)
        if init is not None:
            if isinstance(init, (bytes, bytearray, memoryview)):
                self.load_state(bytes(init))
            elif isinstance(init, (str, Path)):
                self.load_state(Path(init).expanduser().read_bytes())
            else:
                raise TypeError(f"Unsupported Scenario.init for EmulatorEnv reset: {type(init)!r}")
        return Observation(frame=self.frame(), state=self._state_with_step(), step_idx=0, done=False)

    def step(self, action_chunk: np.ndarray) -> Observation:
        self._ensure_emulator()
        for row in self._rows(action_chunk):
            self._emu_set_buttons(self.action_row_to_buttons(row))
            self._emu_step(self.frames_per_row)
        self._emu_set_buttons(set())
        self._step += 1
        return Observation(frame=self.frame(), state=self._state_with_step(), step_idx=self._step, done=False)

    def apply_chunk_capture(
        self, action_chunk: np.ndarray, per_row: float | int | None = None
    ) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        self._ensure_emulator()
        frames = self._frames_from_per_row(per_row)
        out: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
        for row in self._rows(action_chunk):
            self._emu_set_buttons(self.action_row_to_buttons(row))
            self._emu_step(frames)
            out.append((row.copy(), self.frame(), self._state_with_step()))
        self._emu_set_buttons(set())
        return out

    def save_state(self) -> bytes:
        self._ensure_emulator()
        return self._emu_save()

    def load_state(self, state: bytes) -> None:
        self._ensure_emulator()
        self._emu_load(bytes(state))

    def read_state(self) -> dict[str, Any]:
        return {}

    def frame(self) -> np.ndarray:
        self._ensure_emulator()
        frame = np.asarray(self._emu_screen_rgb(), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Expected RGB frame HxWx3, got {frame.shape}")
        return frame.copy()

    def action_row_to_buttons(self, row: np.ndarray) -> set[str]:
        a = np.asarray(row, dtype=np.float32)
        if a.ndim != 1 or a.shape[0] < 25:
            raise ValueError(f"Expected one NitroGen action row with 25 dims; got {a.shape}")

        buttons: set[str] = set()
        lx, ly = float(a[JLX]), float(a[JLY])
        if lx < 0.5 - STICK_THRESH:
            buttons.add("LEFT")
        elif lx > 0.5 + STICK_THRESH:
            buttons.add("RIGHT")
        if ly < 0.5 - STICK_THRESH:
            buttons.add("UP")
        elif ly > 0.5 + STICK_THRESH:
            buttons.add("DOWN")

        for idx, button in self._button_indices.items():
            if float(a[idx]) > 0.5:
                buttons.add(button)
        return buttons

    def action_chunk_to_buttons(self, action_chunk: np.ndarray) -> set[str]:
        """Aggregate a chunk using the Mednafen/SNES `BUTTON_FRAC` convention."""
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            return self.action_row_to_buttons(a)
        if a.ndim != 2 or a.shape[1] < 25:
            raise ValueError(f"Expected NitroGen action chunk with shape (H,25); got {a.shape}")

        buttons: set[str] = set()
        lx, ly = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if lx < 0.5 - STICK_THRESH:
            buttons.add("LEFT")
        elif lx > 0.5 + STICK_THRESH:
            buttons.add("RIGHT")
        if ly < 0.5 - STICK_THRESH:
            buttons.add("UP")
        elif ly > 0.5 + STICK_THRESH:
            buttons.add("DOWN")

        for idx, button in self._button_indices.items():
            if float((a[:, idx] > 0.5).mean()) >= BUTTON_FRAC:
                buttons.add(button)
        return buttons

    def close(self) -> None:
        try:
            if self._emulator is not None:
                self._emu_set_buttons(set())
        except Exception:
            pass
        self._destroy_emulator()


__all__ = [
    "BUTTON_FRAC",
    "DEFAULT_CHUNK_SECONDS",
    "DEFAULT_FRAMES_PER_ROW",
    "EmulatorEnv",
    "STICK_THRESH",
]
