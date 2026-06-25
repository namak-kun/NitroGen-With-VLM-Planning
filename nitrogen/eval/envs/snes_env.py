"""In-process SNES backend for frame-exact NitroGen eval/RL rollouts.

Backend: stable-retro 1.0.0 (`stable_retro.RetroEmulator`) with its bundled
Snes/snes9x libretro core.  This env does not use Retro game integrations:
`RetroEmulator(rom_path)` loads an arbitrary homebrew ROM directly, headless,
as long as the extension is known to stable-retro.  The bundled core declares
`.sfc`; this module also registers `.smc` for Snes at import/runtime.

Frame cadence: SNES reports ~60.0988 FPS, so each NitroGen 30 Hz action row is
held for `frames_per_row=2` emulator frames by default.  A normal H=18 chunk
therefore advances 36 SNES frames (~0.6 s).

Button map: NitroGen SOUTH->SNES B, EAST->A, WEST->Y, NORTH->X,
LEFT_SHOULDER->L, RIGHT_SHOULDER->R, START->START, BACK/SELECT->SELECT, and
left stick or DPAD -> UP/DOWN/LEFT/RIGHT.  The stick convention is NitroGen's
load-bearing 0.5-neutral range with a +/-0.2 threshold, matching the Mednafen
SNES subprocess envs.

Save/load: stable-retro exposes `em.get_state() -> bytes` and
`em.set_state(bytes) -> bool`.  Its set_state restores core execution state but
does not immediately repaint the current Python framebuffer, so SnesEnv's
snapshot bytes include both the core state and the last RGB frame cache.  This
makes `load_state(); frame()` return exactly the saved frame while subsequent
stepping resumes from the restored core state.

Memory: `GameData.update_ram()` exposes core memory blocks.  SNES WRAM is
available at stable-retro's documented Snes `rambase` 0x7E0000 (128 KiB);
`read_memory`, `read_wram`, and `read_state` demonstrate direct access for RL
reward/state code.
"""
from __future__ import annotations

import gc
import json
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from nitrogen.shared import BUTTON_ACTION_TOKENS

from ..core import GameEnv, JLX, JLY, Observation, Scenario

try:  # Prefer the shared base if the parallel agent has already landed it.
    from .emulator_env import EmulatorEnv as _SharedEmulatorEnv
except Exception:  # pragma: no cover - exercised while the shared base is absent.
    _SharedEmulatorEnv = None


STICK_THRESH = 0.2
BUTTON_FRAC = 0.3
DEFAULT_FRAMES_PER_ROW = 2
DEFAULT_CHUNK_SECONDS = 0.6
SNES_WRAM_BASE = 0x7E0000
SNES_BUTTONS = ("B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R")
STATE_MAGIC = b"NITROGEN-SNES-STATE-v1\0"

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROM_PATH = str(REPO / "tmp" / "roms" / "space_rescue_squad.sfc")


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


class _StandaloneEmulatorEnv(GameEnv):
    """Minimal compatible fallback for the shared in-process EmulatorEnv contract."""

    name = "emulator"
    action_hz = 30.0

    def __init__(
        self,
        rom_path: str,
        *,
        frames_per_row: int = DEFAULT_FRAMES_PER_ROW,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        launch: bool = True,
        **_: Any,
    ):
        self.rom_path = str(Path(rom_path).expanduser())
        self.frames_per_row = int(frames_per_row)
        self.chunk_seconds = float(chunk_seconds)
        self._step = 0
        self._emulator = None
        if launch:
            self._emulator = self._make_emulator(self.rom_path)
            self._emu_set_buttons(set())

    # ---- hooks supplied by concrete emulator backends ---------------------------------
    button_map: Mapping[str, str] = {}

    def _make_emulator(self, rom_path: str):
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
    def _ensure_emulator(self) -> None:
        if self._emulator is None:
            self._emulator = self._make_emulator(self.rom_path)
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

    def apply_chunk_capture(self, action_chunk: np.ndarray, per_row: float | int | None = None) -> list:
        self._ensure_emulator()
        frames = self._frames_from_per_row(per_row)
        out = []
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

    def frame(self) -> np.ndarray:
        self._ensure_emulator()
        frame = np.asarray(self._emu_screen_rgb(), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Expected RGB frame HxWx3, got {frame.shape}")
        return frame.copy()

    def action_row_to_buttons(self, row: np.ndarray) -> set[str]:
        raise NotImplementedError

    def read_state(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        try:
            if self._emulator is not None:
                self._emu_set_buttons(set())
        except Exception:
            pass
        self._destroy_emulator()


EmulatorEnv = _SharedEmulatorEnv or _StandaloneEmulatorEnv
USING_SHARED_EMULATOR_ENV = _SharedEmulatorEnv is not None


class SnesEnv(EmulatorEnv):
    """Headless, in-process SNES env with frame stepping, savestates, and RAM reads."""

    name = "snes"
    system = "Snes"
    action_hz = 30.0
    button_map = {
        "SOUTH": "B",
        "EAST": "A",
        "WEST": "Y",
        "NORTH": "X",
        "LEFT_SHOULDER": "L",
        "RIGHT_SHOULDER": "R",
        "START": "START",
        "BACK": "SELECT",
        "SELECT": "SELECT",
        "DPAD_UP": "UP",
        "DPAD_DOWN": "DOWN",
        "DPAD_LEFT": "LEFT",
        "DPAD_RIGHT": "RIGHT",
    }
    _button_indices = _button_index_map(button_map)

    def __init__(
        self,
        rom_path: str = DEFAULT_ROM_PATH,
        *,
        frames_per_row: int = DEFAULT_FRAMES_PER_ROW,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        **kwargs: Any,
    ):
        self._retro = None
        self._game_data = None
        self._frame_cache: np.ndarray | None = None
        self._current_buttons: set[str] = set()
        self._button_order = list(SNES_BUTTONS)
        self._button_to_i = {button: i for i, button in enumerate(self._button_order)}
        self.emulator_fps = 60.0
        self.wram_base = SNES_WRAM_BASE
        super().__init__(
            rom_path=str(Path(rom_path).expanduser()),
            frames_per_row=frames_per_row,
            chunk_seconds=chunk_seconds,
            **kwargs,
        )

    # ---- stable-retro setup ------------------------------------------------------------
    def _import_retro(self):
        try:
            import stable_retro as retro
        except ImportError:  # stable-retro also provides the deprecated `retro` import name.
            import retro  # type: ignore[no-redef]
        return retro

    def _ensure_snes_extensions(self, retro) -> None:
        info = dict(retro.get_system_info("Snes"))
        exts = set(info.get("ext", []))
        if "smc" not in exts:
            info["ext"] = sorted(exts | {"smc"})
            retro.RetroEmulator.load_core_info(json.dumps({"Snes": info}))
            retro.data.EMU_INFO["Snes"] = info
            retro.data.EMU_CORES["Snes"] = "snes9x_libretro.so"
            retro.data.EMU_EXTENSIONS[".smc"] = "Snes"
        retro.data.EMU_EXTENSIONS.setdefault(".sfc", "Snes")

    def _make_emulator(self, rom_path: str):
        retro = self._import_retro()
        self._retro = retro
        self._ensure_snes_extensions(retro)

        path = Path(rom_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SNES ROM not found: {path}")
        if path.suffix.lower() not in {".sfc", ".smc"}:
            raise ValueError(f"SnesEnv expects a .sfc or .smc ROM, got {path}")

        gc.collect()
        emu = retro.RetroEmulator(str(path))
        self._game_data = retro.data.GameData()
        emu.configure_data(self._game_data)
        info = retro.get_system_info("Snes")
        self._button_order = list(info.get("buttons", SNES_BUTTONS))
        self._button_to_i = {button: i for i, button in enumerate(self._button_order)}
        self.wram_base = int(info.get("rambase", SNES_WRAM_BASE))
        try:
            self.emulator_fps = float(emu.get_screen_rate())
        except Exception:
            self.emulator_fps = 60.0
        self._emulator = emu
        self._refresh_memory()
        self._refresh_frame()
        return emu

    # ---- emulator hooks ----------------------------------------------------------------
    def _refresh_frame(self) -> np.ndarray:
        if self._emulator is None:
            raise RuntimeError("SNES emulator is not initialized")
        frame = np.asarray(self._emulator.get_screen(), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"stable-retro returned non-RGB SNES frame {frame.shape}")
        self._frame_cache = frame.copy()
        return self._frame_cache

    def _refresh_memory(self) -> None:
        if self._game_data is not None:
            try:
                self._game_data.update_ram()
            except Exception:
                pass

    def _emu_step(self, n_frames: int) -> None:
        if self._emulator is None:
            raise RuntimeError("SNES emulator is not initialized")
        for _ in range(max(0, int(n_frames))):
            self._emulator.step()
        self._refresh_memory()
        self._refresh_frame()

    def _emu_screen_rgb(self) -> np.ndarray:
        if self._frame_cache is None:
            return self._refresh_frame()
        return self._frame_cache

    def _emu_set_buttons(self, buttons: set[str]) -> None:
        if self._emulator is None:
            return
        clean = {button.upper() for button in buttons if button.upper() in self._button_to_i}
        mask = np.zeros((len(self._button_order),), dtype=np.uint8)
        for button in clean:
            mask[self._button_to_i[button]] = 1
        self._emulator.set_button_mask(mask, 0)
        self._current_buttons = clean

    def _emu_save(self) -> bytes:
        if self._emulator is None:
            raise RuntimeError("SNES emulator is not initialized")
        core_state = bytes(self._emulator.get_state())
        frame = self.frame()
        header = {
            "core_len": len(core_state),
            "frame_shape": list(frame.shape),
            "frame_dtype": "uint8",
            "buttons": sorted(self._current_buttons),
            "system": self.system,
            "backend": "stable-retro",
        }
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return (
            STATE_MAGIC
            + struct.pack("<I", len(header_bytes))
            + header_bytes
            + core_state
            + frame.tobytes()
        )

    def _emu_load(self, state: bytes) -> None:
        if self._emulator is None:
            raise RuntimeError("SNES emulator is not initialized")
        buttons: Iterable[str] = ()
        frame: np.ndarray | None = None
        core_state = state
        if state.startswith(STATE_MAGIC):
            pos = len(STATE_MAGIC)
            (header_len,) = struct.unpack("<I", state[pos : pos + 4])
            pos += 4
            header = json.loads(state[pos : pos + header_len].decode("utf-8"))
            pos += header_len
            core_len = int(header["core_len"])
            core_state = state[pos : pos + core_len]
            pos += core_len
            shape = tuple(int(x) for x in header["frame_shape"])
            frame_bytes = state[pos:]
            expected = int(np.prod(shape))
            if len(frame_bytes) == expected:
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape).copy()
            buttons = header.get("buttons", ())

        ok = self._emulator.set_state(bytes(core_state))
        if ok is False:
            raise RuntimeError("stable-retro rejected SNES savestate bytes")
        self._emu_set_buttons(set(buttons))
        self._refresh_memory()
        self._frame_cache = frame if frame is not None else None
        if self._frame_cache is None:
            self._refresh_frame()

    # ---- public helpers ----------------------------------------------------------------
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

        for idx, snes_button in self._button_indices.items():
            if float(a[idx]) > 0.5:
                buttons.add(snes_button)
        return buttons

    def action_chunk_to_buttons(self, action_chunk: np.ndarray) -> set[str]:
        """Aggregate a whole chunk with Mednafen-style BUTTON_FRAC semantics."""
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
        for idx, snes_button in self._button_indices.items():
            if float((a[:, idx] > 0.5).mean()) >= BUTTON_FRAC:
                buttons.add(snes_button)
        return buttons

    def _memory_blocks(self) -> dict[int, Any]:
        self._refresh_memory()
        memory = getattr(self._game_data, "memory", None)
        blocks = getattr(memory, "blocks", {}) if memory is not None else {}
        return {int(base): block for base, block in blocks.items()}

    def read_memory(self, address: int, length: int) -> bytes:
        """Read `length` bytes from a stable-retro memory block by absolute SNES address."""
        if length < 0:
            raise ValueError("length must be non-negative")
        blocks = self._memory_blocks()
        addr = int(address)
        remaining = int(length)
        out = bytearray()
        while remaining:
            for base, block in sorted(blocks.items()):
                block_len = len(block)
                if base <= addr < base + block_len:
                    off = addr - base
                    take = min(remaining, block_len - off)
                    out.extend(bytes(block[off : off + take]))
                    addr += take
                    remaining -= take
                    break
            else:
                raise ValueError(f"Address range 0x{address:X}+{length} is not in exposed SNES memory")
        return bytes(out)

    def read_wram(self, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Return a copy of SNES WRAM bytes from 0x7E0000 + offset."""
        blocks = self._memory_blocks()
        block = blocks.get(self.wram_base)
        if block is None:
            return np.zeros((0,), dtype=np.uint8)
        arr = np.frombuffer(block, dtype=np.uint8)
        start = max(0, int(offset))
        end = arr.size if length is None else min(arr.size, start + max(0, int(length)))
        return arr[start:end].copy()

    def read_state(self) -> dict[str, Any]:
        wram = self.read_wram()
        sample = wram[:16]
        checksum_window = wram[: min(wram.size, 4096)].astype(np.uint32, copy=False)
        return {
            "backend": "stable-retro",
            "emulator_fps": self.emulator_fps,
            "buttons": sorted(self._current_buttons),
            "wram_base": self.wram_base,
            "wram_size": int(wram.size),
            "wram_first16": sample.tolist(),
            "wram_checksum4096": int(checksum_window.sum()) if checksum_window.size else 0,
        }

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._game_data = None
            self._frame_cache = None
            self._current_buttons = set()
            self._emulator = None
            gc.collect()


__all__ = ["SnesEnv", "EmulatorEnv", "USING_SHARED_EMULATOR_ENV"]
