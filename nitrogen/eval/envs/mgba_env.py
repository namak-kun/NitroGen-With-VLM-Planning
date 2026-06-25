"""mGBA-backed in-process env for GB, GBC, and GBA NitroGen eval/RL.

mGBA is a multi-system core: this single backend loads `.gb`, `.gbc`, and
`.gba` ROMs and lets the core auto-detect the concrete system.  The `system`
argument is retained as a hint for metadata only; ROM extension and mGBA's
loader decide the actual core.

Validated API in this repo: `pygba==0.2.4` wrapping raw `mgba==0.10.5`
bindings.  The PyPI `mgba` wheel was unavailable for CPython 3.12 here, so the
repo-local mGBA 0.10.5 Python build was installed into `.venv`.  API quirks:
`core.save_raw_state()/load_raw_state()` provide frame-exact core state, but
`load_raw_state()` does not repaint the Python framebuffer immediately.  Like
`snes_env.py`, this module wraps savestates with the last RGB frame cache so
`load_state(); frame()` returns exactly to the saved snapshot.

Frame cadence: default `frames_per_row=2`, matching NitroGen's 30 Hz action
rows against the ~60 Hz GB/GBC/GBA video cadence.

Button map: NitroGen SOUTH->A, WEST->B, LEFT_SHOULDER->L, RIGHT_SHOULDER->R,
START->START, BACK/SELECT->SELECT, and left-stick/DPAD directions -> d-pad.
GB/GBC cores ignore L/R.  The stick convention is NitroGen's 0.5-neutral
`JLX/JLY` range with +/-0.2 threshold inherited from `EmulatorEnv`.

Screen buffers are returned as HxWx3 `np.uint8` RGB frames.  mGBA reports
GB/GBC dimensions as 160x144 and GBA as 240x160 (width x height), so frames are
normally shaped `(144, 160, 3)` for GB/GBC and `(160, 240, 3)` for GBA.
"""
from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import struct
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .emulator_env import DEFAULT_FRAMES_PER_ROW, EmulatorEnv


REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROM_PATH = str(REPO / "tmp" / "roms" / "BlindJump.gba")
STATE_MAGIC = b"NITROGEN-MGBA-STATE-v1\0"
DEFAULT_BOOT_FRAMES = 1
DEFAULT_EMULATOR_FPS = 60.0
SUPPORTED_SUFFIXES = {".gb", ".gbc", ".gba"}


class MgbaEnv(EmulatorEnv):
    """Headless, in-process mGBA env with frame stepping, savestates, and RAM reads."""

    name = "mgba"
    system = "auto"
    action_hz = 30.0
    button_map = {
        "SOUTH": "A",
        "WEST": "B",
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

    def __init__(
        self,
        rom_path: str = DEFAULT_ROM_PATH,
        *,
        system: str | None = None,
        frames_per_row: int = DEFAULT_FRAMES_PER_ROW,
        boot_frames: int = DEFAULT_BOOT_FRAMES,
        **kwargs: Any,
    ):
        self.system_hint = system or "auto"
        self.boot_frames = int(boot_frames)
        self.emulator_fps = DEFAULT_EMULATOR_FPS
        self.backend_name = ""
        self.backend_version = ""
        self.mgba_version = ""
        self.pygba_version = ""
        self.detected_system = ""
        self.screen_size = (0, 0)
        self._pygba = None
        self._mgba = None
        self._core = None
        self._framebuffer = None
        self._frame_cache: np.ndarray | None = None
        self._current_buttons: set[str] = set()
        self._button_to_key: dict[str, int] = {}
        super().__init__(
            rom_path=str(Path(rom_path).expanduser()),
            system=system or "auto",
            frames_per_row=frames_per_row,
            **kwargs,
        )

    # ---- backend setup -----------------------------------------------------------------
    def _prepare_headless(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )

    def _import_mgba_modules(self):
        import mgba
        import mgba.core
        import mgba.image

        try:
            import mgba.log

            mgba.log.silence()
        except Exception:
            pass
        return mgba

    def _make_emulator(self, rom_path: str, system: str | None = None):
        self._prepare_headless()
        path = Path(rom_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"mGBA ROM not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise ValueError(f"MgbaEnv expects one of {supported}, got {path}")

        mgba = self._import_mgba_modules()
        self._mgba = mgba
        self.mgba_version = str(getattr(mgba, "__version__", "unknown"))
        core = None

        try:
            import pygba
            from pygba import PyGBA

            wrapper = PyGBA.load(str(path), autoload_save=False)
            core = wrapper.core
            self._pygba = wrapper
            try:
                pygba_version = importlib.metadata.version("pygba")
            except importlib.metadata.PackageNotFoundError:
                pygba_version = getattr(pygba, "__version__", "unknown")
            self.pygba_version = str(pygba_version)
            self.backend_name = "pygba"
            self.backend_version = self.pygba_version
        except Exception:
            self._pygba = None
            core = mgba.core.load_path(str(path))
            self.backend_name = "mgba"
            self.backend_version = self.mgba_version

        if core is None:
            raise ValueError(f"mGBA failed to load ROM: {path}")

        self._core = core
        self.detected_system = self._detect_system(core, path)
        self._configure_keys(core)
        width, height = (int(v) for v in core.desired_video_dimensions())
        self.screen_size = (width, height)
        self._framebuffer = mgba.image.Image(width, height)
        core.set_video_buffer(self._framebuffer)
        core.reset()
        if self.boot_frames > 0:
            self._run_frames(self.boot_frames)
        self._refresh_frame()
        return core

    def _detect_system(self, core: Any, path: Path) -> str:
        suffix = path.suffix.lower()
        module = type(core).__module__.lower().split(".")[-1]
        name = type(core).__name__.lower()
        if module == "gba" or name == "gba":
            return "gba"
        if module == "gb" or name == "gb":
            return "gbc" if suffix == ".gbc" else "gb"
        platform = getattr(core, "platform", None)
        core_cls = getattr(getattr(self._mgba, "core", None), "Core", None)
        if core_cls is not None:
            if platform == getattr(core_cls, "PLATFORM_GBA", object()):
                return "gba"
            if platform == getattr(core_cls, "PLATFORM_GB", object()):
                return "gbc" if suffix == ".gbc" else "gb"
        return suffix.lstrip(".") or "unknown"

    def _configure_keys(self, core: Any) -> None:
        self._button_to_key = {}
        for button in ("A", "B", "L", "R", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"):
            attr = f"KEY_{button}"
            if hasattr(core, attr):
                self._button_to_key[button] = int(getattr(core, attr))
        # GB/GBC classes do not expose L/R; leave them unmapped so they are ignored.

    # ---- emulator hooks ----------------------------------------------------------------
    def _run_frames(self, n_frames: int) -> None:
        if self._core is None:
            raise RuntimeError("mGBA core is not initialized")
        for _ in range(max(0, int(n_frames))):
            self._core.run_frame()

    def _refresh_frame(self) -> np.ndarray:
        if self._framebuffer is None:
            raise RuntimeError("mGBA framebuffer is not initialized")
        image = self._framebuffer.to_pil().convert("RGB")
        frame = np.asarray(image, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"mGBA returned non-RGB frame {frame.shape}")
        self._frame_cache = frame.copy()
        return self._frame_cache

    def _emu_step(self, n_frames: int) -> None:
        self._run_frames(n_frames)
        self._refresh_frame()

    def _emu_screen_rgb(self) -> np.ndarray:
        if self._frame_cache is None:
            return self._refresh_frame()
        return self._frame_cache

    def _emu_set_buttons(self, buttons: set[str]) -> None:
        if self._core is None:
            return
        clean = {button.upper() for button in buttons}
        key_ids = [self._button_to_key[button] for button in sorted(clean) if button in self._button_to_key]
        self._core.set_keys(*key_ids)
        self._current_buttons = {button for button in clean if button in self._button_to_key}

    def _emu_save(self) -> bytes:
        if self._core is None:
            raise RuntimeError("mGBA core is not initialized")
        raw_state = self._core.save_raw_state()
        if raw_state is None:
            raise RuntimeError("mGBA failed to create a savestate")
        core_state = bytes(raw_state)
        frame = self.frame()
        header = {
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "mgba_version": self.mgba_version,
            "pygba_version": self.pygba_version,
            "core_len": len(core_state),
            "frame_shape": list(frame.shape),
            "frame_dtype": "uint8",
            "buttons": sorted(self._current_buttons),
            "system_hint": self.system_hint,
            "detected_system": self.detected_system,
            "screen_size": list(self.screen_size),
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
        if self._core is None:
            raise RuntimeError("mGBA core is not initialized")
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

        ok = self._core.load_raw_state(bytes(core_state))
        if ok is False:
            raise RuntimeError("mGBA rejected savestate bytes")
        self._emu_set_buttons(set(buttons))
        self._frame_cache = frame if frame is not None else None
        if self._frame_cache is None:
            self._refresh_frame()

    # ---- public helpers ----------------------------------------------------------------
    def _memory_regions(self) -> list[tuple[str, Any]]:
        core = self._core
        memory = getattr(core, "memory", None)
        if memory is None:
            return []
        names = (
            "wram",
            "iwram",
            "vram",
            "sram",
            "hram",
            "io",
            "oam",
            "palette",
            "cart",
            "rom",
        )
        regions = []
        for name in names:
            region = getattr(memory, name, None)
            if region is not None and hasattr(region, "base") and hasattr(region, "size"):
                regions.append((name, region))
        return regions

    def read_memory(self, address: int, length: int) -> bytes:
        """Read bytes through mGBA's exposed memory API using an absolute bus address."""
        if length < 0:
            raise ValueError("length must be non-negative")
        addr = int(address)
        remaining = int(length)
        out = bytearray()
        regions = self._memory_regions()
        while remaining:
            for _, region in regions:
                base = int(region.base)
                size = int(region.size)
                if base <= addr < base + size:
                    off = addr - base
                    take = min(remaining, size - off)
                    out.extend(bytes(region[off : off + take]))
                    addr += take
                    remaining -= take
                    break
            else:
                if self._pygba is not None and hasattr(self._pygba, "read_memory"):
                    out.extend(bytes(self._pygba.read_memory(addr, remaining)))
                    remaining = 0
                else:
                    raise ValueError(f"Address range 0x{address:X}+{length} is not exposed by mGBA")
        return bytes(out)

    def _default_memory_sample(self) -> tuple[int, bytes]:
        if self.detected_system == "gba":
            address = 0x02000000
        else:
            address = 0xC000
        try:
            return address, self.read_memory(address, 16)
        except Exception:
            return address, b""

    def read_state(self) -> dict[str, Any]:
        address, sample = self._default_memory_sample()
        checksum = 0
        try:
            checksum = int(sum(self.read_memory(address, 256)))
        except Exception:
            checksum = int(sum(sample))
        frame_counter = None
        try:
            frame_counter = int(self._core.frame_counter) if self._core is not None else None
        except Exception:
            pass
        return {
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "mgba_version": self.mgba_version,
            "pygba_version": self.pygba_version,
            "system_hint": self.system_hint,
            "detected_system": self.detected_system,
            "emulator_fps": self.emulator_fps,
            "screen_size": self.screen_size,
            "frame_counter": frame_counter,
            "game_title": getattr(self._core, "game_title", "") if self._core is not None else "",
            "game_code": getattr(self._core, "game_code", "") if self._core is not None else "",
            "buttons": sorted(self._current_buttons),
            "memory_sample_address": address,
            "memory_first16": list(sample),
            "memory_checksum256": checksum,
        }

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._pygba = None
            self._mgba = None
            self._core = None
            self._framebuffer = None
            self._frame_cache = None
            self._current_buttons = set()
            self._button_to_key = {}
            gc.collect()


__all__ = ["MgbaEnv"]
