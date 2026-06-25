"""Mega Man X8 16-bit fan game as a ProcGameEnv keyboard platformer.

The distributed Windows exe contains an embedded Godot PCK.  The PCK header is
Godot 3.5.3 format, so this runs it with the same native Linux Godot used by the
Castlevania env instead of Wine.
"""
from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_EAST, I_LSHLD, I_LTRIG, I_NORTH = 5, 7, 9, 10
I_RSHLD, I_RTRIG, I_SOUTH, I_START, I_WEST = 14, 16, 18, 19, 20
STICK_THRESH = 0.25


class MMX8Env(ProcGameEnv):
    name = "mmx8"
    window_name = "Mega Man X8"
    control = "keyboard"
    reset_by_relaunch = True             # respawn for a clean state, then skip the intro/title below
    restart_wait = 1.5                   # just enough for the Godot window; the macro drives the intro

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 6.0,
        godot_binary: str | None = None,
        exe_path: str | None = None,
        pck_path: str | None = None,
        **kw,
    ):
        repo = Path(__file__).resolve().parents[3]
        self.godot_binary = godot_binary or str(
            repo / "tmp/godot-3.5.3/Godot_v3.5.3-stable_x11.64"
        )
        self.exe_path = Path(
            exe_path
            or os.environ.get("MMX8_EXE")
            or "/tmp/mmx8/Mega Man X8 16-bit 1.0.0.9.exe"
        )
        self.pck_path = Path(
            pck_path
            or os.environ.get("MMX8_PCK")
            or repo / ".nitrogen-env-build/mmx8/game.pck"
        )
        self._ensure_pck()
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def _ensure_pck(self) -> None:
        if self._pck_ready():
            return
        if not self.exe_path.exists():
            raise FileNotFoundError(f"MMX8 exe not found: {self.exe_path}")
        self.pck_path.parent.mkdir(parents=True, exist_ok=True)
        with self.exe_path.open("rb") as src:
            src.seek(-12, os.SEEK_END)
            pck_size, magic = struct.unpack("<Q4s", src.read(12))
            if magic != b"GDPC":
                raise RuntimeError(f"MMX8 exe has no Godot PCK trailer: {self.exe_path}")
            pck_start = self.exe_path.stat().st_size - 12 - pck_size
            src.seek(pck_start)
            if src.read(4) != b"GDPC":
                raise RuntimeError(f"MMX8 embedded PCK magic missing at {pck_start}")
            src.seek(pck_start)
            with self.pck_path.open("wb") as dst:
                remaining = pck_size
                while remaining:
                    chunk = src.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError("MMX8 embedded PCK ended early")
                    dst.write(chunk)
                    remaining -= len(chunk)

        # Embedded PCK file offsets are absolute within the exe.  After carving to
        # a standalone PCK, rewrite the index offsets to be relative to the PCK.
        with self.pck_path.open("r+b") as pck:
            pck.seek(84)
            file_count = struct.unpack("<I", pck.read(4))[0]
            for _ in range(file_count):
                path_len = struct.unpack("<I", pck.read(4))[0]
                pck.seek(path_len, os.SEEK_CUR)
                offset_pos = pck.tell()
                offset, _size = struct.unpack("<QQ", pck.read(16))
                if offset >= pck_start:
                    pck.seek(offset_pos)
                    pck.write(struct.pack("<Q", offset - pck_start))
                    pck.seek(offset_pos + 16)
                pck.seek(16, os.SEEK_CUR)

    def _pck_ready(self) -> bool:
        if not self.pck_path.exists() or self.pck_path.stat().st_size == 0:
            return False
        try:
            with self.pck_path.open("rb") as pck:
                if pck.read(4) != b"GDPC":
                    return False
                pck.seek(84)
                file_count = struct.unpack("<I", pck.read(4))[0]
                for _ in range(file_count):
                    path_len = struct.unpack("<I", pck.read(4))[0]
                    path = pck.read(path_len).rstrip(b"\0").decode("utf-8", "replace")
                    offset, size = struct.unpack("<QQ", pck.read(16))
                    pck.seek(16, os.SEEK_CUR)
                    if path == "res://project.binary":
                        here = pck.tell()
                        pck.seek(offset)
                        ok = pck.read(4) == b"ECFG" and size > 0
                        pck.seek(here)
                        return ok
        except Exception:
            return False
        return False

    def launch_cmd(self):
        if not Path(self.godot_binary).exists() and shutil.which(self.godot_binary) is None:
            raise FileNotFoundError(f"Godot binary not found: {self.godot_binary}")
        return [
            "env",
            "LIBGL_ALWAYS_SOFTWARE=1",
            self.godot_binary,
            "--video-driver",
            "GLES3",
            "--audio-driver",
            "Dummy",
            "--main-pack",
            str(self.pck_path),
        ]

    def reset_macro(self, scenario):
        # mmx8 (Godot) boots through a splash -> title -> opening dialogue before gameplay. reset()
        # runs this UNFROZEN, so mash Enter+z with waits to skip all of it into the level (empirically
        # reaches gameplay ~12-14s of presses). reset_by_relaunch gives the clean starting state.
        macro = [("wait", 2.0)]
        for _ in range(8):
            macro += [("key", "Return"), ("key", "z"), ("wait", 1.8)]
        macro += [("wait", 1.0)]
        return macro

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "s",      # jump
                I_WEST: "d",       # fire
                I_RTRIG: "d",      # alternate model fire/attack affordance
                I_EAST: "a",       # dash
                I_LTRIG: "a",      # alternate dash affordance
                I_NORTH: "f",      # alt-fire / weapon action
                I_RSHLD: "f",
                I_LSHLD: "z",      # special/select-special
                I_START: "Return",
            },
        )
