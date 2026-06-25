"""thextech_memread.py — read TheXTech game state from process memory WITHOUT a source fork.

The previous approach patched TheXTech's source (`src/graphics/gfx_update.cpp`) to dump player state to
a file each frame — a maintained fork. This reads the SAME globals directly from the live process's
memory instead, so TheXTech can be built 100% stock (only requirement: a NON-STRIPPED build so the
global symbol addresses are resolvable — `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo`).

How it works
------------
TheXTech builds as a NON-PIE executable (`Type: EXEC`), so its global symbols sit at FIXED virtual
addresses == their link addresses (no ASLR for the main image). We:
  1. resolve the global addresses once from the binary with `nm` (Player, Lives, GameMenu, ...),
  2. open `/proc/<pid>/mem` of the running game (the env is the PARENT process, and Linux
     `yama ptrace_scope=1` permits a parent to read its child's memory), and
  3. read the fields at `addr + struct offset`, converting the engine's fixed-point coordinates.

Memory layout (verified on TheXTech build 2026-06, via gdb + live reads)
  * `Player` is a `RangeArr<Player_t,0,200>`; element i is at `&Player + i*sizeof(Player_t)`.
    `sizeof(Player_t) == 488`, `offsetof(Player_t, Location) == 248`. TheXTech is 1-indexed, so the
    human player is `Player[1]`.
  * `Location_t` = 6 × `num_t`: X@0 Y@8 Height@16 Width@24 SpeedX@32 SpeedY@40.
  * `num_t` is 32.32 FIXED-POINT (`int64_t i`, `operator double() = i / (1<<32)`), so a coordinate is
    `read_int64(addr) / 2**32`. (Confirmed in `lib/fixed_point.h`.)
  * `Player_t.Dead` (bool) @ offset 388.
  * Scalars: `Lives` int32, `GameMenu` bool, `LevelSelect` bool, `GamePaused` int32, `EndLevel` bool,
    `LevelBeatCode` int32, `numPlayers` int32.
If the struct layout changes in a future TheXTech version, re-derive offsets with:
    gdb -q -batch -ex 'print sizeof(Player_t)' -ex 'print/d &((Player_t*)0)->Location' \
        -ex 'print/d &((Player_t*)0)->Dead' <binary>
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

# --- struct layout constants (see module docstring) ---------------------------------------------
SIZEOF_PLAYER = 488
OFF_LOCATION = 248
OFF_DEAD = 388
LOC_X, LOC_Y, LOC_SPEEDX, LOC_SPEEDY = 0, 8, 32, 40
FIXED_SHIFT = 32                       # num_t is 32.32 fixed-point: real = raw / 2**32
PLAYER_INDEX = 1                       # TheXTech is 1-indexed; human player is Player[1]

# globals we resolve from the binary -> (symbol name as emitted by `nm`)
_SCALAR_SYMS = ["Lives", "GameMenu", "LevelSelect", "GamePaused", "EndLevel",
                "LevelBeatCode", "numPlayers", "Player"]


class TheXTechMemReader:
    """Resolves symbol addresses from a (non-stripped) TheXTech binary and reads live state from
    /proc/<pid>/mem. Construct with the binary path; call attach(pid) then read()."""

    def __init__(self, binary: str):
        self.binary = str(binary)
        self.addr = self._resolve_symbols(self.binary)
        self.pid: int | None = None

    @staticmethod
    def _resolve_symbols(binary: str) -> dict[str, int]:
        """nm the binary for the globals we need. Raises if the binary is stripped (no symbols)."""
        if not Path(binary).exists():
            raise FileNotFoundError(f"TheXTech binary not found: {binary}")
        out = subprocess.run(["nm", binary], capture_output=True, text=True)
        addr: dict[str, int] = {}
        wanted = set(_SCALAR_SYMS)
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2] in wanted:    # "<hexaddr> <type> <name>"
                addr[parts[2]] = int(parts[0], 16)
        missing = wanted - set(addr)
        if missing:
            raise RuntimeError(
                f"TheXTech symbols not found in {binary}: {sorted(missing)}. The binary must be "
                f"NON-STRIPPED — build with `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo`.")
        return addr

    def attach(self, pid: int) -> None:
        self.pid = int(pid)

    def detach(self) -> None:
        self.pid = None

    # ---- low-level reads ----
    # We open /proc/<pid>/mem FRESH per state read rather than holding a long-lived handle: a persistent
    # handle goes stale across a game relaunch (reset respawns the process), whereas a fresh open per
    # read is cheap and always valid. Reads are wrapped so a transient failure (process mid-load /
    # exited) just yields {} from read().
    def _rd(self, address: int, n: int) -> bytes:
        with open(f"/proc/{self.pid}/mem", "rb", 0) as mem:
            mem.seek(address)
            data = mem.read(n)
        if len(data) != n:
            raise OSError(f"short read at {hex(address)} ({len(data)}/{n})")
        return data

    def _fixed(self, address: int) -> float:
        return struct.unpack("<q", self._rd(address, 8))[0] / (1 << FIXED_SHIFT)

    def _i32(self, address: int) -> int:
        return struct.unpack("<i", self._rd(address, 4))[0]

    def _bool(self, address: int) -> int:
        return 1 if self._rd(address, 1)[0] else 0

    def read(self) -> dict:
        """Return the same dict shape the old file-export produced. Empty dict on any read failure
        (e.g. process not yet mapped / exited)."""
        if self.pid is None:
            return {}
        try:
            ploc = self.addr["Player"] + PLAYER_INDEX * SIZEOF_PLAYER + OFF_LOCATION
            pdead = self.addr["Player"] + PLAYER_INDEX * SIZEOF_PLAYER + OFF_DEAD
            x = self._fixed(ploc + LOC_X)
            y = self._fixed(ploc + LOC_Y)
            vx = self._fixed(ploc + LOC_SPEEDX)
            vy = self._fixed(ploc + LOC_SPEEDY)
            lives = self._i32(self.addr["Lives"])
            dead = self._bool(pdead)
            game_menu = self._bool(self.addr["GameMenu"])
            level_select = self._bool(self.addr["LevelSelect"])
            paused = self._i32(self.addr["GamePaused"])
            end_level = self._bool(self.addr["EndLevel"])
            beat_code = self._i32(self.addr["LevelBeatCode"])
            nplayers = self._i32(self.addr["numPlayers"])
        except Exception:
            return {}
        return {
            "x": round(x, 1), "y": round(y, 1), "vx": round(vx, 2), "vy": round(vy, 2),
            "lives": lives, "dead": dead,
            "in_menu": int(bool(game_menu or level_select or paused)),
            "won": int(end_level and beat_code > 0), "beat_code": beat_code,
            "nplayers": nplayers,
        }
