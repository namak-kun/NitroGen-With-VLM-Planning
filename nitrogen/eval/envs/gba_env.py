"""GBA stable-retro RL/eval envs with direct RAM rewards.

Uses stable-retro's mGBA libretro core (RetroEmulator) and repo-local minimal
integrations under nitrogen/eval/envs/retro_integrations. GameData maps GBA
absolute bus addresses directly (EWRAM 0x02000000, IWRAM 0x03000000, etc.);
no shipped integration or external mGBA Python module is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[3]
INTEGRATIONS = Path(__file__).resolve().parent / "retro_integrations"

JLX, JLY = 21, 22
STICK_THRESH = 0.2
DPAD = {"DOWN": 1, "LEFT": 2, "RIGHT": 3, "UP": 4}
PRIMARY_DIMS = (18, 5)  # NitroGen south/east face buttons -> GBA A
SECONDARY_DIMS = (6, 4)  # optional -> GBA B when present in model rows


def _num(d: dict, k: str, default=0.0) -> float:
    try:
        return float((d or {}).get(k, default))
    except (TypeError, ValueError):
        return float(default)


def reward_topdown(before: dict, after: dict) -> tuple[float, bool]:
    dpos = ((_num(after, "x") - _num(before, "x")) ** 2 +
            (_num(after, "y") - _num(before, "y")) ** 2) ** 0.5
    reward = 0.01 * dpos
    if ((after or {}).get("area"), (after or {}).get("room")) != ((before or {}).get("area"), (before or {}).get("room")):
        reward += 5.0
    if ((after or {}).get("map_bank"), (after or {}).get("map_id")) != ((before or {}).get("map_bank"), (before or {}).get("map_id")):
        reward += 5.0
    return reward, False


def reward_fire_emblem(before: dict, after: dict) -> tuple[float, bool]:
    dpos = abs(_num(after, "cursor_x") - _num(before, "cursor_x")) + abs(_num(after, "cursor_y") - _num(before, "cursor_y"))
    reward = 0.05 * dpos
    if _num(after, "turn") > _num(before, "turn"):
        reward += 1.0
    if _num(after, "chapter") != _num(before, "chapter"):
        reward += 5.0
    return reward, False


@dataclass(frozen=True)
class GbaGameConfig:
    name: str
    rom_path: str
    integration: str
    reward_fn: Callable[[dict, dict], tuple[float, bool]]
    start_kind: str
    boot_note: str
    frames_per_row: int = 4
    max_rows: int = 512
    real_reward: bool = True


GBA_GAMES: dict[str, GbaGameConfig] = {
    "minish_cap": GbaGameConfig(
        name="minish_cap",
        rom_path="Game data/Legend of Zelda, The - The Minish Cap.gba",
        integration="MinishCap-GbAdvance",
        reward_fn=reward_topdown,
        start_kind="bedroom_v1",
        boot_note="fresh boot -> START -> create file named A -> start file -> mash A/START through intro to bedroom control",
    ),
    "pokemon_emerald": GbaGameConfig(
        name="pokemon_emerald",
        rom_path="Game data/Pokemon - Emerald Version.gba",
        integration="PokemonEmerald-GbAdvance",
        reward_fn=reward_topdown,
        start_kind="boot_stub_v1",
        boot_note="fresh boot + START/A title/intro macro; reward addresses wired but no controllable start-state solved yet",
        real_reward=False,
    ),
    "fire_emblem_sacred_stones": GbaGameConfig(
        name="fire_emblem_sacred_stones",
        rom_path="Game data/Fire Emblem - The Sacred Stones.gba",
        integration="FireEmblemSacredStones-GbAdvance",
        reward_fn=reward_fire_emblem,
        start_kind="boot_stub_v1",
        boot_note="fresh boot + START/A title macro; cursor reward addresses wired best-effort, gameplay start unresolved",
        real_reward=False,
    ),
}


class GbaRLEnv:
    def __init__(self, game: str = "minish_cap", start_state: str | None = None, frames_per_row: int | None = None):
        import stable_retro as retro
        if game not in GBA_GAMES:
            raise ValueError(f"unknown GBA game {game!r}; have {sorted(GBA_GAMES)}")
        self.cfg = GBA_GAMES[game]
        self.game = game
        self._retro = retro
        rp = Path(self.cfg.rom_path)
        if not rp.is_absolute():
            rp = REPO / rp
        if not rp.exists():
            raise FileNotFoundError(rp)
        self.rom_path = str(rp)
        self._emu = retro.RetroEmulator(self.rom_path)
        info = retro.get_system_info("GbAdvance")
        self._buttons = list(info["buttons"])
        self._bi = {b: i for i, b in enumerate(self._buttons) if b}
        self._gd = retro.data.GameData()
        idir = INTEGRATIONS / self.cfg.integration
        self._gd.load(str(idir / "data.json"), str(idir / "scenario.json"))
        self._emu.configure_data(self._gd)
        self.frames_per_row = int(frames_per_row or self.cfg.frames_per_row)
        self.max_rows = self.cfg.max_rows
        self._start_state_path = start_state or str(REPO / "tmp" / "states" / f"{self.cfg.integration}_{self.cfg.start_kind}.state")
        self._start_bytes: bytes | None = None
        self._prev: dict = {}
        self._steps = 0
        self.chunk_seconds = 0.6

    def _mask(self, names) -> np.ndarray:
        m = np.zeros(len(self._buttons), np.uint8)
        for n in names:
            if n in self._bi:
                m[self._bi[n]] = 1
        return m

    def _emu_step(self, button_names, n_frames: int) -> None:
        m = self._mask(button_names)
        for _ in range(int(n_frames)):
            self._emu.set_button_mask(m, 0)
            self._emu.step()
        self._gd.update_ram()

    def _vars(self) -> dict:
        self._gd.update_ram()
        try:
            return {k: float(v) for k, v in self._gd.lookup_all().items()}
        except Exception:
            return {}

    def _var(self, name: str) -> float:
        try:
            return float(self._gd.lookup_value(name))
        except Exception:
            return 0.0

    def frame(self) -> np.ndarray:
        return np.asarray(self._emu.get_screen(), dtype=np.uint8).copy()

    def save_state(self) -> bytes:
        return bytes(self._emu.get_state())

    def load_state(self, state: bytes) -> None:
        self._emu.set_state(bytes(state))
        self._gd.update_ram()

    def _ensure_start_state(self) -> bytes:
        if self._start_bytes is not None:
            return self._start_bytes
        p = Path(self._start_state_path)
        if p.exists():
            self._start_bytes = p.read_bytes()
            return self._start_bytes
        if self.game == "minish_cap":
            self._boot_minish_cap()
        elif self.game == "pokemon_emerald":
            self._boot_title_stub()
        else:
            self._boot_title_stub()
        self._start_bytes = self.save_state()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self._start_bytes)
        return self._start_bytes

    def _boot_title_stub(self) -> None:
        self._emu_step([], 240)
        for i in range(80):
            self._emu_step(["START"] if i % 6 == 0 else (["A"] if i % 3 == 0 else []), 30)

    def _boot_minish_cap(self) -> None:
        # Title/file-select/name-entry: choose file 1, accept a one-letter "A" name, start it.
        for names, frames in [
            ([], 180), (["START"], 10), ([], 120), (["A"], 10), ([], 120), (["A"], 10),
            ([], 120), (["A"], 8), ([], 30), (["START"], 8), ([], 120), (["A"], 8),
            ([], 160), (["A"], 8), ([], 600),
        ]:
            self._emu_step(names, frames)
        # Dismiss intro/Princess Zelda text until Link is controllable in the bedroom.
        for i in range(220):
            names = ["A"] if i % 4 == 0 else (["START"] if i % 11 == 0 else [])
            self._emu_step(names, 90)
        self._emu_step([], 60)

    def action_row_to_buttons(self, row: np.ndarray) -> list[str]:
        a = np.asarray(row, dtype=np.float32).ravel()
        out: list[str] = []
        lx, ly = float(a[JLX]), float(a[JLY])
        if lx < 0.5 - STICK_THRESH or a[DPAD["LEFT"]] > 0.5:
            out.append("LEFT")
        elif lx > 0.5 + STICK_THRESH or a[DPAD["RIGHT"]] > 0.5:
            out.append("RIGHT")
        if ly < 0.5 - STICK_THRESH or a[DPAD["UP"]] > 0.5:
            out.append("UP")
        elif ly > 0.5 + STICK_THRESH or a[DPAD["DOWN"]] > 0.5:
            out.append("DOWN")
        if any(d < len(a) and float(a[d]) > 0.5 for d in PRIMARY_DIMS):
            out.append("A")
        if any(d < len(a) and float(a[d]) > 0.5 for d in SECONDARY_DIMS):
            out.append("B")
        return out

    def reset(self, state: bytes | None = None) -> np.ndarray:
        self.load_state(state if state is not None else self._ensure_start_state())
        if self.frame().mean() < 1.0:
            self._emu_step([], 1)
        self._prev = self._vars()
        self._steps = 0
        return self.frame()

    def step(self, action_chunk: np.ndarray):
        rows = np.asarray(action_chunk, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows[None]
        before = self._prev
        for row in rows:
            self._emu_step(self.action_row_to_buttons(row), self.frames_per_row)
            self._steps += 1
        after = self._vars()
        reward, done = self.cfg.reward_fn(before, after)
        self._prev = after
        if self._steps >= self.max_rows:
            done = True
        info = {"steps": self._steps, "state": after, "real_reward": self.cfg.real_reward, "boot_note": self.cfg.boot_note}
        return self.frame(), float(reward), bool(done), info

    def close(self):
        self._emu = None
        self._gd = None


def make_gba_env(name: str) -> GbaRLEnv:
    aliases = {
        "gba_minish_cap": "minish_cap",
        "gba_pokemon_emerald": "pokemon_emerald",
        "gba_fire_emblem_sacred_stones": "fire_emblem_sacred_stones",
    }
    return GbaRLEnv(aliases.get(name, name))


class GbaEvalEnv:
    """GameEnv-compatible adapter for the counterfactual/eval harness."""

    action_hz = 30.0

    def __init__(self, game: str = "minish_cap"):
        from nitrogen.eval.core import Observation
        self._Observation = Observation
        self.rl = GbaRLEnv(game)
        self.name = f"gba_{self.rl.game}"

    def reset(self, scenario=None):
        init = getattr(scenario, "init", None)
        frame = self.rl.reset(init if isinstance(init, (bytes, bytearray)) else None)
        return self._Observation(frame=frame, state=dict(self.rl._prev), step_idx=0, done=False)

    def step(self, action_chunk: np.ndarray):
        frame, _, done, info = self.rl.step(action_chunk)
        return self._Observation(frame=frame, state=info.get("state", {}), step_idx=info.get("steps", 0),
                                 done=done)

    def read_state(self) -> dict:
        return self.rl._vars()

    def save_state(self) -> bytes:
        return self.rl.save_state()

    def load_state(self, state: bytes) -> None:
        self.rl.load_state(state)

    def close(self):
        self.rl.close()


def make_gba_eval_env(name: str) -> GbaEvalEnv:
    aliases = {
        "gba_minish_cap": "minish_cap",
        "gba_pokemon_emerald": "pokemon_emerald",
        "gba_fire_emblem_sacred_stones": "fire_emblem_sacred_stones",
    }
    return GbaEvalEnv(aliases.get(name, name))


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "minish_cap"
    env = make_gba_env(game)
    obs = env.reset()
    print(f"{game} reset frame={obs.shape} state={env._prev} real_reward={env.cfg.real_reward}")
    right = np.full((18, 25), 0.5, np.float32); right[:, JLX] = 1.0
    down = np.full((18, 25), 0.5, np.float32); down[:, JLY] = 1.0
    total = 0.0
    for t, act in enumerate([right, down, right, down]):
        obs, r, done, info = env.step(act)
        total += r
        print(f"  step{t} reward={r:+.3f} state={info['state']} done={done}")
    s = env.save_state(); st = dict(env._prev)
    env.step(right); env.load_state(s); rt = env._vars()
    print(f"save/load exact vars: {st} -> {rt}")
    print(f"total={total:+.3f}")
