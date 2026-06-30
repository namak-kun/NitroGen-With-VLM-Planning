"""retro_rl_env.py -- a self-contained, RL-ready stable-retro env with a DENSE REWARD + frame-exact
save/load + our-own start states. Built for the RL loop (verl/prime-rl), NOT the counterfactual eval
harness (which has no reward). Works for any stable-retro game that ships an integration (data.json
reward variables); defaults to Sonic 2 (Genesis) — the classic gym-retro RL benchmark — which gives a
dense forward-progress reward (screen_x) with our ROM in Game data/.

Why self-contained (not subclassing eval EmulatorEnv): the eval envs expose frames+save/load but no
reward; RL needs (obs, reward, done). This loads the integration's data.json so named variables
(screen_x/score/rings/lives) become a reward, and supports OUR-OWN start states (the shipped .state
files are tied to a specific ROM dump and break on a different dump — verified — so we generate ours).

Key facts (verified 2026-06-26 on Game data/Sonic The Hedgehog 2.md):
- screen_x is a DENSE monotonic forward-progress reward (0->3->62->133->213->308->422 driving RIGHT).
- save_state()/load_state() are frame-exact (load -> screen_x resets exactly; roundtrip exact).
- a fresh boot + scripted START reaches gameplay; we snapshot that as the start state.

NitroGen action layout (per row, 25-dim): buttons[0:21], j_left[21:23] (x,y in [0,1], 0.5=neutral;
x<0.5=left, y<0.5=up), j_right[23:25]. We map stick -> d-pad and a jump-button dim -> the console jump.

Usage:
  from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
  env = RetroRLEnv()                      # Sonic 2, Genesis, dense screen_x reward
  obs = env.reset()                       # loads/creates the start state
  obs, reward, done, info = env.step(action_chunk)   # action_chunk (H,25) or (25,)
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
_RETRO_DATA = None  # filled on first use

# Small per-game RAM additions for integrations that ship only sparse score/coin
# variables. Addresses are stable-retro absolute SNES WRAM addresses (0x7e....).
_EXTRA_INFO: dict[str, dict[str, dict[str, Any]]] = {
    "SuperMarioWorld-Snes-v0": {
        "screen_x": {"address": 0x7E0094, "type": "<u2"},
        "_layer1_x": {"address": 0x7E001A, "type": "<u2"},
        "_level_mode": {"address": 0x7E0100, "type": "|u1"},
        "_message_box": {"address": 0x7E1426, "type": "|u1"},
        "_ow_x": {"address": 0x7E1F17, "type": "<u2"},
        "_ow_y": {"address": 0x7E1F19, "type": "<u2"},
    },
}

JLX, JLY = 21, 22
STICK_THRESH = 0.2
# NitroGen 25-dim button indices that should count as "jump/primary action" (SOUTH=B, EAST=A).
JUMP_DIMS = (18, 5)
# d-pad dims in the 25-dim action (DATASET_COL_TO_MODEL_DIM): down=1,left=2,right=3,up=4.
DPAD = {"DOWN": 1, "LEFT": 2, "RIGHT": 3, "UP": 4}


def _integration_dir(game: str) -> str:
    global _RETRO_DATA
    if _RETRO_DATA is None:
        import stable_retro as retro
        _RETRO_DATA = os.path.join(os.path.dirname(retro.__file__), "data", "stable")
    return os.path.join(_RETRO_DATA, game)


def _data_json(game: str, idir: str) -> str:
    base = os.path.join(idir, "data.json")
    extra = _EXTRA_INFO.get(game)
    if not extra:
        return base
    with open(base) as f:
        data = json.load(f)
    data.setdefault("info", {}).update(extra)
    out_dir = REPO / "tmp" / "retro_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{game}_data_extra.json"
    with open(out, "w") as f:
        json.dump(data, f)
    return str(out)


class RetroRLEnv:
    def __init__(
        self,
        rom_path: str = "Game data/Sonic The Hedgehog 2.md",
        game: str = "SonicTheHedgehog2-Genesis-v0",
        system: str = "Genesis",
        reward_var: str = "screen_x",
        reward_scale: float = 0.01,
        score_var: str | None = "score",
        score_scale: float = 0.01,
        life_var: str | None = "lives",
        done_on_life_zero: bool = True,
        death_penalty: float = 5.0,
        stuck_penalty: float = 0.05,
        stuck_eps: float = 0.5,
        max_stuck: int = 12,
        frames_per_row: int = 4,
        jump_button: str = "A",
        start_state: str | None = None,
        boot_start_frames: int = 600,
        max_rows: int = 256,
    ):
        import stable_retro as retro
        self._retro = retro
        rp = Path(rom_path).expanduser()
        if not rp.is_absolute():
            rp = REPO / rom_path
        if not rp.exists():
            raise FileNotFoundError(f"ROM not found: {rp}")
        self.rom_path = str(rp)
        self.game = game
        self.system = system
        self.reward_var = reward_var
        self.reward_scale = reward_scale
        self.score_var = score_var
        self.score_scale = score_scale
        self.life_var = life_var
        self.done_on_life_zero = done_on_life_zero
        self.death_penalty = death_penalty
        self.stuck_penalty = stuck_penalty
        self.stuck_eps = stuck_eps
        self.max_stuck = max_stuck
        self._stuck_count = 0
        self.frames_per_row = int(frames_per_row)
        self.jump_button = jump_button
        self.boot_start_frames = boot_start_frames
        self.max_rows = max_rows
        self.chunk_seconds = 0.6  # NitroGen chunk; informational

        self._emu = retro.RetroEmulator(self.rom_path)
        info = retro.get_system_info(system)
        self._buttons = list(info.get("buttons"))
        self._bi = {b: i for i, b in enumerate(self._buttons)}
        try:
            self.fps = float(self._emu.get_screen_rate())
        except Exception:
            self.fps = 60.0
        self._gd = retro.data.GameData()
        idir = _integration_dir(game)
        self._gd.load(_data_json(game, idir), os.path.join(idir, "scenario.json"))
        self._emu.configure_data(self._gd)

        start_kind = "smw_yi1_v3" if game == "SuperMarioWorld-Snes-v0" else "fresh"
        self._start_state_path = start_state or str(REPO / "tmp" / "states" /
                                                    f"{game}_{start_kind}.state")
        self._start_bytes: bytes | None = None
        self._steps = 0
        self._last_reward_val = 0.0
        self._last_score = 0.0

    # ---- low-level ---------------------------------------------------------------------
    def _mask(self, names) -> np.ndarray:
        m = np.zeros(len(self._buttons), np.uint8)
        for n in names:
            if n in self._bi:
                m[self._bi[n]] = 1
        return m

    def _emu_step(self, button_names, n_frames):
        m = self._mask(button_names)
        for _ in range(n_frames):
            self._emu.set_button_mask(m, 0)
            self._emu.step()
        self._gd.update_ram()

    def frame(self) -> np.ndarray:
        return np.asarray(self._emu.get_screen(), dtype=np.uint8).copy()

    def _var(self, name):
        try:
            return float(self._gd.lookup_value(name))
        except Exception:
            return 0.0

    def save_state(self) -> bytes:
        return bytes(self._emu.get_state())

    def load_state(self, state: bytes) -> None:
        self._emu.set_state(bytes(state))
        self._gd.update_ram()

    # ---- start state (our own; shipped .state are ROM-dump-specific -> break) -----------
    def _ensure_start_state(self) -> bytes:
        if self._start_bytes is not None:
            return self._start_bytes
        if os.path.exists(self._start_state_path):
            self._start_bytes = open(self._start_state_path, "rb").read()
            return self._start_bytes
        if self.game == "SuperMarioWorld-Snes-v0":
            self._boot_super_mario_world()
        else:
            # fresh boot + mash START to reach gameplay, then snapshot
            for i in range(self.boot_start_frames):
                names = ["START"] if (i // 20) % 2 == 0 else []
                self._emu.set_button_mask(self._mask(names), 0)
                self._emu.step()
            self._gd.update_ram()
        self._start_bytes = bytes(self._emu.get_state())
        os.makedirs(os.path.dirname(self._start_state_path), exist_ok=True)
        open(self._start_state_path, "wb").write(self._start_bytes)
        return self._start_bytes

    def _boot_super_mario_world(self) -> None:
        # Title -> file select -> Yoshi's House intro -> overworld -> Yoshi's Island 1.
        # The first ROM boot has empty SRAM, so this generates a gameplay state instead
        # of relying on stable-retro's hash-specific shipped states.
        self._emu_step([], 180)
        self._emu_step(["START"], 8)
        self._emu_step([], 160)
        self._emu_step(["START"], 8)
        self._emu_step([], 300)
        self._emu_step(["B"], 4)
        self._emu_step([], 120)
        self._emu_step(["B"], 4)

        # Dismiss the Yoshi's House message box once it accepts input, then wait for
        # the overworld map (mode 14). The input latch is delayed, so retry pulses.
        for _ in range(20):
            self._emu_step([], 120)
            if self._var("_level_mode") == 20 or self._var("_message_box"):
                self._emu_step(["B"], 8)
                self._emu_step([], 20)
            if self._var("_level_mode") == 14 and self._var("_message_box") == 0:
                break
        for _ in range(360):
            if self._var("_level_mode") == 14 and self._var("_ow_y") == 120:
                break
            self._emu_step([], 1)
        self._emu_step([], 60)

        # Move one overworld node right to Yoshi's Island 1 and enter it.
        for _ in range(8):
            if self._var("_ow_x") >= 140:
                break
            self._emu_step(["RIGHT"], 20)
        self._emu_step([], 60)
        if self._var("_ow_x") < 140:
            raise RuntimeError("SMW boot failed before Yoshi's Island 1: overworld did not move right")

        self._emu_step(["B"], 8)
        reached_yi1 = False
        for _ in range(1200):
            self._emu_step([], 1)
            reached_yi1 = (self._var("_ow_x") >= 140 and self._var("_level_mode") == 20
                           and self._var("_message_box") == 0 and self._var("lives") > 0
                           and self.frame().mean() > 50)
            if reached_yi1:
                break
        if not reached_yi1:
            raise RuntimeError("SMW boot failed: did not reach Yoshi's Island 1 gameplay")
        self._emu_step([], 30)

    # ---- NitroGen action -> console buttons --------------------------------------------
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
        if any(float(a[d]) > 0.5 for d in JUMP_DIMS):
            out.append(self.jump_button)
        return out

    # ---- RL interface ------------------------------------------------------------------
    def reset(self, state: bytes | None = None) -> np.ndarray:
        self.load_state(state if state is not None else self._ensure_start_state())
        if self.frame().mean() < 1.0:
            self._emu_step([], 1)
        self._steps = 0
        self._stuck_count = 0
        self._last_reward_val = self._var(self.reward_var)
        self._last_score = self._var(self.score_var) if self.score_var else 0.0
        return self.frame()

    def step(self, action_chunk: np.ndarray):
        """Apply an (H,25) NitroGen action chunk (or a single (25,) row). Returns
        (obs, reward, done, info). Reward = scale*Δprogress + score bonus - DEATH penalty - STUCK penalty
        (anti-exploit, per the rubber-duck: screen_x alone trains a fast-suicide policy). Success metrics
        (distance/death/stuck) are tracked SEPARATELY in info, not folded into reward shaping decisions."""
        a = np.asarray(action_chunk, dtype=np.float32)
        rows = a[None] if a.ndim == 1 else a
        lives_before = self._var(self.life_var) if self.life_var else 0.0
        for row in rows:
            self._emu_step(self.action_row_to_buttons(row), self.frames_per_row)
            self._steps += 1
        rv = self._var(self.reward_var)
        dprog = rv - self._last_reward_val
        reward = self.reward_scale * dprog
        self._last_reward_val = rv
        if self.score_var:
            sc = self._var(self.score_var)
            reward += self.score_scale * max(0.0, sc - self._last_score)
            self._last_score = sc
        # anti-exploit shaping
        died = False
        if self.life_var:
            lv = self._var(self.life_var)
            if lv < lives_before:
                reward -= self.death_penalty            # lost a life
            died = lv <= 0
        stuck = abs(dprog) < self.stuck_eps
        if stuck:
            reward -= self.stuck_penalty
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        done = (self.done_on_life_zero and died) or self._steps >= self.max_rows \
            or self._stuck_count >= self.max_stuck
        info = {"steps": self._steps, self.reward_var: rv, "dprog": dprog, "died": died, "stuck": stuck,
                "stuck_count": self._stuck_count,
                **{k: self._var(k) for k in ("score", "rings", "lives", "act") if self.score_var}}
        return self.frame(), float(reward), bool(done), info

    def close(self):
        self._emu = None
        self._gd = None


if __name__ == "__main__":
    # smoke test: dense reward should be positive when driving right from the start state
    env = RetroRLEnv()
    obs = env.reset()
    print("reset frame:", obs.shape, "start", env.reward_var, "=", env._last_reward_val)
    right = np.full((18, 25), 0.5, np.float32)
    right[:, JLX] = 1.0  # hold RIGHT
    right[:, 18] = 1.0   # hold jump (helps Sonic over bumps)
    total = 0.0
    for t in range(6):
        obs, r, done, info = env.step(right)
        total += r
        print(f"  step{t} reward={r:+.3f} screen_x={info.get('screen_x'):.0f} done={done}")
    print(f"total reward driving RIGHT (should be >0): {total:+.3f}")
    # GRPO substrate check: save, perturb, restore
    s = env.save_state(); x_save = env._var(env.reward_var)
    env.step(right); env.load_state(s)
    print(f"save/load roundtrip: screen_x {x_save:.0f} -> {env._var(env.reward_var):.0f} (exact)")
