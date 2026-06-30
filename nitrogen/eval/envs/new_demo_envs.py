"""new_demo_envs.py -- frame-exact envs for the newly-added demo games (MMX, SMB All-Stars), wired so
make_env('mmx')/make_env('smbas') work for demo_bc.py + reach_eval (save/load_state, screen-x progress).

MMX uses the shipped stable-retro experimental integration (xpos/health/lives). SMB All-Stars has NO
integration; we attach a minimal data.json (mario world-x position) discovered at build time. Both expose the
RetroRLEnv-compatible interface demo_bc/eval expect: reset/load_state/save_state/frame/_var/action_row_to_buttons/
step + reward_var/life_var attrs + jump_button. Console->25-dim mapping matches retro_rl_env.

If SMB's progress var is wrong/unknown, eval still degrades gracefully (reward 0) but TRAINING (demo_bc loads
demo.npz directly) is unaffected.
"""
from __future__ import annotations
import gzip
import os

import numpy as np

import stable_retro as retro

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
_RDATA = os.path.join(os.path.dirname(retro.__file__), "data")

# 25-dim NitroGen action layout (matches retro_rl_env): JLX=21, JLY=22, jump dims south=18/east=5, run=west=20.
JLX, JLY = 21, 22
STICK = 0.2


class IntegrationEnv:
    """Minimal RetroRLEnv-compatible env over an arbitrary stable-retro integration dir."""
    def __init__(self, rom, idir, system="Snes", reward_var="xpos", life_var="health",
                 start_state=None, frames_per_row=4, jump_button="B", reward_scale=1.0,
                 death_penalty=5.0, stuck_penalty=0.05, stuck_eps=0.5, max_stuck=12, max_rows=256):
        self.rom_path = os.path.join(_R, rom) if not os.path.isabs(rom) else rom
        self.idir = idir
        self.reward_var = reward_var
        self.life_var = life_var
        self.jump_button = jump_button
        self.frames_per_row = frames_per_row
        self.reward_scale = reward_scale
        self.death_penalty = death_penalty
        self.stuck_penalty = stuck_penalty
        self.stuck_eps = stuck_eps
        self.max_stuck = max_stuck
        self.max_rows = max_rows
        self.done_on_life_zero = True

        self._emu = retro.RetroEmulator(self.rom_path)
        self._buttons = list(retro.get_system_info(system)["buttons"])
        self._bi = {b: i for i, b in enumerate(self._buttons)}
        self._gd = retro.data.GameData()
        scen = os.path.join(idir, "scenario.json")
        self._gd.load(os.path.join(idir, "data.json"), scen if os.path.exists(scen) else None)
        self._emu.configure_data(self._gd)
        # bundled start state (gz or raw)
        self._start = None
        if start_state:
            sp = start_state if os.path.isabs(start_state) else os.path.join(idir, start_state)
            if os.path.exists(sp):
                raw = open(sp, "rb").read()
                try:
                    self._start = gzip.decompress(raw)
                except Exception:
                    self._start = raw
        self._steps = 0
        self._stuck_count = 0
        self._last_reward_val = 0.0

    # ---- low level ----
    def _mask(self, names):
        m = np.zeros(len(self._buttons), np.uint8)
        for n in names:
            if n in self._bi:
                m[self._bi[n]] = 1
        return m

    def _emu_step(self, names, n):
        m = self._mask(names)
        for _ in range(n):
            self._emu.set_button_mask(m, 0)
            self._emu.step()
        self._gd.update_ram()

    def frame(self):
        return np.asarray(self._emu.get_screen(), np.uint8).copy()

    def _var(self, name):
        try:
            return float(self._gd.lookup_value(name))
        except Exception:
            return 0.0

    # ---- RetroRLEnv-compatible API ----
    def reset(self, state=None):
        if state is not None:
            self._emu.set_state(state)
        elif self._start is not None:
            self._emu.set_state(self._start)
        self._gd.update_ram()
        if self.frame().mean() < 1.0:
            self._emu_step([], 1)
        self._steps = 0
        self._stuck_count = 0
        self._last_reward_val = self._var(self.reward_var)
        return self.frame()

    def load_state(self, b):
        self._emu.set_state(b); self._gd.update_ram()
        self._last_reward_val = self._var(self.reward_var)

    def save_state(self):
        return bytes(self._emu.get_state())

    def action_row_to_buttons(self, row):
        a = np.asarray(row, np.float32).ravel()
        out = []
        lx, ly = float(a[JLX]), float(a[JLY])
        if lx < 0.5 - STICK:
            out.append("LEFT")
        elif lx > 0.5 + STICK:
            out.append("RIGHT")
        if ly < 0.5 - STICK:
            out.append("UP")
        elif ly > 0.5 + STICK:
            out.append("DOWN")
        if float(a[18]) > 0.5 or float(a[5]) > 0.5:
            out.append(self.jump_button)
        if float(a[20]) > 0.5:                  # west / run / shoot
            out.append("Y")
        return out

    def step(self, chunk):
        rows = np.asarray(chunk, np.float32)
        rows = rows[None] if rows.ndim == 1 else rows
        lives_before = self._var(self.life_var) if self.life_var else 0.0
        for r in rows:
            self._emu_step(self.action_row_to_buttons(r), self.frames_per_row)
            self._steps += 1
        rv = self._var(self.reward_var)
        dprog = rv - self._last_reward_val
        reward = self.reward_scale * dprog
        self._last_reward_val = rv
        died = False
        if self.life_var:
            lv = self._var(self.life_var)
            if lv < lives_before:
                reward -= self.death_penalty
            died = lv <= 0
        stuck = abs(dprog) < self.stuck_eps
        self._stuck_count = self._stuck_count + 1 if stuck else 0
        if stuck:
            reward -= self.stuck_penalty
        done = (self.done_on_life_zero and died) or self._steps >= self.max_rows or self._stuck_count >= self.max_stuck
        return self.frame(), float(reward), bool(done), {"prog": rv, "dprog": dprog, "died": died}

    def close(self):
        self._emu = None
        self._gd = None


def make_mmx():
    return IntegrationEnv(rom="Game data/Mega Man X.sfc",
                          idir=os.path.join(_RDATA, "experimental", "MegaManX-Snes"),
                          reward_var="xpos", life_var="health", start_state="Level1.state",
                          jump_button="B")


# SMB All-Stars: no shipped integration. We write a minimal data.json with a candidate progress var into
# tmp/retro_data and (optionally) a discovered RAM address. Address resolved by smbas_find_progress.py.
def make_smbas(prog_addr=None, prog_type="<u2", life_addr=None):
    idir = os.path.join(_R, "tmp", "retro_data", "SuperMarioAllStars-Snes")
    os.makedirs(idir, exist_ok=True)
    import json
    info = {}
    if prog_addr is not None:
        info["xpos"] = {"address": int(prog_addr), "type": prog_type}
    if life_addr is not None:
        info["lives"] = {"address": int(life_addr), "type": "|u1"}
    json.dump({"info": info}, open(os.path.join(idir, "data.json"), "w"))
    return IntegrationEnv(rom="Game data/Super Mario All-Stars.sfc", idir=idir,
                          reward_var="xpos" if prog_addr is not None else "none",
                          life_var="lives" if life_addr is not None else None,
                          start_state=None, jump_button="B")
