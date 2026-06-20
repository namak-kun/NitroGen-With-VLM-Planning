"""CaveStoryEnv — closed-loop GameEnv backed by doukutsu-rs (open-source Rust reimplementation
of the freeware Metroidvania Cave Story / Doukutsu Monogatari). Chosen as the first FOSS eval
env because it is: (1) genuinely free (freeware data, no ROM), (2) an open engine we can patch
to expose privileged state (player tile X/Y, map id, life, weapons, flags) for exact success
detection, (3) a 2D action-platformer with real exploration/branching -> planning matters, and
(4) gamepad-native and close to NitroGen's training distribution.

STATUS: integration scaffold. The doukutsu-rs side needs a small patch exposing a control
socket (inject inputs + read state + grab framebuffer). The Python side here speaks that
protocol. Until the engine patch lands, this class documents the exact contract and raises a
clear NotImplementedError so the harness fails loudly rather than silently.

Engine-side contract (to implement in a doukutsu-rs fork, e.g. a --agent-socket flag):
  - TCP/UDS line protocol, one JSON message per control tick.
  - Python -> engine: {"buttons": [..], "lstick": [x,y], "rstick": [x,y], "frames": N}
        held for N engine frames (N = engine_fps / action_hz, e.g. 50/30 ~ 2).
  - engine -> Python: {"frame_png_b64": "...", "state": {"x":..,"y":..,"map":..,"life":..,
        "max_life":..,"weapons":[..],"flags":[..]}, "done": bool}
  - reset: {"reset": {"save": "<scenario init key>"}} -> load a savestate/profile.
The mapping from NitroGen's 25-dim gamepad to Cave Story controls (jump=A, fire=B, weapon
switch=shoulders, d-pad/left-stick = move) is done in `_encode_action`.
"""
from __future__ import annotations

import numpy as np

from ..core import (
    JLX, JLY, N_BUTTONS, GameEnv, Observation, Scenario,
)

# Cave Story controls mapped onto NitroGen button indices (xbox-ish layout, see
# nitrogen/training/actions.py BUTTON_ORDER). Movement uses the left stick / d-pad.
CS_BUTTON_MAP = {
    "jump": 0,            # south / A
    "fire": 1,            # east / B
    "weapon_prev": 4,     # left shoulder
    "weapon_next": 5,     # right shoulder
    "inventory": 9,       # start
}
MOVE_THRESH = 0.4         # left-stick deflection that counts as a directional press


class CaveStoryEnv(GameEnv):
    name = "cave_story"
    action_hz = 30.0

    def __init__(self, engine_addr: str = "127.0.0.1:55355", engine_fps: int = 50,
                 out_size: int = 256, connect: bool = True):
        self.engine_addr = engine_addr
        self.engine_fps = engine_fps
        self.out_size = out_size
        self.hold_frames = max(1, round(engine_fps / self.action_hz))
        self._sock = None
        if connect:
            self._connect()

    # ---- engine link (to be implemented against the doukutsu-rs agent socket) ----------
    def _connect(self):
        raise NotImplementedError(
            "CaveStoryEnv needs the doukutsu-rs agent-socket patch. Build the fork with the "
            "--agent-socket flag (see module docstring for the protocol), then set connect=True. "
            "Until then use DummyGridEnv to exercise the harness.")

    def _send(self, msg: dict) -> dict:
        raise NotImplementedError

    # ---- action encoding ---------------------------------------------------------------
    def _encode_action(self, row: np.ndarray) -> dict:
        """One NitroGen-layout (25,) action -> a Cave Story input message."""
        btn = row[:N_BUTTONS]
        lx, ly = float(row[JLX]), float(row[JLY])
        pressed = {name: bool(btn[idx] > 0.5) for name, idx in CS_BUTTON_MAP.items()}
        move = {
            "left": lx < -MOVE_THRESH, "right": lx > MOVE_THRESH,
            "up": ly < -MOVE_THRESH, "down": ly > MOVE_THRESH,
        }
        return {"buttons": pressed, "move": move, "frames": self.hold_frames}

    # ---- GameEnv interface -------------------------------------------------------------
    def reset(self, scenario: Scenario) -> Observation:
        resp = self._send({"reset": {"save": scenario.init}})
        return self._to_obs(resp)

    def step(self, action_chunk: np.ndarray) -> Observation:
        resp = None
        for row in self.iter_steps(action_chunk):
            resp = self._send(self._encode_action(row))
        return self._to_obs(resp)

    def _to_obs(self, resp: dict) -> Observation:
        import base64, io
        from PIL import Image
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(resp["frame_png_b64"]))).convert("RGB"))
        return Observation(frame=img, state=resp.get("state", {}),
                           step_idx=resp.get("state", {}).get("step", 0),
                           done=bool(resp.get("done", False)))

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
