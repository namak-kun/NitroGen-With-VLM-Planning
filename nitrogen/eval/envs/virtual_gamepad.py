"""Virtual Xbox-360 gamepad via Linux uinput, for feeding NitroGen's GAMEPAD action faithfully
to SDL2 games (doukutsu-rs etc.) — the correct embodiment match (NitroGen outputs analog sticks
+ buttons; we drive analog axes + buttons, NOT discretized keyboard keys).

We mimic a Microsoft Xbox 360 controller (VID 0x045e / PID 0x028e) with the standard axis/button
layout, so SDL2's GameController subsystem recognizes it via its built-in mapping (is_game_controller
== true) with no custom gamecontrollerdb entry needed.

CRITICAL: NitroGen's 25-dim action uses buttons[0:21] in `BUTTON_ACTION_TOKENS` order (back=0,
east=5, guide=6, north=10, south=18, start=19, west=20, ... — NOT an xbox-physical order!), then
j_left[21:23] (x,y in [-1,1], -x=left/-y=up), j_right[23:25]. We map button NAME -> evdev code
(via the canonical order) so the indices stay correct; getting this wrong silently presses the
wrong buttons (e.g. opening menus). The named B_* constants below are the canonical indices for
reset macros.
"""
from __future__ import annotations

import time

from evdev import UInput, ecodes as e
from nitrogen.shared import BUTTON_ACTION_TOKENS

# canonical NitroGen button index for each name (lowercased, matches BUTTON_ORDER)
_NAME2IDX = {t.lower(): i for i, t in enumerate(BUTTON_ACTION_TOKENS)}

# named indices into the 25-dim action vector (for reset macros / masking)
B_SOUTH = _NAME2IDX["south"]      # A / jump / menu_ok
B_EAST = _NAME2IDX["east"]        # B / menu_back
B_WEST = _NAME2IDX["west"]        # X / skip
B_NORTH = _NAME2IDX["north"]      # Y
B_BACK = _NAME2IDX["back"]
B_START = _NAME2IDX["start"]
B_GUIDE = _NAME2IDX["guide"]
B_DUP = _NAME2IDX["dpad_up"]
B_DDOWN = _NAME2IDX["dpad_down"]
B_DLEFT = _NAME2IDX["dpad_left"]
B_DRIGHT = _NAME2IDX["dpad_right"]
MENU_BUTTONS = (B_BACK, B_GUIDE, B_START)  # zero these to stop the model opening menus (NO_MENU)

AXIS_MAX = 32767
JLX, JLY, JRX, JRY = 21, 22, 23, 24

# button NAME -> evdev button code (digital)
_BTN_EVDEV = {
    "south": e.BTN_A, "east": e.BTN_B, "west": e.BTN_X, "north": e.BTN_Y,
    "left_shoulder": e.BTN_TL, "right_shoulder": e.BTN_TR,
    "back": e.BTN_SELECT, "start": e.BTN_START, "guide": e.BTN_MODE,
    "left_thumb": e.BTN_THUMBL, "right_thumb": e.BTN_THUMBR,
}
_BTN_PAIRS = [(_NAME2IDX[n], code) for n, code in _BTN_EVDEV.items() if n in _NAME2IDX]
_I_LTRIG = _NAME2IDX.get("left_trigger")
_I_RTRIG = _NAME2IDX.get("right_trigger")


class VirtualGamepad:
    """A uinput-backed Xbox-360-style controller. Call set_action(row25) each control step."""

    def __init__(self, name: str = "Xbox 360 Controller"):
        cap = {
            e.EV_KEY: [
                e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y,
                e.BTN_TL, e.BTN_TR, e.BTN_SELECT, e.BTN_START, e.BTN_MODE,
                e.BTN_THUMBL, e.BTN_THUMBR,
            ],
            e.EV_ABS: [
                (e.ABS_X, (0, -AXIS_MAX, AXIS_MAX, 0, 0, 0)),
                (e.ABS_Y, (0, -AXIS_MAX, AXIS_MAX, 0, 0, 0)),
                (e.ABS_RX, (0, -AXIS_MAX, AXIS_MAX, 0, 0, 0)),
                (e.ABS_RY, (0, -AXIS_MAX, AXIS_MAX, 0, 0, 0)),
                (e.ABS_Z, (0, 0, 255, 0, 0, 0)),        # left trigger
                (e.ABS_RZ, (0, 0, 255, 0, 0, 0)),       # right trigger
                (e.ABS_HAT0X, (0, -1, 1, 0, 0, 0)),     # d-pad
                (e.ABS_HAT0Y, (0, -1, 1, 0, 0, 0)),
            ],
        }
        self.ui = UInput(cap, name=name, vendor=0x045e, product=0x028e, version=0x0114)
        time.sleep(0.3)

    def _axis(self, v: float) -> int:
        return int(max(-1.0, min(1.0, float(v))) * AXIS_MAX)

    def set_action(self, row) -> None:
        """row: a length-25 NitroGen action vector for ONE control step (BUTTON_ACTION_TOKENS
        order for buttons[0:21], then j_left[21:23], j_right[23:25])."""
        b = row
        ui = self.ui
        ui.write(e.EV_ABS, e.ABS_X, self._axis(b[JLX]))
        ui.write(e.EV_ABS, e.ABS_Y, self._axis(b[JLY]))
        ui.write(e.EV_ABS, e.ABS_RX, self._axis(b[JRX]))
        ui.write(e.EV_ABS, e.ABS_RY, self._axis(b[JRY]))
        if _I_LTRIG is not None:
            ui.write(e.EV_ABS, e.ABS_Z, 255 if b[_I_LTRIG] > 0.5 else 0)
        if _I_RTRIG is not None:
            ui.write(e.EV_ABS, e.ABS_RZ, 255 if b[_I_RTRIG] > 0.5 else 0)
        for idx, code in _BTN_PAIRS:
            ui.write(e.EV_KEY, code, 1 if b[idx] > 0.5 else 0)
        hx = (1 if b[B_DRIGHT] > 0.5 else 0) - (1 if b[B_DLEFT] > 0.5 else 0)
        hy = (1 if b[B_DDOWN] > 0.5 else 0) - (1 if b[B_DUP] > 0.5 else 0)
        ui.write(e.EV_ABS, e.ABS_HAT0X, hx)
        ui.write(e.EV_ABS, e.ABS_HAT0Y, hy)
        ui.syn()

    def neutral(self) -> None:
        ui = self.ui
        for code in (e.ABS_X, e.ABS_Y, e.ABS_RX, e.ABS_RY, e.ABS_Z, e.ABS_RZ,
                     e.ABS_HAT0X, e.ABS_HAT0Y):
            ui.write(e.EV_ABS, code, 0)
        for code in (e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y, e.BTN_TL, e.BTN_TR,
                     e.BTN_SELECT, e.BTN_START, e.BTN_MODE, e.BTN_THUMBL, e.BTN_THUMBR):
            ui.write(e.EV_KEY, code, 0)
        ui.syn()

    def close(self) -> None:
        try:
            self.ui.close()
        except Exception:
            pass


if __name__ == "__main__":
    import numpy as np
    print("button index map:", {n: _NAME2IDX[n] for n in ("south", "east", "west", "north",
          "start", "back", "guide")})
    pad = VirtualGamepad()
    print("created virtual pad; pushing left stick left for 30s...")
    row = np.zeros(25, np.float32); row[JLX] = -1.0
    for _ in range(60):
        pad.set_action(row); time.sleep(0.5)
    pad.close()
