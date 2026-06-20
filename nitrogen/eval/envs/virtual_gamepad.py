"""Virtual Xbox-360 gamepad via Linux uinput, for feeding NitroGen's GAMEPAD action faithfully
to SDL2 games (doukutsu-rs etc.) — the correct embodiment match (NitroGen outputs analog sticks
+ buttons; we drive analog axes + buttons, NOT discretized keyboard keys).

We mimic a Microsoft Xbox 360 controller (VID 0x045e / PID 0x028e) with the standard axis/button
layout, so SDL2's GameController subsystem recognizes it via its built-in mapping (is_game_controller
== true) with no custom gamecontrollerdb entry needed.

NitroGen action layout (per step, 25 dims): buttons[0:21] (BUTTON_ORDER), j_left[21:23] (x,y in
[-1,1], -x=left/-y=up), j_right[23:25]. We map left/right stick -> ABS_X/Y, ABS_RX/RY, and the
relevant buttons -> BTN_*. Trigger buttons map to ABS_Z/RZ.
"""
from __future__ import annotations

import time

from evdev import UInput, ecodes as e

# NitroGen BUTTON_ORDER indices (see nitrogen/training/actions.py / mm_tokenizers). xbox-ish:
B_SOUTH, B_EAST, B_WEST, B_NORTH = 0, 1, 2, 3        # A, B, X, Y
B_LSHOULDER, B_RSHOULDER = 4, 5
B_LTRIGGER, B_RTRIGGER = 6, 7
B_SELECT, B_START = 8, 9
B_LTHUMB, B_RTHUMB = 10, 11
B_DUP, B_DDOWN, B_DLEFT, B_DRIGHT = 12, 13, 14, 15

AXIS_MAX = 32767
JLX, JLY, JRX, JRY = 21, 22, 23, 24


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
        # Xbox 360 wired VID/PID so SDL recognizes the built-in mapping.
        self.ui = UInput(cap, name=name, vendor=0x045e, product=0x028e, version=0x0114)
        time.sleep(0.3)

    def _axis(self, v: float) -> int:
        return int(max(-1.0, min(1.0, float(v))) * AXIS_MAX)

    def set_action(self, row) -> None:
        """row: a length-25 NitroGen action vector for ONE control step."""
        b = row
        ui = self.ui
        # sticks (note: NitroGen y is -up/+down; ABS_Y is -up/+down on Xbox too -> direct)
        ui.write(e.EV_ABS, e.ABS_X, self._axis(b[JLX]))
        ui.write(e.EV_ABS, e.ABS_Y, self._axis(b[JLY]))
        ui.write(e.EV_ABS, e.ABS_RX, self._axis(b[JRX]))
        ui.write(e.EV_ABS, e.ABS_RY, self._axis(b[JRY]))
        # triggers (NitroGen has them as buttons; emit analog full/zero)
        ui.write(e.EV_ABS, e.ABS_Z, 255 if b[B_LTRIGGER] > 0.5 else 0)
        ui.write(e.EV_ABS, e.ABS_RZ, 255 if b[B_RTRIGGER] > 0.5 else 0)
        # face + shoulder + stick-click + start/select buttons
        for idx, code in [
            (B_SOUTH, e.BTN_A), (B_EAST, e.BTN_B), (B_WEST, e.BTN_X), (B_NORTH, e.BTN_Y),
            (B_LSHOULDER, e.BTN_TL), (B_RSHOULDER, e.BTN_TR),
            (B_SELECT, e.BTN_SELECT), (B_START, e.BTN_START),
            (B_LTHUMB, e.BTN_THUMBL), (B_RTHUMB, e.BTN_THUMBR),
        ]:
            ui.write(e.EV_KEY, code, 1 if b[idx] > 0.5 else 0)
        # d-pad -> hat
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
                     e.BTN_SELECT, e.BTN_START, e.BTN_THUMBL, e.BTN_THUMBR):
            ui.write(e.EV_KEY, code, 0)
        ui.syn()

    def close(self) -> None:
        try:
            self.ui.close()
        except Exception:
            pass


if __name__ == "__main__":
    import numpy as np
    pad = VirtualGamepad()
    print("created virtual pad:", pad.ui.device.path if hasattr(pad.ui, "device") else "ok")
    print("Hold it open 30s so SDL/doukutsu-rs can enumerate it; pushing left stick left...")
    row = np.zeros(25, np.float32); row[JLX] = -1.0
    for _ in range(60):
        pad.set_action(row); time.sleep(0.5)
    pad.close()
