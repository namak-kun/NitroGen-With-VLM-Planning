"""WitchBlast env — a Binding-of-Isaac-like twin-stick dungeon shooter (the model was TRAINED on
Isaac, so this is a strong in-distribution candidate). Thin ProcGameEnv subclass.

Controls (readme.txt): WASD = move in 8 directions; Arrow keys = SHOOT in 4 directions; Right Ctrl =
shoot in facing direction; Tab = change shoot type; Space = cast spell. This is a true twin-stick:
  LEFT stick  (JLX=21, JLY=22) -> WASD movement
  RIGHT stick (dims 23/24)     -> arrow-key shooting direction
Sticks are [0,1] with 0.5 = neutral (so 'left' = value < 0.5-thresh, etc).

Built: /tmp/witchblast/build/Witch_Blast (C++/SFML 2.6, dynamically linked -> speedhack works). Runs
from /tmp/witchblast (needs ./data). Window title: "Witch Blast".
"""
from __future__ import annotations

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

# right-stick (aim/shoot) dims and a couple of buttons
RSX, RSY = 23, 24
I_RTRIG, I_SOUTH, I_WEST, I_NORTH = 16, 18, 20, 19
MOVE_THRESH = 0.2
SHOOT_THRESH = 0.25
GAME_DIR = "/tmp/witchblast"


class WitchBlastEnv(ProcGameEnv):
    name = "witchblast"
    window_name = "Witch Blast"
    control = "keyboard"
    # SFML gates ALL input on app->hasFocus() (WitchBlastGame.cpp:3998). Under a bare Xvfb (no WM)
    # the window never gains focus so the game stays frozen + ignores input. A WM is REQUIRED. NOTE
    # (2026-06-23): matchbox alone did NOT make hasFocus() true in testing — the headless SFML-focus
    # fix is still OPEN (try other WMs / synthetic FocusIn / Xephyr). See plan.md "WitchBlast".
    window_manager = "matchbox-window-manager"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 12.0, **kw):
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        return ["sh", "-c", f"cd {GAME_DIR} && exec ./build/Witch_Blast"]

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys = set()
        # LEFT stick -> WASD movement (8-directional)
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < 0.5 - MOVE_THRESH:
            keys.add("a")
        elif mx > 0.5 + MOVE_THRESH:
            keys.add("d")
        if my < 0.5 - MOVE_THRESH:
            keys.add("w")
        elif my > 0.5 + MOVE_THRESH:
            keys.add("s")
        # RIGHT stick -> arrow-key shooting direction
        rx, ry = float(a[:, RSX].mean()), float(a[:, RSY].mean())
        if rx < 0.5 - SHOOT_THRESH:
            keys.add("Left")
        elif rx > 0.5 + SHOOT_THRESH:
            keys.add("Right")
        if ry < 0.5 - SHOOT_THRESH:
            keys.add("Up")
        elif ry > 0.5 + SHOOT_THRESH:
            keys.add("Down")
        # fire button -> shoot in facing direction (held). xdotool keysym for Right Ctrl = Control_R.
        if (a[:, I_RTRIG] > 0.5).mean() >= 0.3 or (a[:, I_SOUTH] > 0.5).mean() >= 0.3:
            keys.add("Control_R")
        if (a[:, I_NORTH] > 0.5).mean() >= 0.3:
            keys.add("space")   # cast spell
        return keys

    def reset_macro(self, scenario):
        # WitchBlast opens on a title/menu; a few Enter/Space presses start a run.
        return [("wait", 1.5), ("key", "Return"), ("wait", 0.5), ("key", "Return"),
                ("wait", 0.5), ("key", "space")]
