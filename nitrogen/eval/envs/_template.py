"""TEMPLATE for adding a new game env. COPY this file to <yourgame>.py, rename the class, fill in
the 3 required hooks, then VET it:

    PYTHONPATH=. .venv/bin/python planner_poc/vet_env.py nitrogen.eval.envs.<yourgame>:<YourGame>Env

The base class (ProcGameEnv) already owns: Xvfb headless display, LD_PRELOAD speedhack
(freeze-during-inference), ffmpeg x11grab capture, xdotool keyboard control, and the
reset/step/close loop. You only declare the GAME-SPECIFIC bits below. See supertuxkart.py
(racing, keyboard) and cavestory.py (platformer, gamepad+state) for two worked examples.

Action layout reminder (load-bearing): a NitroGen action chunk is (H, 25):
  dims 0:21  = buttons (use nitrogen.shared.BUTTON_ACTION_TOKENS for names/indices)
  dims 21:23 = LEFT stick  (JLX=21 = horizontal/steer, JLY=22 = vertical), output in [0,1], 0.5=neutral
  dims 23:25 = RIGHT stick
Threshold the stick: e.g. JLX < 0.5-thresh -> "Left", JLX > 0.5+thresh -> "Right". (When the policy
emits raw [-1,1] you'll instead compare against 0; vet_env prints what your env actually receives.)
"""
from __future__ import annotations

import numpy as np

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons
from ..core import JLX, JLY


class TemplateGameEnv(ProcGameEnv):
    name = "template_game"
    window_name = "ExactWindowTitleSubstring"   # xdotool searches by this; check with `xdotool search --name`
    control = "keyboard"                          # "keyboard" (xdotool) or "gamepad" (VirtualGamepad)

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 15.0, **kw):
        # add game-specific config (level file, difficulty, etc.) here, then:
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    # REQUIRED: how to launch the game (argv list). Runs under the env's DISPLAY + speedhack preload.
    def launch_cmd(self):
        return ["/path/to/game-binary", "--some-flag", f"--width={self.width}", f"--height={self.height}"]

    # REQUIRED (keyboard control): map a NitroGen (H,25) chunk -> set of keysyms to HOLD this step.
    # Easiest path = the helper, giving it a {button_index: keysym} map for this game's controls:
    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            button_map={18: "z",   # SOUTH -> jump key
                        20: "x"},  # WEST  -> attack key
            steer_thresh=0.25, vert_thresh=0.25)
        # ...or hand-roll it like supertuxkart.py if the game wants custom keys (accel=Up, etc.).

    # OPTIONAL: privileged state for a StatePredicateDetector (skip -> frame-based VLMJudgeDetector).
    # def read_state(self) -> dict:
    #     return {}

    # OPTIONAL: keypress macro to reach a reproducible start (skip past menus). Entries are
    # ("key", k) | ("hold", k, seconds) | ("wait", seconds).
    # def reset_macro(self, scenario):
    #     return [("wait", 2.0), ("key", "Return"), ("wait", 1.0)]
