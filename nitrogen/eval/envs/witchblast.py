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

import os
import shlex
import subprocess

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

# right-stick (aim/shoot) dims and a couple of buttons
RSX, RSY = 23, 24
I_RTRIG, I_SOUTH, I_WEST, I_NORTH = 16, 18, 20, 19
MOVE_THRESH = 0.2
SHOOT_THRESH = 0.25
GAME_DIR = "/tmp/witchblast"
PATCHED_BIN = f"{GAME_DIR}/build/Witch_Blast_autostart"


class WitchBlastEnv(ProcGameEnv):
    name = "witchblast"
    window_name = "Witch Blast"
    control = "keyboard"
    # SFML's sf::Keyboard polling reads global XTEST key state, not xdotool's window-targeted events.
    target_keys_to_window = False
    window_manager = None

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 12.0, **kw):
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        _ensure_headless_binary()
        return [
            "sh", "-c",
            f"cd {shlex.quote(GAME_DIR)} && "
            "ALSOFT_DRIVERS=null WITCHBLAST_AUTOSTART=1 WITCHBLAST_FORCE_FOCUS=1 "
            f"exec {shlex.quote(PATCHED_BIN)}",
        ]

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
        if (a[:, I_SOUTH] > 0.5).mean() >= 0.3:
            keys.add("Control_R")
        if (a[:, I_RTRIG] > 0.5).mean() >= 0.3:
            keys.discard("a")
            keys.discard("d")
            keys.add("w")
        if (a[:, I_NORTH] > 0.5).mean() >= 0.3:
            keys.add("space")   # cast spell
        return keys

    def reset_macro(self, scenario):
        return [("wait", 0.5), ("key", "Return"), ("wait", 0.5)]


def _ensure_headless_binary() -> None:
    """Build the NitroGen headless WitchBlast wrapper binary once.

    The stock SFML game hard-gates updates on app->hasFocus(), which is unreliable under Xvfb even
    with WMs. This relinks the already-built game with a tiny source patch that (only when launched
    with WITCHBLAST_FORCE_FOCUS/WITCHBLAST_AUTOSTART) treats the window as focused and starts a run.
    """
    src = f"{GAME_DIR}/src/WitchBlastGame.cpp"
    build = f"{GAME_DIR}/build"
    patched_src = f"{build}/WitchBlastGame_nitrogen.cpp"
    patched_obj = f"{build}/WitchBlastGame_nitrogen.o"
    link_txt = f"{build}/CMakeFiles/Witch_Blast.dir/link.txt"

    if os.path.exists(PATCHED_BIN) and os.path.getmtime(PATCHED_BIN) >= os.path.getmtime(src):
        return

    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace(
        "    if (app->hasFocus())",
        '    if (app->hasFocus() || std::getenv("WITCHBLAST_FORCE_FOCUS"))',
    )
    text = text.replace(
        "  lastTime = getAbsolutTime();\n\n  prepareIntro();",
        '  lastTime = getAbsolutTime();\n\n'
        '  if (std::getenv("WITCHBLAST_AUTOSTART")) {\n'
        '    parameters.playerName = "ai";\n'
        '    saveConfigurationToFile();\n'
        '    startNewGame(false, 1);\n'
        '  }\n'
        '  else\n'
        '    prepareIntro();',
    )
    with open(patched_src, "w", encoding="utf-8") as f:
        f.write(text)

    subprocess.run(
        ["g++", "-std=c++11", "-O3", "-DNDEBUG", "-I../src", "-c", patched_src, "-o", patched_obj],
        cwd=build, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with open(link_txt, "r", encoding="utf-8") as f:
        link_cmd = shlex.split(f.read())
    link_cmd = [
        patched_obj if arg.endswith("src/WitchBlastGame.cpp.o") else
        PATCHED_BIN if previous == "-o" else arg
        for previous, arg in zip([""] + link_cmd[:-1], link_cmd)
    ]
    subprocess.run(link_cmd, cwd=build, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
