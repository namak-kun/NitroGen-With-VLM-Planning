"""Flare: Empyrean Campaign top-down action-RPG env.

Flare is an SDL2 Diablo/Zelda-like ARPG.  We force the pure SDL software renderer
and seed a keyboard-only config so it runs cleanly under Xvfb without GPU/mouse
dependencies.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_RTRIG, I_SOUTH, I_WEST = 16, 18, 20
STICK_THRESH = 0.2
BUTTON_FRAC = 0.3


class FlareARPGEnv(ProcGameEnv):
    name = "flare_arpg"
    window_name = "Flare"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 14.0, **kw):
        repo = Path(__file__).resolve().parents[3]
        self.runtime_dir = repo / ".nitrogen-env-build" / "flare_arpg"
        self.home = self.runtime_dir / "home"
        self.speedhack_ctl = self.runtime_dir / "speedhack.ctl"
        self._ensure_runtime_files(width, height)
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def _ensure_runtime_files(self, width: int, height: int) -> None:
        config_dir = self.home / ".config" / "flare"
        saves_dir = self.home / ".local" / "share" / "flare" / "saves"
        config_dir.mkdir(parents=True, exist_ok=True)
        saves_dir.mkdir(parents=True, exist_ok=True)
        if (saves_dir / "empyrean").exists():
            shutil.rmtree(saves_dir / "empyrean")
        try:
            (config_dir / "flare_lock").unlink()
        except FileNotFoundError:
            pass
        (config_dir / "settings.txt").write_text(
            "\n".join([
                "## flare-engine settings file ##",
                "move_type_dimissed=1",
                "fullscreen=0",
                f"resolution_w={width}",
                f"resolution_h={height}",
                "music_volume=0",
                "sound_volume=0",
                "combat_text=1",
                "mouse_move=0",
                "hwsurface=1",
                "vsync=0",
                "texture_filter=1",
                "dpi_scaling=0",
                "parallax_layers=1",
                "max_fps=60",
                "renderer=sdl",
                "enable_joystick=0",
                "joystick_device=-1",
                "joystick_deadzone=8000",
                "language=en",
                "change_gamma=0",
                "gamma=1",
                "mouse_aim=0",
                "no_mouse=1",
                "show_fps=0",
                "colorblind=0",
                "hardware_cursor=0",
                "dev_mode=0",
                "dev_hud=0",
                "loot_tooltips=0",
                "statbar_labels=0",
                "statbar_autohide=1",
                "auto_equip=1",
                "subtitles=0",
                "minimap_mode=0",
                "mouse_move_swap=0",
                "mouse_move_attack=1",
                "entity_markers=1",
                "prev_save_slot=-1",
                "low_hp_warning_type=0",
                "low_hp_threshold=20",
                "item_compare_tips=0",
                "max_render_size=0",
                "touch_controls=0",
                "touch_scale=1",
                "",
            ]),
            encoding="utf-8",
        )
        (config_dir / "keybindings.txt").write_text(
            "\n".join([
                "# Keybindings",
                "# FORMAT: {ACTION}={BIND},{TYPE}; type 0 is keyboard SDL scancode.",
                "file_version=1.12.90",
                "[user]",
                "cancel=41,0",
                "accept=40,0",
                "accept=44,0",
                "up=26,0",
                "up=82,0",
                "down=22,0",
                "down=81,0",
                "left=4,0",
                "left=80,0",
                "right=7,0",
                "right=79,0",
                "bar1=30,0",
                "bar2=31,0",
                "bar3=32,0",
                "bar4=33,0",
                "bar5=34,0",
                "bar6=35,0",
                "bar7=36,0",
                "bar8=37,0",
                "bar9=38,0",
                "bar0=39,0",
                "main1=44,0",
                "main2=27,0",
                "character=6,0",
                "inventory=12,0",
                "powers=19,0",
                "log=15,0",
                "equipment_swap=20,0",
                "equipment_swap_prev=29,0",
                "minimap_mode=16,0",
                "loot_tooltip_mode=56,0",
                "actionbar=5,0",
                "menu_page_next=78,0",
                "menu_page_prev=75,0",
                "menu_activate=44,0",
                "developer_menu=62,0",
                "",
            ]),
            encoding="utf-8",
        )

    def _env(self) -> dict:
        env = super()._env()
        env.update({
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "XDG_DATA_HOME": str(self.home / ".local" / "share"),
            "SDL_AUDIODRIVER": "dummy",
            "SDL_VIDEODRIVER": "x11",
        })
        return env

    def launch_cmd(self):
        return ["flare", "--renderer=sdl", "--no-audio"]

    def boot(self):
        self._xvfb = subprocess.Popen(
            ["Xvfb", f":{self.display}", "-screen", "0", f"{self.width}x{self.height}x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)
        if self.freeze_during_inference:
            from ..speedhack import SpeedHack
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            self._sh = SpeedHack(ctrl_path=str(self.speedhack_ctl))
        self._game = subprocess.Popen(
            self.launch_cmd(),
            env=self._env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(self.boot_wait)
        self._wid = self._find_window()

    def reset_macro(self, scenario):
        macro = [
            ("wait", 0.3),
            ("key", "Return"),  # title: Play Game
            ("wait", 0.8),
            ("key", "Return"),  # empty slot list: New Game
            ("wait", 0.8),
            ("key", "Return"),  # character screen: Create
            ("wait", 0.8),
        ]
        for _ in range(15):
            macro.extend([("key", "Return"), ("wait", 0.35)])
        macro.append(("wait", 1.0))
        return macro

    def action_to_keys(self, action_chunk):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys = set()
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < 0.5 - STICK_THRESH:
            keys.add("a")
        elif mx > 0.5 + STICK_THRESH:
            keys.add("d")
        if my < 0.5 - STICK_THRESH:
            keys.add("w")
        elif my > 0.5 + STICK_THRESH:
            keys.add("s")
        if (a[:, I_SOUTH] > 0.5).mean() >= BUTTON_FRAC or (a[:, I_RTRIG] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("space")
        if (a[:, I_WEST] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("x")
        return keys

    def save_frame(self, path: str) -> None:
        out = Path(path)
        if out.is_absolute() and out.parts[:2] == ("/", "tmp"):
            out = self.runtime_dir / "vet_frame.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(self._grab()).save(out)

    def close(self) -> None:
        super().close()
        if self._sh is not None:
            self._sh.close()
