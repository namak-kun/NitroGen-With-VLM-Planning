"""OpenMW env using the free OpenMW Example Suite/template content.

The commercial Morrowind data files are intentionally not used. Download and extract
OpenMW/example-suite into .nitrogen-env-build/openmw before launching this env.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv
from ..core import JLX, JLY

I_LTRIG, I_RTRIG, I_SOUTH, I_WEST = 9, 16, 18, 20
STICK_THRESH = 0.2
BUTTON_FRAC = 0.3


class OpenMWEnv(ProcGameEnv):
    name = "openmw"
    window_name = "OpenMW"
    control = "keyboard"
    window_manager = "matchbox-window-manager"
    target_keys_to_window = False

    def __init__(self, data_root: str | None = None, width: int = 800, height: int = 600,
                 boot_wait: float = 18.0, **kw):
        repo = Path(__file__).resolve().parents[3]
        self.data_root = Path(data_root).expanduser().resolve() if data_root else (
            repo / ".nitrogen-env-build" / "openmw")
        self.template_data = self.data_root / "openmw-template" / "data"
        self.example_data = self.data_root / "openmw-ExampleSuite" / "data"
        self.config_dir = self.data_root / "config"
        self.user_data_dir = self.data_root / "user-data"
        self.xdg_config_home = self.data_root / "xdg-config"
        self.xdg_data_home = self.data_root / "xdg-data"
        self.script_data = self.data_root / "nitrogen-scripts"
        self.startup_script = self.data_root / "nitrogen-start.txt"
        self.speedhack_ctl = self.data_root / "speedhack.ctl"
        self._ensure_config(width, height)
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def _ensure_config(self, width: int, height: int) -> None:
        required = [
            self.template_data / "openmw-template.omwgame",
            self.example_data / "ExampleSuite.omwaddon",
            self.example_data / "Land.esp",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(
                "OpenMW free Example Suite data is missing. Expected the extracted "
                "OpenMW/example-suite release under .nitrogen-env-build/openmw; "
                f"missing: {missing}")
        script_dir = self.script_data / "scripts" / "nitrogen_openmw"
        for path in (self.config_dir, self.user_data_dir, self.xdg_config_home,
                     self.xdg_data_home, script_dir):
            path.mkdir(parents=True, exist_ok=True)
        (self.script_data / "nitrogen-openmw.omwscripts").write_text(
            "PLAYER: scripts/nitrogen_openmw/driver.lua\n")
        (script_dir / "driver.lua").write_text("""
local camera = require('openmw.camera')
local input = require('openmw.input')
local self = require('openmw.self')

local function down(name)
  local key = input.KEY[name]
  return key ~= nil and input.isKeyPressed(key)
end

return { engineHandlers = {
  onActive = function()
    self:enableAI(false)
  end,
  onFrame = function(dt)
    local w, a, s, d = down('W'), down('A'), down('S'), down('D')
    camera.setMode(camera.MODE.FirstPerson)
    camera.setPitch((w or a or s or d or down('E')) and 1.0 or -1.4)
    self.controls.movement = (w and 1 or 0) + (s and -1 or 0)
    self.controls.sideMovement = (d and 1 or 0) + (a and -1 or 0)
    self.controls.run = w
    self.controls.jump = down('E')
  end
} }
""".lstrip())
        self.startup_script.write_text("set GameHour to 12\nToggleGodMode\n")
        (self.config_dir / "openmw.cfg").write_text(
            "\n".join([
                "# NitroGen OpenMW free example-suite configuration.",
                f'data="{self.template_data}"',
                f'data="{self.example_data}"',
                f'data="{self.script_data}"',
                "content=openmw-template.omwgame",
                "content=ExampleSuite.omwaddon",
                "content=Land.esp",
                "content=nitrogen-openmw.omwscripts",
                "",
            ]))
        (self.config_dir / "settings.cfg").write_text(
            "\n".join([
                "[Video]",
                f"resolution x = {width}",
                f"resolution y = {height}",
                "fullscreen = false",
                "vsync = false",
                "",
                "[Input]",
                "grab cursor = false",
                "",
                "[Sound]",
                "enabled = false",
                "",
            ]))

    def _env(self) -> dict:
        env = super()._env()
        env.update({
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "SDL_VIDEODRIVER": "x11",
            "XDG_CONFIG_HOME": str(self.xdg_config_home),
            "XDG_DATA_HOME": str(self.xdg_data_home),
        })
        return env

    def launch_cmd(self):
        return [
            "openmw",
            "--config", str(self.config_dir),
            "--user-data", str(self.user_data_dir),
            "--skip-menu",
            "--no-grab",
            "--no-sound",
            "--script-run", str(self.startup_script),
        ]

    def boot(self):
        self._xvfb = subprocess.Popen(
            ["Xvfb", f":{self.display}", "-screen", "0", f"{self.width}x{self.height}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        if self.window_manager:
            self._wm = subprocess.Popen(self.window_manager.split(), env=self._env(),
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
        if self.freeze_during_inference:
            from ..speedhack import SpeedHack
            self._sh = SpeedHack(ctrl_path=str(self.speedhack_ctl))
        self._game = subprocess.Popen(self.launch_cmd(), env=self._env(),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(self.boot_wait)
        self._wid = self._find_window()
        if self._wid:
            self._focus_window()

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
        if (a[:, I_RTRIG] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("w")
        if (a[:, I_LTRIG] > 0.5).mean() >= BUTTON_FRAC:
            keys.discard("w")
            keys.add("s")
        if (a[:, I_SOUTH] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("e")
        if (a[:, I_WEST] > 0.5).mean() >= BUTTON_FRAC:
            keys.add("space")
        return keys

    def reset_macro(self, scenario):
        return []

    def save_frame(self, path: str) -> None:
        out = Path(path)
        if out.is_absolute() and out.parts[:2] == ("/", "tmp"):
            out = self.data_root / "vet_frame.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(self._grab()).save(out)

    def close(self) -> None:
        super().close()
        if self._sh is not None:
            self._sh.close()
