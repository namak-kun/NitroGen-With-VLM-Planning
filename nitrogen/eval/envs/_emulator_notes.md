# Emulator harness probe: Mednafen NES

Validated path: `mednafen` 1.29.0 from apt (`/usr/games/mednafen`). `ldd` resolves `libc.so.6` and `libSDL2-2.0.so.0`, so NitroGen's LD_PRELOAD speedhack attaches.

Probe setup used a locally generated legal NES homebrew ROM that cycles the backdrop and latches any controller input. Run under Xvfb with:

```bash
SDL_VIDEODRIVER=x11 SDL_AUDIODRIVER=dummy MEDNAFEN_HOME=/path/to/isolated/home \
LD_PRELOAD=/home/t-nagupta/NitroGen/nitrogen/eval/speedhack/libspeedhack.so \
SPEEDHACK_CTRL=/path/to/speedhack.ctl \
mednafen -force_module nes -sound 0 -video.driver softfb -video.fs 0 \
  -nes.xscale 2 -nes.yscale 2 -nes.stretch 0 -nothrottle 0 game.nes
```

Results on 800x600 x11grab:

- Capture works: frame mean `71.964`, std `92.351`, min/max `0/252`.
- Normal run advances: 0.7s frame diff mean `63.698489`, max `252`, changed pixels `229376`.
- Speedhack exact freeze works: `set_speed(0.0)`, 1.5s frame diff mean `0.000000`, max `0`, changed pixels `0`.
- Resume works: 0.4s post-unfreeze diff mean `71.317333`, max `220`, changed pixels `229376`.
- Keyboard control works: Mednafen default NES `d` = Right; transition diff mean `57.280000`, max `252`, changed pixels `229120`, then post-key stability diff `0`.

Important wiring notes:

- Mednafen tolerates exact `pause_scale=0.0`. ProcGameEnv's default `SpeedHack.pause_scale=0.02` is only slow motion for this emulator; over 1.5s it still advanced (diff mean `54.734222`). For strict inference freeze, set `self._sh.pause_scale = 0.0` after `super().boot()`, or instantiate `SpeedHack(pause_scale=0.0)`.
- `xdotool search --name` did not find the SDL window, but `xdotool search --class mednafen` did. A ProcGameEnv subclass can override `_find_window()` to search class `mednafen`.
- Default NES keyboard map: `w/a/s/d` = D-pad, `z` = B, `[` = A, `Enter` = Start, `Tab` = Select.
- Native fallback exists but was not needed: Mednafen hotkeys `Pause` = pause, `Alt+A` = frame advance, `Alt+R` = run normal.

Minimal subclass shape:

```python
import os
import subprocess

class MednafenNesEnv(ProcGameEnv):
    name = "mednafen_nes"
    target_keys_to_window = True

    def __init__(self, rom_path: str, mednafen_home: str, **kw):
        self.rom_path = rom_path
        self.mednafen_home = mednafen_home
        os.makedirs(self.mednafen_home, exist_ok=True)
        super().__init__(width=800, height=600, boot_wait=3.0, **kw)

    def _env(self):
        env = super()._env()
        env.update({
            "SDL_VIDEODRIVER": "x11",
            "SDL_AUDIODRIVER": "dummy",
            "MEDNAFEN_HOME": self.mednafen_home,
            "MEDNAFEN_ALLOWMULTI": "1",
        })
        return env

    def launch_cmd(self):
        return [
            "mednafen", "-force_module", "nes", "-sound", "0",
            "-video.driver", "softfb", "-video.fs", "0",
            "-nes.xscale", "2", "-nes.yscale", "2", "-nes.stretch", "0",
            "-nothrottle", "0", self.rom_path,
        ]

    def boot(self):
        super().boot()
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def _find_window(self):
        out = subprocess.run(
            ["xdotool", "search", "--class", "mednafen"],
            env=self._env(), capture_output=True, text=True,
        )
        ids = [line for line in out.stdout.split() if line.strip()]
        return ids[-1] if ids else ""
```
