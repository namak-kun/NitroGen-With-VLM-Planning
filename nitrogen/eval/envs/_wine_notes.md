# Wine harness probe

Validated on Wine `wine-9.0 (Ubuntu 9.0~repack-4build3)`.

Setup used an isolated win64 prefix under the repo scratch area:

```bash
DISPLAY=:101 WINEPREFIX=/path/to/wineprefix WINEARCH=win64 \
WINEDLLOVERRIDES="mscoree,mshtml=" WINEDEBUG=-all \
/usr/lib/wine/wine64 wineboot -i
```

Test program was a locally built free/legal Win32 probe game, compiled as both PE32 and PE32+,
with a `QueryPerformanceCounter` animation, a `SetTimer` frame-counter animation, and an
arrow-key-controlled square. This tests both clock-based and timer/message-loop advancement.

Results on an 800x600 Xvfb display:

- Capture works: frame mean `27.172`, std `26.119`, min/max `16/255`.
- Normal PE32 run under `/usr/bin/wine` advances: 1.5s diff mean `3.917300`, max `235`,
  changed pixels `14949`.
- PE32 under `/usr/bin/wine` with current `SpeedHack.env()` and `set_speed(0.0)` freezes:
  1.5s diff mean `0.000000`, max `0`, changed pixels `0`.
- PE64 under `/usr/lib/wine/wine64` with the same speedhack also freezes: 1.5s diff mean
  `0.000000`, max `0`, changed pixels `0`.
- Keyboard control works through Wine/X11: while frozen, `xdotool key --window $wid Right`
  changed the PE32 frame with diff mean `4.752328`, max `239`, changed pixels `16714`.

Verdict: the Wine path is viable. Wine's Windows timing/timer plumbing is affected by NitroGen's
LD_PRELOAD `clock_gettime` shim, including for a PE32 Windows process in a win64 prefix.

Launch recipe:

```python
from nitrogen.eval.speedhack import SpeedHack

sh = SpeedHack(ctrl_path="/path/to/wine_speedhack.ctl", initial_scale=1.0, pause_scale=0.0)
env = {
    **os.environ,
    "DISPLAY": ":101",
    "WINEPREFIX": "/path/to/wineprefix",
    "WINEARCH": "win64",
    "WINEDLLOVERRIDES": "mscoree,mshtml=",
    "WINEDEBUG": "-all",
    **sh.env(),
}
subprocess.Popen(["wine", "/path/to/game.exe"], env=env)
sh.set_speed(0.0)  # exact freeze during inference
sh.set_speed(1.0)  # run during action playback
```

Window/control recipe:

```bash
wid=$(DISPLAY=:101 xdotool search --name "Game window title" | tail -1)
DISPLAY=:101 xdotool windowfocus --sync "$wid"
DISPLAY=:101 xdotool key --window "$wid" Right
```

Use each game's real keyboard map; typical fan-game defaults are arrows plus `z`/`x`/`Shift`/
`Return`.

Important caveat: exact `pause_scale=0.0` also freezes ffmpeg if the capture subprocess inherits
`LD_PRELOAD` and `SPEEDHACK_CTRL`; in the probe, `ffmpeg x11grab` timed out in that configuration.
For exact freezes, run capture with a clean env containing `DISPLAY` but not the speedhack variables
(or add a capture-env hook/override). Near-zero `pause_scale` avoids that at the cost of slow drift.

Minimal env shape:

```python
class WineGameEnv(ProcGameEnv):
    name = "wine_game"
    target_keys_to_window = True

    def __init__(self, exe_path: str, wineprefix: str, **kw):
        self.exe_path = exe_path
        self.wineprefix = wineprefix
        super().__init__(width=800, height=600, boot_wait=5.0, **kw)

    def _env(self):
        env = super()._env()
        env.update({
            "WINEPREFIX": self.wineprefix,
            "WINEARCH": "win64",
            "WINEDLLOVERRIDES": "mscoree,mshtml=",
            "WINEDEBUG": "-all",
        })
        return env

    def boot(self):
        super().boot()
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def launch_cmd(self):
        return ["wine", self.exe_path]

    def _find_window(self):
        # Search by the actual game title; fall back to Wine class/title if needed.
        return super()._find_window()

    # If using exact pause_scale=0.0, override _grab or add a capture-env hook so ffmpeg does not
    # inherit LD_PRELOAD/SPEEDHACK_CTRL.
```
