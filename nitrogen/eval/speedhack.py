"""SpeedHack — Python controller for libspeedhack.so (the LD_PRELOAD time-scaling shim). The
Linux equivalent of NitroGen's `xspeedhack` client: create a tiny mmap'd control file, set the
scale live (0.0 = freeze, 1.0 = run), and hand the right LD_PRELOAD/SPEEDHACK_CTRL env to the
launched game. Mirrors GamepadEnv.pause()/unpause().

Usage:
    sh = SpeedHack()                      # builds the .so if needed, creates the control file
    proc = subprocess.Popen(cmd, env={**os.environ, **sh.env()})
    sh.set_speed(0.0)                     # freeze (during inference)
    sh.set_speed(1.0)                     # run (during action playback)
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "speedhack", "libspeedhack.c")
_SO = os.path.join(_HERE, "speedhack", "libspeedhack.so")


def build_so(force: bool = False) -> str:
    """Compile libspeedhack.so if missing/stale. Returns its path."""
    if force or not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_SRC):
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", _SRC, "-o", _SO, "-ldl", "-lpthread"],
            check=True)
    return _SO


class SpeedHack:
    def __init__(self, ctrl_path: str | None = None, initial_scale: float = 1.0,
                 pause_scale: float = 0.02):
        # pause_scale: the scale used by pause(). NOT 0.0 by default: a *full* freeze (0.0)
        # trips an edge case in some engines' frame-skip/catch-up logic (doukutsu-rs does NOT
        # resume after a 0.0 freeze), so we use a tiny near-freeze. At 0.02x, 0.4s of inference
        # advances the game ~8ms -> effectively frozen (the player barely drifts) while the
        # timing stays in its normal regime so resume works. Set 0.0 for engines that tolerate it.
        self.pause_scale = pause_scale
        self.so = build_so()
        self.ctrl_path = ctrl_path or tempfile.mktemp(prefix="speedhack_", suffix=".ctl")
        with open(self.ctrl_path, "wb") as f:
            f.write(struct.pack("d", float(initial_scale)))
        self._f = open(self.ctrl_path, "r+b", buffering=0)

    def env(self) -> dict:
        """LD_PRELOAD + SPEEDHACK_CTRL to pass to the launched game."""
        return {"LD_PRELOAD": self.so, "SPEEDHACK_CTRL": self.ctrl_path}

    def set_speed(self, scale: float) -> None:
        self._f.seek(0)
        self._f.write(struct.pack("d", float(scale)))
        self._f.flush()

    def pause(self) -> None:
        self.set_speed(self.pause_scale)

    def unpause(self) -> None:
        self.set_speed(1.0)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
        try:
            os.remove(self.ctrl_path)
        except OSError:
            pass


if __name__ == "__main__":
    # Self-test: launch a Python child under the shim and show its monotonic clock FREEZE.
    import sys, time
    sh = SpeedHack()
    child = (
        "import time,sys\n"
        "for i in range(20):\n"
        "  sys.stdout.write(f'{time.monotonic():.3f}\\n'); sys.stdout.flush(); time.sleep(0.1)\n"
    )
    env = {**os.environ, **sh.env()}
    p = subprocess.Popen([sys.executable, "-u", "-c", child], env=env,
                         stdout=subprocess.PIPE, text=True)
    time.sleep(0.6); print("[controller] set_speed(0.0)  # FREEZE"); sh.set_speed(0.0)
    time.sleep(0.8); print("[controller] set_speed(1.0)  # RUN"); sh.set_speed(1.0)
    time.sleep(0.6); print("[controller] set_speed(3.0)  # 3x"); sh.set_speed(3.0)
    out, _ = p.communicate(timeout=10)
    print("child monotonic readings (should plateau while frozen, jump fast at 3x):")
    print(out)
    sh.close()
