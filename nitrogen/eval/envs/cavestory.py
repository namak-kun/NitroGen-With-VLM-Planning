"""CaveStoryEnv — closed-loop GameEnv backed by doukutsu-rs (open-source Rust reimplementation
of the freeware Metroidvania Cave Story). FEASIBILITY-PROVEN, HEADLESS, purely GAMEPAD-driven —
  * render/observe: the game runs under Xvfb with software OpenGL (Mesa llvmpipe); we grab the
    framebuffer with ffmpeg x11grab -> the RGB frame the policy sees.
  * act: NitroGen outputs a GAMEPAD action (analog sticks + buttons), so we drive a VIRTUAL
    XBOX-360 GAMEPAD (uinput) — the faithful embodiment match — which doukutsu-rs's SDL2
    GameController subsystem auto-detects ("Connected gamepad: Xbox 360 Controller"). This
    preserves the analog stick signal and per-step timing (vs. discretizing to keyboard keys).
    SDL reads the pad directly from the device (evdev), so NO window focus / xdotool is needed —
    even menu navigation goes through the gamepad.
Validated end-to-end: gamepad detected, drove the title menu (A=confirm), reached gameplay.

REQUIRES the doukutsu-rs build patched so Player 1 defaults to Gamepad(0) (on desktop Linux it
defaults to Keyboard, leaving the pad unused). One-line change in src/game/settings.rs
`default_p1_controller_type` -> ControllerType::Gamepad(0); see scripts/setup_cavestory.md.

This v1 is REAL-TIME and asynchronous (the game keeps running while the GPU policy thinks),
which actually matches the intended System-2(async)/System-1(real-time) deployment. A future
v2 engine patch (a socket PlayerController + state export + frame-stepping) would add exact
determinism + privileged state (player x/y/map/life/flags) for StatePredicateDetector; until
then use VLMJudgeDetector on frames. The engine-patch protocol is documented at the bottom.

Setup (once): build doukutsu-rs (cargo), drop the freeware Cave Story `data/` + `Doukutsu.exe`
next to the binary, install Xvfb + ffmpeg + xdotool. See scripts/setup_cavestory.md.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

import numpy as np

from ..core import JLX, JLY, N_BUTTONS, GameEnv, Observation, Scenario
from .virtual_gamepad import VirtualGamepad, B_SOUTH, B_EAST, B_DUP, B_DDOWN, B_DLEFT, B_DRIGHT

# Named gamepad buttons for reset macros (menu navigation). doukutsu-rs default gamepad map:
# menu_ok=South(A), menu_back=East(B), move=d-pad/left-stick.
MENU_OK, MENU_BACK = B_SOUTH, B_EAST


def _free_display() -> int:
    for d in range(101, 200):
        if not os.path.exists(f"/tmp/.X{d}-lock"):
            return d
    raise RuntimeError("no free X display")


class CaveStoryEnv(GameEnv):
    name = "cave_story"
    action_hz = 30.0

    def __init__(self, drs_dir: str = "/tmp/doukutsu-rs/target/release",
                 width: int = 640, height: int = 480, display: int | None = None,
                 reset_macro: list | None = None, boot_wait: float = 10.0,
                 chunk_seconds: float = 0.6, launch: bool = True, use_gamepad: bool = True,
                 freeze_during_inference: bool = True):
        self.drs_dir = drs_dir
        self.width, self.height = width, height
        self.display = display if display is not None else _free_display()
        self.boot_wait = boot_wait
        self.chunk_seconds = chunk_seconds
        self.use_gamepad = use_gamepad
        # freeze_during_inference: replicate NitroGen's GamepadEnv — keep the game (near-)frozen
        # except while applying an action, so the ~0.4s/chunk GPU inference doesn't let the game
        # run uncontrolled (the real-time gap). Uses the LD_PRELOAD speedhack (set_speed).
        self.freeze_during_inference = freeze_during_inference
        self._sh = None
        # reset_macro: list of ("key", keysym) | ("wait", seconds) to drive menus to a known
        # start (e.g. New Save -> Normal -> Single Player -> gameplay). Game-flow-specific, so
        # the caller/scenario supplies it. Default = boot only.
        self.reset_macro = reset_macro or []
        self._xvfb = None
        self._game = None
        self._wid = None
        self._pad = None
        self._step = 0
        req = ("Xvfb", "ffmpeg") if use_gamepad else ("Xvfb", "ffmpeg", "xdotool")
        for tool in req:
            if shutil.which(tool) is None:
                raise RuntimeError(f"{tool} not found; see scripts/setup_cavestory.md")
        if launch:
            self._boot()

    def _press_btn(self, idx: int, hold: float = 0.12):
        """Press a gamepad button (by NitroGen button index) for menu navigation."""
        row = np.zeros(25, np.float32); row[idx] = 1.0
        self._pad.set_action(row); time.sleep(hold); self._pad.neutral(); time.sleep(0.25)

    def _hold_dir(self, dx: int, dy: int, hold: float = 0.15):
        """Hold the left stick in a direction (menu cursor move / nudge)."""
        row = np.zeros(25, np.float32); row[JLX] = dx; row[JLY] = dy
        self._pad.set_action(row); time.sleep(hold); self._pad.neutral(); time.sleep(0.2)

    # ---- process / display management --------------------------------------------------
    def _env(self):
        return dict(os.environ, DISPLAY=f":{self.display}")

    def _fix_input_perms(self):
        # New uinput nodes are root:root 0600; SDL (our user) needs read access. Best-effort.
        subprocess.run("sudo chmod 666 /dev/input/event* /dev/input/js* 2>/dev/null",
                       shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _boot(self):
        # Create the virtual gamepad BEFORE launching the game so SDL enumerates it at startup.
        if self.use_gamepad:
            self._pad = VirtualGamepad()
            time.sleep(0.3)
            self._fix_input_perms()
        # Speedhack (LD_PRELOAD) so we can freeze the game during inference. Built lazily.
        game_env = self._env()
        if self.freeze_during_inference:
            from ..speedhack import SpeedHack
            self._sh = SpeedHack()           # pause_scale ~0.02 near-freeze (0.0 hangs doukutsu-rs)
            game_env = {**game_env, **self._sh.env()}
        self._xvfb = subprocess.Popen(
            ["Xvfb", f":{self.display}", "-screen", "0", f"{self.width}x{self.height}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        binp = os.path.join(self.drs_dir, "doukutsu-rs")
        self._game = subprocess.Popen(
            [binp, "--window-width", str(self.width), "--window-height", str(self.height)],
            cwd=self.drs_dir, env=game_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(self.boot_wait)
        # window id only needed for keyboard injection; gamepad reads the device directly.
        self._wid = None if self.use_gamepad else self._find_window()

    def _find_window(self) -> str:
        out = subprocess.run(["xdotool", "search", "--name", "Cave Story"],
                             env=self._env(), capture_output=True, text=True)
        ids = [x for x in out.stdout.split() if x.strip()]
        if not ids:
            raise RuntimeError("doukutsu-rs window not found on the display")
        return ids[-1]

    def _xdo(self, *args):
        subprocess.run(["xdotool", *args], env=self._env(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _press(self, key: str, hold: float = 0.12):
        self._xdo("keydown", "--window", self._wid, key)
        time.sleep(hold)
        self._xdo("keyup", "--window", self._wid, key)

    # ---- observation: grab the framebuffer --------------------------------------------
    def _grab(self) -> np.ndarray:
        p = subprocess.run(
            ["ffmpeg", "-loglevel", "quiet", "-f", "x11grab",
             "-video_size", f"{self.width}x{self.height}", "-i", f":{self.display}.0",
             "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            capture_output=True)
        buf = p.stdout
        n = self.width * self.height * 3
        if len(buf) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(buf[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    # ---- action: drive the virtual gamepad with per-step timing ------------------------
    def _apply_chunk(self, action_chunk: np.ndarray) -> None:
        """Replay the (H,25) chunk on the virtual gamepad, one row at a time, preserving the
        analog stick signal AND intra-chunk timing (each row held chunk_seconds/H). This is the
        faithful embodiment: NitroGen's analog output -> analog axes (vs. discretized keys).

        Mirrors NitroGen GamepadEnv.perform_action: if freeze_during_inference, the game is
        (near-)frozen between steps and UNPAUSED only while the action is applied, so inference
        latency doesn't let the game run uncontrolled. We busy-wait (not time.sleep) per row so
        the real elapsed time is precise."""
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        h = a.shape[0]
        per = self.chunk_seconds / max(h, 1)
        if self._sh is not None:
            self._sh.unpause()
        for row in a:
            self._pad.set_action(row)
            t = time.perf_counter()
            while time.perf_counter() - t < per:
                pass
        if self._sh is not None:
            self._sh.pause()                 # near-freeze again until the next step

    # ---- GameEnv interface -------------------------------------------------------------
    def reset(self, scenario: Scenario) -> Observation:
        # Menu-setup macro navigates to a known start via the GAMEPAD. Entry forms:
        #   ("btn", idx)        press a gamepad button (e.g. MENU_OK to confirm)
        #   ("dir", dx, dy)     hold the left stick (menu cursor move)
        #   ("wait", seconds)   wait for loads/cutscenes
        #   ("key", keysym)     keyboard (only when use_gamepad=False)
        # The menu macro runs at NORMAL speed (game must advance to load); we (near-)freeze only
        # after reaching the start state, so the first inference doesn't let the game run.
        if self._sh is not None:
            self._sh.unpause()
        if self._pad is not None:
            self._pad.neutral()
        macro = (scenario.success_spec or {}).get("reset_macro", self.reset_macro)
        for entry in macro:
            kind = entry[0]
            if kind == "btn":
                self._press_btn(int(entry[1]))
            elif kind == "dir":
                self._hold_dir(float(entry[1]), float(entry[2]))
            elif kind == "wait":
                time.sleep(float(entry[1]))
            elif kind == "key":
                self._press(entry[1])
        if self._sh is not None:
            self._sh.pause()
        self._step = 0
        return Observation(frame=self._grab(), state={"step": 0}, step_idx=0, done=False)

    def step(self, action_chunk: np.ndarray) -> Observation:
        if self._pad is not None:
            self._apply_chunk(action_chunk)
            self._pad.neutral()
        else:  # keyboard fallback (discretized) — not recommended; loses analog
            self._step_keyboard(action_chunk)
        self._step += 1
        return Observation(frame=self._grab(), state={"step": self._step},
                           step_idx=self._step, done=False)

    def _step_keyboard(self, action_chunk: np.ndarray) -> None:
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        keys = []
        mx, my = float(a[:, JLX].mean()), float(a[:, JLY].mean())
        if mx < -MOVE_THRESH: keys.append("Left")
        if mx > MOVE_THRESH: keys.append("Right")
        if my < -MOVE_THRESH: keys.append("Up")
        if my > MOVE_THRESH: keys.append("Down")
        if (a[:, :N_BUTTONS][:, 0] > 0.5).mean() >= 0.3: keys.append(KEY_JUMP)
        for k in keys:
            self._xdo("keydown", "--window", self._wid, k)
        time.sleep(self.chunk_seconds)
        for k in keys:
            self._xdo("keyup", "--window", self._wid, k)

    def save_frame(self, path: str) -> None:
        from PIL import Image
        Image.fromarray(self._grab()).save(path)

    def close(self) -> None:
        if self._pad is not None:
            self._pad.close()
            self._pad = None
        if self._sh is not None:
            self._sh.close()
            self._sh = None
        for proc in (self._game, self._xvfb):
            if proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        lock = f"/tmp/.X{self.display}-lock"
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except OSError:
                pass


# ---- v2 (deterministic) engine-patch protocol, for reference --------------------------
# A doukutsu-rs fork with `--agent-socket <addr>` would enable a synchronous, exact env:
#   - a socket-driven PlayerController (template: src/input/dummy_player_controller.rs or
#     replay_player_controller.rs) that blocks each tick reading one input message;
#   - export player state each tick (src/game/player Player {x,y,life,...} + stage id + flags);
#   - send the rendered framebuffer (already in the GL backend) as PNG/raw.
#   Protocol (one JSON msg per control tick):
#     py->engine: {"buttons":{...},"move":{"left":bool,...},"frames":N} | {"reset":{"save":k}}
#     engine->py: {"frame_b64":..,"state":{"x":..,"y":..,"map":..,"life":..,"flags":[..]},"done":bool}
# This removes real-time noise and gives exact StatePredicateDetector success detection.
