"""Zelda Classic / ZQuest Classic env.

Uses the prebuilt Linux ZQuest Classic release in
``.nitrogen-env-build/zelda_classic/runtime`` and its bundled classic module
quest.  The player boots directly into a standalone top-down Zelda-style quest.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_EAST, I_LTRIG, I_RTRIG, I_SOUTH, I_START, I_WEST = 5, 9, 16, 18, 19, 20
STICK_THRESH = 0.25
BUTTON_FRAC = 0.3

REPO = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPO / ".nitrogen-env-build" / "zelda_classic" / "runtime"
RUNS_ROOT = REPO / ".nitrogen-env-build" / "zelda_classic" / "runs"


class ZeldaClassicEnv(ProcGameEnv):
    name = "zelda_classic"
    window_name = "ZQuest Classic"
    control = "keyboard"

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 14.0,
        runtime_dir: str | None = None,
        quest_path: str | None = None,
        **kw,
    ):
        self.width, self.height = width, height
        self.runtime_dir = Path(runtime_dir) if runtime_dir else self._find_runtime()
        self.binary = self.runtime_dir / "bin" / "zplayer"
        self.quest_path = (
            Path(quest_path)
            if quest_path
            else self.runtime_dir / "share" / "zquestclassic" / "modules" / "classic" / "default.qst"
        )
        self.share_dir = self.runtime_dir / "share" / "zquestclassic"
        self._ensure_runtime_files()

        self._old_tmpdir_env = os.environ.get("TMPDIR")
        self._old_tempfile_tempdir = tempfile.tempdir
        self.run_dir = RUNS_ROOT / str(os.getpid())
        self.home_dir = self.run_dir / "home"
        self.xdg_config_dir = self.run_dir / "xdg_config"
        self.xdg_data_dir = self.run_dir / "xdg_data"
        self.tmp_dir = self.run_dir / "tmp"
        for path in (self.home_dir, self.xdg_config_dir, self.xdg_data_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.save_path = self.run_dir / "zelda_classic.sav"
        os.environ["TMPDIR"] = str(self.tmp_dir)
        tempfile.tempdir = str(self.tmp_dir)

        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def _find_runtime(self) -> Path:
        if os.environ.get("ZELDA_CLASSIC_DIR"):
            return Path(os.environ["ZELDA_CLASSIC_DIR"])
        return RUNTIME_ROOT

    def _ensure_runtime_files(self) -> None:
        if not self.binary.exists():
            raise FileNotFoundError(
                f"ZQuest Classic zplayer not found: {self.binary}. "
                "Expected the Linux release extracted under .nitrogen-env-build/zelda_classic/runtime."
            )
        if not self.quest_path.exists():
            raise FileNotFoundError(f"Zelda Classic quest not found: {self.quest_path}")

        (self.share_dir / "zc.cfg").write_text(
            "[Controls]\n"
            "global_control_scheme = Custom\n\n"
            "[zeldadx]\n"
            "replay_upload_prompt = 1\n"
            "replay_upload = 0\n"
            "replay_new_saves = 0\n"
            "nosound = 1\n"
            "pause_in_background = 0\n"
            "fullscreen = 0\n"
            f"window_width = {int(self.width)}\n"
            f"window_height = {int(self.height)}\n"
            "throttlefps = 1\n"
            "vsync = 0\n"
            "name_entry_mode = 0\n",
            encoding="utf-8",
        )

    def launch_cmd(self):
        env = [
            "SDL_AUDIODRIVER=dummy",
            "ALLEGRO_AUDIO_DRIVER=null",
            "LIBGL_ALWAYS_SOFTWARE=1",
            f"HOME={shlex.quote(str(self.home_dir))}",
            f"XDG_CONFIG_HOME={shlex.quote(str(self.xdg_config_dir))}",
            f"XDG_DATA_HOME={shlex.quote(str(self.xdg_data_dir))}",
            f"TMPDIR={shlex.quote(str(self.tmp_dir))}",
        ]
        cmd = (
            f"cd {shlex.quote(str(self.runtime_dir))} && "
            f"exec env {' '.join(env)} {shlex.quote(str(self.binary))} "
            f"-windowed -res {int(self.width)} {int(self.height)} -s "
            f"-standalone {shlex.quote(str(self.quest_path.resolve()))} "
            f"{shlex.quote(str(self.save_path.resolve()))}"
        )
        return ["bash", "-lc", cmd]

    def reset_macro(self, scenario):
        return [("wait", 0.2)]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "z",    # A / sword
                I_RTRIG: "z",    # vet_env strong action also swings
                I_WEST: "x",     # B / item
                I_LTRIG: "x",
                I_EAST: "Return",
                I_START: "Return",
            },
            button_frac=BUTTON_FRAC,
        )

    def _tool_env(self) -> dict:
        env = dict(os.environ, DISPLAY=f":{self.display}")
        env.pop("LD_PRELOAD", None)
        env.pop("SPEEDHACK_CTRL", None)
        return env

    def _xdo(self, *args):
        subprocess.run(
            ["xdotool", *args],
            env=self._tool_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _grab(self) -> np.ndarray:
        p = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-f",
                "x11grab",
                "-video_size",
                f"{self.width}x{self.height}",
                "-i",
                f":{self.display}.0",
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            env=self._tool_env(),
            capture_output=True,
        )
        n = self.width * self.height * 3
        if len(p.stdout) < n:
            return np.zeros((64, 64, 3), np.uint8)
        frame = np.frombuffer(p.stdout[:n], np.uint8).reshape(self.height, self.width, 3)
        crop_w = max(1, int(self.width * 64 / 800))
        crop_h = max(1, int(self.height * 64 / 600))
        x0 = int(self.width * 128 / 800)
        y0 = int(self.height * 173 / 600)
        x1, y1 = min(self.width, x0 + crop_w), min(self.height, y0 + crop_h)
        return frame[y0:y1, x0:x1].copy()

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            RUNS_ROOT.mkdir(parents=True, exist_ok=True)
            path = str(RUNS_ROOT / Path(path).name)
        super().save_frame(path)

    def close(self) -> None:
        try:
            super().close()
        finally:
            if getattr(self, "_sh", None) is not None:
                self._sh.close()
            if self._old_tmpdir_env is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = self._old_tmpdir_env
            tempfile.tempdir = self._old_tempfile_tempdir
            shutil.rmtree(self.run_dir, ignore_errors=True)
