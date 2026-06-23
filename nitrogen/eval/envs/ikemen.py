"""Ikemen-GO quick-versus env using bundled free KFM content."""
from __future__ import annotations

import os
import signal
import shlex
import time
from pathlib import Path

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_EAST, I_LSHLD, I_LTRIG, I_NORTH = 5, 7, 9, 10
I_RSHLD, I_RTRIG, I_SOUTH, I_START, I_WEST = 14, 16, 18, 19, 20
STICK_THRESH = 0.2
BUTTON_FRAC = 0.3


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_game_dir() -> Path:
    candidates = [
        os.environ.get("NITROGEN_IKEMEN_DIR"),
        "/tmp/ikemen",
        str(_repo_root() / "tmp" / "ikemen"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return Path(candidate).expanduser()
    return Path(candidates[-1])


class IkemenEnv(ProcGameEnv):
    name = "ikemen"
    window_name = "Ikemen"
    control = "keyboard"

    def __init__(
        self,
        game_dir: str | os.PathLike | None = None,
        p1: str = "kfm",
        p2: str = "kfm",
        stage: str = "stages/kfm.def",
        p2_ai: int = 4,
        width: int = 800,
        height: int = 600,
        boot_wait: float = 18.0,
        freeze_during_inference: bool = True,
        **kw,
    ):
        self.game_dir = Path(game_dir).expanduser() if game_dir is not None else _default_game_dir()
        self.binary = self._find_binary()
        self.p1, self.p2, self.stage, self.p2_ai = p1, p2, stage, p2_ai
        self._sigstop_freeze = freeze_during_inference
        self._game_stopped = False
        super().__init__(
            width=width,
            height=height,
            boot_wait=boot_wait,
            freeze_during_inference=False,
            **kw,
        )

    def _find_binary(self) -> Path:
        for name in ("Ikemen_GO_Linux", "Ikemen_GO_linux", "Ikemen_GO"):
            path = self.game_dir / name
            if path.exists():
                return path
        found = sorted(p for p in self.game_dir.glob("Ikemen*") if p.is_file() and os.access(p, os.X_OK))
        if found:
            return found[0]
        raise FileNotFoundError(
            f"Ikemen-GO binary not found under {self.game_dir}; set NITROGEN_IKEMEN_DIR"
        )

    def launch_cmd(self):
        if not (self.game_dir / "chars" / self.p1).exists():
            raise FileNotFoundError(f"Ikemen free character missing: chars/{self.p1}")
        if not (self.game_dir / self.stage).exists():
            raise FileNotFoundError(f"Ikemen free stage missing: {self.stage}")
        argv = [
            f"./{self.binary.name}",
            "-windowed",
            "-width",
            str(self.width),
            "-height",
            str(self.height),
            "-nojoy",
            "-nomusic",
            "-p1",
            self.p1,
            "-p2",
            self.p2,
            "-p2.ai",
            str(self.p2_ai),
            "-s",
            self.stage,
            "-time",
            "-1",
            "-rounds",
            "999",
        ]
        lib_dir = self.game_dir / "lib"
        script = (
            f"cd {shlex.quote(str(self.game_dir))} && "
            "export SDL_AUDIODRIVER=dummy LIBGL_ALWAYS_SOFTWARE=1 "
            f"LD_LIBRARY_PATH={shlex.quote(str(lib_dir))}; "
            f"exec {shlex.join(argv)}"
        )
        return ["sh", "-c", script]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_frac=BUTTON_FRAC,
            button_map={
                I_SOUTH: "z",   # KFM command a
                I_WEST: "x",    # KFM command b
                I_RTRIG: "c",   # KFM command c; vet_env's strong action uses this
                I_EAST: "a",    # KFM command x
                I_NORTH: "s",   # KFM command y
                I_RSHLD: "d",   # KFM command z
                I_LSHLD: "q",   # KFM command d
                I_LTRIG: "w",   # KFM command w
                I_START: "Return",
            },
        )

    def reset_macro(self, scenario):
        return [("wait", 0.5)]

    def _resume_game(self) -> None:
        if self._sigstop_freeze and self._game_stopped and self._game is not None:
            if self._game.poll() is None:
                os.kill(self._game.pid, signal.SIGCONT)
                time.sleep(0.03)
            self._game_stopped = False

    def _pause_game(self) -> None:
        if self._sigstop_freeze and not self._game_stopped and self._game is not None:
            if self._game.poll() is None:
                os.kill(self._game.pid, signal.SIGSTOP)
                self._game_stopped = True
                time.sleep(0.03)

    def reset(self, scenario=None):
        self._resume_game()
        obs = super().reset(scenario)
        self._pause_game()
        obs.frame = self._grab()
        return obs

    def _apply(self, action_chunk):
        self._resume_game()
        try:
            super()._apply(action_chunk)
        finally:
            self._pause_game()

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            out_dir = _repo_root() / "tmp" / "ikemen"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = str(out_dir / Path(path).name)
        super().save_frame(path)

    def close(self) -> None:
        try:
            self._resume_game()
            super().close()
        finally:
            if getattr(self, "_sh", None) is not None:
                self._sh.close()
