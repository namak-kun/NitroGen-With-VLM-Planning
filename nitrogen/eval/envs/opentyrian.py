"""OpenTyrian env — vertical shmup control via keyboard.

OpenTyrian controls (README): arrows move the ship, Space fires weapons.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_RTRIG, I_SOUTH = 16, 18
STICK_THRESH = 0.2


def _game_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = []
    if os.environ.get("OPENTYRIAN_DIR"):
        candidates.append(Path(os.environ["OPENTYRIAN_DIR"]))
    candidates.extend([
        Path("/tmp/opentyrian"),
        repo_root / ".nitrogen-env-build" / "opentyrian",
    ])
    for path in candidates:
        if (path / "opentyrian").is_file():
            return path
    return candidates[-1]


class OpenTyrianEnv(ProcGameEnv):
    name = "opentyrian"
    window_name = "OpenTyrian"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 12.0, **kw):
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)

    def launch_cmd(self):
        game_dir = _game_dir()
        data_dir = game_dir / "data" / "tyrian21"
        return [
            "sh",
            "-c",
            "cd {game_dir} && SDL_AUDIODRIVER=dummy exec {binary} "
            "-t {data_dir} -j -s -x".format(
                game_dir=shlex.quote(str(game_dir)),
                binary=shlex.quote(str(game_dir / "opentyrian")),
                data_dir=shlex.quote(str(data_dir)),
            ),
        ]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "space",
                I_RTRIG: "space",
            },
            button_frac=0.3,
        )

    def reset_macro(self, scenario):
        return [
            ("wait", 0.5),
            ("key", "Return"),
            ("wait", 0.7),
            ("key", "Down"),
            ("wait", 0.2),
            ("key", "Return"),
            ("wait", 0.7),
            ("key", "Return"),
            ("wait", 0.7),
            ("key", "Return"),
            ("wait", 3.0),
        ]
