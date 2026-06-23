"""Mighty Retro Zero under system Wine."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .wine_game_env import WineGameEnv

EXE_NAME = "MightyRetroZero_alpha011.exe"
ARCHIVE_NAME = "MightyRetroZero_alpha_011.sh"
REPO = Path(__file__).resolve().parents[3]
BUILD_ROOT = REPO / ".nitrogen-env-build" / "mighty_retro_zero"


def _candidate_exes() -> list[Path]:
    paths: list[Path] = []
    if os.environ.get("NITROGEN_MRZ_EXE"):
        paths.append(Path(os.environ["NITROGEN_MRZ_EXE"]).expanduser())
    paths.extend(
        [
            Path("/tmp/mrz/cde-root/tmp/Mighty") / EXE_NAME,
            Path("/tmp/mrz/Mighty") / EXE_NAME,
            Path("/tmp/mrz") / EXE_NAME,
            Path("/tmp/Mighty") / EXE_NAME,
            Path("/tmp") / EXE_NAME,
        ]
    )
    return paths


def _candidate_archives() -> list[Path]:
    paths: list[Path] = []
    if os.environ.get("NITROGEN_MRZ_ARCHIVE"):
        paths.append(Path(os.environ["NITROGEN_MRZ_ARCHIVE"]).expanduser())
    paths.extend([Path("/tmp") / ARCHIVE_NAME, BUILD_ROOT / ARCHIVE_NAME])
    paths.extend(sorted(Path("/tmp").glob("MightyRetroZero*.sh")))
    return paths


def _locate_exe(auto_extract: bool = True) -> Path:
    for path in _candidate_exes():
        if path.exists():
            return path
    if auto_extract:
        target = Path(os.environ.get("NITROGEN_MRZ_EXTRACT_DIR", "/tmp/mrz")).expanduser()
        for archive in _candidate_archives():
            if not archive.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["sh", str(archive), "--noexec", "--keep", "--target", str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=180,
            )
            matches = sorted(target.rglob(EXE_NAME))
            if matches:
                return matches[0]
    searched = "\n  ".join(str(p) for p in _candidate_exes())
    raise FileNotFoundError(f"{EXE_NAME} not found. Searched:\n  {searched}")


class MightyRetroZeroEnv(WineGameEnv):
    """Clickteam-Fusion Mega-Man-style run-n-gun: arrows, Z jump, X shoot."""

    name = "mighty_retro_zero"
    window_name = "Mighty"
    window_name_candidates = ("Mighty Retro Zero", "MightyRetroZero", "Mighty", EXE_NAME)

    def __init__(
        self,
        exe_path: str | os.PathLike[str] | None = None,
        wineprefix: str | os.PathLike[str] | None = None,
        boot_wait: float = 20.0,
        auto_extract: bool = True,
        **kw,
    ):
        exe = Path(exe_path).expanduser() if exe_path else _locate_exe(auto_extract=auto_extract)
        prefix = wineprefix or os.environ.get(
            "NITROGEN_MRZ_WINEPREFIX", f"/tmp/mrz-wineprefix-{os.getpid()}"
        )
        super().__init__(exe_path=exe, wineprefix=prefix, boot_wait=boot_wait, **kw)

    def launch_cmd(self) -> list[str]:
        exe = Path(self.exe_path)
        cmd = (
            f"cd {shlex.quote(str(exe.parent))} && "
            f"exec {shlex.quote(self.wine_binary)} {shlex.quote('./' + exe.name)}"
        )
        return ["sh", "-c", cmd]

    def _find_window(self) -> str:
        for name in self.window_name_candidates:
            out = subprocess.run(
                ["xdotool", "search", "--name", name],
                env=self._tool_env(),
                capture_output=True,
                text=True,
            )
            ids = [line for line in out.stdout.split() if line.strip()]
            if ids:
                return ids[-1]
        out = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", "."],
            env=self._tool_env(),
            capture_output=True,
            text=True,
        )
        ids = [line for line in out.stdout.split() if line.strip()]
        return ids[-1] if ids else ""

    def reset_macro(self, scenario):
        return [("wait", 0.5), ("key", "x"), ("wait", 1.5)]
