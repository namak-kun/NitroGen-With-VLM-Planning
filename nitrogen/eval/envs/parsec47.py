"""Parsec47 env — high-speed abstract bullet-hell shmup control via keyboard.

Parsec47 controls: arrows move the ship, Z fires, X slows / special.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .proc_game_env import ProcGameEnv, keys_from_dirs_and_buttons

I_RTRIG, I_SOUTH, I_WEST = 16, 18, 20
STICK_THRESH = 0.2
BUTTON_FRAC = 0.3


_SDL_SPEEDHACK_C = r"""
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/time.h>

typedef void (*delay_t)(uint32_t);

static volatile double *ctrl = (double *)0;
static delay_t real_delay = (delay_t)0;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t anc_real = 0;
static uint64_t anc_scaled = 0;
static double cur_scale = 1.0;
static int tick_init = 0;
static int g_init = 0;

static uint64_t real_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (uint64_t)tv.tv_sec * 1000ULL + (uint64_t)tv.tv_usec / 1000ULL;
}

static void ensure_init(void) {
    if (g_init) return;
    real_delay = (delay_t)dlsym(RTLD_NEXT, "SDL_Delay");
    const char *path = getenv("SPEEDHACK_CTRL");
    if (path) {
        int fd = open(path, O_RDWR);
        if (fd >= 0) {
            void *m = mmap((void *)0, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (m != MAP_FAILED) ctrl = (double *)m;
            close(fd);
        }
    }
    g_init = 1;
}

uint32_t SDL_GetTicks(void) {
    if (!g_init) ensure_init();
    uint64_t real = real_ms();
    pthread_mutex_lock(&lock);
    double target = ctrl ? *ctrl : 1.0;
    if (!tick_init) {
        anc_real = real;
        anc_scaled = 0;
        cur_scale = target;
        tick_init = 1;
    }
    if (target != cur_scale) {
        anc_scaled += (uint64_t)((double)(real - anc_real) * cur_scale);
        anc_real = real;
        cur_scale = target;
    }
    uint64_t scaled = anc_scaled + (uint64_t)((double)(real - anc_real) * cur_scale);
    pthread_mutex_unlock(&lock);
    return (uint32_t)scaled;
}

void SDL_Delay(uint32_t ms) {
    if (!g_init) ensure_init();
    double scale = ctrl ? *ctrl : 1.0;
    while (ctrl && scale <= 0.001) {
        usleep(10000U);
        scale = *ctrl;
    }
    uint32_t sleep_ms = ms;
    if (scale > 0.001) {
        sleep_ms = (uint32_t)((double)ms / scale);
    }
    if (real_delay) {
        real_delay(sleep_ms);
    } else {
        usleep((useconds_t)sleep_ms * 1000U);
    }
}
"""


def _build_sdl_speedhack(run_dir: Path) -> Path:
    src = run_dir / "parsec47_sdl_speedhack.c"
    so = run_dir / "libparsec47_sdl_speedhack.so"
    if not src.exists() or src.read_text() != _SDL_SPEEDHACK_C:
        src.write_text(_SDL_SPEEDHACK_C)
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", str(src), "-o", str(so), "-ldl", "-lpthread"],
            check=True,
        )
    return so


class Parsec47Env(ProcGameEnv):
    name = "parsec47"
    window_name = "PARSEC47"
    control = "keyboard"

    def __init__(self, width: int = 800, height: int = 600, boot_wait: float = 12.0, **kw):
        self.run_dir = Path.cwd() / ".nitrogen-env-build" / "parsec47"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._sdl_speedhack = _build_sdl_speedhack(self.run_dir)
        os.environ["TMPDIR"] = str(self.run_dir)
        tempfile.tempdir = str(self.run_dir)
        super().__init__(width=width, height=height, boot_wait=boot_wait, **kw)
        if self._sh is not None:
            self._sh.pause_scale = 0.0

    def _env(self) -> dict:
        env = super()._env()
        preload = env.get("LD_PRELOAD", "")
        sdl_speedhack = str(self._sdl_speedhack)
        env["LD_PRELOAD"] = f"{sdl_speedhack}:{preload}" if preload else sdl_speedhack
        return env

    def launch_cmd(self):
        return [
            "/usr/bin/env",
            "SDL_AUDIODRIVER=dummy",
            "/usr/games/parsec47",
            "-nosound",
            "-window",
        ]

    def action_to_keys(self, action_chunk):
        return keys_from_dirs_and_buttons(
            action_chunk,
            steer_thresh=STICK_THRESH,
            vert_thresh=STICK_THRESH,
            button_map={
                I_SOUTH: "z",
                I_RTRIG: "z",
                I_WEST: "x",
            },
            button_frac=BUTTON_FRAC,
        )

    def reset_macro(self, scenario):
        return [
            ("wait", 0.5),
            ("key", "z"),
            ("wait", 1.0),
        ]

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
            env=dict(os.environ, DISPLAY=f":{self.display}"),
            capture_output=True,
        )
        n = self.width * self.height * 3
        if len(p.stdout) < n:
            return np.zeros((self.height, self.width, 3), np.uint8)
        return np.frombuffer(p.stdout[:n], np.uint8).reshape(self.height, self.width, 3).copy()

    def save_frame(self, path: str) -> None:
        if path.startswith("/tmp/"):
            path = str(self.run_dir / Path(path).name)
        super().save_frame(path)

    def close(self) -> None:
        try:
            super().close()
        finally:
            if getattr(self, "_sh", None) is not None:
                self._sh.close()
