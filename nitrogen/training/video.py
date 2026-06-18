"""Video frame ingestion for the action-only NitroGen dataset.

The dataset ships actions only; frames must be fetched from the source video URL
in each chunk's metadata.json. This module downloads (and caches) the relevant
slice of a video with yt-dlp + ffmpeg and extracts frames at given timestamps,
masks the on-screen controller overlay bbox, and resizes to the model resolution.

NOTE on access: YouTube blocks unauthenticated requests from datacenter IPs
("Sign in to confirm you're not a bot"). Provide a Netscape cookies file via
`cookies_file=` (export from a logged-in browser) or a `proxy=`. Without one of
these, frame fetching will fail in such environments. All non-network logic
(bbox masking, resizing, action assembly) is independent of this.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VideoFetchConfig:
    cache_dir: str = "/home/t-nagupta/NitroGen/frame_cache"
    cookies_file: Optional[str] = None      # Netscape cookies.txt for YouTube auth
    cookies_from_browser: Optional[str] = None  # e.g. "chrome", "firefox"
    proxy: Optional[str] = None
    out_size: int = 256                      # SigLIP input resolution
    fmt: str = "bestvideo[height<=720]/best[height<=720]/best"
    quiet: bool = True
    # --- YouTube anti-bot plumbing (required as of 2026) ---
    # PO Token: requires the bgutil provider HTTP server running (default
    # http://127.0.0.1:4416); yt-dlp auto-detects it. n-challenge: requires a JS
    # runtime (Deno recommended) on PATH + the EJS solver scripts, enabled via
    # remote_components. See docs/SETUP_YOUTUBE.md.
    remote_components: tuple = ("ejs:github",)
    deno_path: Optional[str] = "/home/t-nagupta/.deno/bin"  # prepended to PATH if set


class VideoFrameFetcher:
    def __init__(self, config: VideoFetchConfig):
        self.cfg = config
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        # Ensure the JS runtime (Deno) used for the n-challenge is discoverable.
        if self.cfg.deno_path and self.cfg.deno_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = self.cfg.deno_path + os.pathsep + os.environ.get("PATH", "")

    # ---- yt-dlp options shared across calls
    def _ydl_auth_opts(self) -> dict:
        opts = {}
        if self.cfg.cookies_file:
            opts["cookiefile"] = self.cfg.cookies_file
        if self.cfg.cookies_from_browser:
            opts["cookiesfrombrowser"] = (self.cfg.cookies_from_browser,)
        if self.cfg.proxy:
            opts["proxy"] = self.cfg.proxy
        if self.cfg.remote_components:
            opts["remote_components"] = list(self.cfg.remote_components)
        return opts

    def _cache_path(self, video_id: str, start: float, end: float) -> str:
        key = f"{video_id}_{start:.2f}_{end:.2f}"
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        return os.path.join(self.cfg.cache_dir, f"{video_id}_{h}.mp4")

    def download_slice(self, url: str, video_id: str, start: float, end: float) -> str:
        """Download [start, end] seconds of a video to the cache; return path.

        Uses yt-dlp download_ranges so we don't fetch the whole video.
        """
        out_path = self._cache_path(video_id, start, end)
        if os.path.exists(out_path):
            return out_path
        import yt_dlp
        from yt_dlp.utils import download_range_func

        opts = {
            "format": self.cfg.fmt,
            "outtmpl": out_path,
            "quiet": self.cfg.quiet,
            "no_warnings": self.cfg.quiet,
            "download_ranges": download_range_func(None, [(start, end)]),
            "force_keyframes_at_cuts": True,
            **self._ydl_auth_opts(),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return out_path

    def extract_frame(self, video_path: str, t_in_slice: float, size: int | None = None) -> np.ndarray:
        """Extract a single RGB frame (H,W,3 uint8) at `t_in_slice` seconds into
        the downloaded slice, using ffmpeg.

        If `size` is None, the frame is extracted at its native resolution (so the
        controller bbox, which is in native pixel space, can be masked directly,
        and the model's AutoImageProcessor can do resize+normalize afterward).
        """
        scale = ["-vf", f"scale={size}:{size}"] if size else []
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-ss", f"{t_in_slice:.3f}", "-i", video_path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", *scale, "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            raise RuntimeError(f"ffmpeg frame extract failed: {proc.stderr.decode()[:200]}")
        from PIL import Image
        import io
        return np.asarray(Image.open(io.BytesIO(proc.stdout)).convert("RGB"))

    @staticmethod
    def mask_controller(frame: np.ndarray, bbox_xywh, resolution_hw=None) -> np.ndarray:
        """Zero out the on-screen controller overlay region (in native pixel space).

        `bbox_xywh` is in the ORIGINAL video pixel space from metadata.json. If
        `frame` is at a different resolution than `resolution_hw`, the bbox is
        rescaled; otherwise it is applied directly.
        """
        H_out, W_out = frame.shape[:2]
        x, y, w, h = bbox_xywh
        if resolution_hw is not None:
            orig_h, orig_w = resolution_hw
            sx, sy = W_out / orig_w, H_out / orig_h
        else:
            sx = sy = 1.0
        x0, y0 = max(0, int(x * sx)), max(0, int(y * sy))
        x1, y1 = min(W_out, int((x + w) * sx)), min(H_out, int((y + h) * sy))
        out = frame.copy()
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 0
        return out
