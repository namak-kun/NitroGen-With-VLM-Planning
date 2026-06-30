"""yt_farm.py -- YouTube gameplay -> frames (+ metadata/chapters) farming for NitroGen data bootstrap.

Wraps yt-dlp with the bits this box needs to get past YouTube's JS "n-challenge":
  * cookies file (NitroGen/cookies.txt)
  * an external JS runtime (deno) + the EJS challenge-solver scripts fetched from npm

Validated 2026-06-24: downloads precise time-ranges of a 2.6h SMW longplay at 360p and ffmpeg-extracts
clean gameplay frames. Frames are for the DiT/plan conditioning + (later) IDM action pseudo-labeling.

Examples:
  # find candidate longplays for a game
  python yt_farm.py search "Super Mario World 100% longplay" -n 5
  # metadata + chapters (chapters == free coarse OBJECTIVES if the uploader added them)
  python yt_farm.py meta "https://www.youtube.com/watch?v=ID"
  # download time windows (sec) and extract frames at 4 fps -> docs/yt_farm/<id>/
  python yt_farm.py grab "https://www.youtube.com/watch?v=ID" --sections 600-660 1200-1260 --fps 4

NOTE (copyright): commercial-game longplays are processed LOCALLY for research only; do not redistribute
the source video. ROMs/video stay on disk; only derived frames/actions feed training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COOKIES = REPO / "cookies.txt"
DENO = Path(os.environ.get("DENO_BIN", "/home/t-nagupta/.deno/bin/deno"))
OUT_ROOT = REPO / "docs" / "yt_farm"
PY = sys.executable


def _base_cmd() -> list[str]:
    """yt-dlp invocation with cookies + deno EJS so YouTube's n-challenge gets solved."""
    cmd = [PY, "-m", "yt_dlp", "--no-warnings"]
    if COOKIES.exists():
        cmd += ["--cookies", str(COOKIES)]
    if DENO.exists():
        cmd += ["--js-runtimes", f"deno:{DENO}", "--remote-components", "ejs:npm"]
    return cmd


def search(query: str, n: int = 5) -> list[dict]:
    cmd = _base_cmd() + ["--skip-download", "--flat-playlist", "-J", f"ytsearch{n}:{query}"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    data = json.loads(out.stdout or "{}")
    rows = []
    for e in data.get("entries", []):
        rows.append({"id": e.get("id"), "title": e.get("title"),
                     "duration": e.get("duration"), "channel": e.get("channel")})
    return rows


def meta(url: str) -> dict:
    cmd = _base_cmd() + ["--skip-download", "-J", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    d = json.loads(out.stdout or "{}")
    return {"id": d.get("id"), "title": d.get("title"), "duration": d.get("duration"),
            "uploader": d.get("uploader"),
            "chapters": [{"start": c.get("start_time"), "end": c.get("end_time"),
                          "title": c.get("title")} for c in (d.get("chapters") or [])]}


def grab(url: str, sections: list[str], fps: float = 4.0, height: int = 360, fmt: str | None = None) -> Path:
    """Download the given second-ranges (e.g. '600-660') and extract frames at fps. Returns out dir.

    Format: prefer PROGRESSIVE (combined audio+video) streams. yt-dlp video-ONLY formats (`bestvideo...`)
    reliably 403 on `--download-sections` range fetches (confirmed); progressive `best[...]` / itag 18 (360p
    mp4) work. Order progressive first, fall back to muxed video+audio only as a last resort."""
    vid = url.rsplit("=", 1)[-1].rsplit("/", 1)[-1]
    out_dir = OUT_ROOT / vid
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    fstr = fmt or (f"best[height<={height}][ext=mp4]/18/best[height<={height}]/"
                   f"bestvideo[height<={height}][ext=mp4]+bestaudio/best")
    cmd = _base_cmd() + [
        "-f", fstr,
        "-o", str(out_dir / "clip_%(section_start)s.%(ext)s"),
    ]
    for s in sections:
        a, b = s.split("-")
        cmd += ["--download-sections", f"*{a}-{b}"]
    cmd.append(url)
    subprocess.run(cmd, check=True, timeout=900)
    # extract frames from every downloaded clip
    n = 0
    for clip in sorted(out_dir.glob("clip_*.mp4")):
        tag = clip.stem.replace("clip_", "")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip),
                        "-vf", f"fps={fps}", str(out_dir / "frames" / f"{tag}_%04d.png")],
                       check=True)
    n = len(list((out_dir / "frames").glob("*.png")))
    (out_dir / "meta.json").write_text(json.dumps(meta(url), indent=2))
    print(f"[yt_farm] {vid}: {n} frames -> {out_dir/'frames'}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-n", type=int, default=5)
    m = sub.add_parser("meta"); m.add_argument("url")
    g = sub.add_parser("grab"); g.add_argument("url")
    g.add_argument("--sections", nargs="+", required=True, help="second ranges e.g. 600-660 1200-1260")
    g.add_argument("--fps", type=float, default=4.0); g.add_argument("--height", type=int, default=360)
    g.add_argument("--format", default=None, help="override yt-dlp -f format string (else progressive default)")
    a = ap.parse_args()
    if a.cmd == "search":
        for r in search(a.query, a.n):
            print(f"{r['id']}  {int(r['duration'] or 0)//60}m  {r['title']}  [{r['channel']}]")
    elif a.cmd == "meta":
        print(json.dumps(meta(a.url), indent=2))
    elif a.cmd == "grab":
        grab(a.url, a.sections, a.fps, a.height, a.format)


if __name__ == "__main__":
    main()
