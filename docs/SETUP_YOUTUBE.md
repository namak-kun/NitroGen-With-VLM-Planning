# YouTube frame ingestion — environment setup

The NitroGen dataset ships **actions only**; frames must be fetched from source
videos. As of 2026 YouTube enforces several anti-bot measures, so a bare yt-dlp
returns only storyboards ("No video formats found"). The full working recipe
(all verified working on this host):

## Requirements

1. **Cookies** — Netscape `cookies.txt` exported from a logged-in (throwaway)
   YouTube account. Lives at `NitroGen/cookies.txt` (git-ignored). Passed via
   `VideoFetchConfig.cookies_file`.

2. **PO Token provider (bgutil)** — defeats YouTube's Proof-of-Origin requirement.
   - yt-dlp plugin: `pip install bgutil-ytdlp-pot-provider` (installed in .venv).
   - Node server (built from `Brainicism/bgutil-ytdlp-pot-provider`):
     ```bash
     git clone https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git
     cd bgutil-ytdlp-pot-provider/server && npm install && npx tsc
     node build/main.js          # serves http://127.0.0.1:4416 — keep alive
     ```
   - yt-dlp auto-detects the server at `127.0.0.1:4416`. Health: `curl :4416/ping`.

3. **JS runtime for the `n`-challenge** — Deno (preferred):
   ```bash
   curl -fsSL https://deno.land/install.sh | sh -s -- -y   # -> ~/.deno/bin/deno
   ```
   `VideoFrameFetcher` prepends `~/.deno/bin` to PATH automatically
   (`VideoFetchConfig.deno_path`).

4. **EJS solver scripts** — yt-dlp downloads the challenge-solver lib on demand via
   `remote_components=['ejs:github']` (set by default in `VideoFetchConfig`).

5. **ffmpeg** — for frame extraction (`apt install ffmpeg`).

## Verified end-to-end

```
cookies + PO token (bgutil :4416) + Deno + EJS  ->  real mp4/webm formats
download-sections "*START-END"  ->  ~3-10 MB per 20s slice (cached)
ffmpeg -ss T -i slice.mp4 -frames:v 1  ->  native-res RGB frame
mask controller bbox -> AutoImageProcessor -> 256x256 SigLIP input
```

Survival: sampled YouTube-source videos were **16/16 live**; Twitch VODs ~1/21
(mostly expired — skip `source=="twitch"`).

## Gotchas

- **Plugin double-registration**: importing yt-dlp concurrently across threads
  raises `PoTokenProvider ... already registered`. Pre-fetch sequentially, or
  `import yt_dlp` once in the main thread before any worker pool.
- Cookies expire; re-export periodically. Keep the PO-token server running for the
  whole fetch/training session.

## Launching training with real frames

Ensure the PO-token server is running, then:
```bash
PATH=~/.deno/bin:$PATH python scripts/train_planner.py \
  --ng-ckpt ckpts/nitrogen/ng.pt --qwen ckpts/qwen35-0.8b \
  --shard-root <data-dir-with-chunks> \
  --cookies-file cookies.txt --steps 1000
```
(omit `--cookies-file` / add `--stub-frames` for a network-free dry run.)
