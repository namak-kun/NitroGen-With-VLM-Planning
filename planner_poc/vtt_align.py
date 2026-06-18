"""Parse YouTube VTT (with word-level timestamps), dedup the auto-caption rolling-window
repetition, and expose (start_sec, end_sec, text) cues. Then align cues to a chunk's
[start_time, end_time] slice window.
"""
import re
import glob
import os


def _ts(s):
    h, m, rest = s.split(":")
    sec, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def parse_vtt(path):
    """Return list of (start, end, text) cues with inline <c> word tags stripped and the
    auto-caption rolling duplication removed (keep only the NEW text per cue)."""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    cues = []
    block = None
    for line in raw.splitlines():
        m = re.match(r"(\d\d:\d\d:\d\d\.\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d\.\d\d\d)", line)
        if m:
            if block:
                cues.append(block)
            block = [_ts(m.group(1)), _ts(m.group(2)), []]
        elif block is not None and line.strip() and not line.startswith("WEBVTT") \
                and not line.startswith("Kind:") and not line.startswith("Language:"):
            txt = re.sub(r"<[^>]+>", "", line).strip()
            if txt:
                block[2].append(txt)
    if block:
        cues.append(block)
    # join lines, dedup rolling repetition: each cue's text often = prev tail + new words
    out = []
    prev = ""
    for s, e, lines in cues:
        text = " ".join(lines).strip()
        if not text:
            continue
        # remove the longest prefix that equals a suffix of prev (rolling overlap)
        new = text
        if prev:
            words = text.split()
            pw = prev.split()
            # find largest k s.t. last k words of prev == first k words of text
            best = 0
            for k in range(1, min(len(words), len(pw)) + 1):
                if pw[-k:] == words[:k]:
                    best = k
            new = " ".join(words[best:])
        if new.strip():
            out.append((s, e, new.strip()))
        prev = text
    return out


def window_text(cues, start, end):
    """All cue text whose [start,end] overlaps the [start,end] window (seconds)."""
    parts = [t for (s, e, t) in cues if e >= start and s <= end]
    return " ".join(parts).strip()


def best_vtt(transcript_dir, vid):
    """Prefer manual (.en.vtt) over auto (.en-orig.vtt)."""
    for suf in (".en.vtt", ".en-orig.vtt"):
        p = os.path.join(transcript_dir, vid + suf)
        if os.path.exists(p):
            return p
    g = glob.glob(os.path.join(transcript_dir, vid + "*.vtt"))
    return g[0] if g else None


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    cues = parse_vtt(p)
    print(f"{len(cues)} deduped cues")
    for s, e, t in cues[:15]:
        print(f"  [{s:7.1f}-{e:7.1f}] {t}")
