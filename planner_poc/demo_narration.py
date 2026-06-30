"""demo_narration.py — human timestamped narration for gold demos (System-2 rationale).

Each demo (docs/demos/demos/<Game>/<ts>/) gets a HUMAN-WRITTEN `narration.txt` you fill while watching
episode.mp4. This tool (a) scaffolds the skeleton and (b) parses it into a frame-aligned `narration.json`
that pairs each note with the EXACT actions/frames you took (episode.mp4 is 60fps real-time, frame == sec*60,
and demo.npz actions[i] <-> observations[i]). The narration is uncontaminated gold objective/rationale: which
level, what you intended, what was required vs bonus.

HUMAN FORMAT (narration.txt) — easiest to type while watching:
    # game: SuperMarioWorld-Snes        <- auto-filled headers (don't edit)
    # demo: 20260627-105939
    # fps: 60   duration_s: 127.97
    # level: <fill, or use @level markers below for multi-level demos>
    #
    # One note per line:  M:SS  your text   (M:SS as seen in episode.mp4; also S, S.s, MM:SS, H:MM:SS ok)
    #   - "@level <name>" as the text marks a level/area change at that time
    #   - "#" lines are comments/headers, ignored by the parser
    0:00 @level Donut Plains 1
    0:00 spawn; running right to build P-meter speed
    0:07 jumped early over the Rex to keep momentum (required)
    0:15 detoured up for the dragon coin — bonus, not needed to clear

USAGE:
    python planner_poc/demo_narration.py init               # scaffold narration.txt in every demo (skip existing)
    python planner_poc/demo_narration.py init --game SuperMarioWorld-Snes --force
    python planner_poc/demo_narration.py parse              # narration.txt -> narration.json (frame-aligned) for all
    python planner_poc/demo_narration.py parse --game SuperMarioWorld-Snes
"""
from __future__ import annotations
import argparse, glob, json, os, re

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS = os.path.join(_R, "docs/demos/demos")

TS_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)\s+(.*\S)\s*$")   # [H:]M:SS[.s] text
TS_SEC_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(.*\S)\s*$")                       # bare seconds  text
HDR_RE = re.compile(r"^\s*#\s*([a-zA-Z_]+)\s*:\s*(.*\S)\s*$")                     # # key: value


RANGE_RE = re.compile(r"^(\s*(?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?)\s*-\s*(?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?(\s+.*)$")


def _parse_ts(line: str):
    """Return (seconds, text) or None for a non-entry line. Accepts a range form 'M:SS-M:SS text'
    (uses the START time)."""
    r = RANGE_RE.match(line)            # 'M:SS-M:SS text' -> collapse to 'M:SS text' (start)
    if r:
        line = r.group(1) + r.group(2)
    m = TS_RE.match(line)
    if m:
        h = int(m.group(1)) if m.group(1) else 0
        return h * 3600 + int(m.group(2)) * 60 + float(m.group(3)), m.group(4)
    m = TS_SEC_RE.match(line)
    if m:
        return float(m.group(1)), m.group(2)
    return None


def _demo_dirs(game=None):
    pat = os.path.join(DEMOS, game or "*", "*")
    return [d for d in sorted(glob.glob(pat)) if os.path.isfile(os.path.join(d, "meta.json"))]


def cmd_init(args):
    n = 0
    for d in _demo_dirs(args.game):
        out = os.path.join(d, "narration.txt")
        if os.path.exists(out) and not args.force:
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        fps = meta.get("fps", 60)
        dur = meta.get("duration_seconds", "?")
        body = (
            f"# game: {meta.get('game_id', os.path.basename(os.path.dirname(d)))}\n"
            f"# demo: {os.path.basename(d)}\n"
            f"# fps: {fps}   duration_s: {dur}\n"
            f"# level: <fill, or use @level markers below for multi-level demos>\n"
            f"#\n"
            f"# One note per line:  M:SS  your text   (timestamps as seen in episode.mp4, real-time)\n"
            f"#   - \"@level <name>\" as the text marks a level/area change at that time\n"
            f"#   - mark bonus/optional actions explicitly (e.g. 'bonus, not required')\n"
            f"#   - '#' lines are comments, ignored by the parser\n"
            f"# Example:\n"
            f"#   0:00 @level <fill>\n"
            f"#   0:00 spawn; running right to build speed\n"
            f"#   0:07 jumped early to clear the enemy without losing momentum\n"
            f"\n"
        )
        with open(out, "w") as f:
            f.write(body)
        n += 1
        print(f"  scaffolded {os.path.relpath(out, _R)}")
    print(f"init: wrote {n} narration.txt skeleton(s)" + ("" if n else " (all already present; use --force to overwrite)"))


def _align_and_write(d, lines, total_box):
    """Parse narration `lines` (list of str) for demo dir `d`, frame-align, write narration.json. Returns
    n_entries. total_box is a 1-elem list accumulator."""
    meta = json.load(open(os.path.join(d, "meta.json")))
    fps = float(meta.get("fps", 60))
    num_steps = int(meta.get("num_steps", 0))
    dur = float(meta.get("duration_seconds", num_steps / fps if fps else 0))
    headers, entries, warnings = {}, [], []
    for ln, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        h = HDR_RE.match(line)
        if h:
            headers[h.group(1).lower()] = h.group(2)
            continue
        if line.lstrip().startswith("#"):
            continue
        p = _parse_ts(line)
        if p is None:
            warnings.append(f"L{ln}: unparseable, skipped: {line!r}")
            continue
        t, text = p
        frame = int(round(t * fps))
        if dur and t > dur + 0.5:
            warnings.append(f"L{ln}: t={t:.2f}s exceeds duration {dur:.2f}s")
        is_level = text.lower().startswith("@level")
        entries.append({"t": round(t, 3), "frame": frame,
                        "level_marker": text[6:].strip() if is_level else None,
                        "text": text})
    entries.sort(key=lambda e: e["frame"])
    cur_level = headers.get("level") if headers.get("level", "").strip() not in (
        "", "<fill, or use @level markers below for multi-level demos>") else None
    for i, e in enumerate(entries):
        e["frame_end"] = entries[i + 1]["frame"] if i + 1 < len(entries) else num_steps
        if e["level_marker"]:
            cur_level = e["level_marker"]
        e["level"] = cur_level
    out = {"game_id": headers.get("game", meta.get("game_id")),
           "demo": os.path.basename(d), "fps": fps, "num_steps": num_steps,
           "duration_s": dur, "headers": headers, "n_entries": len(entries), "entries": entries}
    with open(os.path.join(d, "narration.json"), "w") as f:
        json.dump(out, f, indent=2)
    total_box[0] += len(entries)
    flag = f"  [!] {len(warnings)} warning(s)" if warnings else ""
    print(f"  {out['game_id']}/{out['demo']}: {len(entries)} notes -> narration.json{flag}")
    for w in warnings[:6]:
        print(f"       {w}")
    return len(entries)


def cmd_parse(args):
    total = [0]
    for d in _demo_dirs(args.game):
        src = os.path.join(d, "narration.txt")
        if not os.path.exists(src):
            continue
        _align_and_write(d, list(open(src)), total)
    print(f"parse: {total[0]} narrated note(s) across demos")


def cmd_ingest(args):
    """Ingest an aggregated narration file (e.g. demo_explanations.md) with '## <demo-dir-path>' section
    headers, distributing each section to that demo's narration.json (frame-aligned). Also mirrors each
    section to the demo's narration.txt unless --no-mirror."""
    path = args.file if os.path.isabs(args.file) else os.path.join(_R, args.file)
    sections, cur_dir, cur_lines = [], None, []
    for raw in open(path):
        m = re.match(r"^##\s+(\S+)", raw)
        if m:
            if cur_dir is not None:
                sections.append((cur_dir, cur_lines))
            cur_dir, cur_lines = m.group(1), []
        elif cur_dir is not None:
            cur_lines.append(raw)
    if cur_dir is not None:
        sections.append((cur_dir, cur_lines))
    total = [0]
    for rel, lines in sections:
        d = rel if os.path.isabs(rel) else os.path.join(_R, rel)
        if not os.path.isfile(os.path.join(d, "meta.json")):
            print(f"  [skip] no meta.json at {rel}")
            continue
        if not args.no_mirror:
            with open(os.path.join(d, "narration.txt"), "w") as f:
                f.writelines(lines)
        _align_and_write(d, lines, total)
    print(f"ingest: {total[0]} note(s) from {os.path.relpath(path, _R)} across {len(sections)} section(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("init", help="scaffold narration.txt skeletons")
    pi.add_argument("--game", default=None, help="only this game dir (e.g. SuperMarioWorld-Snes)")
    pi.add_argument("--force", action="store_true", help="overwrite existing narration.txt")
    pi.set_defaults(func=cmd_init)
    pp = sub.add_parser("parse", help="narration.txt -> frame-aligned narration.json")
    pp.add_argument("--game", default=None)
    pp.set_defaults(func=cmd_parse)
    pg = sub.add_parser("ingest", help="ingest an aggregated file (## <demo-path> sections) -> per-demo narration.json")
    pg.add_argument("--file", default="demo_explanations.md", help="aggregated narration file (repo-relative or absolute)")
    pg.add_argument("--no-mirror", action="store_true", help="don't also write each section to the demo's narration.txt")
    pg.set_defaults(func=cmd_ingest)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
