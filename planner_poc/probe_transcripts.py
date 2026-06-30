"""Transcript feasibility probe (Stage 2). For each unique source video in the working
set, check whether YouTube captions (manual or auto) exist and, if so, fetch + measure
basic stats: how much text, density (words/min), and a crude "motor-plan-ness" keyword
hit rate (directional/action words). This is a GO/NO-GO probe for Stage 2 BEFORE building
the alignment pipeline: if transcripts are sparse, absent, or pure chit-chat, Stage 2 is a
data-quality problem, not a modeling one.
"""
import glob, json, os, re, subprocess, sys, tempfile

COOKIES = os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "cookies.txt")
DENO = os.path.expanduser("~/.deno/bin")
if DENO not in os.environ.get("PATH", ""):
    os.environ["PATH"] = DENO + os.pathsep + os.environ.get("PATH", "")

# motor/plan vocabulary — words a streamer would say IF narrating near-term intent
PLAN_WORDS = set("""left right up down jump dash attack hit kick punch block dodge roll
grab boost drift brake turn go stop move run walk climb shoot fire reload aim throw
forward back behind around over under push pull switch combo special""".split())


def videos():
    seen = {}
    for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True):
        m = json.load(open(md)); ov = m["original_video"]
        seen.setdefault(ov["video_id"], {"url": ov["url"], "game": m.get("game"),
                                         "start": ov.get("start_time"), "end": ov.get("end_time")})
    return seen


def probe_one(vid, info):
    """Return dict of availability + stats. Uses yt-dlp to list+fetch subs to a temp dir."""
    out = {"vid": vid, "game": info["game"], "has_manual": False, "has_auto": False,
           "words": 0, "dur_min": 0.0, "wpm": 0.0, "plan_hit": 0.0}
    with tempfile.TemporaryDirectory() as td:
        # list available subs first
        cmd = [".venv/bin/yt-dlp", "--cookies", COOKIES, "--skip-download", "--list-subs",
               "--extractor-args", "youtube:player_client=default,-tv",
               info["url"]]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            listing = r.stdout + r.stderr
        except Exception as e:
            out["error"] = repr(e)[:80]; return out
        out["has_manual"] = ("Available subtitles" in listing and "en" in listing)
        out["has_auto"] = "Available automatic captions" in listing and re.search(r"\ben\b", listing) is not None
        if not (out["has_manual"] or out["has_auto"]):
            return out
        # fetch (prefer manual, fall back to auto) as vtt
        cmd2 = [".venv/bin/yt-dlp", "--cookies", COOKIES, "--skip-download",
                "--write-subs", "--write-auto-subs", "--sub-langs", "en.*",
                "--sub-format", "vtt", "--extractor-args", "youtube:player_client=default,-tv",
                "-o", os.path.join(td, "%(id)s.%(ext)s"), info["url"]]
        try:
            subprocess.run(cmd2, capture_output=True, text=True, timeout=180)
        except Exception as e:
            out["error"] = repr(e)[:80]; return out
        vtts = glob.glob(os.path.join(td, "*.vtt"))
        if not vtts:
            return out
        text = open(vtts[0], encoding="utf-8", errors="ignore").read()
        # strip vtt timing + tags
        lines = [l for l in text.splitlines() if l and "-->" not in l and not l.startswith("WEBVTT")
                 and not re.match(r"^\d+$", l)]
        words = re.findall(r"[a-zA-Z']+", " ".join(lines).lower())
        out["words"] = len(words)
        try:
            out["dur_min"] = (float(info["end"]) - float(info["start"])) / 60.0
        except Exception:
            out["dur_min"] = 0.0
        out["wpm"] = out["words"] / out["dur_min"] if out["dur_min"] else 0.0
        out["plan_hit"] = (sum(w in PLAN_WORDS for w in words) / len(words)) if words else 0.0
    return out


def main():
    vids = videos()
    print(f"probing {len(vids)} videos for transcripts...\n")
    rows = []
    for vid, info in vids.items():
        r = probe_one(vid, info)
        rows.append(r)
        avail = "manual" if r["has_manual"] else ("auto" if r["has_auto"] else "NONE")
        print(f"  {vid}  {str(r['game'])[:22]:22s} subs={avail:6s} words={r['words']:5d} "
              f"wpm={r['wpm']:5.1f} plan%={100*r['plan_hit']:4.1f}" + (f"  ERR {r.get('error')}" if r.get("error") else ""))
    n = len(rows)
    has = sum(r["has_manual"] or r["has_auto"] for r in rows)
    withtext = [r for r in rows if r["words"] > 0]
    print(f"\n=== SUMMARY ===")
    print(f"  videos with ANY en subs: {has}/{n}")
    if withtext:
        import statistics as st
        print(f"  median words: {int(st.median([r['words'] for r in withtext]))}")
        print(f"  median wpm:   {st.median([r['wpm'] for r in withtext]):.1f}")
        print(f"  median plan-word rate: {100*st.median([r['plan_hit'] for r in withtext]):.1f}%")
    print("\n  GO/NO-GO heuristic: plan-word rate >~2-3% and wpm >~60 suggests usable")
    print("  narration; <1% or very low wpm suggests chit-chat/music -> Stage 2 is a")
    print("  data-quality problem (would need motor-plan-dense channels or relabeling).")


if __name__ == "__main__":
    main()
