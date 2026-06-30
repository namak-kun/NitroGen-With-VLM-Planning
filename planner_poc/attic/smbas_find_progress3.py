"""smbas_find_progress3.py — SMB1 All-Stars world-x via a 2-BYTE (page,offset) scan, cross-validated across
demos (both models' R8 pick). SMB1 x-progress = 256*page + x_in_page; a single byte wraps each screen. We
replay several right-running demos, read live WRAM (GameData.memory), and for every adjacent (lo@a, hi@a+1)
pair compute v=256*hi+lo, scoring monotone-non-decreasing fraction WITHIN level segments (segment on big
drops = death/level reset). Require the SAME address to win across >=3 demos. Writes smbas_progress.json.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import stable_retro as retro

_R = "/home/t-nagupta/NitroGen-With-VLM-Planning"
ROM = os.path.join(_R, "Game data/Super Mario All-Stars.sfc")
NAMES = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


def mb(snes12, buttons):
    m = np.zeros(len(buttons), np.uint8)
    for i, on in enumerate(snes12):
        if on and NAMES[i] in buttons:
            m[buttons.index(NAMES[i])] = 1
    return m


def wram(gd):
    addr, buf = max(gd.memory.blocks.items(), key=lambda kv: len(kv[1]))
    return addr, np.frombuffer(bytes(buf), np.uint8)


def seg_mono(v):
    """monotone-non-decreasing fraction within segments (split on drops > 64 = level/death reset)."""
    d = np.diff(v.astype(np.int32))
    resets = np.where(d < -64)[0]
    good = tot = 0
    start = 0
    for r in list(resets) + [len(v) - 1]:
        seg = v[start:r + 1]
        if len(seg) > 3:
            dd = np.diff(seg.astype(np.int32))
            good += np.sum(dd >= 0); tot += len(dd)
        start = r + 1
    return good / max(tot, 1), (v.max() - v.min())


def scan_demo(emu, gd, buttons, acts, base_addr, S, every=15, maxf=3000):
    H = []
    for i in range(min(len(acts), maxf)):
        emu.set_button_mask(mb(acts[i], buttons), 0); emu.step(); gd.update_ram()
        if i % every == 0:
            _, w = wram(gd); H.append(w.copy())
    H = np.stack(H).astype(np.int32)
    # candidate addresses: low byte varies a lot
    var = np.where(H.std(0) > 1)[0]
    scores = {}
    for a in var:
        if a + 1 >= S:
            continue
        v = 256 * H[:, a + 1] + H[:, a]            # little-endian <u2
        mono, rng = seg_mono(v)
        if mono > 0.9 and rng > 100:
            scores[int(a)] = (float(mono), float(rng))
    return scores


def main():
    emu = retro.RetroEmulator(ROM)
    buttons = list(retro.get_system_info("Snes")["buttons"])
    gd = retro.data.GameData()
    idir = os.path.join(_R, "tmp", "retro_data", "SuperMarioAllStars-Snes")
    os.makedirs(idir, exist_ok=True)
    json.dump({"info": {}}, open(os.path.join(idir, "data.json"), "w"))
    gd.load(os.path.join(idir, "data.json"), None)
    emu.configure_data(gd)
    base_addr, w0 = wram(gd); S = len(w0)
    print(f"WRAM base 0x{base_addr:X} size {S}", flush=True)

    # use 4 long right-running demos (likely level completions): #8,#10,#13,#20 (high-activity ends)
    dirs = sorted(glob.glob(os.path.join(_R, "docs/demos/demos/SuperMarioAllStars-Snes/*")))
    pick = [dirs[i] for i in (7, 9, 12, 19) if i < len(dirs)]
    per_demo = []
    for d in pick:
        acts = np.load(os.path.join(d, "demo.npz"))["actions"]
        sc = scan_demo(emu, gd, buttons, acts, base_addr, S)
        per_demo.append(set(sc.keys()))
        print(f"  {os.path.basename(d)}: {len(sc)} candidate addrs", flush=True)
    # intersection across demos
    common = set.intersection(*per_demo) if per_demo else set()
    print(f"  addrs monotone in ALL {len(pick)} demos: {sorted(hex(base_addr+a) for a in common)[:10]}", flush=True)
    if common:
        # pick the lowest address (usually the canonical x-pos pair)
        a = sorted(common)[0]
        abs_addr = base_addr + a
        out = os.path.join(_R, "tmp", "retro_data", "smbas_progress.json")
        json.dump({"prog_addr": int(abs_addr), "prog_type": "<u2"}, open(out, "w"), indent=2)
        print(f"\nWROTE {out}: prog_addr={abs_addr} (0x{abs_addr:X}) <u2  => SMB1 now EVAL-ready", flush=True)
    else:
        print("\nno common 2-byte progress addr; SMB1 stays train-only.", flush=True)


if __name__ == "__main__":
    main()
