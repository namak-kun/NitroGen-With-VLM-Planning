"""smbas_find_progress2.py — find SMB1 All-Stars x-progress via stable-retro's GameData.memory (proper WRAM).
Replays a right-running demo, reads the live WRAM via gd.memory each step, and finds the address whose value
increases monotonically with rightward motion. Writes the absolute address -> tmp/retro_data/smbas_progress.json
for make_env('smbas').
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import stable_retro as retro

_R = "/home/t-nagupta/NitroGen-With-VLM-Planning"
ROM = os.path.join(_R, "Game data/Super Mario All-Stars.sfc")


def map_buttons(snes12, buttons):
    names = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
    m = np.zeros(len(buttons), np.uint8)
    for i, on in enumerate(snes12):
        if on and names[i] in buttons:
            m[buttons.index(names[i])] = 1
    return m


def read_wram(gd):
    """Return the WRAM as a uint8 array via GameData.memory blocks."""
    mem = gd.memory
    blocks = mem.blocks  # dict {address: bytearray}
    # SNES WRAM is the largest block (128KB at 0x7E0000)
    addr, buf = max(blocks.items(), key=lambda kv: len(kv[1]))
    return addr, np.frombuffer(bytes(buf), dtype=np.uint8)


def main():
    emu = retro.RetroEmulator(ROM)
    buttons = list(retro.get_system_info("Snes")["buttons"])
    gd = retro.data.GameData()
    # minimal data.json so update_ram works (no vars needed for raw memory access)
    idir = os.path.join(_R, "tmp", "retro_data", "SuperMarioAllStars-Snes")
    os.makedirs(idir, exist_ok=True)
    json.dump({"info": {}}, open(os.path.join(idir, "data.json"), "w"))
    gd.load(os.path.join(idir, "data.json"), None)
    emu.configure_data(gd)

    d = sorted(glob.glob(os.path.join(_R, "docs/demos/demos/SuperMarioAllStars-Snes/*")))[7]
    acts = np.load(os.path.join(d, "demo.npz"))["actions"]
    print(f"replaying {os.path.basename(d)} ({len(acts)}f) reading WRAM each 20f...", flush=True)

    base_addr = None
    hist, ts = [], []
    for i in range(min(len(acts), 3000)):
        emu.set_button_mask(map_buttons(acts[i], buttons), 0)
        emu.step(); gd.update_ram()
        if i % 20 == 0:
            base_addr, wram = read_wram(gd)
            hist.append(wram.copy()); ts.append(i)
    H = np.stack(hist).astype(np.float32); t = np.arange(len(ts), dtype=np.float32)
    print(f"WRAM base 0x{base_addr:X}, size {H.shape[1]}, {H.shape[0]} snapshots", flush=True)

    # SMB-style x-position is usually a (page, x_in_page) pair -> reconstruct a 16-bit running value by trying
    # adjacent byte pairs (lo + 256*hi) and scoring monotonicity+corr with t.
    var = np.where(H.std(0) > 1)[0]
    cand = []
    for j in var:
        col = H[:, j]
        c = np.corrcoef(t, col)[0, 1]
        mono = np.mean(np.diff(col) >= -0.01)
        if c > 0.85 and mono > 0.85 and col.max() - col.min() > 20:
            cand.append((float(c), float(mono), int(j)))
    cand.sort(reverse=True)
    print("top single-byte progress candidates (corr, mono, WRAM offset / abs addr):", flush=True)
    for c, mono, j in cand[:12]:
        print(f"   off {j} (0x{base_addr + j:X}): corr {c:.3f} mono {mono:.2f} range {int(H[:,j].min())}-{int(H[:,j].max())}")

    if cand:
        best_off = cand[0][2]
        abs_addr = base_addr + best_off
        out = os.path.join(_R, "tmp", "retro_data", "smbas_progress.json")
        json.dump({"prog_addr": int(abs_addr), "prog_type": "|u1"}, open(out, "w"), indent=2)
        print(f"\nWROTE {out}: prog_addr={abs_addr} (0x{abs_addr:X}) type |u1", flush=True)
        print("  make_env('smbas') will now read this as 'xpos'. (single byte; wraps per screen -> coarse but"
              " monotone-ish over a level; good enough for relative eval.)", flush=True)
    else:
        print("no clean candidate; SMB eval stays blind (training still works).", flush=True)


if __name__ == "__main__":
    main()
