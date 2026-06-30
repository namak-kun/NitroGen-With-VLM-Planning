"""smbas_find_progress.py — find the SMB1 (Super Mario All-Stars) horizontal-progress RAM address so we can
build a minimal stable-retro integration for eval (reward/save-state). Strategy: load a demo, replay its
actions while snapshotting the full RAM each chunk, then find the byte(s) that increase MONOTONICALLY with
rightward progress (high correlation with frame index during right-running, and that JUMP on level changes).

Writes the best candidate to tmp/retro_data/smbas_progress.json (consumed by make_env('smbas')).
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import stable_retro as retro

_R = "/home/t-nagupta/NitroGen-With-VLM-Planning"
ROM = os.path.join(_R, "Game data/Super Mario All-Stars.sfc")
DEMO_GLOB = "docs/demos/demos/SuperMarioAllStars-Snes/*"


def map_buttons(snes12, buttons):
    # demo 12-button order: B,Y,SELECT,START,UP,DOWN,LEFT,RIGHT,A,X,L,R
    names = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
    m = np.zeros(len(buttons), np.uint8)
    for i, on in enumerate(snes12):
        if on and names[i] in buttons:
            m[buttons.index(names[i])] = 1
    return m


def main():
    emu = retro.RetroEmulator(ROM)
    buttons = list(retro.get_system_info("Snes")["buttons"])

    # pick a long, RIGHT-heavy demo (clean monotone progress); demo #8 (194103) was high-activity end (a clear win)
    d = sorted(glob.glob(os.path.join(_R, DEMO_GLOB)))[7]
    z = np.load(os.path.join(d, "demo.npz"))
    acts = z["actions"]
    print(f"using demo {os.path.basename(d)}: {len(acts)} frames", flush=True)

    # replay, snapshot RAM every 30 frames
    ram_hist = []
    idxs = []
    emu.set_button_mask(np.zeros(len(buttons), np.uint8), 0)
    for i in range(0, min(len(acts), 3000)):
        emu.set_button_mask(map_buttons(acts[i], buttons), 0)
        emu.step()
        if i % 30 == 0:
            ram = np.frombuffer(bytes(emu.get_state()), dtype=np.uint8)  # full state incl RAM
            ram_hist.append(ram.copy()); idxs.append(i)
    H = np.stack(ram_hist)  # (T, S)
    t = np.arange(len(idxs), dtype=np.float32)
    print(f"snapshots {H.shape}; scanning for monotone-increasing bytes...", flush=True)

    # candidate single bytes: high positive correlation with t AND mostly non-decreasing
    T, S = H.shape
    Hf = H.astype(np.float32)
    # restrict to bytes that actually vary
    varying = np.where(Hf.std(0) > 2)[0]
    best = []
    for j in varying:
        col = Hf[:, j]
        c = np.corrcoef(t, col)[0, 1]
        mono = np.mean(np.diff(col) >= 0)
        if c > 0.8 and mono > 0.7:
            best.append((float(c), float(mono), int(j)))
    best.sort(reverse=True)
    print(f"top monotone-with-progress byte offsets (corr, mono%, offset):", flush=True)
    for c, mono, j in best[:15]:
        print(f"   offset {j}: corr {c:.3f} mono {mono:.2f} range {int(Hf[:,j].min())}-{int(Hf[:,j].max())}")

    # NOTE: these offsets are into the SAVE-STATE blob, not the live RAM address space. To get the actual
    # data.json address we need the WRAM offset. stable-retro data.json uses absolute addresses (e.g. SNES
    # WRAM 0x7E0000 = decimal 8257536). Try the common SMB1 x-pos pair (player x hi/lo) by absolute address.
    # SMB1 (NES) uses $006D (page) + $0086 (x in page); All-Stars SNES remaps. We emit the best save-state
    # offset for the env to read via a RAW reader fallback, plus print guidance.
    out = os.path.join(_R, "tmp", "retro_data")
    os.makedirs(out, exist_ok=True)
    cand = best[:5]
    json.dump({"savestate_offsets": [j for _, _, j in cand],
               "note": "save-state blob offsets, monotone with rightward progress; use raw-RAM reader"},
              open(os.path.join(out, "smbas_progress_scan.json"), "w"), indent=2)
    print(f"\nwrote scan -> tmp/retro_data/smbas_progress_scan.json", flush=True)
    print("NEXT: map a save-state offset to a WRAM address, or use a raw-offset reader in IntegrationEnv.", flush=True)


if __name__ == "__main__":
    main()
