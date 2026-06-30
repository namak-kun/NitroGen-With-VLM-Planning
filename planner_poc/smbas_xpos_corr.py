"""smbas_xpos_corr.py — find SMB1 All-Stars world-x by CORRELATING WRAM bytes with measured screen-scroll
(rightward progress) across a real demo. We don't trust online maps; the demo frames ARE the ground truth:
when Mario advances, the screen scrolls right (background columns shift left). We compute per-chunk scroll via
horizontal cross-correlation of consecutive frames, accumulate it = true x-progress, then find the 2-byte WRAM
pair most correlated with it. Validated across multiple demos. Writes smbas_progress.json (<u2).
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import stable_retro as retro

_R = "/home/t-nagupta/NitroGen-With-VLM-Planning"
ROM = os.path.join(_R, "Game data/Super Mario All-Stars.sfc")
NAMES = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


def mb(s, buttons):
    m = np.zeros(len(buttons), np.uint8)
    for i, on in enumerate(s):
        if on and NAMES[i] in buttons:
            m[buttons.index(NAMES[i])] = 1
    return m


def hscroll(a, b):
    """estimate horizontal shift b->a by best column cross-correlation of mean-row (range +-8)."""
    ca = a.mean(axis=(0, 2)); cb = b.mean(axis=(0, 2))
    ca = ca - ca.mean(); cb = cb - cb.mean()
    best, bv = 0, -1e9
    for s in range(-8, 9):
        v = np.dot(ca, np.roll(cb, s))
        if v > bv:
            bv, best = v, s
    return best  # >0 means content moved right->camera right (progress)


def wram(gd):
    return np.frombuffer(bytes(gd.memory.blocks[0x7e0000]), np.uint8)


def scan(emu, gd, buttons, acts, obs, every=20, maxf=2400):
    H, prog, pframe = [], [], None
    cum = 0
    for i in range(min(len(acts), maxf)):
        emu.set_button_mask(mb(acts[i], buttons), 0); emu.step(); gd.update_ram()
        if i % every == 0:
            f = obs[i]
            if pframe is not None:
                cum += hscroll(f, pframe)
            pframe = f
            H.append(wram(gd).copy()); prog.append(cum)
    return np.stack(H).astype(np.float64), np.array(prog, float)


def main():
    emu = retro.RetroEmulator(ROM); buttons = list(retro.get_system_info("Snes")["buttons"])
    gd = retro.data.GameData(); idir = os.path.join(_R, "tmp/retro_data/SuperMarioAllStars-Snes")
    os.makedirs(idir, exist_ok=True); json.dump({"info": {}}, open(idir + "/data.json", "w"))
    gd.load(idir + "/data.json", None); emu.configure_data(gd)

    dirs = sorted(glob.glob(os.path.join(_R, "docs/demos/demos/SuperMarioAllStars-Snes/*")))
    cand_scores = {}
    for di in (7, 9, 12):
        z = np.load(os.path.join(dirs[di], "demo.npz")); acts, obs = z["actions"], z["observations"]
        raw = open(os.path.join(dirs[di], "initial.state"), "rb").read()
        st = __import__("gzip").decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        emu.set_state(st); gd.update_ram()
        H, prog = scan(emu, gd, buttons, acts, obs)
        if prog.std() < 1:
            continue
        for a in range(len(H[0]) - 1):
            v = 256 * H[:, a + 1] + H[:, a]
            if v.std() < 1:
                continue
            c = abs(np.corrcoef(prog, v)[0, 1])
            cand_scores.setdefault(a, []).append(c)
        print(f"demo {os.path.basename(dirs[di])}: scrolled cum {prog[-1]:.0f}", flush=True)
    # rank addresses by min corr across demos (must be robust)
    ranked = sorted(((min(cs), np.mean(cs), a) for a, cs in cand_scores.items() if len(cs) >= 3), reverse=True)
    print("top 2-byte addrs by min-corr-with-scroll across demos:")
    for mn, mean, a in ranked[:12]:
        print(f"  0x7E{a:04X}: min {mn:.3f} mean {mean:.3f}")
    if ranked and ranked[0][0] > 0.6:
        a = ranked[0][2]; abs_addr = 0x7e0000 + a
        json.dump({"prog_addr": abs_addr, "prog_type": "<u2"},
                  open(os.path.join(_R, "tmp/retro_data/smbas_progress.json"), "w"), indent=2)
        print(f"\nWROTE prog_addr=0x{abs_addr:X} <u2 (corr-validated) => SMB1 EVAL-ready")
    else:
        print("\nno robust addr (min-corr<=0.6); keep train-only.")


if __name__ == "__main__":
    main()
