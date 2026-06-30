"""Offline shoot-button test: the model is Markov (single frame), so feed it Cave Story frames
that CONTAIN enemies / a boss and read which face button it presses. Answers: does NitroGen
shoot with WEST(X) (the CS+ default) or EAST(B)? Compares to a no-enemy frame (the intro room)
to see which button SPIKES when enemies are present.

Run: PYTHONPATH=. python planner_poc/shoot_button_test.py [ckpt] [cfg]
"""
import os, sys
import numpy as np
from PIL import Image
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
import os; sys.path.insert(0, os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "planner_poc"))
from nitrogen.shared import BUTTON_ACTION_TOKENS
from eval_policy import NitroGenPolicy

_I = {t.lower(): i for i, t in enumerate(BUTTON_ACTION_TOKENS)}
FACE = ["south", "east", "west", "north"]          # A, B, X, Y
NAMES = {"south": "A(south)", "east": "B(east)", "west": "X(west)", "north": "Y(north)"}

# enemy/boss frames (doukutsu-rendered) vs a no-enemy control (our intro room capture)
FRAMES = {
    "boss_fight (drs_7)": "/tmp/enemy_frames/drs_7.png",
    "enemy_near (drs_9)": "/tmp/enemy_frames/drs_9.png",
    "boss (drs_6)": "/tmp/enemy_frames/drs_6.png",
    "NO_enemy (intro)": "/tmp/nitrogen_run/f000.png",
}
PLANS = ["shoot the enemy", "attack and defeat the enemies", ""]  # last = null/unconditioned


def rates(chunk):
    """fraction of the 18 action rows pressing each face button."""
    return {n: float((chunk[:, _I[n]] > 0.5).mean()) for n in FACE}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student/plan_stage1_2500.pt"
    cfg = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    pol = NitroGenPolicy(ckpt, default_cfg=cfg)
    print(f"shoot-button test | ckpt={ckpt} cfg={cfg}\n")
    print(f"{'frame':22} {'plan':28} " + " ".join(f"{NAMES[n]:>9}" for n in FACE))
    agg = {n: [] for n in FACE}
    enemy_agg = {n: [] for n in FACE}
    noenemy_agg = {n: [] for n in FACE}
    for label, path in FRAMES.items():
        if not os.path.exists(path):
            print(f"{label:22} MISSING {path}"); continue
        fr = np.asarray(Image.open(path).convert("RGB"))
        for plan in PLANS:
            ch = pol._sample_chunk(fr, plan, cfg)
            r = rates(ch)
            ptxt = plan if plan else "(null)"
            print(f"{label:22} {ptxt:28} " + " ".join(f"{r[n]:>9.2f}" for n in FACE))
            for n in FACE:
                agg[n].append(r[n])
                (noenemy_agg if "NO_enemy" in label else enemy_agg)[n].append(r[n])
        print()
    print("=== ENEMY frames vs NO-ENEMY (mean face-button rate) ===")
    print(f"{'button':10} {'enemy':>8} {'no-enemy':>9} {'delta':>8}")
    for n in FACE:
        e = float(np.mean(enemy_agg[n])) if enemy_agg[n] else 0.0
        ne = float(np.mean(noenemy_agg[n])) if noenemy_agg[n] else 0.0
        print(f"{NAMES[n]:10} {e:>8.2f} {ne:>9.2f} {e-ne:>+8.2f}")
    print("\n=> the button that SPIKES on enemy frames (high delta) is the model's SHOOT button.")


if __name__ == "__main__":
    main()
