"""EXP-051 decisive eval: does ANY variant make the frozen DiT OBEY a counterfactual plan?

The EXP-050 CFG sweep showed the mm/con students never move left against a rightward-prior frame
at ANY CFG weight. Here we sweep CFG for each checkpoint across several real game frames, with a
plan that CONTRADICTS the frame's likely action (e.g. 'move left' on a right-drifting frame), and
measure the left-stick x. If a variant's stick_x goes NEGATIVE (left) as w rises, the plan is now
OVERRIDING the frame -> the authority ceiling is broken.

For each frame we test opposing plans (left vs right) and report the SEPARATION
(stick_x[right] - stick_x[left]); a large positive separation = the plan controls direction.

Run: PYTHONPATH=. python planner_poc/eval_cfg_override.py
"""
import os, sys
import numpy as np
from PIL import Image
import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
from eval_policy import NitroGenPolicy

JLX, JLY, I_SOUTH = 21, 22, 18
CKPTS = {
    "mm":         "runs/stage2_student_mm/plan_stage1_2500.pt",
    "mm_con":     "runs/stage2_student_mm_con/plan_stage1_2500.pt",
    "mm_lora":    "runs/stage2_student_mm_lora/plan_stage1_2500.pt",
    "mm_lora_cf": "runs/stage2_student_mm_lora_cf/plan_stage1_2500.pt",
}
# real Cave Story frames the model saw in play (varied positions).
FRAMES = {
    "corner":  "/tmp/run_mm/f024.png",     # stuck right corner (rightward prior)
    "start":   "/tmp/run_mm/f000.png",     # intro room center
    "ledge":   "/tmp/run_text/f012.png",   # on the ledge mid-room
}
WS = [1, 2, 4, 8, 16]
PLAN_L = "move left"
PLAN_R = "move right"


def stick_x(pol, frame, plan, w, mm):
    ch = pol._sample_chunk(frame, plan, float(w), plan_frames=[frame] if mm else None)
    return float(ch[:, JLX].mean())


def main():
    results = {}
    for name, ck in CKPTS.items():
        if not os.path.exists(ck):
            print(f"[skip] {name}: no ckpt"); continue
        mm = True  # all are frame-conditioned students
        pol = NitroGenPolicy(ck, default_cfg=1.0); pol.mm_mode = mm
        print(f"\n===== {name} =====")
        print(f"{'frame':8} {'w':>3} {'L:stick_x':>10} {'R:stick_x':>10} {'sep(R-L)':>9}")
        for fname, fpath in FRAMES.items():
            if not os.path.exists(fpath):
                continue
            fr = np.asarray(Image.open(fpath).convert("RGB"))
            for w in WS:
                xl = stick_x(pol, fr, PLAN_L, w, mm)
                xr = stick_x(pol, fr, PLAN_R, w, mm)
                sep = xr - xl
                flag = "  <-- separates!" if sep > 0.15 else ("  L-obey" if xl < -0.1 else "")
                print(f"{fname:8} {w:>3} {xl:>10.2f} {xr:>10.2f} {sep:>9.2f}{flag}")
                results.setdefault(name, []).append(sep)
        del pol
        import torch; torch.cuda.empty_cache()
    print("\n===== SUMMARY: mean direction-separation (R-L stick_x); >0 = plan controls dir =====")
    for name, seps in results.items():
        s = np.array(seps)
        print(f"  {name:12} mean_sep={s.mean():+.3f}  max_sep={s.max():+.3f}  "
              f"frac_separating={np.mean(s>0.15):.2f}")


if __name__ == "__main__":
    main()
