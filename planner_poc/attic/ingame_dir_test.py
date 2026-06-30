"""Clean in-game override measurement: from First Cave start, drive a CONSTANT plan for N chunks
and measure net dx (tiles). Compares null / 'go left' / 'go right' for the chosen variant+CFG, to
see whether the plan can move Quote LEFT against the rightward prior in-game (the POC crux).

Run: PYTHONPATH=. python planner_poc/ingame_dir_test.py [ckpt] [cfg] [nchunks]  (MM=1 TXT=1 env)
"""
import os, sys
import numpy as np
import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
from nitrogen.eval import Scenario
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, B_NORTH
from eval_policy import NitroGenPolicy

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
GAMEPLAY_MASK = tuple(MENU_BUTTONS) + (B_NORTH,)


def run(pol, plan, cfg, nchunks, mm, null=False):
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    try:
        sc = Scenario("dir", plan=plan, objective=plan, success_spec={"reset_macro": BOOT},
                      max_steps=nchunks, cfg_scale=cfg)
        obs = env.reset(sc)
        x0 = obs.state.get("tile_x"); xs = [x0]
        cur = obs.frame
        for _ in range(nchunks):
            ch = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur] if mm else None, null=null)
            for b in GAMEPLAY_MASK:
                ch[:, b] = 0.0
            obs = env.step(ch); cur = obs.frame
            xs.append(obs.state.get("tile_x"))
        xs = [x for x in xs if x is not None]
        return x0, xs[-1], (xs[-1] - x0)
    finally:
        env.close()
        import torch; torch.cuda.empty_cache()


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student_mm_txt_lora_cf/plan_stage1_2500.pt"
    cfg = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    nchunks = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    mm = os.environ.get("MM", "1") == "1"; txt = os.environ.get("TXT", "1") == "1"
    qwen = os.environ.get("QWEN", None)
    pol = NitroGenPolicy(ckpt, default_cfg=cfg, qwen=qwen); pol.mm_mode = mm; pol.mm_text_only = txt
    print(f"in-game dir test | {ckpt.split('/')[1]} cfg={cfg} mm={mm} txt={txt} nchunks={nchunks}\n")
    print(f"{'plan':12} {'x0':>7} {'xT':>7} {'net dx':>8}  (dx<0 = LEFT, overriding the right prior)")
    for plan, kw in [("(null)", dict(null=True)), ("go left", {}), ("go right", {})]:
        x0, xt, dx = run(pol, plan if plan != "(null)" else "go left", cfg, nchunks, mm, **kw)
        tag = "  <-- LEFT!" if dx < -1 else ""
        print(f"{plan:12} {x0:>7.1f} {xt:>7.1f} {dx:>+8.1f}{tag}", flush=True)


if __name__ == "__main__":
    main()
