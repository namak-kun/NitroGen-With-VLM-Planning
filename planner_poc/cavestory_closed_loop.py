"""Closed-loop System-2 demo: the planner is RE-INVOKED every `replan` chunks. A capable VLM
(Qwen3.5-9B) re-reads the recent frames, judges whether Quote is stuck/looping, and issues a
fresh corrective plan (e.g. "jump left to get off the wall"). The frozen 0.8B mm-encoder + DiT
(System-1) then execute it. This fixes the failure mode where a FIXED plan let Quote corner
itself with no recovery.

Produces an annotated MP4 where the (updating) plan is drawn on every frame, and a log of every
plan change + per-action detail.

Run: PYTHONPATH=. python planner_poc/cavestory_closed_loop.py [ckpt] [objective] [cfg] [nchunks] [replan]
Out: docs/cavestory_play/closed_loop.mp4 + actions_closed_loop.txt
"""
import os, sys, subprocess, re
import numpy as np
import torch
from PIL import Image
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
import os; sys.path.insert(0, os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "planner_poc"))

from transformers import AutoProcessor, AutoModelForImageTextToText
from nitrogen.eval import Scenario, JLX, JLY
from nitrogen.eval.envs.cavestory import CaveStoryEnv, MENU_OK, SKIP
from nitrogen.eval.envs.virtual_gamepad import MENU_BUTTONS, _NAME2IDX, B_NORTH
from eval_policy import NitroGenPolicy
from cavestory_play_annotated import annotate, _RPT  # reuse the frame annotator

BOOT = ([("wait", 1.2), ("btn", MENU_OK)] * 4 + [("wait", 4.0)]
        + [("hold", SKIP, 4.0), ("btn", MENU_OK)] * 4 + [("wait", 1.0)])
GAMEPLAY_MASK = tuple(MENU_BUTTONS) + (B_NORTH,)
OUT = "/tmp/cl_frames"
PLANNER_VLM = os.environ.get("PLANNER_VLM", "Qwen/Qwen3.5-9B")

SYS = ("You guide Quote (the small character) in the 2D platformer Cave Story. You see two recent "
       "frames in time order. Judge whether Quote is making progress or is STUCK/looping against a "
       "wall or in a corner. The goal is to explore and reach the exit door. Output ONLY one short "
       "imperative plan (max 10 words), no analysis or numbering. If stuck, give a CORRECTIVE plan "
       "that moves AWAY from the wall Quote is jammed against. Example: 'move left and jump onto the ledge'.")


class System2:
    """Capable VLM planner, invoked closed-loop from recent frames."""
    def __init__(self, model_id=PLANNER_VLM, device="cuda"):
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.m = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device).eval()
        self.device = device

    @torch.no_grad()
    def plan(self, frames, prev_plan=None):
        imgs = [f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f)).convert("RGB")
                for f in frames]
        q = "Quote's recent frames. Give one short plan."
        if prev_plan:
            q = f"Previous plan was '{prev_plan}'. {q} Revise it if Quote is stuck."
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": q}]
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inp = self.proc(text=[text], images=imgs, return_tensors="pt").to(self.device)
        out = self.m.generate(**inp, max_new_tokens=40, do_sample=False)
        ans = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()
        ans = ans.splitlines()[0].strip().strip('"').strip() if ans else ""
        return ans


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student_mm/plan_stage1_2500.pt"
    objective = sys.argv[2] if len(sys.argv) > 2 else "explore and reach the exit door"
    cfg = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    nchunks = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    replan = int(sys.argv[5]) if len(sys.argv) > 5 else 2          # re-invoke planner every N chunks
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    print(f"closed-loop | ckpt={ckpt} cfg={cfg} nchunks={nchunks} replan_every={replan}")
    pol = NitroGenPolicy(ckpt, default_cfg=cfg); pol.mm_mode = True
    s2 = System2()
    env = CaveStoryEnv(boot_wait=11.0, freeze_during_inference=True)
    fi = 0
    log = [f"objective: {objective}", f"planner: {PLANNER_VLM}  replan_every: {replan}", ""]
    try:
        sc = Scenario("cl", plan=objective, objective=objective,
                      success_spec={"reset_macro": BOOT}, max_steps=nchunks, cfg_scale=cfg)
        obs = env.reset(sc)
        cur_frame = obs.frame
        hist = [cur_frame]              # chunk-boundary frames for the System-2 planner
        plan = s2.plan([cur_frame, cur_frame])      # initial plan from the first frame
        log.append(f"[chunk 0] INITIAL PLAN: {plan!r}")
        print(f" initial plan: {plan!r}")
        for c in range(nchunks):
            # System-2 cadence: re-invoke the planner from the last 2 boundary frames.
            if c > 0 and c % replan == 0:
                prev = plan
                plan = s2.plan(hist[-2:], prev_plan=prev) or prev
                changed = plan != prev
                log.append(f"[chunk {c}] REPLAN ({'CHANGED' if changed else 'same'}): {plan!r}")
                print(f"  chunk {c}: REPLAN -> {plan!r}  {'(changed)' if changed else ''}")
            chunk = pol._sample_chunk(cur_frame, plan, cfg, plan_frames=[cur_frame])
            for b in GAMEPLAY_MASK:
                chunk[:, b] = 0.0
            rows = env.apply_chunk_capture(chunk)
            st = env._read_state(); pos = (st.get("tile_x"), st.get("tile_y"))
            for ri, (row, frame) in enumerate(rows):
                img = annotate(frame, row, c, ri, plan, "CLOSED-LOOP", pos)
                Image.fromarray(img).save(f"{OUT}/f{fi:04d}.png"); fi += 1
            cur_frame = rows[-1][1]; hist.append(cur_frame)
            print(f"  chunk {c:2d}: Quote@({pos[0]},{pos[1]})  plan={plan!r}", flush=True)
    finally:
        env.close()
    outdir = os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "docs/cavestory_play")
    outmp4 = f"{outdir}/closed_loop.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "quiet", "-y", "-framerate", "6",
                    "-i", f"{OUT}/f%04d.png", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", outmp4])
    with open(f"{outdir}/actions_closed_loop.txt", "w") as fh:
        fh.write("\n".join(log) + "\n")
    print(f"\nsaved {fi} frames -> {outmp4}")
    print(f" plan-change log -> {outdir}/actions_closed_loop.txt")


if __name__ == "__main__":
    main()
