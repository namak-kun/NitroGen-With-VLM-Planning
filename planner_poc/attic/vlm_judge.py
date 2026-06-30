"""A concrete VLM judge_fn for nitrogen.eval.VLMJudgeDetector — the most GENERAL way to define a
goal (works on any game, including static-background platformers where optical-flow steering fails).

It reuses the planner's Qwen3.5 VL backbone (already loaded for the policy) to answer a yes/no
question about the recent frames. CAVEATS (verified 2026-06-23): (1) circularity — judging with the
SAME VLM that plans shares blind spots; (2) the planner's Qwen3.5-2B is a WEAK judge (failed a trivial
brighten-detection smoke test). For a RELIABLE metric, pass a proper INSTRUCTION-TUNED VLM
(e.g. Qwen2.5-VL-Instruct / Gemma-3-VL) as `plan_encoder` here instead. The wiring is correct; only
the backing model quality limits it. For a POC, reusing the loaded backbone is cheap but soft.

Usage:
    from planner_poc.vlm_judge import make_vlm_judge
    judge = make_vlm_judge(policy.pl)              # policy.pl is the PlanEncoder
    det = VLMJudgeDetector(judge)                  # plug into the harness
    # success_spec={'judge_instruction': 'the character has moved to the right side of the screen'}
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

_YES = re.compile(r"\b(yes|yeah|yep|true|correct|done|reached|complete)\b", re.I)
_NO = re.compile(r"\b(no|not|nope|false|hasn'?t|didn'?t|fail)\b", re.I)


def make_vlm_judge(plan_encoder, device: str = "cuda", max_new_tokens: int = 6) -> Callable[[list, str], bool]:
    """Return judge_fn(frames, instruction)->bool. Asks the VL backbone a yes/no completion question
    over the recent frames. The instruction should read as a state to verify, e.g.
    'the kart is turning left' / 'the character reached the door' / 'the player moved right'."""

    def judge_fn(frames: list, instruction: str) -> bool:
        if not frames:
            return False
        # show the judge a few spread-out frames (oldest->newest) for temporal context
        fr = frames if len(frames) <= 4 else [frames[0], frames[len(frames) // 2], frames[-1]]
        q = (f"Looking at these game frames in order, answer with a single word YES or NO: "
             f"is the following TRUE? \"{instruction}\"")
        try:
            ans = plan_encoder.generate_plan(
                [np.asarray(f) for f in fr], device, instruction=q,
                system="You are a precise game-state judge. Answer only YES or NO.",
                max_new_tokens=max_new_tokens)
        except Exception:
            return False
        ans = (ans or "").strip().lower()
        if _YES.search(ans) and not _NO.search(ans):
            return True
        if _NO.search(ans):
            return False
        return ans.startswith("y")

    return judge_fn


if __name__ == "__main__":
    # smoke test: load the planner backbone + judge a trivial pair of frames.
    import os
    import sys
    import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))
    from nitrogen.planner import PlanEncoder, PlannerConfig
    qwen = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=qwen)); pl.load()
    judge = make_vlm_judge(pl)
    # a black frame and a white frame; ask an obviously-true and obviously-false question
    black = np.zeros((120, 160, 3), np.uint8)
    white = np.full((120, 160, 3), 255, np.uint8)
    print("judge(black->white, 'the screen got brighter'):", judge([black, white], "the screen got brighter"))
    print("judge(white->black, 'the screen got brighter'):", judge([white, black], "the screen got brighter"))
