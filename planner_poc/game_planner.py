"""game_planner.py -- per-game CORRECT planner prompts/controls + a closed-loop planner with a two-part
<learnings>/<plan> output and carried-forward context (experiential learning across a game).

WHY: the old closed-loop used SYS[_syskey] + a global INSTR that (a) miswired Fire Emblem to a
"Mario-style platformer" prompt and (b) hardcoded "jump" into EVERY game's instruction -> Minish/FE were
told to JUMP (an action they don't have) and FE was told to reach platforms. This module gives each game
its true genre, control vocabulary, and objective.

OUTPUT FORMAT (the user's context-handling idea): the planner emits
    <learnings>
    - durable facts about THIS game/level discovered so far (mechanics, hazards, what works)
    </learnings>
    <plan>
    one short imperative next action, grounded in the listed controls
    </plan>
The ClosedLoopPlanner carries the LEARNINGS forward into the next prompt (accumulating game knowledge),
so the planner builds a running understanding instead of re-deriving from 4 pixels each time.

Ablation modes: 'base' (null DiT, no planner), 'plan' (per-game prompt, plan only, no memory),
'learn' (per-game prompt + carried <learnings>).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

# ---- per-game planner config: true genre + correct controls + objective. -----------------------------
GAMES = {
    "smw": dict(
        title="Super Mario World (SNES), a side-scrolling platformer.",
        controls="Controls: move LEFT or RIGHT, hold RUN to dash, JUMP (to clear gaps/enemies and reach "
                 "higher ledges), crouch DOWN. Moving sideways alone will NOT climb — you must JUMP.",
        objective="Goal: advance RIGHT through the level and reach the exit; jump over pits and stomp or "
                  "avoid enemies.",
    ),
    "sonic": dict(
        title="Sonic the Hedgehog 2 (Genesis), a FAST side-scrolling platformer.",
        controls="Controls: move LEFT or RIGHT (build momentum), JUMP (one button) to clear gaps/enemies, "
                 "hold DOWN then JUMP to SPIN-DASH for a speed burst. Rolling down slopes builds speed.",
        objective="Goal: advance RIGHT to the end of the act, keeping momentum. RINGS are OPTIONAL "
                  "collectibles (they only protect from one hit) — do NOT chase rings or treat them as the "
                  "objective; just keep moving RIGHT.",
    ),
    "minish": dict(
        title="The Legend of Zelda: The Minish Cap (GBA), a TOP-DOWN action-adventure.",
        controls="Controls (TOP-DOWN, absolute screen directions): move UP, DOWN, LEFT, or RIGHT; ATTACK "
                 "with the sword; USE the equipped item. THERE IS NO JUMP — never plan to jump.",
        objective="Goal: explore the area and move toward doors, exits, or new screens; attack enemies or "
                  "bushes blocking the way. Use UP/DOWN/LEFT/RIGHT, never 'forward'.",
    ),
    "fireemblem": dict(
        title="Fire Emblem: The Sacred Stones (GBA), a TURN-BASED TACTICS game on a grid.",
        controls="Controls: move the CURSOR UP/DOWN/LEFT/RIGHT over grid tiles; SELECT/CONFIRM to pick a "
                 "unit, tile, or menu option; CANCEL to go back; open the MENU to act. THERE IS NO JUMPING "
                 "and NO platforming — this is a menu-and-grid strategy game.",
        objective="Goal: select your units and move them toward enemies, confirm attacks, and navigate "
                  "menus (Attack/Wait/Item). Think in terms of cursor moves and unit actions, NOT movement "
                  "through a level.",
    ),
}

PLAN_ONLY_INSTR = ("In one short sentence, tell the controller what to do NEXT, grounded ONLY in the listed "
                   "controls for this game. Output just the plan sentence.")

# The LEARN instruction now frames the VLM as System 2 reviewing System 1's execution of its OWN prior plan.
# Behavioural learning is only possible when a prior plan + the play that followed it are in context (the
# user's point): without them the model can only restate controls. So learnings are grounded on the
# plan->execution->outcome gap and on anything SURPRISING about the controller (DiT) or the game.
LEARN_INSTR = (
    "Now update your LEARNINGS and issue the next plan. For LEARNINGS, record ONLY durable, SURPRISING, or "
    "useful discoveries — especially about how the CONTROLLER (System 1) behaves: did it actually follow "
    "your previous plan? did an input do something unexpected, overshoot, stall, or get stuck/loop? plus "
    "durable game facts (hazards, what advances vs what doesn't). Do NOT restate the controls and do NOT "
    "narrate the momentary scene. Output EXACTLY this format and nothing else:\n"
    "<learnings>\n- (<=6 concise bullets; update/extend the prior learnings; behaviour + durable facts "
    "only)\n</learnings>\n"
    "<plan>\n(one short imperative next action for the controller, grounded ONLY in the listed "
    "controls)\n</plan>"
)

# Per-game verbs to describe the action the CONTROLLER ACTUALLY APPLIED (env.action_row_to_buttons), so the
# trace names buttons by their game meaning instead of the raw Xbox-platformer vocabulary (which makes the
# VLM hallucinate moves like a Sonic "spin-dash" or a Minish "jump" that the game/env never produced).
_DIRSET = ("LEFT", "RIGHT", "UP", "DOWN")
ACT_VERBS = {
    "smw":        {"_dir": "move {d}", "A": "JUMP", "B": "JUMP", "Y": "run", "X": "spin"},
    "sonic":      {"_dir": "move {d}", "A": "JUMP", "B": "JUMP", "C": "JUMP"},
    "minish":     {"_dir": "move {d}", "A": "use item", "B": "sword attack"},
    "fireemblem": {"_dir": "move the cursor {d}", "A": "confirm/select", "B": "cancel/back"},
}


def applied_action_desc(env, game: str, row) -> str:
    """Describe ONE action row as the env actually applies it (directions + face buttons), in this game's
    verbs. Uses env.action_row_to_buttons so the trace reflects what the game received, not raw DiT dims."""
    toks = list(env.action_row_to_buttons(row))
    vmap = ACT_VERBS.get(game, {})
    dirs = [t for t in toks if t in _DIRSET]
    acts = [t for t in toks if t not in _DIRSET]
    parts = []
    if dirs:
        parts.append(vmap.get("_dir", "move {d}").format(d="+".join(d.lower() for d in dirs)))
    for a in acts:
        parts.append(vmap.get(a, a))
    return ", ".join(parts) if parts else "no input (idle)"


def system_prompt(game: str, mode: str) -> str:
    g = GAMES[game]
    base = (
        f"You are System 2, the high-level PLANNER for an agent playing {g['title']} You do NOT press "
        f"buttons yourself: a separate fast controller (System 1) reads your short plan and turns it into "
        f"the actual inputs. You are invoked periodically. You CANNOT read the controller's inputs from a "
        f"still frame — when shown the recent play, the inputs the controller took are written out for you. "
        f"{g['controls']} {g['objective']}")
    if mode == "learn":
        return base + (" Each time, you see your PREVIOUS plan, then what the controller actually did and "
                       "the frames that resulted, then the current situation. Judge whether the controller "
                       "followed your plan and whether it worked, then plan the next move.\n" + LEARN_INSTR)
    return base + " You are shown the most recent frames (oldest first).\n" + PLAN_ONLY_INSTR


def parse_learn_plan(text: str) -> tuple[str, str]:
    """Extract (learnings, plan) from the tagged output; robust to truncation / missing tags.

    Layered fallbacks (per the user) so a malformed generation never yields garbage:
      LEARNINGS:  (1) closed <learnings>..</learnings>; (2) UNCLOSED <learnings>.. up to <plan>/end;
                  (3) everything after </think> and before <plan> (when no learnings tag survived).
      PLAN:       (1) closed <plan>..</plan>; (2) UNCLOSED <plan>..end; (3) everything after </learnings>.
    Then PLAN is cleaned to the first non-tag, non-bullet line so it never becomes a literal '<learnings>'
    (which previously fed back and corrupted the next prompt). THINK blocks are stripped first."""
    # strip a think block if present (the chat template emits one; in think mode it may be filled).
    after_think = text.split("</think>", 1)[1] if "</think>" in text else text

    # ---------- LEARNINGS ----------
    learnings = ""
    m = re.search(r"<learnings>(.*?)</learnings>", text, re.S | re.I)            # (1) closed
    if m and m.group(1).strip():
        learnings = m.group(1)
    else:
        m = re.search(r"<learnings>(.*?)(?=<plan>|$)", text, re.S | re.I)         # (2) unclosed -> <plan>/end
        if m and m.group(1).strip():
            learnings = m.group(1)
        else:                                                                    # (3) after </think>, before <plan>
            learnings = re.split(r"<plan>", after_think, flags=re.I)[0]

    # ---------- PLAN ----------
    cand = ""
    pm = re.search(r"<plan>(.*?)</plan>", text, re.S | re.I)                      # (1) closed
    if pm and pm.group(1).strip():
        cand = pm.group(1)
    else:
        po = re.search(r"<plan>(.*)$", text, re.S | re.I)                        # (2) unclosed
        if po and po.group(1).strip():
            cand = po.group(1)
        elif re.search(r"</learnings>", text, re.I):                             # (3) after </learnings>
            cand = re.split(r"</learnings>", text, flags=re.I)[-1]
        else:
            cand = after_think
    plan = ""
    for ln in cand.splitlines():
        s = ln.strip().strip('"').strip()
        if not s or s.startswith("<") or s.startswith("-") or s.startswith("•"):
            continue                                                             # skip tags + bullet lines
        plan = s
        break
    learnings = re.sub(r"</?(learnings|plan|think)>", "", learnings, flags=re.I).strip()
    return learnings, plan


class ClosedLoopPlanner:
    """Closed-loop game planner. mode: 'plan' (stateless, single-shot) or 'learn' (carry <learnings> AND
    review the controller's execution of the PREVIOUS plan). In learn mode the caller should feed the play
    that happened since the last plan via observe(action_desc, frame) -- those (what System 1 did, what
    resulted) pairs plus the prior plan are what make BEHAVIOURAL learning possible (without a prior plan to
    check against, the model can only restate controls)."""

    def __init__(self, pol, game: str, mode: str = "learn", device: str = "cuda",
                 max_new_tokens: int = 220, trace_cap: int = 4):
        self.pol = pol; self.game = game; self.mode = mode; self.device = device
        self.max_new_tokens = max_new_tokens
        self.trace_cap = trace_cap
        self.learnings = ""        # carried-forward game knowledge
        self.last_plan = ""
        self.trace = []            # [(action_desc, frame)] executed SINCE the last plan

    def reset(self):
        self.learnings = ""; self.last_plan = ""; self.trace = []

    def observe(self, action_desc: str, frame) -> None:
        """Record one executed step (what the controller did + the resulting frame) since the last plan."""
        self.trace.append((action_desc, frame))
        if len(self.trace) > self.trace_cap:
            self.trace = self.trace[-self.trace_cap:]

    @staticmethod
    def _pil(f):
        from PIL import Image
        return f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f)).convert("RGB")

    def plan(self, frames) -> str:
        import torch
        pl = self.pol.pl
        pl.load()
        if next(pl.backbone.parameters()).device != torch.device(self.device):
            pl.backbone.to(self.device)
        # The FIRST plan has no prior plan/trace to assess, so behavioural learnings would be fabricated
        # (the user's point). Treat the first call as plan-only and start learnings only once there is play
        # to review.
        first_step = not (self.last_plan and self.trace)
        learn = (self.mode == "learn") and not first_step
        sys_p = system_prompt(self.game, "learn" if learn else "plan")
        content, imgs = [], []

        def add_img(f):
            content.append({"type": "image"}); imgs.append(self._pil(f))

        if learn:
            # 1) the prior plan + the play that followed it (so the model can assess adherence/efficacy).
            content.append({"type": "text", "text": f'Your PREVIOUS plan was: "{self.last_plan}".'})
            content.append({"type": "text", "text": "What the controller (System 1) then actually did, and "
                                                    "the frame that resulted after each step:"})
            for desc, fr in self.trace:
                content.append({"type": "text", "text": f"- controller input: {desc}; result:"})
                add_img(fr)
        # 2) the current situation.
        content.append({"type": "text", "text": "Current situation now (most recent frame last):"})
        for f in frames:
            add_img(f)
        # 3) the prior learnings + the output instruction.
        if learn:
            prior = self.learnings or "(nothing recorded yet)"
            content.append({"type": "text", "text": f"Your LEARNINGS so far:\n{prior}\n\n{LEARN_INSTR}"})
        else:
            content.append({"type": "text", "text": PLAN_ONLY_INSTR})

        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": content}]
        text = pl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = pl.processor(text=[text], images=imgs, return_tensors="pt").to(self.device)
        budget = (self.max_new_tokens + 100) if learn else 64   # learn needs room for learnings THEN plan
        with torch.no_grad():
            out = pl.backbone.generate(**inp, max_new_tokens=budget, do_sample=False)
        gen = pl.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        if learn:
            learn_txt, plan = parse_learn_plan(gen)
            if learn_txt:
                self.learnings = learn_txt      # carry forward
        else:
            plan = gen.strip().strip('"').split("\n")[0].strip()
        # Guard: never store an empty or tag-like plan (it would feed back and corrupt the next prompt).
        if plan and not plan.lstrip().startswith("<"):
            self.last_plan = plan
        elif not self.last_plan:
            self.last_plan = "move right"
        self.trace = []                          # consumed; next window starts fresh
        return self.last_plan
