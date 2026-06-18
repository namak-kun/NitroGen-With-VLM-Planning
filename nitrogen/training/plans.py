"""Synthetic plans for Stage-1 alignment (schedule-based).

A plan's *language* specifies the temporal structure of the action chunk, and we
build a full-chunk counterfactual target to match it:
  * HOLD ("keep going left")        -> direction held for the whole chunk
  * TAP  ("tap left, then keep playing") -> brief direction, then reactive/real
  * SEQ  ("go left then right")     -> two (or three) segments in sequence
  * BUTTON ("jump")                -> brief button press, then reactive/real
  * IDLE ("do nothing")            -> zeros

Each plan is a list of `Segment`s over the normalized chunk [0,1]. Segment kinds:
  - "dir":    drive a direction (stick and/or d-pad, per detected modality)
  - "button": press a named button
  - "free":   fill with the streamer's REAL action (resume reactive policy)
  - "idle":   zeros

The contrastive training signal: null plan -> reproduce the streamer's REAL chunk;
synthetic plan -> reproduce its scheduled counterfactual chunk. Action layout is
NitroGen's true [buttons(21) @0..20, j_left @21..22, j_right @23..24] (actions.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .actions import BUTTON_ORDER, NUM_BUTTONS

_BTN = {name: i for i, name in enumerate(BUTTON_ORDER)}


def _zeros_chunk(H: int):
    return {
        "buttons": np.zeros((H, NUM_BUTTONS), dtype=np.float32),
        "j_left": np.zeros((H, 2), dtype=np.float32),
        "j_right": np.zeros((H, 2), dtype=np.float32),
    }


# ---- direction helpers (modality-aware: stick and/or d-pad) ----
_DPAD = {"left": "dpad_left", "right": "dpad_right", "up": "dpad_up", "down": "dpad_down"}
_DIRS = {
    "left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1),
    "up_left": (-1, -1), "up_right": (1, -1), "down_left": (-1, 1), "down_right": (1, 1),
}
CARDINALS = ["left", "right", "up", "down"]


def _dpad_names(dx, dy):
    out = []
    if dx < 0: out.append(_DPAD["left"])
    if dx > 0: out.append(_DPAD["right"])
    if dy < 0: out.append(_DPAD["up"])
    if dy > 0: out.append(_DPAD["down"])
    return out


def _set_dir(chunk, lo, hi, dx, dy, modality, mag=1.0):
    if modality in ("stick", "both", "none"):
        chunk["j_left"][lo:hi, 0] = dx * mag
        chunk["j_left"][lo:hi, 1] = dy * mag
    if modality in ("dpad", "both"):
        for name in _dpad_names(dx, dy):
            chunk["buttons"][lo:hi, _BTN[name]] = 1.0


@dataclass
class Segment:
    start: float            # fraction in [0,1)
    end: float              # fraction in (0,1]
    kind: str               # "dir" | "button" | "free" | "idle"
    payload: object = None  # (dx,dy) for dir; button-name str for button


@dataclass
class SyntheticPlan:
    name: str
    phrasings: list[str]
    schedule: list[Segment]
    grounded: bool = True  # True = direction/idle (documented); False = button


def build_target(plan: SyntheticPlan, H: int, real_chunk: dict | None,
                 modality: str = "stick") -> dict:
    """Materialize a plan's schedule into an (H, ...) counterfactual chunk."""
    c = _zeros_chunk(H)
    for seg in plan.schedule:
        lo, hi = int(round(seg.start * H)), int(round(seg.end * H))
        lo, hi = max(0, lo), min(H, hi)
        if hi <= lo:
            continue
        if seg.kind == "dir":
            dx, dy = seg.payload
            _set_dir(c, lo, hi, dx, dy, modality)
        elif seg.kind == "button":
            c["buttons"][lo:hi, _BTN[seg.payload]] = 1.0
        elif seg.kind == "free":
            if real_chunk is not None:
                for k in ("buttons", "j_left", "j_right"):
                    c[k][lo:hi] = real_chunk[k][lo:hi]
        # "idle" -> leave zeros
    return c


# ---------------------------------------------------------------------------
# Plan library (temporally diverse), generated symmetrically over directions so
# the four cardinals are balanced.
# ---------------------------------------------------------------------------
_HOLD_PHRASES = ["keep going {d}", "hold {d} the whole time", "move {d} continuously",
                 "go {d} and don't stop", "stay {d}"]
_TAP_PHRASES = ["tap {d} then keep playing", "briefly go {d}", "nudge {d} then continue",
                "a quick step {d}, then react"]
_SEQ_PHRASES = ["go {a} then {b}", "first {a}, then {b}", "{a}, then switch to {b}",
                "head {a} and afterwards {b}"]


def _make_library() -> list[SyntheticPlan]:
    plans: list[SyntheticPlan] = []
    for d in CARDINALS:  # HOLD
        plans.append(SyntheticPlan(
            f"hold_{d}", [p.format(d=d) for p in _HOLD_PHRASES],
            [Segment(0.0, 1.0, "dir", _DIRS[d])]))
    for d in CARDINALS:  # TAP
        plans.append(SyntheticPlan(
            f"tap_{d}", [p.format(d=d) for p in _TAP_PHRASES],
            [Segment(0.0, 0.33, "dir", _DIRS[d]), Segment(0.33, 1.0, "free")]))
    seq_pairs = [("left", "right"), ("right", "left"), ("up", "down"), ("down", "up"),
                 ("left", "up"), ("up", "right"), ("right", "down"), ("down", "left")]
    for a, b in seq_pairs:  # SEQ2
        plans.append(SyntheticPlan(
            f"seq_{a}_{b}", [p.format(a=a, b=b) for p in _SEQ_PHRASES],
            [Segment(0.0, 0.5, "dir", _DIRS[a]), Segment(0.5, 1.0, "dir", _DIRS[b])]))
    for a, b, cc in [("left", "up", "right"), ("up", "right", "down"), ("down", "left", "up")]:  # SEQ3
        plans.append(SyntheticPlan(
            f"seq3_{a}_{b}_{cc}",
            [f"go {a}, then {b}, then {cc}", f"first {a}, then {b}, finally {cc}"],
            [Segment(0.0, 1/3, "dir", _DIRS[a]), Segment(1/3, 2/3, "dir", _DIRS[b]),
             Segment(2/3, 1.0, "dir", _DIRS[cc])]))
    # SEQHET: heterogeneous 2-segment plans mixing a stick DIRECTION and a BUTTON press
    # IN SEQUENCE, both orders. Hardest routing test: the two segments live in DIFFERENT
    # parts of the 25-dim action vector (stick dims 21-24 vs button dims 0-20), so the
    # model must route segment-i to the right region AND modality. grounded=False (these
    # involve a button), so they enter training only when grounded_only=False
    # (the --het-heavy / button-inclusive recipe). Opposite orders share no label, so the
    # order-aware contrastive loss separates "dir then button" from "button then dir".
    _HET_BTN = {"jump": "south", "attack": "west"}
    # Fully balanced: every cardinal x every button x both orders, so up/down get the
    # SAME supervision as left/right (EXP-021 found up/down regressed because they were
    # under-represented 2:1 in heterogeneous plans). 4 dirs x 2 buttons = 8 pairs x 2
    # orders = 16 plans; each direction appears in 4 plans, each button in 8.
    het_pairs = [(d, b) for d in ("left", "right", "up", "down") for b in ("jump", "attack")]
    for d, btn in het_pairs:
        bkey = _HET_BTN[btn]
        plans.append(SyntheticPlan(  # direction first, then button
            f"seqhet_{d}_{btn}",
            [f"go {d}, then {btn}", f"first {d}, then {btn}", f"{d}, then {btn}"],
            [Segment(0.0, 0.5, "dir", _DIRS[d]), Segment(0.5, 1.0, "button", bkey)],
            grounded=False))
        plans.append(SyntheticPlan(  # button first, then direction
            f"seqhet_{btn}_{d}",
            [f"{btn}, then go {d}", f"first {btn}, then {d}", f"{btn}, then head {d}"],
            [Segment(0.0, 0.5, "button", bkey), Segment(0.5, 1.0, "dir", _DIRS[d])],
            grounded=False))
    # SEQDUR: UNEVEN-duration 2-segment plans. Same two directions, but the TEXT implies
    # how long the first segment lasts, and the TARGET splits at the matching point. This
    # tests whether routing keys off plan CONTENT (the duration words) vs a fixed even
    # 50/50 split prior. Each (a,b,split) is a distinct contrastive label so the loss can
    # separate "briefly a" from "mostly a". Splits at 1/4, 1/2, 3/4 of the 18-step chunk.
    _DUR_PHRASES = {
        25: ["briefly go {a}, then {b}", "a quick {a}, then {b}", "tap {a}, then go {b}"],
        50: ["go {a} then {b}", "first {a}, then {b}", "{a}, then switch to {b}"],
        75: ["go {a} for a long time, then {b}", "mostly {a}, then a bit of {b}",
             "hold {a}, then briefly {b}"],
    }
    for a, b in [("left", "right"), ("right", "left"), ("up", "down"), ("down", "up")]:
        for pct, split in ((25, 0.25), (50, 0.5), (75, 0.75)):
            plans.append(SyntheticPlan(
                f"seqdur_{a}_{b}_{pct}",
                [p.format(a=a, b=b) for p in _DUR_PHRASES[pct]],
                [Segment(0.0, split, "dir", _DIRS[a]), Segment(split, 1.0, "dir", _DIRS[b])]))
    plans.append(SyntheticPlan(  # IDLE
        "idle", ["do nothing", "stay idle", "wait", "hold still", "stop moving"],
        [Segment(0.0, 1.0, "idle")]))
    plans.append(SyntheticPlan(  # BUTTON taps
        "jump", ["jump", "jump now", "press jump", "hop up"],
        [Segment(0.0, 0.15, "button", "south"), Segment(0.15, 1.0, "free")], grounded=False))
    plans.append(SyntheticPlan(
        "attack", ["attack", "hit it", "strike now", "swing"],
        [Segment(0.0, 0.2, "button", "west"), Segment(0.2, 1.0, "free")], grounded=False))
    return plans


PLANS: list[SyntheticPlan] = _make_library()
GROUNDED_PLANS = [p for p in PLANS if p.grounded]
PLAN_BY_NAME = {p.name: p for p in PLANS}


# ---------------------------------------------------------------------------
# Cross-chunk plans: ONE plan spans A consecutive action-chunks. Each chunk has its
# own within-chunk schedule + contrastive label. The model emits K*A plan tokens (A
# blocks of K); a per-chunk cursor a selects block a.
#   * HOLD-per-chunk  (cc_*):    chunk0 holds left, chunk1 holds right (EXP-027).
#   * NESTED (ccnest_*):         chunk0 does left->right WITHIN it, chunk1 up->down.
# So the SEQ can happen ACROSS chunks (cursor) AND WITHIN a chunk (action positions).
# ---------------------------------------------------------------------------
@dataclass
class CrossChunkPlan:
    name: str
    phrasings: list[str]
    chunk_scheds: list[list[Segment]]  # per-chunk within-chunk schedule (len == A)
    chunk_labels: list[str]            # per-chunk contrastive label key (dir_x / seq_x_y)
    grounded: bool = True

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_scheds)


_CC_PHRASES = ["go {a}, then {b}", "first {a}, then {b}", "{a}, then {b}",
               "head {a}, then go {b}"]


def _hold_sched(d: str) -> list[Segment]:
    return [Segment(0.0, 1.0, "dir", _DIRS[d])]


def _seq_sched(a: str, b: str) -> list[Segment]:
    return [Segment(0.0, 0.5, "dir", _DIRS[a]), Segment(0.5, 1.0, "dir", _DIRS[b])]


def _make_cross_chunk_library() -> list[CrossChunkPlan]:
    pairs = [("left", "right"), ("right", "left"), ("up", "down"), ("down", "up"),
             ("left", "up"), ("up", "right"), ("right", "down"), ("down", "left")]
    out = []
    for a, b in pairs:
        out.append(CrossChunkPlan(
            f"cc_{a}_{b}", [p.format(a=a, b=b) for p in _CC_PHRASES],
            [_hold_sched(a), _hold_sched(b)], [f"dir_{a}", f"dir_{b}"]))
    return out


def _make_nested_chunk_library() -> list[CrossChunkPlan]:
    # Each chunk is a within-chunk SEQ. chunk0 = a->b, chunk1 = c->d.
    combos = [(("left", "right"), ("up", "down")), (("up", "down"), ("left", "right")),
              (("left", "right"), ("right", "left")), (("up", "down"), ("down", "up")),
              (("right", "left"), ("down", "up")), (("down", "up"), ("right", "left"))]
    out = []
    for (a, b), (c, d) in combos:
        out.append(CrossChunkPlan(
            f"ccnest_{a}_{b}__{c}_{d}",
            [f"go {a} then {b}, then {c} then {d}",
             f"first {a} then {b}, after that {c} then {d}"],
            [_seq_sched(a, b), _seq_sched(c, d)], [f"seq_{a}_{b}", f"seq_{c}_{d}"]))
    return out


def _phrase_seq(dirs: list[str]) -> list[str]:
    body = ", then ".join(dirs)
    return [f"go {body}", "go " + " then ".join(dirs)]


def _make_cc_hold_A(dir_seqs: list[list[str]]) -> list[CrossChunkPlan]:
    """A-chunk hold plans: each chunk holds dirs[a]. Generalizes the A=2 cc_* family to
    arbitrary horizon (e.g. A=4 rotating the 4 cardinals -> each cursor a distinct dir)."""
    out = []
    for dirs in dir_seqs:
        out.append(CrossChunkPlan(
            f"cc{len(dirs)}_" + "_".join(dirs), _phrase_seq(dirs),
            [_hold_sched(d) for d in dirs], [f"dir_{d}" for d in dirs]))
    return out


CROSS_CHUNK_PLANS: list[CrossChunkPlan] = _make_cross_chunk_library()
NESTED_CHUNK_PLANS: list[CrossChunkPlan] = _make_nested_chunk_library()
# A=4 (longer-horizon) hold plans: rotate the 4 cardinals so each of the 4 cursors holds
# a distinct direction (clean cursor-scaling test). num_queries = K*4 = 32.
CROSS_CHUNK_PLANS_A4: list[CrossChunkPlan] = _make_cc_hold_A([
    ["left", "up", "right", "down"], ["up", "right", "down", "left"],
    ["right", "down", "left", "up"], ["down", "left", "up", "right"]])
CC_PLAN_BY_NAME = {p.name: p for p in (CROSS_CHUNK_PLANS + NESTED_CHUNK_PLANS + CROSS_CHUNK_PLANS_A4)}


def build_cc_target(plan: CrossChunkPlan, cursor: int, H: int,
                    real_chunk: dict | None, modality: str = "stick") -> dict:
    """Materialize chunk `cursor` of a cross-chunk plan via its within-chunk schedule."""
    fake = SyntheticPlan(plan.name, plan.phrasings, plan.chunk_scheds[cursor], plan.grounded)
    return build_target(fake, H, real_chunk, modality)


def cc_chunk_label(plan: CrossChunkPlan, cursor: int) -> str:
    """Contrastive label key for chunk `cursor` (dir_<d> for holds, seq_<a>_<b> nested)."""
    return plan.chunk_labels[cursor]


class CrossChunkPlanSampler:
    """Samples a cross-chunk plan, a cursor a in [0,A), and builds chunk a's target.

    Returns (plan_name, plan_text, cursor, target, label_key). The same plan text is
    reused for every cursor; only the cursor + target differ, so the model must route
    block a -> chunk a's behavior. `pool` selects hold / nested / both plan families.
    """

    def __init__(self, action_horizon: int = 18, modality: str = "auto",
                 seed: int | None = None, pool: str = "hold"):
        self.H = action_horizon
        self.modality = modality
        self.rng = random.Random(seed)
        if pool == "nested":
            self.plans = NESTED_CHUNK_PLANS
        elif pool == "both":
            self.plans = CROSS_CHUNK_PLANS + NESTED_CHUNK_PLANS
        elif pool == "hold4":
            self.plans = CROSS_CHUNK_PLANS_A4
        else:
            self.plans = CROSS_CHUNK_PLANS

    def sample(self, real_chunk: dict | None = None, cursor: int | None = None):
        modality = self.modality
        if modality == "auto":
            if real_chunk is not None:
                from .actions import movement_modality
                modality = movement_modality(real_chunk["buttons"], real_chunk["j_left"])
                if modality == "none":
                    modality = "both"
            else:
                modality = "both"
        plan = self.rng.choice(self.plans)
        if cursor is None:
            cursor = self.rng.randrange(plan.num_chunks)
        text = self.rng.choice(plan.phrasings)
        target = build_cc_target(plan, cursor, self.H, real_chunk, modality)
        return plan.name, text, cursor, target, cc_chunk_label(plan, cursor)


def _group(name: str) -> str:
    if name.startswith("seqhet_"): return "seqhet"
    if name.startswith("seqdur_"): return "seqdur"
    if name.startswith("hold_"): return "hold"
    if name.startswith("tap_"): return "tap"
    if name.startswith("seq3_"): return "seq3"
    if name.startswith("seq_"): return "seq"
    if name in ("jump", "attack"): return "button"
    return "idle"


class SyntheticPlanSampler:
    """Samples a synthetic plan + builds its scheduled counterfactual target.

    Sampling is balanced across plan GROUPS (hold/tap/seq/seq3/idle[/button]) and,
    within a group, uniformly over its members — so cardinal directions and
    temporal structures are evenly represented (addresses direction imbalance).
    """

    def __init__(self, action_horizon: int = 18, grounded_only: bool = True,
                 modality: str = "auto", group_weights: dict | None = None,
                 seed: int | None = None, **_legacy):
        self.H = action_horizon
        self.modality = modality
        pool = GROUNDED_PLANS if grounded_only else PLANS
        self.groups: dict[str, list[SyntheticPlan]] = {}
        for p in pool:
            self.groups.setdefault(_group(p.name), []).append(p)
        self.group_names = list(self.groups.keys())
        self.group_weights = group_weights
        self.rng = random.Random(seed)

    def sample(self, real_chunk: dict | None = None,
               exclude: str | None = None) -> tuple[str, str, dict]:
        """Returns (plan_name, plan_text, target_chunk)."""
        modality = self.modality
        if modality == "auto":
            if real_chunk is not None:
                from .actions import movement_modality
                modality = movement_modality(real_chunk["buttons"], real_chunk["j_left"])
                if modality == "none":
                    modality = "both"
            else:
                modality = "both"
        gnames = self.group_names
        if self.group_weights:
            weights = [self.group_weights.get(g, 1.0) for g in gnames]
            g = self.rng.choices(gnames, weights=weights, k=1)[0]
        else:
            g = self.rng.choice(gnames)
        members = [p for p in self.groups[g] if p.name != exclude] or self.groups[g]
        plan = self.rng.choice(members)
        text = self.rng.choice(plan.phrasings)
        target = build_target(plan, self.H, real_chunk, modality)
        return plan.name, text, target
