"""Action assembly: dataset parquet rows -> NitroGen action chunks.

The HF dataset stores one row per video frame at 60 fps, with 17 boolean button
columns and two joystick columns (each an [x, y] list in [-1, 1]). NitroGen's true
25-d action layout (verified empirically AND from nitrogen/shared.py) is:
    [ buttons(21) @ dims 0..20 , j_left @ 21..22 , j_right @ 23..24 ]
where the 21 button dims follow `BUTTON_ACTION_TOKENS` (the order play.py uses to
decode model output -> gamepad). The dataset's 17 columns are a SUBSET; the model
has 4 extra slots (RIGHT_BOTTOM/LEFT/RIGHT/UP) absent from the dataset (left 0).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..shared import BUTTON_ACTION_TOKENS

# Authoritative 21-dim model button order (index == model output dim 0..20),
# from nitrogen/shared.py / scripts/play.py (zip(TOKEN_SET, button_vector)).
MODEL_BUTTON_TOKENS = [t.lower() for t in BUTTON_ACTION_TOKENS]  # 21
NUM_BUTTON_SLOTS = len(MODEL_BUTTON_TOKENS)  # 21

# The 17 dataset parquet button columns.
DATASET_BUTTON_COLS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "north", "right_shoulder",
    "right_thumb", "right_trigger", "south", "start", "west",
]
# Map each dataset column -> its model output-dim index (4 model slots have no
# dataset column: right_bottom/right_left/right_right/right_up -> always 0).
DATASET_COL_TO_MODEL_DIM = {c: MODEL_BUTTON_TOKENS.index(c) for c in DATASET_BUTTON_COLS}

# Back-compat alias used elsewhere; now refers to the 21-dim model order.
BUTTON_ORDER = MODEL_BUTTON_TOKENS
NUM_BUTTONS = NUM_BUTTON_SLOTS  # 21


def load_chunk_actions(parquet_path: str, processed: bool = True) -> dict:
    """Load a chunk parquet -> dict of numpy arrays in MODEL button order.

    Returns:
        buttons:  (T, 21) float32 in {0,1}  -- placed at their model dims;
                  the 4 dataset-absent slots (right_bottom/left/right/up) are 0.
        j_left:   (T, 2)  float32 in [-1, 1]
        j_right:  (T, 2)  float32 in [-1, 1]
    """
    df = pl.read_parquet(parquet_path)
    T = df.height
    buttons = np.zeros((T, NUM_BUTTON_SLOTS), dtype=np.float32)
    for col, dim in DATASET_COL_TO_MODEL_DIM.items():
        if col in df.columns:
            buttons[:, dim] = df[col].to_numpy().astype(np.float32)
    j_left = np.asarray(df["j_left"].to_list(), dtype=np.float32)
    j_right = np.asarray(df["j_right"].to_list(), dtype=np.float32)
    return {"buttons": buttons, "j_left": j_left, "j_right": j_right}


def action_density(buttons: np.ndarray, j_left: np.ndarray, j_right: np.ndarray,
                   joy_eps: float = 0.1) -> np.ndarray:
    """Per-frame 'is the player doing something' mask -> (T,) bool.

    Used for IDLE-frame detection (idle == low density window). A frame is active
    if any button is pressed OR either joystick is deflected beyond joy_eps.
    """
    btn_active = buttons.astype(bool).any(axis=1)
    joy_active = (np.abs(j_left).sum(1) + np.abs(j_right).sum(1)) > joy_eps
    return btn_active | joy_active


# D-pad model dims (for movement-modality detection / d-pad direction targets).
DPAD_DIMS = {d: DATASET_COL_TO_MODEL_DIM[f"dpad_{d}"] for d in ["left", "right", "up", "down"]}


def movement_modality(buttons: np.ndarray, j_left: np.ndarray,
                      joy_eps: float = 0.2, ratio: float = 1.5) -> str:
    """Detect which control the streamer uses for directional movement.

    "go left" is control-scheme-ambiguous (d-pad-left vs left-stick-left), and
    which one actually moves the character is game-dependent. We infer it from the
    real actions so synthetic directional counterfactuals stay on-distribution.

    Returns one of: "stick", "dpad", "both", "none".
    """
    dpad = buttons[:, [DPAD_DIMS["left"], DPAD_DIMS["right"], DPAD_DIMS["up"], DPAD_DIMS["down"]]]
    dpad_use = dpad.astype(bool).any(1).mean()
    stick_use = (np.abs(j_left).sum(1) > joy_eps).mean()
    if dpad_use < 0.02 and stick_use < 0.02:
        return "none"
    if stick_use > ratio * dpad_use:
        return "stick"
    if dpad_use > ratio * stick_use:
        return "dpad"
    return "both"


def assemble_chunk(buttons: np.ndarray, j_left: np.ndarray, j_right: np.ndarray,
                   start_idx: int, action_horizon: int = 18, frame_stride: int = 2):
    """Slice an action chunk starting at frame `start_idx`.

    The dataset is 60 fps; NitroGen control frequency is lower. `frame_stride`
    decimates frames (stride 2 -> 30 Hz). Returns chunk arrays of length
    `action_horizon`, or None if there aren't enough frames.

    Returns dict with buttons (H,17), j_left (H,2), j_right (H,2).
    """
    idxs = start_idx + np.arange(action_horizon) * frame_stride
    if idxs[-1] >= buttons.shape[0]:
        return None
    return {
        "buttons": buttons[idxs],
        "j_left": j_left[idxs],
        "j_right": j_right[idxs],
    }


def is_idle_window(buttons, j_left, j_right, start_idx, action_horizon=18,
                   frame_stride=2, density_thresh=0.5) -> bool:
    """True if the action chunk starting at start_idx is low-activity (idle)."""
    idxs = start_idx + np.arange(action_horizon) * frame_stride
    if idxs[-1] >= buttons.shape[0]:
        return False
    dens = action_density(buttons[idxs], j_left[idxs], j_right[idxs]).mean()
    return dens < density_thresh


def chunk_dominant_dir(chunk: dict, eps: float = 0.15) -> str | None:
    """Post-hoc dominant cardinal direction of a REAL action chunk (stick + d-pad),
    for R0 post-hoc plans. Returns 'left'/'right'/'up'/'down' or None if no clear
    direction (so post-hoc plans only describe chunks the streamer clearly steered).
    j_left is in [-1,1] (x: -left/+right, y: -up/+down); d-pad adds a discrete vote.
    """
    jl = chunk["j_left"]; btn = chunk["buttons"]
    x = float(jl[:, 0].mean()) + float(btn[:, DPAD_DIMS["right"]].mean() - btn[:, DPAD_DIMS["left"]].mean())
    y = float(jl[:, 1].mean()) + float(btn[:, DPAD_DIMS["down"]].mean() - btn[:, DPAD_DIMS["up"]].mean())
    if max(abs(x), abs(y)) < eps:
        return None
    if abs(x) >= abs(y):
        return "right" if x > 0 else "left"
    return "down" if y > 0 else "up"

# Human-readable names for the buttons that matter for play (Xbox-ish gamepad).
_BTN_LABEL = {
    "south": "A/jump", "east": "B", "west": "X/attack", "north": "Y",
    "left_shoulder": "LB", "right_shoulder": "RB",
    "left_trigger": "LT/brake", "right_trigger": "RT/accelerate",
    "left_thumb": "L3", "right_thumb": "R3", "start": "start", "back": "back",
    "dpad_up": "dpad-up", "dpad_down": "dpad-down", "dpad_left": "dpad-left", "dpad_right": "dpad-right",
}
_BTN_IDX = {n: i for i, n in enumerate(BUTTON_ORDER)}


def _stick_dir(xy, eps: float = 0.25) -> str:
    x, y = float(xy[0]), float(xy[1])
    if max(abs(x), abs(y)) < eps:
        return "neutral"
    parts = []
    if y < -eps: parts.append("up")
    if y > eps: parts.append("down")
    if x < -eps: parts.append("left")
    if x > eps: parts.append("right")
    return "-".join(parts) if parts else "neutral"


def summarize_chunk(chunk: dict, n_seg: int = 3, btn_label: dict | None = None) -> str:
    """Terse natural-language description of an 18-step action chunk: left-stick motion over
    n_seg sub-segments + any buttons pressed (with rough duration). Used as the action-grounded
    CONTEXT for VLM plan generation and as the privileged action text for distillation teachers.
    Canonical home (planner_poc/action_summary.py re-exports this).

    btn_label: optional per-game {button_name -> human label} override. The default _BTN_LABEL is an
    Xbox/PLATFORMER vocabulary ("south"->"A/jump", "west"->"X/attack"), which is WRONG for top-down
    (Minish: no jump) or turn-based (Fire Emblem: confirm, not jump) games and makes the VLM hallucinate
    moves (e.g. Sonic "spin-dash"). Pass a per-game map so the action trace names buttons correctly. A
    mapping to "" (empty string) SUPPRESSES that button (use for buttons with no game meaning)."""
    labels = btn_label if btn_label is not None else _BTN_LABEL
    H = chunk["buttons"].shape[0]
    jl = chunk["j_left"]; btn = chunk["buttons"]
    seg = max(1, H // n_seg)
    seg_descs = []
    for k in range(n_seg):
        lo, hi = k * seg, (H if k == n_seg - 1 else (k + 1) * seg)
        seg_descs.append(_stick_dir(jl[lo:hi].mean(0)))
    collapsed = []
    for d in seg_descs:
        if not collapsed or collapsed[-1] != d:
            collapsed.append(d)
    parts = []
    if all(d == "neutral" for d in collapsed):
        parts.append("left stick: mostly neutral")
    else:
        parts.append("left stick: " + " then ".join(collapsed))
    held = []
    for name, idx in _BTN_IDX.items():
        frac = float((btn[:, idx] > 0.5).mean())
        if frac > 0.1:
            label = labels.get(name, name if btn_label is None else "")
            if not label:                 # suppressed (no game meaning for this button)
                continue
            when = "throughout" if frac > 0.7 else ("briefly" if frac < 0.35 else "for a while")
            held.append(f"{label} {when}")
    parts.append("buttons: " + ", ".join(held) if held else "no buttons")
    return "; ".join(parts)
