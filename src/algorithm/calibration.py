"""
Compute the four calibration offsets that relate detectable signal features
(Z-position troughs and Y-rotation peaks) to ground-truth gait events,
and persist them to / load them from a JSON file.

Offset definitions
------------------
offset1 : sample distance from Heel Strike candidate to the following Z-position trough
offset2 : sample distance from the Z-position trough to the following Toe Off
offset3 : sample distance from Right Heel Strike to the nearest Y-rotation peak
offset4 : sample distance from Heel Strike to its paired Toe Off
"""

import json
import numpy as np


def compute_offsets(
    z_pos, y_rot,
    HS_left_idx, HS_right_idx,
    TO_left_idx, TO_right_idx,
) -> dict:
    """
    Compute mean calibration offsets from ground-truth event indices.

    Parameters
    ----------
    z_pos        : (N,) array — filtered vertical pelvis position
    y_rot        : (N,) array — filtered pelvis Y rotation (Euler angle)
    HS_left_idx  : int array — ground-truth left heel strike sample indices
    HS_right_idx : int array — ground-truth right heel strike sample indices
    TO_left_idx  : int array — ground-truth left toe off sample indices
    TO_right_idx : int array — ground-truth right toe off sample indices

    Returns
    -------
    dict with keys offset1, offset2, offset3, offset4 (integers, mean values)
    """
    z_grad = np.diff(z_pos, prepend=z_pos[0])
    y_grad = np.diff(y_rot, prepend=y_rot[0])

    troughs = _find_troughs(z_grad)
    y_peaks = _find_peaks(y_grad)

    all_HS = np.concatenate([HS_left_idx, HS_right_idx])
    all_TO = np.concatenate([TO_left_idx, TO_right_idx])

    # offset1: HS → nearest following trough
    offset1_vals = []
    for hs in all_HS:
        future = troughs[troughs > hs]
        if len(future):
            offset1_vals.append(future[0] - hs)

    # offset2: nearest preceding trough → TO
    offset2_vals = []
    for to in all_TO:
        past = troughs[troughs < to]
        if len(past):
            offset2_vals.append(to - past[-1])

    # offset3: RHS → nearest Y-rotation peak
    offset3_vals = []
    for rhs in HS_right_idx:
        if len(y_peaks):
            nearest = y_peaks[np.argmin(np.abs(y_peaks - rhs))]
            offset3_vals.append(rhs - nearest)

    # offset4: HS → paired (next) TO
    offset4_vals = []
    for hs in all_HS:
        future = all_TO[all_TO > hs]
        if len(future):
            offset4_vals.append(future[0] - hs)

    return {
        "offset1": int(round(np.mean(offset1_vals))) if offset1_vals else 0,
        "offset2": int(round(np.mean(offset2_vals))) if offset2_vals else 0,
        "offset3": int(round(np.mean(offset3_vals))) if offset3_vals else 0,
        "offset4": int(round(np.mean(offset4_vals))) if offset4_vals else 0,
    }


def save_offsets(offsets: dict, path: str) -> None:
    """Persist calibration offsets to a JSON file."""
    with open(path, "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"Offsets saved to {path}: {offsets}")


def load_offsets(path: str) -> dict:
    """Load calibration offsets from a JSON file."""
    with open(path) as f:
        offsets = json.load(f)
    return {k: int(v) for k, v in offsets.items()}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _find_troughs(z_grad):
    """Return indices where z_grad transitions from negative to positive."""
    return np.array([
        n for n in range(1, len(z_grad))
        if z_grad[n] > 0 and z_grad[n - 1] < 0
    ])


def _find_peaks(y_grad):
    """Return indices where y_grad transitions from positive to negative."""
    return np.array([
        n for n in range(1, len(y_grad))
        if y_grad[n] < 0 and y_grad[n - 1] > 0
    ])
