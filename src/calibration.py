"""
Compute the four calibration offsets that relate detectable signal features
(Z-position troughs and Y-rotation peaks) to ground-truth gait events.

Offset definitions
------------------
offset1 : index distance from Heel Strike to the following Z-position trough
offset2 : index distance from the Z-position trough to the following Toe Off
offset3 : index distance from Right Heel Strike to the nearest Y-rotation peak
offset4 : index distance from Heel Strike to its paired Toe Off
"""

import numpy as np


def _find_troughs(z_grad):
    """Return indices where z_grad transitions from negative to positive (trough)."""
    troughs = []
    for n in range(1, len(z_grad)):
        if z_grad[n] > 0 and z_grad[n - 1] < 0:
            troughs.append(n)
    return np.array(troughs)


def _find_peaks(y_grad):
    """Return indices where y_grad transitions from positive to negative (peak)."""
    peaks = []
    for n in range(1, len(y_grad)):
        if y_grad[n] < 0 and y_grad[n - 1] > 0:
            peaks.append(n)
    return np.array(peaks)


def compute_offsets(z_pos, y_rot, HS_left_idx, HS_right_idx, TO_left_idx, TO_right_idx):
    """
    Compute mean calibration offsets from ground-truth event indices.

    Parameters
    ----------
    z_pos        : (N,) array — filtered, transformed vertical pelvis position
    y_rot        : (N,) array — filtered pelvis Y rotation (Euler angle)
    HS_left_idx  : int array — ground-truth left heel strike sample indices
    HS_right_idx : int array — ground-truth right heel strike sample indices
    TO_left_idx  : int array — ground-truth left toe off sample indices
    TO_right_idx : int array — ground-truth right toe off sample indices

    Returns
    -------
    dict with keys offset1, offset2, offset3, offset4 (all integers, mean values)
    """
    z_grad = np.diff(z_pos, prepend=z_pos[0])
    y_grad = np.diff(y_rot, prepend=y_rot[0])

    troughs = _find_troughs(z_grad)
    y_peaks = _find_peaks(y_grad)

    all_HS = np.concatenate([HS_left_idx, HS_right_idx])
    all_TO = np.concatenate([TO_left_idx, TO_right_idx])

    # offset1: HS → nearest trough (trough follows HS)
    offset1_vals = []
    for hs in all_HS:
        future_troughs = troughs[troughs > hs]
        if len(future_troughs):
            offset1_vals.append(future_troughs[0] - hs)

    # offset2: nearest trough → TO (TO follows trough)
    offset2_vals = []
    for to in all_TO:
        past_troughs = troughs[troughs < to]
        if len(past_troughs):
            offset2_vals.append(to - past_troughs[-1])

    # offset3: RHS → nearest Y-rotation peak
    offset3_vals = []
    for rhs in HS_right_idx:
        if len(y_peaks):
            nearest_peak = y_peaks[np.argmin(np.abs(y_peaks - rhs))]
            offset3_vals.append(rhs - nearest_peak)

    # offset4: HS → paired TO
    offset4_vals = []
    for hs in all_HS:
        future_TOs = all_TO[all_TO > hs]
        if len(future_TOs):
            offset4_vals.append(future_TOs[0] - hs)

    return {
        "offset1": int(round(np.mean(offset1_vals))) if offset1_vals else 0,
        "offset2": int(round(np.mean(offset2_vals))) if offset2_vals else 0,
        "offset3": int(round(np.mean(offset3_vals))) if offset3_vals else 0,
        "offset4": int(round(np.mean(offset4_vals))) if offset4_vals else 0,
    }
