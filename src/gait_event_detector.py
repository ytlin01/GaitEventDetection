"""
Gait Event Detector — offline post-processing algorithm.

Detects four gait events per stride from filtered pelvis motion data:
  LHS  Left Heel Strike
  RHS  Right Heel Strike
  LTO  Left Toe Off
  RTO  Right Toe Off

Algorithm overview
------------------
1. Compute discrete gradients of z_pos and y_rot.
2. At each downward gradient transition (start of descent):
     Look ahead offset1 samples. If the gradient is positive by then,
     a Z-position trough is confirmed → candidate Heel Strike.
     Classify as RHS or LHS by checking whether a Y-rotation peak
     falls within offset3 samples of this index.
3. At each upward gradient transition (start of ascent = trough moment):
     Search backward offset2 samples to confirm a trough preceded it → candidate Toe Off.
     Identify the paired Heel Strike at index n − offset4.
     If that HS was RHS → this event is LTO; if LHS → RTO.

See algorithms/Real-Time Algorithm Pipeline.drawio for the full flowchart.
"""

import numpy as np


class GaitEventDetector:
    def __init__(self, offset1: int, offset2: int, offset3: int, offset4: int):
        """
        Parameters
        ----------
        offset1 : samples from Heel Strike to the Z-position trough
        offset2 : samples from the Z-position trough to Toe Off
        offset3 : samples between Right Heel Strike and nearest Y-rotation peak
        offset4 : samples from Heel Strike to its paired Toe Off
        """
        self.offset1 = offset1
        self.offset2 = offset2
        self.offset3 = offset3
        self.offset4 = offset4

    def detect(self, z_pos: np.ndarray, y_rot: np.ndarray) -> dict:
        """
        Run gait event detection on preprocessed pelvis signals.

        Parameters
        ----------
        z_pos : (N,) array — filtered vertical pelvis position
        y_rot : (N,) array — filtered pelvis Y rotation (Euler angle)

        Returns
        -------
        dict with keys 'LHS', 'RHS', 'LTO', 'RTO', each a sorted list of sample indices
        """
        z_grad = np.diff(z_pos, prepend=z_pos[0])
        N = len(z_pos)

        labeled = {}  # index → event label

        for n in range(1, N):

            # --- Downward transition: start of descent → Heel Strike candidate ---
            if z_grad[n] < 0 and z_grad[n - 1] >= 0:
                lookahead = n + self.offset1
                if lookahead < N and z_grad[lookahead] > 0:
                    label = self._classify_hs(n, y_rot)
                    labeled[n] = label

            # --- Upward transition: trough moment → Toe Off candidate ---
            elif z_grad[n] > 0 and z_grad[n - 1] <= 0:
                lo = max(0, n - self.offset2)
                trough_confirmed = np.any(z_grad[lo:n] < 0)

                if trough_confirmed:
                    paired_idx = n - self.offset4
                    if paired_idx in labeled:
                        if labeled[paired_idx] == "RHS":
                            labeled[n] = "LTO"
                        elif labeled[paired_idx] == "LHS":
                            labeled[n] = "RTO"

        events = {"LHS": [], "RHS": [], "LTO": [], "RTO": []}
        for idx, label in sorted(labeled.items()):
            events[label].append(idx)
        return events

    def _classify_hs(self, n: int, y_rot: np.ndarray) -> str:
        """
        Classify a Heel Strike at index n as Right or Left by searching for a
        Y-rotation peak within offset3 samples.
        """
        lo = max(0, n - abs(self.offset3))
        hi = min(len(y_rot), n + abs(self.offset3) + 1)
        window = y_rot[lo:hi]
        local = n - lo  # position of n within the window

        is_peak = (
            0 < local < len(window) - 1
            and window[local] > window[local - 1]
            and window[local] > window[local + 1]
        )
        return "RHS" if is_peak else "LHS"
