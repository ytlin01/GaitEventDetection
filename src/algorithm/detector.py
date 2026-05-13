"""
Gait Event Detector — offline and real-time modes.

Detects four gait events per stride from filtered pelvis motion data:
  LHS  Left Heel Strike
  RHS  Right Heel Strike
  LTO  Left Toe Off
  RTO  Right Toe Off

Algorithm overview
------------------
1. Compute discrete gradients of z_pos.
2. At each downward gradient transition (start of descent):
     Queue a Heel Strike candidate. Confirm it offset1 samples later
     when the gradient turns positive (trough passed). At confirmation
     time, classify as RHS or LHS by checking whether a Y-rotation peak
     falls within offset3 samples of the candidate index.
3. At each upward gradient transition (trough moment):
     Search backward offset2 samples to confirm a trough preceded it.
     Identify the paired Heel Strike at index n − offset4.
     If that HS was RHS → this event is LTO; if LHS → RTO.

Both detect() (offline, full-array) and detect_step() (real-time,
sample-by-sample) implement this identical logic.
"""

import numpy as np


class GaitEventDetector:
    def __init__(self, offset1: int, offset2: int, offset3: int, offset4: int):
        """
        Parameters
        ----------
        offset1 : samples from Heel Strike candidate to Z-position trough confirmation
        offset2 : samples to search backward from upward transition for a trough
        offset3 : samples around a Heel Strike to search for a Y-rotation peak
        offset4 : samples from a Heel Strike to its paired Toe Off
        """
        self.offset1 = offset1
        self.offset2 = offset2
        self.offset3 = offset3
        self.offset4 = offset4

        # Real-time state
        self._z_buf: list[float] = []
        self._yrot_buf: list[float] = []
        self._pending_hs: dict[int, bool] = {}   # step → True (awaiting confirmation)
        self._labeled: dict[int, str] = {}        # step → confirmed event label
        self._step: int = 0

    # ------------------------------------------------------------------
    # Offline
    # ------------------------------------------------------------------

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
        labeled: dict[int, str] = {}

        for n in range(1, N):

            # Downward transition: queue HS candidate, confirm offset1 later
            if z_grad[n] < 0 and z_grad[n - 1] >= 0:
                lookahead = n + self.offset1
                if lookahead < N and z_grad[lookahead] > 0:
                    label = self._classify_hs(n, y_rot)
                    labeled[n] = label

            # Upward transition: trough moment → Toe Off candidate
            elif z_grad[n] > 0 and z_grad[n - 1] <= 0:
                lo = max(0, n - self.offset2)
                if np.any(z_grad[lo:n] < 0):
                    paired_idx = n - self.offset4
                    if paired_idx in labeled:
                        if labeled[paired_idx] == "RHS":
                            labeled[n] = "LTO"
                        elif labeled[paired_idx] == "LHS":
                            labeled[n] = "RTO"

        events: dict[str, list[int]] = {"LHS": [], "RHS": [], "LTO": [], "RTO": []}
        for idx, label in sorted(labeled.items()):
            events[label].append(idx)
        return events

    # ------------------------------------------------------------------
    # Real-time
    # ------------------------------------------------------------------

    def detect_step(self, z_val: float, yrot_val: float) -> list[str]:
        """
        Process one incoming sample. Returns a list of event labels that
        fire at this step (e.g. ['RHS'] or []).

        Classification is deferred to confirmation time (offset1 steps
        after the downward transition) so that the Y-rotation window
        around the candidate index is fully available.
        """
        self._z_buf.append(z_val)
        self._yrot_buf.append(yrot_val)
        n = self._step
        events: list[str] = []

        if n == 0:
            self._step += 1
            return events

        z_grad_curr = self._z_buf[n] - self._z_buf[n - 1]
        z_grad_prev = self._z_buf[n - 1] - self._z_buf[n - 2] if n >= 2 else 0.0

        # Confirm or expire any pending HS that is now offset1 steps old
        pending_n = n - self.offset1
        if pending_n in self._pending_hs:
            if z_grad_curr > 0:
                label = self._classify_hs(pending_n, self._yrot_buf)
                self._labeled[pending_n] = label
                events.append(label)
            del self._pending_hs[pending_n]

        # New downward transition → queue HS candidate
        if z_grad_curr < 0 and z_grad_prev >= 0:
            self._pending_hs[n] = True

        # Upward transition → Toe Off candidate
        if z_grad_curr > 0 and z_grad_prev <= 0:
            lo = max(0, n - self.offset2)
            grads = [self._z_buf[i] - self._z_buf[i - 1] for i in range(max(1, lo), n)]
            if any(g < 0 for g in grads):
                paired_step = n - self.offset4
                if paired_step in self._labeled:
                    if self._labeled[paired_step] == "RHS":
                        events.append("LTO")
                    elif self._labeled[paired_step] == "LHS":
                        events.append("RTO")

        self._step += 1
        return events

    def reset(self):
        """Reset real-time state (call after calibration or session restart)."""
        self._z_buf.clear()
        self._yrot_buf.clear()
        self._pending_hs.clear()
        self._labeled.clear()
        self._step = 0

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _classify_hs(self, n: int, y_rot) -> str:
        """
        Classify a Heel Strike candidate at index n as Right or Left by
        searching for a Y-rotation peak within offset3 samples of n.
        """
        lo = max(0, n - abs(self.offset3))
        hi = min(len(y_rot), n + abs(self.offset3) + 1)
        window = y_rot[lo:hi]
        local = n - lo

        is_peak = (
            0 < local < len(window) - 1
            and window[local] > window[local - 1]
            and window[local] > window[local + 1]
        )
        return "RHS" if is_peak else "LHS"
