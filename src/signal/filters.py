import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter


# ---------- Offline (zero-phase, full-array) ----------

def butterworth_filter(data, cutoff, fs, order=4):
    """Zero-phase Butterworth low-pass filter. Requires the full signal."""
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype="low", analog=False)
    return filtfilt(b, a, data)


def apply_butterworth_to_df(df, cutoff, fs, order=4):
    """Apply zero-phase Butterworth low-pass filter to every numeric column."""
    result = df.copy()
    for col in result.columns:
        result[col] = butterworth_filter(result[col].values, cutoff, fs, order)
    return result


# ---------- Real-time (causal, single-sample) ----------

class ButterLowPass_RT:
    """Causal Butterworth low-pass filter for sample-by-sample processing."""

    def __init__(self, order=4, cutoff=1, fs=100):
        self.nyquist = fs / 2.0
        self.b, self.a = butter(order, cutoff / self.nyquist, btype="low", analog=False)

    def apply(self, signal):
        """Apply to a list/array of samples accumulated so far."""
        return lfilter(self.b, self.a, signal)


class ButterHighPass_RT:
    """Causal Butterworth high-pass filter for sample-by-sample processing."""

    def __init__(self, order=4, cutoff=1, fs=100):
        self.nyquist = fs / 2.0
        self.b, self.a = butter(order, cutoff / self.nyquist, btype="high", analog=False)

    def apply(self, signal):
        return lfilter(self.b, self.a, signal)
