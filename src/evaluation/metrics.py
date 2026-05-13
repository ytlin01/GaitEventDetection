"""
Evaluation metrics for comparing detected gait events against ground truth.
"""

import numpy as np


def match_events(detected_indices: list, truth_indices: list, tolerance: int) -> list:
    """
    Greedily match detected events to ground truth within a sample tolerance.

    Parameters
    ----------
    detected_indices : sorted list of detected sample indices
    truth_indices    : sorted list of ground-truth sample indices
    tolerance        : maximum sample distance for a valid match

    Returns
    -------
    List of (detected_idx, truth_idx) matched pairs
    """
    matched = []
    used_truth = set()
    for d in sorted(detected_indices):
        for t in sorted(truth_indices):
            if t in used_truth:
                continue
            if abs(d - t) <= tolerance:
                matched.append((d, t))
                used_truth.add(t)
                break
    return matched


def precision_recall(detected: dict, ground_truth: dict, tolerance: int = 5) -> dict:
    """
    Compute per-event-type precision, recall, and F1 score.

    Parameters
    ----------
    detected, ground_truth : dicts with keys LHS, RHS, LTO, RTO
                             mapping to lists of sample indices
    tolerance              : max allowed sample offset for a match

    Returns
    -------
    dict: {event_type: {"precision": float, "recall": float, "f1": float}}
    """
    results = {}
    for event in ("LHS", "RHS", "LTO", "RTO"):
        d = detected.get(event, [])
        t = ground_truth.get(event, [])
        matched = match_events(d, t, tolerance)
        tp = len(matched)
        prec = tp / len(d) if d else 0.0
        rec = tp / len(t) if t else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[event] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    return results


def timing_errors(detected: dict, ground_truth: dict, tolerance: int = 5) -> dict:
    """
    Compute mean and std timing error (in samples) for each event type.

    Parameters
    ----------
    detected, ground_truth : same format as precision_recall
    tolerance              : max sample offset for a match

    Returns
    -------
    dict: {event_type: {"mean_error": float, "std_error": float, "n_matched": int}}
    """
    results = {}
    for event in ("LHS", "RHS", "LTO", "RTO"):
        d = detected.get(event, [])
        t = ground_truth.get(event, [])
        matched = match_events(d, t, tolerance)
        errors = [abs(dm - tm) for dm, tm in matched]
        results[event] = {
            "mean_error": round(float(np.mean(errors)), 3) if errors else float("nan"),
            "std_error": round(float(np.std(errors)), 3) if errors else float("nan"),
            "n_matched": len(matched),
        }
    return results


def print_report(detected: dict, ground_truth: dict, fs: float = 100.0, tolerance: int = 5) -> None:
    """Print a formatted evaluation report to stdout."""
    pr = precision_recall(detected, ground_truth, tolerance)
    te = timing_errors(detected, ground_truth, tolerance)

    print(f"\n{'Event':<6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  "
          f"{'MeanErr(ms)':>12}  {'StdErr(ms)':>11}  {'Matched':>8}")
    print("-" * 68)
    for event in ("LHS", "RHS", "LTO", "RTO"):
        p = pr[event]
        t = te[event]
        mean_ms = t["mean_error"] / fs * 1000 if not np.isnan(t["mean_error"]) else float("nan")
        std_ms = t["std_error"] / fs * 1000 if not np.isnan(t["std_error"]) else float("nan")
        print(
            f"{event:<6}  {p['precision']:>6.3f}  {p['recall']:>6.3f}  {p['f1']:>6.3f}  "
            f"{mean_ms:>12.1f}  {std_ms:>11.1f}  {t['n_matched']:>8}"
        )
    print()
