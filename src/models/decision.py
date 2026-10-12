"""Canonical score normalization and original VisualAnomalyScreening thresholding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_detection_intervals(
    score_vector, alpha, method="mean", smoothing=True, sliding=False, anomaly_padding=0
):
    """
    Given an anomaly score vector, compute detection intervals using either
    global thresholding or a sliding-window threshold, and optionally
    smooth the scores first via an exponentially-weighted moving average.

    Parameters
    ----------
    score_vector : array-like, shape (T,)
        The aligned anomaly score vector.
    alpha : float
        Gaussian tail parameter for mean/median + norm.ppf(1-alpha) * spread;
        it does not force a fixed proportion of detected points.
    method : {'mean', 'median'}, default='mean'
        Whether to compute central tendency + spread as (mean, std) or
        (median, MAD).
    smoothing : bool, default=False
        If True, first replace `score_vector` with its EWMA (alpha = smoothing_alpha).
    smoothing_alpha : float in (0,1), default=0.3
        The alpha for the EWMA if `smoothing=True`.
    sliding : bool, default=False
        If True, compute a local threshold in a sliding window (size T/3, step T/10)
        and mark any point exceeding its window's threshold as anomalous.
        Otherwise use a single global threshold.
    anomaly_padding : int, default=0
        Number of time points to pad before and after each detected interval.

    Returns
    -------
    detection_intervals : list of (start, end) tuples
        Contiguous index ranges (padded) where the (smoothed) score exceeds the threshold.
    threshold : float or None
        The global threshold (if sliding=False), else None.
    scores : np.ndarray
        The (optionally smoothed) score vector.
    """
    scores = np.array(score_vector, dtype=float)
    T = len(scores)

    if smoothing:
        span = max(1, int(len(scores) * 0.01))
        scores = pd.Series(scores).ewm(span=span).mean().values

    # Precompute the Gaussian multiplier
    z = norm.ppf(1 - alpha)

    # 2) Build a boolean mask of anomalies
    anomaly_flags = np.zeros(T, dtype=bool)

    if not sliding:
        # 2a) Global threshold
        if method == "mean":
            central = scores.mean()
            spread = scores.std()
        elif method == "median":
            central = np.median(scores)
            spread = np.median(np.abs(scores - central))
        else:
            raise ValueError("method must be 'mean' or 'median'")
        threshold = central + z * spread
        anomaly_flags = scores > threshold

    else:
        # 2b) Sliding‑window threshold
        threshold = 0  # With sliding window based method, threshold varies.
        win = max(1, T // 3)
        step = max(1, T // 10)
        for start in range(0, T, step):
            end = min(start + win, T)
            segment = scores[start:end]
            if method == "mean":
                central = segment.mean()
                spread = segment.std()
            else:  # median
                central = np.median(segment)
                spread = np.median(np.abs(segment - central))
            thresh_local = central + z * spread
            # mark any point in [start:end) above its local thresh
            anomaly_flags[start:end] |= segment > thresh_local

    # 3) Extract contiguous intervals from the boolean mask
    detection_intervals = []
    in_int = False
    for i, flag in enumerate(anomaly_flags):
        if flag and not in_int:
            in_int = True
            start = i
        elif not flag and in_int:
            in_int = False
            detection_intervals.append((start, i - 1))
    if in_int:
        detection_intervals.append((start, T - 1))

    # 4) Apply padding if requested
    if anomaly_padding > 0:
        padded = []
        for s, e in detection_intervals:
            s_pad = max(0, s - anomaly_padding)
            e_pad = min(T - 1, e + anomaly_padding)
            padded.append((s_pad, e_pad))

        # If nothing was detected, stay empty
        if not padded:
            detection_intervals = []
        else:
            # 5) Merge overlapping or contiguous intervals
            padded.sort(key=lambda x: x[0])
            merged = []
            cur_s, cur_e = padded[0]
            for s_next, e_next in padded[1:]:
                if s_next <= cur_e + 1:  # overlap or contiguous
                    cur_e = max(cur_e, e_next)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s_next, e_next
            merged.append((cur_s, cur_e))

            detection_intervals = merged

    return detection_intervals, threshold, scores


def robust_positive_z(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        std = np.std(x)
        if std < 1e-12:
            return np.zeros_like(x)
        z = (x - med) / (std + 1e-12)
    else:
        z = (x - med) / (1.4826 * mad + 1e-12)
    return np.maximum(z, 0.0)
