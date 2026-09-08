"""Decision metrics (spec section 5): constant-threshold calibration, exceedance
rates with Wilson confidence intervals, residual quantiles, and the late-window
growth-slope fit used for the C2 encoding-ranking preview.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "calibrate_threshold",
    "exceed_counts",
    "wilson_ci",
    "signed_quantiles",
    "late_slope",
]


def calibrate_threshold(abs_r_t0, fpr0: float) -> float:
    """theta = (1 - fpr0) quantile of |r| at t0, pooled per configuration.

    Pooling across all arrays of a configuration is deliberate and slightly
    optimistic for the baseline (spec section 5, stated explicitly).
    """
    return float(np.quantile(np.asarray(abs_r_t0, dtype=float).ravel(), 1.0 - fpr0))


def exceed_counts(r, theta: float):
    """(k, n): number of |r| > theta exceedances and total sample count."""
    r = np.asarray(r)
    return int((np.abs(r) > theta).sum()), int(r.size)


def wilson_ci(k: int, n: int, z: float = 1.959964):
    """Wilson score interval for a binomial rate; returns (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def signed_quantiles(r, levels):
    """Signed residual percentiles at the given levels (spec section 5)."""
    return np.percentile(np.asarray(r, dtype=float).ravel(), list(levels))


def late_slope(t_rel, sigma_r, t_min: float = 1e2):
    """Fit ln sigma_r = slope * ln ln(t_rel) + intercept over t_rel >= t_min.

    Under the first-order skeleton sigma_r ~ sigma_nu * ln(t/t0) the slope is
    approximately 1; deviations feed the C2 ranking preview. Returns
    (slope, intercept, n_points); NaN slope when fewer than 2 valid points.
    """
    t = np.asarray(t_rel, dtype=float)
    s = np.asarray(sigma_r, dtype=float)
    mask = (t >= t_min) & (s > 0.0) & np.isfinite(s)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan"), int(mask.sum())
    x = np.log(np.log(t[mask]))
    y = np.log(s[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), int(mask.sum())
