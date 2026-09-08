"""Per-cell power-law conductance drift for PCM arrays.

Spec: docs/E1_Design_Spec_2026-08-22.md, section 3.

Model
-----
    g(t)      = g0 * (t/t0)^(-nu)          (t0 = 1 s, convention)
    nu        = mu_nu(g0) + eta,  eta ~ N(0, sigma_nu^2), clipped to nu_clip
    mu_nu(g0) = nu_hi + (nu_lo - nu_hi) * (ln g0 - ln g_min) / (ln g_max - ln g_min)

mu_nu is monotonically decreasing in g0 (higher conductance, lower drift
exponent). The numerical endpoints are PROVISIONAL and must be recalibrated
against Boniardi & Ielmini (APL 2011) and Antolini et al. (JETCAS 2023,
Fig. 8) before any value enters the paper. E1 conclusions rely only on
(i) monotonicity of mu_nu and (ii) sigma_nu > 0 (spec section 3 discipline
box).
"""

from __future__ import annotations

import numpy as np

__all__ = ["mu_nu", "sample_nu", "drift_factor", "conductance_at"]


def mu_nu(g0, *, g_min, g_max, nu_hi, nu_lo):
    """Mean drift exponent as a log-linear, monotone-decreasing function of g0.

    g0 values outside [g_min, g_max] (programming-spread tails) extrapolate the
    same log-linear form; downstream clipping of nu keeps values physical.
    """
    g0 = np.asarray(g0, dtype=float)
    frac = (np.log(g0) - np.log(g_min)) / (np.log(g_max) - np.log(g_min))
    return nu_hi + (nu_lo - nu_hi) * frac


def sample_nu(g0, sigma_nu, rng, *, g_min, g_max, nu_hi, nu_lo, clip):
    """Draw per-cell drift exponents nu = mu_nu(g0) + N(0, sigma_nu^2), clipped."""
    g0 = np.asarray(g0, dtype=float)
    nu = mu_nu(g0, g_min=g_min, g_max=g_max, nu_hi=nu_hi, nu_lo=nu_lo)
    nu = nu + rng.normal(0.0, sigma_nu, size=g0.shape)
    return np.clip(nu, clip[0], clip[1])


def drift_factor(nu, t_rel):
    """(t/t0)^(-nu); t_rel is a positive scalar, nu an array or scalar."""
    return np.power(float(t_rel), -np.asarray(nu, dtype=float))


def conductance_at(g0, nu, t_rel):
    """Conductance at relative time t_rel under power-law drift."""
    return np.asarray(g0, dtype=float) * drift_factor(nu, t_rel)
