"""Analytical residual-moment model for E2 (docs/E2_Derivation_2026-08-29.md).

Computes, without simulation, the moments of the checksum residual
r_k(x, t) = y_ck - lam * sum_{j in S_k} y_j under the E1 physical model
(power-law drift with nu-g0 correlation, programming spread, clipped nu),
for encodings a (bias) and b (differential), both readout modes.

Structure (derivation document sections in parentheses):
  cell level     m1, m2 of one PCM cell's drifted conductance (sec. 2)
  logical level  deviation e(v, t) of one logical cell from its ideal (sec. 3)
  group level    defect Delta = e_ck - (1/m) sum e_data: mean/variance (sec. 4)
  array level    mu_r, sigma_r of the pooled residual (sec. 5)
  readout        exact ratio scaling r_on = s * r_off, E[s], E[s^2] (sec. 6)
  decision       Gaussian FPR prediction against a fixed threshold (sec. 7)

All integrals are evaluated by dense Gaussian-weighted grids with the nu clip
and the g0 floor applied inside the integrand, so the model is exact for the
simulator's truncated distributions (no small-sigma expansion). mu_nu is
imported from drift_model: one source of truth for the physics. The encoding
target maps mirror encodings.py; the MC self-test (mc_selftest) guards the
mirror against divergence.

Encoding c is out of scope for the v1 model (sign flips and the g_min digital
subtraction break the linear-deviation structure; see derivation doc sec. 11).
"""

from __future__ import annotations

import dataclasses
from math import erfc, sqrt
from typing import Dict, Tuple

import numpy as np

from .drift_model import mu_nu

__all__ = [
    "ModelParams",
    "cell_moments",
    "value_nodes_w",
    "value_nodes_V",
    "logical_e_moments",
    "dist_e_moments",
    "delta_moments",
    "ref_scale_moments",
    "r_moments",
    "predict_fpr",
    "mc_selftest",
]


@dataclasses.dataclass(frozen=True)
class ModelParams:
    """Physical parameters of the run being modeled (read from its config.json,
    never from the current E1_config, per the traceability discipline)."""

    R: int
    C: int
    g_min: float
    g_max: float
    nu_hi: float
    nu_lo: float
    nu_clip: tuple
    prog_spread: bool
    sigma_p: float
    g0_floor: float
    w_sigma: float
    ref_g0: float
    fpr0: float
    sign_g_hi: float = 0.8
    sign_g_lo: float = 0.2

    @classmethod
    def from_run_config(cls, cfg_dict: dict) -> "ModelParams":
        c = cfg_dict["config"] if "config" in cfg_dict else cfg_dict
        return cls(
            R=c["R"],
            C=c["C"],
            g_min=c["g_min"],
            g_max=c["g_max"],
            nu_hi=c["nu_hi"],
            nu_lo=c["nu_lo"],
            nu_clip=tuple(c["nu_clip"]),
            prog_spread=c["prog_spread"],
            sigma_p=c["sigma_p"],
            g0_floor=c["g0_floor"],
            w_sigma=c["w_sigma"],
            ref_g0=c["ref_g0"],
            fpr0=c["fpr0"],
            sign_g_hi=c.get("sign_g_hi", 0.8),
            sign_g_lo=c.get("sign_g_lo", 0.2),
        )

    # encoding target maps -- must mirror encodings.py builders
    @property
    def s_d(self) -> float:
        return self.g_max - self.g_min

    @property
    def g_off(self) -> float:
        return 0.5 * (self.g_max + self.g_min)

    @property
    def slope_a(self) -> float:
        return 0.5 * (self.g_max - self.g_min)


# ---------------------------------------------------------------------------
# integration grids
# ---------------------------------------------------------------------------


def _gauss_nodes(sigma: float, half_width: float, n: int):
    """Uniform grid on +-half_width*sigma with renormalized Gaussian weights."""
    if sigma <= 0.0:
        return np.zeros(1), np.ones(1)
    x = np.linspace(-half_width * sigma, half_width * sigma, n)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return x, w / w.sum()


def cell_moments(
    g_bar: np.ndarray,
    L: float,
    sigma_nu: float,
    p: ModelParams,
    n_eps: int = 41,
    n_eta: int = 241,
) -> Tuple[np.ndarray, np.ndarray]:
    """First and second moments of one cell's drifted conductance g(t).

    g0 = max(g_bar*(1+eps), g0_floor), eps ~ N(0, sigma_p^2) (if prog_spread)
    nu = clip(mu_nu(g0) + eta, nu_clip),  eta ~ N(0, sigma_nu^2)
    m1 = E[g0 e^{-nu L}],  m2 = E[g0^2 e^{-2 nu L}]

    Exact for the truncated distributions (clip and floor applied inside the
    integrand). Vectorized over an array of targets g_bar.
    """
    g_bar = np.atleast_1d(np.asarray(g_bar, dtype=float))
    eps, w_eps = _gauss_nodes(p.sigma_p if p.prog_spread else 0.0, 6.0, n_eps)
    eta, w_eta = _gauss_nodes(sigma_nu, 8.0, n_eta)
    g0 = np.maximum(g_bar[:, None] * (1.0 + eps[None, :]), p.g0_floor)  # [Nv, Ne]
    mu = mu_nu(g0, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo)
    nu = np.clip(mu[:, :, None] + eta[None, None, :], p.nu_clip[0], p.nu_clip[1])
    F = np.exp(-nu * L)  # [Nv, Ne, Nk]
    w2 = w_eps[None, :, None] * w_eta[None, None, :]
    m1 = (g0[:, :, None] * F * w2).sum(axis=(1, 2))
    m2 = ((g0**2)[:, :, None] * F**2 * w2).sum(axis=(1, 2))
    return m1, m2


# ---------------------------------------------------------------------------
# stored-value distributions
# ---------------------------------------------------------------------------


def value_nodes_w(p: ModelParams, n: int = 401):
    """Nodes/weights of the data-value distribution: N(0, w_sigma^2) clipped to
    [-1, 1] (interior grid plus the two clip atoms), mirroring make_weights."""
    x = np.linspace(-1.0, 1.0, n + 2)[1:-1]
    w = np.exp(-0.5 * (x / p.w_sigma) ** 2)
    w = w / w.sum()
    atom = 0.5 * erfc(1.0 / (p.w_sigma * sqrt(2.0)))  # P(N > 1)
    w = w * (1.0 - 2.0 * atom)
    nodes = np.concatenate([x, [-1.0, 1.0]])
    wts = np.concatenate([w, [atom, atom]])
    return nodes, wts


def value_nodes_V(p: ModelParams, m: int, n: int = 401):
    """Checksum-value distribution V = (1/m) sum of m data values, by the CLT
    normal N(0, var_wc/m); var_wc is the exact clipped-w variance. For m >= 8
    the CLT error is negligible for the smooth integrands used here (stated as
    approximation A5 in the derivation doc)."""
    xw, ww = value_nodes_w(p)
    var_wc = float((ww * xw**2).sum())
    sig_V = sqrt(var_wc / m)
    return _gauss_nodes(sig_V, 8.0, n)


# ---------------------------------------------------------------------------
# logical-cell deviation moments
# ---------------------------------------------------------------------------


def logical_e_moments(
    encoding: str, v: np.ndarray, L: float, sigma_nu: float, p: ModelParams
) -> Tuple[np.ndarray, np.ndarray]:
    """E[e | v] and Var[e | v] of the logical-cell deviation e = readout - ideal
    at time L = ln(t/t0), for stored values v (vectorized).

    encoding b: e = (g+(t) - g-(t)) - s_d*v, two independent physical cells
    encoding a: e = g(t) - (g_off + slope_a*v), one physical cell
    """
    v = np.asarray(v, dtype=float)
    if encoding == "b":
        gp = p.g_min + p.s_d * np.maximum(v, 0.0)
        gn = p.g_min + p.s_d * np.maximum(-v, 0.0)
        both = np.concatenate([gp, gn])
        m1, m2 = cell_moments(both, L, sigma_nu, p)
        m1p, m1n = m1[: v.size], m1[v.size :]
        m2p, m2n = m2[: v.size], m2[v.size :]
        Ee = (m1p - m1n) - (gp - gn)
        Ve = (m2p - m1p**2) + (m2n - m1n**2)
        return Ee, Ve
    if encoding == "a":
        gb = p.g_off + p.slope_a * v
        m1, m2 = cell_moments(gb, L, sigma_nu, p)
        return m1 - gb, m2 - m1**2
    raise ValueError(f"model does not cover encoding {encoding!r} (see doc sec. 11)")


def dist_e_moments(
    encoding: str, nodes: np.ndarray, wts: np.ndarray, L: float, sigma_nu: float, p: ModelParams
) -> Tuple[float, float]:
    """Mean and variance of e over a stored-value distribution:
    Var[e] = E_v[Var[e|v]] + Var_v[E[e|v]] (device + value-heterogeneity)."""
    Ee, Ve = logical_e_moments(encoding, nodes, L, sigma_nu, p)
    mean = float((wts * Ee).sum())
    var = float((wts * (Ve + Ee**2)).sum() - mean**2)
    return mean, var


# ---------------------------------------------------------------------------
# group and array level
# ---------------------------------------------------------------------------


_TABLE_CACHE: dict = {}


def _h_tables(encoding: str, L: float, sigma_nu: float, p: ModelParams, nv: int = 1201):
    """Dense tables of h(v) = E[e|v] and Ve(v) = Var[e|v] on v in [-1, 1],
    cached per (encoding, L, sigma_nu, params). All group-level statistics are
    evaluated by interpolation on these tables."""
    key = (encoding, round(L, 12), sigma_nu, p)
    if key not in _TABLE_CACHE:
        v = np.linspace(-1.0, 1.0, nv)
        Ee, Ve = logical_e_moments(encoding, v, L, sigma_nu, p)
        _TABLE_CACHE[key] = (v, Ee, Ve)
    return _TABLE_CACHE[key]


def delta_moments(
    encoding: str, m: int, L: float, sigma_nu: float, p: ModelParams, lam_scale: int = 1
) -> Dict[str, float]:
    """Moments of the group defect Delta = e_ck(V) - (1/m) sum_{j=1..m} e_j(w_j)
    with V = (1/m) sum_j w_j (the checksum stores the scaled sum of the SAME
    row's data values -- V is dependent on the w_j, not an independent draw).

    Exact decomposition (derivation doc sec. 4):
      device part   E_V[Ve(V)] + E_w[Ve(w)]/m            (independent cells)
      value part    Var[h(V) - (1/m) sum_j h(w_j)]
                  = Var[h(V)] + Var[h(w)]/m - 2 Cov(h(V), h(w_1))
    The value part vanishes identically for linear h (the checksum invariant is
    transparent to linear value maps; only the nonlinearity of h survives), and
    vanishes exactly at m = 1. The covariance is computed on the joint grid
    V = w_1/m + ((m-1)/m) V', with V' the CLT normal of the other m-1 values
    (approximation A5; exact in w_1).
    """
    if encoding == "a" and lam_scale != 1:
        raise ValueError("lam_scale != 1 is defined for encoding b only (E3 spec P-B)")
    c = float(lam_scale)
    v_tab, h_tab, ve_tab = _h_tables(encoding, L, sigma_nu, p)
    xw, ww = value_nodes_w(p)
    hw = np.interp(xw, v_tab, h_tab)
    vew = np.interp(xw, v_tab, ve_tab)
    Ed = float((ww * hw).sum())
    Vphi = float((ww * hw**2).sum() - Ed**2)
    Ve_data = float((ww * vew).sum())

    # joint checksum-value grid: exact in w_1, CLT normal for the other m-1.
    # The checksum cell stores u = clip(c*V, -1, 1); its total value-level
    # deviation is htil(V) = h(u) + s_d*(u - c*V): the second term is the
    # ideal-cancellation mismatch caused by clipping (zero when |c*V| <= 1).
    var_wc = float((ww * xw**2).sum())
    if m == 1:
        Vgrid = xw[:, None]
        wq = np.ones(1)
    else:
        xq, wq = _gauss_nodes(sqrt(var_wc / (m - 1)), 8.0, 161)
        Vgrid = xw[:, None] / m + ((m - 1) / m) * xq[None, :]
    U = np.clip(c * Vgrid, -1.0, 1.0)
    hV = np.interp(U, v_tab, h_tab)
    if encoding == "b":
        hV = hV + p.s_d * (U - c * Vgrid)
    veV = np.interp(U, v_tab, ve_tab)
    wj = ww[:, None] * wq[None, :]
    clip_frac = float((wj * (np.abs(c * Vgrid) > 1.0)).sum())
    Ec = float((wj * hV).sum())
    VhV = float((wj * hV**2).sum() - Ec**2)
    Ve_ck = float((wj * veV).sum())
    EhVphi = float((wj * hV * hw[:, None]).sum())
    C = EhVphi - Ec * Ed

    val_part = VhV + (c * c) * Vphi / m - 2.0 * c * C
    return {
        "E_delta": Ec - c * Ed,
        "V_delta": (Ve_ck + (c * c) * Ve_data / m) + max(val_part, 0.0),
        "E_e_data": Ed,
        "V_e_data": Ve_data + Vphi,
        "E_e_ck": Ec,
        "V_e_ck": Ve_ck + VhV,
        "V_device": Ve_ck + (c * c) * Ve_data / m,
        "V_value": val_part,
        "cov_term": C,
        "clip_frac": clip_frac,
    }


def ref_scale_moments(L: float, sigma_nu: float, p: ModelParams) -> Tuple[float, float]:
    """E[s] and E[s^2] of the ratio-readout scale s = e^{+nu_ref L}, where the
    reference cell is programmed at ref_g0 with spread and its nu follows the
    same clipped model. The clip at nu_clip[1] bounds s, so the integral is
    well defined at all L."""
    eps, w_eps = _gauss_nodes(p.sigma_p if p.prog_spread else 0.0, 6.0, 41)
    eta, w_eta = _gauss_nodes(sigma_nu, 8.0, 241)
    g0 = np.maximum(p.ref_g0 * (1.0 + eps), p.g0_floor)
    mu = mu_nu(g0, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo)
    nu = np.clip(mu[:, None] + eta[None, :], p.nu_clip[0], p.nu_clip[1])
    s = np.exp(nu * L)
    w2 = w_eps[:, None] * w_eta[None, :]
    return float((s * w2).sum()), float((s**2 * w2).sum())


def r_moments(
    encoding: str, m: int, L: float, sigma_nu: float, p: ModelParams, ratio_norm: bool,
    lam_scale: int = 1,
) -> Tuple[float, float]:
    """(mu_r, sigma_r) of the pooled residual at time L = ln(t/t0).

    r = sum_i Delta_i x_i over R rows, x_i ~ U(0,1):
      mu_r0     = (R/2) E[Delta]
      sigma_r0^2 = (R/3) Var[Delta] + (R/12) E[Delta]^2
    (pooled over inputs and arrays; row/array independence per seed policy).
    Ratio readout multiplies r by the array-level scale s exactly (a, b), so
      mu_on = E[s] mu_r0,  sigma_on^2 = E[s^2](sigma_r0^2 + mu_r0^2) - mu_on^2.
    """
    d = delta_moments(encoding, m, L, sigma_nu, p, lam_scale=lam_scale)
    mu0 = 0.5 * p.R * d["E_delta"]
    var0 = (p.R / 3.0) * d["V_delta"] + (p.R / 12.0) * d["E_delta"] ** 2
    if not ratio_norm:
        return mu0, sqrt(var0)
    Es, Es2 = ref_scale_moments(L, sigma_nu, p)
    mu = Es * mu0
    var = Es2 * (var0 + mu0**2) - mu**2
    return mu, sqrt(var)


# ---------------------------------------------------------------------------
# decision layer
# ---------------------------------------------------------------------------


def _Q(z: float) -> float:
    return 0.5 * erfc(z / sqrt(2.0))


def predict_fpr(mu: float, sigma: float, theta: float) -> float:
    """P(|r| > theta) for r ~ N(mu, sigma^2)."""
    if sigma <= 0.0:
        return 0.0 if abs(mu) <= theta else 1.0
    return _Q((theta - mu) / sigma) + _Q((theta + mu) / sigma)


# ---------------------------------------------------------------------------
# Monte Carlo self-test (guards the model against the simulator's code path)
# ---------------------------------------------------------------------------


def mc_selftest(p: ModelParams, rng: np.random.Generator, n: int = 2_000_000) -> Dict[str, float]:
    """Cross-check cell_moments against direct sampling through the simulator's
    own physics functions (drift_model.sample_nu path semantics). Returns the
    worst relative error over a spread of (g_bar, sigma_nu, L) cases."""
    from .drift_model import sample_nu

    worst = 0.0
    cases = [(0.1, 0.02, np.log(1e4)), (0.5, 0.04, np.log(1e8)), (0.95, 0.005, np.log(1e2))]
    for g_bar, s_nu, L in cases:
        eps = rng.normal(0.0, p.sigma_p if p.prog_spread else 0.0, n)
        g0 = np.maximum(g_bar * (1.0 + eps), p.g0_floor)
        nu = sample_nu(
            g0, s_nu, rng,
            g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo, clip=p.nu_clip,
        )
        F = np.exp(-nu * L)
        mc1, mc2 = float((g0 * F).mean()), float(((g0 * F) ** 2).mean())
        m1, m2 = cell_moments(np.array([g_bar]), L, s_nu, p)
        worst = max(worst, abs(m1[0] / mc1 - 1.0), abs(m2[0] / mc2 - 1.0))
    return {"worst_rel_err": worst}


# ---------------------------------------------------------------------------
# E3 fault layer: minimum detectable amplitude under the drift-aware band
# ---------------------------------------------------------------------------


def fault_dm_nodes(
    encoding: str, kind: str, d: float, L: float, sigma_nu: float, p: ModelParams,
    n_w: int = 21, n_nu: int = 15,
):
    """Nodes/weights of the logical matrix-entry deviation dM(t) induced by one
    random data-cell fault of kind 'soft_abs' (absolute deviation d, random
    sign) or 'sa0'/'sa1'. Integrates over the stored value w ~ P_w, the branch
    (encoding b: pos/neg, logical sign +/-), the deviation sign, and the cell's
    clipped nu. Programming spread of the faulted cell is neglected (A7: sub-%
    effect on |dM|, quantified by mc_selftest_fault)."""
    xw2, ww2 = value_nodes_w(p, n=n_w)
    if kind == "sign_flip":
        xw2, ww2 = value_nodes_w(p, n=max(n_w, 101))
        eta, weta = _gauss_nodes(sigma_nu, 6.0, max(n_nu, 31))
        if encoding == "a":
            gbar = p.g_off + p.slope_a * xw2
            mu = mu_nu(gbar, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo)
            nu = np.clip(mu[:, None] + eta[None, :], p.nu_clip[0], p.nu_clip[1])
            dm = (-2.0 * p.slope_a * xw2)[:, None] * np.exp(-nu * L)
            w = ww2[:, None] * weta[None, :]
        elif encoding == "b":
            gp = p.g_min + p.s_d * np.maximum(xw2, 0.0)
            gn = p.g_min + p.s_d * np.maximum(-xw2, 0.0)
            mup = mu_nu(gp, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo)
            mun = mu_nu(gn, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo)
            Fp = np.exp(-np.clip(mup[:, None] + eta[None, :], p.nu_clip[0], p.nu_clip[1]) * L)
            Fn = np.exp(-np.clip(mun[:, None] + eta[None, :], p.nu_clip[0], p.nu_clip[1]) * L)
            dm = -2.0 * (gp[:, None, None] * Fp[:, :, None] - gn[:, None, None] * Fn[:, None, :])
            w = ww2[:, None, None] * weta[None, :, None] * weta[None, None, :]
        else:
            raise ValueError("sign_flip nodes are defined for encodings a/b only (E5 spec P-A)")
        w = w.ravel()
        return dm.ravel(), w / w.sum()
    branches = [("+", 1.0)] if encoding == "a" else [("+", 1.0), ("-", -1.0)]
    vals, wts = [], []
    for bname, bsign in branches:
        if encoding == "a":
            gbar = p.g_off + p.slope_a * xw2
        else:
            v_eff = xw2 if bname == "+" else -xw2
            gbar = p.g_min + p.s_d * np.maximum(v_eff, 0.0)
        mu = mu_nu(gbar, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo)
        eta, weta = _gauss_nodes(sigma_nu, 6.0, n_nu)
        nu = np.clip(mu[:, None] + eta[None, :], p.nu_clip[0], p.nu_clip[1])
        F = np.exp(-nu * L)  # [Nw, Nnu]
        for sgn in (+1.0, -1.0):
            if kind == "soft_abs":
                g_new = np.maximum(gbar + sgn * d, p.g0_floor)
                dm = bsign * (g_new - gbar)[:, None] * F
            elif kind in ("sa0", "sa1"):
                g_stuck = p.g_min if kind == "sa0" else p.g_max
                dm = bsign * (g_stuck - gbar[:, None] * F)
            else:
                raise ValueError(kind)
            w2 = (ww2[:, None] * weta[None, :]) * (0.5 / len(branches))
            vals.append(dm.ravel())
            wts.append(w2.ravel())
            if kind in ("sa0", "sa1"):
                break  # stuck faults have no deviation sign
    v = np.concatenate(vals)
    w = np.concatenate(wts)
    return v, w / w.sum()


def predict_p1(
    encoding: str, m: int, L: float, sigma_nu: float, p: ModelParams, ratio_norm: bool,
    f_band: float, dm_vals: np.ndarray, dm_wts: np.ndarray, lam_scale: int = 1,
    n_x: int = 17, n_s: int = 15, mu_hat: float = 0.0,
) -> float:
    """Per-MVM detection probability p1 = P(|r'| > f) (encoding a: centered on
    the model mean) for a persistent single-cell fault with deviation nodes
    dm_vals. Uses the exact shift identity r' = r - lam*x*dM on the OFF-side
    residual; ratio readout is folded in exactly via r_on = s*r_off, i.e. the
    threshold becomes f/s integrated over the reference-scale distribution.
    mu_hat is the constant the detector subtracts in the ON domain (encoding
    a's population mean track); per reference scale s the effective centering
    is mu_hat/s, which reproduces the per-array mean mismatch of the mean
    mixture."""
    mu0, sd0 = r_moments(encoding, m, L, sigma_nu, p, False, lam_scale=lam_scale)
    lam = lam_scale / m
    x = (np.arange(n_x) + 0.5) / n_x
    sh = -lam * x[:, None] * dm_vals[None, :]           # [Nx, Nd]
    wxd = (dm_wts[None, :] / n_x)
    if ratio_norm:
        eta, weta = _gauss_nodes(sigma_nu, 6.0, n_s)
        mur = mu_nu(np.array([p.ref_g0]), g_min=p.g_min, g_max=p.g_max,
                    nu_hi=p.nu_hi, nu_lo=p.nu_lo)[0]
        nur = np.clip(mur + eta, p.nu_clip[0], p.nu_clip[1])
        s_nodes = np.exp(nur * L)
        s_wts = weta.copy()
    else:
        s_nodes = np.array([1.0]); s_wts = np.array([1.0])
    fs = (f_band / s_nodes)[:, None, None]
    mh = (mu_hat / s_nodes)[:, None, None]
    z1 = (fs + mh - mu0 - sh[None, :, :]) / sd0
    z2 = (fs - mh + mu0 + sh[None, :, :]) / sd0
    tot = (_Q_arr(z1) + _Q_arr(z2)) * wxd[None, :, :]
    return float((s_wts * tot.sum(axis=(1, 2))).sum())


def _Q_vec(z: np.ndarray) -> np.ndarray:
    from numpy import vectorize
    return _Q_arr(z)


def _Q_arr(z: np.ndarray) -> np.ndarray:
    # vectorized standard normal tail via erfc on arrays
    from math import sqrt as _s
    try:
        from scipy.special import erfc as _e  # pragma: no cover
        return 0.5 * _e(z / _s(2.0))
    except Exception:
        x = np.asarray(z, dtype=float) / _s(2.0)
        # Abramowitz-Stegun 7.1.26 complementary error function, |eps|<1.5e-7
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        erf_ax = 1.0 - poly * np.exp(-ax * ax)
        erf_x = sign * erf_ax
        return 0.5 * (1.0 - erf_x)


def mc_selftest_fault(p: ModelParams, rng: np.random.Generator, n: int = 400_000) -> Dict[str, float]:
    """Cross-check fault_dm_nodes against direct sampling through the
    simulator's own ops (faults.faulted_cell + drift factor), programming
    spread included on the MC side so the A7 omission is measured, not
    assumed. Compares E[|dM|] and E[dM^2] for a soft_abs case."""
    from .faults import FaultSpec, faulted_cell

    d, s_nu, L = 0.3, 0.02, np.log(1e2)
    w = np.clip(rng.normal(0.0, p.w_sigma, n), -1.0, 1.0)
    branch = rng.integers(0, 2, n)
    v_eff = np.where(branch == 0, w, -w)
    gbar = p.g_min + p.s_d * np.maximum(v_eff, 0.0)
    eps = rng.normal(0.0, p.sigma_p if p.prog_spread else 0.0, n)
    g0 = np.maximum(gbar * (1.0 + eps), p.g0_floor)
    from .drift_model import sample_nu
    nu = sample_nu(g0, s_nu, rng, g_min=p.g_min, g_max=p.g_max,
                   nu_hi=p.nu_hi, nu_lo=p.nu_lo, clip=p.nu_clip)
    sgn = np.where(rng.integers(0, 2, n) == 0, 1, -1)
    g_new = np.maximum(g0 + sgn * d, p.g0_floor)
    dm = np.where(branch == 0, 1.0, -1.0) * (g_new - g0) * np.exp(-nu * L)
    vals, wts = fault_dm_nodes("b", "soft_abs", d, L, s_nu, p)
    q_abs = float((wts * np.abs(vals)).sum()); q_sq = float((wts * vals**2).sum())
    m_abs = float(np.abs(dm).mean()); m_sq = float((dm**2).mean())
    worst = max(abs(q_abs / m_abs - 1.0), abs(q_sq / m_sq - 1.0))
    # spot-check the per-cell op agrees with faulted_cell on one draw
    g_chk, nu_chk = faulted_cell(float(g0[0]), float(nu[0]),
                                 FaultSpec("soft_abs", d, 0, 0, "pos", int(sgn[0])), p)
    assert abs(g_chk - g_new[0]) < 1e-15 and nu_chk == nu[0]
    return {"worst_rel_err": worst, "mc_E_abs": m_abs, "quad_E_abs": q_abs}


def p_flip_model(L: float, sigma_nu: float, p: ModelParams, m: int = 16,
                 n_eps: int = 9, n_eta: int = 161) -> Dict[str, float]:
    """Sign-flip probability under the frozen pcm_ref comparator (E5 spec
    section 1) and the per-readout group contamination fraction, exact
    conditional on the shared sign-reference trajectory.

    The sharp dimension (the sign cell's own nu) is integrated in closed form:
    a hi cell flips iff nu_s > thr with thr = ln(g_s0/ref_t)/L, so its
    conditional probability is the exact clipped-Gaussian tail; only the
    smooth outer dimensions (programming spreads and the reference nu) are
    gridded. This removes the deep-tail resolution loss of an indicator
    quadrature at small sigma_nu."""
    eps, weps = _gauss_nodes(p.sigma_p if p.prog_spread else 0.0, 6.0, n_eps)
    eta, weta = _gauss_nodes(sigma_nu, 8.0, n_eta)
    lo_c, hi_c = p.nu_clip

    def tail(thr, mu):
        # P(clip(mu + eta) > thr) for scalar-broadcastable thr, mu
        z = (thr - mu) / max(sigma_nu, 1e-300)
        out = 0.5 * _erfc_arr(z / sqrt(2.0))
        out = np.where(thr >= hi_c, 0.0, out)
        out = np.where(thr < lo_c, 1.0, out)
        return out

    # reference grid (outer, smooth)
    g_r0 = np.maximum(p.ref_g0 * (1.0 + eps), p.g0_floor)
    mu_r = mu_nu(g_r0, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo)
    nu_r = np.clip(mu_r[:, None] + eta[None, :], lo_c, hi_c)
    ref_t = (g_r0[:, None] * np.exp(-nu_r * L)).ravel()
    w_ref = (weps[:, None] * weta[None, :]).ravel()

    def family(target, flip_when_low):
        g0 = np.maximum(target * (1.0 + eps), p.g0_floor)
        mu = mu_nu(g0, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo)
        if L <= 1e-12:
            ind = (g0[:, None] < ref_t[None, :]) if flip_when_low \
                else (g0[:, None] > ref_t[None, :])
            pf = ind.astype(float)
        else:
            thr = np.log(g0[:, None] / ref_t[None, :]) / L
            pf = tail(thr, mu[:, None])
            if not flip_when_low:
                pf = 1.0 - pf
        return (weps[:, None] * pf).sum(axis=0)  # conditional on each ref node

    p_hi = family(p.sign_g_hi, True)   # hi flips when gs drops below ref
    p_lo = family(p.sign_g_lo, False)  # lo flips when gs rises above ref
    n_half = 0.5 * p.R * (m + 1)
    contam = 1.0 - (1.0 - p_hi) ** n_half * (1.0 - p_lo) ** n_half
    pc = 0.5 * (p_hi + p_lo)
    Ep = float((w_ref * pc).sum())
    Ec = float((w_ref * contam).sum())
    return {
        "p_flip": Ep,
        "contamination": Ec,
        "p_hi": float((w_ref * p_hi).sum()),
        "p_lo": float((w_ref * p_lo).sum()),
        # reference-conditional spreads: one shared reference cell per array
        # makes the ensemble estimate scatter by sd_ref/sqrt(n_arrays)
        "p_flip_sd_ref": float(np.sqrt(max((w_ref * pc**2).sum() - Ep**2, 0.0))),
        "contam_sd_ref": float(np.sqrt(max((w_ref * contam**2).sum() - Ec**2, 0.0))),
    }


def _erfc_arr(z: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erfc as _e  # pragma: no cover
        return _e(z)
    except Exception:
        return 2.0 * _Q_arr(np.asarray(z, dtype=float) * sqrt(2.0))


def mc_selftest_signflip(p: ModelParams, rng: np.random.Generator, n: int = 400_000) -> Dict[str, float]:
    """Sampled sign_flip deviation moments (encoding b, spread included on the
    MC side) against fault_dm_nodes quadrature."""
    from .drift_model import sample_nu

    s_nu, L = 0.02, np.log(1e2)
    w = np.clip(rng.normal(0.0, p.w_sigma, n), -1.0, 1.0)
    gp = np.maximum((p.g_min + p.s_d * np.maximum(w, 0.0)) * (1 + rng.normal(0, p.sigma_p, n)), p.g0_floor)
    gn = np.maximum((p.g_min + p.s_d * np.maximum(-w, 0.0)) * (1 + rng.normal(0, p.sigma_p, n)), p.g0_floor)
    nup = sample_nu(gp, s_nu, rng, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo, clip=p.nu_clip)
    nun = sample_nu(gn, s_nu, rng, g_min=p.g_min, g_max=p.g_max, nu_hi=p.nu_hi, nu_lo=p.nu_lo, clip=p.nu_clip)
    dm = -2.0 * (gp * np.exp(-nup * L) - gn * np.exp(-nun * L))
    vals, wts = fault_dm_nodes("b", "sign_flip", 0.0, L, s_nu, p)
    q_abs = float((wts * np.abs(vals)).sum()); q_sq = float((wts * vals ** 2).sum())
    worst = max(abs(q_abs / np.abs(dm).mean() - 1.0), abs(q_sq / (dm ** 2).mean() - 1.0))
    return {"worst_rel_err": float(worst)}


def mc_selftest_pflip(p: ModelParams, rng: np.random.Generator,
                      n_ref: int = 4000, m: int = 16) -> Dict[str, float]:
    """Sampled flip probability and contamination against p_flip_model at a
    late time. The contamination check uses the direct any-flip indicator per
    reference (unbiased for the nonlinear group statistic); the marginal uses
    the pooled cell count."""
    from .drift_model import sample_nu

    s_nu, L = 0.02, np.log(1e8)
    ref = p_flip_model(L, s_nu, p, m=m)
    n_half = int(p.R * (m + 1) / 2)

    def draw(target, shape):
        g0 = np.maximum(target * (1 + rng.normal(0, p.sigma_p, shape)), p.g0_floor)
        nu = sample_nu(g0, s_nu, rng, g_min=p.g_min, g_max=p.g_max,
                       nu_hi=p.nu_hi, nu_lo=p.nu_lo, clip=p.nu_clip)
        return g0 * np.exp(-nu * L)

    flips = tot = anyf = 0
    chunk = 400
    for lo_i in range(0, n_ref, chunk):
        k = min(chunk, n_ref - lo_i)
        refs = draw(p.ref_g0, k)[:, None]
        hi = draw(p.sign_g_hi, (k, n_half))
        lo = draw(p.sign_g_lo, (k, n_half))
        fh, fl = hi < refs, lo > refs
        flips += fh.sum() + fl.sum()
        tot += 2 * k * n_half
        anyf += (fh.any(axis=1) | fl.any(axis=1)).sum()
    p_mc = flips / tot
    c_mc = anyf / n_ref
    worst = max(abs(ref["p_flip"] / p_mc - 1.0),
                abs(ref["contamination"] / max(c_mc, 1e-12) - 1.0))
    return {"worst_rel_err": float(worst), "p_mc": float(p_mc),
            "p_model": ref["p_flip"], "c_mc": float(c_mc),
            "c_model": ref["contamination"]}


def sigma_digitized(sigma_analog: float, q: float, m: int) -> float:
    """Prop 3(i): fault-free residual scale after b_y-bit output quantization.
    sigma_dig^2 = sigma^2 + c_q q^2 with c_q = (1 + 1/m)/12 (checksum column
    plus the 1/m-scaled data sum, independent uniform rounding errors)."""
    return float(np.sqrt(sigma_analog ** 2 + (1.0 + 1.0 / m) / 12.0 * q * q))
