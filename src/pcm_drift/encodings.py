"""Encoding builders and effective readout matrices (spec section 4).

Three encodings, each producing (CrossbarState, ColumnLayout):

    E1.a  build_bias     bias mapping with a physical bias column
    E1.b  build_diff     spatial differential pairs (two physical columns per
                         logical column)
    E1.c  build_signmag  sign-magnitude with separate PCM sign cells

Logical column order is always [data 0..C-1 | checksum 0..K-1]; data groups are
contiguous ranges of m columns, so group k occupies logical columns
[k*m, (k+1)*m) and its checksum is logical column C + k. Checksum cells store
the scaled group sum lam * sum_{j in S_k} w_ij with lam = 1/m (single-cell
storability; spec section 2). `effective_matrix` returns M such that Y = X @ M
yields the logical column outputs on which the residuals of spec section 2 are
defined.

Under lam = 1/m the bias terms of encoding a cancel exactly in the residual for
both BIAS_MODE variants (spec section 4.1, finding F1); the bias column is
still programmed for layout fidelity but is not consumed by any E1 metric.

The sampling order inside each builder is fixed and documented so that a given
device RNG reproduces an array bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .crossbar import CrossbarState, RefCell
from .drift_model import conductance_at, drift_factor, sample_nu

__all__ = [
    "ColumnLayout",
    "build",
    "build_bias",
    "build_diff",
    "build_signmag",
    "effective_matrix",
    "sign_flip_matrices",
]


@dataclass
class ColumnLayout:
    """Mapping between logical columns (data + checksum) and physical columns."""

    encoding: str
    R: int
    C: int
    m: int
    K: int
    lam: float
    n_logical: int  # C + K
    bias_col: Optional[int] = None  # physical index of the bias column (encoding a)
    pos_cols: Optional[np.ndarray] = None  # [C+K] physical indices (encoding b)
    neg_cols: Optional[np.ndarray] = None  # [C+K] physical indices (encoding b)

    def group_cols(self, k: int) -> np.ndarray:
        """Logical data column indices of group k."""
        return np.arange(k * self.m, (k + 1) * self.m)

    def cksum_col(self, k: int) -> int:
        """Logical column index of group k's checksum column."""
        return self.C + k


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _nu_kwargs(cfg) -> dict:
    return dict(
        g_min=cfg.g_min,
        g_max=cfg.g_max,
        nu_hi=cfg.nu_hi,
        nu_lo=cfg.nu_lo,
        clip=cfg.nu_clip,
    )


def _program(g_target, rng, cfg) -> np.ndarray:
    """Apply multiplicative programming spread and a positivity floor.

    g0 = g_target * (1 + e), e ~ N(0, sigma_p^2), floored at cfg.g0_floor.
    Applied to every PCM cell (data, checksum, bias, sign, reference) when
    cfg.prog_spread is True; this static spread is the nonzero t0 residual
    floor required for the degenerate-case claim (spec section 5, note B1).
    """
    g = np.asarray(g_target, dtype=float)
    if cfg.prog_spread:
        g = g * (1.0 + rng.normal(0.0, cfg.sigma_p, size=g.shape))
    return np.maximum(g, cfg.g0_floor)


def _make_ref(rng, sigma_nu, cfg) -> RefCell:
    """Program one reference cell at cfg.ref_g0 (spread + nu from the same model)."""
    g0 = float(_program(np.array(cfg.ref_g0), rng, cfg))
    nu = float(sample_nu(np.array(g0), sigma_nu, rng, **_nu_kwargs(cfg)))
    return RefCell(g0=g0, nu=nu)


def _checksum_values(W: np.ndarray, m: int) -> np.ndarray:
    """Scaled group sums V [R, K], V[:, k] = (1/m) * sum over group k columns."""
    R, C = W.shape
    if C % m != 0:
        raise ValueError(f"C={C} must be divisible by m={m}")
    K = C // m
    return W.reshape(R, K, m).sum(axis=2) / m


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_bias(W: np.ndarray, m: int, sigma_nu: float, rng, cfg):
    """E1.a bias mapping: g = g_off + s * w; physical bias column appended last.

    Sampling order: (1) programming spread over the whole physical array
    [data | checksum | bias], (2) per-cell nu, (3) ratio reference cell.
    """
    R, C = W.shape
    V = _checksum_values(W, m)
    K = V.shape[1]
    g_off = 0.5 * (cfg.g_max + cfg.g_min)
    s = 0.5 * (cfg.g_max - cfg.g_min)
    targets = np.concatenate(
        [g_off + s * W, g_off + s * V, np.full((R, 1), g_off)], axis=1
    )
    g0 = _program(targets, rng, cfg)
    nu = sample_nu(g0, sigma_nu, rng, **_nu_kwargs(cfg))
    ref_ratio = _make_ref(rng, sigma_nu, cfg)
    layout = ColumnLayout("a", R, C, m, K, 1.0 / m, C + K, bias_col=C + K)
    state = CrossbarState(g0=g0, nu=nu, ref_ratio=ref_ratio, W=W.copy())
    return state, layout


def build_diff(W: np.ndarray, m: int, sigma_nu: float, rng, cfg, lam_scale: int = 1):
    """E1.b spatial differential: per logical value v a pair
    g+ = g_min + s_d * max(v, 0), g- = g_min + s_d * max(-v, 0).

    Physical layout interleaves [pos_0, neg_0, pos_1, neg_1, ...] over the
    logical order [data | checksum]. Sampling order: (1) programming spread
    over the interleaved physical array, (2) per-cell nu, (3) ratio reference.
    """
    R, C = W.shape
    V = np.clip(lam_scale * _checksum_values(W, m), -1.0, 1.0)
    K = V.shape[1]
    vals = np.concatenate([W, V], axis=1)  # [R, C+K]
    s_d = cfg.g_max - cfg.g_min
    g_pos = cfg.g_min + s_d * np.maximum(vals, 0.0)
    g_neg = cfg.g_min + s_d * np.maximum(-vals, 0.0)
    P = 2 * (C + K)
    targets = np.empty((R, P), dtype=float)
    targets[:, 0::2] = g_pos
    targets[:, 1::2] = g_neg
    g0 = _program(targets, rng, cfg)
    nu = sample_nu(g0, sigma_nu, rng, **_nu_kwargs(cfg))
    ref_ratio = _make_ref(rng, sigma_nu, cfg)
    layout = ColumnLayout(
        "b",
        R,
        C,
        m,
        K,
        lam_scale / m,
        C + K,
        pos_cols=np.arange(0, P, 2),
        neg_cols=np.arange(1, P, 2),
    )
    state = CrossbarState(g0=g0, nu=nu, ref_ratio=ref_ratio, W=W.copy())
    return state, layout


def build_signmag(W: np.ndarray, m: int, sigma_nu: float, rng, cfg):
    """E1.c sign-magnitude: magnitude g = g_min + s_d * |v|; per-cell PCM sign
    cell programmed to sign_g_hi (positive) or sign_g_lo (negative).

    Sampling order: (1) spread on magnitude cells, (2) nu magnitude, (3) spread
    on sign cells, (4) nu sign, (5) ratio reference, (6) sign reference.
    """
    R, C = W.shape
    V = _checksum_values(W, m)
    K = V.shape[1]
    vals = np.concatenate([W, V], axis=1)  # [R, C+K]
    s_d = cfg.g_max - cfg.g_min
    mag_t = cfg.g_min + s_d * np.abs(vals)
    sign_true = np.where(vals >= 0.0, 1.0, -1.0)
    sign_t = np.where(sign_true > 0.0, cfg.sign_g_hi, cfg.sign_g_lo)
    g0 = _program(mag_t, rng, cfg)
    nu = sample_nu(g0, sigma_nu, rng, **_nu_kwargs(cfg))
    sg0 = _program(sign_t, rng, cfg)
    snu = sample_nu(sg0, sigma_nu, rng, **_nu_kwargs(cfg))
    ref_ratio = _make_ref(rng, sigma_nu, cfg)
    ref_sign = _make_ref(rng, sigma_nu, cfg)
    layout = ColumnLayout("c", R, C, m, K, 1.0 / m, C + K)
    state = CrossbarState(
        g0=g0,
        nu=nu,
        ref_ratio=ref_ratio,
        ref_sign=ref_sign,
        sign_g0=sg0,
        sign_nu=snu,
        sign_true=sign_true,
        W=W.copy(),
    )
    return state, layout


_BUILDERS = {"a": build_bias, "b": build_diff, "c": build_signmag}


def build(encoding: str, W: np.ndarray, m: int, sigma_nu: float, rng, cfg, lam_scale: int = 1):
    """Dispatch to the encoding-specific builder."""
    if lam_scale != 1:
        if encoding != "b":
            raise ValueError("lam_scale != 1 is defined for encoding b only (E3 spec P-B)")
        return build_diff(W, m, sigma_nu, rng, cfg, lam_scale=lam_scale)
    return _BUILDERS[encoding](W, m, sigma_nu, rng, cfg)


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------


def _ratio_scale(state: CrossbarState, t_rel: float, ratio_norm: bool) -> float:
    """scale = g_ref(t0)/g_ref(t) = (t/t0)^{+nu_ref}; 1.0 when normalization off.

    Programming spread on the reference cell cancels in the ratio; only nu_ref
    (itself tied to the actual programmed conductance) matters.
    """
    if not ratio_norm:
        return 1.0
    return 1.0 / float(drift_factor(state.ref_ratio.nu, t_rel))


def effective_matrix(
    state: CrossbarState, layout: ColumnLayout, t_rel: float, ratio_norm: bool, cfg
) -> Tuple[np.ndarray, Dict]:
    """Effective logical readout matrix M [R, C+K] at time t_rel.

    Y = X @ M yields the logical column outputs consumed by
    checksum.residuals. Returns (M, info); info carries 'scale' and, for
    encoding c, 'sigma_hat' and 'flips' (both input-independent).

    Ratio normalization multiplies the conductance estimate by
    scale = g_ref(t0)/g_ref(t) before any digital constant (the nominal g_min
    of encoding c) is subtracted; for encodings a and b this reduces to scaling
    the outputs. Sign resolution under 'pcm_ref' compares raw cell currents and
    is therefore scale-invariant; under 'fixed_th' the compensated estimate is
    compared against the constant threshold.
    """
    scale = _ratio_scale(state, t_rel, ratio_norm)
    g_t = conductance_at(state.g0, state.nu, t_rel)

    if layout.encoding == "a":
        # bias column (last physical column) not consumed by E1 metrics (F1)
        M = scale * g_t[:, : layout.n_logical]
        return M, {"scale": scale}

    if layout.encoding == "b":
        M = scale * (g_t[:, layout.pos_cols] - g_t[:, layout.neg_cols])
        return M, {"scale": scale}

    # encoding c
    gs_t = conductance_at(state.sign_g0, state.sign_nu, t_rel)
    if cfg.sign_ref == "pcm_ref":
        ref_t = state.ref_sign.g_at(t_rel)
        sigma_hat = np.where(gs_t > ref_t, 1.0, -1.0)
    elif cfg.sign_ref == "fixed_th":
        sigma_hat = np.where(scale * gs_t > cfg.fixed_sign_th, 1.0, -1.0)
    else:
        raise ValueError(f"unknown sign_ref: {cfg.sign_ref}")
    M = sigma_hat * (scale * g_t - cfg.g_min)
    flips = sigma_hat != state.sign_true
    return M, {"scale": scale, "sigma_hat": sigma_hat, "flips": flips}


def sign_flip_matrices(state: CrossbarState, layout: ColumnLayout, t_rel: float, cfg) -> Dict[str, np.ndarray]:
    """Both sign-resolution rules evaluated at t_rel (encoding c only).

    Returns bool [R, C+K] flip matrices under three readout variants:
      'pcm_ref'        raw-current comparison against the drifting sign
                       reference cell (ratio-normalization invariant)
      'fixed_th_raw'   normalization off: raw estimate vs constant threshold
      'fixed_th_norm'  normalization on: compensated estimate vs constant

    Reported side by side per spec section 4.3 regardless of which rule the
    configured readout (cfg.sign_ref) uses.
    """
    gs_t = conductance_at(state.sign_g0, state.sign_nu, t_rel)
    ref_t = state.ref_sign.g_at(t_rel)
    scale_on = 1.0 / float(drift_factor(state.ref_ratio.nu, t_rel))
    out = {
        "pcm_ref": np.where(gs_t > ref_t, 1.0, -1.0) != state.sign_true,
        "fixed_th_raw": np.where(gs_t > cfg.fixed_sign_th, 1.0, -1.0) != state.sign_true,
        "fixed_th_norm": np.where(scale_on * gs_t > cfg.fixed_sign_th, 1.0, -1.0)
        != state.sign_true,
    }
    return out
