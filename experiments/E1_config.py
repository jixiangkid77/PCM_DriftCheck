"""E1 parameter single source of truth (spec section 6).

Every factor, default and seed policy of experiment E1 lives here; the entry
script E1_naive_checksum_failure.py and later experiments import from this
module instead of redefining numbers. All device-physics values are
PROVISIONAL placeholders pending calibration against Boniardi & Ielmini
(APL 2011) and Antolini et al. (JETCAS 2023, Fig. 8); they must not be quoted
in the paper (spec section 3 discipline box).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "E1Config",
    "ENC_IDS",
    "smoke_config",
    "rng_weights",
    "rng_inputs",
    "rng_device",
    "rng_fault",
    "make_weights",
    "make_inputs",
]


@dataclass(frozen=True)
class E1Config:
    # geometry (spec section 2, frozen)
    R: int = 128
    C: int = 128
    m_grid: tuple = (8, 16, 32)

    # normalized conductance domain -- PROVISIONAL
    g_min: float = 0.05
    g_max: float = 1.00

    # drift model (spec section 3) -- PROVISIONAL
    t0_s: float = 1.0
    nu_hi: float = 0.10
    nu_lo: float = 0.02
    nu_clip: tuple = (0.001, 0.20)
    sigma_nu_grid: tuple = (0.005, 0.01, 0.02, 0.04)

    # relative time grids (spec section 6); index 0 must be t0 (= 1.0)
    t_rel_grid: tuple = tuple(10.0 ** e for e in range(9))
    t_rel_grid_dense: tuple = tuple(10.0 ** (e / 2.0) for e in range(17))

    # programming spread (spec section 3, note B1) -- PROVISIONAL
    prog_spread: bool = True
    sigma_p: float = 0.02
    g0_floor: float = 1e-3

    # encodings (spec section 4)
    encodings: tuple = ("a", "b", "c")
    bias_mode: str = "column"  # {'column','digital'}; no effect on r under lam=1/m (F1)
    sign_ref: str = "pcm_ref"  # {'pcm_ref','fixed_th'}
    fixed_sign_th: float = 0.5
    sign_g_lo: float = 0.2
    sign_g_hi: float = 0.8
    ref_g0: float = 0.5  # ratio and sign reference cells (two separate cells)

    # readout variants
    ratio_norm_modes: tuple = (False, True)

    # sampling (spec section 6)
    n_arrays: int = 20
    n_mvm: int = 1000
    seed_base: int = 20260821
    w_sigma: float = 0.5  # weights ~ N(0, w_sigma^2) clipped to [-1, 1] (Bi 2023 precedent)

    # decision rule (spec section 5)
    fpr0: float = 0.01
    quantile_levels: tuple = (1, 5, 25, 50, 75, 95, 99)
    slope_t_min: float = 1e2  # late-window lower edge for the C2 slope fit

    # Phase B slice (spec section 6)
    b_sigma_nu: float = 0.02
    b_m: int = 16
    b_fault_kinds: tuple = ("sa0", "sa1", "soft")
    b_soft_d_grid: tuple = (0.10, 0.25, 0.50)
    b_n_trials: int = 200
    b_n_mvm: int = 100


ENC_IDS = {"a": 0, "b": 1, "c": 2}

# named substreams for the SeedSequence-based policy below
_STREAM_INPUTS = 1
_STREAM_DEVICE = 2
_STREAM_FAULT = 3


def smoke_config(cfg: E1Config) -> E1Config:
    """Tiny configuration for pipeline verification (--smoke)."""
    return replace(
        cfg,
        n_arrays=3,
        n_mvm=64,
        sigma_nu_grid=(0.02,),
        m_grid=(16,),
        t_rel_grid=(1.0, 1e2, 1e4, 1e6, 1e8),
        b_n_trials=12,
        b_n_mvm=16,
    )


# ---------------------------------------------------------------------------
# seed policy
# ---------------------------------------------------------------------------
# Array k's logical weight matrix W_k uses seed_base + k literally (spec
# section 6 wording) and is therefore SHARED across encodings, sigma_nu, m and
# readout variants: every configuration sees the same logical weights, so
# encoding comparisons are controlled comparisons. Inputs are shared the same
# way (and across t points: paired sampling reduces Monte Carlo noise in the
# FPR(t) trend). Device randomness (programming spread, per-cell nu, reference
# cells) is drawn per (array, encoding, sigma_nu index, m index) so different
# physical configurations are independent samples, reproducible bit for bit.


def rng_weights(cfg: E1Config, k: int) -> np.random.Generator:
    return np.random.default_rng(cfg.seed_base + k)


def rng_inputs(cfg: E1Config, k: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([cfg.seed_base, _STREAM_INPUTS, k]))


def rng_device(cfg: E1Config, k: int, enc: str, si: int, mi: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([cfg.seed_base, _STREAM_DEVICE, k, ENC_IDS[enc], si, mi])
    )


def rng_fault(cfg: E1Config, enc: str, cond_id: int, trial: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([cfg.seed_base, _STREAM_FAULT, ENC_IDS[enc], cond_id, trial])
    )


def make_weights(cfg: E1Config, k: int) -> np.ndarray:
    """Logical weights of array k: N(0, w_sigma^2) clipped to [-1, 1]."""
    W = rng_weights(cfg, k).normal(0.0, cfg.w_sigma, size=(cfg.R, cfg.C))
    return np.clip(W, -1.0, 1.0)


def make_inputs(cfg: E1Config, k: int) -> np.ndarray:
    """Input batch of array k: uniform [0, 1), shape [n_mvm, R]."""
    return rng_inputs(cfg, k).uniform(0.0, 1.0, size=(cfg.n_mvm, cfg.R))
