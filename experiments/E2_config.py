"""E2 parameter single source of truth.

E2 validates the analytical residual-moment model (src/pcm_drift/model.py,
derivation in docs/E2_Derivation_2026-08-29.md) against the stored E1 residual
samples. Physical parameters are NOT defined here: the validation reads them
from the source E1 run's config.json, so model and data always refer to the
same physics (traceability discipline). This module holds only E2-side knobs:
pre-registered agreement tolerances, quadrature sizes, and figure slices.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["E2Config"]


@dataclass(frozen=True)
class E2Config:
    # encodings covered by the v1 model (encoding c: doc sec. 11, out of scope)
    encodings: tuple = ("a", "b")

    # pre-registered agreement tolerances (doc sec. 9); exceedances are
    # reported as WARN in the log and flagged in model_vs_data.csv
    tol_sigma_rel: float = 0.10  # |sigma_model/sigma_data - 1|
    tol_mu_rel: float = 0.15  # enc a mean, relative, where |mu_data| > mu_floor
    mu_floor: float = 0.05  # below this |mu_data| the relative test is skipped
    tol_fpr_abs: float = 0.08  # |FPR_model - FPR_data|

    # Monte Carlo self-test of the cell-moment quadrature
    mc_n: int = 2_000_000
    mc_tol_rel: float = 0.01
    mc_seed: int = 20260829

    # figure slice
    fig_m: int = 16
