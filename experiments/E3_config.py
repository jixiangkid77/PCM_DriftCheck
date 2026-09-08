"""E3 parameter single source of truth (spec: docs/E3_Design_Spec_2026-08-29.md).

Physics parameters are NOT defined here: E3 imports E1Config so simulator and
model share one physical parameter set. This module holds only E3-side knobs:
the scan grids, the detection-latency budgets, pre-registered tolerances, and
ensemble sizes.

Detection metric (spec amendment A-2): the band keeps the per-MVM alarm rate
at FPR0 = 1% (continuity with E1/E2); a persistent fault with per-MVM hit
probability p1 is detected within N MVMs with probability 1 - (1-p1)^N, so the
budget N* maps to the threshold p1* = 1 - 0.1^(1/N*). d_min is the smallest
absolute conductance deviation reaching p1*, primary budget N* = 20.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["E3Config"]


def _t_grid():
    return tuple(float(10**k) for k in range(9))


@dataclass(frozen=True)
class E3Config:
    # detection protocol
    fpr0: float = 0.01  # per-MVM alarm budget (band design point)
    n_star_grid: tuple = (5, 20, 100)  # MVMs-to-90%-detection budgets
    n_star_primary: int = 20

    # scan grids (encoding b primary; encoding a slice at m=16, c=1)
    sigma_nu_grid: tuple = (0.005, 0.01, 0.02, 0.04)
    m_grid: tuple = (8, 16, 32)
    c_grid: tuple = (1, 2, 3)  # lam = c/m, encoding b only (spec P-A/P-B)
    t_rel_grid: tuple = field(default_factory=_t_grid)
    off_slice: tuple = (0.02, 16, 1)  # (sigma_nu, m, c) checked at ratio_norm off
    sa_slice: tuple = (0.02, 16, 1)  # sa0/sa1 reference curves (norm on)

    # fault amplitude axis (absolute conductance deviation, spec amendment A-1)
    delta_model_grid_lo: float = 0.005
    delta_ceiling: float = 0.95
    n_delta_model: int = 16  # log-spaced model curve
    sim_delta_factors: tuple = (0.5, 0.75, 1.0, 1.5, 2.25)  # around model dmin
    closed_probe: tuple = (0.3, 0.6, 0.95)  # sim probes when model says closed

    # ensembles and trials
    n_arrays: int = 20
    n_mvm: int = 2000
    n_trials: int = 4000
    seed_base: int = 20260821  # branches: device 5, inputs 6, faults 7

    # pre-registered tolerances (spec section 6)
    tol_dmin_rel: float = 0.15
    tol_p1_abs: float = 0.05
    fpr_gate: tuple = (0.005, 0.02)
    fpr_gate_hi_sigma_on: tuple = (0.003, 0.04)  # sigma_nu = 0.04, norm on

    # model quadrature sizes
    mc_tol_rel: float = 0.02
    mc_seed: int = 20260829
