"""E5 parameter single source of truth (spec: docs/E5_Design_Spec_2026-08-30.md, v2 frozen).

Fault-class coverage across the three encodings under the drift-aware band
(encodings a/b: deployed band f(t) = theta0 * rho_model; encoding c: oracle
per-t empirical band per ruling P-A case three), plus the sign-flip phase
boundary t_c from the p_flip model against kappa * FPR0. Physics from
E1Config; ensemble seed coordinates from E3Config (device branch 5, inputs 6);
new streams: faults branch 9, flip-ensemble devices branch 10.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["E5Config"]


@dataclass(frozen=True)
class E5Config:
    # main comparison slice (spec section 4)
    sigma_nu: float = 0.02
    m: int = 16
    c: int = 1
    encodings: tuple = ("a", "b", "c")
    # (kind, delta) fault conditions; delta used by soft_abs only
    classes: tuple = (("sa0", 0.0), ("sa1", 0.0), ("soft_abs", 0.3),
                      ("soft_abs", 0.95), ("sign_flip", 0.0))
    n_trials: int = 4000
    fault_seed_branch: int = 9

    # sign-flip phase boundary (spec section 1)
    flip_sigma_grid: tuple = (0.005, 0.01, 0.02, 0.04)
    n_arrays_flip: int = 1600
    flip_seed_branch: int = 10
    kappa: float = 0.5  # budget share of FPR0 allowed to flip contamination
    n_tc_grid: int = 61  # log-spaced model grid for the t_c solve

    # detection budget language (inherited from E3)
    fpr0: float = 0.01
    n_star_primary: int = 20

    # pre-registered tolerances (spec section 6)
    tol_p1_abs: float = 0.05
    tol_pflip_rel: float = 0.25  # descriptive only; the gate is the z test
    z_gate: float = 2.5
    min_events: int = 20
    mc_tol_rel: float = 0.02
    mc_seed: int = 20260830

    # shift-identity audit
    audit_samples: int = 3
    audit_tol: float = 1e-9
