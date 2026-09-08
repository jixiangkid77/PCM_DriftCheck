"""E4 parameter single source of truth (spec: docs/E4_Design_Spec_2026-08-30.md).

E4 compares three concurrent-detection decision rules on the E3 ensembles
(seed-identical rebuild): the exact-equality baseline at b-bit output
quantization (Parrini family, theta -> 0), the constant-tolerance baseline
(X-ABFT family, rho == 1), and the drift-aware band f(t) = theta0 * rho(t).
Physics comes from E1Config; ensemble seed coordinates come from E3Config so
detector C reproduces the E3 official record bit for bit (spec gate Q3).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["E4Config"]


@dataclass(frozen=True)
class E4Config:
    # comparison slice (encoding b, m = 16, c = 1 fixed by spec section 2)
    sigma_nu_on: tuple = (0.005, 0.01, 0.02, 0.04)  # ratio_norm on panels
    sigma_off_slice: float = 0.02                   # ratio_norm off panel
    m: int = 16
    c: int = 1

    # baseline A: ADC bit-width family; full scale from the t0 healthy output
    # span of all logical columns (most favorable calibration for the baseline)
    adc_bits: tuple = (6, 8, 10)

    # discrimination probes (persistent single-cell soft_abs faults)
    probe_deltas: tuple = (0.3, 0.95)
    probe_t: tuple = (1.0, 100.0, 10000.0)
    n_trials: int = 4000
    fault_seed_branch: int = 8  # SeedSequence([base, 8, ...]); base from E3Config

    # detector C regression gate (spec Q3): bitwise equality against the E3
    # official record is asserted downstream in the analysis, not here; the
    # script logs C rows at full precision for that comparison.
    fpr0: float = 0.01
