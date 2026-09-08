"""Fault injection for Phase B (spec section 5).

Faults are applied at t0 and persist. Stuck cells stop drifting (nu set to 0);
soft-deviation cells keep their originally sampled nu: the deviation is a
post-programming disturbance and the exponent stays tied to the original
programming event (implementation-level decision, declared in the delivery
note). Encoding b faults hit one physical branch (pos or neg) of a random
logical data column; encoding c faults target magnitude cells only, sign-cell
fault classes belong to E5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crossbar import CrossbarState
from .encodings import ColumnLayout

__all__ = ["FaultSpec", "sample_fault", "inject", "faulted_cell"]


@dataclass
class FaultSpec:
    kind: str  # 'sa0' | 'sa1' | 'soft' | 'soft_abs'
    d: float  # soft deviation magnitude: relative for 'soft', absolute
    # conductance units for 'soft_abs' (ignored for sa0/sa1)
    row: int
    col: int  # logical data column index in [0, C)
    branch: str  # 'pos' | 'neg' for encoding b, '' otherwise
    soft_sign: int  # +1 / -1: direction of the soft deviation


def sample_fault(kind: str, d: float, layout: ColumnLayout, rng) -> FaultSpec:
    """Draw a random fault location (uniform over data cells)."""
    row = int(rng.integers(0, layout.R))
    col = int(rng.integers(0, layout.C))
    branch = ""
    if layout.encoding == "b":
        branch = "pos" if int(rng.integers(0, 2)) == 0 else "neg"
    soft_sign = 1 if int(rng.integers(0, 2)) == 0 else -1
    return FaultSpec(kind=kind, d=d, row=row, col=col, branch=branch, soft_sign=soft_sign)


def _phys_col(spec: FaultSpec, layout: ColumnLayout) -> int:
    if layout.encoding == "b":
        cols = layout.pos_cols if spec.branch == "pos" else layout.neg_cols
        return int(cols[spec.col])
    return spec.col  # encodings a and c: physical index = logical index for data columns


def inject(state: CrossbarState, layout: ColumnLayout, spec: FaultSpec, cfg):
    """Return (faulted copy of state, affected group index).

    Only the magnitude g0/nu arrays are copied; reference cells, sign cells and
    W are shared with the base state (they are never mutated here).
    """
    st = CrossbarState(
        g0=state.g0.copy(),
        nu=state.nu.copy(),
        ref_ratio=state.ref_ratio,
        ref_sign=state.ref_sign,
        sign_g0=state.sign_g0,
        sign_nu=state.sign_nu,
        sign_true=state.sign_true,
        W=state.W,
    )
    p = _phys_col(spec, layout)
    i = spec.row
    g_new, nu_new = faulted_cell(st.g0[i, p], st.nu[i, p], spec, cfg)
    st.g0[i, p] = g_new
    st.nu[i, p] = nu_new
    return st, spec.col // layout.m


def faulted_cell(g0: float, nu: float, spec: FaultSpec, cfg):
    """Single source of truth for the per-cell fault operation.

    Returns (g0_new, nu_new). 'sa0'/'sa1' freeze the cell at g_min/g_max with
    nu = 0; 'soft' applies a relative deviation g0*(1 +- d); 'soft_abs' applies
    an absolute conductance deviation g0 +- d (E3 primary axis, aligned with
    the perturbation-bound convention of the analog-ECC line). Both soft kinds
    keep the originally sampled nu (post-programming disturbance)."""
    if spec.kind == "sa0":
        return cfg.g_min, 0.0
    if spec.kind == "sa1":
        return cfg.g_max, 0.0
    if spec.kind == "soft":
        return max(g0 * (1.0 + spec.soft_sign * spec.d), cfg.g0_floor), nu
    if spec.kind == "soft_abs":
        return max(g0 + spec.soft_sign * spec.d, cfg.g0_floor), nu
    raise ValueError(f"unknown fault kind: {spec.kind}")
