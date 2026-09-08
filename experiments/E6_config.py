"""E6 parameter single source of truth (spec: docs/E6_Design_Spec_2026-08-30.md, frozen).

Hardware realization of the drift-aware band: exact fixed-point checker
datapath (the only approximation is the f(t) representation, spec section 1
decision 1), PWL band evaluator sized by the Prop 3 error budget, bit-true
FPR backtest on the E3 ensemble slice, and sky130 synthesis of both variants.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["E6Config"]


@dataclass(frozen=True)
class E6Config:
    # slice (encoding b, deployed band; E3 seed coordinates)
    sigma_nu: float = 0.02
    m: int = 16
    c: int = 1

    # digitization axis (Prop 3); FS convention follows E4: twice the maximum
    # absolute t0 output over the ensemble, per bit-width
    by_grid: tuple = (6, 8, 10)
    by_main: int = 8

    # error budget (Prop 3(ii)); kappa_p mirrors the phase-lemma share language
    fpr0: float = 0.01
    kappa_p: float = 0.5          # |dFPR| <= kappa_p * FPR0
    eps_f_target: float = 0.02    # engineering target, 3x margin under budget

    # PWL sizing (self-contained variant, main bit-width)
    n_seg_log: int = 5            # 32 segments uniform in L (16 missed the 2% target, see T1 verdict)
    u_bits: int = 8
    f_frac_bits: int = 3          # threshold stored in 1/8 code units

    # phases
    n_vectors: int = 4000         # Phase F vectors per testbench
    n_trials_spot: int = 4000     # Phase FPR p1 spot checks per class
    spot_classes: tuple = (("sign_flip", 0.0), ("soft_abs", 0.95))
    vec_seed: int = 20260830
    tol_fpr_abs: float = 0.005    # kappa_p * FPR0
    gate_by: tuple = (8, 10)      # hard-gate axis; 6 bit is characterization (spec v2)
    tol_p1_abs: float = 0.02  # analysis-layer gate, b_y = 10 only (spec v2, sec 5); 6/8-bit rows are T3 characterization

    # synthesis variants: (BY, LOGM, CSC) for the minimal checker
    synth_variants: tuple = ((8, 4, 1), (8, 3, 1), (8, 5, 1),
                             (6, 4, 1), (10, 4, 1), (8, 4, 3))
    liberty: str = "synth/sky130_fd_sc_hd__tt_025C_1v80.lib"
