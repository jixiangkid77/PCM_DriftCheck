"""pcm_drift: shared simulation library for the PCM_DriftCheck (D2) experiment series.

Instrument layer of the two-layer architecture defined in
docs/E1_Design_Spec_2026-08-22.md, section 7: all experiments E1..E7 import
this single package so that every reported number originates from one verified
code path.

Modules
-------
drift_model : per-cell power-law conductance drift, nu sampling (nu-g0 correlation)
crossbar    : array state container and batched matrix-vector products
encodings   : bias / differential / sign-magnitude builders, effective readout matrices
checksum    : checksum residual computation (the decision quantity r_k)
faults      : fault injection (sa0 / sa1 / soft deviation)
metrics     : threshold calibration, FPR/TPR, Wilson CI, quantiles, late-slope fit
io_utils    : runstamp directories, JST logging, config snapshots
model       : E2 analytical residual-moment model (quadrature-exact)

All numerical parameter defaults live in experiments/E1_config.py (single
source of truth); this package holds mechanisms only.
"""

__version__ = "0.3.2"

from . import drift_model, crossbar, encodings, checksum, faults, metrics, io_utils, model

__all__ = [
    "drift_model",
    "crossbar",
    "encodings",
    "checksum",
    "faults",
    "metrics",
    "io_utils",
    "model",
]
