"""Crossbar state container and batched MVM.

Geometry convention (spec section 2, frozen): rows are wordlines carrying the
input vector x, columns are bitlines producing outputs y_j = sum_i G_ij x_i.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .drift_model import conductance_at

__all__ = ["RefCell", "CrossbarState", "matmul"]


@dataclass
class RefCell:
    """A single reference PCM cell (drifts like any other cell)."""

    g0: float
    nu: float

    def g_at(self, t_rel: float) -> float:
        return float(conductance_at(self.g0, self.nu, t_rel))


@dataclass
class CrossbarState:
    """Programmed physical state of one array instance (post programming spread).

    Fields
    ------
    g0, nu             [R, P] magnitude (physical) cells
    ref_ratio          reference cell for ratio-readout normalization
    ref_sign           reference cell for sign resolution (encoding c), else None
    sign_g0, sign_nu   [R, L] sign cells (encoding c), else None; L = logical columns
    sign_true          [R, L] programmed signs in {+1, -1} (encoding c), else None
    W                  [R, C] logical weight matrix (provenance copy; not consumed
                       by any E1 metric)
    """

    g0: np.ndarray
    nu: np.ndarray
    ref_ratio: RefCell
    ref_sign: Optional[RefCell] = None
    sign_g0: Optional[np.ndarray] = None
    sign_nu: Optional[np.ndarray] = None
    sign_true: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None


def matmul(X: np.ndarray, M: np.ndarray, backend: str = "numpy") -> np.ndarray:
    """Batched MVM: Y = X @ M with X [N, R], M [R, P] -> Y [N, P].

    backend 'torch' uses PyTorch (CUDA if available) and silently falls back to
    numpy when torch is not importable; at E1 problem sizes numpy is entirely
    sufficient (spec section 6 runtime estimate).
    """
    if backend == "torch":
        try:
            import torch  # local import: optional dependency

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            Xt = torch.as_tensor(X, device=dev)
            Mt = torch.as_tensor(M, device=dev)
            return (Xt @ Mt).cpu().numpy()
        except ImportError:
            pass
    return X @ M
