"""Checksum residuals: the decision quantity of spec section 2.

    r_k(x, t) = y_ck(t) - lam * sum_{j in S_k} y_j(t)

computed on logical column outputs Y = X @ M (see encodings.effective_matrix).
Units stay on the checksum-column output side; no 1/lam amplification (frozen
in spec section 2).
"""

from __future__ import annotations

import numpy as np

from .encodings import ColumnLayout

__all__ = ["residuals", "group_flip_flags"]


def residuals(Y: np.ndarray, layout: ColumnLayout) -> np.ndarray:
    """Y [N, C+K] logical outputs -> residuals r [N, K].

    Relies on the frozen logical order [data 0..C-1 | checksum 0..K-1] with
    contiguous m-column groups.
    """
    N = Y.shape[0]
    C, K, m = layout.C, layout.K, layout.m
    Yd = Y[:, :C].reshape(N, K, m)
    return Y[:, C : C + K] - layout.lam * Yd.sum(axis=2)


def group_flip_flags(flips: np.ndarray, layout: ColumnLayout) -> np.ndarray:
    """Per-group sign-flip flags (encoding c; input-independent).

    flips: bool [R, C+K] from encodings.effective_matrix info. Returns bool [K]:
    True when any sign cell among the group's data columns or its checksum
    column has flipped.
    """
    C, K, m = layout.C, layout.K, layout.m
    data_any = flips[:, :C].reshape(-1, K, m).any(axis=(0, 2))
    ck_any = flips[:, C : C + K].any(axis=0)
    return data_any | ck_any
