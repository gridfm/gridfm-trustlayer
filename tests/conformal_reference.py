"""Reference split-conformal quantile used to build test expectations.

Kept deliberately simple (a full sort and an explicit rank) so that it reads as
the textbook definition rather than as a copy of the library implementation.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(values: np.ndarray, level: float) -> np.ndarray:
    """Split-conformal residual quantile along axis 0.

    Upper tail (`level >= 0.5`): the `ceil((n + 1) level)`-th smallest value,
    `+inf` if that rank exceeds `n`. Lower tail: the `floor((n + 1) level)`-th
    smallest value, `-inf` if that rank is below 1.
    """

    ordered = np.sort(np.asarray(values, dtype=np.float64), axis=0)
    n = ordered.shape[0]
    if level >= 0.5:
        rank = math.ceil((n + 1) * level - 1e-9)
        if rank > n:
            return np.full(ordered.shape[1:], np.inf, dtype=np.float32)
    else:
        rank = math.floor((n + 1) * level + 1e-9)
        if rank < 1:
            return np.full(ordered.shape[1:], -np.inf, dtype=np.float32)
    return ordered[rank - 1].astype(np.float32)
