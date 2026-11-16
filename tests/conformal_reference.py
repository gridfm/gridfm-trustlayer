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


def weighted_conformal_offsets(
    weights: np.ndarray,
    residuals: np.ndarray,
    valid: np.ndarray,
    level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted split-conformal bounds with a point mass for the test residual.

    The full-scan definition used by kernel-weighted CP before the tail-only
    search, in float64: for each test row and element, sort the calibration
    residuals, accumulate the kernel weights of the valid ones, and take

    - upper: the first sorted residual whose cumulative weight reaches
      `level * (total + 1)`;
    - lower: the last sorted residual whose weight from there upward reaches it.

    Args:
        weights: `(n_test, n_calibration)` kernel weights.
        residuals: `(n_calibration, n_elements)` calibration residuals.
        valid: `(n_calibration, n_elements)` boolean mask of usable residuals.
        level: `1 - alpha / 2`.

    Returns:
        `(lower, upper, reached)`, each `(n_test, n_elements)`; where `reached` is
        False the level cannot be attained and the bounds are undefined (NaN).
    """

    order = np.argsort(residuals, axis=0, kind="stable")
    sorted_residuals = np.take_along_axis(residuals, order, axis=0).astype(np.float64)
    sorted_valid = np.take_along_axis(valid, order, axis=0).astype(np.float64)
    permuted = np.asarray(weights, dtype=np.float64)[:, order] * sorted_valid
    cumulative = np.cumsum(permuted, axis=1)
    total = cumulative[:, -1, :]
    threshold = level * (total + 1.0)
    reached = total >= threshold
    upper_index = np.argmax(cumulative >= threshold[:, None, :], axis=1)
    suffix = total[:, None, :] - cumulative + permuted
    lower_index = np.clip((suffix >= threshold[:, None, :]).sum(axis=1) - 1, 0, None)
    element = np.arange(residuals.shape[1])
    upper = np.where(reached, sorted_residuals[upper_index, element], np.nan)
    lower = np.where(reached, sorted_residuals[lower_index, element], np.nan)
    return lower, upper, reached


def pooled_embeddings_frame_reference(
    path: str, pooling: str, scenario_ids: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Scenario embeddings pooled through one pandas frame, as in 0.1.1.

    Reads every column at once, sorts the rows by (scenario, bus), reshapes to
    `(n_scenarios, n_buses, d)` and applies the pooling: the straightforward
    version that the column-by-column loader must reproduce bit for bit.
    """

    import pandas as pd  # type: ignore[import-untyped]

    frame = pd.read_parquet(path)
    if scenario_ids is not None:
        frame = frame[frame["scenario"].isin(np.asarray(scenario_ids))]
    columns = sorted(c for c in frame.columns if str(c).startswith("emb_"))
    frame = frame[["scenario", "bus", *columns]].sort_values(
        ["scenario", "bus"], kind="stable"
    )
    scenarios, counts = np.unique(frame["scenario"].to_numpy(), return_counts=True)
    values = frame[columns].to_numpy(dtype=np.float32, copy=False)
    stacked = values.reshape(len(scenarios), int(counts[0]), len(columns))
    pooled = {
        "mean": lambda: stacked.mean(axis=1),
        "max": lambda: stacked.max(axis=1),
        "meanmax": lambda: np.concatenate(
            [stacked.mean(axis=1), stacked.max(axis=1)], axis=1
        ),
        "concat": lambda: stacked.reshape(stacked.shape[0], -1),
    }[pooling]()
    return scenarios, np.ascontiguousarray(pooled, dtype=np.float32)
