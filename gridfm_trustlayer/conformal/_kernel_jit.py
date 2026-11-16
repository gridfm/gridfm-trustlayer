"""Compiled inner loops of the kernel calibrator (numba).

Each function handles one block of test scenarios, releases the GIL and uses no
numba threading: `KernelIntervalCalibrator` spreads the blocks over its own
thread pool, which works with every numba threading layer and never nests
parallel regions. `cache=True` stores the compiled code next to this module (or
in numba's user cache directory), so it is compiled once per installation.

The functions follow the numpy definitions in `kernel.py` operation by
operation, so that they give the same results: see `solve_bandwidths` and
`tail_bounds` for the summation orders.
"""

from __future__ import annotations

import numba  # type: ignore[import-untyped]
import numpy as np

# Positions summed together in the first level of the tail scan; the two-level
# order of these sums is part of the result (see `_first_crossing`).
SCAN_BLOCK = 16


@numba.njit(nogil=True, cache=True)
def row_extremes(distances, minimum, nearest, farthest):  # pragma: no cover - compiled
    """Per row: smallest distance, smallest positive distance (inf if none), largest."""

    n_rows, n_columns = distances.shape
    for row in range(n_rows):
        low = distances[row, 0]
        high = distances[row, 0]
        positive = np.float32(np.inf)
        for column in range(n_columns):
            value = distances[row, column]
            low = min(low, value)
            high = max(high, value)
            if value > 0.0 and value < positive:
                positive = value
        minimum[row] = low
        nearest[row] = positive
        farthest[row] = high


@numba.njit(nogil=True, cache=True)
def _n_eff_and_slope(row, row_minimum, bandwidth):  # pragma: no cover - compiled
    """`n_eff` and `d log n_eff / d log h` of one row (see `kernel._effective_sample_size_and_slope`)."""

    weight_sum = 0.0
    squared_sum = 0.0
    weighted_scaled = 0.0
    squared_scaled = 0.0
    for column in range(row.shape[0]):
        # float32 difference, then float64 division, as the numpy version.
        scaled = np.float64(row[column] - row_minimum) / bandwidth
        weight = np.exp(-scaled)
        squared = weight * weight
        weight_sum += weight
        squared_sum += squared
        weighted_scaled += weight * scaled
        squared_scaled += squared * scaled
    n_eff = weight_sum * weight_sum / squared_sum
    slope = 2.0 * weighted_scaled / weight_sum - 2.0 * squared_scaled / squared_sum
    return n_eff, slope


@numba.njit(nogil=True, cache=True)
def solve_bandwidths(  # pragma: no cover - compiled
    distances,
    row_minimum,
    log_lower,
    log_upper,
    target,
    rtol,
    newton_iterations,
    max_iterations,
    bandwidth_out,
    n_eff_out,
):
    """Per-row bandwidth whose kernel weights reach `target` n_eff.

    Newton steps on `log n_eff` against `log h`, starting at the middle of the
    bracket `[log_lower, log_upper]`, which every evaluation narrows; a step that
    would leave the bracket is replaced by bisection. After `newton_iterations`
    unconverged iterations only bisection is used: where `n_eff` is flat at one
    end of the bracket and steep at the other, Newton steps can alternate
    between the two ends and never converge.
    """

    log_target = np.log(target)
    for row in range(distances.shape[0]):
        values = distances[row]
        lower = log_lower[row]
        upper = log_upper[row]
        point = 0.5 * (lower + upper)
        for iteration in range(max_iterations):
            bandwidth = np.exp(point)
            n_eff, slope = _n_eff_and_slope(values, row_minimum[row], bandwidth)
            bandwidth_out[row] = bandwidth
            n_eff_out[row] = n_eff
            if abs(n_eff - target) <= rtol * target:
                break
            if n_eff < target:
                lower = point
            else:
                upper = point
            step = np.nan
            if iteration < newton_iterations and slope != 0.0:
                step = point - (np.log(n_eff) - log_target) / slope
            if np.isfinite(step) and lower < step < upper:
                point = step
            else:
                point = 0.5 * (lower + upper)


@numba.njit(nogil=True, cache=True)
def _first_crossing(weights, order, slack, from_top):  # pragma: no cover - compiled
    """First scan position at which the accumulated weight exceeds `slack`.

    `order` lists calibration indices in sorted order (an invalid entry points
    at the zero weight `weights[-1]`); the scan runs from the top or from the
    bottom. The sums follow the numpy scan of 0.1.1 exactly: float64 sums of
    blocks of `SCAN_BLOCK` positions, accumulated block by block, and inside
    the block that crosses, a running sum added to the total before it.
    """

    n = order.shape[0]
    before = 0.0
    start = 0
    while start < n:
        block = 0.0
        for offset in range(SCAN_BLOCK):
            position = start + offset
            if position < n:
                index = order[n - 1 - position] if from_top else order[position]
                block += np.float64(weights[index])
        if before + block > slack:
            running = 0.0
            for offset in range(SCAN_BLOCK):
                position = start + offset
                if position < n:
                    index = order[n - 1 - position] if from_top else order[position]
                    running += np.float64(weights[index])
                if before + running > slack:
                    return position
            return min(start + SCAN_BLOCK, n) - 1
        before = before + block
        start += SCAN_BLOCK
    return n - 1


@numba.njit(nogil=True, cache=True)
def tail_bounds(  # pragma: no cover - compiled
    weights, slack, reached, order, sorted_residuals, lower_out, upper_out
):
    """Weighted conformal bounds of every reached (row, element).

    `weights` is `(n_rows, n_calibration)` float32, `slack` and `reached` are
    `(n_rows, n_elements)`, `order` and `sorted_residuals` are
    `(n_elements, n_calibration)`: per element, the calibration indices in
    sorted order (invalid ones pointing at `n_calibration`) and the sorted
    residuals. The upper bound is the residual at the first position from the
    top whose accumulated weight exceeds the slack, the lower bound mirrors it
    from the bottom. Unreached entries are left untouched.
    """

    n_rows, n_calibration = weights.shape
    n_elements = order.shape[0]
    row_weights = np.empty(n_calibration + 1, dtype=np.float32)
    for row in range(n_rows):
        for column in range(n_calibration):
            row_weights[column] = weights[row, column]
        row_weights[n_calibration] = 0.0
        for element in range(n_elements):
            if not reached[row, element]:
                continue
            limit = slack[row, element]
            above = _first_crossing(row_weights, order[element], limit, True)
            upper_out[row, element] = sorted_residuals[
                element, n_calibration - 1 - above
            ]
            below = _first_crossing(row_weights, order[element], limit, False)
            lower_out[row, element] = sorted_residuals[element, below]
