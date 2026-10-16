"""Status of predicted values and operating scenarios against operating limits.

A value is `within_limits` when its whole interval lies inside the limits,
`violation` when its whole interval lies beyond a limit, and `uncertain` when
the interval straddles a limit. An operating scenario takes the worst status of
its values: `secure`, `uncertain` or `insecure`. Passing the point prediction
as both bounds classifies the prediction alone, which is then only ever
`within_limits` or `violation`.

The CLI commands `screen` and `apply` use these functions; they are public so
that the same rule can be applied to the intervals of the Python API.
"""

from __future__ import annotations

import math

import numpy as np

ELEMENT_STATUSES = ("within_limits", "uncertain", "violation")
SCENARIO_STATUSES = ("secure", "uncertain", "insecure")


def status_codes(
    lower: np.ndarray,
    upper: np.ndarray,
    lower_limit: float,
    upper_limit: float,
) -> np.ndarray:
    """0 within limits, 1 uncertain, 2 violation; a NaN limit is ignored.

    The limits are compared strictly: a bound equal to the limit does not
    cross it. A missing (NaN) bound is uncertain: nothing shows the value is
    within the limits.
    """

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    beyond = np.zeros(np.broadcast(lower, upper).shape, dtype=bool)
    crosses = np.zeros_like(beyond)
    if not math.isnan(upper_limit):
        beyond |= lower > upper_limit
        crosses |= upper > upper_limit
    if not math.isnan(lower_limit):
        beyond |= upper < lower_limit
        crosses |= lower < lower_limit
    crosses |= np.isnan(lower) | np.isnan(upper)
    return np.where(beyond, 2, np.where(crosses, 1, 0)).astype(np.int8)


def classify_status(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    upper_limit: float | None = None,
    lower_limit: float | None = None,
) -> np.ndarray:
    """Status of each value: `within_limits`, `uncertain` or `violation`.

    Args:
        lower: Lower bounds, any shape (e.g. scenarios x elements).
        upper: Upper bounds, the same shape.
        upper_limit: Upper operating limit, e.g. `1.0` for line loading.
        lower_limit: Lower operating limit, e.g. `0.95` for `Vm`.

    Returns:
        An array of status names of the bounds' shape. Pass the point
        prediction as both `lower` and `upper` to classify it alone.

    Raises:
        ValueError: If neither limit is given.
    """

    if upper_limit is None and lower_limit is None:
        raise ValueError("Give upper_limit, lower_limit or both.")
    codes = status_codes(
        lower,
        upper,
        math.nan if lower_limit is None else float(lower_limit),
        math.nan if upper_limit is None else float(upper_limit),
    )
    return np.asarray(ELEMENT_STATUSES, dtype=object)[codes]


def scenario_status(
    statuses: np.ndarray,
    *,
    axis: int = -1,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Status of each operating scenario: the worst status of its values.

    Args:
        statuses: Value statuses from `classify_status`, e.g. scenarios x
            elements. Statuses of several variables can be concatenated along
            `axis` first.
        axis: The axis of the values within a scenario.
        valid: Optional mask of the same shape; `False` values (e.g. known
            set-points or out-of-service lines) are ignored. Missing statuses
            (`None` or NaN, as `apply` writes for those values) are ignored too.

    Returns:
        `secure`, `uncertain` or `insecure` per scenario. A scenario without
        any valid value is `secure`.
    """

    lookup = {name: code for code, name in enumerate(ELEMENT_STATUSES)}

    def code(status: object) -> int:
        # Anything but a status name (None, NaN, pandas NA) is a missing status.
        return lookup[status] if isinstance(status, str) else 0

    array = np.asarray(statuses, dtype=object)
    codes = (
        np.vectorize(code, otypes=[np.int8])(array)
        if array.size
        else np.zeros(array.shape, dtype=np.int8)
    )
    if valid is not None:
        codes = np.where(np.asarray(valid, dtype=bool), codes, 0)
    # A scenario with no values at all is secure, like one with no valid value.
    worst = codes.max(axis=axis, initial=0)
    return np.asarray(SCENARIO_STATUSES, dtype=object)[worst]
