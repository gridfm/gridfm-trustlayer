"""Coverage and screening reports from Python, the tables of `calibrate` and `screen`.

`coverage_report` gives the coverage and width of the intervals per variable
and stratum (the CLI's `coverage_by_stratum.csv`), or per element.
`screening_report` gives, against operating limits, the alarms caught, missed
and false (with true values), the status counts, and the status of every
scenario (the CLI's `status_by_stratum.csv` and `scenarios.csv`).

Both take `ChannelSet`s (one, or several such as buses and lines) and the
intervals from `predict_intervals`; values marked invalid in the predictions or
the true values are left out. Unlike the CLI, which widens the limits by 1e-6
to absorb rounding, the limits are used as given (as in `classify_status`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.conformal.calibrator import ChannelSet, _limits
from gridfm_trustlayer.screening import ELEMENT_STATUSES, status_codes
from gridfm_trustlayer.strata import OUTAGE_TYPES

Intervals = Mapping[str, tuple[np.ndarray, np.ndarray]]
Sets = ChannelSet | Sequence[ChannelSet]


def coverage_report(
    predictions: Sets,
    true_values: Sets,
    intervals: Intervals,
    *,
    alpha: float,
    strat: str | None = None,
    by_element: bool = False,
) -> pd.DataFrame:
    """Coverage and width of the intervals, overall and per stratum.

    Args:
        predictions: The predictions the intervals were computed for (their
            valid masks and, with `strat`, the stratum column are used), or a
            list of `ChannelSet`s such as `[buses, lines]`.
        true_values: The true values of the same scenarios and elements (a
            list in the same order when `predictions` is one).
        intervals: `{variable: (lower, upper)}` from `predict_intervals`, for
            any of the variables.
        alpha: The calibrator's `alpha`; the target coverage is `1 - alpha`.
        strat: A stratum column of `predictions` to report per stratum.
        by_element: One row per element (and stratum) instead of per stratum.

    Returns:
        Per variable and stratum (`all` first): `n_scenarios`, `n_elements`,
        `n_values` (counted), `n_excluded` (marked invalid),
        `target_coverage`, `coverage` with an approximate 95% interval
        (`coverage_low`, `coverage_high`; scenarios as independent draws),
        `worst_element_coverage` (elements with enough values),
        `mean_width` (values with finite bounds) and `n_elements_infinite`.
        With `by_element`: per element, `n_values`, `coverage`,
        `mean_width`, `min_width` and `max_width`.
    """

    from gridfm_trustlayer.cli import reports

    holders = _holders(predictions, true_values, intervals, "intervals")
    records = []
    for name, (lower, upper) in intervals.items():
        data, truth_set = holders[name]
        assert truth_set is not None
        valid, truth = _valid_truth(data, truth_set, name)
        lower, upper = _bounds(lower, upper, truth.shape, name)
        covered = (lower <= truth) & (truth <= upper)
        width = np.where(np.isnan(lower) | np.isnan(upper), np.nan, upper - lower)
        for stratification, value, label, rows in _scopes(data, strat):
            for column, element in enumerate(np.asarray(data.entity_ids)):
                use = valid[rows, column]
                widths = width[rows, column][use]
                widths = widths[~np.isnan(widths)]
                finite = widths[np.isfinite(widths)]
                n_values = int(use.sum())
                records.append(
                    {
                        "variable": name,
                        "stratification": stratification,
                        "stratum_value": value,
                        "stratum": label,
                        "element": int(element),
                        "n_values": n_values,
                        "n_known": int((~use).sum()),
                        "n_pooled": 0,
                        "n_scenarios": n_values,
                        "coverage": covered[rows, column][use].mean()
                        if n_values
                        else math.nan,
                        "mean_width": widths.mean() if widths.size else math.nan,
                        "finite_width_sum": float(finite.sum()),
                        "finite_width_count": int(finite.size),
                        "min_width": widths.min() if widths.size else math.nan,
                        "max_width": widths.max() if widths.size else math.nan,
                    }
                )
    elements = pd.DataFrame.from_records(records)
    if by_element:
        table = reports.sort_by_stratum(elements)
        return table.loc[
            :,
            [
                "variable",
                "stratum",
                "element",
                "n_values",
                "coverage",
                "mean_width",
                "min_width",
                "max_width",
            ],
        ].reset_index(drop=True)
    by_stratum = reports.coverage_by_stratum(elements, alpha)
    by_stratum = by_stratum.rename(columns={"n_known": "n_excluded"})
    return by_stratum.drop(
        columns=["stratification", "stratum_value", "n_pooled"]
    ).reset_index(drop=True)


@dataclass
class ScreeningReport:
    """The tables of `screening_report`.

    Attributes:
        alarms: With true values, per variable, stratum and method
            (`interval`, or `point` for the prediction alone): `n_values`,
            `violations` (true values beyond a limit), `alarms` (a bound, or
            the point, beyond a limit), `caught`, `missed`, `false_alarms`,
            `cleared`, `recall` and `precision`. None without true values.
        statuses: Counts per level (`scenario`, `element`), variable (`all`
            for every variable together), stratum, method and status, with
            `truly_violated` when the true values are known.
        scenarios: One row per scenario: its interval and point status,
            overall and per variable, the counts of uncertain values and
            violations, and the truth when known.
    """

    alarms: pd.DataFrame | None
    statuses: pd.DataFrame
    scenarios: pd.DataFrame


def screening_report(
    predictions: Sets,
    intervals: Intervals,
    limits: Mapping[str, Mapping[str, float]],
    *,
    true_values: Sets | None = None,
    strat: str | None = None,
) -> ScreeningReport:
    """Screening against operating limits, as `screen` (with true values) and `apply`.

    A value is `within_limits` when its whole interval is inside the limits,
    a `violation` when it is entirely beyond one, `uncertain` otherwise; a
    scenario takes the worst status of its values (`secure`, `uncertain`,
    `insecure`). An alarm is raised when a bound crosses a limit.

    Pass buses and lines together (`[buses, lines]`) for the scenario status
    to cover every screened variable.

    Args:
        predictions: The predictions the intervals were computed for, or a
            list of `ChannelSet`s such as `[buses, lines]`.
        intervals: `{variable: (lower, upper)}` from `predict_intervals`.
        limits: Per variable to screen, e.g. `{"line_loading": {"upper": 1.0},
            "Vm": {"lower": 0.95, "upper": 1.05}}`.
        true_values: The true values, to count caught, missed and false alarms
            (a list in the same order when `predictions` is one).
        strat: A stratum column of `predictions` to report per stratum.

    Returns:
        A `ScreeningReport` with `alarms`, `statuses` and `scenarios`.
    """

    from gridfm_trustlayer.cli import status

    unknown = sorted(set(limits) - set(intervals))
    if unknown:
        raise ValueError(f"limits name variables without intervals: {unknown}.")
    holders = _holders(predictions, true_values, limits, "limits")
    truth_known = true_values is not None
    strat_columns = [] if strat is None else [strat]
    values: list[pd.DataFrame] = []
    alarm_rows: list[dict[str, object]] = []
    for name, bounds in limits.items():
        data, truth_set = holders[name]
        limit = _limits(bounds, name)
        low = limit.get("lower_limit", math.nan)
        high = limit.get("upper_limit", math.nan)
        point = np.asarray(data.channels[name], dtype=float)
        lower, upper = _bounds(*intervals[name], point.shape, name)
        if truth_set is None:
            valid = _valid(data, name)
            truth = None
        else:
            valid, truth = _valid_truth(data, truth_set, name)
        scopes = _scopes(data, strat)
        scenario_ids = np.asarray(data.scenario_ids)
        interval_codes = status_codes(lower, upper, low, high)
        point_codes = status_codes(point, point, low, high)
        rows, columns = np.nonzero(valid)
        frame = pd.DataFrame(
            {
                "scenario": scenario_ids[rows],
                "variable": name,
                "status": pd.Categorical.from_codes(
                    interval_codes[rows, columns], ELEMENT_STATUSES
                ),
                "point_status": pd.Categorical.from_codes(
                    point_codes[rows, columns], ELEMENT_STATUSES
                ),
            }
        )
        if strat is not None:
            frame[strat] = np.asarray(data.scenario_columns[strat])[rows]
        if truth is not None:
            beyond = np.zeros(truth.shape, dtype=bool)
            if not math.isnan(high):
                beyond |= truth > high
            if not math.isnan(low):
                beyond |= truth < low
            frame["truth"] = beyond[rows, columns]
            for method, codes in (("interval", interval_codes), ("point", point_codes)):
                alarm = codes > 0
                for _, _, label, scope in scopes:
                    use = valid & scope[:, None]
                    alarm_rows.append(
                        _alarm_counts(name, label, method, alarm[use], beyond[use])
                    )
        values.append(frame)
    combined = pd.concat(values, ignore_index=True)
    label_of = _label_function(strat)
    element_counts = status.element_counts(
        combined, strat_columns, label=label_of, truth=truth_known
    )
    table = status.scenario_table(
        [status.scenario_rows(combined, strat_columns, truth=truth_known)],
        truth=truth_known,
    )
    scenario_counts = status.scenario_counts(
        table, strat_columns, label=label_of, truth=truth_known
    )
    counts = status.combine_counts([scenario_counts, element_counts], truth=truth_known)
    return ScreeningReport(
        alarms=pd.DataFrame.from_records(alarm_rows) if truth_known else None,
        statuses=counts.drop(columns=["stratification", "stratum_value"]),
        scenarios=status.labelled_scenarios(table, truth=truth_known),
    )


def _alarm_counts(
    variable: str, stratum: str, method: str, alarm: np.ndarray, violation: np.ndarray
) -> dict[str, object]:
    caught = int((alarm & violation).sum())
    missed = int((~alarm & violation).sum())
    false_alarms = int((alarm & ~violation).sum())
    violations, alarms = caught + missed, caught + false_alarms
    return {
        "variable": variable,
        "stratum": stratum,
        "method": method,
        "n_values": int(alarm.size),
        "violations": violations,
        "alarms": alarms,
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "cleared": int((~alarm & ~violation).sum()),
        "recall": caught / violations if violations else math.nan,
        "precision": caught / alarms if alarms else math.nan,
    }


def _holders(
    predictions: Sets,
    true_values: Sets | None,
    variables: Mapping[str, object],
    what: str,
) -> dict[str, tuple[ChannelSet, ChannelSet | None]]:
    """The predictions and true values holding each variable of `variables`."""

    sets = [predictions] if isinstance(predictions, ChannelSet) else list(predictions)
    if true_values is None:
        truths: list[ChannelSet | None] = [None] * len(sets)
    else:
        truths = (
            [true_values] if isinstance(true_values, ChannelSet) else list(true_values)
        )
        if len(truths) != len(sets):
            raise ValueError(
                f"Got {len(sets)} prediction sets and {len(truths)} true-value sets; "
                "pass them in pairs, in the same order."
            )
    holders: dict[str, tuple[ChannelSet, ChannelSet | None]] = {}
    for data, truth in zip(sets, truths, strict=True):
        if truth is not None:
            _check_pair(data, truth)
        for name in data.channels:
            if name in holders:
                raise ValueError(
                    f"The variable {name!r} is in more than one prediction set."
                )
            if truth is not None and name not in truth.channels:
                raise ValueError(f"true_values has no variable {name!r}.")
            holders[name] = (data, truth)
    missing = sorted(set(variables) - set(holders))
    if missing:
        raise ValueError(
            f"{what} name variables that are not in the predictions: {missing}."
        )
    return holders


def _bounds(
    lower: np.ndarray, upper: np.ndarray, shape: tuple[int, ...], name: str
) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    for bound in bounds:
        if bound.shape != shape:
            raise ValueError(
                f"The intervals of {name!r} have shape {bound.shape}; the "
                f"predictions {shape}. Use the intervals predicted for these "
                "predictions."
            )
    return bounds


def _check_pair(predictions: ChannelSet, true_values: ChannelSet) -> None:
    if not (
        np.array_equal(predictions.scenario_ids, true_values.scenario_ids)
        and np.array_equal(predictions.entity_ids, true_values.entity_ids)
    ):
        raise ValueError(
            "predictions and true_values must have the same scenarios and elements, "
            "in the same order (check_inputs says what differs)."
        )


def _valid(data: ChannelSet, name: str) -> np.ndarray:
    mask = data.valid_masks.get(name)
    shape = np.shape(data.channels[name])
    return np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


def _valid_truth(
    predictions: ChannelSet, true_values: ChannelSet, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Values counted (valid in both, with a finite true value), and the true values."""

    truth = np.asarray(true_values.channels[name], dtype=float)
    valid = _valid(predictions, name) & _valid(true_values, name) & np.isfinite(truth)
    return valid, truth


def _scopes(
    predictions: ChannelSet, strat: str | None
) -> list[tuple[str, float, str, np.ndarray]]:
    """`(stratification, value, label, scenario mask)`: overall, then each stratum."""

    n = len(predictions.scenario_ids)
    scopes = [("", math.nan, "all", np.ones(n, dtype=bool))]
    if strat is None:
        return scopes
    if strat not in predictions.scenario_columns:
        raise ValueError(
            f"predictions has no stratum column {strat!r} (it has "
            f"{sorted(predictions.scenario_columns)})."
        )
    values = np.asarray(predictions.scenario_columns[strat])
    label = _label_function(strat)
    for value in np.unique(values):
        scopes.append(
            (strat, float(value), label(pd.DataFrame(), strat, value), values == value)
        )
    return scopes


def _label_function(strat: str | None) -> Callable[[pd.DataFrame, str, object], str]:
    """Stratum labels as in the CLI reports: `n_outages=2`, `outage_type=mixed`."""

    def label(_: pd.DataFrame, stratification: str, value: object) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "all"
        number = int(float(str(value)))
        if stratification == "outage_type" and 0 <= number < len(OUTAGE_TYPES):
            return f"{stratification}={OUTAGE_TYPES[number]}"
        return f"{stratification}={number}"

    return label
