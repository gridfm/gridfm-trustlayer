"""Results from Python: `ChannelSet.to_frame`, `coverage_report` and `screening_report`."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer import ChannelSet, coverage_report, screening_report

INF = np.inf


def _set(values: list[list[float]], *, name: str = "x", **kwargs: Any) -> ChannelSet:
    array = np.asarray(values, dtype=np.float32)
    defaults: dict[str, Any] = {
        "scenario_ids": np.arange(1, array.shape[0] + 1),
        "entity_ids": np.arange(10, 10 * array.shape[1] + 1, 10),
        "entity_name": "branch",
    }
    defaults.update(kwargs)
    return ChannelSet(channels={name: array}, **defaults)


# ------------------------------------------------------------------ coverage


def _coverage_data() -> tuple[ChannelSet, ChannelSet, dict[str, Any]]:
    """4 scenarios x 2 lines; one value invalid, one interval unbounded."""

    strata = {"k": np.array([0, 0, 1, 1])}
    valid = np.ones((4, 2), dtype=bool)
    valid[2, 1] = False
    predictions = _set(
        [[1.0, 0.5]] * 4, valid_masks={"x": valid}, scenario_columns=strata
    )
    true_values = _set(
        [[1.05, 0.5], [0.8, 0.55], [1.2, 0.4], [1.0, 0.5]], scenario_columns=strata
    )
    lower = np.array([[0.9, 0.4]] * 4)
    upper = np.array([[1.1, 0.6]] * 4)
    upper[3, 1] = INF
    return predictions, true_values, {"x": (lower, upper)}


def test_coverage_report_pools_the_valid_values_per_stratum() -> None:
    predictions, true_values, intervals = _coverage_data()

    table = coverage_report(predictions, true_values, intervals, alpha=0.1, strat="k")

    assert table["stratum"].tolist() == ["all", "k=0", "k=1"]
    row = table.set_index("stratum")
    # Covered: 5 of the 7 valid values (scenario 2 and 3 miss on line 10).
    assert row.loc["all", "coverage"] == pytest.approx(5 / 7)
    assert row.loc["k=0", "coverage"] == pytest.approx(3 / 4)
    assert row.loc["k=1", "coverage"] == pytest.approx(2 / 3)
    assert row["n_values"].tolist() == [7, 4, 3]
    assert row["n_excluded"].tolist() == [1, 0, 1]
    assert row["n_elements_infinite"].tolist() == [1, 0, 1]
    np.testing.assert_allclose(row["mean_width"], 0.2, rtol=1e-6)
    assert (row["target_coverage"] == 0.9).all()
    assert (row["coverage_low"] <= row["coverage"]).all()
    assert (row["coverage"] <= row["coverage_high"]).all()


def test_coverage_report_per_element() -> None:
    predictions, true_values, intervals = _coverage_data()

    table = coverage_report(
        predictions, true_values, intervals, alpha=0.1, by_element=True
    )

    assert table.columns.tolist() == [
        "variable",
        "stratum",
        "element",
        "n_values",
        "coverage",
        "mean_width",
        "min_width",
        "max_width",
    ]
    assert table["element"].tolist() == [10, 20]
    assert table["n_values"].tolist() == [4, 3]
    assert table["coverage"].tolist() == [0.5, 1.0]
    assert table["max_width"].tolist()[1] == INF


def test_coverage_report_leaves_out_invalid_and_unknown_true_values() -> None:
    predictions, true_values, intervals = _coverage_data()
    true_values.channels["x"][0, 0] = np.nan  # covered before
    true_values.valid_masks["x"] = np.ones((4, 2), dtype=bool)
    true_values.valid_masks["x"][1, 0] = False  # missed before

    table = coverage_report(predictions, true_values, intervals, alpha=0.1)

    assert table["n_values"].tolist() == [5]
    assert table["coverage"].tolist() == pytest.approx([4 / 5])


# ------------------------------------------------------------------ screening


def _screening_data() -> tuple[list[ChannelSet], list[ChannelSet], dict[str, Any]]:
    """Lines (`x`, upper limit 1.0) and one bus (`v`, lower limit 0.95)."""

    strata = {"k": np.array([0, 1, 1])}
    lines = (
        _set([[0.6, 0.95], [0.6, 1.2], [0.6, 0.7]], scenario_columns=strata),
        _set([[0.6, 1.05], [0.6, 0.98], [0.6, 0.7]], scenario_columns=strata),
    )
    buses = tuple(
        _set(
            values,
            name="v",
            entity_ids=np.array([1]),
            entity_name="bus",
            scenario_columns=strata,
        )
        for values in ([[1.0], [1.0], [0.97]], [[1.0], [1.0], [0.96]])
    )
    intervals = {
        # statuses: within + uncertain, within + violation, within + within
        "x": (
            np.array([[0.5, 0.9], [0.5, 1.1], [0.5, 0.6]]),
            np.array([[0.7, 1.1], [0.8, 1.3], [0.7, 0.8]]),
        ),
        # within, within, uncertain
        "v": (np.array([[0.98], [0.98], [0.94]]), np.array([[1.02], [1.02], [1.0]])),
    }
    return [lines[0], buses[0]], [lines[1], buses[1]], intervals


LIMITS = {"x": {"upper": 1.0}, "v": {"lower": 0.95}}


def _counts(statuses: pd.DataFrame, **where: str) -> dict[str, int]:
    rows = statuses
    for column, value in where.items():
        rows = rows[rows[column] == value]
    return dict(zip(rows["status"], rows["n"], strict=True))


def test_screening_report_counts_alarms_against_the_true_values() -> None:
    predictions, true_values, intervals = _screening_data()

    report = screening_report(
        predictions, intervals, LIMITS, true_values=true_values, strat="k"
    )

    assert report.alarms is not None
    alarms = report.alarms.set_index(["variable", "stratum", "method"])
    # Line 20 is truly overloaded in scenario 1 only; the intervals raise
    # alarms in scenarios 1 (uncertain) and 2 (violation), the point in 2.
    interval = alarms.loc[("x", "all", "interval")]
    assert (interval.violations, interval.alarms) == (1, 2)
    assert (interval.caught, interval.missed, interval.false_alarms) == (1, 0, 1)
    assert (interval.cleared, interval.recall, interval.precision) == (4, 1.0, 0.5)
    point = alarms.loc[("x", "all", "point")]
    assert (point.caught, point.missed, point.false_alarms) == (0, 1, 1)
    assert point.recall == 0.0
    assert alarms.loc[("x", "k=1", "interval")].tolist()[:3] == [4, 0, 1]
    voltage = alarms.loc[("v", "all", "interval")]
    assert (voltage.violations, voltage.false_alarms) == (0, 1)
    assert math.isnan(voltage.recall)


def test_screening_report_status_of_values_and_scenarios() -> None:
    predictions, true_values, intervals = _screening_data()

    report = screening_report(
        predictions, intervals, LIMITS, true_values=true_values, strat="k"
    )

    statuses = report.statuses
    common = {"stratum": "all", "method": "interval"}
    assert _counts(statuses, level="element", variable="x", **common) == {
        "within_limits": 4,
        "uncertain": 1,
        "violation": 1,
    }
    # Scenarios take the worst status over lines and buses.
    assert _counts(statuses, level="scenario", variable="all", **common) == {
        "secure": 0,
        "uncertain": 2,
        "insecure": 1,
    }
    assert _counts(
        statuses, level="scenario", variable="all", stratum="k=1", method="interval"
    ) == {"secure": 0, "uncertain": 1, "insecure": 1}

    scenarios = report.scenarios.set_index("scenario")
    assert scenarios["status"].tolist() == ["uncertain", "insecure", "uncertain"]
    assert scenarios["point_status"].tolist() == ["secure", "insecure", "secure"]
    assert scenarios["true_status"].tolist() == ["insecure", "secure", "secure"]
    assert scenarios["x_status"].tolist() == ["uncertain", "insecure", "secure"]
    assert scenarios["v_status"].tolist() == ["secure", "secure", "uncertain"]
    assert scenarios["k"].tolist() == [0, 1, 1]


def test_screening_report_without_true_values() -> None:
    predictions, _, intervals = _screening_data()

    report = screening_report(predictions, intervals, LIMITS)

    assert report.alarms is None
    assert "truly_violated" not in report.statuses.columns
    assert "true_status" not in report.scenarios.columns
    assert report.scenarios["status"].tolist() == ["uncertain", "insecure", "uncertain"]


def test_screening_report_of_one_set_screens_only_its_variables() -> None:
    predictions, true_values, intervals = _screening_data()

    report = screening_report(
        predictions[0],
        {"x": intervals["x"]},
        {"x": LIMITS["x"]},
        true_values=true_values[0],
    )

    assert report.scenarios["status"].tolist() == ["uncertain", "insecure", "secure"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda p, t, i, lim: (p, t, i, {**lim, "y": {"upper": 1}}),
            "without intervals",
        ),
        (
            lambda p, t, i, lim: (p, t, {**i, "y": i["x"]}, {**lim, "y": {"upper": 1}}),
            "not in the predictions",
        ),
        (lambda p, t, i, lim: (p, t[:1], i, lim), "pass them in pairs"),
        (lambda p, t, i, lim: ([p[0], p[0]], [t[0], t[0]], i, lim), "more than one"),
        (lambda p, t, i, lim: (p, t[::-1], i, lim), "same scenarios and elements"),
        (
            lambda p, t, i, lim: (p, t, {**i, "x": (i["x"][0].T, i["x"][1].T)}, lim),
            "have shape",
        ),
        (lambda p, t, i, lim: (p, t, i, {"x": {"max": 1.0}}), "take 'lower'"),
    ],
)
def test_screening_report_refuses_inputs_that_do_not_match(change, message) -> None:
    predictions, true_values, intervals, limits = change(*_screening_data(), LIMITS)

    with pytest.raises(ValueError, match=message):
        screening_report(predictions, intervals, limits, true_values=true_values)


# ------------------------------------------------------------------ to_frame


def test_to_frame_gives_one_row_per_scenario_and_element() -> None:
    predictions, _, intervals = _coverage_data()

    frame = predictions.to_frame(
        intervals, entity="line", limits={"x": {"upper": 1.05}}
    )

    assert frame.columns.tolist() == [
        "scenario",
        "line",
        "k",
        "x",
        "x_valid",
        "x_lower",
        "x_upper",
        "x_status",
        "x_point_status",
    ]
    assert frame["scenario"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert frame["line"].tolist() == [10, 20] * 4
    assert frame["k"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    invalid = frame.iloc[5]
    assert not invalid["x_valid"]
    assert math.isnan(invalid["x_lower"])
    assert pd.isna(invalid["x_status"])
    assert frame["x_status"].tolist()[:2] == ["uncertain", "within_limits"]
    assert frame["x_point_status"].tolist()[:2] == ["within_limits"] * 2
    assert frame["x_upper"].tolist()[7] == INF


def test_to_frame_inverts_from_frame() -> None:
    predictions, _, _ = _coverage_data()

    frame = predictions.to_frame(entity="line")
    rebuilt = ChannelSet.from_frame(
        frame, ["x"], entity="line", valid="x_valid", strata=["k"]
    )

    np.testing.assert_array_equal(rebuilt.channels["x"], predictions.channels["x"])
    np.testing.assert_array_equal(
        rebuilt.valid_masks["x"], predictions.valid_masks["x"]
    )
    np.testing.assert_array_equal(rebuilt.scenario_columns["k"], [0, 0, 1, 1])
    assert frame.columns.tolist() == ["scenario", "line", "k", "x", "x_valid"]


def test_to_frame_refuses_limits_it_cannot_apply() -> None:
    predictions, _, intervals = _coverage_data()

    with pytest.raises(ValueError, match="need intervals"):
        predictions.to_frame(limits={"x": {"upper": 1.0}})
    with pytest.raises(ValueError, match="without intervals"):
        predictions.to_frame(intervals, limits={"y": {"upper": 1.0}})
    with pytest.raises(ValueError, match="take 'lower'"):
        predictions.to_frame(intervals, limits={"x": {"high": 1.0}})
