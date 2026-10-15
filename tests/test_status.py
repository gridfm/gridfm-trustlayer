"""The status rule shared by `screen` and `apply`."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.cli import status


def _classify(pred, lower, upper, lower_limit=0.95, upper_limit=1.05):
    interval, point = status.classify(
        np.asarray(pred, float),
        np.asarray(lower, float),
        np.asarray(upper, float),
        lower_limit,
        upper_limit,
    )
    return list(interval), list(point)


def test_interval_status_against_two_limits() -> None:
    interval, point = _classify(
        pred=[1.00, 1.04, 1.07, 0.96, 0.90, 1.00],
        lower=[0.98, 1.02, 1.06, 0.94, 0.85, 0.94],
        upper=[1.02, 1.06, 1.09, 0.97, 0.93, 1.06],
    )
    assert interval == [
        "within_limits",
        "uncertain",  # straddles the upper limit
        "violation",  # entirely above it
        "uncertain",  # straddles the lower limit
        "violation",  # entirely below it
        "uncertain",  # crosses both
    ]
    assert point == [
        "within_limits",
        "within_limits",
        "violation",
        "within_limits",
        "violation",
        "within_limits",
    ]


def test_upper_limit_only_and_edge_cases() -> None:
    interval, point = _classify(
        pred=[0.5, 0.9, 1.2, 0.9],
        lower=[-2.0, 0.8, 1.1, -math.inf],
        upper=[1.0, math.inf, 1.3, 0.95],
        lower_limit=math.nan,
        upper_limit=1.0,
    )
    # A bound equal to the limit does not cross it, as for the alarms of `screen`.
    assert interval == ["within_limits", "uncertain", "violation", "within_limits"]
    assert point == ["within_limits", "within_limits", "violation", "within_limits"]


def test_a_scenario_takes_the_worst_status_of_its_values() -> None:
    values = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1, 2, 2],
            "variable": ["Vm"] * 6,
            "status": [
                "within_limits",
                "within_limits",
                "uncertain",
                "within_limits",
                "uncertain",
                "violation",
            ],
            "point_status": ["within_limits"] * 5 + ["violation"],
            "truth": [False, False, True, False, False, True],
        }
    )
    rows = status.scenario_rows(values, [], truth=True)
    table = status.labelled_scenarios(
        status.scenario_table([rows], truth=True), truth=True
    )
    assert table["status"].tolist() == ["secure", "uncertain", "insecure"]
    assert table["point_status"].tolist() == ["secure", "secure", "insecure"]
    assert table["true_status"].tolist() == ["secure", "insecure", "insecure"]
    assert table["Vm_n_uncertain"].tolist() == [0, 1, 1]
    assert table["Vm_n_violation"].tolist() == [0, 0, 1]


def test_public_status_functions() -> None:
    from gridfm_trustlayer import classify_status, scenario_status

    lower = np.array([[0.5, 0.9], [0.95, 1.1]])
    upper = np.array([[0.8, 1.2], [0.99, 1.3]])
    statuses = classify_status(lower, upper, upper_limit=1.0)
    assert statuses.tolist() == [
        ["within_limits", "uncertain"],
        ["within_limits", "violation"],
    ]
    assert scenario_status(statuses, axis=1).tolist() == ["uncertain", "insecure"]
    # Masked entries (e.g. out-of-service lines) are ignored.
    valid = np.array([[True, False], [True, True]])
    assert scenario_status(statuses, axis=1, valid=valid).tolist() == [
        "secure",
        "insecure",
    ]
    both = classify_status(lower, upper, lower_limit=0.6, upper_limit=1.0)
    assert both[0, 0] == "uncertain"
    try:
        classify_status(lower, upper)
    except ValueError:
        pass
    else:
        raise AssertionError("a limit is required")
