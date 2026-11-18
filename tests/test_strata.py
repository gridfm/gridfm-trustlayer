"""Strata from your own data: `OutageStrata` and `LoadBins` give the CLI's strata."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer import ChannelSet, LoadBins, OutageStrata
from gridfm_trustlayer.cli.common import (
    _assign_system_load_bins,
    _compute_system_load_bin_edges,
)
from gridfm_trustlayer.io.ground_truth import (
    _classify_outage_type,
    _count_reference_outages,
)
from gridfm_trustlayer.strata import OUTAGE_TYPES, strata_from_dict


def _in_service(rng: np.random.Generator, n: int, width: int) -> np.ndarray:
    flags = rng.random((n, width)) > 0.15
    flags[:, 0] = False  # never in service, e.g. a line not built yet
    return flags


def test_outage_strata_match_the_cli_with_the_calibration_set_as_reference() -> None:
    rng = np.random.default_rng(0)
    calibration = _in_service(rng, 300, 12), _in_service(rng, 300, 4)
    test = _in_service(rng, 80, 12), _in_service(rng, 80, 4)
    test[0][:, 0] = rng.random(80) > 0.5  # in service only in the test set

    strata = OutageStrata.fit(*calibration)
    for branch, generator in (calibration, test):
        expected_branch = _count_reference_outages(
            branch.astype(np.float32), strata.branches[None, :]
        )
        expected_generator = _count_reference_outages(
            generator.astype(np.float32), np.asarray(strata.generators)[None, :]
        )
        np.testing.assert_array_equal(
            strata.n_outages(branch, generator), expected_branch + expected_generator
        )
        np.testing.assert_array_equal(
            strata.outage_type(branch, generator),
            _classify_outage_type(expected_branch, expected_generator),
        )
    assert not strata.branches[0]


def test_outage_strata_of_wide_frames_are_series_ready_for_from_frame() -> None:
    branch = pd.DataFrame(
        {"l1": [1, 1, 0, 0], "l2": [1, 0, 1, 0]}, index=pd.Index([30, 10, 20, 40])
    )
    generator = pd.DataFrame({"g1": [True, True, False, True]}, index=branch.index)
    strata = OutageStrata.fit(branch, generator)

    n_outages = strata.n_outages(branch, generator)
    outage_type = strata.outage_type(branch, generator)
    assert isinstance(n_outages, pd.Series)
    assert n_outages.name == "n_outages"
    assert n_outages.to_dict() == {30: 0, 10: 1, 20: 2, 40: 2}
    assert [OUTAGE_TYPES[code] for code in outage_type] == [
        "none",
        "branch_only",
        "mixed",
        "branch_only",
    ]

    rows = pd.DataFrame(
        {"scenario": [10, 20, 30, 40], "bus": 1, "x": [1.0, 2.0, 3.0, 4.0]}
    )
    channel_set = ChannelSet.from_frame(
        rows, ["x"], entity="bus", strata={"n_outages": n_outages}
    )
    np.testing.assert_array_equal(
        channel_set.scenario_columns["n_outages"], [1, 2, 0, 2]
    )

    with pytest.raises(ValueError, match="no value for 1 scenario"):
        ChannelSet.from_frame(
            rows, ["x"], entity="bus", strata={"n_outages": n_outages.iloc[1:]}
        )


def test_outage_strata_count_branches_only_when_fitted_without_generators() -> None:
    branch = np.array([[1, 1], [0, 1], [0, 0]], dtype=bool)
    strata = OutageStrata.fit(branch)

    np.testing.assert_array_equal(strata.n_outages(branch), [0, 1, 2])
    np.testing.assert_array_equal(strata.outage_type(branch), [0, 1, 1])
    with pytest.raises(ValueError, match="Fitted without generators"):
        strata.n_outages(branch, branch)


@pytest.mark.parametrize(
    ("branch", "generator", "message"),
    [
        (np.ones((3, 2), dtype=bool), None, "pass generator_in_service"),
        (np.ones((3, 5), dtype=bool), np.ones((3, 1)), "Got 5 branches"),
        (np.ones((3, 2), dtype=bool), np.ones((3, 4)), "Got 4 generators"),
        (np.ones((3, 2), dtype=bool), np.ones((2, 1)), "3 scenarios"),
        (np.full((3, 2), 0.5), np.ones((3, 1)), "True/False"),
        (np.ones(3, dtype=bool), np.ones((3, 1)), "must be 2-D"),
    ],
)
def test_outage_strata_refuse_inputs_that_do_not_fit(
    branch: np.ndarray, generator: np.ndarray | None, message: str
) -> None:
    strata = OutageStrata.fit(np.ones((4, 2), dtype=bool), np.ones((4, 1)))
    with pytest.raises(ValueError, match=message):
        strata.n_outages(branch, generator)


def _reference_edges(loads: np.ndarray, n_bins: int) -> np.ndarray:
    """The CLI's edges up to 0.1.1, before they moved to `gridfm_trustlayer.strata`."""

    finite = np.asarray(loads, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    edges = np.quantile(finite, np.linspace(0.0, 1.0, num=n_bins + 1))
    return np.unique(edges.astype(np.float32)[1:-1])


@pytest.mark.parametrize("n_bins", [2, 5, 10])
def test_load_bins_match_the_cli(n_bins: int) -> None:
    rng = np.random.default_rng(n_bins)
    calibration = rng.gamma(4.0, 50.0, size=500).astype(np.float32)
    calibration[::7] = np.round(calibration[::7], -1)  # some ties
    test = rng.gamma(4.0, 50.0, size=120).astype(np.float32)

    bins = LoadBins.fit(calibration, n_bins)
    expected = _reference_edges(calibration, n_bins)
    np.testing.assert_array_equal(bins.edges, expected)
    np.testing.assert_array_equal(
        _compute_system_load_bin_edges(calibration, n_bins), expected
    )
    for loads in (calibration, test):
        reference = np.searchsorted(expected, loads.astype(np.float32), side="right")
        np.testing.assert_array_equal(bins.assign(loads), reference)
        np.testing.assert_array_equal(
            _assign_system_load_bins(loads, bins.edges), reference
        )
    counts = np.bincount(bins.assign(calibration))
    assert counts.size == n_bins
    assert counts.min() > 0


def test_load_bins_keep_the_index_of_a_series_and_refuse_bad_loads() -> None:
    loads = pd.Series([3.0, 1.0, 2.0, 4.0], index=[7, 5, 6, 8])
    bins = LoadBins.fit(loads, 2)

    assigned = bins.assign(loads)
    assert isinstance(assigned, pd.Series)
    assert assigned.name == "system_load_bin"
    assert assigned.to_dict() == {7: 1, 5: 0, 6: 0, 8: 1}

    with pytest.raises(ValueError, match="at least 2"):
        LoadBins.fit(loads, 1)
    with pytest.raises(ValueError, match="two distinct bins"):
        LoadBins.fit(np.ones(10), 4)
    with pytest.raises(ValueError, match="finite"):
        bins.assign(np.array([1.0, np.nan]))


def test_outage_strata_match_frame_columns_by_label() -> None:
    calibration = pd.DataFrame({7: [1, 1, 0], 3: [1, 0, 1], 5: [0, 0, 0]})
    strata = OutageStrata.fit(calibration)

    shuffled = calibration[[5, 7, 3]]
    np.testing.assert_array_equal(strata.n_outages(shuffled), [0, 1, 1])
    np.testing.assert_array_equal(strata.n_outages(calibration.to_numpy()), [0, 1, 1])
    with pytest.raises(ValueError, match="1 missing"):
        strata.n_outages(calibration[[7, 3]])
    with pytest.raises(ValueError, match="1 not seen at fit"):
        strata.n_outages(calibration.assign(**{"9": 1}))


def test_strata_survive_a_json_round_trip() -> None:
    branch = pd.DataFrame({"a": [1, 0], "b": [1, 1]}, index=[4, 5])
    generator = pd.DataFrame({10: [1, 1], 11: [0, 1]}, index=[4, 5])
    outages = OutageStrata.fit(branch, generator)
    bins = LoadBins.fit(np.array([0.1, 0.7, 0.3, 0.9, 0.5], dtype=np.float32), 3)

    for strata in (outages, bins):
        restored = strata_from_dict(json.loads(json.dumps(strata.to_dict())))
        assert type(restored) is type(strata)
    restored_outages = strata_from_dict(json.loads(json.dumps(outages.to_dict())))
    assert isinstance(restored_outages, OutageStrata)
    np.testing.assert_array_equal(
        restored_outages.n_outages(branch, generator),
        outages.n_outages(branch, generator),
    )
    restored_bins = strata_from_dict(json.loads(json.dumps(bins.to_dict())))
    assert isinstance(restored_bins, LoadBins)
    np.testing.assert_array_equal(restored_bins.edges, bins.edges)
    assert restored_bins.edges.dtype == np.float32
    with pytest.raises(ValueError, match="Unknown strata kind"):
        strata_from_dict({"kind": "weather"})
