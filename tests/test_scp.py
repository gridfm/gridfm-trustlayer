"""Tests for naive per-bus interval calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from gridfm_trustlayer.conformal import ChannelSet, NaiveIntervalCalibrator
from gridfm_trustlayer.io.ground_truth import GroundTruthSet
from gridfm_trustlayer.io.predictions import PredictionSet
from tests.conformal_reference import conformal_quantile


def test_naive_interval_fit_and_predict() -> None:
    predictions, ground_truth = _make_aligned_sets()

    # Four calibration scenarios: alpha = 0.5 is the smallest round value whose
    # split-conformal ranks stay within 1..4, so both tails are finite.
    calibrator = NaiveIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.5,
        channel_names=("Vm", "Va"),
    )

    vm_residuals = ground_truth.Vm - predictions.Vm
    va_residuals = ground_truth.Va - predictions.Va
    np.testing.assert_allclose(
        calibrator.lower_residual_quantiles["Vm"],
        conformal_quantile(vm_residuals, 0.25),
    )
    np.testing.assert_allclose(
        calibrator.upper_residual_quantiles["Vm"],
        conformal_quantile(vm_residuals, 0.75),
    )
    np.testing.assert_allclose(
        calibrator.lower_residual_quantiles["Va"],
        conformal_quantile(va_residuals, 0.25),
    )
    np.testing.assert_allclose(
        calibrator.upper_residual_quantiles["Va"],
        conformal_quantile(va_residuals, 0.75),
    )

    intervals = calibrator.predict_intervals(predictions)
    expected_vm_lower = (
        predictions.Vm + calibrator.lower_residual_quantiles["Vm"][None, :]
    )
    expected_vm_upper = (
        predictions.Vm + calibrator.upper_residual_quantiles["Vm"][None, :]
    )
    np.testing.assert_allclose(intervals["Vm"][0], expected_vm_lower)
    np.testing.assert_allclose(intervals["Vm"][1], expected_vm_upper)


def test_naive_interval_roundtrip() -> None:
    predictions, ground_truth = _make_aligned_sets()
    calibrator = NaiveIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.2,
    )

    loaded = NaiveIntervalCalibrator.from_dict(calibrator.to_dict())

    assert loaded.alpha == calibrator.alpha
    np.testing.assert_array_equal(loaded.bus_ids, calibrator.bus_ids)
    np.testing.assert_allclose(
        loaded.lower_residual_quantiles["Vm"],
        calibrator.lower_residual_quantiles["Vm"],
    )
    np.testing.assert_allclose(
        loaded.upper_residual_quantiles["Va"],
        calibrator.upper_residual_quantiles["Va"],
    )


def test_naive_interval_stratified_by_n_outages() -> None:
    scenario_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    bus_ids = np.array([0, 1], dtype=np.int64)
    scenario_columns = {"n_outages": np.array([0, 0, 1, 1], dtype=np.int32)}
    predictions = ChannelSet(
        scenario_ids=scenario_ids,
        entity_ids=bus_ids,
        entity_name="bus",
        channels={
            "Vm": np.array(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                dtype=np.float32,
            )
        },
        scenario_columns=scenario_columns,
    )
    ground_truth = ChannelSet(
        scenario_ids=scenario_ids,
        entity_ids=bus_ids,
        entity_name="bus",
        channels={
            "Vm": np.array(
                [[1.1, 0.9], [1.2, 0.8], [1.5, 0.5], [1.6, 0.4]],
                dtype=np.float32,
            )
        },
        scenario_columns=scenario_columns,
    )

    # Two calibration scenarios per stratum: alpha = 0.9 keeps both tails finite.
    calibrator = NaiveIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.9,
        channel_names=("Vm",),
        strat="n_outages",
    )

    assert calibrator.strat == "n_outages"
    np.testing.assert_array_equal(
        calibrator.stratification_values,
        np.array([0, 1], dtype=np.int64),
    )
    np.testing.assert_allclose(
        calibrator.stratified_lower_residual_quantiles["Vm"][0],
        conformal_quantile(
            np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32), 0.45
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        calibrator.stratified_upper_residual_quantiles["Vm"][1],
        conformal_quantile(
            np.array([[0.5, -0.5], [0.6, -0.6]], dtype=np.float32), 0.55
        ),
        atol=1e-6,
    )

    intervals = calibrator.predict_intervals(predictions)
    np.testing.assert_allclose(
        intervals["Vm"][0][0],
        predictions.channels["Vm"][0]
        + calibrator.stratified_lower_residual_quantiles["Vm"][0],
    )
    np.testing.assert_allclose(
        intervals["Vm"][1][3],
        predictions.channels["Vm"][3]
        + calibrator.stratified_upper_residual_quantiles["Vm"][1],
    )

    loaded = NaiveIntervalCalibrator.from_dict(calibrator.to_dict())
    assert loaded.strat == "n_outages"
    np.testing.assert_array_equal(loaded.stratification_values, np.array([0, 1]))
    np.testing.assert_allclose(
        loaded.stratified_lower_residual_quantiles["Vm"],
        calibrator.stratified_lower_residual_quantiles["Vm"],
    )


def test_naive_interval_stratified_by_outage_type() -> None:
    scenario_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    bus_ids = np.array([0, 1], dtype=np.int64)
    scenario_columns = {"outage_type": np.array([0, 1, 2, 3], dtype=np.int32)}
    predictions = ChannelSet(
        scenario_ids=scenario_ids,
        entity_ids=bus_ids,
        entity_name="bus",
        channels={
            "Vm": np.ones((4, 2), dtype=np.float32),
        },
        scenario_columns=scenario_columns,
    )
    ground_truth = ChannelSet(
        scenario_ids=scenario_ids,
        entity_ids=bus_ids,
        entity_name="bus",
        channels={
            "Vm": np.array(
                [[1.0, 1.0], [1.1, 0.9], [1.2, 0.8], [1.3, 0.7]],
                dtype=np.float32,
            )
        },
        scenario_columns=scenario_columns,
    )

    calibrator = NaiveIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.2,
        channel_names=("Vm",),
        strat="outage_type",
    )

    assert calibrator.strat == "outage_type"
    np.testing.assert_array_equal(
        calibrator.stratification_values,
        np.array([0, 1, 2, 3], dtype=np.int64),
    )

    # Each outage type holds a single calibration scenario. One residual cannot
    # support a 1 - alpha/2 = 0.9 bound, so every bound is infinite rather than a
    # zero-width band around that one residual.
    intervals = calibrator.predict_intervals(predictions)
    assert np.all(np.isneginf(intervals["Vm"][0]))
    assert np.all(np.isposinf(intervals["Vm"][1]))

    loaded = NaiveIntervalCalibrator.from_dict(calibrator.to_dict())
    assert loaded.strat == "outage_type"
    np.testing.assert_array_equal(loaded.stratification_values, np.array([0, 1, 2, 3]))


def test_naive_interval_stratified_by_system_load_bin_roundtrip() -> None:
    scenario_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    bus_ids = np.array([0, 1], dtype=np.int64)
    scenario_columns = {"system_load_bin": np.array([0, 0, 1, 1], dtype=np.int32)}
    predictions = ChannelSet(
        scenario_ids=scenario_ids,
        entity_ids=bus_ids,
        entity_name="bus",
        channels={"Vm": np.ones((4, 2), dtype=np.float32)},
        scenario_columns=scenario_columns,
    )
    ground_truth = ChannelSet(
        scenario_ids=scenario_ids,
        entity_ids=bus_ids,
        entity_name="bus",
        channels={
            "Vm": np.array(
                [[1.0, 1.0], [1.1, 0.9], [1.4, 0.6], [1.5, 0.5]],
                dtype=np.float32,
            )
        },
        scenario_columns=scenario_columns,
    )

    calibrator = NaiveIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.2,
        channel_names=("Vm",),
        strat="system_load_bin",
        system_load_n_bins=2,
        system_load_bin_edges=np.array([50.0], dtype=np.float32),
    )

    assert calibrator.strat == "system_load_bin"
    assert calibrator.system_load_n_bins == 2
    assert calibrator.system_load_bin_edges is not None
    np.testing.assert_allclose(
        calibrator.system_load_bin_edges,
        np.array([50.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        calibrator.stratification_values,
        np.array([0, 1], dtype=np.int64),
    )

    loaded = NaiveIntervalCalibrator.from_dict(calibrator.to_dict())
    assert loaded.strat == "system_load_bin"
    assert loaded.system_load_n_bins == 2
    assert loaded.system_load_bin_edges is not None
    np.testing.assert_allclose(
        loaded.system_load_bin_edges,
        np.array([50.0], dtype=np.float32),
    )


def test_naive_interval_stratified_by_pred_bins_roundtrip() -> None:
    scenario_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    bus_ids = np.array([0, 1], dtype=np.int64)
    predictions = ChannelSet(
        scenario_ids=scenario_ids,
        entity_ids=bus_ids,
        entity_name="bus",
        channels={
            "Vm": np.array(
                [[0.90, 1.00], [0.95, 1.05], [1.10, 1.20], [1.15, 1.25]],
                dtype=np.float32,
            )
        },
    )
    ground_truth = ChannelSet(
        scenario_ids=scenario_ids,
        entity_ids=bus_ids,
        entity_name="bus",
        channels={
            "Vm": np.array(
                [[0.95, 1.02], [0.97, 1.08], [1.20, 1.35], [1.25, 1.40]],
                dtype=np.float32,
            )
        },
    )

    calibrator = NaiveIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.9,
        channel_names=("Vm",),
        strat="pred_bins",
        pred_n_bins=2,
        pred_bin_edges={"Vm": np.array([1.025], dtype=np.float32)},
    )

    assert calibrator.strat == "pred_bins"
    assert calibrator.pred_n_bins == 2
    np.testing.assert_allclose(
        calibrator.pred_bin_edges["Vm"],
        np.array([1.025], dtype=np.float32),
    )
    intervals = calibrator.predict_intervals(predictions)
    # Bin 0 holds two residuals for bus 0 (finite bounds at alpha = 0.9) but a
    # single residual for bus 1 (infinite bounds).
    expected_lower_bin0 = np.array(
        [
            0.90 + conformal_quantile(np.array([[0.05], [0.02]]), 0.45)[0],
            1.00 + conformal_quantile(np.array([[0.02]]), 0.45)[0],
        ],
        dtype=np.float32,
    )
    expected_upper_bin0 = np.array(
        [
            0.90 + conformal_quantile(np.array([[0.05], [0.02]]), 0.55)[0],
            1.00 + conformal_quantile(np.array([[0.02]]), 0.55)[0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        intervals["Vm"][0][0],
        expected_lower_bin0,
    )
    np.testing.assert_allclose(
        intervals["Vm"][1][0],
        expected_upper_bin0,
    )

    loaded = NaiveIntervalCalibrator.from_dict(calibrator.to_dict())
    assert loaded.strat == "pred_bins"
    assert loaded.pred_n_bins == 2
    np.testing.assert_allclose(
        loaded.pred_bin_edges["Vm"],
        np.array([1.025], dtype=np.float32),
    )


def _single_entity_sets(residuals: np.ndarray) -> tuple[ChannelSet, ChannelSet]:
    """Zero predictions and targets equal to `residuals`, shape (scenarios, entities)."""

    residuals = np.asarray(residuals, dtype=np.float32)
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    n_scenarios, n_entities = residuals.shape
    common: dict[str, Any] = {
        "scenario_ids": np.arange(n_scenarios, dtype=np.int64),
        "entity_ids": np.arange(n_entities, dtype=np.int64),
        "entity_name": "bus",
    }
    predictions = ChannelSet(channels={"Vm": np.zeros_like(residuals)}, **common)
    targets = ChannelSet(channels={"Vm": residuals}, **common)
    return predictions, targets


def test_split_conformal_quantile_is_the_corrected_order_statistic() -> None:
    # Residuals 1..19 in shuffled order. At alpha = 0.2 each tail is at level 0.1,
    # so the bounds are the ceil(20 * 0.9) = 18th and floor(20 * 0.1) = 2nd
    # smallest residuals. The plain empirical quantile would interpolate to
    # 17.2 and 2.8 instead, which is what made the old intervals too narrow.
    residuals = np.random.default_rng(0).permutation(np.arange(1, 20))
    calibrator = NaiveIntervalCalibrator.fit(
        *_single_entity_sets(residuals), alpha=0.2, channel_names=("Vm",)
    )

    assert calibrator.upper_residual_quantiles["Vm"][0] == 18.0
    assert calibrator.lower_residual_quantiles["Vm"][0] == 2.0


def test_too_few_calibration_samples_give_infinite_bounds() -> None:
    # The upper rank ceil((n + 1) * 0.9) first fits within n at n = 9.
    short = NaiveIntervalCalibrator.fit(
        *_single_entity_sets(np.arange(1, 9)), alpha=0.2, channel_names=("Vm",)
    )
    enough = NaiveIntervalCalibrator.fit(
        *_single_entity_sets(np.arange(1, 10)), alpha=0.2, channel_names=("Vm",)
    )

    assert np.isposinf(short.upper_residual_quantiles["Vm"][0])
    assert np.isneginf(short.lower_residual_quantiles["Vm"][0])
    assert enough.upper_residual_quantiles["Vm"][0] == 9.0
    assert enough.lower_residual_quantiles["Vm"][0] == 1.0


def test_intervals_reach_finite_sample_coverage() -> None:
    # Each of the 4,000 entities is an independent replication: 19 calibration
    # residuals and one test residual from the same distribution. For continuous
    # residuals the two-sided coverage is exactly (18 - 2) / 20 = 0.8 = 1 - alpha.
    # The uncorrected empirical quantile gives about 0.72 here, so this test
    # fails if the finite-sample correction is ever lost.
    rng = np.random.default_rng(42)
    n_calibration, n_replications, alpha = 19, 4000, 0.2
    calibration = rng.standard_normal((n_calibration, n_replications))
    test_residual = rng.standard_normal(n_replications).astype(np.float32)

    calibrator = NaiveIntervalCalibrator.fit(
        *_single_entity_sets(calibration), alpha=alpha, channel_names=("Vm",)
    )
    test_predictions, _ = _single_entity_sets(np.zeros((1, n_replications)))
    lower, upper = calibrator.predict_intervals(test_predictions)["Vm"]
    covered = (lower[0] <= test_residual) & (test_residual <= upper[0])

    # Standard error at 4,000 replications is about 0.0063.
    assert abs(covered.mean() - (1.0 - alpha)) < 0.025


def _make_aligned_sets() -> tuple[PredictionSet, GroundTruthSet]:
    scenario_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    bus_ids = np.array([0, 1], dtype=np.int64)
    vm_pred = np.array(
        [[1.0, 0.9], [1.1, 1.0], [0.95, 1.05], [1.05, 0.98]],
        dtype=np.float32,
    )
    va_pred = np.array(
        [[0.0, 0.1], [0.2, -0.1], [-0.2, 0.3], [0.1, 0.0]],
        dtype=np.float32,
    )
    vm_target = np.array(
        [[1.05, 0.85], [1.12, 0.97], [1.0, 1.08], [1.0, 1.0]],
        dtype=np.float32,
    )
    va_target = np.array(
        [[0.02, 0.12], [0.16, -0.05], [-0.1, 0.25], [0.08, -0.02]],
        dtype=np.float32,
    )
    prediction_set = PredictionSet(
        task_name="PowerFlow",
        channel_names=("Vm", "Va"),
        scenario_ids=scenario_ids,
        bus_ids=bus_ids,
        Vm=vm_pred,
        Va=va_pred,
        channels={},
        bus_type_masks={},
    )
    ground_truth = GroundTruthSet(
        scenario_ids=scenario_ids,
        bus_ids=bus_ids,
        branch_ids=np.array([0], dtype=np.int64),
        n_outages=np.zeros(len(scenario_ids), dtype=np.int32),
        outage_type=np.zeros(len(scenario_ids), dtype=np.int32),
        system_load=np.zeros(len(scenario_ids), dtype=np.float32),
        removed_mask=np.zeros((len(scenario_ids), 1), dtype=bool),
        Vm=vm_target,
        Va=va_target,
        channels={},
        admittances={
            key: np.zeros((len(scenario_ids), 1), dtype=np.float32)
            for key in (
                "Yff_r",
                "Yff_i",
                "Yft_r",
                "Yft_i",
                "Ytf_r",
                "Ytf_i",
                "Ytt_r",
                "Ytt_i",
            )
        },
        f_bus=np.array([0], dtype=np.int32),
        t_bus=np.array([1], dtype=np.int32),
        rate_a=np.array([1.0], dtype=np.float32),
        base_kv=np.array([100.0, 100.0], dtype=np.float32),
        sn_mva=100.0,
        bus_type_masks={},
    )
    return prediction_set, ground_truth
