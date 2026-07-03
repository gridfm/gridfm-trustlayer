"""Tests for naive per-bus interval calibration."""

from __future__ import annotations

import numpy as np

from gridfm_trustlayer.conformal import NaiveIntervalCalibrator
from gridfm_trustlayer.io.predictions import PredictionSet
from gridfm_trustlayer.io.ground_truth import GroundTruthSet


def test_naive_interval_fit_and_predict() -> None:
    predictions, ground_truth = _make_aligned_sets()

    calibrator = NaiveIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.2,
        channel_names=("Vm", "Va"),
    )

    vm_residuals = ground_truth.Vm - predictions.Vm
    va_residuals = ground_truth.Va - predictions.Va
    np.testing.assert_allclose(
        calibrator.lower_residual_quantiles["Vm"],
        np.quantile(vm_residuals, 0.1, axis=0).astype(np.float32),
    )
    np.testing.assert_allclose(
        calibrator.upper_residual_quantiles["Vm"],
        np.quantile(vm_residuals, 0.9, axis=0).astype(np.float32),
    )
    np.testing.assert_allclose(
        calibrator.lower_residual_quantiles["Va"],
        np.quantile(va_residuals, 0.1, axis=0).astype(np.float32),
    )
    np.testing.assert_allclose(
        calibrator.upper_residual_quantiles["Va"],
        np.quantile(va_residuals, 0.9, axis=0).astype(np.float32),
    )

    intervals = calibrator.predict_intervals(predictions)
    expected_vm_lower = predictions.Vm + calibrator.lower_residual_quantiles["Vm"][None, :]
    expected_vm_upper = predictions.Vm + calibrator.upper_residual_quantiles["Vm"][None, :]
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
        removed_mask=np.zeros((len(scenario_ids), 1), dtype=bool),
        Vm=vm_target,
        Va=va_target,
        channels={},
        admittances={key: np.zeros((len(scenario_ids), 1), dtype=np.float32) for key in (
            "Yff_r",
            "Yff_i",
            "Yft_r",
            "Yft_i",
            "Ytf_r",
            "Ytf_i",
            "Ytt_r",
            "Ytt_i",
        )},
        f_bus=np.array([0], dtype=np.int32),
        t_bus=np.array([1], dtype=np.int32),
        rate_a=np.array([1.0], dtype=np.float32),
        base_kv=np.array([100.0, 100.0], dtype=np.float32),
        sn_mva=100.0,
        bus_type_masks={},
    )
    return prediction_set, ground_truth