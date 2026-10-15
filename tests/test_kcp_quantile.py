"""The kernel's tail-only weighted quantile equals the full-scan definition.

`_weighted_tail_offsets` reads only the ends of the sorted residuals (widening
where needed) and takes the total weight from a matrix product. These tests
compare it, and whole `predict_intervals` runs, with the straightforward full
scan in `tests/conformal_reference.py`, and require identical bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

import gridfm_trustlayer.conformal.kernel as kernel_module
from gridfm_trustlayer.conformal import ChannelSet, KernelIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import (
    BANDWIDTH_N_EFF,
    _effective_sample_size_and_slope,
    _prepare_embeddings,
    _solve_bandwidth_for_n_eff,
    _squared_distances,
    _StratumBank,
    _weighted_tail_offsets,
)
from tests.conformal_reference import weighted_conformal_offsets


def _bank(residuals: np.ndarray, valid: np.ndarray) -> _StratumBank:
    order = np.argsort(residuals, axis=0, kind="stable").astype(np.int32)
    sorted_valid = np.take_along_axis(valid, order, axis=0)
    return _StratumBank(
        rows=np.arange(residuals.shape[0]),
        sorted_residuals=np.take_along_axis(residuals, order, axis=0),
        sort_order=order,
        sorted_valid=sorted_valid.astype(np.float32),
        valid=valid.astype(np.float64),
        sort_order_masked=np.where(sorted_valid, order, residuals.shape[0]).astype(
            np.int32
        ),
        fallback_lower=np.zeros(residuals.shape[1], dtype=np.float32),
        fallback_upper=np.zeros(residuals.shape[1], dtype=np.float32),
        has_data=np.asarray(valid.any(axis=0)),
    )


def _case(seed: int, n_calibration: int, n_elements: int, kind: str):
    rng = np.random.default_rng(seed)
    position = rng.normal(size=n_calibration)
    residuals = rng.normal(size=(n_calibration, n_elements))
    if kind == "heteroscedastic":
        # Large residuals sit where the weights are large: the tails hold
        # much of the weight, so the first scan window is often too short.
        residuals *= np.exp(2.0 * position)[:, None]
    if kind == "ties":
        residuals = np.round(residuals, 0)
    residuals = residuals.astype(np.float32)
    valid = rng.random((n_calibration, n_elements)) > 0.1
    valid[:, 0] = False  # an element without any valid residual
    test_position = rng.normal(size=(17, 1))
    bandwidth = {"peaked": 0.05, "flat": 50.0}.get(kind, 1.0)
    weights = np.exp(-((test_position - position[None, :]) ** 2) / bandwidth)
    if kind == "underflow":
        weights[:, rng.random(n_calibration) < 0.5] = 0.0
    return weights.astype(np.float32), residuals, valid


CASES = [
    (seed, n, kind)
    for seed, (n, kind) in enumerate(
        (n, kind)
        for n in (1, 2, 19, 60, 333)
        for kind in ("smooth", "heteroscedastic", "ties", "peaked", "flat", "underflow")
    )
]


@pytest.mark.parametrize("alpha", [0.02, 0.1, 0.2, 0.5])
@pytest.mark.parametrize(("seed", "n_calibration", "kind"), CASES)
def test_tail_quantile_equals_the_full_scan(seed, n_calibration, kind, alpha) -> None:
    weights, residuals, valid = _case(seed, n_calibration, 5, kind)
    level = 1.0 - alpha / 2.0

    lower, upper, reached = _weighted_tail_offsets(
        weights, _bank(residuals, valid), level
    )
    ref_lower, ref_upper, ref_reached = weighted_conformal_offsets(
        weights, residuals, valid, level
    )

    np.testing.assert_array_equal(reached, ref_reached)
    np.testing.assert_array_equal(lower[reached], ref_lower[reached])
    np.testing.assert_array_equal(upper[reached], ref_upper[reached])


def test_tail_quantile_widens_the_scan_until_every_bound_settles(monkeypatch) -> None:
    # The smallest possible first window forces several doublings.
    monkeypatch.setattr(kernel_module, "_TAIL_WINDOW_FACTOR", 0.0)
    monkeypatch.setattr(kernel_module, "_MIN_TAIL_WINDOW", 1)
    weights, residuals, valid = _case(7, 333, 6, "heteroscedastic")

    lower, upper, reached = _weighted_tail_offsets(
        weights, _bank(residuals, valid), 0.95
    )
    ref_lower, ref_upper, ref_reached = weighted_conformal_offsets(
        weights, residuals, valid, 0.95
    )

    assert reached.any()
    np.testing.assert_array_equal(reached, ref_reached)
    np.testing.assert_array_equal(lower[reached], ref_lower[reached])
    np.testing.assert_array_equal(upper[reached], ref_upper[reached])


def _reference_predict(calibrator: KernelIntervalCalibrator, test: ChannelSet):
    """`predict_intervals` written out with the full-scan quantile."""

    assert test.scenario_embeddings is not None
    standardized = _prepare_embeddings(
        test.scenario_embeddings,
        calibrator.embedding_mean,
        calibrator.embedding_scale,
        calibrator.distance,
    )
    strata = test.scenario_columns["n_outages"]
    level = 1.0 - calibrator.alpha / 2.0
    offsets = {
        name: [np.zeros(test.channels[name].shape) for _ in range(2)]
        for name in calibrator.channel_names
    }
    fallback = np.zeros(len(strata), dtype=bool)
    for index, value in enumerate(calibrator.stratification_values):
        rows = np.flatnonzero(strata == value)
        bank_rows = np.flatnonzero(calibrator.calibration_strata == value)
        calibration = calibrator.calibration_embeddings[bank_rows]
        distances = _squared_distances(
            standardized[rows],
            calibration,
            np.einsum("ij,ij->i", calibration, calibration),
        )
        if calibrator.bandwidth == BANDWIDTH_N_EFF:
            target = min(calibrator.n_eff_target, float(len(bank_rows)))
            bandwidth, _ = _solve_bandwidth_for_n_eff(distances, target)
        else:
            k = min(calibrator.k_nn, len(bank_rows)) - 1
            bandwidth = np.partition(distances, k, axis=1)[:, k]
        bandwidth = np.maximum(bandwidth, np.float32(1e-12))
        weights = np.exp(-distances / bandwidth[:, None]).astype(np.float32)
        n_eff = _effective_sample_size_and_slope(
            distances, bandwidth.astype(np.float64)
        )[0]
        degenerate = n_eff < calibrator.n_eff_min
        fallback[rows] |= degenerate
        for name in calibrator.channel_names:
            residuals = calibrator.residuals[name][bank_rows]
            valid = calibrator.valid_masks[name][bank_rows]
            lower, upper, reached = weighted_conformal_offsets(
                weights, residuals, valid, level
            )
            use = reached & ~degenerate[:, None]
            offsets[name][0][rows] = np.where(
                use, lower, calibrator.fallback_lower_quantiles[name][index]
            )
            offsets[name][1][rows] = np.where(
                use, upper, calibrator.fallback_upper_quantiles[name][index]
            )
            fallback[rows] |= ~(reached | ~valid.any(axis=0)).all(axis=1)
    return offsets, fallback


@pytest.mark.parametrize(
    ("bandwidth", "options", "embedding_dim"),
    [
        ("n_eff", {"n_eff_target": 40}, 2),
        ("n_eff", {"n_eff_target": 150}, 3),
        ("knn", {"k_nn": 15}, 3),
        # Wide embedding: distances concentrate and the level is unreachable for
        # most scenarios, which the kernel now detects before any quantile work.
        ("n_eff", {"n_eff_target": 100}, 64),
    ],
)
def test_predict_intervals_equals_the_full_scan_definition(
    bandwidth, options, embedding_dim
) -> None:
    rng = np.random.default_rng(embedding_dim)
    n_calibration, n_test, n_elements = 600, 120, 7

    def channel_set(n: int, values: dict[str, np.ndarray], embeddings, strata):
        return ChannelSet(
            scenario_ids=np.arange(n),
            entity_ids=np.arange(n_elements),
            entity_name="bus",
            channels=values,
            scenario_columns={"n_outages": strata},
            scenario_embeddings=embeddings,
        )

    embeddings = rng.normal(size=(n_calibration + n_test, embedding_dim)).astype(
        np.float32
    )
    strata = rng.integers(0, 3, size=n_calibration + n_test)
    scale = 0.01 * (1.0 + np.abs(embeddings[:, :1]))
    predicted = {
        name: rng.normal(size=(n_calibration + n_test, n_elements)).astype(np.float32)
        for name in ("Vm", "Va")
    }
    true = {
        name: (values + scale * rng.normal(size=values.shape)).astype(np.float32)
        for name, values in predicted.items()
    }
    cal, test = slice(0, n_calibration), slice(n_calibration, None)
    calibrator = KernelIntervalCalibrator.fit(
        channel_set(
            n_calibration,
            {k: v[cal] for k, v in predicted.items()},
            embeddings[cal],
            strata[cal],
        ),
        channel_set(
            n_calibration,
            {k: v[cal] for k, v in true.items()},
            embeddings[cal],
            strata[cal],
        ),
        alpha=0.1,
        channel_names=("Vm", "Va"),
        strat="n_outages",
        embeddings=embeddings[cal],
        embedding_pooling="mean",
        bandwidth=bandwidth,
        n_eff_min=5,
        **options,
    )
    test_set = channel_set(
        n_test,
        {k: v[test] for k, v in predicted.items()},
        embeddings[test],
        strata[test],
    )

    intervals = calibrator.predict_intervals(test_set)
    offsets, fallback = _reference_predict(calibrator, test_set)

    for name in ("Vm", "Va"):
        lower, upper = intervals[name]
        np.testing.assert_array_equal(
            lower, test_set.channels[name] + offsets[name][0].astype(np.float32)
        )
        np.testing.assert_array_equal(
            upper, test_set.channels[name] + offsets[name][1].astype(np.float32)
        )
    np.testing.assert_array_equal(calibrator.diagnostics["kernel_fallback"], fallback)
    if embedding_dim == 64:
        assert fallback.mean() > 0.5
