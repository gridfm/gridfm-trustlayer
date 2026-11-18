"""The kernel's tail-only weighted quantile equals the full-scan definition.

`_weighted_tail_offsets` accumulates weight only from the ends of the sorted
residuals and takes the total weight from a matrix product. These tests
compare it, and whole `predict_intervals` runs, with the straightforward full
scan in `tests/conformal_reference.py`, and require identical bounds. They
also check the bandwidth solve, and that sharing the kernel work between
calibrators and targets gives the results of separate runs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gridfm_trustlayer.conformal import ChannelSet, KernelIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import (
    _N_EFF_SOLVE_RTOL,
    BANDWIDTH_N_EFF,
    _effective_sample_size_and_slope,
    _predict_targets,
    _prepare_embeddings,
    _solve_bandwidth_for_n_eff,
    _squared_distances,
    _StratumBank,
    _weighted_tail_offsets,
    predict_intervals_jointly,
)
from tests.conformal_reference import weighted_conformal_offsets

# These tests compare bounds with their definitions on small synthetic data,
# where most scenarios fall back to the stratum quantiles; that is checked,
# so the KernelFallbackWarning is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore::gridfm_trustlayer.KernelFallbackWarning"
)


def _bank(
    residuals: np.ndarray,
    valid: np.ndarray,
    executor: ThreadPoolExecutor | None = None,
) -> _StratumBank:
    zeros = np.zeros(residuals.shape[1], dtype=np.float32)
    return _StratumBank.build(
        np.arange(residuals.shape[0]), residuals, valid, zeros, zeros, executor
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


def test_bank_sorted_on_threads_equals_one_stable_sort() -> None:
    rng = np.random.default_rng(5)
    # Ties (rounded values) and more elements than one sorting chunk.
    residuals = np.round(rng.normal(size=(257, 150)), 1).astype(np.float32)
    valid = rng.random(residuals.shape) > 0.2

    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded = _bank(residuals, valid, executor)
    serial = _bank(residuals, valid)

    order = np.argsort(residuals, axis=0, kind="stable")
    expected_order = np.where(
        np.take_along_axis(valid, order, axis=0), order, residuals.shape[0]
    )
    for bank in (threaded, serial):
        np.testing.assert_array_equal(bank.order, expected_order.T)
        np.testing.assert_array_equal(
            bank.sorted_residuals, np.take_along_axis(residuals, order, axis=0).T
        )


def test_bandwidth_solve_converges_where_newton_steps_alternate() -> None:
    # Nearly flat n_eff at one end of the bracket and steep at the other: plain
    # Newton steps jump between the two ends (0.1.1 stopped at n_eff 310 here).
    row = np.concatenate(
        [np.full(31, 1.0), np.full(29, 20.968), np.full(339, 124.197), [22517.4]]
    ).astype(np.float32)

    bandwidth, n_eff = _solve_bandwidth_for_n_eff(row[None, :], 100.0)

    assert abs(n_eff[0] - 100.0) <= _N_EFF_SOLVE_RTOL * 100.0
    recomputed = _effective_sample_size_and_slope(
        row[None, :], bandwidth.astype(np.float64)
    )[0]
    np.testing.assert_allclose(recomputed, 100.0, rtol=1e-5)


def test_bandwidth_solve_reaches_the_target_on_every_row() -> None:
    rng = np.random.default_rng(11)
    centers = rng.normal(size=(6, 8)) * 3.0
    calibration = (centers[rng.integers(0, 6, 900)] + rng.normal(size=(900, 8))).astype(
        np.float32
    )
    test = (centers[rng.integers(0, 6, 200)] + rng.normal(size=(200, 8))).astype(
        np.float32
    )
    distances = _squared_distances(
        test, calibration, np.einsum("ij,ij->i", calibration, calibration)
    )
    for target in (5.0, 60.0, 450.0):
        bandwidth, n_eff = _solve_bandwidth_for_n_eff(distances, target)
        assert np.all(np.abs(n_eff - target) <= _N_EFF_SOLVE_RTOL * target)
        assert np.all(bandwidth > 0)


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


def _two_families(
    volt_options: dict, line_options: dict, seed: int = 4
) -> tuple[list[KernelIntervalCalibrator], list[ChannelSet]]:
    """A voltage-like and a line-like calibrator fitted on the same scenarios."""

    rng = np.random.default_rng(seed)
    n_calibration, n_test = 500, 150
    n = n_calibration + n_test
    embeddings = rng.normal(size=(n, 4)).astype(np.float32)
    strata = rng.integers(0, 3, size=n)
    scale = 0.01 * (1.0 + np.abs(embeddings[:, :1]))
    families = {"volt": ("Vm", "Va"), "line": ("line_loading",)}
    widths = {"volt": 5, "line": 7}
    calibrators, tests = [], []
    for family, channels in families.items():
        width = widths[family]
        predicted = {
            name: rng.normal(size=(n, width)).astype(np.float32) for name in channels
        }
        true = {
            name: (values + scale * rng.normal(size=values.shape)).astype(np.float32)
            for name, values in predicted.items()
        }
        valid = {name: rng.random((n, width)) > 0.05 for name in channels}

        def channel_set(
            rows: slice,
            values: dict,
            width: int = width,
            entity_name: str = "branch" if family == "line" else "bus",
            valid: dict = valid,
        ) -> ChannelSet:
            return ChannelSet(
                scenario_ids=np.arange(n)[rows],
                entity_ids=np.arange(width),
                entity_name=entity_name,
                channels={name: v[rows] for name, v in values.items()},
                valid_masks={name: v[rows] for name, v in valid.items()},
                scenario_columns={"n_outages": strata[rows]},
                scenario_embeddings=embeddings[rows],
            )

        cal, test = slice(0, n_calibration), slice(n_calibration, None)
        options = volt_options if family == "volt" else line_options
        calibrators.append(
            KernelIntervalCalibrator.fit(
                channel_set(cal, predicted),
                channel_set(cal, true),
                alpha=0.1,
                channel_names=channels,
                strat="n_outages",
                embeddings=embeddings[cal],
                n_eff_min=5,
                **options,
            )
        )
        tests.append(channel_set(test, predicted))
    return calibrators, tests


def _separate(calibrators, tests):
    results = []
    for calibrator, test in zip(calibrators, tests, strict=True):
        intervals = calibrator.predict_intervals(test)
        diagnostics = {k: v.copy() for k, v in calibrator.diagnostics.items()}
        pooled = {k: v.copy() for k, v in calibrator.pooled_masks.items()}
        results.append((intervals, diagnostics, pooled))
    return results


def _assert_same(first, second) -> None:
    for (intervals, diagnostics, pooled), (intervals2, diagnostics2, pooled2) in zip(
        first, second, strict=True
    ):
        assert intervals.keys() == intervals2.keys()
        for name in intervals:
            np.testing.assert_array_equal(intervals[name][0], intervals2[name][0])
            np.testing.assert_array_equal(intervals[name][1], intervals2[name][1])
        for key in diagnostics:
            np.testing.assert_array_equal(diagnostics[key], diagnostics2[key])
        assert pooled.keys() == pooled2.keys()
        for key in pooled:
            np.testing.assert_array_equal(pooled[key], pooled2[key])


@pytest.mark.parametrize(
    ("volt_options", "line_options"),
    [
        ({"n_eff_target": 60}, {"n_eff_target": 60}),
        ({"n_eff_target": 60}, {"n_eff_target": 120}),
        ({"bandwidth": "knn", "k_nn": 15}, {"n_eff_target": 60}),
    ],
)
def test_joint_prediction_equals_separate_predictions(
    volt_options, line_options
) -> None:
    calibrators, tests = _two_families(volt_options, line_options)
    separate = _separate(calibrators, tests)

    intervals = predict_intervals_jointly(calibrators, tests)
    joint = [
        (
            result,
            {k: v.copy() for k, v in calibrator.diagnostics.items()},
            {k: v.copy() for k, v in calibrator.pooled_masks.items()},
        )
        for result, calibrator in zip(intervals, calibrators, strict=True)
    ]

    _assert_same(separate, joint)


def test_joint_prediction_of_unrelated_calibrators_runs_them_separately() -> None:
    first, first_tests = _two_families({"n_eff_target": 60}, {"n_eff_target": 60})
    second, second_tests = _two_families(
        {"n_eff_target": 60}, {"n_eff_target": 60}, seed=9
    )
    calibrators, tests = [first[0], second[1]], [first_tests[0], second_tests[1]]
    separate = _separate(calibrators, tests)

    predict_intervals_jointly(calibrators, tests)
    for calibrator, (_, diagnostics, _) in zip(calibrators, separate, strict=True):
        for key, values in diagnostics.items():
            np.testing.assert_array_equal(calibrator.diagnostics[key], values)


def test_several_targets_equal_one_fit_per_target() -> None:
    calibrators, tests = _two_families({"n_eff_target": 60}, {"n_eff_target": 60})
    targets = (20.0, 60.0, 150.0)

    shared = _predict_targets(calibrators, tests, targets)

    for calibrator, test, by_target in zip(calibrators, tests, shared, strict=True):
        for target in targets:
            calibrator.n_eff_target = target
            calibrator._banks = {}
            expected = calibrator.predict_intervals(test)
            intervals, diagnostics = by_target[target]
            for name in expected:
                np.testing.assert_array_equal(intervals[name][0], expected[name][0])
                np.testing.assert_array_equal(intervals[name][1], expected[name][1])
            for key, values in calibrator.diagnostics.items():
                np.testing.assert_array_equal(diagnostics[key], values)
