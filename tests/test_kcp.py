"""Tests for kernel-weighted conformal interval calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gridfm_trustlayer.conformal import ChannelSet, KernelIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import DEFAULT_N_EFF_TARGET_SCALE
from gridfm_trustlayer.io.artifact import (
    load_interval_artifacts,
    save_interval_artifacts,
)
from gridfm_trustlayer.io.embeddings import (
    POOLING_REGISTRY,
    load_scenario_embeddings,
    validate_pooling,
)

N_ELEMENTS = 4
EMBEDDING_DIM = 3


def _channel_set(
    predictions: np.ndarray,
    embeddings: np.ndarray,
    strata: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> ChannelSet:
    return ChannelSet(
        scenario_ids=np.arange(len(predictions), dtype=np.int64),
        entity_ids=np.arange(N_ELEMENTS, dtype=np.int64),
        entity_name="bus",
        channels={"Vm": predictions.astype(np.float32)},
        valid_masks={} if valid is None else {"Vm": valid},
        scenario_columns={"n_outages": strata},
        scenario_embeddings=embeddings.astype(np.float32),
    )


def _make_sets(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, EMBEDDING_DIM)).astype(np.float32)
    strata = rng.integers(0, 2, size=n).astype(np.int64)
    predictions = rng.normal(size=(n, N_ELEMENTS)).astype(np.float32)
    # Error magnitude is driven by the first embedding coordinate, so a kernel
    # localized in embedding space should recover the local error scale.
    scale = (0.1 + np.abs(embeddings[:, 0]))[:, None]
    targets = predictions + (rng.normal(size=(n, N_ELEMENTS)) * scale).astype(
        np.float32
    )
    return (
        _channel_set(predictions, embeddings, strata),
        _channel_set(targets, embeddings, strata),
        embeddings,
    )


def _fit(
    alpha: float = 0.1, **kwargs: Any
) -> tuple[KernelIntervalCalibrator, ChannelSet]:
    predictions, ground_truth, embeddings = _make_sets()
    calibrator = KernelIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=alpha,
        channel_names=("Vm",),
        strat="n_outages",
        embeddings=embeddings,
        embedding_pooling="mean",
        **kwargs,
    )
    return calibrator, predictions


def test_intervals_bracket_predictions_and_vary_by_scenario() -> None:
    calibrator, predictions = _fit()
    lower, upper = calibrator.predict_intervals(predictions)["Vm"]

    assert lower.shape == predictions.channels["Vm"].shape
    assert np.all(lower <= upper)
    # A kernel-weighted quantile must differ across test points; an identical
    # offset everywhere would mean the localization silently collapsed to SCP.
    offsets = upper - predictions.channels["Vm"]
    assert offsets.std(axis=0).max() > 0.0


def test_empirical_coverage_meets_target() -> None:
    predictions, ground_truth, embeddings = _make_sets()
    calibrator = KernelIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.1,
        channel_names=("Vm",),
        strat="n_outages",
        embeddings=embeddings,
        embedding_pooling="mean",
    )
    lower, upper = calibrator.predict_intervals(predictions)["Vm"]
    targets = ground_truth.channels["Vm"]
    coverage = ((targets >= lower) & (targets <= upper)).mean()
    assert coverage >= 0.9


def test_low_effective_sample_size_falls_back_to_stratum_quantiles() -> None:
    # n_eff can never reach this threshold, so every scenario must fall back.
    calibrator, predictions = _fit(n_eff_min=1e9)
    _, upper = calibrator.predict_intervals(predictions)["Vm"]

    assert calibrator.diagnostics["kernel_fallback"].all()
    strata = predictions.scenario_columns["n_outages"]
    for index, stratum_value in enumerate(calibrator.stratification_values):
        rows = strata == stratum_value
        expected_upper = calibrator.fallback_upper_quantiles["Vm"][index]
        offsets = (upper - predictions.channels["Vm"])[rows]
        np.testing.assert_allclose(
            offsets, np.broadcast_to(expected_upper, offsets.shape), atol=1e-6
        )


def test_diagnostics_are_reported_per_scenario() -> None:
    calibrator, predictions = _fit()
    calibrator.predict_intervals(predictions)
    diagnostics = calibrator.diagnostics

    n_scenarios = len(predictions.scenario_ids)
    assert set(diagnostics) == {"n_eff", "kernel_bandwidth_sq", "kernel_fallback"}
    for values in diagnostics.values():
        assert values.shape == (n_scenarios,)
    assert np.all(diagnostics["n_eff"] >= 0.0)
    assert np.all(diagnostics["kernel_bandwidth_sq"] > 0.0)


def test_invalid_entries_carry_no_kernel_weight() -> None:
    predictions, ground_truth, embeddings = _make_sets()
    valid = np.ones(predictions.channels["Vm"].shape, dtype=bool)
    valid[:, 0] = False
    # Poison the masked column: if invalid rows leaked into the weighted
    # quantile the resulting offsets would be enormous.
    poisoned = ground_truth.channels["Vm"].copy()
    poisoned[:, 0] += 1e4
    predictions = _channel_set(
        predictions.channels["Vm"],
        embeddings,
        predictions.scenario_columns["n_outages"],
        valid=valid,
    )
    ground_truth = _channel_set(
        poisoned,
        embeddings,
        ground_truth.scenario_columns["n_outages"],
        valid=valid,
    )
    calibrator = KernelIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.1,
        channel_names=("Vm",),
        strat="n_outages",
        embeddings=embeddings,
        embedding_pooling="mean",
    )
    lower, upper = calibrator.predict_intervals(predictions)["Vm"]
    # Masked entries collapse to the point prediction, valid ones stay sane.
    np.testing.assert_allclose(lower[:, 0], predictions.channels["Vm"][:, 0])
    np.testing.assert_allclose(upper[:, 0], predictions.channels["Vm"][:, 0])
    assert np.abs(upper[:, 1:] - predictions.channels["Vm"][:, 1:]).max() < 10.0


def test_chunking_does_not_change_results() -> None:
    calibrator, predictions = _fit()
    single_threaded = calibrator.predict_intervals(predictions)["Vm"]
    calibrator.n_jobs = 4
    multi_threaded = calibrator.predict_intervals(predictions)["Vm"]
    np.testing.assert_array_equal(single_threaded[0], multi_threaded[0])
    np.testing.assert_array_equal(single_threaded[1], multi_threaded[1])


def test_pred_bins_stratification_is_rejected() -> None:
    predictions, ground_truth, embeddings = _make_sets()
    with pytest.raises(ValueError, match="pred_bins"):
        KernelIntervalCalibrator.fit(
            predictions,
            ground_truth,
            alpha=0.1,
            channel_names=("Vm",),
            strat="pred_bins",
            embeddings=embeddings,
            embedding_pooling="mean",
        )


def test_missing_test_embeddings_raise() -> None:
    calibrator, predictions = _fit()
    predictions.scenario_embeddings = None
    with pytest.raises(ValueError, match="requires scenario embeddings"):
        calibrator.predict_intervals(predictions)


def test_artifact_roundtrip_preserves_intervals(tmp_path) -> None:
    calibrator, predictions = _fit()
    expected = calibrator.predict_intervals(predictions)["Vm"]

    artifact_path = tmp_path / "kernel.trustlayer"
    save_interval_artifacts({"volt": calibrator}, str(artifact_path))
    loaded = load_interval_artifacts(str(artifact_path))["volt"]

    assert isinstance(loaded, KernelIntervalCalibrator)
    assert loaded.k_nn == calibrator.k_nn
    assert loaded.embedding_pooling == calibrator.embedding_pooling
    actual = loaded.predict_intervals(predictions)["Vm"]
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_embedding_pooling_strategies(tmp_path) -> None:
    import pandas as pd  # type: ignore[import-untyped]

    n_scenarios, n_buses, dim = 5, 3, 4
    rng = np.random.default_rng(0)
    values = rng.normal(size=(n_scenarios, n_buses, dim)).astype(np.float32)
    frame = pd.DataFrame(
        {
            "scenario": np.repeat(np.arange(n_scenarios), n_buses),
            "bus": np.tile(np.arange(n_buses), n_scenarios),
        }
    )
    flat = values.reshape(-1, dim)
    for index in range(dim):
        frame[f"emb_{index:03d}"] = flat[:, index]
    path = tmp_path / "emb.parquet"
    frame.to_parquet(path, index=False)

    expected_dims = {
        "mean": dim,
        "max": dim,
        "meanmax": 2 * dim,
        "concat": n_buses * dim,
    }
    for pooling in POOLING_REGISTRY:
        loaded = load_scenario_embeddings(str(path), pooling=pooling)
        assert loaded.dim == expected_dims[pooling]
        assert loaded.scenario_ids.tolist() == list(range(n_scenarios))
    np.testing.assert_allclose(
        load_scenario_embeddings(str(path), pooling="mean").embeddings,
        values.mean(axis=1),
        rtol=1e-6,
    )


def test_validate_pooling_rejects_unknown() -> None:
    assert validate_pooling(None) == "mean"
    with pytest.raises(ValueError, match="Unsupported embedding_pooling"):
        validate_pooling("median")


def test_n_eff_bandwidth_hits_its_target() -> None:
    calibrator, predictions = _fit(bandwidth="n_eff", n_eff_target=120.0)
    calibrator.predict_intervals(predictions)
    n_eff = calibrator.diagnostics["n_eff"]
    # The solve is a bisection to a relative tolerance, so allow a small band.
    np.testing.assert_allclose(n_eff, 120.0, rtol=5e-3)


def test_n_eff_bandwidth_is_invariant_to_embedding_width() -> None:
    """The point of `bandwidth: n_eff`: k_nn drifts with dimension, n_eff does not."""

    spreads: dict[str, list[float]] = {}
    for dim in (2, 64):
        rng = np.random.default_rng(0)
        n = 400
        embeddings = rng.normal(size=(n, dim)).astype(np.float32)
        strata = np.zeros(n, dtype=np.int64)
        preds = rng.normal(size=(n, N_ELEMENTS)).astype(np.float32)
        scale = (0.1 + np.abs(embeddings[:, 0]))[:, None]
        targets = preds + (rng.normal(size=(n, N_ELEMENTS)) * scale).astype(np.float32)
        prediction_set = _channel_set(preds, embeddings, strata)
        truth_set = _channel_set(targets, embeddings, strata)
        modes: list[tuple[str, dict[str, Any]]] = [
            ("knn", {"bandwidth": "knn", "k_nn": 20}),
            ("n_eff", {"bandwidth": "n_eff", "n_eff_target": 100.0}),
        ]
        for mode, kwargs in modes:
            calibrator = KernelIntervalCalibrator.fit(
                prediction_set,
                truth_set,
                alpha=0.1,
                channel_names=("Vm",),
                strat="n_outages",
                embeddings=embeddings,
                embedding_pooling="mean",
                **kwargs,
            )
            calibrator.predict_intervals(prediction_set)
            spreads.setdefault(mode, []).append(
                float(np.median(calibrator.diagnostics["n_eff"]))
            )

    knn_low, knn_high = spreads["knn"]
    n_eff_low, n_eff_high = spreads["n_eff"]
    # k_nn localization collapses as distances concentrate...
    assert knn_high > 2.0 * knn_low
    # ...while the n_eff solve holds the target regardless of width.
    np.testing.assert_allclose([n_eff_low, n_eff_high], 100.0, rtol=5e-3)


def test_n_eff_target_defaults_to_twenty_over_alpha() -> None:
    calibrator, _ = _fit(alpha=0.2, bandwidth="n_eff")
    assert calibrator.n_eff_target == DEFAULT_N_EFF_TARGET_SCALE / 0.2


def test_n_eff_target_saturates_at_stratum_size() -> None:
    # Target far larger than any stratum: the solve should saturate rather than fail.
    calibrator, predictions = _fit(bandwidth="n_eff", n_eff_target=1e6)
    lower, upper = calibrator.predict_intervals(predictions)["Vm"]
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    strata = predictions.scenario_columns["n_outages"]
    largest_stratum = max(int((strata == v).sum()) for v in np.unique(strata))
    assert calibrator.diagnostics["n_eff"].max() <= largest_stratum + 1e-3


def test_unsupported_bandwidth_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported bandwidth"):
        _fit(bandwidth="silverman")


def test_n_eff_target_below_n_eff_min_is_rejected() -> None:
    with pytest.raises(ValueError, match="below n_eff_min"):
        _fit(bandwidth="n_eff", n_eff_target=3.0, n_eff_min=10.0)


def test_bandwidth_survives_artifact_roundtrip(tmp_path) -> None:
    calibrator, predictions = _fit(bandwidth="n_eff", n_eff_target=150.0)
    expected = calibrator.predict_intervals(predictions)["Vm"]

    artifact_path = tmp_path / "kernel_n_eff.trustlayer"
    save_interval_artifacts({"volt": calibrator}, str(artifact_path))
    loaded = load_interval_artifacts(str(artifact_path))["volt"]

    assert isinstance(loaded, KernelIntervalCalibrator)
    assert loaded.bandwidth == "n_eff"
    assert loaded.n_eff_target == 150.0
    actual = loaded.predict_intervals(predictions)["Vm"]
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_knn_remains_the_default() -> None:
    calibrator, _ = _fit()
    assert calibrator.bandwidth == "knn"


def test_cosine_distance_projects_onto_the_unit_sphere() -> None:
    calibrator, _ = _fit(distance="cosine")
    norms = np.linalg.norm(calibrator.calibration_embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_euclidean_remains_the_default() -> None:
    calibrator, _ = _fit()
    assert calibrator.distance == "euclidean"
    norms = np.linalg.norm(calibrator.calibration_embeddings, axis=1)
    assert norms.std() > 0.0


def test_cosine_changes_the_resulting_intervals() -> None:
    euclidean, predictions = _fit()
    cosine, _ = _fit(distance="cosine")
    euclidean_upper = euclidean.predict_intervals(predictions)["Vm"][1]
    cosine_upper = cosine.predict_intervals(predictions)["Vm"][1]
    assert not np.allclose(euclidean_upper, cosine_upper)


def test_cosine_keeps_valid_coverage() -> None:
    predictions, ground_truth, embeddings = _make_sets()
    calibrator = KernelIntervalCalibrator.fit(
        predictions,
        ground_truth,
        alpha=0.1,
        channel_names=("Vm",),
        strat="n_outages",
        embeddings=embeddings,
        embedding_pooling="mean",
        distance="cosine",
    )
    lower, upper = calibrator.predict_intervals(predictions)["Vm"]
    targets = ground_truth.channels["Vm"]
    assert ((targets >= lower) & (targets <= upper)).mean() >= 0.9


def test_cosine_tolerates_a_scenario_on_the_mean() -> None:
    # A standardized row of exact zeros has no direction; it must not produce NaN.
    predictions, ground_truth, embeddings = _make_sets()
    embeddings = embeddings.copy()
    embeddings[0] = embeddings.mean(axis=0)
    calibrator = KernelIntervalCalibrator.fit(
        _channel_set(
            predictions.channels["Vm"],
            embeddings,
            predictions.scenario_columns["n_outages"],
        ),
        _channel_set(
            ground_truth.channels["Vm"],
            embeddings,
            ground_truth.scenario_columns["n_outages"],
        ),
        alpha=0.1,
        channel_names=("Vm",),
        strat="n_outages",
        embeddings=embeddings,
        embedding_pooling="mean",
        distance="cosine",
    )
    assert np.all(np.isfinite(calibrator.calibration_embeddings))


def test_unsupported_distance_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported distance"):
        _fit(distance="manhattan")


def test_distance_survives_artifact_roundtrip(tmp_path) -> None:
    calibrator, predictions = _fit(distance="cosine")
    expected = calibrator.predict_intervals(predictions)["Vm"]

    artifact_path = tmp_path / "kernel_cosine.trustlayer"
    save_interval_artifacts({"volt": calibrator}, str(artifact_path))
    loaded = load_interval_artifacts(str(artifact_path))["volt"]

    assert isinstance(loaded, KernelIntervalCalibrator)
    assert loaded.distance == "cosine"
    actual = loaded.predict_intervals(predictions)["Vm"]
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])
