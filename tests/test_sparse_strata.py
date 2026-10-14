"""Sparse strata: too few calibration samples in a stratum for a finite bound.

By default (`sparse_strata="pooled"`) such an element gets its quantile over
all calibration scenarios; `sparse_strata="infinite"` gives it infinite bounds,
which keep the per-stratum guarantee. A stratum absent from the calibration
data counts as sparse, for both methods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer.cli import main
from gridfm_trustlayer.conformal import (
    ChannelSet,
    KernelIntervalCalibrator,
    NaiveIntervalCalibrator,
)
from gridfm_trustlayer.io.artifact import (
    load_interval_artifacts,
    save_interval_artifacts,
)
from tests.conformal_reference import conformal_quantile
from tests.test_streaming import _config

ALPHA = 0.2  # a finite bound needs ceil(2 / alpha) - 1 = 9 samples
N_ELEMENTS = 3


def _channel_set(strata: np.ndarray, seed: int) -> ChannelSet:
    rng = np.random.default_rng(seed)
    n = len(strata)
    return ChannelSet(
        scenario_ids=np.arange(n),
        entity_ids=np.arange(N_ELEMENTS),
        entity_name="bus",
        channels={"Vm": rng.normal(size=(n, N_ELEMENTS)).astype(np.float32)},
        scenario_columns={"n_outages": strata},
        scenario_embeddings=rng.normal(size=(n, 4)).astype(np.float32),
    )


def _fit(method: str, sparse_strata: str, predictions, truth):
    if method == "element":
        return NaiveIntervalCalibrator.fit(
            predictions,
            truth,
            alpha=ALPHA,
            channel_names=("Vm",),
            strat="n_outages",
            sparse_strata=sparse_strata,
        )
    return KernelIntervalCalibrator.fit(
        predictions,
        truth,
        alpha=ALPHA,
        channel_names=("Vm",),
        strat="n_outages",
        embeddings=predictions.scenario_embeddings,
        embedding_pooling="mean",
        n_eff_target=40,
        sparse_strata=sparse_strata,
    )


@pytest.fixture
def calibration() -> tuple[ChannelSet, ChannelSet]:
    # Stratum 1: 100 scenarios; stratum 2: 5, too few for a finite bound.
    strata = np.r_[np.ones(100), np.full(5, 2)].astype(np.int64)
    return _channel_set(strata, seed=1), _channel_set(strata, seed=2)


@pytest.mark.parametrize("method", ["element", "kernel"])
def test_sparse_and_unseen_strata_get_infinite_bounds_when_asked(
    calibration, method
) -> None:
    calibrator = _fit(method, "infinite", *calibration)
    # One test scenario per stratum: dense, sparse, and absent from calibration.
    test = _channel_set(np.array([1, 2, 3]), seed=3)
    lower, upper = calibrator.predict_intervals(test)["Vm"]

    assert np.isfinite(lower[0]).all() and np.isfinite(upper[0]).all()
    assert np.isneginf(lower[1:]).all() and np.isposinf(upper[1:]).all()
    assert calibrator.pooled_masks == {}
    assert calibrator.sparse_strata_counts() == {"Vm": {2: N_ELEMENTS}}


@pytest.mark.parametrize("method", ["element", "kernel"])
def test_pooled_sparse_strata_use_the_quantile_of_all_scenarios(
    calibration, method
) -> None:
    predictions, truth = calibration
    calibrator = _fit(method, "pooled", predictions, truth)
    test = _channel_set(np.array([1, 2, 3]), seed=3)
    lower, upper = calibrator.predict_intervals(test)["Vm"]

    residuals = truth.channels["Vm"] - predictions.channels["Vm"]
    pooled_lower = conformal_quantile(residuals, ALPHA / 2)
    pooled_upper = conformal_quantile(residuals, 1 - ALPHA / 2)
    prediction = test.channels["Vm"]
    np.testing.assert_allclose(lower[1:], prediction[1:] + pooled_lower, rtol=1e-6)
    np.testing.assert_allclose(upper[1:], prediction[1:] + pooled_upper, rtol=1e-6)
    np.testing.assert_array_equal(
        calibrator.pooled_masks["Vm"],
        np.array([[False] * N_ELEMENTS, [True] * N_ELEMENTS, [True] * N_ELEMENTS]),
    )
    # The dense stratum is untouched.
    default = _fit(method, "infinite", predictions, truth)
    default_lower, default_upper = default.predict_intervals(test)["Vm"]
    np.testing.assert_array_equal(lower[0], default_lower[0])
    np.testing.assert_array_equal(upper[0], default_upper[0])


@pytest.mark.parametrize("method", ["element", "kernel"])
def test_pooled_is_the_default(calibration, method) -> None:
    predictions, truth = calibration
    calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator
    if method == "element":
        calibrator = NaiveIntervalCalibrator.fit(
            predictions, truth, alpha=ALPHA, channel_names=("Vm",), strat="n_outages"
        )
    else:
        calibrator = KernelIntervalCalibrator.fit(
            predictions,
            truth,
            alpha=ALPHA,
            channel_names=("Vm",),
            strat="n_outages",
            embeddings=predictions.scenario_embeddings,
            embedding_pooling="mean",
            n_eff_target=40,
        )
    assert calibrator.sparse_strata == "pooled"
    lower, upper = calibrator.predict_intervals(_channel_set(np.array([2]), seed=3))[
        "Vm"
    ]
    assert np.isfinite(lower).all() and np.isfinite(upper).all()


def test_kernel_marks_an_unseen_stratum_as_a_fallback(calibration) -> None:
    calibrator = _fit("kernel", "infinite", *calibration)
    calibrator.predict_intervals(_channel_set(np.array([1, 3]), seed=3))
    # The scenario of stratum 3 has no calibration neighbours at all.
    assert bool(calibrator.diagnostics["kernel_fallback"][1])
    assert float(calibrator.diagnostics["n_eff"][1]) == 0.0


@pytest.mark.parametrize("method", ["element", "kernel"])
def test_sparse_strata_survives_the_artifact(tmp_path, calibration, method) -> None:
    calibrator = _fit(method, "pooled", *calibration)
    path = str(tmp_path / "layer.trustlayer")
    save_interval_artifacts({"volt": calibrator}, path)
    assert load_interval_artifacts(path)["volt"].sparse_strata == "pooled"


def test_invalid_sparse_strata_is_rejected(calibration) -> None:
    with pytest.raises(ValueError, match="sparse_strata"):
        _fit("element", "nearest", *calibration)


def _calibrated_vm(tmp_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(next((tmp_path / "run" / "calibrated").glob("*vm_va*")))
    return frame[~frame["Vm_known"]]


def test_cli_sparse_strata(tmp_path, capsys) -> None:
    config = _config(tmp_path, "run", ["cp_type: element", "strat: n_outages"])
    assert main(["compute", "--config", str(config)]) == 0
    assert main(["calibrate", "--config", str(config)]) == 0
    printed = capsys.readouterr().out
    # The synthetic data has rare outage counts, too sparse at alpha = 0.2.
    assert "compute: too few calibration samples for a finite bound: Vm:" in printed
    assert "(sparse_strata: pooled)" in printed
    pooled = _calibrated_vm(tmp_path)
    assert np.isfinite(pooled["Vm_pi_upper"].to_numpy()).all()
    assert pooled["Vm_pooled"].any()
    coverage_dir = tmp_path / "run" / "calibrated" / "coverage"
    by_stratum = pd.read_csv(coverage_dir / "coverage_by_stratum.csv")
    vm = by_stratum[by_stratum["variable"].eq("Vm")].set_index("stratum")
    assert vm.loc["all", "n_pooled"] == int(pooled["Vm_pooled"].sum())
    summary = (coverage_dir / "summary.md").read_text(encoding="utf-8")
    assert "Too few calibration samples for a finite bound" in summary
    assert "with pooled bounds" in summary

    config.write_text(
        config.read_text(encoding="utf-8") + "\nsparse_strata: infinite\n",
        encoding="utf-8",
    )
    assert main(["compute", "--config", str(config)]) == 0
    assert main(["calibrate", "--config", str(config)]) == 0
    assert "(sparse_strata: infinite)" in capsys.readouterr().out
    strict = _calibrated_vm(tmp_path)
    assert "Vm_pooled" not in strict.columns
    # The pooled flag marked exactly the bounds that are infinite without it.
    infinite = ~np.isfinite(strict["Vm_pi_upper"].to_numpy())
    merged = strict[["scenario", "bus"]].assign(infinite=infinite)
    merged = merged.merge(
        pooled[["scenario", "bus", "Vm_pooled"]], on=["scenario", "bus"]
    )
    np.testing.assert_array_equal(merged["Vm_pooled"], merged["infinite"])
