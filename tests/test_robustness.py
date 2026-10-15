"""Inputs that used to give wrong results quietly, or to fail badly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer import (
    ChannelSet,
    KernelIntervalCalibrator,
    NaiveIntervalCalibrator,
    classify_status,
    scenario_status,
)
from gridfm_trustlayer.cli import main
from gridfm_trustlayer.cli.config import validate_command_config
from gridfm_trustlayer.cli.main import DEBUG_VARIABLE
from gridfm_trustlayer.conformal.tuning import TuningOptions
from gridfm_trustlayer.io.artifact import load_ground_truth_reference
from tests.test_apply import _config, _rows
from tests.test_streaming import METHODS


def _single_scenario(test: Path, scenario: int, target: Path) -> dict[str, Path]:
    """One scenario of a dataset, alone, as new scenarios may come."""

    target.mkdir()
    paths = {
        "predictions": test / "predictions.parquet",
        "pf_node": test / "raw" / "bus_data.parquet",
        "pf_edge": test / "raw" / "branch_data.parquet",
        "gen": test / "raw" / "gen_data.parquet",
        "embeddings": test / "embeddings.parquet",
    }
    out = {}
    for name, path in paths.items():
        frame = pd.read_parquet(path)
        frame = frame[frame["scenario"] == scenario].drop(
            columns=["scenario_partition"], errors="ignore"
        )
        file_name = "gen_data.parquet" if name == "gen" else f"{name}.parquet"
        out[name] = target / file_name
        frame.to_parquet(out[name], index=False)
    return out


@pytest.mark.parametrize("method", ["element_n_outages", "kernel_n_outages"])
def test_a_scenario_has_the_same_stratum_alone_as_in_its_dataset(
    tmp_path, method
) -> None:
    config = _config(tmp_path, METHODS[method], {}, "[Vm, Va, line_loading]")
    for command in ("compute", "calibrate"):
        assert main([command, "--config", str(config)]) == 0
    stored = load_ground_truth_reference(str(tmp_path / "layer.trustlayer"))
    assert stored is not None and stored["va_in_degrees"] is True

    calibrated = _rows(tmp_path / "calibrated", "volt")
    outages = calibrated.groupby("scenario")["n_outages"].first()
    # In its own dataset, an out-of-service element is in service elsewhere;
    # alone, nothing shows it exists. Its outages must count all the same.
    scenario = int(outages[outages >= 2].index[0])
    alone = _single_scenario(tmp_path / "test", scenario, tmp_path / "alone")
    head, apply_block = config.read_text().split("  apply:\n")
    for key in ("predictions", "pf_node", "pf_edge", "embeddings"):
        apply_block = apply_block.replace(
            str(
                tmp_path
                / "test"
                / {
                    "predictions": "predictions.parquet",
                    "pf_node": "raw/bus_data.parquet",
                    "pf_edge": "raw/branch_data.parquet",
                    "embeddings": "embeddings.parquet",
                }[key]
            ),
            str(alone[key]),
        )
    config.write_text(head + "  apply:\n" + apply_block, encoding="utf-8")
    assert main(["apply", "--config", str(config)]) == 0

    applied = _rows(tmp_path / "applied", "volt")
    assert applied["n_outages"].unique().tolist() == [outages[scenario]]
    bounds = [c for c in applied.columns if c.endswith(("_pi_lower", "_pi_upper"))]
    expected = calibrated[calibrated["scenario"] == scenario].reset_index(drop=True)
    pd.testing.assert_frame_equal(applied[bounds], expected[bounds])


def _channel_set(values: np.ndarray, **extra) -> ChannelSet:
    n, m = values.shape
    return ChannelSet(
        scenario_ids=np.arange(n),
        entity_ids=np.arange(m),
        entity_name="bus",
        channels={"Vm": values.astype(np.float32)},
        **extra,
    )


def test_missing_true_values_are_left_out_of_the_fit() -> None:
    rng = np.random.default_rng(0)
    predicted = rng.normal(size=(200, 3)).astype(np.float32)
    true = predicted + rng.normal(scale=0.1, size=predicted.shape).astype(np.float32)
    gappy = true.copy()
    gappy[:5, 0] = np.nan  # missing solutions
    gappy[5, 1] = np.inf
    embeddings = rng.normal(size=(200, 2)).astype(np.float32)

    element = NaiveIntervalCalibrator.fit(
        _channel_set(predicted), _channel_set(gappy), alpha=0.1, channel_names=("Vm",)
    )
    assert np.isfinite(element.upper_residual_quantiles["Vm"]).all()
    assert np.isfinite(element.lower_residual_quantiles["Vm"]).all()

    def kernel(truth: np.ndarray) -> KernelIntervalCalibrator:
        return KernelIntervalCalibrator.fit(
            _channel_set(predicted, scenario_embeddings=embeddings),
            _channel_set(truth),
            alpha=0.1,
            channel_names=("Vm",),
            embeddings=embeddings,
            embedding_pooling="mean",
            n_eff_target=50,
        )

    fitted = kernel(gappy)
    valid = fitted.valid_masks["Vm"]
    assert not valid[:5, 0].any() and not valid[5, 1]
    # The same as fitting without those scenarios' values: a missing truth is not
    # a perfect prediction.
    assert valid.sum() == valid.size - 6


def test_a_missing_bound_is_uncertain_and_missing_statuses_are_ignored() -> None:
    statuses = classify_status(
        np.array([0.5, np.nan, 0.5]), np.array([0.9, 0.9, np.nan]), upper_limit=1.0
    )
    assert statuses.tolist() == ["within_limits", "uncertain", "uncertain"]

    # `apply` writes no status for known values and out-of-service lines.
    rows = np.array([["within_limits", None], ["violation", np.nan]], dtype=object)
    assert scenario_status(rows).tolist() == ["secure", "insecure"]
    valid = np.array([[True, False], [False, False]])
    assert scenario_status(rows, valid=valid).tolist() == ["secure", "secure"]
    with pytest.raises(KeyError):
        scenario_status(np.array([["not a status"]], dtype=object))


def _kernel_diagnostics(width: int, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Fit on 400 scenarios and apply to 50, with `valid` over all 450."""

    rng = np.random.default_rng(1)
    n_calibration = 400
    embeddings = rng.normal(size=(valid.shape[0], width)).astype(np.float32)
    # Drawn for two elements, so that one element alone gets the same values.
    predicted = rng.normal(size=(valid.shape[0], 2)).astype(np.float32)
    true = predicted + rng.normal(scale=0.1, size=predicted.shape).astype(np.float32)
    predicted, true = predicted[:, : valid.shape[1]], true[:, : valid.shape[1]]
    calibration, test = slice(0, n_calibration), slice(n_calibration, None)
    calibrator = KernelIntervalCalibrator.fit(
        _channel_set(
            predicted[calibration],
            valid_masks={"Vm": valid[calibration]},
            scenario_embeddings=embeddings[calibration],
        ),
        _channel_set(true[calibration]),
        alpha=0.1,
        channel_names=("Vm",),
        embeddings=embeddings[calibration],
        embedding_pooling="mean",
        n_eff_target=40,
    )
    calibrator.predict_intervals(
        _channel_set(
            predicted[test],
            valid_masks={"Vm": valid[test]},
            scenario_embeddings=embeddings[test],
        )
    )
    return calibrator.diagnostics


def test_kernel_n_eff_is_reported_for_wide_embeddings() -> None:
    # Far apart in 512 dimensions, the weights underflow in float32 (and the
    # level is out of reach, so the bounds fall back); the reported n_eff is
    # still the target the bandwidth was solved for.
    diagnostics = _kernel_diagnostics(512, np.ones((450, 2), dtype=bool))
    np.testing.assert_allclose(diagnostics["n_eff"], 40.0, rtol=1e-4)


def test_a_known_value_does_not_make_a_scenario_fall_back() -> None:
    # Element 1 is a set-point: valid in a few calibration scenarios only, too
    # few to reach the level, and known (no interval) in every test scenario,
    # as Vm at a generator bus.
    valid = np.ones((450, 2), dtype=bool)
    valid[10:, 1] = False
    without_it = _kernel_diagnostics(2, valid[:, :1])["kernel_fallback"]
    with_it = _kernel_diagnostics(2, valid)["kernel_fallback"]
    assert not without_it.all()
    np.testing.assert_array_equal(with_it, without_it)
    # Where the element does need a bound, the fallback is reported.
    valid[400:, 1] = True
    assert _kernel_diagnostics(2, valid)["kernel_fallback"].all()


def test_tuning_set_for_compute_only_does_not_break_the_other_commands() -> None:
    config = {
        "cp_type": "kernel",
        "n_eff_target": 100.0,
        "tuning": {"grid": [50, 100]},
        "embeddings": "e.parquet",
    }
    validate_command_config("calibrate", config)
    with pytest.raises(ValueError, match="tuning block only applies"):
        validate_command_config("compute", config)


def test_random_search_range_must_hold_a_whole_number() -> None:
    with pytest.raises(ValueError, match="whole number"):
        TuningOptions.from_config({"search": "random", "n_eff_range": [1.5, 1.7]})


def test_cli_reports_errors_in_one_line(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv(DEBUG_VARIABLE)
    missing = tmp_path / "missing.yaml"
    assert main(["calibrate", "--config", str(missing)]) == 2
    error = capsys.readouterr().err.strip()
    assert error.startswith("gridfm_trustlayer calibrate: error:")
    assert "missing.yaml" in error and "Traceback" not in error


def test_pred_bins_tables_have_one_overall_row_per_variable(tmp_path) -> None:
    config = _config(tmp_path, METHODS["element_pred_bins"], {}, "[Vm, Va]")
    for command in ("compute", "calibrate"):
        assert main([command, "--config", str(config)]) == 0
    table = pd.read_csv(tmp_path / "coverage" / "coverage_by_stratum.csv")
    overall = table[table["stratum"] == "all"]
    assert overall["variable"].value_counts().max() == 1
    assert table["coverage"].notna().all()
