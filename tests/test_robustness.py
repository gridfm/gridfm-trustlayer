"""Inputs that used to give wrong results quietly, or to fail badly."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------- 0.1.1 fixes
DEMO_DATA = Path(__file__).resolve().parents[1] / "examples" / "demo" / "data"


@pytest.mark.skipif(not DEMO_DATA.exists(), reason="needs the demo data")
def test_apply_exits_after_writing_its_outputs(tmp_path) -> None:
    # With pyarrow 26, datasets still cached at exit deadlocked Arrow's thread
    # pool: 0.1.0's apply wrote everything and never exited when its output was
    # piped. Whether it happens depends on what is cached at exit, so this is a
    # smoke test; the fix (io.parquet releases the cache at exit) is general.
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "task_name: PowerFlow",
                "alpha: 0.1",
                "strat: n_outages",
                "variables: [Vm, Va, line_loading]",
                "inputs:",
                "  calibration:",
                *(
                    f"    {key}: {DEMO_DATA / 'calibration' / name}"
                    for key, name in (
                        ("predictions", "case14_ieee_predictions.parquet"),
                        ("pf_node", "raw/bus_data.parquet"),
                        ("pf_edge", "raw/branch_data.parquet"),
                    )
                ),
                "  apply:",
                *(
                    f"    {key}: {DEMO_DATA / 'test' / name}"
                    for key, name in (
                        ("predictions", "case14_ieee_predictions.parquet"),
                        ("pf_node", "raw/bus_data.parquet"),
                        ("pf_edge", "raw/branch_data.parquet"),
                    )
                ),
                "outputs:",
                f"  artifact: {tmp_path / 'layer.trustlayer'}",
                f"  applied_predictions: {tmp_path / 'applied'}",
            ]
        ),
        encoding="utf-8",
    )
    for command in ("compute", "apply"):
        finished = subprocess.run(
            [
                sys.executable,
                "-m",
                "gridfm_trustlayer",
                command,
                "--config",
                str(config),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert finished.returncode == 0, finished.stderr


def test_commands_find_each_others_default_outputs(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, METHODS["kernel_n_outages"], {}, "[Vm, Va]")
    text = config.read_text()
    config.write_text(text.split("outputs:")[0], encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for command in ("compute", "calibrate", "apply"):
        assert main([command, "--config", str(config)]) == 0
    folder = tmp_path / "trustlayer_outputs" / "powerflow" / "kernel"
    assert (folder / "predictions_vm_va.trustlayer").is_file()


def test_missing_input_names_the_config_key(tmp_path, monkeypatch, capsys) -> None:
    config = _config(tmp_path, METHODS["element_n_outages"], {}, "[Vm]")
    config.write_text(
        config.read_text().split("  apply:")[0]
        + "outputs:\n"
        + config.read_text().split("outputs:\n")[1],
        encoding="utf-8",
    )
    monkeypatch.delenv(DEBUG_VARIABLE)
    assert main(["compute", "--config", str(config)]) == 0
    assert main(["apply", "--config", str(config)]) == 2
    assert "Missing inputs.apply.predictions in the config" in capsys.readouterr().err


def _screen_config(tmp_path: Path, method: str, variables: str) -> Path:
    config = _config(tmp_path, METHODS[method], {}, variables)
    config.write_text(
        config.read_text()
        + f"\n  screen_dir: {tmp_path / 'screen'}\n"
        + "screen_bounds:\n  Vm: {lower: 0.98, upper: 1.02}\n"
        + "  line_loading: {lower: 0.5, upper: 1.0}\n",
        encoding="utf-8",
    )
    return config


def test_screen_refuses_a_variable_that_was_not_calibrated(tmp_path) -> None:
    config = _screen_config(tmp_path, "element_n_outages", "[Vm, Va]")
    for command in ("compute", "calibrate"):
        assert main([command, "--config", str(config)]) == 0
    with pytest.raises(ValueError, match="line_loading not found in the calibrated"):
        main(["screen", "--config", str(config), "--variables", "Vm,line_loading"])


def test_screen_and_apply_ignore_a_line_loading_lower_limit_alike(tmp_path) -> None:
    config = _screen_config(tmp_path, "element_n_outages", "[line_loading]")
    for command in ("compute", "calibrate", "screen", "apply"):
        assert main([command, "--config", str(config)]) == 0
    screened = pd.read_csv(tmp_path / "screen" / "scenarios.csv").set_index("scenario")
    applied = pd.read_csv(next(tmp_path.rglob("status/scenarios.csv"))).set_index(
        "scenario"
    )
    pd.testing.assert_series_equal(
        screened["status"], applied.loc[screened.index, "status"]
    )


def test_switching_between_one_and_two_families_replaces_the_outputs(
    tmp_path,
) -> None:
    config = _screen_config(tmp_path, "element_n_outages", "[Vm, Va]")
    volt = config.read_text()
    mixed = volt.replace("variables: [Vm, Va]", "variables: [Vm, Va, line_loading]")
    for text, screened in ((volt, "Vm"), (mixed, "Vm,line_loading"), (volt, "Vm")):
        config.write_text(text, encoding="utf-8")
        for command in ("compute", "calibrate"):
            assert main([command, "--config", str(config)]) == 0
        assert main(["screen", "--config", str(config), "--variables", screened]) == 0
    # The second volt run left no line-loading report behind.
    assert not list((tmp_path / "coverage").glob("*line_loading*"))
    assert not list((tmp_path / "screen").glob("*line_loading*"))


def test_compute_refuses_to_overwrite_a_file_that_is_not_an_artifact(
    tmp_path,
) -> None:
    config = _config(tmp_path, METHODS["element_n_outages"], {}, "[Vm]")
    victim = tmp_path / "victim.parquet"
    victim.write_bytes(b"not an artifact")
    with pytest.raises(ValueError, match="is not a trustlayer artifact"):
        main(["compute", "--config", str(config), "--output", str(victim)])
    assert victim.read_bytes() == b"not an artifact"


def test_calibrate_and_apply_report_pooled_bounds(tmp_path, capsys) -> None:
    config = _config(tmp_path, METHODS["element_n_outages"], {}, "[Vm, Va]")
    for command in ("compute", "calibrate", "apply"):
        assert main([command, "--config", str(config)]) == 0
    printed = capsys.readouterr().out
    # 60 synthetic scenarios over several outage counts: some strata are sparse.
    assert "calibrate: Vm:" in printed and "use pooled bounds" in printed
    assert "apply: Vm:" in printed


def test_apply_and_screen_agree_on_missing_predictions(tmp_path) -> None:
    config = _screen_config(tmp_path, "element_n_outages", "[Vm, line_loading]")
    test = tmp_path / "test" / "predictions.parquet"
    frame = pd.read_parquet(test)
    frame.loc[frame.index[::7], "Vm_pred"] = np.nan  # some predictions missing
    frame.to_parquet(test, index=False)
    for command in ("compute", "calibrate", "screen", "apply"):
        assert main([command, "--config", str(config)]) == 0
    screened = pd.read_csv(tmp_path / "screen" / "scenarios.csv").set_index("scenario")
    applied = pd.read_csv(next(tmp_path.rglob("status/scenarios.csv"))).set_index(
        "scenario"
    )
    for column in ("status", "point_status"):
        pd.testing.assert_series_equal(
            screened[column], applied.loc[screened.index, column]
        )
    assert (applied["point_status"] == "uncertain").any()


def test_line_bounds_do_not_depend_on_the_true_values(tmp_path) -> None:
    config = _config(tmp_path, METHODS["element_n_outages"], {}, "[line_loading]")
    bus = tmp_path / "test" / "raw" / "bus_data.parquet"
    frame = pd.read_parquet(bus)
    frame.loc[frame["scenario"] == 25, "Vm"] = np.nan  # an unsolved scenario
    shutil.rmtree(bus)
    frame.to_parquet(bus, partition_cols=["scenario_partition"])
    for command in ("compute", "calibrate", "apply"):
        assert main([command, "--config", str(config)]) == 0
    # One family: each output is the dataset itself.
    calibrated, applied = (
        pd.read_parquet(tmp_path / name)
        .sort_values(["scenario", "branch"])
        .reset_index(drop=True)
        for name in ("calibrated", "applied")
    )
    bounds = ["line_loading_pi_lower", "line_loading_pi_upper"]
    pd.testing.assert_frame_equal(calibrated[bounds], applied[bounds])


def test_one_nan_embedding_is_refused(tmp_path) -> None:
    config = _config(tmp_path, METHODS["kernel_n_outages"], {}, "[Vm]")
    embeddings = tmp_path / "calibration" / "embeddings.parquet"
    frame = pd.read_parquet(embeddings)
    frame.loc[3, "emb_001"] = np.nan
    frame.to_parquet(embeddings, index=False)
    with pytest.raises(ValueError, match="NaN or infinite values in 1 scenario"):
        main(["compute", "--config", str(config)])


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ("Vm: {lower: 1.05, upper: 0.95}", "must be below its upper limit"),
        ("Vm: {lower: 0.95, uper: 1.05}", "unknown keys"),
        ("Vm: {lower: low, upper: 1.05}", "screen_bounds.Vm.lower must be a number"),
    ],
)
def test_screen_bounds_are_checked(tmp_path, bounds, message) -> None:
    config = _config(tmp_path, METHODS["element_n_outages"], {}, "[Vm]")
    config.write_text(
        config.read_text() + f"\nscreen_bounds:\n  {bounds}\n", encoding="utf-8"
    )
    assert main(["compute", "--config", str(config)]) == 0
    with pytest.raises(ValueError, match=message):
        main(["apply", "--config", str(config)])


def test_config_values_are_checked(tmp_path) -> None:
    for line, message in (
        ("alpha: 1.5", "alpha must be between 0 and 1"),
        ("version: 2", "Unsupported config version"),
    ):
        config = _config(tmp_path / line[:5], METHODS["element_n_outages"], {}, "[Vm]")
        config.write_text(
            config.read_text().replace("alpha: 0.2", line)
            if line.startswith("alpha")
            else config.read_text().replace("version: 1", line),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=message):
            main(["compute", "--config", str(config)])


def test_task_name_spelling(tmp_path, capsys) -> None:
    from gridfm_trustlayer import load_predictions

    paths = _config(tmp_path, METHODS["element_n_outages"], {}, "[Vm]")
    predictions = tmp_path / "calibration" / "predictions.parquet"
    assert load_predictions(str(predictions), task_name="Power Flow").task_name == (
        "PowerFlow"
    )
    paths.write_text(
        paths.read_text().replace("task_name: PowerFlow", "task_name: PowerFlows"),
        encoding="utf-8",
    )
    assert main(["compute", "--config", str(paths)]) == 0
    assert "is not PowerFlow or OptimalPowerFlow" in capsys.readouterr().out


def test_python_api_inputs_are_checked() -> None:
    values = np.zeros((4, 2), dtype=np.float32)
    common: dict[str, Any] = {
        "scenario_ids": np.arange(4),
        "entity_ids": np.arange(2),
        "entity_name": "bus",
    }
    with pytest.raises(ValueError, match="expected"):
        ChannelSet(channels={"Vm": np.zeros(4)}, **common)
    with pytest.raises(ValueError, match="scenario column"):
        ChannelSet(
            channels={"Vm": values}, scenario_columns={"k": np.array([1, 2])}, **common
        )
    channel_set = ChannelSet(channels={"Vm": values}, valid_masks=None, **common)  # type: ignore[arg-type]
    assert channel_set.valid_masks == {}
    strata = {"k": np.array([1.0, 1.5, 2.0, 2.0])}
    with pytest.raises(ValueError, match="whole numbers"):
        NaiveIntervalCalibrator.fit(
            ChannelSet(channels={"Vm": values}, scenario_columns=strata, **common),
            ChannelSet(channels={"Vm": values}, scenario_columns=strata, **common),
            alpha=0.5,
            channel_names=("Vm",),
            strat="k",
        )
    assert scenario_status(np.empty((3, 0), dtype=object)).tolist() == ["secure"] * 3


def test_build_config_refuses_unknown_variables() -> None:
    from gridfm_trustlayer.interactive import build_config

    paths = {"predictions": "p.parquet", "pf_node": "b.parquet", "pf_edge": "e.parquet"}
    with pytest.raises(ValueError, match="Unsupported variables"):
        build_config(calibration=paths, test=paths, variables=["Vm", "Pg"])


def test_screen_leaves_out_values_without_a_true_solution(tmp_path) -> None:
    config = _screen_config(tmp_path, "element_n_outages", "[Vm]")
    test = tmp_path / "test" / "predictions.parquet"
    frame = pd.read_parquet(test)
    unsolved = frame.index[frame["PQ"] == 1][::5]
    frame.loc[unsolved, "Vm_target"] = np.nan
    frame.to_parquet(test, index=False)
    for command in ("compute", "calibrate", "screen"):
        assert main([command, "--config", str(config)]) == 0
    table = pd.read_csv(tmp_path / "screen" / "screening_by_stratum.csv")
    overall = table[table["stratum"] == "all"].iloc[0]
    known = int((frame["PV"] == 1).sum())
    assert int(overall["n_values"]) == len(frame) - known - len(unsolved)
