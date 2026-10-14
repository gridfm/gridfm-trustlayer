"""CLI tests for naive Vm/Va interval computation and application."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer.cli import main
from gridfm_trustlayer.cli.config import load_cli_config, normalize_variables
from gridfm_trustlayer.io import (
    compute_line_loading,
    load_ground_truth,
    load_predictions,
)
from tests.conformal_reference import conformal_quantile


def test_compute_then_calibrate_roundtrip(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(
        config_path, predictions_path, test_predictions_path, pf_node_path, pf_edge_path
    )

    assert (
        main(
            [
                "compute",
                "--config",
                str(config_path),
                "--predictions",
                str(predictions_path),
                "--pf-node",
                str(pf_node_path),
                "--pf-edge",
                str(pf_edge_path),
                "--alpha",
                "0.5",
                "--output",
                str(artifact_path),
                "--task-name",
                "StateEstimation",
            ],
        )
        == 0
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["method"] == "naive_residual_quantiles"
    assert artifact["channel_names"] == ["Vm", "Va"]

    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--predictions",
                str(test_predictions_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
                "--task-name",
                "StateEstimation",
            ],
        )
        == 0
    )

    output_frame = pd.read_parquet(output_path).sort_values(["scenario", "bus"])
    assert {
        "Vm_pi_lower",
        "Vm_pi_upper",
        "Va_pi_lower",
        "Va_pi_upper",
    }.issubset(output_frame.columns)

    calibration_frame = pd.read_parquet(predictions_path).sort_values(
        ["scenario", "bus"]
    )
    test_frame = pd.read_parquet(test_predictions_path).sort_values(["scenario", "bus"])
    vm_residuals = (
        pd.read_parquet(pf_node_path).sort_values(["scenario", "bus"])["Vm"].to_numpy()
        - calibration_frame["Vm_pred"].to_numpy()
    ).reshape(4, 2)
    # Four calibration scenarios at alpha = 0.5: finite split-conformal ranks.
    expected_vm_lower = conformal_quantile(vm_residuals, 0.25)
    expected_vm_upper = conformal_quantile(vm_residuals, 0.75)
    expected_rows = test_frame["Vm_pred"].to_numpy().reshape(4, 2)
    np.testing.assert_allclose(
        output_frame["Vm_pi_lower"].to_numpy().reshape(4, 2),
        expected_rows + expected_vm_lower[None, :],
        rtol=1e-6,  # float32 parquet output against a float64 expectation
    )
    np.testing.assert_allclose(
        output_frame["Vm_pi_upper"].to_numpy().reshape(4, 2),
        expected_rows + expected_vm_upper[None, :],
        rtol=1e-6,  # float32 parquet output against a float64 expectation
    )
    coverage_dir = tmp_path / "coverage"
    elements = pd.read_csv(coverage_dir / "coverage_by_element_voltage.csv")
    by_stratum = pd.read_csv(coverage_dir / "coverage_by_stratum.csv")
    groups = pd.read_csv(coverage_dir / "coverage_by_group.csv")
    assert list(elements.columns) == [
        "variable",
        "stratum",
        "bus",
        "bus_type",
        "n_values",
        "n_known",
        "coverage",
        "mean_width",
        "min_width",
        "max_width",
    ]
    assert set(elements["bus_type"]) == {"PQ", "PV"}
    assert set(groups["group"]) == {"bus_type=PQ", "bus_type=PV"}
    assert by_stratum["variable"].tolist() == ["Vm", "Va"]
    assert set(by_stratum["stratum"]) == {"all"}
    np.testing.assert_allclose(by_stratum["target_coverage"], 0.5)
    # Coverage pools the values of all buses.
    vm = elements[elements["variable"].eq("Vm")]
    pooled = (vm["coverage"] * vm["n_values"]).sum() / vm["n_values"].sum()
    vm_coverage = by_stratum.loc[by_stratum["variable"].eq("Vm"), "coverage"]
    assert vm_coverage.item() == pytest.approx(pooled, rel=1e-5)
    for name in ("coverage_by_stratum", "coverage_by_bus", "width_by_stratum"):
        assert (coverage_dir / "figures" / f"vm_{name}.png").exists()
        assert (coverage_dir / "figures" / f"va_{name}.png").exists()
    summary = (coverage_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# Coverage report")
    assert "figures/vm_coverage_by_stratum.png" in summary


def test_default_common_output_folder(tmp_path, monkeypatch) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(
        config_path, predictions_path, test_predictions_path, pf_node_path, pf_edge_path
    )
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "compute",
                "--config",
                str(config_path),
                "--predictions",
                str(predictions_path),
                "--pf-node",
                str(pf_node_path),
                "--pf-edge",
                str(pf_edge_path),
                "--alpha",
                "0.2",
                "--task-name",
                "StateEstimation",
            ],
        )
        == 0
    )

    output_dir = tmp_path / "trustlayer_outputs" / "stateestimation" / "element"
    artifact_path = output_dir / "predictions_vm_va.trustlayer"
    assert artifact_path.exists()

    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--predictions",
                str(test_predictions_path),
                "--artifact",
                str(artifact_path),
                "--task-name",
                "StateEstimation",
            ],
        )
        == 0
    )

    output_path = output_dir / "test_predictions_vm_va_with_intervals.parquet"
    assert output_path.exists()
    coverage_dir = output_dir / "coverage"
    assert (coverage_dir / "coverage_by_element_voltage.csv").exists()
    assert (coverage_dir / "coverage_by_stratum.csv").exists()
    assert (coverage_dir / "figures" / "vm_width_by_stratum.png").exists()
    output_frame = pd.read_parquet(output_path)
    assert {"Vm_pi_lower", "Va_pi_upper"}.issubset(output_frame.columns)


def test_calibrate_no_figures_skips_figure_outputs(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(
        config_path, predictions_path, test_predictions_path, pf_node_path, pf_edge_path
    )

    assert (
        main(
            [
                "compute",
                "--config",
                str(config_path),
                "--output",
                str(artifact_path),
            ],
        )
        == 0
    )
    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
                "--no-figures",
            ],
        )
        == 0
    )

    assert output_path.exists()
    coverage_dir = tmp_path / "coverage"
    for name in (
        "coverage_by_stratum.csv",
        "coverage_by_group.csv",
        "coverage_by_element_voltage.csv",
        "summary.md",
    ):
        assert (coverage_dir / name).exists()
    assert not (coverage_dir / "figures").exists()
    assert "## Figures" not in (coverage_dir / "summary.md").read_text(encoding="utf-8")


def test_config_driven_compute_and_calibrate(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "nested" / "artifacts" / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "nested" / "outputs" / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: element",
                "alpha: 0.2",
                "variables:",
                "  - Vm",
                "  - Va",
                *_inputs_yaml(
                    predictions_path, test_predictions_path, pf_node_path, pf_edge_path
                ),
                "outputs:",
                f"  artifact: {artifact_path}",
                f"  calibrated_predictions: {output_path}",
                "commands:",
                "  calibrate:",
                "    cp_type: element",
            ],
        ),
        encoding="utf-8",
    )

    assert main(["compute", "--config", str(config_path)]) == 0
    assert artifact_path.exists()
    assert main(["calibrate", "--config", str(config_path)]) == 0
    assert output_path.exists()

    output_frame = pd.read_parquet(output_path)
    assert {"Vm_pi_lower", "Va_pi_upper"}.issubset(output_frame.columns)


def test_compute_then_calibrate_with_n_outages_stratification(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_generator_status_fixture(
        tmp_path / "gen_data.parquet",
        in_service=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    )
    _write_cli_config(
        config_path,
        predictions_path,
        test_predictions_path,
        pf_node_path,
        pf_edge_path,
        alpha=0.9,
        strat="n_outages",
    )

    assert (
        main(["compute", "--config", str(config_path), "--output", str(artifact_path)])
        == 0
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["strat"] == "n_outages"
    assert artifact["stratification_values"] == [0, 1]

    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    output_frame = pd.read_parquet(output_path).sort_values(["scenario", "bus"])
    outage_one = output_frame[output_frame["scenario"].eq(3)].reset_index(drop=True)
    outage_zero = output_frame[output_frame["scenario"].eq(0)].reset_index(drop=True)
    # N-0 holds three calibration scenarios (finite bounds at alpha = 0.9); N-1
    # holds one, which cannot support a finite bound: it takes the quantile of
    # all four calibration scenarios (sparse_strata: pooled, the default).
    assert np.all(np.isfinite(outage_zero["Vm_pi_upper"].to_numpy()))
    assert not outage_zero["Vm_pooled"].any()
    assert outage_one["Vm_pooled"].all()
    calibration = pd.read_parquet(predictions_path).sort_values(["scenario", "bus"])
    residuals = (calibration["Vm_target"] - calibration["Vm_pred"]).to_numpy()
    np.testing.assert_allclose(
        outage_one["Vm_pi_upper"].to_numpy(),
        outage_one["Vm_pred"].to_numpy()
        + conformal_quantile(residuals.reshape(4, 2), 0.55),
        rtol=1e-6,
    )
    coverage_dir = output_path.parent / "coverage"
    elements = pd.read_csv(coverage_dir / "coverage_by_element_voltage.csv")
    by_stratum = pd.read_csv(coverage_dir / "coverage_by_stratum.csv")
    assert set(elements["stratum"]) == {"all", "n_outages=0", "n_outages=1"}
    assert by_stratum["stratum"].tolist() == ["all", "n_outages=0", "n_outages=1"] * 2
    # N-1 uses pooled bounds at both buses: reported, not hidden.
    n_minus_1 = by_stratum[by_stratum["stratum"].eq("n_outages=1")]
    assert (n_minus_1["n_elements_infinite"] == 0).all()
    assert (n_minus_1["n_pooled"] == 2).all()
    summary = (coverage_dir / "summary.md").read_text(encoding="utf-8")
    assert (
        "Too few calibration samples for a finite bound: Vm: 2 buses (n_outages=1: 2)"
        in summary
    )
    assert (coverage_dir / "figures" / "vm_coverage_by_stratum.png").exists()
    assert (coverage_dir / "figures" / "vm_width_by_stratum.png").exists()


def test_compute_then_calibrate_with_outage_type_stratification(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    branch_frame = pd.read_parquet(pf_edge_path)
    branch_frame["br_status"] = [1.0, 0.0, 1.0, 0.0]
    branch_frame.to_parquet(pf_edge_path, index=False)
    _write_generator_status_fixture(
        tmp_path / "gen_data.parquet",
        in_service=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    )
    _write_cli_config(
        config_path,
        predictions_path,
        test_predictions_path,
        pf_node_path,
        pf_edge_path,
        strat="outage_type",
    )

    assert (
        main(["compute", "--config", str(config_path), "--output", str(artifact_path)])
        == 0
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["strat"] == "outage_type"
    assert artifact["stratification_values"] == [0, 1, 2, 3]

    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    output_frame = pd.read_parquet(output_path).sort_values(["scenario", "bus"])
    mixed = output_frame[output_frame["scenario"].eq(3)].reset_index(drop=True)
    no_outage = output_frame[output_frame["scenario"].eq(0)].reset_index(drop=True)
    # Each outage type holds a single calibration scenario, so no finite bound
    # exists; the pipeline must still run end to end and report it as infinite.
    assert np.all(np.isposinf(mixed["Vm_pi_upper"].to_numpy()))
    assert np.all(np.isposinf(no_outage["Vm_pi_upper"].to_numpy()))
    assert {"outage_type", "outage_type_label"}.issubset(output_frame.columns)
    assert set(output_frame["outage_type_label"].unique().tolist()) == {
        "none",
        "branch_only",
        "gen_only",
        "mixed",
    }
    assert set(
        output_frame.loc[output_frame["outage_type"].eq(3), "outage_type_label"]
        .unique()
        .tolist()
    ) == {"mixed"}

    coverage_dir = output_path.parent / "coverage"
    elements = pd.read_csv(coverage_dir / "coverage_by_element_voltage.csv")
    assert set(elements["stratum"]) == {
        "all",
        "outage_type=none",
        "outage_type=branch_only",
        "outage_type=gen_only",
        "outage_type=mixed",
    }
    assert (coverage_dir / "figures" / "vm_width_by_stratum.png").exists()


def test_compute_then_calibrate_with_system_load_bin_stratification(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(
        config_path,
        predictions_path,
        test_predictions_path,
        pf_node_path,
        pf_edge_path,
        alpha=0.9,
        strat="system_load_bin",
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\nsystem_load_n_bins: 2\n",
        encoding="utf-8",
    )

    assert (
        main(["compute", "--config", str(config_path), "--output", str(artifact_path)])
        == 0
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["strat"] == "system_load_bin"
    assert artifact["system_load_n_bins"] == 2
    assert artifact["system_load_bin_edges"] == [50.0]
    assert artifact["stratification_values"] == [0, 1]

    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    output_frame = pd.read_parquet(output_path).sort_values(["scenario", "bus"])
    assert {"system_load_bin", "system_load_bin_label"}.issubset(output_frame.columns)
    assert set(output_frame["system_load_bin_label"].unique().tolist()) == {
        "p00-p50 (<50)",
        "p50-p100 (>=50)",
    }
    low_load = output_frame[output_frame["scenario"].eq(0)].reset_index(drop=True)
    high_load = output_frame[output_frame["scenario"].eq(3)].reset_index(drop=True)
    assert not np.allclose(
        low_load["Vm_pi_upper"].to_numpy(),
        high_load["Vm_pi_upper"].to_numpy(),
    )
    elements = pd.read_csv(
        output_path.parent / "coverage" / "coverage_by_element_voltage.csv"
    )
    assert set(elements["stratum"]) == {
        "all",
        "system_load_bin=p00-p50 (<50)",
        "system_load_bin=p50-p100 (>=50)",
    }


def test_compute_then_calibrate_with_pred_bins_stratification(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(
        config_path,
        predictions_path,
        test_predictions_path,
        pf_node_path,
        pf_edge_path,
        strat="pred_bins",
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\npred_n_bins: 2\n",
        encoding="utf-8",
    )

    assert (
        main(["compute", "--config", str(config_path), "--output", str(artifact_path)])
        == 0
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["strat"] == "pred_bins"
    assert artifact["pred_n_bins"] == 2
    assert set(artifact["pred_bin_edges"]) == {"Vm", "Va"}

    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    output_frame = pd.read_parquet(output_path).sort_values(["scenario", "bus"])
    assert {
        "pred_bins_vm",
        "pred_bins_vm_label",
        "pred_bins_va",
        "pred_bins_va_label",
    }.issubset(output_frame.columns)
    assert set(output_frame["pred_bins_vm_label"].unique().tolist()) == {
        "p00-p50 (<1)",
        "p50-p100 (>=1)",
    }
    same_scenario = output_frame[output_frame["scenario"].eq(0)].sort_values("bus")
    np.testing.assert_array_equal(
        same_scenario["pred_bins_vm"].to_numpy(dtype=float),
        np.array([1.0, 0.0]),
    )
    elements = pd.read_csv(
        output_path.parent / "coverage" / "coverage_by_element_voltage.csv"
    )
    vm_strata = set(elements.loc[elements["variable"].eq("Vm"), "stratum"])
    assert "all" in vm_strata
    assert all(s == "all" or s.startswith("pred_bins_vm=") for s in vm_strata)


def test_compute_then_calibrate_line_loading_roundtrip(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "line_intervals.trustlayer"
    output_path = tmp_path / "line_predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(
        config_path, predictions_path, test_predictions_path, pf_node_path, pf_edge_path
    )

    assert (
        main(
            [
                "compute",
                "--config",
                str(config_path),
                "--predictions",
                str(predictions_path),
                "--pf-node",
                str(pf_node_path),
                "--pf-edge",
                str(pf_edge_path),
                "--alpha",
                "0.5",
                "--variables",
                "line_loading",
                "--output",
                str(artifact_path),
                "--task-name",
                "StateEstimation",
            ],
        )
        == 0
    )

    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--predictions",
                str(test_predictions_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
                "--pf-node",
                str(pf_node_path),
                "--pf-edge",
                str(pf_edge_path),
                "--task-name",
                "StateEstimation",
            ],
        )
        == 0
    )

    output_frame = pd.read_parquet(output_path).sort_values(["scenario", "branch"])
    assert {
        "scenario",
        "branch",
        "from_bus",
        "to_bus",
        "rate_a",
        "line_loading_pred",
        "line_loading_target",
        "line_loading_pi_lower",
        "line_loading_pi_upper",
    }.issubset(output_frame.columns)

    predictions = load_predictions(str(predictions_path), task_name="StateEstimation")
    ground_truth = load_ground_truth(str(pf_node_path), str(pf_edge_path), sn_mva=100.0)
    predicted_loading = compute_line_loading(
        predictions.Vm, predictions.Va, ground_truth
    )
    target_loading = compute_line_loading(
        ground_truth.Vm, ground_truth.Va, ground_truth
    )
    residuals = target_loading - predicted_loading
    expected_lower = conformal_quantile(residuals[:, :1], 0.25)[0]
    expected_upper = conformal_quantile(residuals[:, :1], 0.75)[0]
    np.testing.assert_allclose(
        output_frame["line_loading_pi_lower"].to_numpy(),
        output_frame["line_loading_pred"].to_numpy() + expected_lower,
    )
    np.testing.assert_allclose(
        output_frame["line_loading_pi_upper"].to_numpy(),
        output_frame["line_loading_pred"].to_numpy() + expected_upper,
    )

    coverage_dir = tmp_path / "coverage"
    elements = pd.read_csv(coverage_dir / "coverage_by_element_line_loading.csv")
    groups = pd.read_csv(coverage_dir / "coverage_by_group.csv")
    by_stratum = pd.read_csv(coverage_dir / "coverage_by_stratum.csv")
    assert list(elements.columns) == [
        "variable",
        "stratum",
        "branch",
        "from_bus",
        "to_bus",
        "rate_a",
        "n_values",
        "coverage",
        "mean_width",
        "min_width",
        "max_width",
    ]
    assert groups["group"].tolist() == ["rate_a=10"]
    np.testing.assert_allclose(by_stratum["target_coverage"], 0.5)
    assert (coverage_dir / "figures" / "line_loading_width_by_stratum.png").exists()
    assert (coverage_dir / "figures" / "line_loading_coverage_by_line.png").exists()


def test_compute_then_calibrate_line_loading_with_pred_bins_stratification(
    tmp_path,
) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "line_intervals.trustlayer"
    output_path = tmp_path / "line_predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config_line.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: element",
                "alpha: 0.2",
                "strat: pred_bins",
                "pred_n_bins: 2",
                "variables:",
                "  - line_loading",
                *_inputs_yaml(
                    predictions_path, test_predictions_path, pf_node_path, pf_edge_path
                ),
                "outputs:",
                f"  artifact: {artifact_path}",
                f"  calibrated_predictions: {output_path}",
            ],
        ),
        encoding="utf-8",
    )

    assert main(["compute", "--config", str(config_path)]) == 0
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["strat"] == "pred_bins"
    assert set(artifact["pred_bin_edges"]) == {"line_loading"}

    assert main(["calibrate", "--config", str(config_path)]) == 0
    output_frame = pd.read_parquet(output_path).sort_values(["scenario", "branch"])
    assert {"pred_bins_line_loading", "pred_bins_line_loading_label"}.issubset(
        output_frame.columns
    )


def test_config_driven_line_loading_compute_and_calibrate(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "nested" / "artifacts" / "line_intervals.trustlayer"
    output_path = (
        tmp_path / "nested" / "outputs" / "line_predictions_with_intervals.parquet"
    )
    config_path = tmp_path / "trust_config_line.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: element",
                "alpha: 0.2",
                "variables:",
                "  - line_loading",
                *_inputs_yaml(
                    predictions_path, test_predictions_path, pf_node_path, pf_edge_path
                ),
                "outputs:",
                f"  artifact: {artifact_path}",
                f"  calibrated_predictions: {output_path}",
            ],
        ),
        encoding="utf-8",
    )

    assert main(["compute", "--config", str(config_path)]) == 0
    assert artifact_path.exists()
    assert main(["calibrate", "--config", str(config_path)]) == 0
    assert output_path.exists()

    output_frame = pd.read_parquet(output_path)
    assert {
        "branch",
        "line_loading_pred",
        "line_loading_target",
        "line_loading_pi_lower",
        "line_loading_pi_upper",
    }.issubset(output_frame.columns)
    coverage_dir = output_path.parent / "coverage"
    assert (coverage_dir / "coverage_by_element_line_loading.csv").exists()
    assert (coverage_dir / "figures" / "line_loading_width_by_stratum.png").exists()


def test_config_driven_mixed_compute_and_calibrate(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "nested" / "artifacts" / "mixed_intervals.trustlayer"
    outputs_dir = tmp_path / "nested" / "outputs"
    volt_output_path = outputs_dir / "predictions_vm_va_with_intervals.parquet"
    line_output_path = outputs_dir / "predictions_line_loading_with_intervals.parquet"
    config_path = tmp_path / "trust_config_mixed.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: element",
                "alpha: 0.2",
                "variables:",
                "  - Vm",
                "  - Va",
                "  - line_loading",
                *_inputs_yaml(
                    predictions_path, test_predictions_path, pf_node_path, pf_edge_path
                ),
                "outputs:",
                f"  artifact: {artifact_path}",
                f"  calibrated_predictions_volt: {volt_output_path}",
                f"  calibrated_predictions_line: {line_output_path}",
            ],
        ),
        encoding="utf-8",
    )

    assert main(["compute", "--config", str(config_path)]) == 0
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["version"] == "trustlayer_interval_bundle_v1"
    assert set(artifact["artifacts"]) == {"volt", "line"}

    assert main(["calibrate", "--config", str(config_path)]) == 0
    assert volt_output_path.exists()
    assert line_output_path.exists()

    volt_frame = pd.read_parquet(volt_output_path)
    line_frame = pd.read_parquet(line_output_path)
    assert {"Vm_pi_lower", "Va_pi_upper"}.issubset(volt_frame.columns)
    assert {"line_loading_pred", "line_loading_pi_upper"}.issubset(line_frame.columns)
    coverage_dir = outputs_dir / "coverage"
    by_stratum = pd.read_csv(coverage_dir / "coverage_by_stratum.csv")
    assert by_stratum["variable"].tolist() == ["Vm", "Va", "line_loading"]
    assert (coverage_dir / "coverage_by_element_voltage.csv").exists()
    assert (coverage_dir / "coverage_by_element_line_loading.csv").exists()
    assert (coverage_dir / "figures" / "vm_width_by_stratum.png").exists()
    assert (coverage_dir / "figures" / "line_loading_width_by_stratum.png").exists()


def test_config_driven_mixed_screen(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "nested" / "artifacts" / "mixed_intervals.trustlayer"
    outputs_dir = tmp_path / "nested" / "outputs"
    volt_output_path = outputs_dir / "predictions_vm_va_with_intervals.parquet"
    line_output_path = outputs_dir / "predictions_line_loading_with_intervals.parquet"
    screen_dir = tmp_path / "nested" / "screen"
    config_path = tmp_path / "trust_config_screen.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: element",
                "alpha: 0.2",
                "variables:",
                "  - Vm",
                "  - Va",
                "  - line_loading",
                "screen_bounds:",
                "  Vm:",
                "    lower: 0.95",
                "    upper: 1.05",
                "  line_loading:",
                "    upper: 1.0",
                *_inputs_yaml(
                    predictions_path, test_predictions_path, pf_node_path, pf_edge_path
                ),
                "outputs:",
                f"  artifact: {artifact_path}",
                f"  calibrated_predictions_volt: {volt_output_path}",
                f"  calibrated_predictions_line: {line_output_path}",
                f"  screen_dir: {screen_dir}",
                "commands:",
                "  screen:",
                "    variables:",
                "      - Vm",
                "      - line_loading",
            ],
        ),
        encoding="utf-8",
    )

    assert main(["compute", "--config", str(config_path)]) == 0
    assert main(["calibrate", "--config", str(config_path)]) == 0
    assert main(["screen", "--config", str(config_path)]) == 0

    volt_detail = pd.read_parquet(screen_dir / "screening_detail_voltage.parquet")
    line_detail = pd.read_parquet(screen_dir / "screening_detail_line_loading.parquet")
    by_stratum = pd.read_csv(screen_dir / "screening_by_stratum.csv")
    by_group = pd.read_csv(screen_dir / "screening_by_group.csv")
    volt_elements = pd.read_csv(screen_dir / "screening_by_element_voltage.csv")

    assert set(volt_detail["variable"].unique().tolist()) == {"Vm"}
    assert set(line_detail["variable"].unique().tolist()) == {"line_loading"}
    flags = [
        "pred_lower_violation",
        "true_lower_violation",
        "pred_upper_violation",
        "true_upper_violation",
    ]
    values = ["variable", "target", "pi_lower", "pi_upper", *flags]
    partition = ["scenario_partition"]
    assert list(volt_detail.columns) == [
        "scenario",
        "bus",
        "bus_type",
        *values,
        *partition,
    ]
    assert list(line_detail.columns) == [
        "scenario",
        "branch",
        "from_bus",
        "to_bus",
        "rate_a",
        *values,
        *partition,
    ]
    # One row per variable, stratum and screened side; line loading only upper.
    assert by_stratum[["variable", "side"]].values.tolist() == [
        ["Vm", "lower"],
        ["Vm", "upper"],
        ["Vm", "any"],
        ["line_loading", "upper"],
    ]
    assert list(by_stratum.columns) == [
        "variable",
        "stratum",
        "side",
        "n_values",
        "n_known",
        "violations",
        "alarms",
        "caught",
        "missed",
        "false_alarms",
        "cleared",
        "recall",
        "precision",
    ]
    # The stored flags reproduce the counts.
    predicted = (
        volt_detail["pred_lower_violation"] | volt_detail["pred_upper_violation"]
    )
    true = volt_detail["true_lower_violation"] | volt_detail["true_upper_violation"]
    vm_any = by_stratum[
        by_stratum["variable"].eq("Vm") & by_stratum["side"].eq("any")
    ].iloc[0]
    assert int(vm_any["caught"]) == int((predicted & true).sum())
    assert int(vm_any["false_alarms"]) == int((predicted & ~true).sum())
    assert int(vm_any["missed"]) == int((~predicted & true).sum())
    assert int(vm_any["n_values"]) == len(volt_detail)
    assert int(
        vm_any["caught"] + vm_any["missed"] + vm_any["false_alarms"] + vm_any["cleared"]
    ) == len(volt_detail)
    assert set(by_group["group"]) == {"all", "bus_type=PQ", "bus_type=PV", "rate_a=10"}
    assert {"bus", "bus_type", "side", "recall"}.issubset(volt_elements.columns)
    # line_loading has only an upper limit.
    assert not line_detail["pred_lower_violation"].any()
    assert not line_detail["true_lower_violation"].any()

    assert (screen_dir / "figures" / "vm_confusion.png").exists()
    assert (screen_dir / "figures" / "line_loading_confusion.png").exists()
    summary = (screen_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# Screening report")
    assert "### Vm (either limit)" in summary


def _screen_config(tmp_path, outputs: list[str]) -> Path:
    predictions_path = tmp_path / "predictions.parquet"
    test_predictions_path = tmp_path / "test_predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    _write_prediction_fixture(predictions_path)
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path = tmp_path / "trust_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: element",
                "alpha: 0.2",
                "variables: [Vm, Va, line_loading]",
                "screen_bounds:",
                "  Vm: {lower: 0.95, upper: 1.05}",
                "  line_loading: {upper: 1.0}",
                *_inputs_yaml(
                    predictions_path, test_predictions_path, pf_node_path, pf_edge_path
                ),
                "outputs:",
                f"  artifact: {tmp_path / 'intervals.trustlayer'}",
                *outputs,
                "commands:",
                "  screen:",
                "    variables: [Vm, line_loading]",
            ],
        ),
        encoding="utf-8",
    )
    return config_path


def test_screen_finds_mixed_outputs_in_a_shared_directory(tmp_path) -> None:
    # calibrate writes test_predictions_vm_va_... into the directory, while only
    # Vm is screened: screen must find the file by family, not by its own variables.
    calibrated_dir = tmp_path / "calibrated"
    screen_dir = tmp_path / "screen"
    config_path = _screen_config(
        tmp_path,
        [
            f"  calibrated_predictions: {calibrated_dir}",
            f"  screen_dir: {screen_dir}",
        ],
    )

    assert main(["compute", "--config", str(config_path)]) == 0
    assert main(["calibrate", "--config", str(config_path)]) == 0
    assert main(["screen", "--config", str(config_path)]) == 0

    assert (screen_dir / "screening_by_stratum.csv").exists()
    assert (screen_dir / "screening_by_element_line_loading.csv").exists()


def test_screen_finds_mixed_outputs_in_the_default_directory(
    tmp_path, monkeypatch
) -> None:
    config_path = _screen_config(tmp_path, [])
    monkeypatch.chdir(tmp_path)

    assert main(["compute", "--config", str(config_path)]) == 0
    assert main(["calibrate", "--config", str(config_path)]) == 0
    assert main(["screen", "--config", str(config_path)]) == 0

    screen_dir = tmp_path / "trustlayer_outputs" / "stateestimation" / "element"
    assert (screen_dir / "screening_by_stratum.csv").exists()
    assert (screen_dir / "screening_by_element_line_loading.csv").exists()


def test_screen_no_figures_skips_the_figures(tmp_path) -> None:
    screen_dir = tmp_path / "screen"
    config_path = _screen_config(
        tmp_path,
        [
            f"  calibrated_predictions: {tmp_path / 'calibrated'}",
            f"  screen_dir: {screen_dir}",
        ],
    )
    for command in ("compute", "calibrate"):
        assert main([command, "--config", str(config_path)]) == 0
    assert main(["screen", "--config", str(config_path), "--no-figures"]) == 0

    assert (screen_dir / "screening_by_stratum.csv").exists()
    assert not (screen_dir / "figures").exists()
    summary = (screen_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Figures" not in summary


def test_single_family_run_honours_the_family_specific_output_key(tmp_path) -> None:
    # calibrate and screen must agree on the path when only *_volt is set.
    volt_path = tmp_path / "out" / "volt.parquet"
    config_path = _screen_config(
        tmp_path,
        [
            f"  calibrated_predictions_volt: {volt_path}",
            f"  screen_dir: {tmp_path / 'screen'}",
        ],
    )
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("[Vm, Va, line_loading]", "[Vm, Va]").replace(
            "[Vm, line_loading]", "[Vm]"
        ),
        encoding="utf-8",
    )

    assert main(["compute", "--config", str(config_path)]) == 0
    assert main(["calibrate", "--config", str(config_path)]) == 0
    assert volt_path.exists()
    assert main(["screen", "--config", str(config_path)]) == 0
    assert (tmp_path / "screen" / "screening_by_element_voltage.csv").exists()


def test_screen_explains_a_missing_calibrated_parquet(tmp_path) -> None:
    config_path = _screen_config(
        tmp_path, [f"  calibrated_predictions: {tmp_path / 'empty'}"]
    )
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="Run calibrate first"):
        main(["screen", "--config", str(config_path)])


def test_normalize_variables_accepts_mixed_line_and_voltage_selection() -> None:
    assert normalize_variables("vm,va") == ("Vm", "Va")
    assert normalize_variables(["line-loading"]) == ("line_loading",)
    assert normalize_variables(["Vm", "line_loading", "va"]) == (
        "Vm",
        "line_loading",
        "Va",
    )


def test_cli_flags_override_config(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    config_artifact_path = tmp_path / "config_intervals.trustlayer"
    override_artifact_path = tmp_path / "override_intervals.trustlayer"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    test_predictions_path = tmp_path / "test_predictions.parquet"
    _write_prediction_fixture(test_predictions_path, shift=0.01)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "cp_type: element",
                "alpha: 0.2",
                "variables:",
                "  - Vm",
                *_inputs_yaml(
                    predictions_path, test_predictions_path, pf_node_path, pf_edge_path
                ),
                "outputs:",
                f"  artifact: {config_artifact_path}",
            ],
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "compute",
                "--config",
                str(config_path),
                "--output",
                str(override_artifact_path),
                "--variables",
                "Vm",
                "Va",
            ],
        )
        == 0
    )

    assert override_artifact_path.exists()
    assert not config_artifact_path.exists()
    artifact = json.loads(override_artifact_path.read_text(encoding="utf-8"))
    assert artifact["channel_names"] == ["Vm", "Va"]


def test_calibrate_refuses_the_calibration_data_under_another_name(tmp_path) -> None:
    # A copy of the calibration predictions passes any path check, but applying the
    # layer to it would still measure coverage in-sample. The artifact's fingerprint
    # of the calibration predictions catches it.
    predictions_path = tmp_path / "predictions.parquet"
    copy_path = tmp_path / "renamed_copy.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "intervals.trustlayer"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    copy_path.write_bytes(predictions_path.read_bytes())
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(
        config_path, predictions_path, copy_path, pf_node_path, pf_edge_path
    )

    assert (
        main(["compute", "--config", str(config_path), "--output", str(artifact_path)])
        == 0
    )
    with pytest.raises(ValueError, match="identical to the calibration predictions"):
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--artifact",
                str(artifact_path),
            ]
        )


def test_config_rejects_the_same_file_for_calibration_and_test(tmp_path) -> None:
    config_path = tmp_path / "trust_config.yaml"
    predictions = tmp_path / "predictions.parquet"
    _write_cli_config(
        config_path,
        predictions,
        tmp_path / "." / "predictions.parquet",
        tmp_path / "pf_node.parquet",
        tmp_path / "pf_edge.parquet",
    )
    with pytest.raises(ValueError, match="point to the same file"):
        load_cli_config(str(config_path))


def test_config_rejects_a_single_shared_inputs_block(tmp_path) -> None:
    config_path = tmp_path / "trust_config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """\
            version: 1
            task_name: PowerFlow
            cp_type: element
            alpha: 0.1
            variables:
              - Vm
            inputs:
              predictions: predictions.parquet
              pf_node: bus_data.parquet
              pf_edge: branch_data.parquet"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be separate datasets"):
        load_cli_config(str(config_path))


def test_cli_requires_config_flag() -> None:
    with pytest.raises(SystemExit):
        main(["compute"])
    with pytest.raises(SystemExit):
        main(["calibrate"])


def test_load_cli_config_rejects_unknown_top_level_keys(tmp_path) -> None:
    config_path = tmp_path / "trust_config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """\
            version: 1
            task_name: StateEstimation
            cp_type: element
            alpha: 0.2
            variables:
              - Vm
            mystery_key: value"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="Unsupported top-level trustlayer config keys"
    ):
        load_cli_config(str(config_path))


def _write_cli_config(
    config_path,
    predictions_path,
    test_predictions_path,
    pf_node_path,
    pf_edge_path,
    *,
    alpha: float = 0.2,
    task_name: str = "StateEstimation",
    strat: str | None = None,
) -> None:
    lines = [
        "version: 1",
        f"task_name: {task_name}",
        "cp_type: element",
        f"alpha: {alpha}",
    ]
    if strat is not None:
        lines.append(f"strat: {strat}")
    lines.extend(
        [
            "variables:",
            "  - Vm",
            "  - Va",
            *_inputs_yaml(
                predictions_path, test_predictions_path, pf_node_path, pf_edge_path
            ),
        ]
    )
    config_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _inputs_yaml(
    calibration_predictions, test_predictions, pf_node_path, pf_edge_path
) -> list[str]:
    """YAML lines for separate calibration and test datasets sharing the topology."""

    return [
        "inputs:",
        "  calibration:",
        f"    predictions: {calibration_predictions}",
        f"    pf_node: {pf_node_path}",
        f"    pf_edge: {pf_edge_path}",
        "  test:",
        f"    predictions: {test_predictions}",
        f"    pf_node: {pf_node_path}",
        f"    pf_edge: {pf_edge_path}",
    ]


def _write_prediction_fixture(path, *, shift: float = 0.0) -> None:
    """Predictions for four scenarios; `shift` offsets every predicted value so the
    test set differs from the calibration set while sharing the ground truth."""
    frame = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1, 2, 2, 3, 3],
            "bus": [0, 1, 0, 1, 0, 1, 0, 1],
            "Vm_target": [1.05, 0.85, 1.12, 0.97, 1.0, 1.08, 1.0, 1.0],
            "Va_target": [0.02, 0.12, 0.16, -0.05, -0.1, 0.25, 0.08, -0.02],
            "Vm_pred": [
                v + shift for v in [1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 1.05, 0.98]
            ],
            "Va_pred": [v + shift for v in [0.0, 0.1, 0.2, -0.1, -0.2, 0.3, 0.1, 0.0]],
            "P_inj_target": [2.0, -2.0, 2.1, -2.1, 1.9, -1.9, 2.0, -2.0],
            "P_inj_pred": [
                v + shift for v in [1.8, -1.8, 2.0, -2.0, 1.8, -1.8, 1.9, -1.9]
            ],
            "PQ": [0, 1, 0, 1, 0, 1, 0, 1],
            "PV": [1, 0, 1, 0, 1, 0, 1, 0],
            "REF": [0] * 8,
        },
    )
    frame.to_parquet(path, index=False)


def _write_ground_truth_fixture(bus_path, branch_path) -> None:
    bus_frame = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1, 2, 2, 3, 3],
            "bus": [0, 1, 0, 1, 0, 1, 0, 1],
            "Pd": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0],
            "Qd": [0.0] * 8,
            "Vm": [1.05, 0.85, 1.12, 0.97, 1.0, 1.08, 1.0, 1.0],
            "Va": [0.02, 0.12, 0.16, -0.05, -0.1, 0.25, 0.08, -0.02],
            "Pg": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "Qg": [0.0] * 8,
            "PQ": [0, 1, 0, 1, 0, 1, 0, 1],
            "PV": [1, 0, 1, 0, 1, 0, 1, 0],
            "REF": [0] * 8,
            "vn_kv": [100.0] * 8,
        },
    )
    branch_frame = pd.DataFrame(
        {
            "scenario": [0, 1, 2, 3],
            "idx": [0, 0, 0, 0],
            "from_bus": [0, 0, 0, 0],
            "to_bus": [1, 1, 1, 1],
            "rate_a": [10.0, 10.0, 10.0, 10.0],
            "br_status": [1.0, 1.0, 1.0, 1.0],
            "Yff_r": [1.0, 1.0, 1.0, 1.0],
            "Yff_i": [0.0, 0.0, 0.0, 0.0],
            "Yft_r": [-1.0, -1.0, -1.0, -1.0],
            "Yft_i": [0.0, 0.0, 0.0, 0.0],
            "Ytf_r": [-1.0, -1.0, -1.0, -1.0],
            "Ytf_i": [0.0, 0.0, 0.0, 0.0],
            "Ytt_r": [1.0, 1.0, 1.0, 1.0],
            "Ytt_i": [0.0, 0.0, 0.0, 0.0],
        },
    )
    bus_frame.to_parquet(bus_path, index=False)
    branch_frame.to_parquet(branch_path, index=False)


def _write_generator_status_fixture(path, *, in_service: list[float]) -> None:
    gen_frame = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1, 2, 2, 3, 3],
            "idx": [0, 1, 0, 1, 0, 1, 0, 1],
            "bus": [0, 1, 0, 1, 0, 1, 0, 1],
            "in_service": in_service,
        }
    )
    gen_frame.to_parquet(path, index=False)


def test_version_flag_and_module_entry_point(capsys) -> None:
    import subprocess
    import sys

    from gridfm_trustlayer import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"gridfm_trustlayer {__version__}"

    result = subprocess.run(
        [sys.executable, "-m", "gridfm_trustlayer", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"gridfm_trustlayer {__version__}"


def test_commands_report_a_summary(tmp_path, capsys) -> None:
    config_path = _screen_config(
        tmp_path,
        [
            f"  calibrated_predictions: {tmp_path / 'calibrated'}",
            f"  screen_dir: {tmp_path / 'screen'}",
        ],
    )

    for command in ("compute", "calibrate", "screen"):
        assert main([command, "--config", str(config_path)]) == 0
    printed = capsys.readouterr().out

    assert (
        "compute: fitted element for Vm, Va, line_loading on 4 calibration" in printed
    )
    assert "calibrate: Vm: coverage" in printed
    assert "calibrate: line_loading: coverage" in printed
    assert f"reports in {tmp_path / 'calibrated' / 'coverage'}" in printed
    assert "screen: Vm (either limit):" in printed
    assert "true violations flagged" in printed
    assert f"screen: reports in {tmp_path / 'screen'}" in printed
