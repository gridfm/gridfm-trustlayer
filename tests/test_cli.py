"""CLI tests for naive Vm/Va interval computation and application."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer.cli import main
from gridfm_trustlayer.cli.config import normalize_variables
from gridfm_trustlayer.io import compute_line_loading, load_ground_truth, load_predictions


def test_compute_then_calibrate_roundtrip(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(config_path, predictions_path, pf_node_path, pf_edge_path)

    assert main(
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
            "--output",
            str(artifact_path),
            "--task-name",
            "StateEstimation",
        ],
    ) == 0

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["method"] == "naive_residual_quantiles"
    assert artifact["channel_names"] == ["Vm", "Va"]

    assert main(
        [
            "calibrate",
            "--config",
            str(config_path),
            "--predictions",
            str(predictions_path),
            "--artifact",
            str(artifact_path),
            "--output",
            str(output_path),
            "--task-name",
            "StateEstimation",
        ],
    ) == 0

    output_frame = pd.read_parquet(output_path).sort_values(["scenario", "bus"])
    assert {
        "Vm_pi_lower",
        "Vm_pi_upper",
        "Va_pi_lower",
        "Va_pi_upper",
    }.issubset(output_frame.columns)

    original_frame = pd.read_parquet(predictions_path).sort_values(["scenario", "bus"])
    vm_residuals = (
        pd.read_parquet(pf_node_path).sort_values(["scenario", "bus"])["Vm"].to_numpy()
        - original_frame["Vm_pred"].to_numpy()
    ).reshape(4, 2)
    expected_vm_lower = np.quantile(vm_residuals, 0.1, axis=0).astype(np.float32)
    expected_vm_upper = np.quantile(vm_residuals, 0.9, axis=0).astype(np.float32)
    expected_rows = original_frame["Vm_pred"].to_numpy().reshape(4, 2)
    np.testing.assert_allclose(
        output_frame["Vm_pi_lower"].to_numpy().reshape(4, 2),
        expected_rows + expected_vm_lower[None, :],
    )
    np.testing.assert_allclose(
        output_frame["Vm_pi_upper"].to_numpy().reshape(4, 2),
        expected_rows + expected_vm_upper[None, :],
    )
    analysis_path = tmp_path / "trust_analysis_volt.csv"
    summary_path = tmp_path / "trust_summary_volt.csv"
    figures_dir = tmp_path / "figures"
    assert analysis_path.exists()
    assert summary_path.exists()
    analysis_frame = pd.read_csv(analysis_path)
    summary_frame = pd.read_csv(summary_path)
    assert {
        "variable",
        "bus",
        "bus_type",
        "width_mean",
        "width_min",
        "width_max",
        "width_mean_degrees",
        "width_min_degrees",
        "width_max_degrees",
        "empirical_coverage",
        "n_samples",
    }.issubset(analysis_frame.columns)
    va_rows = analysis_frame[analysis_frame["variable"].eq("Va")].copy()
    np.testing.assert_allclose(
        va_rows["width_mean_degrees"].to_numpy(),
        np.rad2deg(va_rows["width_mean"].to_numpy()),
    )
    assert set(analysis_frame["bus_type"].unique().tolist()) == {"PQ", "PV"}
    assert {
        "scope",
        "variable",
        "bus_type",
        "n_buses",
        "n_samples_min",
        "n_samples_max",
        "width_mean_across_buses",
        "width_min_across_buses",
        "width_max_across_buses",
        "empirical_coverage_mean_across_buses",
        "empirical_coverage_min_across_buses",
        "empirical_coverage_max_across_buses",
        "width_mean_degrees_across_buses",
        "width_min_degrees_across_buses",
        "width_max_degrees_across_buses",
    }.issubset(summary_frame.columns)
    assert set(summary_frame["scope"].unique().tolist()) == {"system", "bus_type"}
    assert len(summary_frame) == 6
    assert figures_dir.joinpath("vm_width_summary.png").exists()
    assert figures_dir.joinpath("vm_coverage_summary.png").exists()
    assert figures_dir.joinpath("vm_width_density.png").exists()
    assert figures_dir.joinpath("vm_coverage_density.png").exists()
    assert figures_dir.joinpath("va_width_summary.png").exists()
    assert figures_dir.joinpath("va_coverage_summary.png").exists()
    assert figures_dir.joinpath("va_width_density.png").exists()
    assert figures_dir.joinpath("va_coverage_density.png").exists()


def test_default_common_output_folder(tmp_path, monkeypatch) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(config_path, predictions_path, pf_node_path, pf_edge_path)
    monkeypatch.chdir(tmp_path)

    assert main(
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
    ) == 0

    output_dir = tmp_path / "trustlayer_outputs" / "stateestimation" / "base"
    artifact_path = output_dir / "predictions_vm_va.trustlayer"
    assert artifact_path.exists()

    assert main(
        [
            "calibrate",
            "--config",
            str(config_path),
            "--predictions",
            str(predictions_path),
            "--artifact",
            str(artifact_path),
            "--task-name",
            "StateEstimation",
        ],
    ) == 0

    output_path = output_dir / "predictions_vm_va_with_intervals.parquet"
    assert output_path.exists()
    assert (output_dir / "trust_analysis_volt.csv").exists()
    assert (output_dir / "trust_summary_volt.csv").exists()
    assert (output_dir / "figures" / "vm_width_summary.png").exists()
    assert (output_dir / "figures" / "vm_width_density.png").exists()
    output_frame = pd.read_parquet(output_path)
    assert {"Vm_pi_lower", "Va_pi_upper"}.issubset(output_frame.columns)


def test_config_driven_compute_and_calibrate(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "nested" / "artifacts" / "vm_va_intervals.trustlayer"
    output_path = tmp_path / "nested" / "outputs" / "predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: base",
                "alpha: 0.2",
                "variables:",
                "  - Vm",
                "  - Va",
                "inputs:",
                f"  predictions: {predictions_path}",
                f"  pf_node: {pf_node_path}",
                f"  pf_edge: {pf_edge_path}",
                "outputs:",
                f"  artifact: {artifact_path}",
                f"  calibrated_predictions: {output_path}",
                "commands:",
                "  calibrate:",
                "    cp_type: base",
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
    assert (output_path.parent / "trust_analysis_volt.csv").exists()
    assert (output_path.parent / "trust_summary_volt.csv").exists()
    assert (output_path.parent / "figures" / "va_coverage_summary.png").exists()
    assert (output_path.parent / "figures" / "va_coverage_density.png").exists()


def test_compute_then_calibrate_line_loading_roundtrip(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "line_intervals.trustlayer"
    output_path = tmp_path / "line_predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(config_path, predictions_path, pf_node_path, pf_edge_path)

    assert main(
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
            "--variables",
            "line_loading",
            "--output",
            str(artifact_path),
            "--task-name",
            "StateEstimation",
        ],
    ) == 0

    assert main(
        [
            "calibrate",
            "--config",
            str(config_path),
            "--predictions",
            str(predictions_path),
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
    ) == 0

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
    predicted_loading = compute_line_loading(predictions.Vm, predictions.Va, ground_truth)
    target_loading = compute_line_loading(ground_truth.Vm, ground_truth.Va, ground_truth)
    residuals = target_loading - predicted_loading
    expected_lower = np.quantile(residuals[:, 0], 0.1).astype(np.float32)
    expected_upper = np.quantile(residuals[:, 0], 0.9).astype(np.float32)
    np.testing.assert_allclose(
        output_frame["line_loading_pi_lower"].to_numpy(),
        output_frame["line_loading_pred"].to_numpy() + expected_lower,
    )
    np.testing.assert_allclose(
        output_frame["line_loading_pi_upper"].to_numpy(),
        output_frame["line_loading_pred"].to_numpy() + expected_upper,
    )

    analysis_frame = pd.read_csv(tmp_path / "trust_analysis_line.csv")
    summary_frame = pd.read_csv(tmp_path / "trust_summary_line.csv")
    assert {"variable", "branch", "from_bus", "to_bus", "rate_a", "width_mean", "empirical_coverage", "n_samples"}.issubset(analysis_frame.columns)
    assert {"scope", "variable", "rate_a", "n_lines", "width_mean_across_lines", "empirical_coverage_mean_across_lines"}.issubset(summary_frame.columns)
    assert set(summary_frame["scope"].unique().tolist()) == {"system", "rate_a"}
    rate_rows = summary_frame[summary_frame["scope"].eq("rate_a")].copy()
    assert len(rate_rows) == 1
    np.testing.assert_allclose(rate_rows["rate_a"].to_numpy(), np.array([10.0]))
    assert (tmp_path / "figures" / "line_loading_width_summary.png").exists()
    assert (tmp_path / "figures" / "line_loading_coverage_density.png").exists()


def test_config_driven_line_loading_compute_and_calibrate(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "nested" / "artifacts" / "line_intervals.trustlayer"
    output_path = tmp_path / "nested" / "outputs" / "line_predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config_line.yaml"

    _write_prediction_fixture(predictions_path)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: naive",
                "alpha: 0.2",
                "variables:",
                "  - line_loading",
                "inputs:",
                f"  predictions: {predictions_path}",
                f"  pf_node: {pf_node_path}",
                f"  pf_edge: {pf_edge_path}",
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
    assert (output_path.parent / "trust_analysis_line.csv").exists()
    assert (output_path.parent / "trust_summary_line.csv").exists()
    assert (output_path.parent / "figures" / "line_loading_width_summary.png").exists()


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
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: naive",
                "alpha: 0.2",
                "variables:",
                "  - Vm",
                "  - Va",
                "  - line_loading",
                "inputs:",
                f"  predictions: {predictions_path}",
                f"  pf_node: {pf_node_path}",
                f"  pf_edge: {pf_edge_path}",
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
    assert (outputs_dir / "trust_analysis_volt.csv").exists()
    assert (outputs_dir / "trust_summary_volt.csv").exists()
    assert (outputs_dir / "trust_analysis_line.csv").exists()
    assert (outputs_dir / "trust_summary_line.csv").exists()
    assert (outputs_dir / "figures" / "vm_width_summary.png").exists()
    assert (outputs_dir / "figures" / "line_loading_width_summary.png").exists()


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
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: StateEstimation",
                "cp_type: naive",
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
                "    lower: 0.0",
                "    upper: 1.0",
                "inputs:",
                f"  predictions: {predictions_path}",
                f"  pf_node: {pf_node_path}",
                f"  pf_edge: {pf_edge_path}",
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

    volt_detail = pd.read_csv(screen_dir / "trust_screen_detail_volt.csv")
    volt_entity = pd.read_csv(screen_dir / "trust_screen_entity_volt.csv")
    volt_summary = pd.read_csv(screen_dir / "trust_screen_summary_volt.csv")
    line_detail = pd.read_csv(screen_dir / "trust_screen_detail_line.csv")
    line_summary = pd.read_csv(screen_dir / "trust_screen_summary_line.csv")

    assert set(volt_detail["variable"].unique().tolist()) == {"Vm"}
    assert set(line_detail["variable"].unique().tolist()) == {"line_loading"}
    assert {"pred_any_violation", "true_any_violation", "any_tp", "any_fp", "any_fn", "any_tn"}.issubset(volt_detail.columns)
    assert {"lower_precision", "upper_recall", "any_precision", "any_recall"}.issubset(volt_entity.columns)
    assert {"scope", "variable", "bus_type", "lower_tp", "upper_fp", "any_recall"}.issubset(volt_summary.columns)
    assert {"scope", "variable", "rate_a", "lower_tp", "upper_fp", "any_precision"}.issubset(line_summary.columns)

    np.testing.assert_array_equal(
        volt_detail["pred_any_violation"].to_numpy(dtype=bool),
        volt_detail["pred_lower_violation"].to_numpy(dtype=bool) | volt_detail["pred_upper_violation"].to_numpy(dtype=bool),
    )
    system_volt = volt_summary[volt_summary["scope"].eq("system")].iloc[0]
    assert int(system_volt["n_samples"]) == len(volt_detail)
    assert int(system_volt["any_tp"] + system_volt["any_fp"] + system_volt["any_fn"] + system_volt["any_tn"]) == len(volt_detail)
    system_line = line_summary[line_summary["scope"].eq("system")].iloc[0]
    assert int(system_line["n_samples"]) == len(line_detail)
    assert set(line_summary["scope"].unique().tolist()) == {"system", "rate_a"}

    assert (screen_dir / "screen_figures" / "vm_precision_bus_summary.png").exists()
    assert (screen_dir / "screen_figures" / "vm_any_confusion.png").exists()
    assert (screen_dir / "screen_figures" / "line_loading_recall_branch_summary.png").exists()


def test_normalize_variables_accepts_mixed_line_and_voltage_selection() -> None:
    assert normalize_variables("vm,va") == ("Vm", "Va")
    assert normalize_variables(["line-loading"]) == ("line_loading",)
    assert normalize_variables(["Vm", "line_loading", "va"]) == ("Vm", "line_loading", "Va")


def test_cli_flags_override_config(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    config_artifact_path = tmp_path / "config_intervals.trustlayer"
    override_artifact_path = tmp_path / "override_intervals.trustlayer"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    config_path.write_text(
        "\n".join(
            [
                "cp_type: naive",
                "alpha: 0.2",
                "variables:",
                "  - Vm",
                "inputs:",
                f"  predictions: {predictions_path}",
                f"  pf_node: {pf_node_path}",
                f"  pf_edge: {pf_edge_path}",
                "outputs:",
                f"  artifact: {config_artifact_path}",
            ],
        ),
        encoding="utf-8",
    )

    assert main(
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
    ) == 0

    assert override_artifact_path.exists()
    assert not config_artifact_path.exists()
    artifact = json.loads(override_artifact_path.read_text(encoding="utf-8"))
    assert artifact["channel_names"] == ["Vm", "Va"]


def test_compute_with_calibration_fraction_then_calibrate_same_pool_uses_holdout_only(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pf_node_path = tmp_path / "pf_node.parquet"
    pf_edge_path = tmp_path / "pf_edge.parquet"
    artifact_path = tmp_path / "split_intervals.trustlayer"
    output_path = tmp_path / "holdout_predictions_with_intervals.parquet"
    config_path = tmp_path / "trust_config.yaml"

    _write_prediction_fixture(predictions_path)
    _write_ground_truth_fixture(pf_node_path, pf_edge_path)
    _write_cli_config(config_path, predictions_path, pf_node_path, pf_edge_path)

    assert main(
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
            "--calibration-fraction",
            "0.5",
            "--random-seed",
            "7",
            "--output",
            str(artifact_path),
            "--task-name",
            "StateEstimation",
        ],
    ) == 0

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    calibration_scenarios = set(artifact["calibration_scenario_ids"])
    application_scenarios = set(artifact["application_scenario_ids"])
    assert calibration_scenarios
    assert application_scenarios
    assert calibration_scenarios.isdisjoint(application_scenarios)

    assert main(
        [
            "calibrate",
            "--config",
            str(config_path),
            "--predictions",
            str(predictions_path),
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
    ) == 0

    output_frame = pd.read_parquet(output_path)
    output_scenarios = set(output_frame["scenario"].unique().tolist())
    assert output_scenarios == application_scenarios
    assert output_scenarios.isdisjoint(calibration_scenarios)


def test_cli_requires_config_flag() -> None:
    with pytest.raises(SystemExit):
        main(["compute"])
    with pytest.raises(SystemExit):
        main(["calibrate"])


def _write_cli_config(
    config_path,
    predictions_path,
    pf_node_path,
    pf_edge_path,
    *,
    alpha: float = 0.2,
    task_name: str = "StateEstimation",
) -> None:
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                f"task_name: {task_name}",
                "cp_type: naive",
                f"alpha: {alpha}",
                "variables:",
                "  - Vm",
                "  - Va",
                "inputs:",
                f"  predictions: {predictions_path}",
                f"  pf_node: {pf_node_path}",
                f"  pf_edge: {pf_edge_path}",
            ],
        ),
        encoding="utf-8",
    )


def _write_prediction_fixture(path) -> None:
    frame = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1, 2, 2, 3, 3],
            "bus": [0, 1, 0, 1, 0, 1, 0, 1],
            "Vm_target": [1.05, 0.85, 1.12, 0.97, 1.0, 1.08, 1.0, 1.0],
            "Va_target": [0.02, 0.12, 0.16, -0.05, -0.1, 0.25, 0.08, -0.02],
            "Vm_pred": [1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 1.05, 0.98],
            "Va_pred": [0.0, 0.1, 0.2, -0.1, -0.2, 0.3, 0.1, 0.0],
            "P_inj_target": [2.0, -2.0, 2.1, -2.1, 1.9, -1.9, 2.0, -2.0],
            "P_inj_pred": [1.8, -1.8, 2.0, -2.0, 1.8, -1.8, 1.9, -1.9],
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