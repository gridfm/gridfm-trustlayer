"""`apply`: bounds for new scenarios, without their AC solution.

On labeled data, `apply` must give exactly the bounds `calibrate` gives; on
unlabeled data (no solved values in the predictions or the scenario files) it
must still run, reading only what each layer needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer.cli import main
from tests.test_streaming import METHODS, _write_dataset

ROWS = {"volt": ["scenario", "bus"], "line": ["scenario", "branch"]}


def _config(
    tmp_path: Path, method: list[str], apply: dict[str, Path], variables: str
) -> Path:
    calibration = _write_dataset(tmp_path / "calibration", seed=1)
    test = _write_dataset(tmp_path / "test", seed=2)
    kernel = "cp_type: kernel" in method

    def block(name: str, paths: dict[str, Path]) -> list[str]:
        keys = ["predictions", "pf_node", "pf_edge"] + (
            ["embeddings"] if kernel else []
        )
        return [
            f"  {name}:",
            *(f"    {key}: {paths[key]}" for key in keys if key in paths),
        ]

    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: PowerFlow",
                *method,
                "alpha: 0.2",
                f"variables: {variables}",
                "save_figures: false",
                "inputs:",
                *block("calibration", calibration),
                *block("test", test),
                *block("apply", apply or test),
                "outputs:",
                f"  artifact: {tmp_path / 'layer.trustlayer'}",
                f"  calibrated_predictions: {tmp_path / 'calibrated'}",
                f"  applied_predictions: {tmp_path / 'applied'}",
            ]
        ),
        encoding="utf-8",
    )
    return config


def _rows(directory: Path, kind: str) -> pd.DataFrame:
    path = next(directory.glob(f"*{'line_loading' if kind == 'line' else 'vm_va'}*"))
    return pd.read_parquet(path).sort_values(ROWS[kind]).reset_index(drop=True)


def _bound_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if c.endswith(("_pi_lower", "_pi_upper")) or c.startswith("kcp_")
    ]


@pytest.mark.parametrize("method", list(METHODS))
def test_apply_gives_the_bounds_of_calibrate_on_labeled_data(tmp_path, method) -> None:
    config = _config(tmp_path, METHODS[method], {}, "[Vm, Va, line_loading]")
    for command in ("compute", "calibrate", "apply"):
        assert main([command, "--config", str(config)]) == 0

    for kind in ("volt", "line"):
        calibrated = _rows(tmp_path / "calibrated", kind)
        applied = _rows(tmp_path / "applied", kind)
        columns = _bound_columns(calibrated)
        assert columns
        pd.testing.assert_frame_equal(applied[columns], calibrated[columns])
    # No coverage reports: there is no truth to compare with.
    assert not list((tmp_path / "applied").glob("*.csv"))


def _strip_solution(paths: dict[str, Path], target: Path) -> dict[str, Path]:
    """Copies of a dataset without any solved value: what new scenarios look like."""

    target.mkdir()
    predictions = pd.read_parquet(paths["predictions"])
    # Only the set-points survive: Vm at PV buses and Va at the reference bus,
    # which are inputs of a power flow. Every other target is unknown.
    predictions["Vm_target"] = predictions["Vm_target"].where(predictions["PV"] == 1)
    predictions["Va_target"] = predictions["Va_target"].where(predictions["REF"] == 1)
    predictions.to_parquet(target / "predictions.parquet", index=False)
    bus = pd.read_parquet(paths["pf_node"]).drop(columns=["Vm", "Va", "Pg", "Qg"])
    bus.to_parquet(target / "bus_data.parquet", index=False)
    pd.read_parquet(paths["pf_edge"]).to_parquet(
        target / "branch_data.parquet", index=False
    )
    pd.read_parquet(paths["pf_node"].parent / "gen_data.parquet").to_parquet(
        target / "gen_data.parquet", index=False
    )
    return {
        "predictions": target / "predictions.parquet",
        "pf_node": target / "bus_data.parquet",
        "pf_edge": target / "branch_data.parquet",
        "embeddings": paths["embeddings"],
    }


def test_apply_needs_no_solved_values(tmp_path) -> None:
    labeled = _write_dataset(tmp_path / "new", seed=3)
    unlabeled = _strip_solution(labeled, tmp_path / "unlabeled")
    config = _config(
        tmp_path, METHODS["kernel_n_outages"], unlabeled, "[Vm, Va, line_loading]"
    )
    assert main(["compute", "--config", str(config)]) == 0
    assert main(["apply", "--config", str(config)]) == 0
    unlabeled_rows = {kind: _rows(tmp_path / "applied", kind) for kind in ROWS}

    config.write_text(
        config.read_text()
        .replace(
            str(tmp_path / "unlabeled" / "predictions.parquet"),
            str(labeled["predictions"]),
        )
        .replace(
            str(tmp_path / "unlabeled" / "bus_data.parquet"), str(labeled["pf_node"])
        )
        .replace(
            str(tmp_path / "unlabeled" / "branch_data.parquet"), str(labeled["pf_edge"])
        ),
        encoding="utf-8",
    )
    assert main(["apply", "--config", str(config)]) == 0
    for kind in ROWS:
        labeled_rows = _rows(tmp_path / "applied", kind)
        columns = _bound_columns(labeled_rows)
        pd.testing.assert_frame_equal(
            unlabeled_rows[kind][columns], labeled_rows[columns]
        )
        # No true values in the output: for new scenarios they are unknown.
        for rows in (unlabeled_rows[kind], labeled_rows):
            assert not [c for c in rows.columns if c.endswith("_target")]


def test_apply_reads_scenario_descriptions_only_when_the_layer_needs_them(
    tmp_path,
) -> None:
    unlabeled = _strip_solution(
        _write_dataset(tmp_path / "new", seed=3), tmp_path / "unlabeled"
    )
    predictions_only = {"predictions": unlabeled["predictions"]}

    config = _config(tmp_path, ["cp_type: element"], predictions_only, "[Vm]")
    assert main(["compute", "--config", str(config)]) == 0
    assert main(["apply", "--config", str(config)]) == 0
    applied = pd.read_parquet(next((tmp_path / "applied").glob("*")))
    assert np.isfinite(applied["Vm_pi_upper"]).all()

    stratified = tmp_path / "stratified"
    stratified.mkdir()
    config = _config(
        stratified, ["cp_type: element", "strat: n_outages"], predictions_only, "[Vm]"
    )
    assert main(["compute", "--config", str(config)]) == 0
    with pytest.raises(ValueError, match="strat: n_outages"):
        main(["apply", "--config", str(config)])


def test_apply_does_not_overwrite_calibrate_by_default(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, ["cp_type: element"], {}, "[Vm]")
    text = config.read_text().split("\n")
    config.write_text(
        "\n".join(
            line
            for line in text
            if "calibrated_predictions" not in line
            and "applied_predictions" not in line
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for command in ("compute", "calibrate", "apply"):
        assert main([command, "--config", str(config)]) == 0

    outputs = sorted(
        path.name
        for path in (
            tmp_path / "trustlayer_outputs" / "powerflow" / "element"
        ).iterdir()
    )
    assert "predictions_vm_with_intervals.parquet" in outputs
    assert "predictions_vm_bounds.parquet" in outputs


def test_apply_requires_the_power_flow_set_points(tmp_path) -> None:
    unlabeled = _strip_solution(
        _write_dataset(tmp_path / "new", seed=3), tmp_path / "unlabeled"
    )
    predictions = pd.read_parquet(unlabeled["predictions"])
    predictions.drop(columns=["Vm_target", "Va_target"]).to_parquet(
        unlabeled["predictions"], index=False
    )
    config = _config(
        tmp_path,
        ["cp_type: element"],
        {"predictions": unlabeled["predictions"]},
        "[Vm]",
    )
    assert main(["compute", "--config", str(config)]) == 0
    with pytest.raises(ValueError, match="set-points"):
        main(["apply", "--config", str(config)])
