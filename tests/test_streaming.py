"""Block-wise processing: `calibrate` and `screen` must not depend on the block size.

The test set is read, calibrated, screened and written one block of scenarios at
a time. These tests run the same configuration as a single block and split into
many blocks, and require identical reports and rows. The synthetic data is built
so that per-block shortcuts would show: one branch is out of service in every
scenario of the first block only (the outage reference must span the dataset),
and the angles of the first block are small enough to look like radians (the
degree detection must span the dataset too).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import gridfm_trustlayer.io.parquet as parquet_io
from gridfm_trustlayer.cli import main
from gridfm_trustlayer.io import load_ground_truth
from gridfm_trustlayer.io.ground_truth import scan_ground_truth_reference

# The package re-exports `main` the function, which shadows the module name.
cli_main = importlib.import_module("gridfm_trustlayer.cli.main")

N_SCENARIOS = 60
FIRST_BLOCK = 10  # scenarios per block in the split runs
BRANCHES = [(0, 1), (1, 2), (0, 2)]
BUS_TYPES = {0: "REF", 1: "PV", 2: "PQ"}


def _write_dataset(root: Path, seed: int) -> dict[str, Path]:
    """Calibration-or-test dataset: partitioned ground truth, flat predictions."""

    rng = np.random.default_rng(seed)
    scenarios = np.arange(N_SCENARIOS)
    n_buses = len(BUS_TYPES)

    status = (rng.random((N_SCENARIOS, len(BRANCHES))) > 0.25).astype(float)
    status[:, 0] = 1.0
    status[:FIRST_BLOCK, 2] = 0.0  # out in the whole first block only
    status[FIRST_BLOCK:, 2] = 1.0
    gen_status = (rng.random((N_SCENARIOS, 2)) > 0.2).astype(float)
    outages = (status == 0).sum(axis=1) + (gen_status == 0).sum(axis=1)

    vm = 1.0 + 0.02 * rng.standard_normal((N_SCENARIOS, n_buses))
    va = 0.1 * rng.standard_normal((N_SCENARIOS, n_buses))
    va[:FIRST_BLOCK] *= 0.05  # below 2 pi even in degrees
    load = rng.uniform(10.0, 50.0, (N_SCENARIOS, n_buses))
    flags = {
        name: np.array([BUS_TYPES[b] == name for b in range(n_buses)])
        for name in ("REF", "PV", "PQ")
    }

    raw = root / "raw"
    bus_rows = pd.DataFrame(
        {
            "scenario": np.repeat(scenarios, n_buses),
            "bus": np.tile(np.arange(n_buses), N_SCENARIOS),
            "load_scenario_idx": np.repeat(scenarios // 2, n_buses),
            "Pd": load.reshape(-1),
            "Qd": 0.0,
            "Vm": vm.reshape(-1),
            "Va": np.rad2deg(va).reshape(-1),  # datakit stores degrees
            "Pg": 0.0,
            "Qg": 0.0,
            "vn_kv": 100.0,
            **{
                name: np.tile(mask, N_SCENARIOS).astype(float)
                for name, mask in flags.items()
            },
        }
    )
    branch_rows = pd.DataFrame(
        {
            "scenario": np.repeat(scenarios, len(BRANCHES)),
            "idx": np.tile(np.arange(len(BRANCHES)), N_SCENARIOS),
            "from_bus": np.tile([f for f, _ in BRANCHES], N_SCENARIOS),
            "to_bus": np.tile([t for _, t in BRANCHES], N_SCENARIOS),
            "rate_a": 0.05,
            "br_status": status.reshape(-1),
            "Yff_r": 1.0,
            "Yff_i": -5.0,
            "Yft_r": -1.0,
            "Yft_i": 5.0,
            "Ytf_r": -1.0,
            "Ytf_i": 5.0,
            "Ytt_r": 1.0,
            "Ytt_i": -5.0,
        }
    )
    gen_rows = pd.DataFrame(
        {
            "scenario": np.repeat(scenarios, 2),
            "idx": np.tile([0, 1], N_SCENARIOS),
            "bus": np.tile([0, 1], N_SCENARIOS),
            "in_service": gen_status.reshape(-1),
        }
    )
    for name, frame in (
        ("bus_data", bus_rows),
        ("branch_data", branch_rows),
        ("gen_data", gen_rows),
    ):
        frame = frame.assign(scenario_partition=frame["scenario"] // 20)
        frame.to_parquet(raw / f"{name}.parquet", partition_cols=["scenario_partition"])

    noise = 0.004 * (1 + outages)[:, None] * rng.standard_normal((N_SCENARIOS, n_buses))
    predictions = pd.DataFrame(
        {
            "scenario": np.repeat(scenarios, n_buses),
            "bus": np.tile(np.arange(n_buses), N_SCENARIOS),
            "Vm_target": vm.reshape(-1),
            "Va_target": va.reshape(-1),
            "Vm_pred": (vm + noise).reshape(-1),
            "Va_pred": (va + noise).reshape(-1),
            **{
                name: np.tile(mask, N_SCENARIOS).astype(float)
                for name, mask in flags.items()
            },
        }
    )
    predictions.to_parquet(root / "predictions.parquet", index=False)

    embeddings = pd.DataFrame(
        {
            "scenario": np.repeat(scenarios, n_buses),
            "bus": np.tile(np.arange(n_buses), N_SCENARIOS),
            **{
                f"emb_{i:03d}": rng.standard_normal(N_SCENARIOS * n_buses)
                for i in range(3)
            },
        }
    )
    embeddings.to_parquet(root / "embeddings.parquet", index=False)
    return {
        "predictions": root / "predictions.parquet",
        "embeddings": root / "embeddings.parquet",
        "pf_node": raw / "bus_data.parquet",
        "pf_edge": raw / "branch_data.parquet",
    }


def _config(tmp_path: Path, run: str, method: list[str]) -> Path:
    calibration = _write_dataset(tmp_path / "calibration", seed=1)
    test = _write_dataset(tmp_path / "test", seed=2)
    kernel = "cp_type: kernel" in method

    def block(name: str, paths: dict[str, Path]) -> list[str]:
        lines = [f"  {name}:"]
        for key in ("predictions", "pf_node", "pf_edge"):
            lines.append(f"    {key}: {paths[key]}")
        if kernel:
            lines.append(f"    embeddings: {paths['embeddings']}")
        return lines

    out = tmp_path / run
    config = tmp_path / f"{run}.yaml"
    config.write_text(
        "\n".join(
            [
                "version: 1",
                "task_name: PowerFlow",
                *method,
                "alpha: 0.2",
                "variables: [Vm, Va, line_loading]",
                "screen_bounds:",
                "  Vm: {lower: 0.98, upper: 1.02}",
                "  line_loading: {upper: 1.0}",
                "inputs:",
                *block("calibration", calibration),
                *block("test", test),
                "outputs:",
                f"  artifact: {tmp_path / 'layer.trustlayer'}",
                f"  calibrated_predictions: {out / 'calibrated'}",
                f"  screen_dir: {out / 'screen'}",
                "save_figures: false",
                "commands:",
                "  screen:",
                "    variables: [Vm, line_loading]",
            ]
        ),
        encoding="utf-8",
    )
    return config


def _run(config: Path) -> None:
    for command in ("calibrate", "screen"):
        assert main([command, "--config", str(config)]) == 0


def _read_rows(path: Path, keys: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path).drop(columns=parquet_io.PARTITION_COLUMN)
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.StringDtype):
            frame[column] = frame[column].astype(object)
    return frame.sort_values(keys).reset_index(drop=True)


METHODS = {
    "element_n_outages": ["cp_type: element", "strat: n_outages"],
    "element_outage_type": ["cp_type: element", "strat: outage_type"],
    "element_system_load_bin": [
        "cp_type: element",
        "strat: system_load_bin",
        "system_load_n_bins: 3",
    ],
    "element_pred_bins": ["cp_type: element", "strat: pred_bins", "pred_n_bins: 3"],
    "kernel_n_outages": [
        "cp_type: kernel",
        "strat: n_outages",
        "n_eff_target: 8",
        "n_eff_min: 2",
    ],
}


@pytest.mark.parametrize("method", list(METHODS))
def test_calibrate_and_screen_do_not_depend_on_the_block_size(
    tmp_path, monkeypatch, method
) -> None:
    whole = _config(tmp_path, "whole", METHODS[method])
    assert main(["compute", "--config", str(whole)]) == 0
    _run(whole)

    split = Path(str(whole).replace("whole", "split"))
    split.write_text(whole.read_text().replace("/whole/", "/split/"), encoding="utf-8")
    monkeypatch.setattr(parquet_io, "SCENARIOS_PER_PARTITION", FIRST_BLOCK)
    monkeypatch.setattr(cli_main, "_BLOCK_ROWS", 1)
    _run(split)

    whole_dir, split_dir = tmp_path / "whole", tmp_path / "split"
    assert len(
        list((split_dir / "screen" / "screening_detail_line_loading.parquet").iterdir())
    ) == (N_SCENARIOS // FIRST_BLOCK)
    csv_files = sorted(path.relative_to(whole_dir) for path in whole_dir.rglob("*.csv"))
    assert len(csv_files) == 8
    for relative in csv_files:
        pd.testing.assert_frame_equal(
            pd.read_csv(split_dir / relative),
            pd.read_csv(whole_dir / relative),
            check_exact=False,
            rtol=1e-12,
            obj=str(relative),
        )
    parquets = {
        "calibrated/predictions_vm_va_with_intervals.parquet": ["scenario", "bus"],
        "calibrated/predictions_line_loading_with_intervals.parquet": [
            "scenario",
            "branch",
        ],
        "screen/screening_detail_voltage.parquet": ["variable", "scenario", "bus"],
        "screen/screening_detail_line_loading.parquet": [
            "variable",
            "scenario",
            "branch",
        ],
    }
    for relative, keys in parquets.items():
        pd.testing.assert_frame_equal(
            _read_rows(split_dir / relative, keys),
            _read_rows(whole_dir / relative, keys),
            obj=relative,
        )


def test_block_loading_with_the_dataset_reference_matches_a_full_load(tmp_path) -> None:
    paths = _write_dataset(tmp_path, seed=3)
    full = load_ground_truth(str(paths["pf_node"]), str(paths["pf_edge"]))
    reference = scan_ground_truth_reference(
        str(paths["pf_node"]), str(paths["pf_edge"])
    )
    first = np.arange(FIRST_BLOCK)

    block = load_ground_truth(
        str(paths["pf_node"]),
        str(paths["pf_edge"]),
        scenario_ids=first,
        reference=reference,
    )
    np.testing.assert_array_equal(block.n_outages, full.n_outages[first])
    np.testing.assert_array_equal(block.outage_type, full.outage_type[first])
    np.testing.assert_allclose(block.Va, full.Va[first])

    # Without the reference, the first block alone would see branch 2 as never in
    # service (no outage) and its small angles as radians.
    naive = load_ground_truth(
        str(paths["pf_node"]), str(paths["pf_edge"]), scenario_ids=first
    )
    assert not np.array_equal(naive.n_outages, full.n_outages[first])
    assert not np.allclose(naive.Va, full.Va[first])


def test_kernel_runs_read_embeddings_once_or_per_block_with_the_same_result(
    tmp_path, monkeypatch
) -> None:
    once = _config(tmp_path, "once", METHODS["kernel_n_outages"])
    assert main(["compute", "--config", str(once)]) == 0
    _run(once)

    per_block = Path(str(once).replace("once", "per_block"))
    per_block.write_text(
        once.read_text().replace("/once/", "/per_block/"), encoding="utf-8"
    )
    monkeypatch.setattr(cli_main, "_EMBEDDINGS_PRELOAD_BYTES", 0)
    monkeypatch.setattr(parquet_io, "SCENARIOS_PER_PARTITION", FIRST_BLOCK)
    monkeypatch.setattr(cli_main, "_BLOCK_ROWS", 1)
    _run(per_block)

    relative = "calibrated/predictions_vm_va_with_intervals.parquet"
    keys = ["scenario", "bus"]
    pd.testing.assert_frame_equal(
        _read_rows(tmp_path / "per_block" / relative, keys),
        _read_rows(tmp_path / "once" / relative, keys),
    )
