"""Tests for the Phase 2 data I/O layer."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.io import (
    compute_line_loading,
    load_ground_truth,
    load_predictions,
)


def test_predictions_roundtrip(tmp_path) -> None:
    path = tmp_path / "predictions.parquet"
    _write_pf_predictions_fixture(path)

    prediction_set = load_predictions(str(path), apply_task_correction=False)

    assert prediction_set.task_name == "PowerFlow"
    assert prediction_set.channel_names == ("Vm", "Va", "Pg", "Qg")
    assert prediction_set.Vm.shape == (2, 3)
    assert prediction_set.Va.shape == (2, 3)
    assert prediction_set.channels["Pg"].shape == (2, 3)
    assert prediction_set.channels["Qg"].shape == (2, 3)
    np.testing.assert_array_equal(prediction_set.scenario_ids, np.array([10, 11]))
    np.testing.assert_array_equal(prediction_set.bus_ids, np.array([0, 1, 2]))
    assert prediction_set.Vm[0, 1] == np.float32(0.91)
    assert prediction_set.Va[0, 0] == np.float32(1.0)


def test_pf_mask_correction(tmp_path) -> None:
    path = tmp_path / "predictions.parquet"
    _write_pf_predictions_fixture(path)

    prediction_set = load_predictions(str(path), apply_task_correction=True)

    assert prediction_set.Vm[0, 0] == np.float32(0.90)
    assert prediction_set.Va[0, 0] == np.float32(0.0)
    assert prediction_set.Vm[0, 1] == np.float32(1.02)
    assert prediction_set.Va[0, 1] == np.float32(0.2)
    assert prediction_set.Vm[0, 2] == np.float32(0.98)
    assert prediction_set.channels["Pg"][0, 0] == np.float32(120.0)
    assert prediction_set.channels["Pg"][0, 1] == np.float32(80.0)
    assert prediction_set.channels["Pg"][0, 2] == np.float32(0.0)
    assert prediction_set.channels["Qg"][0, 1] == np.float32(9.0)
    assert prediction_set.channels["Qg"][0, 2] == np.float32(0.0)


def test_opf_prediction_artifacts(tmp_path) -> None:
    bus_path = tmp_path / "case24_ieee_rts_predictions.parquet"
    gen_path = tmp_path / "case24_ieee_rts_gen_predictions.parquet"
    _write_opf_predictions_fixture(bus_path, gen_path)

    prediction_set = load_predictions(
        str(bus_path),
        task_name="OptimalPowerFlow",
        generator_parquet_path=str(gen_path),
    )

    assert prediction_set.task_name == "OptimalPowerFlow"
    assert prediction_set.generator_predictions is not None
    assert prediction_set.generator_predictions.channel_names == ("p_mw",)
    assert prediction_set.generator_predictions.channels["p_mw"].shape == (4,)
    np.testing.assert_array_equal(
        prediction_set.generator_predictions.bus_ids,
        np.array([0, 1, 0, 1]),
    )
    np.testing.assert_array_equal(
        prediction_set.generator_predictions.scenario_ids,
        np.array([0, 0, 1, 1]),
    )
    np.testing.assert_allclose(
        prediction_set.generator_predictions.channels["p_mw"],
        np.array([19.0, 11.0, 20.0, 9.5], dtype=np.float32),
    )


def test_task_adapter_selection(tmp_path) -> None:
    pf_path = tmp_path / "pf_predictions.parquet"
    _write_pf_predictions_fixture(pf_path)
    opf_bus_path = tmp_path / "case24_ieee_rts_predictions.parquet"
    opf_gen_path = tmp_path / "case24_ieee_rts_gen_predictions.parquet"
    _write_opf_predictions_fixture(opf_bus_path, opf_gen_path)

    pf_prediction_set = load_predictions(str(pf_path), apply_task_correction=False)
    opf_prediction_set = load_predictions(
        str(opf_bus_path), apply_task_correction=False
    )

    assert pf_prediction_set.task_name == "PowerFlow"
    assert pf_prediction_set.generator_predictions is None
    assert opf_prediction_set.task_name == "OptimalPowerFlow"
    assert opf_prediction_set.generator_predictions is not None


def test_opf_clamping_rules(tmp_path) -> None:
    bus_path = tmp_path / "case24_ieee_rts_predictions.parquet"
    gen_path = tmp_path / "case24_ieee_rts_gen_predictions.parquet"
    _write_opf_predictions_fixture(bus_path, gen_path)

    prediction_set = load_predictions(
        str(bus_path),
        task_name="OptimalPowerFlow",
        generator_parquet_path=str(gen_path),
    )

    assert prediction_set.Vm[0, 0] == np.float32(0.97)
    assert prediction_set.Va[0, 0] == np.float32(0.0)
    assert prediction_set.channels["Pg"][0, 1] == np.float32(8.0)
    assert prediction_set.channels["Qg"][0, 1] == np.float32(4.2)


def test_ground_truth_roundtrip_and_outages(tmp_path) -> None:
    bus_path = tmp_path / "bus_data.parquet"
    branch_path = tmp_path / "branch_data.parquet"
    _write_ground_truth_fixture(bus_path, branch_path)

    ground_truth = load_ground_truth(str(bus_path), str(branch_path))

    assert ground_truth.Vm.shape == (2, 2)
    assert ground_truth.Va.shape == (2, 2)
    assert ground_truth.channels["Pg"].shape == (2, 2)
    assert ground_truth.channels["Qg"].shape == (2, 2)
    np.testing.assert_array_equal(
        ground_truth.n_outages, np.array([0, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        ground_truth.outage_type, np.array([0, 3], dtype=np.int32)
    )
    np.testing.assert_allclose(
        ground_truth.system_load, np.array([0.0, 0.0], dtype=np.float32)
    )
    assert not bool(ground_truth.removed_mask[0, 0])
    assert bool(ground_truth.removed_mask[1, 0])


def test_ground_truth_converts_degree_angles_to_radians(tmp_path) -> None:
    bus_path = tmp_path / "bus_data.parquet"
    branch_path = tmp_path / "branch_data.parquet"
    _write_ground_truth_fixture(bus_path, branch_path)

    bus_frame = pd.read_parquet(bus_path)
    bus_frame["Va"] = [180.0, -90.0, 45.0, 0.0]
    bus_frame.to_parquet(bus_path, index=False)

    ground_truth = load_ground_truth(str(bus_path), str(branch_path))

    np.testing.assert_allclose(
        ground_truth.Va,
        np.array([[np.pi, -np.pi / 2.0], [np.pi / 4.0, 0.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_line_loading_simple_case_and_removed_sentinel(tmp_path) -> None:
    bus_path = tmp_path / "bus_data.parquet"
    branch_path = tmp_path / "branch_data.parquet"
    _write_ground_truth_fixture(bus_path, branch_path)

    ground_truth = load_ground_truth(str(bus_path), str(branch_path), sn_mva=100.0)
    loading = compute_line_loading(ground_truth.Vm, ground_truth.Va, ground_truth)

    assert loading.shape == (2, 1)
    np.testing.assert_allclose(loading[0, 0], np.float32(1.0), atol=1e-6)
    assert loading[1, 0] == np.float32(-1.0)


def test_task_specific_channels_are_preserved(tmp_path) -> None:
    path = tmp_path / "predictions.parquet"
    frame = pd.DataFrame(
        {
            "scenario": [0, 0],
            "bus": [0, 1],
            "Vm_target": [1.0, 0.99],
            "Va_target": [0.0, 0.1],
            "Vm_pred": [0.98, 0.97],
            "Va_pred": [0.02, 0.08],
            "P_inj_target": [2.0, -2.0],
            "Q_inj_target": [1.0, -1.0],
            "P_inj_pred": [1.5, -1.5],
            "Q_inj_pred": [0.8, -0.8],
        },
    )
    frame.to_parquet(path, index=False)

    prediction_set = load_predictions(
        str(path),
        apply_task_correction=False,
        task_name="StateEstimation",
    )

    assert prediction_set.task_name == "StateEstimation"
    assert prediction_set.channel_names == ("Vm", "Va", "P_inj", "Q_inj")
    np.testing.assert_allclose(
        prediction_set.channels["P_inj"], np.array([[1.5, -1.5]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        prediction_set.channels["Q_inj"], np.array([[0.8, -0.8]], dtype=np.float32)
    )


def test_io_imports_without_graphkit_runtime_dependency() -> None:
    import gridfm_trustlayer.io as io_module

    assert io_module.load_predictions is not None
    assert io_module.load_ground_truth is not None
    assert io_module.compute_line_loading is not None


def _write_pf_predictions_fixture(path) -> None:
    frame = pd.DataFrame(
        {
            "scenario": [10, 10, 10, 11, 11, 11],
            "bus": [0, 1, 2, 0, 1, 2],
            "Pd": [0.0] * 6,
            "Qd": [0.0] * 6,
            "Vm_min": [0.95] * 6,
            "Vm_max": [1.05] * 6,
            "Qg_min": [-50.0] * 6,
            "Qg_max": [50.0] * 6,
            "Vm_target": [1.03, 1.02, 0.99, 1.04, 1.01, 0.98],
            "Va_target": [0.0, 0.5, -1.0, 0.0, 0.4, -0.8],
            "Pg_target": [100.0, 80.0, 0.0, 101.0, 81.0, 0.0],
            "Qg_target": [10.0, 8.0, 0.0, 11.0, 7.0, 0.0],
            "PQ": [0, 0, 1, 0, 0, 1],
            "PV": [0, 1, 0, 0, 1, 0],
            "REF": [1, 0, 0, 1, 0, 0],
            "Vm_pred": [0.90, 0.91, 0.98, 0.92, 0.93, 0.97],
            "Va_pred": [1.0, 0.2, -1.1, 0.8, 0.1, -0.9],
            "Pg_pred": [120.0, 75.0, 7.0, 118.0, 76.0, 6.0],
            "Qg_pred": [12.0, 9.0, 5.0, 13.0, 8.5, 4.5],
            "active res. (MW)": [0.0] * 6,
            "reactive res. (MVar)": [0.0] * 6,
            "PBE": [0.0] * 6,
        },
    )
    frame.to_parquet(path, index=False)


def _write_opf_predictions_fixture(bus_path, gen_path) -> None:
    bus_frame = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1],
            "bus": [0, 1, 0, 1],
            "Pd": [10.0, 6.0, 11.0, 5.5],
            "Qd": [3.0, 2.0, 3.1, 1.9],
            "Vm_min": [0.95, 0.95, 0.95, 0.95],
            "Vm_max": [1.05, 1.05, 1.05, 1.05],
            "Qg_min": [-50.0, -50.0, -50.0, -50.0],
            "Qg_max": [80.0, 80.0, 80.0, 80.0],
            "Vm_target": [1.02, 1.01, 1.03, 1.00],
            "Va_target": [0.0, 0.1, 0.0, 0.2],
            "Pg_target": [20.0, 10.0, 21.0, 9.0],
            "Qg_target": [5.0, 4.0, 5.5, 3.8],
            "PQ": [0, 0, 0, 0],
            "PV": [0, 1, 0, 1],
            "REF": [1, 0, 1, 0],
            "Vm_pred": [0.97, 0.96, 0.98, 0.95],
            "Va_pred": [0.3, 0.05, 0.35, 0.18],
            "Pg_pred": [19.0, 8.0, 20.0, 9.5],
            "Qg_pred": [5.2, 4.2, 5.1, 3.9],
            "active res. (MW)": [0.0, 0.0, 0.0, 0.0],
            "reactive res. (MVar)": [0.0, 0.0, 0.0, 0.0],
            "PBE": [0.0, 0.0, 0.0, 0.0],
        },
    )
    gen_frame = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1],
            "idx": [0, 1, 0, 1],
            "bus": [0, 1, 0, 1],
            "p_mw_target": [20.0, 10.0, 21.0, 9.0],
            "p_mw_pred": [19.0, 11.0, 20.0, 9.5],
            "min_p_mw": [16.0, 8.0, 16.0, 8.0],
            "max_p_mw": [24.0, 12.0, 24.0, 12.0],
            "cp0_eur": [1.0, 2.0, 1.0, 2.0],
            "cp1_eur_per_mw": [0.1, 0.2, 0.1, 0.2],
            "cp2_eur_per_mw2": [0.01, 0.02, 0.01, 0.02],
        },
    )
    bus_frame.to_parquet(bus_path, index=False)
    gen_frame.to_parquet(gen_path, index=False)


def _write_ground_truth_fixture(bus_path, branch_path) -> None:
    bus_frame = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1],
            "bus": [0, 1, 0, 1],
            "Pd": [0.0, 0.0, 0.0, 0.0],
            "Qd": [0.0, 0.0, 0.0, 0.0],
            "Pg": [1.0, 0.0, 1.0, 0.0],
            "Qg": [0.0, 0.0, 0.0, 0.0],
            "Vm": [1.0, 0.9, 1.0, 0.9],
            "Va": [0.0, 0.0, 0.0, 0.0],
            "PQ": [0, 1, 0, 1],
            "PV": [1, 0, 1, 0],
            "REF": [0, 0, 0, 0],
            "vn_kv": [100.0, 100.0, 100.0, 100.0],
        },
    )
    branch_frame = pd.DataFrame(
        {
            "scenario": [0, 1],
            "idx": [0, 0],
            "from_bus": [0, 0],
            "to_bus": [1, 1],
            "rate_a": [10.0, 10.0],
            "br_status": [1.0, 0.0],
            "Yff_r": [1.0, 1.0],
            "Yff_i": [0.0, 0.0],
            "Yft_r": [-1.0, -1.0],
            "Yft_i": [0.0, 0.0],
            "Ytf_r": [-1.0, -1.0],
            "Ytf_i": [0.0, 0.0],
            "Ytt_r": [1.0, 1.0],
            "Ytt_i": [0.0, 0.0],
        },
    )
    gen_frame = pd.DataFrame(
        {
            "scenario": [0, 0, 1, 1],
            "idx": [0, 1, 0, 1],
            "bus": [0, 1, 0, 1],
            "in_service": [1.0, 1.0, 1.0, 0.0],
        },
    )
    bus_frame.to_parquet(bus_path, index=False)
    branch_frame.to_parquet(branch_path, index=False)
    gen_frame.to_parquet(bus_path.parent / "gen_data.parquet", index=False)
