"""Load datakit ground-truth parquets into aligned scenario-by-bus arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


FLOAT32_DTYPE = np.dtype(np.float32)
OUTAGE_TYPE_NONE = np.int32(0)
OUTAGE_TYPE_BRANCH_ONLY = np.int32(1)
OUTAGE_TYPE_GEN_ONLY = np.int32(2)
OUTAGE_TYPE_MIXED = np.int32(3)


@dataclass
class GroundTruthSet:
    """Ground-truth state and topology aligned by scenario, bus, and branch."""

    scenario_ids: np.ndarray
    bus_ids: np.ndarray
    branch_ids: np.ndarray
    n_outages: np.ndarray
    outage_type: np.ndarray
    system_load: np.ndarray
    removed_mask: np.ndarray
    Vm: np.ndarray
    Va: np.ndarray
    channels: dict[str, np.ndarray]
    admittances: dict[str, np.ndarray]
    f_bus: np.ndarray
    t_bus: np.ndarray
    rate_a: np.ndarray
    base_kv: np.ndarray
    sn_mva: float
    bus_type_masks: dict[str, np.ndarray]


def load_ground_truth(
    pf_node_parquet: str,
    pf_edge_parquet: str,
    bus_params_csv: str | None = None,
    edge_params_csv: str | None = None,
    *,
    scenario_ids: Iterable[int] | None = None,
    channel_names: Iterable[str] | None = None,
    sn_mva: float = 100.0,
) -> GroundTruthSet:
    """Load datakit raw bus and branch parquets into aligned arrays.

    The current datakit raw parquet files already contain the bus-level and
    branch-level metadata required for Phase 2. Optional CSV paths are kept in
    the signature for future compatibility, but the loader prefers the parquet
    content when it is available.
    """

    bus_frame = pd.read_parquet(pf_node_parquet)
    branch_frame = pd.read_parquet(pf_edge_parquet)
    _require_columns(bus_frame, ("scenario", "bus", "Vm", "Va"))
    _require_columns(
        branch_frame,
        (
            "scenario",
            "idx",
            "from_bus",
            "to_bus",
            "rate_a",
            "br_status",
            "Yff_r",
            "Yff_i",
            "Yft_r",
            "Yft_i",
            "Ytf_r",
            "Ytf_i",
            "Ytt_r",
            "Ytt_i",
        ),
    )

    bus_frame = bus_frame.copy()
    branch_frame = branch_frame.copy()
    bus_frame["scenario"] = bus_frame["scenario"].astype(np.int64)
    bus_frame["bus"] = bus_frame["bus"].astype(np.int64)
    branch_frame["scenario"] = branch_frame["scenario"].astype(np.int64)
    branch_frame["idx"] = branch_frame["idx"].astype(np.int64)
    branch_frame["from_bus"] = branch_frame["from_bus"].astype(np.int32)
    branch_frame["to_bus"] = branch_frame["to_bus"].astype(np.int32)

    selected_scenarios = _resolve_scenario_ids(bus_frame, scenario_ids)
    bus_frame = bus_frame[bus_frame["scenario"].isin(selected_scenarios)].copy()
    branch_frame = branch_frame[
        branch_frame["scenario"].isin(selected_scenarios)
    ].copy()

    bus_ids = np.sort(bus_frame["bus"].unique().astype(np.int64, copy=False))
    branch_ids = np.sort(branch_frame["idx"].unique().astype(np.int64, copy=False))
    sorted_bus = _sort_rectangular_frame(bus_frame, selected_scenarios, bus_ids, "bus")
    sorted_branch = _sort_rectangular_frame(
        branch_frame, selected_scenarios, branch_ids, "idx"
    )

    vm = _reshape_column(
        sorted_bus, "Vm", selected_scenarios, bus_ids, "bus", FLOAT32_DTYPE
    )
    va = _reshape_column(
        sorted_bus, "Va", selected_scenarios, bus_ids, "bus", FLOAT32_DTYPE
    )
    va = _normalize_angle_units(va)

    direct_channel_names = _resolve_direct_channel_names(
        sorted_bus.columns, channel_names
    )
    channels = {
        channel_name: _reshape_column(
            sorted_bus,
            channel_name,
            selected_scenarios,
            bus_ids,
            "bus",
            FLOAT32_DTYPE,
        )
        for channel_name in direct_channel_names
    }

    bus_type_masks: dict[str, np.ndarray] = {}
    for name in ("PQ", "PV", "REF"):
        if name in sorted_bus.columns:
            bus_type_masks[name] = _reshape_column(
                sorted_bus,
                name,
                selected_scenarios,
                bus_ids,
                "bus",
                FLOAT32_DTYPE,
            ).astype(bool)

    admittances = {
        key: _reshape_column(
            sorted_branch,
            key,
            selected_scenarios,
            branch_ids,
            "idx",
            FLOAT32_DTYPE,
        )
        for key in (
            "Yff_r",
            "Yff_i",
            "Yft_r",
            "Yft_i",
            "Ytf_r",
            "Ytf_i",
            "Ytt_r",
            "Ytt_i",
        )
    }
    br_status = _reshape_column(
        sorted_branch,
        "br_status",
        selected_scenarios,
        branch_ids,
        "idx",
        FLOAT32_DTYPE,
    )
    removed_mask = br_status != 1.0
    branch_outages = _count_reference_outages(br_status)
    generator_outages = _load_generator_outage_counts(
        pf_node_parquet,
        pf_edge_parquet,
        selected_scenarios,
    )
    n_outages = (branch_outages + generator_outages).astype(np.int32, copy=False)
    outage_type = _classify_outage_type(branch_outages, generator_outages)
    system_load = _resolve_system_load(bus_frame, selected_scenarios)

    first_branch_rows = (
        sorted_branch[sorted_branch["scenario"] == selected_scenarios[0]]
        .sort_values("idx", kind="stable")
        .reset_index(drop=True)
    )
    f_bus = first_branch_rows["from_bus"].to_numpy(dtype=np.int32, copy=True)
    t_bus = first_branch_rows["to_bus"].to_numpy(dtype=np.int32, copy=True)
    rate_a = first_branch_rows["rate_a"].to_numpy(dtype=np.float32, copy=True)

    base_kv = _resolve_base_kv(sorted_bus, bus_params_csv, bus_ids)
    if edge_params_csv is not None:
        edge_params = pd.read_csv(edge_params_csv)
        if "rate_a" in edge_params.columns and len(edge_params) == len(rate_a):
            rate_a = edge_params["rate_a"].to_numpy(dtype=np.float32, copy=True)

    return GroundTruthSet(
        scenario_ids=selected_scenarios,
        bus_ids=bus_ids,
        branch_ids=branch_ids,
        n_outages=n_outages,
        outage_type=outage_type,
        system_load=system_load,
        removed_mask=removed_mask,
        Vm=vm,
        Va=va,
        channels=channels,
        admittances=admittances,
        f_bus=f_bus,
        t_bus=t_bus,
        rate_a=rate_a,
        base_kv=base_kv,
        sn_mva=float(sn_mva),
        bus_type_masks=bus_type_masks,
    )


def _resolve_base_kv(
    sorted_bus: pd.DataFrame,
    bus_params_csv: str | None,
    bus_ids: np.ndarray,
) -> np.ndarray:
    if "vn_kv" in sorted_bus.columns:
        first_bus_rows = (
            sorted_bus[["bus", "vn_kv"]]
            .drop_duplicates(subset=["bus"])
            .sort_values("bus", kind="stable")
        )
        return first_bus_rows["vn_kv"].to_numpy(dtype=np.float32, copy=True)

    if bus_params_csv is not None:
        bus_params = pd.read_csv(bus_params_csv)
        base_column = "vn_kv" if "vn_kv" in bus_params.columns else "base_kv"
        ordered = (
            bus_params.sort_values("bus", kind="stable")
            if "bus" in bus_params.columns
            else bus_params
        )
        values = ordered[base_column].to_numpy(dtype=np.float32, copy=True)
        if len(values) != len(bus_ids):
            raise ValueError("bus_params_csv does not match the number of buses.")
        return values

    raise ValueError(
        "Unable to resolve bus base voltage from parquet or bus_params_csv."
    )


def _resolve_direct_channel_names(
    columns: pd.Index,
    requested_channel_names: Iterable[str] | None,
) -> list[str]:
    if requested_channel_names is not None:
        selected = [
            name for name in requested_channel_names if name not in {"Vm", "Va"}
        ]
        return [name for name in selected if name in columns]

    defaults = [name for name in ("Pg", "Qg") if name in columns]
    return defaults


def _resolve_system_load(
    bus_frame: pd.DataFrame,
    scenario_ids: np.ndarray,
) -> np.ndarray:
    if "Pd" not in bus_frame.columns:
        return np.full(len(scenario_ids), np.nan, dtype=np.float32)
    per_scenario = bus_frame.groupby("scenario", sort=False)["Pd"].sum()
    ordered = per_scenario.reindex(scenario_ids.tolist())
    return ordered.to_numpy(dtype=np.float32, copy=True)


def _load_generator_outage_counts(
    pf_node_parquet: str,
    pf_edge_parquet: str,
    scenario_ids: np.ndarray,
) -> np.ndarray:
    generator_path = _default_generator_parquet_path(pf_node_parquet, pf_edge_parquet)
    if generator_path is None:
        return np.zeros(len(scenario_ids), dtype=np.int32)

    gen_frame = pd.read_parquet(generator_path)
    _require_columns(gen_frame, ("scenario", "idx", "in_service"))
    gen_frame = gen_frame.copy()
    gen_frame["scenario"] = gen_frame["scenario"].astype(np.int64)
    gen_frame["idx"] = gen_frame["idx"].astype(np.int64)
    gen_frame = gen_frame[gen_frame["scenario"].isin(scenario_ids)].copy()
    if gen_frame.empty:
        return np.zeros(len(scenario_ids), dtype=np.int32)

    generator_ids = np.sort(gen_frame["idx"].unique().astype(np.int64, copy=False))
    sorted_gen = _sort_rectangular_frame(gen_frame, scenario_ids, generator_ids, "idx")
    gen_status = _reshape_column(
        sorted_gen,
        "in_service",
        scenario_ids,
        generator_ids,
        "idx",
        FLOAT32_DTYPE,
    )
    return _count_reference_outages(gen_status)


def _default_generator_parquet_path(
    pf_node_parquet: str,
    pf_edge_parquet: str,
) -> str | None:
    for parent in (Path(pf_node_parquet).parent, Path(pf_edge_parquet).parent):
        candidate = parent / "gen_data.parquet"
        if candidate.exists():
            return str(candidate)
    return None


def _count_reference_outages(status: np.ndarray) -> np.ndarray:
    if status.size == 0:
        return np.zeros(status.shape[0], dtype=np.int32)
    reference_active = np.nanmax(status, axis=0, keepdims=True) == 1.0
    return np.sum(reference_active & (status != 1.0), axis=1).astype(np.int32)


def _classify_outage_type(
    branch_outages: np.ndarray,
    generator_outages: np.ndarray,
) -> np.ndarray:
    branch_present = branch_outages > 0
    generator_present = generator_outages > 0
    outage_type = np.full(len(branch_outages), OUTAGE_TYPE_NONE, dtype=np.int32)
    outage_type[branch_present & (~generator_present)] = OUTAGE_TYPE_BRANCH_ONLY
    outage_type[(~branch_present) & generator_present] = OUTAGE_TYPE_GEN_ONLY
    outage_type[branch_present & generator_present] = OUTAGE_TYPE_MIXED
    return outage_type


def _normalize_angle_units(values: np.ndarray) -> np.ndarray:
    """Return voltage angles in radians.

    Datakit raw bus parquets store `Va` in degrees for the real OPF/PF exports
    used in this repository, while graphkit prediction artifacts expose `Va`
    and `Va_target` in radians. Normalize the raw ground truth at the I/O
    boundary so calibration compares like with like.

    The heuristic is intentionally conservative: only convert when at least one
    angle magnitude exceeds $2\\pi$, which is not expected for the direct-output
    prediction representation but is common for degree-scale exports.
    """

    if values.size == 0:
        return values
    if float(np.nanmax(np.abs(values))) > float(2.0 * np.pi):
        return np.deg2rad(values).astype(FLOAT32_DTYPE, copy=False)
    return values


def _resolve_scenario_ids(
    bus_frame: pd.DataFrame,
    requested_scenario_ids: Iterable[int] | None,
) -> np.ndarray:
    available = set(bus_frame["scenario"].unique().tolist())
    if requested_scenario_ids is None:
        return np.sort(np.array(list(available), dtype=np.int64))

    ordered = np.array(
        [int(scenario_id) for scenario_id in requested_scenario_ids], dtype=np.int64
    )
    missing = [
        scenario_id for scenario_id in ordered.tolist() if scenario_id not in available
    ]
    if missing:
        raise ValueError(f"Requested scenarios not present in bus parquet: {missing}")
    return ordered


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _sort_rectangular_frame(
    frame: pd.DataFrame,
    scenario_ids: np.ndarray,
    entity_ids: np.ndarray,
    entity_column: str,
) -> pd.DataFrame:
    expected_rows = len(scenario_ids) * len(entity_ids)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected one row per (scenario, {entity_column}); got {len(frame)} rows for "
            f"{len(scenario_ids)} scenarios and {len(entity_ids)} {entity_column} values.",
        )

    ordered = frame.copy()
    ordered["_scenario_order"] = pd.Categorical(
        ordered["scenario"],
        categories=scenario_ids,
        ordered=True,
    )
    ordered["_entity_order"] = pd.Categorical(
        ordered[entity_column],
        categories=entity_ids,
        ordered=True,
    )
    ordered = ordered.sort_values(["_scenario_order", "_entity_order"], kind="stable")
    ordered = ordered.drop(columns=["_scenario_order", "_entity_order"]).reset_index(
        drop=True
    )

    expected_scenarios = np.repeat(scenario_ids, len(entity_ids))
    expected_entities = np.tile(entity_ids, len(scenario_ids))
    if not np.array_equal(ordered["scenario"].to_numpy(), expected_scenarios):
        raise ValueError(
            "Ground-truth parquet is missing or duplicating scenario rows."
        )
    if not np.array_equal(ordered[entity_column].to_numpy(), expected_entities):
        raise ValueError(
            f"Ground-truth parquet is missing or duplicating {entity_column} rows.",
        )
    return ordered


def _reshape_column(
    frame: pd.DataFrame,
    column: str,
    scenario_ids: np.ndarray,
    entity_ids: np.ndarray,
    entity_column: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    ordered = _sort_rectangular_frame(
        frame[["scenario", entity_column, column]],
        scenario_ids,
        entity_ids,
        entity_column,
    )
    values = ordered[column].to_numpy(dtype=dtype, copy=True)
    return values.reshape(len(scenario_ids), len(entity_ids))
