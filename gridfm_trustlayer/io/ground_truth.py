"""Load datakit ground-truth parquets into aligned scenario-by-bus arrays."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.io.parquet import open_dataset, read_frame

FLOAT32_DTYPE = np.dtype(np.float32)
OUTAGE_TYPE_NONE = np.int32(0)
OUTAGE_TYPE_BRANCH_ONLY = np.int32(1)
OUTAGE_TYPE_GEN_ONLY = np.int32(2)
OUTAGE_TYPE_MIXED = np.int32(3)

_BRANCH_COLUMNS = (
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
)
_BUS_COLUMNS = ("scenario", "bus", "Vm", "Va", "Pd", "PQ", "PV", "REF", "vn_kv")
# Bus columns that hold the AC solution rather than the scenario description.
_SOLUTION_BUS_COLUMNS = ("Vm", "Va")


@dataclass
class GroundTruthSet:
    """True AC solution and network data of one dataset, from `load_ground_truth`.

    Attributes:
        scenario_ids: Scenario identifiers, shape `(n_scenarios,)`.
        bus_ids: Bus identifiers, shape `(n_buses,)`.
        branch_ids: Branch identifiers, shape `(n_branches,)`.
        n_outages: Out-of-service branches plus generators, per scenario.
        outage_type: Outage regime per scenario: 0 `none`, 1 `branch_only`,
            2 `gen_only`, 3 `mixed`.
        system_load: Total active load per scenario.
        removed_mask: `(n_scenarios, n_branches)`, `True` for out-of-service branches.
        Vm: True voltage magnitudes, `(n_scenarios, n_buses)`.
        Va: True voltage angles in radians, `(n_scenarios, n_buses)`.
        channels: Every true bus variable, including `Vm` and `Va`.
        admittances: Branch admittance coefficients, `(n_scenarios, n_branches)` each.
        f_bus: From-bus of each branch.
        t_bus: To-bus of each branch.
        rate_a: Rating of each branch.
        base_kv: Base voltage of each bus.
        sn_mva: System base power.
        bus_type_masks: Boolean `PQ`, `PV` and `REF` masks.
    """

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


@dataclass(frozen=True)
class GroundTruthReference:
    """Dataset-wide facts that loading a subset of scenarios must not recompute.

    `n_outages` counts the branches and generators out of service among those in
    service in at least one scenario, and `Va` is converted from degrees when
    some angle exceeds 2 pi. Both depend on which scenarios are loaded, so a
    reader that loads a dataset block by block computes them once, with
    `scan_ground_truth_reference`, and passes them to `load_ground_truth`.
    """

    branch_ids: np.ndarray
    branch_in_service: np.ndarray
    generator_ids: np.ndarray | None
    generator_in_service: np.ndarray | None
    va_in_degrees: bool


def scan_ground_truth_reference(
    pf_node_parquet: str,
    pf_edge_parquet: str,
) -> GroundTruthReference:
    """Compute a `GroundTruthReference` over a whole dataset, one batch at a time."""

    branch_ids, branch_in_service = _scan_in_service(pf_edge_parquet, "br_status")
    generator_path = _default_generator_parquet_path(pf_node_parquet, pf_edge_parquet)
    generator_ids = generator_in_service = None
    if generator_path is not None:
        generator_ids, generator_in_service = _scan_in_service(
            generator_path, "in_service"
        )
    va_max = 0.0
    bus_dataset = open_dataset(pf_node_parquet)
    va_batches = (
        bus_dataset.to_batches(columns=["Va"])
        if "Va" in bus_dataset.schema.names
        else []
    )
    for batch in va_batches:
        values = np.abs(batch.column(0).to_numpy(zero_copy_only=False))
        if values.size and np.isfinite(values).any():
            va_max = max(va_max, float(np.nanmax(values)))
    return GroundTruthReference(
        branch_ids=branch_ids,
        branch_in_service=branch_in_service,
        generator_ids=generator_ids,
        generator_in_service=generator_in_service,
        va_in_degrees=va_max > float(2.0 * np.pi),
    )


def _scan_in_service(path: str, status_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Element ids, and whether each is in service in at least one scenario."""

    in_service: dict[int, bool] = {}
    for batch in open_dataset(path).to_batches(columns=["idx", status_column]):
        ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        active = batch.column(1).to_numpy(zero_copy_only=False) == 1
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        any_active = np.zeros(len(unique_ids), dtype=bool)
        np.logical_or.at(any_active, inverse, active)
        for element, flag in zip(unique_ids.tolist(), any_active.tolist(), strict=True):
            in_service[element] = in_service.get(element, False) or flag
    element_ids = np.array(sorted(in_service), dtype=np.int64)
    return element_ids, np.array(
        [in_service[i] for i in element_ids.tolist()], dtype=bool
    )


def _reference_in_service(
    reference_ids: np.ndarray,
    reference_in_service: np.ndarray,
    element_ids: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(reference_ids, element_ids)
    positions = np.clip(positions, 0, max(len(reference_ids) - 1, 0))
    if len(reference_ids) == 0 or not np.array_equal(
        reference_ids[positions], element_ids
    ):
        raise ValueError("Ground-truth elements do not match the dataset reference.")
    return reference_in_service[positions][None, :]


def load_ground_truth(
    pf_node_parquet: str,
    pf_edge_parquet: str,
    bus_params_csv: str | None = None,
    edge_params_csv: str | None = None,
    *,
    scenario_ids: Iterable[int] | None = None,
    channel_names: Iterable[str] | None = None,
    sn_mva: float = 100.0,
    reference: GroundTruthReference | None = None,
    solved: bool = True,
) -> GroundTruthSet:
    """Read datakit's raw bus and branch parquets into arrays.

    Args:
        pf_node_parquet: Bus ground truth, datakit `raw/bus_data.parquet`.
        pf_edge_parquet: Branch ground truth, datakit `raw/branch_data.parquet`.
        bus_params_csv: Unused; kept for compatibility.
        edge_params_csv: Unused; kept for compatibility.
        scenario_ids: Scenarios to load, e.g. `predictions.scenario_ids`, so
            that the result is aligned with the predictions. All when not given.
        channel_names: Bus variables to load besides `Vm` and `Va`.
        sn_mva: System base power.
        reference: Dataset-wide outage reference and angle unit, from
            `scan_ground_truth_reference`, when loading a subset of a dataset
            block by block. Computed from the loaded scenarios when not given.
        solved: Whether to read the AC solution (`Vm`, `Va` and the other bus
            channels). With `False`, only the description of each scenario is
            read (topology, branch parameters, loads, bus types), which is all
            that bounds for new scenarios need; `Vm` and `Va` are then NaN.

    Returns:
        The ground truth. `Va` is converted to radians if stored in degrees.
    """

    requested_channels = (
        ["Pg", "Qg"] if channel_names is None else [str(name) for name in channel_names]
    )
    bus_columns = [*_BUS_COLUMNS, *requested_channels]
    if not solved:
        bus_columns = [
            name for name in _BUS_COLUMNS if name not in _SOLUTION_BUS_COLUMNS
        ]
    bus_frame = read_frame(pf_node_parquet, bus_columns, scenario_ids=scenario_ids)
    branch_frame = read_frame(
        pf_edge_parquet, _BRANCH_COLUMNS, scenario_ids=scenario_ids
    )
    _require_columns(
        bus_frame, ("scenario", "bus", "Vm", "Va") if solved else ("scenario", "bus")
    )
    _require_columns(branch_frame, _BRANCH_COLUMNS)

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

    if solved:
        vm = _reshape_column(
            sorted_bus, "Vm", selected_scenarios, bus_ids, "bus", FLOAT32_DTYPE
        )
        va = _reshape_column(
            sorted_bus, "Va", selected_scenarios, bus_ids, "bus", FLOAT32_DTYPE
        )
        va = _normalize_angle_units(
            va, in_degrees=None if reference is None else reference.va_in_degrees
        )
    else:
        vm = np.full((len(selected_scenarios), len(bus_ids)), np.nan, FLOAT32_DTYPE)
        va = vm.copy()

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
    branch_outages = _count_reference_outages(
        br_status,
        None
        if reference is None
        else _reference_in_service(
            reference.branch_ids, reference.branch_in_service, branch_ids
        ),
    )
    generator_outages = _load_generator_outage_counts(
        pf_node_parquet,
        pf_edge_parquet,
        selected_scenarios,
        reference,
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
    reference: GroundTruthReference | None = None,
) -> np.ndarray:
    generator_path = _default_generator_parquet_path(pf_node_parquet, pf_edge_parquet)
    if generator_path is None:
        return np.zeros(len(scenario_ids), dtype=np.int32)

    gen_frame = read_frame(
        generator_path, ("scenario", "idx", "in_service"), scenario_ids=scenario_ids
    )
    _require_columns(gen_frame, ("scenario", "idx", "in_service"))
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
    reference_active = None
    if reference is not None and reference.generator_ids is not None:
        assert reference.generator_in_service is not None
        reference_active = _reference_in_service(
            reference.generator_ids, reference.generator_in_service, generator_ids
        )
    return _count_reference_outages(gen_status, reference_active)


def _default_generator_parquet_path(
    pf_node_parquet: str,
    pf_edge_parquet: str,
) -> str | None:
    for parent in (Path(pf_node_parquet).parent, Path(pf_edge_parquet).parent):
        candidate = parent / "gen_data.parquet"
        if candidate.exists():
            return str(candidate)
    return None


def _count_reference_outages(
    status: np.ndarray,
    reference_active: np.ndarray | None = None,
) -> np.ndarray:
    """Elements out of service per scenario, among those in service somewhere.

    `reference_active` (shape `(1, n_elements)`) marks the elements in service in
    at least one scenario of the dataset; computed from `status` when not given.
    """

    if status.size == 0:
        return np.zeros(status.shape[0], dtype=np.int32)
    if reference_active is None:
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


def _normalize_angle_units(
    values: np.ndarray, *, in_degrees: bool | None = None
) -> np.ndarray:
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
    if in_degrees is None:
        in_degrees = float(np.nanmax(np.abs(values))) > float(2.0 * np.pi)
    if in_degrees:
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

    if _in_rectangular_order(frame, scenario_ids, entity_ids, entity_column):
        return frame.reset_index(drop=True)
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
    if not _in_rectangular_order(frame, scenario_ids, entity_ids, entity_column):
        frame = _sort_rectangular_frame(
            frame[["scenario", entity_column, column]],
            scenario_ids,
            entity_ids,
            entity_column,
        )
    values = frame[column].to_numpy(dtype=dtype, copy=True)
    return values.reshape(len(scenario_ids), len(entity_ids))


def _in_rectangular_order(
    frame: pd.DataFrame,
    scenario_ids: np.ndarray,
    entity_ids: np.ndarray,
    entity_column: str,
) -> bool:
    """Whether rows are already one per (scenario, entity), in that order."""

    shape = (len(scenario_ids), len(entity_ids))
    if len(frame) != shape[0] * shape[1]:
        return False
    scenarios = frame["scenario"].to_numpy().reshape(shape)
    entities = frame[entity_column].to_numpy().reshape(shape)
    return bool(
        (scenarios == np.asarray(scenario_ids)[:, None]).all()
        and (entities == np.asarray(entity_ids)[None, :]).all()
    )
