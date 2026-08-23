"""Load graphkit prediction parquets into aligned scenario-by-bus arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.io.task_adapters import BusPredictionBundle, resolve_task_adapter

FLOAT32_DTYPE = np.dtype(np.float32)


@dataclass
class GeneratorPredictionSet:
    """Generator prediction outputs preserved in the graphkit long-form layout.

    The current real OPF export is scenario-local rather than rectangular on a
    stable generator identity. Do not assume that `idx == i` refers to the same
    physical generator across scenarios.
    """

    channel_names: tuple[str, ...]
    scenario_ids: np.ndarray
    generator_ids: np.ndarray
    bus_ids: np.ndarray
    channels: dict[str, np.ndarray]
    target_channels: dict[str, np.ndarray]
    static_columns: dict[str, np.ndarray]


@dataclass
class PredictionSet:
    """Direct prediction outputs aligned by scenario and bus."""

    task_name: str
    channel_names: tuple[str, ...]
    scenario_ids: np.ndarray
    bus_ids: np.ndarray
    Vm: np.ndarray
    Va: np.ndarray
    channels: dict[str, np.ndarray]
    bus_type_masks: dict[str, np.ndarray]
    generator_predictions: GeneratorPredictionSet | None = None
    # Scenario-level representations z(x), attached by the CLI when
    # cp_type 'kernel' is requested. Rows align with `scenario_ids`.
    scenario_embeddings: np.ndarray | None = None


def load_predictions(
    parquet_path: str,
    apply_task_correction: bool = True,
    task_name: str | None = None,
    generator_parquet_path: str | None = None,
    *,
    apply_pf_mask: bool | None = None,
) -> PredictionSet:
    """Load a graphkit predictions parquet into aligned scenario-by-bus arrays.

    The loader keeps parquet-schema handling inside the Phase 2 I/O boundary,
    applies task-specific correction rules through explicit adapters, and loads
    OPF generator artifacts when present.
    """

    if apply_pf_mask is not None:
        apply_task_correction = apply_pf_mask

    frame = pd.read_parquet(parquet_path)
    _require_columns(frame, ("scenario", "bus"))
    frame = frame.copy()
    frame["scenario"] = frame["scenario"].astype(np.int64)
    frame["bus"] = frame["bus"].astype(np.int64)

    scenario_ids = _sorted_unique(frame["scenario"].to_numpy())
    bus_ids = _sorted_unique(frame["bus"].to_numpy())
    sorted_frame = _sort_rectangular_frame(frame, scenario_ids, bus_ids, "bus")
    lookup = {column.lower(): column for column in sorted_frame.columns}

    vm_pred_col = _resolve_column(lookup, "vm_pred")
    va_pred_col = _resolve_column(lookup, "va_pred")
    vm_target_col = _resolve_column(lookup, "vm_target", required=False)
    va_target_col = _resolve_column(lookup, "va_target", required=False)
    assert vm_pred_col is not None
    assert va_pred_col is not None

    vm = _reshape_column(
        sorted_frame,
        vm_pred_col,
        scenario_ids,
        bus_ids,
        "bus",
        FLOAT32_DTYPE,
    )
    va = _reshape_column(
        sorted_frame,
        va_pred_col,
        scenario_ids,
        bus_ids,
        "bus",
        FLOAT32_DTYPE,
    )

    bus_type_masks = _extract_bus_type_masks(sorted_frame, scenario_ids, bus_ids)
    adapter = resolve_task_adapter(
        task_name,
        parquet_path,
        generator_parquet_path,
        infer_power_flow={"PQ", "PV", "REF"}.issubset(bus_type_masks),
    )

    target_channels: dict[str, np.ndarray] = {}
    if vm_target_col is not None:
        target_channels["Vm"] = _reshape_column(
            sorted_frame,
            vm_target_col,
            scenario_ids,
            bus_ids,
            "bus",
            FLOAT32_DTYPE,
        )
    if va_target_col is not None:
        target_channels["Va"] = _reshape_column(
            sorted_frame,
            va_target_col,
            scenario_ids,
            bus_ids,
            "bus",
            FLOAT32_DTYPE,
        )

    channels: dict[str, np.ndarray] = {}
    extra_channel_names: list[str] = []
    for channel_name in _discover_prediction_channel_names(
        sorted_frame.columns,
        lookup,
        excluded_prefixes={"vm", "va"},
    ):
        pred_col = _resolve_column(lookup, f"{channel_name.lower()}_pred")
        assert pred_col is not None
        channels[channel_name] = _reshape_column(
            sorted_frame,
            pred_col,
            scenario_ids,
            bus_ids,
            "bus",
            FLOAT32_DTYPE,
        )
        target_col = _resolve_column(
            lookup, f"{channel_name.lower()}_target", required=False
        )
        if target_col is not None:
            target_channels[channel_name] = _reshape_column(
                sorted_frame,
                target_col,
                scenario_ids,
                bus_ids,
                "bus",
                FLOAT32_DTYPE,
            )
        extra_channel_names.append(channel_name)

    bus_predictions = BusPredictionBundle(Vm=vm, Va=va, channels=channels)
    if apply_task_correction:
        bus_predictions = adapter.apply_bus_corrections(
            bus_predictions,
            target_channels,
            bus_type_masks,
        )

    generator_predictions = None
    generator_path = adapter.resolve_generator_artifact_path(
        parquet_path, generator_parquet_path
    )
    if generator_path is not None:
        generator_predictions = _load_generator_predictions(
            generator_path, scenario_ids
        )

    channel_names = ("Vm", "Va", *extra_channel_names)
    return PredictionSet(
        task_name=adapter.task_name,
        channel_names=channel_names,
        scenario_ids=scenario_ids,
        bus_ids=bus_ids,
        Vm=bus_predictions.Vm,
        Va=bus_predictions.Va,
        channels=bus_predictions.channels,
        bus_type_masks=bus_type_masks,
        generator_predictions=generator_predictions,
    )


def _discover_prediction_channel_names(
    columns: pd.Index,
    lookup: dict[str, str],
    excluded_prefixes: set[str] | None = None,
) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()
    excluded = excluded_prefixes or set()
    for column in columns:
        lower = column.lower()
        if not lower.endswith("_pred"):
            continue
        prefix = column[:-5]
        if prefix.lower() in excluded:
            continue
        if f"{prefix.lower()}_target" not in lookup:
            continue
        if prefix.lower() in seen:
            continue
        discovered.append(prefix)
        seen.add(prefix.lower())
    return discovered


def _extract_bus_type_masks(
    frame: pd.DataFrame,
    scenario_ids: np.ndarray,
    bus_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    lookup = {column.lower(): column for column in frame.columns}
    masks: dict[str, np.ndarray] = {}
    for name in ("PQ", "PV", "REF"):
        column = lookup.get(name.lower())
        if column is None:
            continue
        masks[name] = _reshape_column(
            frame,
            column,
            scenario_ids,
            bus_ids,
            "bus",
            FLOAT32_DTYPE,
        ).astype(bool)
    return masks


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _resolve_column(
    lookup: dict[str, str],
    logical_name: str,
    required: bool = True,
) -> str | None:
    column = lookup.get(logical_name.lower())
    if column is None and required:
        raise ValueError(f"Missing required column: {logical_name}")
    return column


def _sorted_unique(values: np.ndarray) -> np.ndarray:
    return np.sort(np.unique(values))


def _sort_rectangular_frame(
    frame: pd.DataFrame,
    scenario_ids: np.ndarray,
    entity_ids: np.ndarray,
    entity_column: str,
) -> pd.DataFrame:
    expected_rows = len(scenario_ids) * len(entity_ids)
    if len(frame) != expected_rows:
        raise ValueError(
            "Expected a rectangular predictions parquet with one row per "
            f"(scenario, {entity_column}); got {len(frame)} rows for "
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
        raise ValueError("Predictions parquet is missing or duplicating scenario rows.")
    if not np.array_equal(ordered[entity_column].to_numpy(), expected_entities):
        raise ValueError(
            f"Predictions parquet is missing or duplicating {entity_column} rows.",
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


def _load_generator_predictions(
    parquet_path: str,
    expected_scenario_ids: np.ndarray,
) -> GeneratorPredictionSet:
    # Keep OPF generator artifacts in their exported long form. The current
    # graphkit export does not provide a stable cross-scenario generator ID, so
    # forcing `(scenario, idx)` into a rectangular matrix would misalign rows.
    frame = pd.read_parquet(parquet_path)
    _require_columns(frame, ("scenario", "idx", "bus"))
    frame = frame.copy()
    frame["scenario"] = frame["scenario"].astype(np.int64)
    frame["idx"] = frame["idx"].astype(np.int64)
    frame["bus"] = frame["bus"].astype(np.int64)

    scenario_ids = _sorted_unique(frame["scenario"].to_numpy())
    if not np.array_equal(scenario_ids, expected_scenario_ids):
        raise ValueError(
            "Generator predictions parquet scenarios do not match the bus predictions parquet.",
        )

    sorted_frame = frame.sort_values(
        ["scenario", "idx", "bus"], kind="stable"
    ).reset_index(drop=True)
    lookup = {column.lower(): column for column in sorted_frame.columns}

    channel_names = _discover_prediction_channel_names(sorted_frame.columns, lookup)
    channels: dict[str, np.ndarray] = {}
    target_channels: dict[str, np.ndarray] = {}
    dynamic_columns: set[str] = {"scenario", "idx", "bus"}
    for channel_name in channel_names:
        pred_col = _resolve_column(lookup, f"{channel_name.lower()}_pred")
        assert pred_col is not None
        channels[channel_name] = sorted_frame[pred_col].to_numpy(
            dtype=FLOAT32_DTYPE, copy=True
        )
        dynamic_columns.add(pred_col)

        target_col = _resolve_column(
            lookup, f"{channel_name.lower()}_target", required=False
        )
        if target_col is not None:
            target_channels[channel_name] = sorted_frame[target_col].to_numpy(
                dtype=FLOAT32_DTYPE,
                copy=True,
            )
            dynamic_columns.add(target_col)

    static_columns = {
        column: sorted_frame[column].to_numpy(copy=True)
        for column in sorted_frame.columns
        if column not in dynamic_columns
    }
    return GeneratorPredictionSet(
        channel_names=tuple(channel_names),
        scenario_ids=sorted_frame["scenario"].to_numpy(dtype=np.int64, copy=True),
        generator_ids=sorted_frame["idx"].to_numpy(dtype=np.int64, copy=True),
        bus_ids=sorted_frame["bus"].to_numpy(dtype=np.int64, copy=True),
        channels=channels,
        target_channels=target_channels,
        static_columns=static_columns,
    )
