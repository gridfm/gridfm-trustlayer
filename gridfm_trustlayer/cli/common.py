"""Helpers shared by the CLI commands: data assembly, strata, paths and statistics."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.cli import reports, status
from gridfm_trustlayer.cli.config import (
    coalesce_option,
    normalize_variables,
)
from gridfm_trustlayer.conformal import ChannelSet, NaiveIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import (
    KernelIntervalCalibrator,
)
from gridfm_trustlayer.io import (
    GroundTruthSet,
    PredictionSet,
    compute_line_loading,
)
from gridfm_trustlayer.io.embeddings import (
    load_scenario_embeddings,
)
from gridfm_trustlayer.io.ground_truth import (
    GroundTruthReference,
    scan_ground_truth_reference,
)
from gridfm_trustlayer.io.parquet import (
    DEFAULT_BLOCK_ROWS,
    PartitionedWriter,
    column_names,
    open_dataset,
)

SCREEN_BOUND_TOLERANCE = 1e-6


# Rows (scenario x entity) that calibrate and screen hold in memory at once.
_BLOCK_ROWS = DEFAULT_BLOCK_ROWS


# Scenarios whose predictions enter the in-sample fingerprint (with every id).
_FINGERPRINT_SCENARIOS = 1000


# Test embeddings are read and pooled once when the file is small enough to
# decode at once (e.g. one row per scenario); larger ones, such as per-bus
# embeddings of a big grid, are read again for each block.
_EMBEDDINGS_PRELOAD_BYTES = 1 << 30


BASE_ANALYSIS_STRAT_COLUMNS = ("n_outages", "outage_type", "system_load_bin")


def _sparse_strata_listing(
    calibrators: Mapping[str, NaiveIntervalCalibrator | KernelIntervalCalibrator],
) -> str:
    """E.g. `Vm: 27 buses (n_outages=1: 27)`, or '' when no stratum is sparse."""

    parts: list[str] = []
    for calibrator in calibrators.values():
        nouns = (
            ("line", "lines")
            if calibrator.entity_name == "branch"
            else ("bus", "buses")
        )
        counts = calibrator.sparse_strata_counts()
        for variable in sorted(counts, key=reports.variable_rank):
            by_stratum = counts[variable]
            if not by_stratum:
                continue
            total = sum(by_stratum.values())
            where = ", ".join(
                f"{_sparse_stratum_label(calibrator.strat, variable, value)}: {count}"
                for value, count in by_stratum.items()
            )
            parts.append(f"{variable}: {total} {nouns[total != 1]} ({where})")
    return "; ".join(parts)


def _sparse_stratum_label(strat: str | None, variable: str, value: int) -> str:
    if strat == "pred_bins":
        return f"pred_bins_{variable.lower()} bin {value}"
    return _format_stratum_label(str(strat), value)


class _EmbeddingSource:
    """Scenario embeddings for a kernel artifact, attached block by block.

    Small files are read and pooled once; larger ones (e.g. per-bus embeddings
    of a big grid) are read again for each block, which keeps memory bounded.
    """

    def __init__(
        self,
        path: str,
        pooling: str,
        scenario_ids: np.ndarray,
        *,
        preload: bool = True,
    ) -> None:
        self.path = path
        self.pooling = pooling
        self.preloaded = None
        # Parallel workers read only their own blocks (`preload=False`).
        if preload and _decoded_size_estimate(path) <= _EMBEDDINGS_PRELOAD_BYTES:
            self.preloaded = load_scenario_embeddings(
                path, pooling=pooling, scenario_ids=scenario_ids
            )

    @classmethod
    def for_artifact(
        cls,
        calibrators: dict[str, NaiveIntervalCalibrator | KernelIntervalCalibrator],
        args: argparse.Namespace,
        config: dict[str, object],
        scenario_ids: np.ndarray,
    ) -> _EmbeddingSource | None:
        spec = _embedding_spec(calibrators, args, config)
        return None if spec is None else cls(*spec, scenario_ids)

    def attach(self, predictions: PredictionSet) -> None:
        if self.preloaded is not None:
            predictions.scenario_embeddings = self.preloaded.align_to(
                predictions.scenario_ids
            )
        else:
            _attach_scenario_embeddings(predictions, self.path, pooling=self.pooling)


def _ground_truth_reference(
    calibrators: Iterable[NaiveIntervalCalibrator | KernelIntervalCalibrator],
    paths: tuple[str, str],
) -> GroundTruthReference:
    """The calibration set's reference stored in the artifact, else one of `paths`.

    Outages are counted, and angles converted, as in the calibration set, so a
    scenario gets the same stratum whatever other scenarios it comes with.
    Artifacts written before the reference was stored fall back to a scan of
    the data at hand.
    """

    for calibrator in calibrators:
        if calibrator.ground_truth_reference is not None:
            return GroundTruthReference.from_dict(calibrator.ground_truth_reference)
    return scan_ground_truth_reference(*paths)


def _embedding_spec(
    calibrators: dict[str, NaiveIntervalCalibrator | KernelIntervalCalibrator],
    args: argparse.Namespace,
    config: dict[str, object],
) -> tuple[str, str] | None:
    """`(path, pooling)` of the embeddings a kernel artifact needs, or None."""

    poolings = {
        calibrator.embedding_pooling
        for calibrator in calibrators.values()
        if isinstance(calibrator, KernelIntervalCalibrator)
    }
    if not poolings:
        return None
    if len(poolings) != 1:
        raise ValueError(
            f"Kernel artifact mixes embedding_pooling strategies: {sorted(poolings)}."
        )
    path = coalesce_option(args.embeddings, config, "embeddings", required=True)
    return str(path), next(iter(poolings))


# Quantities that are inputs, not results, of the task's problem: the voltage
# magnitude of PV buses and the angle of the reference bus in a power flow, and
# the reference angle in an OPF. The loaders replace the model's predictions there
# with these set-points, read from the `<variable>_target` columns, so new
# scenarios need them too (their values at other buses are not used).
_SET_POINTS = {
    "PowerFlow": (("Vm", "PV"), ("Va", "REF")),
    "OptimalPowerFlow": (("Va", "REF"),),
}


_ELEMENT_NAMES = {"bus": "bus", "branch": "line"}


def _add_known_columns(
    frame: pd.DataFrame, variables: tuple[str, ...], task_name: str
) -> pd.DataFrame:
    """Mark the rows whose value is a known set-point, `<variable>_known`.

    These values are inputs of the problem, so their bounds have zero width and
    contain the value by construction: coverage and screening leave them out.
    """

    for variable, bus_type in _SET_POINTS.get(task_name, ()):
        if variable in variables and bus_type in frame.columns:
            frame[f"{variable}_known"] = frame[bus_type].to_numpy() == 1
    return frame


def _report_progress(command: str, index: int, blocks: list[np.ndarray]) -> None:
    """One line per block, so that long runs show where they are."""

    if len(blocks) > 1:
        done = sum(len(block) for block in blocks[:index])
        total = sum(len(block) for block in blocks)
        print(
            f"{command}: block {index}/{len(blocks)} ({done:,}/{total:,} scenarios)",
            flush=True,
        )


def _common_parent(paths: Iterable[str]) -> str:
    parents = [Path(path).parent for path in paths]
    return str(
        parents[0] if len(set(parents)) == 1 else Path(os.path.commonpath(parents))
    )


def _voltage_target_frame(
    ground_truth: GroundTruthSet,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    """True bus values as `<variable>_target` columns, one row per scenario and bus."""

    frame = pd.DataFrame(
        {
            "scenario": np.repeat(ground_truth.scenario_ids, len(ground_truth.bus_ids)),
            "bus": np.tile(ground_truth.bus_ids, len(ground_truth.scenario_ids)),
        },
    )
    for variable in variables:
        frame[f"{variable}_target"] = _ground_truth_channel(
            ground_truth, variable
        ).reshape(-1)
    return frame


def _ground_truth_paths(
    args: argparse.Namespace,
    config: dict[str, object],
) -> tuple[str, str]:
    pf_node = coalesce_option(args.pf_node, config, "pf_node")
    pf_edge = coalesce_option(args.pf_edge, config, "pf_edge")
    if pf_node is None or pf_edge is None:
        raise ValueError(
            "The test ground truth is required: set inputs.test.pf_node and "
            "inputs.test.pf_edge (or --pf-node / --pf-edge).",
        )
    return str(pf_node), str(pf_edge)


def _decoded_size_estimate(path: str) -> int:
    """Rough in-memory size of a whole dataset once decoded, from its metadata."""

    dataset = open_dataset(path)
    return dataset.count_rows() * len(dataset.schema.names) * 4 * 3


def _scan_bus_types(path: str) -> dict[int, str]:
    """Bus type of every bus (REF, PV, PQ), from the type flags across all scenarios."""

    flags = [name for name in ("REF", "PV", "PQ") if name in column_names(path)]
    if not flags:
        return {}
    maxima: list[pd.DataFrame] = []
    for batch in open_dataset(path).to_batches(columns=["bus", *flags]):
        frame = batch.to_pandas()
        maxima.append(frame.groupby("bus").max(numeric_only=True))
    combined = pd.concat(maxima).groupby(level=0).max()
    return {
        int(bus): _bus_type_label_from_flags(
            ref=float(row.get("REF", 0.0)),
            pv=float(row.get("PV", 0.0)),
            pq=float(row.get("PQ", 0.0)),
        )
        for bus, row in combined.iterrows()
    }


_SCENARIOS_FILE_TEXT = (
    "one row per operating scenario: its status overall and per variable, by "
    "interval and by point prediction"
)


_STATUS_FILE_TEXT = (
    "number of scenarios and values in each status, per variable and stratum, by "
    "interval and by point prediction"
)


def _scenario_strata(columns: Iterable[str]) -> list[str]:
    """The scenario-level stratification columns (and labels) among `columns`."""

    base = set(BASE_ANALYSIS_STRAT_COLUMNS)
    return [c for c in columns if c in base or c.removesuffix("_label") in base]


def _write_status_tables(
    scenario_parts: list[pd.DataFrame],
    element_counts: list[pd.DataFrame],
    directory: Path,
    *,
    truth: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`scenarios.csv` and `status_by_stratum.csv`; returns the table and the counts."""

    table = status.scenario_table(scenario_parts, truth=truth)
    scenario_counts = status.scenario_counts(
        table,
        [c for c in BASE_ANALYSIS_STRAT_COLUMNS if c in table.columns],
        label=_stratum_label,
        truth=truth,
    )
    counts = status.combine_counts([scenario_counts, *element_counts], truth=truth)
    reports.write_csv(
        status.labelled_scenarios(table, truth=truth), directory / "scenarios.csv"
    )
    reports.write_csv(
        counts.drop(columns=["stratification", "stratum_value"]),
        directory / "status_by_stratum.csv",
    )
    return table, counts


def _partition_variables(
    channel_names: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for channel_name in channel_names:
        target_kind = "line" if channel_name == "line_loading" else "volt"
        grouped.setdefault(target_kind, []).append(channel_name)
    return {target_kind: tuple(names) for target_kind, names in grouped.items()}


def _prediction_fingerprint(scenario_ids: np.ndarray, sample: PredictionSet) -> str:
    """SHA-256 over all scenario ids and the predictions of the first scenarios.

    `sample` holds (at least) the first `_FINGERPRINT_SCENARIOS` scenarios in id
    order. That is enough to recognize the same dataset under another name, and
    `calibrate` can check it without reading the whole test set.
    """

    count = min(_FINGERPRINT_SCENARIOS, len(sample.scenario_ids))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(scenario_ids, dtype=np.int64).tobytes())
    for array in (
        sample.scenario_ids[:count],
        sample.bus_ids,
        sample.Vm[:count],
        sample.Va[:count],
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    for name in sorted(sample.channels):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(sample.channels[name][:count]).tobytes())
    return digest.hexdigest()


def _parse_screen_bounds(raw_bounds: object) -> dict[str, dict[str, float]]:
    """`screen_bounds` of the config, as `{variable: {"lower": ..., "upper": ...}}`."""

    bounds_by_variable: dict[str, dict[str, float]] = {}
    if raw_bounds is not None:
        if not isinstance(raw_bounds, dict):
            raise ValueError(
                "screen_bounds must be a mapping from variable name to lower/upper bounds."
            )
        for raw_name, raw_value in raw_bounds.items():
            variable = normalize_variables([str(raw_name)])[0]
            if not isinstance(raw_value, dict):
                raise ValueError(
                    f"screen_bounds.{raw_name} must be a mapping with screening bounds."
                )
            if variable == "line_loading":
                if "upper" not in raw_value:
                    raise ValueError(f"screen_bounds.{raw_name} must define 'upper'.")
                lower_value = float(raw_value.get("lower", float("nan")))
            else:
                if "lower" not in raw_value or "upper" not in raw_value:
                    raise ValueError(
                        f"screen_bounds.{raw_name} must define both 'lower' and 'upper'."
                    )
                lower_value = float(raw_value["lower"])
            bounds_by_variable[variable] = {
                "lower": lower_value,
                "upper": float(raw_value["upper"]),
            }
    return bounds_by_variable


def _build_interval_frame(
    predictions: ChannelSet,
    intervals: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    scenario_ids = predictions.scenario_ids
    entity_ids = predictions.entity_ids
    entity_column = predictions.entity_name
    frame = pd.DataFrame(
        {
            "scenario": np.repeat(scenario_ids, len(entity_ids)),
            entity_column: np.tile(entity_ids, len(scenario_ids)),
        },
    )
    _append_scenario_columns_to_frame(
        frame,
        scenario_ids,
        len(entity_ids),
        predictions.scenario_columns,
    )
    for channel_name, (lower, upper) in intervals.items():
        frame[f"{channel_name}_pi_lower"] = lower.reshape(-1)
        frame[f"{channel_name}_pi_upper"] = upper.reshape(-1)
    return frame


def _build_entity_output_frame(
    predictions: ChannelSet,
    ground_truth: ChannelSet | None,
    intervals: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    scenario_ids = predictions.scenario_ids
    entity_ids = predictions.entity_ids
    entity_column = predictions.entity_name
    frame = pd.DataFrame(
        {
            "scenario": np.repeat(scenario_ids, len(entity_ids)),
            entity_column: np.tile(entity_ids, len(scenario_ids)),
        },
    )
    _append_scenario_columns_to_frame(
        frame,
        scenario_ids,
        len(entity_ids),
        predictions.scenario_columns,
    )
    for column_name, values in predictions.entity_columns.items():
        frame[column_name] = np.tile(values, len(scenario_ids))
    for channel_name, prediction_values in predictions.channels.items():
        frame[f"{channel_name}_pred"] = prediction_values.reshape(-1)
        if ground_truth is not None:
            frame[f"{channel_name}_target"] = ground_truth.channels[
                channel_name
            ].reshape(-1)
        lower, upper = intervals[channel_name]
        frame[f"{channel_name}_pi_lower"] = lower.reshape(-1)
        frame[f"{channel_name}_pi_upper"] = upper.reshape(-1)
    return frame


def _append_scenario_columns_to_frame(
    frame: pd.DataFrame,
    scenario_ids: np.ndarray,
    entity_count: int,
    scenario_columns: dict[str, np.ndarray],
) -> None:
    for column_name, values in scenario_columns.items():
        if len(values) != len(scenario_ids):
            raise ValueError(
                f"Scenario column {column_name!r} does not align with the scenario axis."
            )
        repeated_values = np.repeat(values, entity_count)
        frame[column_name] = repeated_values
        if f"{column_name}_label" in scenario_columns:
            continue
        label_column = _scenario_label_column(column_name, repeated_values)
        if label_column is not None:
            frame[f"{column_name}_label"] = label_column


def _resolve_target_kind(channel_names: tuple[str, ...]) -> str:
    if tuple(channel_names) == ("line_loading",):
        return "line"
    return "volt"


def _build_calibration_sets(
    predictions,
    ground_truth,
    channel_names: tuple[str, ...],
    *,
    strat: str | None = None,
    system_load_n_bins: int | None = None,
    system_load_bin_edges: np.ndarray | None = None,
    pred_n_bins: int | None = None,
    pred_bin_edges: dict[str, np.ndarray] | None = None,
) -> tuple[ChannelSet, ChannelSet]:
    calibration_predictions, calibration_ground_truth = _build_target_channel_sets(
        predictions,
        ground_truth,
        channel_names,
    )
    scenario_columns = _build_scenario_columns(
        calibration_predictions,
        ground_truth,
        strat,
        system_load_n_bins=system_load_n_bins,
        system_load_bin_edges=system_load_bin_edges,
        pred_n_bins=pred_n_bins,
        pred_bin_edges=pred_bin_edges,
    )
    return (
        _with_scenario_columns(calibration_predictions, scenario_columns),
        _with_scenario_columns(calibration_ground_truth, scenario_columns),
    )


def _build_target_channel_sets(
    predictions,
    ground_truth,
    channel_names: tuple[str, ...],
) -> tuple[ChannelSet, ChannelSet]:
    if _resolve_target_kind(channel_names) == "line":
        return _build_line_loading_sets(
            predictions,
            ground_truth,
            scenario_columns=None,
        )
    return (
        _build_bus_channel_set(
            predictions,
            channel_names,
            scenario_columns=None,
        ),
        _build_ground_truth_bus_channel_set(
            ground_truth,
            channel_names,
            scenario_columns=None,
        ),
    )


def _build_bus_channel_set(
    predictions,
    channel_names: tuple[str, ...],
    *,
    scenario_columns: dict[str, np.ndarray] | None = None,
) -> ChannelSet:
    return ChannelSet(
        scenario_ids=predictions.scenario_ids.copy(),
        entity_ids=predictions.bus_ids.copy(),
        entity_name="bus",
        channels={
            channel_name: _prediction_channel(predictions, channel_name).copy()
            for channel_name in channel_names
        },
        valid_masks=_unknown_masks(predictions, channel_names),
        entity_columns={
            "bus_type": _bus_type_labels_array(
                predictions.bus_ids, predictions.bus_type_masks
            ),
        },
        scenario_columns={
            name: values.copy() for name, values in (scenario_columns or {}).items()
        },
        scenario_embeddings=predictions.scenario_embeddings,
    )


def _unknown_masks(
    predictions: PredictionSet, channel_names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Per variable, the entries that are predicted rather than known set-points.

    Known entries (e.g. `Vm` at PV buses) equal their set-point, so their
    residual is zero. Fitting on them would shrink the bounds of a bus in the
    scenarios where it is not PV (e.g. its generator is out); as invalid
    entries they get no weight when fitting and keep their value when applied.
    """

    masks: dict[str, np.ndarray] = {}
    for variable, bus_type in _SET_POINTS.get(predictions.task_name, ()):
        if variable in channel_names and bus_type in predictions.bus_type_masks:
            masks[variable] = ~predictions.bus_type_masks[bus_type]
    return masks


def _build_ground_truth_bus_channel_set(
    ground_truth,
    channel_names: tuple[str, ...],
    *,
    scenario_columns: dict[str, np.ndarray] | None = None,
) -> ChannelSet:
    return ChannelSet(
        scenario_ids=ground_truth.scenario_ids.copy(),
        entity_ids=ground_truth.bus_ids.copy(),
        entity_name="bus",
        channels={
            channel_name: _ground_truth_channel(ground_truth, channel_name).copy()
            for channel_name in channel_names
        },
        entity_columns={
            "bus_type": _bus_type_labels_array(
                ground_truth.bus_ids, ground_truth.bus_type_masks
            ),
        },
        scenario_columns={
            name: values.copy() for name, values in (scenario_columns or {}).items()
        },
    )


def _build_line_loading_sets(
    predictions,
    ground_truth,
    *,
    scenario_columns: dict[str, np.ndarray] | None = None,
) -> tuple[ChannelSet, ChannelSet]:
    predicted_loading = compute_line_loading(
        predictions.Vm, predictions.Va, ground_truth
    )
    target_loading = compute_line_loading(
        ground_truth.Vm, ground_truth.Va, ground_truth
    )
    valid_mask = (predicted_loading >= 0.0) & (target_loading >= 0.0)
    entity_columns = {
        "from_bus": ground_truth.f_bus.astype(np.int64, copy=True),
        "to_bus": ground_truth.t_bus.astype(np.int64, copy=True),
        "rate_a": ground_truth.rate_a.astype(np.float32, copy=True),
    }
    return (
        ChannelSet(
            scenario_ids=predictions.scenario_ids.copy(),
            entity_ids=ground_truth.branch_ids.copy(),
            entity_name="branch",
            channels={"line_loading": predicted_loading},
            valid_masks={"line_loading": valid_mask},
            entity_columns=entity_columns,
            scenario_columns={
                name: values.copy() for name, values in (scenario_columns or {}).items()
            },
            scenario_embeddings=predictions.scenario_embeddings,
        ),
        ChannelSet(
            scenario_ids=ground_truth.scenario_ids.copy(),
            entity_ids=ground_truth.branch_ids.copy(),
            entity_name="branch",
            channels={"line_loading": target_loading},
            valid_masks={"line_loading": valid_mask},
            entity_columns=entity_columns,
            scenario_columns={
                name: values.copy() for name, values in (scenario_columns or {}).items()
            },
            scenario_embeddings=predictions.scenario_embeddings,
        ),
    )


def _build_scenario_columns(
    calibration_predictions: ChannelSet,
    ground_truth,
    strat: str | None,
    *,
    system_load_n_bins: int | None = None,
    system_load_bin_edges: np.ndarray | None = None,
    pred_n_bins: int | None = None,
    pred_bin_edges: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    if strat is None:
        return {}
    if strat == "n_outages":
        return {"n_outages": ground_truth.n_outages.astype(np.int32, copy=True)}
    if strat == "outage_type":
        return {"outage_type": ground_truth.outage_type.astype(np.int32, copy=True)}
    if strat == "system_load_bin":
        edges = system_load_bin_edges
        if edges is None:
            if system_load_n_bins is None:
                raise ValueError(
                    "system_load_n_bins is required to build system_load_bin strata."
                )
            edges = _compute_system_load_bin_edges(
                ground_truth.system_load,
                int(system_load_n_bins),
            )
        load_bins = _assign_system_load_bins(ground_truth.system_load, edges)
        return {
            "system_load_bin": load_bins,
            "system_load_bin_label": _build_system_load_bin_labels(
                load_bins,
                edges,
            ),
        }
    if strat == "pred_bins":
        return {}
    raise ValueError(f"Unsupported stratification: {strat!r}")


def _with_scenario_columns(
    channel_set: ChannelSet,
    scenario_columns: dict[str, np.ndarray],
) -> ChannelSet:
    return ChannelSet(
        scenario_ids=channel_set.scenario_ids.copy(),
        entity_ids=channel_set.entity_ids.copy(),
        entity_name=channel_set.entity_name,
        channels={name: values.copy() for name, values in channel_set.channels.items()},
        valid_masks={
            name: values.copy() for name, values in channel_set.valid_masks.items()
        },
        entity_columns={
            name: values.copy() for name, values in channel_set.entity_columns.items()
        },
        scenario_columns={
            name: values.copy() for name, values in scenario_columns.items()
        },
        scenario_embeddings=channel_set.scenario_embeddings,
    )


def _build_pred_bin_frame(
    predictions: ChannelSet,
    pred_bin_edges: dict[str, np.ndarray],
) -> pd.DataFrame:
    scenario_ids = predictions.scenario_ids
    entity_ids = predictions.entity_ids
    entity_column = predictions.entity_name
    frame = pd.DataFrame(
        {
            "scenario": np.repeat(scenario_ids, len(entity_ids)),
            entity_column: np.tile(entity_ids, len(scenario_ids)),
        },
    )
    for channel_name, prediction_values in predictions.channels.items():
        if channel_name not in pred_bin_edges:
            continue
        column_name = _pred_bin_column_name(channel_name)
        row_bins = _assign_prediction_bins_to_rows(
            prediction_values,
            pred_bin_edges[channel_name],
            valid_mask=predictions.valid_masks.get(channel_name),
        )
        frame[column_name] = row_bins.reshape(-1)
        frame[f"{column_name}_label"] = _build_prediction_bin_label_array(
            row_bins.reshape(-1),
            pred_bin_edges[channel_name],
        )
    return frame


def _assign_prediction_bins_to_rows(
    prediction_values: np.ndarray,
    pred_bin_edges: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(prediction_values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if valid_mask is not None:
        finite_mask &= valid_mask
    bins = np.full(values.shape, np.nan, dtype=np.float32)
    if np.any(finite_mask):
        bins[finite_mask] = np.searchsorted(
            np.asarray(pred_bin_edges, dtype=np.float32),
            values[finite_mask],
            side="right",
        ).astype(np.float32, copy=False)
    return bins


def _build_prediction_bin_label_array(
    flattened_bin_values: np.ndarray,
    pred_bin_edges: np.ndarray,
) -> pd.Categorical:
    total_bins = len(np.asarray(pred_bin_edges, dtype=np.float32)) + 1
    return _categorical_labels(
        flattened_bin_values,
        lambda value: _format_system_load_bin_value(
            value, total_bins=total_bins, edges=pred_bin_edges
        ),
    )


def _compute_system_load_bin_edges(
    system_load: np.ndarray,
    n_bins: int,
    *,
    stratification: str = "system_load_bin",
) -> np.ndarray:
    """Interior percentile edges of `n_bins` bins (also used for `pred_bins`)."""

    finite_load = np.asarray(system_load, dtype=np.float32)
    finite_load = finite_load[np.isfinite(finite_load)]
    if finite_load.size == 0:
        raise ValueError(
            "Unable to derive system_load_bin strata because bus demand column 'Pd' is missing or invalid."
            if stratification == "system_load_bin"
            else f"Unable to derive {stratification} strata: no finite values."
        )
    quantiles = np.linspace(0.0, 1.0, num=int(n_bins) + 1)
    edges = np.quantile(finite_load, quantiles).astype(np.float32, copy=False)
    interior_edges = np.unique(edges[1:-1])
    assigned = _assign_system_load_bins(finite_load, interior_edges)
    if np.unique(assigned).size < 2:
        raise ValueError(
            f"{stratification} stratification requires at least two distinct percentile bins."
        )
    return interior_edges.astype(np.float32, copy=True)


def _assign_system_load_bins(
    system_load: np.ndarray,
    system_load_bin_edges: np.ndarray,
) -> np.ndarray:
    values = np.asarray(system_load, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "system_load_bin stratification requires finite per-scenario system load values."
        )
    edges = np.asarray(system_load_bin_edges, dtype=np.float32)
    return np.searchsorted(edges, values, side="right").astype(np.int32, copy=False)


def _build_system_load_bin_labels(
    system_load_bins: np.ndarray,
    system_load_bin_edges: np.ndarray,
) -> np.ndarray:
    total_bins = len(np.asarray(system_load_bin_edges, dtype=np.float32)) + 1
    return np.asarray(
        [
            _format_system_load_bin_value(
                int(value),
                total_bins=total_bins,
                edges=system_load_bin_edges,
            )
            for value in system_load_bins.tolist()
        ],
        dtype=object,
    )


def _merge_pooled_columns(
    frame: pd.DataFrame,
    calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator,
    application: ChannelSet,
) -> pd.DataFrame:
    """`<variable>_pooled`: the bound used the quantile of all strata (sparse_strata: pooled)."""

    masks = calibrator.pooled_masks
    if not masks:
        return frame
    entity = application.entity_name
    n_scenarios, n_entities = len(application.scenario_ids), len(application.entity_ids)
    pooled_frame = pd.DataFrame(
        {
            "scenario": np.repeat(
                np.asarray(application.scenario_ids, dtype=np.int64), n_entities
            ),
            entity: np.tile(
                np.asarray(application.entity_ids, dtype=np.int64), n_scenarios
            ),
            **{f"{name}_pooled": mask.reshape(-1) for name, mask in masks.items()},
        }
    )
    return frame.merge(pooled_frame, on=["scenario", entity], how="left")


def _merge_kernel_diagnostics(
    frame: pd.DataFrame,
    calibrator,
    scenario_ids: np.ndarray,
) -> pd.DataFrame:
    """Attach per-scenario KCP diagnostics so localization can be audited.

    `n_eff` near the calibration-set size means the kernel is not localizing;
    `kernel_fallback` marks scenarios that reverted to the stratum-wide
    (SCP) quantile.
    """

    if not isinstance(calibrator, KernelIntervalCalibrator):
        return frame
    diagnostics = calibrator.diagnostics
    if not diagnostics:
        return frame
    diagnostic_frame = pd.DataFrame(
        {
            "scenario": np.asarray(scenario_ids, dtype=np.int64),
            "kcp_n_eff": diagnostics["n_eff"],
            "kcp_bandwidth_sq": diagnostics["kernel_bandwidth_sq"],
            "kcp_fallback": diagnostics["kernel_fallback"],
        }
    )
    return frame.merge(diagnostic_frame, on="scenario", how="left")


def _attach_scenario_embeddings(
    predictions,
    embeddings_path: str,
    *,
    pooling: str,
):
    """Load, pool, and align scenario embeddings onto a prediction set."""

    embedding_set = load_scenario_embeddings(
        embeddings_path,
        pooling=pooling,
        scenario_ids=predictions.scenario_ids,
    )
    predictions.scenario_embeddings = embedding_set.align_to(predictions.scenario_ids)
    return predictions


def _default_output_dir(task_name: str, cp_type: str) -> Path:
    output_dir = (
        Path.cwd()
        / "trustlayer_outputs"
        / _normalize_output_segment(task_name)
        / _normalize_output_segment(cp_type)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _normalize_output_segment(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-" for character in value.strip()
    )
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or "unknown"


def _known_mask(frame: pd.DataFrame, variable: str) -> np.ndarray:
    """Rows whose value of `variable` is a known set-point (see `_SET_POINTS`)."""

    return _flag(frame, f"{variable}_known")


def _flag(frame: pd.DataFrame, column: str) -> np.ndarray:
    """A boolean column of the frame, or all False when it is absent."""

    if column not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    return frame[column].to_numpy(dtype=bool)


def _stratum_label(work: pd.DataFrame, stratification: str, value: object) -> str:
    stratum_value = None if pd.isna(value) else int(cast(float, value))
    label_column = f"{stratification}_label"
    if stratum_value is not None and label_column in work.columns:
        labels = work.loc[work[stratification] == value, label_column].dropna()
        if not labels.empty:
            return f"{stratification}={labels.iloc[0]}"
    return _format_stratum_label(stratification, stratum_value)


def _analysis_scenario_metadata_columns_from_frame(
    frame: pd.DataFrame,
    *,
    variable: str | None = None,
) -> list[str]:
    columns: list[str] = []
    for column in _ordered_analysis_strat_columns(frame.columns):
        if variable is not None and not _stratification_applies_to_variable(
            column, variable
        ):
            continue
        if column not in frame.columns:
            continue
        columns.append(column)
        label_column = f"{column}_label"
        if label_column in frame.columns:
            columns.append(label_column)
    return columns


def _ordered_analysis_strat_columns(column_names) -> list[str]:
    available = set(column_names)
    columns = [column for column in BASE_ANALYSIS_STRAT_COLUMNS if column in available]
    columns.extend(
        sorted(
            column
            for column in available
            if _is_analysis_strat_column(column)
            and column not in BASE_ANALYSIS_STRAT_COLUMNS
        )
    )
    return columns


def _is_analysis_strat_column(column_name: str) -> bool:
    return column_name in BASE_ANALYSIS_STRAT_COLUMNS or (
        column_name.startswith("pred_bins_") and not column_name.endswith("_label")
    )


def _stratification_applies_to_variable(stratification: str, variable: str) -> bool:
    if stratification.startswith("pred_bins_"):
        return stratification == _pred_bin_column_name(variable)
    return True


def _resolve_bus_type_labels_from_masks(
    bus_ids: np.ndarray,
    bus_type_masks: dict[str, np.ndarray],
) -> dict[int, str]:
    labels: dict[int, str] = {}
    for index, bus_id in enumerate(bus_ids.tolist()):
        labels[int(bus_id)] = _bus_type_label_from_values(
            ref=bus_type_masks.get("REF"),
            pv=bus_type_masks.get("PV"),
            pq=bus_type_masks.get("PQ"),
            column_index=index,
        )
    return labels


def _bus_type_labels_array(
    bus_ids: np.ndarray,
    bus_type_masks: dict[str, np.ndarray],
) -> np.ndarray:
    labels = _resolve_bus_type_labels_from_masks(bus_ids, bus_type_masks)
    return np.asarray(
        [labels.get(int(bus_id), "Unknown") for bus_id in bus_ids.tolist()],
        dtype=object,
    )


def _bus_type_label_from_values(
    *,
    ref: np.ndarray | None,
    pv: np.ndarray | None,
    pq: np.ndarray | None,
    column_index: int,
) -> str:
    return _bus_type_label_from_flags(
        ref=0.0 if ref is None else float(np.any(ref[:, column_index])),
        pv=0.0 if pv is None else float(np.any(pv[:, column_index])),
        pq=0.0 if pq is None else float(np.any(pq[:, column_index])),
    )


def _bus_type_label_from_flags(*, ref: float, pv: float, pq: float) -> str:
    if ref >= 0.5:
        return "REF"
    if pv >= 0.5:
        return "PV"
    if pq >= 0.5:
        return "PQ"
    return "Unknown"


def _format_stratum_label(stratification: str, stratum_value: object) -> str:
    if pd.isna(stratum_value):
        return "all"
    if isinstance(stratum_value, (int, float, np.integer, np.floating)):
        normalized_value = int(float(stratum_value))
        if stratification == "outage_type":
            return f"{stratification}={_format_outage_type_value(normalized_value)}"
        if stratification == "system_load_bin":
            return f"{stratification}={_format_system_load_bin_value(normalized_value)}"
        if stratification.startswith("pred_bins_"):
            return f"{stratification}={_format_system_load_bin_value(normalized_value)}"
        return f"{stratification}={normalized_value}"
    return f"{stratification}={stratum_value!s}"


def _format_outage_type_value(value: int) -> str:
    labels = {
        0: "none",
        1: "branch_only",
        2: "gen_only",
        3: "mixed",
    }
    return labels.get(int(value), str(int(value)))


def _format_system_load_bin_value(
    value: int,
    *,
    total_bins: int | None = None,
    edges: np.ndarray | None = None,
) -> str:
    if total_bins is None or edges is None:
        return f"bin_{int(value)}"
    lower_pct = round((100.0 * int(value)) / total_bins)
    upper_pct = round((100.0 * (int(value) + 1)) / total_bins)
    edges_array = np.asarray(edges, dtype=np.float32)
    label = f"p{lower_pct:02d}-p{upper_pct:02d}"
    if edges_array.size == 0:
        return label
    if int(value) == 0:
        return f"{label} (<{edges_array[0]:g})"
    if int(value) == total_bins - 1:
        return f"{label} (>={edges_array[-1]:g})"
    return f"{label} [{edges_array[int(value) - 1]:g}, {edges_array[int(value)]:g})"


def _scenario_label_column(
    column_name: str,
    repeated_values: np.ndarray,
) -> pd.Categorical | None:
    if column_name.endswith("_label"):
        return None
    if column_name == "outage_type":
        return _categorical_labels(repeated_values, _format_outage_type_value)
    if column_name == "system_load_bin" or column_name.startswith("pred_bins_"):
        return _categorical_labels(repeated_values, _format_system_load_bin_value)
    return None


def _categorical_labels(
    values: np.ndarray,
    format_value: Callable[[int], str],
) -> pd.Categorical:
    """Readable labels for integer codes, missing where the code is NaN.

    Built as a categorical, with one string per distinct code rather than one
    per row: label columns span every row of a block, and millions of small
    strings fragment memory that the process then cannot give back.
    """

    numeric = np.asarray(values, dtype=float)
    present = ~np.isnan(numeric)
    distinct = np.unique(numeric[present]).astype(np.int64)
    codes = np.full(numeric.shape[0], -1, dtype=np.int64)
    codes[present] = np.searchsorted(distinct, numeric[present].astype(np.int64))
    return pd.Categorical.from_codes(
        codes, categories=[format_value(int(value)) for value in distinct]
    )


def _pred_bin_column_name(channel_name: str) -> str:
    return f"pred_bins_{channel_name.lower()}"


def _ground_truth_channel(ground_truth, channel_name: str) -> np.ndarray:
    if channel_name == "Vm":
        return ground_truth.Vm
    if channel_name == "Va":
        return ground_truth.Va
    return ground_truth.channels[channel_name]


def _prediction_channel(predictions, channel_name: str) -> np.ndarray:
    if channel_name == "Vm":
        return predictions.Vm
    if channel_name == "Va":
        return predictions.Va
    return predictions.channels[channel_name]


def _n_jobs(args: argparse.Namespace, config: dict[str, object]) -> int | None:
    """`--n-jobs`, else `n_jobs` of the config; None when neither is set."""

    value = coalesce_option(getattr(args, "n_jobs", None), config, "n_jobs")
    return None if value is None else int(cast(int, value))


def _block_jobs(args: argparse.Namespace, config: dict[str, object]) -> int:
    """Processes for the blocks of scenarios: one unless `n_jobs` is set.

    Reading and writing the blocks is mostly parquet decoding, which Arrow
    already spreads over threads; on IEEE-118 test sets of 5,000 and 20,000
    scenarios, parallel blocks were slower and took more memory, so they are
    opt-in. The cross-validated bandwidth choice, which is pure computation,
    runs in parallel by default.
    """

    n_jobs = _n_jobs(args, config)
    return 1 if n_jobs is None else n_jobs


def _write_block(
    path: str, frame: pd.DataFrame, index: int, schemas: dict[str, Any], key: str
) -> None:
    """Add one block to a partitioned output, with the column types of the first."""

    writer = PartitionedWriter(path, clear=False, schema=schemas.get(key))
    writer.write(frame, block_index=index)
    if schemas.get(key) is None:
        schemas[key] = writer.schema


def _share_threads(
    calibrators: Mapping[str, NaiveIntervalCalibrator | KernelIntervalCalibrator],
    threads: int | None,
) -> None:
    """Give kernel calibrators their process's share of the CPUs (None: all)."""

    for calibrator in calibrators.values():
        if isinstance(calibrator, KernelIntervalCalibrator):
            calibrator.n_jobs = threads
