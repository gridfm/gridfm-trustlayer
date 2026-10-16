"""`apply`: bounds and statuses for new scenarios, without their AC solution."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.cli import common, reports, status
from gridfm_trustlayer.cli.common import (
    _SCENARIOS_FILE_TEXT,
    _SET_POINTS,
    _STATUS_FILE_TEXT,
    SCREEN_BOUND_TOLERANCE,
    _add_known_columns,
    _analysis_scenario_metadata_columns_from_frame,
    _artifact_path,
    _block_jobs,
    _bounds_notes,
    _build_bus_channel_set,
    _build_entity_output_frame,
    _build_interval_frame,
    _build_pred_bin_frame,
    _build_scenario_columns,
    _check_artifact_matches,
    _common_parent,
    _default_output_dir,
    _embedding_spec,
    _EmbeddingSource,
    _ground_truth_reference,
    _known_mask,
    _merge_kernel_diagnostics,
    _merge_pooled_columns,
    _ordered_analysis_strat_columns,
    _output_task_name,
    _parse_screen_bounds,
    _report_progress,
    _require_set_points,
    _scenario_strata,
    _share_threads,
    _stratification_applies_to_variable,
    _stratum_label,
    _write_block,
    _write_status_tables,
)
from gridfm_trustlayer.cli.config import (
    coalesce_option,
    load_cli_config,
    resolve_command_config,
    validate_command_config,
)
from gridfm_trustlayer.conformal import ChannelSet, NaiveIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import KernelIntervalCalibrator
from gridfm_trustlayer.io import (
    GroundTruthSet,
    PredictionSet,
    compute_line_loading,
    load_ground_truth,
    load_predictions,
)
from gridfm_trustlayer.io.artifact import load_interval_artifacts
from gridfm_trustlayer.io.ground_truth import (
    GroundTruthReference,
)
from gridfm_trustlayer.io.parquet import (
    clear_partitioned_output,
    column_names,
    is_trustlayer_output,
    read_frame,
    read_scenario_ids,
    rows_per_scenario,
    scenario_blocks,
)
from gridfm_trustlayer.parallel import block_rows, map_blocks

# Stratifications computed from the description of each scenario: its outages
# (branch and generator status) or its total load.
_DESCRIPTION_STRATA = {"n_outages", "outage_type", "system_load_bin"}


def _run_apply(args: argparse.Namespace) -> None:
    """Bounds for new scenarios, which need no known AC solution.

    Same computation as `calibrate`, without the coverage analysis, and with
    `screen_bounds` the statuses of `screen`, without the truth: only the
    predictions are needed, plus the description of each scenario (topology,
    branch parameters, loads) when the layer is stratified by outages or load
    or calibrates line loading, and the embeddings for `cp_type: kernel`.
    """

    config = resolve_command_config(load_cli_config(args.config), "apply")
    validate_command_config("apply", config)
    predictions_path = str(
        coalesce_option(args.predictions, config, "predictions", required=True)
    )
    artifact_path = _artifact_path(
        args, config, explicit=args.artifact, for_reading=True
    )
    calibrators = load_interval_artifacts(artifact_path)
    _check_artifact_matches(calibrators, args, config, artifact_path, command="apply")
    task_name = coalesce_option(args.task_name, config, "task_name")
    generator_path = coalesce_option(
        args.generator_predictions, config, "generator_predictions"
    )
    scenario_ids = read_scenario_ids(predictions_path)
    if scenario_ids.size == 0:
        raise ValueError(f"{predictions_path} contains no scenarios.")
    task = load_predictions(
        predictions_path,
        task_name=task_name,
        generator_parquet_path=generator_path,
        scenario_ids=scenario_ids[:1],
    ).task_name
    _require_set_point_columns(predictions_path, task)

    reasons = sorted(
        {
            f"strat: {calibrator.strat}"
            for calibrator in calibrators.values()
            if calibrator.strat in _DESCRIPTION_STRATA
        }
        | {
            "line_loading"
            for calibrator in calibrators.values()
            if calibrator.entity_name == "branch"
        }
    )
    description_paths = None
    reference = None
    if reasons:
        pf_node = coalesce_option(args.pf_node, config, "pf_node")
        pf_edge = coalesce_option(args.pf_edge, config, "pf_edge")
        if pf_node is None or pf_edge is None:
            raise ValueError(
                f"This layer ({', '.join(reasons)}) needs the description of the new "
                "scenarios: set inputs.apply.pf_node and inputs.apply.pf_edge (or "
                "--pf-node / --pf-edge) to their bus and branch files. Their topology, "
                "branch parameters and loads are read; no AC solution is needed.",
            )
        description_paths = (str(pf_node), str(pf_edge))
        reference = _ground_truth_reference(calibrators.values(), description_paths)
    has_bus_targets = any(
        calibrator.entity_name == "bus" for calibrator in calibrators.values()
    )
    output_paths = {
        target_kind: _resolve_apply_output_path(
            args,
            config,
            predictions_path,
            calibrator,
            target_kind=target_kind,
            multiple_targets=len(calibrators) > 1,
        )
        for target_kind, calibrator in calibrators.items()
    }
    embeddings = _embedding_spec(calibrators, args, config)
    inputs = (
        predictions_path,
        *(description_paths or ()),
        artifact_path,
        generator_path,
        None if embeddings is None else embeddings[0],
    )
    for path in output_paths.values():
        # Both families go into a folder; at that path, a previous run with one
        # family left a dataset, replaced here.
        if len(output_paths) > 1 and is_trustlayer_output(Path(path).parent):
            clear_partitioned_output(Path(path).parent, inputs=inputs)
        clear_partitioned_output(path, inputs=inputs)
    # With screen_bounds, every value and scenario also gets its status.
    variables = sorted(
        (name for c in calibrators.values() for name in c.channel_names),
        key=reports.variable_rank,
    )
    screen_bounds = {
        variable: bounds
        for variable, bounds in _parse_screen_bounds(
            config.get("screen_bounds")
        ).items()
        if variable in variables
    }
    n_jobs = _block_jobs(args, config)
    rows = rows_per_scenario(predictions_path, len(scenario_ids))
    blocks = scenario_blocks(
        scenario_ids,
        rows,
        block_rows(common._BLOCK_ROWS, rows * len(scenario_ids), n_jobs),
    )
    job = _ApplyJob(
        predictions_path=predictions_path,
        task_name=task_name,
        generator_path=generator_path,
        description_paths=description_paths,
        reference=reference,
        artifact_path=str(artifact_path),
        output_paths={kind: str(path) for kind, path in output_paths.items()},
        has_bus_targets=has_bus_targets,
        embeddings=embeddings,
        scenario_ids=scenario_ids,
        screen_bounds=screen_bounds,
        calibrators=calibrators,
    )
    results = map_blocks(
        job,
        blocks,
        n_jobs=n_jobs,
        progress=partial(_report_progress, "apply", blocks=blocks),
    )
    status_counts = [part for result in results for part in result[0]]
    scenario_parts = [part for result in results for part in result[1]]
    data_dir = _common_parent(output_paths.values())
    print(
        f"apply: bounds for {', '.join(variables)} in {len(scenario_ids):,} scenarios; "
        f"outputs in "
        f"{next(iter(output_paths.values())) if len(output_paths) == 1 else data_dir}",
        flush=True,
    )
    notes = _bounds_notes(output_paths.values())
    for note in notes:
        print(f"apply: {note}", flush=True)
    if scenario_parts:
        status_dir = Path(str(config.get("status_dir") or Path(data_dir) / "status"))
        status_dir.mkdir(parents=True, exist_ok=True)
        reports.clear_report(status_dir)
        table, counts = _write_status_tables(
            scenario_parts, status_counts, status_dir, truth=False
        )
        limits = _limits_text(screen_bounds)
        status.write_apply_summary(
            status_dir / "summary.md",
            counts,
            table,
            files=[
                ("scenarios.csv", _SCENARIOS_FILE_TEXT),
                ("status_by_stratum.csv", _STATUS_FILE_TEXT),
                *(
                    (
                        os.path.relpath(path, status_dir),
                        _APPLIED_DATA_TEXT,
                    )
                    for path in output_paths.values()
                ),
            ],
            context=[
                f"- Limits: {limits}.",
                f"- New scenarios: `{predictions_path}` ({len(scenario_ids):,} scenarios).",
                f"- Artifact: `{artifact_path}`.",
                "- Known values (set-points) and out-of-service lines are not screened.",
                *(f"- {note[0].upper()}{note[1:]}." for note in notes),
            ],
        )
        print(status.terminal_line(counts, command="apply", truth=False), flush=True)
        print(f"apply: status report in {status_dir}", flush=True)


@dataclass
class _ApplyJob:
    """`apply` on one block of new scenarios (see `parallel.BlockJob`).

    Writes the block's bounds (and statuses) and returns the block's status
    counts and per-scenario statuses.
    """

    predictions_path: str
    task_name: str | None
    generator_path: str | None
    description_paths: tuple[str, str] | None
    reference: GroundTruthReference | None
    artifact_path: str
    output_paths: dict[str, str]
    has_bus_targets: bool
    embeddings: tuple[str, str] | None
    scenario_ids: np.ndarray
    screen_bounds: dict[str, dict[str, float]]
    calibrators: (
        dict[str, NaiveIntervalCalibrator | KernelIntervalCalibrator] | None
    ) = None
    schemas: dict[str, Any] = field(default_factory=dict)
    _embeddings: _EmbeddingSource | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "calibrators": None, "_embeddings": None}

    def setup(self, threads: int | None) -> None:
        if self.calibrators is None:
            self.calibrators = load_interval_artifacts(self.artifact_path)
        _share_threads(self.calibrators, threads)
        if self.embeddings is not None:
            path, pooling = self.embeddings
            self._embeddings = _EmbeddingSource(
                path, pooling, self.scenario_ids, preload=threads is None
            )

    def run(
        self, index: int, block: np.ndarray
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        assert self.calibrators is not None
        predictions = load_predictions(
            self.predictions_path,
            task_name=self.task_name,
            generator_parquet_path=self.generator_path,
            scenario_ids=block,
        )
        _require_set_points(predictions)
        if self._embeddings is not None:
            self._embeddings.attach(predictions)
        description = None
        if self.description_paths is not None:
            description = load_ground_truth(
                *self.description_paths,
                scenario_ids=predictions.scenario_ids,
                reference=self.reference,
                solved=False,
            )
        original_frame = (
            read_frame(self.predictions_path, scenario_ids=block)
            if self.has_bus_targets
            else None
        )
        status_counts: list[pd.DataFrame] = []
        scenario_parts: list[pd.DataFrame] = []
        for kind, calibrator in self.calibrators.items():
            frame = _bounds_block_frame(
                calibrator, predictions, description, original_frame
            )
            screened = [v for v in calibrator.channel_names if v in self.screen_bounds]
            if screened:
                frame, values = _add_status_columns(frame, screened, self.screen_bounds)
                status_counts.append(
                    status.element_counts(
                        values,
                        _ordered_analysis_strat_columns(values.columns),
                        label=_stratum_label,
                        truth=False,
                        applies=_stratification_applies_to_variable,
                    )
                )
                scenario_parts.append(
                    status.scenario_rows(
                        values, _scenario_strata(values.columns), truth=False
                    )
                )
            _write_block(self.output_paths[kind], frame, index, self.schemas, kind)
        return status_counts, scenario_parts


def _limits_text(bounds_by_variable: Mapping[str, Mapping[str, float]]) -> str:
    return "; ".join(
        f"`{variable}` ≤ {bounds['upper']:g}"
        if math.isnan(bounds["lower"])
        else f"`{variable}` in [{bounds['lower']:g}, {bounds['upper']:g}]"
        for variable, bounds in bounds_by_variable.items()
    )


def _add_status_columns(
    frame: pd.DataFrame,
    variables: Sequence[str],
    bounds_by_variable: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`<variable>_status` and `<variable>_point_status` of new scenarios.

    Known values (set-points) and out-of-service lines are not screened: their
    statuses are empty. Returns the frame and the screened values, for counting.
    """

    values: list[pd.DataFrame] = []
    for variable in variables:
        bounds = bounds_by_variable[variable]
        pred = frame[f"{variable}_pred"].to_numpy(dtype=float)
        screened = ~_known_mask(frame, variable)
        if variable == "line_loading":
            # -1 marks an out-of-service or unrated line; a missing (NaN)
            # prediction is screened, and uncertain, as in `screen`.
            screened &= ~(pred < 0.0)
        lower_limit = float(bounds["lower"]) - SCREEN_BOUND_TOLERANCE
        interval, point = status.classify(
            pred[screened],
            frame[f"{variable}_pi_lower"].to_numpy(dtype=float)[screened],
            frame[f"{variable}_pi_upper"].to_numpy(dtype=float)[screened],
            float("nan") if math.isnan(lower_limit) else lower_limit,
            float(bounds["upper"]) + SCREEN_BOUND_TOLERANCE,
        )
        for column, statuses in (("status", interval), ("point_status", point)):
            codes = np.full(len(frame), -1, dtype=np.int8)
            codes[screened] = statuses.codes
            frame[f"{variable}_{column}"] = pd.Categorical.from_codes(
                codes, status.ELEMENT_STATUSES
            )
        strata = _analysis_scenario_metadata_columns_from_frame(
            frame, variable=variable
        )
        rows = frame.loc[screened, ["scenario", *strata]].copy()
        rows["variable"] = variable
        rows["status"] = interval
        rows["point_status"] = point
        values.append(rows)
    return frame, pd.concat(values, ignore_index=True)


def _require_set_point_columns(predictions_path: str, task: str) -> None:
    columns = {name.lower() for name in column_names(predictions_path)}
    missing = [
        f"{variable}_target"
        for variable, _ in _SET_POINTS.get(task, ())
        if f"{variable.lower()}_target" not in columns
    ]
    if missing:
        raise ValueError(
            f"{task} predictions need the known set-points of each scenario in "
            f"{', '.join(missing)}: "
            + ", ".join(f"{v} at {bus} buses" for v, bus in _SET_POINTS[task])
            + ". The model's predictions there are replaced by these inputs, as when "
            "the layer was fitted. Values at other buses may be missing.",
        )


def _resolve_apply_output_path(
    args: argparse.Namespace,
    config: dict[str, object],
    predictions_path: str,
    calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator,
    *,
    target_kind: str,
    multiple_targets: bool,
) -> str:
    """Where `apply` writes one target family.

    The default name ends in `_bounds.parquet`, not `calibrate`'s
    `_with_intervals.parquet`: new and test predictions often share a file
    name, and the two runs must not overwrite each other.
    """

    stem = Path(predictions_path).stem
    variables = "_".join(name.lower() for name in calibrator.channel_names)
    default_name = f"{stem}_{variables}_bounds.parquet"
    if args.output is not None:
        output = Path(str(args.output))
        if not multiple_targets:
            return str(output)
        if output.suffix.lower() == ".parquet" or output.is_file():
            raise ValueError(
                "Bounds for voltages and line loading are written separately, so "
                "--output must be a directory. Use outputs.applied_predictions_volt "
                "and outputs.applied_predictions_line for explicit file paths.",
            )
        return str(output / default_name)
    explicit = config.get(f"applied_predictions_{target_kind}")
    if explicit is not None:
        return str(explicit)
    shared = config.get("applied_predictions")
    if shared is not None:
        if not multiple_targets:
            return str(shared)
        if Path(str(shared)).suffix.lower() == ".parquet":
            raise ValueError(
                "Bounds for voltages and line loading are written separately. Set "
                "outputs.applied_predictions to a directory, or set "
                "outputs.applied_predictions_volt and outputs.applied_predictions_line.",
            )
        return str(Path(str(shared)) / default_name)
    task_name = _output_task_name(args, config)
    cp_type = (
        "kernel" if isinstance(calibrator, KernelIntervalCalibrator) else "element"
    )
    return str(_default_output_dir(task_name, cp_type) / default_name)


def _bounds_block_frame(
    calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator,
    predictions: PredictionSet,
    description: GroundTruthSet | None,
    original_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Rows of one block of new scenarios with their bounds, for one target family."""

    scenario_columns: dict[str, np.ndarray] = {}
    if description is not None:
        scenario_columns = _build_scenario_columns(
            _build_bus_channel_set(predictions, ("Vm",), scenario_columns=None),
            description,
            calibrator.strat,
            system_load_n_bins=calibrator.system_load_n_bins,
            system_load_bin_edges=calibrator.system_load_bin_edges,
            pred_n_bins=calibrator.pred_n_bins,
            pred_bin_edges=calibrator.pred_bin_edges,
        )
    application: ChannelSet
    if calibrator.entity_name == "branch":
        assert description is not None
        loading = compute_line_loading(predictions.Vm, predictions.Va, description)
        application = ChannelSet(
            scenario_ids=predictions.scenario_ids.copy(),
            entity_ids=description.branch_ids.copy(),
            entity_name="branch",
            channels={"line_loading": loading},
            # Out-of-service or unrated branches carry -1 and get no interval.
            valid_masks={"line_loading": loading >= 0.0},
            entity_columns={
                "from_bus": description.f_bus.astype(np.int64, copy=True),
                "to_bus": description.t_bus.astype(np.int64, copy=True),
                "rate_a": description.rate_a.astype(np.float32, copy=True),
            },
            scenario_columns=scenario_columns,
            scenario_embeddings=predictions.scenario_embeddings,
        )
        intervals = calibrator.predict_intervals(application)
        output_frame = _build_entity_output_frame(application, None, intervals)
    else:
        assert original_frame is not None
        application = _build_bus_channel_set(
            predictions, calibrator.channel_names, scenario_columns=scenario_columns
        )
        intervals = calibrator.predict_intervals(application)
        # For new scenarios the `*_target` columns hold real values only at the
        # set-points (Vm at PV buses, Va at the reference bus) and placeholders
        # elsewhere, which would read as true values; they are not written.
        targets = [c for c in original_frame.columns if c.lower().endswith("_target")]
        output_frame = original_frame.drop(columns=targets).merge(
            _build_interval_frame(application, intervals),
            on=["scenario", "bus"],
            how="left",
            validate="one_to_one",
        )
    if calibrator.strat == "pred_bins":
        entity = application.entity_name
        output_frame = output_frame.merge(
            _build_pred_bin_frame(application, calibrator.pred_bin_edges),
            on=["scenario", entity],
            how="left",
            validate="one_to_one",
        )
    return _merge_kernel_diagnostics(
        _merge_pooled_columns(
            _add_known_columns(
                output_frame, calibrator.channel_names, predictions.task_name
            ),
            calibrator,
            application,
        ),
        calibrator,
        application.scenario_ids,
    )


_APPLIED_DATA_TEXT = (
    "the bounds, with `<variable>_status` and `<variable>_point_status` for every "
    "screened value"
)
