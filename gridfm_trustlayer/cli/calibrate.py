"""`calibrate`: bounds and coverage report for the test set."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.cli import common, reports
from gridfm_trustlayer.cli.common import (
    _ELEMENT_NAMES,
    _FINGERPRINT_SCENARIOS,
    _SET_POINTS,
    _add_known_columns,
    _analysis_scenario_metadata_columns_from_frame,
    _artifact_path,
    _block_jobs,
    _bounds_notes,
    _build_bus_channel_set,
    _build_calibration_sets,
    _build_entity_output_frame,
    _build_interval_frame,
    _build_pred_bin_frame,
    _build_scenario_columns,
    _check_artifact_matches,
    _common_parent,
    _default_output_dir,
    _embedding_spec,
    _EmbeddingSource,
    _flag,
    _format_stratum_label,
    _ground_truth_paths,
    _ground_truth_reference,
    _known_mask,
    _merge_kernel_diagnostics,
    _merge_pooled_columns,
    _ordered_analysis_strat_columns,
    _output_task_name,
    _prediction_fingerprint,
    _report_progress,
    _require_set_points,
    _scan_bus_types,
    _share_threads,
    _sparse_strata_listing,
    _stratification_applies_to_variable,
    _stratum_label,
    _voltage_target_frame,
    _write_block,
)
from gridfm_trustlayer.cli.config import (
    coalesce_option,
    load_cli_config,
    resolve_command_config,
    validate_command_config,
    validate_cp_type,
    validate_strat,
)
from gridfm_trustlayer.conformal import NaiveIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import (
    KernelIntervalCalibrator,
)
from gridfm_trustlayer.io import (
    GroundTruthSet,
    PredictionSet,
    load_ground_truth,
    load_predictions,
)
from gridfm_trustlayer.io.artifact import (
    load_interval_artifacts,
)
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


def _run_calibrate(args: argparse.Namespace) -> None:
    config = resolve_command_config(load_cli_config(args.config), "calibrate")
    validate_command_config("calibrate", config)
    cp_type = validate_cp_type(coalesce_option(args.cp_type, config, "cp_type"))
    requested_strat = validate_strat(coalesce_option(args.strat, config, "strat"))
    save_figures_value = coalesce_option(args.save_figures, config, "save_figures")
    save_figures = True if save_figures_value is None else bool(save_figures_value)
    predictions_path = str(
        coalesce_option(args.predictions, config, "predictions", required=True)
    )
    artifact_path = _artifact_path(
        args, config, explicit=args.artifact, for_reading=True
    )
    calibrators = load_interval_artifacts(artifact_path)
    _check_artifact_matches(
        calibrators, args, config, artifact_path, command="calibrate"
    )
    if requested_strat is not None:
        for calibrator in calibrators.values():
            if calibrator.strat != requested_strat:
                raise ValueError(
                    "Requested stratification does not match the saved interval artifact."
                )
    task_name = coalesce_option(args.task_name, config, "task_name")
    generator_path = coalesce_option(
        args.generator_predictions, config, "generator_predictions"
    )
    scenario_ids = read_scenario_ids(predictions_path)
    if scenario_ids.size == 0:
        raise ValueError(f"{predictions_path} contains no scenarios.")
    sample = load_predictions(
        predictions_path,
        task_name=task_name,
        generator_parquet_path=generator_path,
        scenario_ids=scenario_ids[:_FINGERPRINT_SCENARIOS],
    )
    _refuse_calibration_data(calibrators, scenario_ids, sample)

    # The test set is processed in blocks of scenarios: each block's rows are
    # written to the partitioned outputs, and only per-entity sums are kept for
    # the reports, so memory does not grow with the size of the test set.
    columns = set(column_names(predictions_path))
    has_bus_targets = any(
        calibrator.entity_name == "bus" for calibrator in calibrators.values()
    )
    needs_ground_truth = any(
        calibrator.strat is not None
        or calibrator.entity_name == "branch"
        or any(f"{name}_target" not in columns for name in calibrator.channel_names)
        for calibrator in calibrators.values()
    )
    ground_truth_paths = None
    reference = None
    if needs_ground_truth:
        ground_truth_paths = _ground_truth_paths(args, config)
        reference = _ground_truth_reference(calibrators.values(), ground_truth_paths)
    bus_types = _scan_bus_types(predictions_path) if has_bus_targets else {}
    output_paths = {
        target_kind: _resolve_output_path(
            args,
            config,
            predictions_path,
            cp_type,
            calibrator.channel_names,
            target_kind=target_kind,
            multiple_targets=len(calibrators) > 1,
            task_name=_output_task_name(args, config),
        )
        for target_kind, calibrator in calibrators.items()
    }
    embeddings = _embedding_spec(calibrators, args, config)
    inputs = (
        predictions_path,
        *(ground_truth_paths or ()),
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
    n_jobs = _block_jobs(args, config)
    rows = rows_per_scenario(predictions_path, len(scenario_ids))
    blocks = scenario_blocks(
        scenario_ids,
        rows,
        block_rows(common._BLOCK_ROWS, rows * len(scenario_ids), n_jobs),
    )
    job = _CalibrateJob(
        predictions_path=predictions_path,
        task_name=task_name,
        generator_path=generator_path,
        ground_truth_paths=ground_truth_paths,
        reference=reference,
        artifact_path=str(artifact_path),
        output_paths={kind: str(path) for kind, path in output_paths.items()},
        has_bus_targets=has_bus_targets,
        embeddings=embeddings,
        scenario_ids=scenario_ids,
        calibrators=calibrators,
    )
    results = map_blocks(
        job,
        blocks,
        n_jobs=n_jobs,
        progress=partial(_report_progress, "calibrate", blocks=blocks),
    )
    partials = {kind: [result[kind] for result in results] for kind in calibrators}

    data_dir = Path(_common_parent(output_paths.values()))
    coverage_dir = Path(str(config.get("coverage_dir") or data_dir / "coverage"))
    reports.clear_report(coverage_dir)
    strata: list[pd.DataFrame] = []
    groups: list[pd.DataFrame] = []
    figure_tasks: list[reports.FigureTask] = []
    files: list[tuple[str, str]] = [
        ("coverage_by_stratum.csv", "coverage and width per variable and stratum"),
        ("coverage_by_group.csv", "the same per bus type (voltages) or line rating"),
    ]
    data_files: list[tuple[str, str]] = []
    for target_kind, calibrator in calibrators.items():
        entity_column = calibrator.entity_name
        elements = _combine_analysis_partials(
            partials[target_kind], entity_column=entity_column, bus_types=bus_types
        )
        by_stratum = reports.coverage_by_stratum(elements, float(calibrator.alpha))
        strata.append(by_stratum)
        groups.append(reports.coverage_by_group(elements, entity_column))
        element_file = f"coverage_by_element_{reports.family_name(entity_column)}.csv"
        reports.write_csv(
            reports.coverage_by_element(elements, entity_column),
            coverage_dir / element_file,
        )
        files.append(
            (
                element_file,
                f"coverage and width of every {_ELEMENT_NAMES[entity_column]}",
            )
        )
        data_files.append(
            (
                os.path.relpath(output_paths[target_kind], coverage_dir),
                "the test predictions with their bounds (partitioned parquet)",
            )
        )
        if save_figures:
            figure_tasks += reports.coverage_figure_tasks(
                by_stratum,
                elements,
                coverage_dir / "figures",
                entity_column=entity_column,
            )
    figures = reports.render_figures(figure_tasks)
    by_stratum = reports.sort_by_stratum(pd.concat(strata, ignore_index=True))
    files += data_files
    reports.write_csv(
        by_stratum.drop(columns=["stratification", "stratum_value"]),
        coverage_dir / "coverage_by_stratum.csv",
    )
    reports.write_csv(
        reports.sort_by_variable(pd.concat(groups, ignore_index=True)).drop(
            columns=["stratification", "stratum_value"], errors="ignore"
        ),
        coverage_dir / "coverage_by_group.csv",
    )
    first = next(iter(calibrators.values()))
    method = f"`cp_type: {cp_type}`" + (
        f", `strat: {first.strat}`" if first.strat else ""
    )
    reports.write_coverage_summary(
        coverage_dir / "summary.md",
        by_stratum,
        figures,
        files,
        context=[
            (
                f"- Method: {method}, `alpha: {first.alpha:g}` "
                f"(target coverage {1 - first.alpha:.0%})."
            ),
            f"- Test predictions: `{predictions_path}` ({len(scenario_ids):,} scenarios).",
            f"- Artifact: `{artifact_path}`.",
            *_sparse_strata_context(calibrators),
        ],
        known_where=_known_where(sample.task_name),
    )
    _report_coverage(by_stratum)
    for note in _bounds_notes(output_paths.values()):
        print(f"calibrate: {note}", flush=True)
    data_text = (
        next(iter(output_paths.values())) if len(output_paths) == 1 else data_dir
    )
    print(f"calibrate: data in {data_text}; reports in {coverage_dir}", flush=True)


@dataclass
class _CalibrateJob:
    """`calibrate` on one block of test scenarios (see `parallel.BlockJob`).

    Writes the block's rows with their bounds and returns, per target family,
    the block's per-element coverage sums.
    """

    predictions_path: str
    task_name: str | None
    generator_path: str | None
    ground_truth_paths: tuple[str, str] | None
    reference: GroundTruthReference | None
    artifact_path: str
    output_paths: dict[str, str]
    has_bus_targets: bool
    embeddings: tuple[str, str] | None
    scenario_ids: np.ndarray
    calibrators: (
        dict[str, NaiveIntervalCalibrator | KernelIntervalCalibrator] | None
    ) = None
    schemas: dict[str, Any] = field(default_factory=dict)
    _embeddings: _EmbeddingSource | None = None

    def __getstate__(self) -> dict[str, Any]:
        # Workers rebuild the heavy state in `setup`.
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

    def run(self, index: int, block: np.ndarray) -> dict[str, pd.DataFrame]:
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
        ground_truth = None
        if self.ground_truth_paths is not None:
            ground_truth = load_ground_truth(
                *self.ground_truth_paths,
                scenario_ids=predictions.scenario_ids,
                reference=self.reference,
            )
        original_frame = (
            read_frame(self.predictions_path, scenario_ids=block)
            if self.has_bus_targets
            else None
        )
        partials: dict[str, pd.DataFrame] = {}
        for kind, calibrator in self.calibrators.items():
            frame = _calibrated_block_frame(
                calibrator, predictions, ground_truth, original_frame
            )
            _write_block(self.output_paths[kind], frame, index, self.schemas, kind)
            partials[kind] = _analysis_partials(
                frame, calibrator.channel_names, entity_column=calibrator.entity_name
            )
        return partials


_BUS_TYPE_WORDS = {"PV": "PV buses", "REF": "the reference bus"}


def _known_where(task_name: str) -> dict[str, str]:
    """Where each variable has known values, in words, for the reports."""

    return {
        variable: _BUS_TYPE_WORDS[bus_type]
        for variable, bus_type in _SET_POINTS.get(task_name, ())
    }


def _report_coverage(by_stratum: pd.DataFrame) -> None:
    """Overall coverage and width per variable, and the weakest stratum."""

    for variable, rows in by_stratum.groupby("variable", sort=False):
        overall = rows[rows["stratification"].eq("")].iloc[0]
        unit, factor = reports.unit_of(str(variable))
        width = float(overall["mean_width"]) * factor
        width_text = (
            f"mean width {width:.4g} {unit}".rstrip()
            if np.isfinite(width)
            else "mean width infinite (some strata have too few calibration samples)"
        )
        line = (
            f"calibrate: {variable}: coverage "
            f"{overall['coverage']:.3f} "
            f"(target {overall['target_coverage']:.3f}), {width_text}"
        )
        strata = rows[rows["stratification"].ne("")]
        if not strata.empty:
            weakest = strata.loc[strata["coverage"].idxmin()]
            line += f"; lowest stratum {weakest['stratum']} {weakest['coverage']:.3f}"
        if int(overall["n_known"]):
            line += f"; {int(overall['n_known']):,} known values not counted"
        print(line, flush=True)


def _calibrated_block_frame(
    calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator,
    predictions: PredictionSet,
    ground_truth: GroundTruthSet | None,
    original_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Rows of one block of test scenarios with their bounds, for one target family.

    Bus targets keep every column of the predictions file and add the bounds;
    line loading gets one row per branch with prediction, target and bounds.
    """

    if calibrator.entity_name == "branch":
        assert ground_truth is not None
        application_predictions, application_ground_truth = _build_calibration_sets(
            predictions,
            ground_truth,
            calibrator.channel_names,
            strat=calibrator.strat,
            system_load_n_bins=calibrator.system_load_n_bins,
            system_load_bin_edges=calibrator.system_load_bin_edges,
            pred_n_bins=calibrator.pred_n_bins,
            pred_bin_edges=calibrator.pred_bin_edges,
        )
        intervals = calibrator.predict_intervals(application_predictions)
        output_frame = _build_entity_output_frame(
            application_predictions,
            application_ground_truth,
            intervals,
        )
        if calibrator.strat == "pred_bins":
            output_frame = output_frame.merge(
                _build_pred_bin_frame(
                    application_predictions,
                    calibrator.pred_bin_edges,
                ),
                on=["scenario", "branch"],
                how="left",
                validate="one_to_one",
            )
    else:
        assert original_frame is not None
        scenario_columns: dict[str, np.ndarray] = {}
        if ground_truth is not None:
            scenario_columns = _build_scenario_columns(
                _build_bus_channel_set(
                    predictions, calibrator.channel_names, scenario_columns=None
                ),
                ground_truth,
                calibrator.strat,
                system_load_n_bins=calibrator.system_load_n_bins,
                system_load_bin_edges=calibrator.system_load_bin_edges,
                pred_n_bins=calibrator.pred_n_bins,
                pred_bin_edges=calibrator.pred_bin_edges,
            )
        application_predictions = _build_bus_channel_set(
            predictions,
            calibrator.channel_names,
            scenario_columns=scenario_columns,
        )
        intervals = calibrator.predict_intervals(application_predictions)
        interval_frame = _build_interval_frame(application_predictions, intervals)
        if calibrator.strat == "pred_bins":
            interval_frame = interval_frame.merge(
                _build_pred_bin_frame(
                    application_predictions,
                    calibrator.pred_bin_edges,
                ),
                on=["scenario", "bus"],
                how="left",
                validate="one_to_one",
            )
        output_frame = original_frame.merge(
            interval_frame,
            on=["scenario", "bus"],
            how="left",
            validate="one_to_one",
        )
        missing_targets = [
            name
            for name in calibrator.channel_names
            if f"{name}_target" not in output_frame.columns
        ]
        if missing_targets:
            assert ground_truth is not None
            output_frame = output_frame.merge(
                _voltage_target_frame(ground_truth, tuple(missing_targets)),
                on=["scenario", "bus"],
                how="left",
                validate="one_to_one",
            )
    return _merge_kernel_diagnostics(
        _merge_pooled_columns(
            _add_known_columns(
                output_frame, calibrator.channel_names, predictions.task_name
            ),
            calibrator,
            application_predictions,
        ),
        calibrator,
        application_predictions.scenario_ids,
    )


def _refuse_calibration_data(
    calibrators: dict[str, NaiveIntervalCalibrator | KernelIntervalCalibrator],
    scenario_ids: np.ndarray,
    sample: PredictionSet,
) -> None:
    """Refuse to apply an artifact to the predictions it was fitted on.

    Coverage measured on the calibration data is in-sample and says nothing about
    new scenarios, so `calibrate` requires a test dataset distinct from the one
    `compute` used.
    """

    fingerprints = {
        calibrator.calibration_fingerprint
        for calibrator in calibrators.values()
        if calibrator.calibration_fingerprint is not None
    }
    if _prediction_fingerprint(scenario_ids, sample) in fingerprints:
        raise ValueError(
            "The test predictions are identical to the calibration predictions this "
            "artifact was fitted on. Apply the layer to a separate test dataset "
            "(inputs.test) so that coverage is measured on unseen scenarios.",
        )


def _resolve_output_path(
    args: argparse.Namespace,
    config: dict[str, object],
    predictions_path: str,
    cp_type: str,
    channel_names: tuple[str, ...],
    *,
    target_kind: str,
    multiple_targets: bool,
    task_name: str,
) -> str:
    default_name = _default_calibrated_name(predictions_path, cp_type, channel_names)
    if not multiple_targets:
        output_path = coalesce_option(args.output, config, "output")
        if output_path is None:
            # The family-specific key first, as `screen` reads it first too.
            output_path = config.get(f"calibrated_predictions_{target_kind}")
        if output_path is None:
            output_path = coalesce_option(args.output, config, "calibrated_predictions")
        if output_path is None:
            output_path = str(
                _default_output_dir(task_name, cp_type).joinpath(default_name)
            )
        return str(output_path)

    generic_output = coalesce_option(args.output, config, "output")
    if generic_output is not None:
        output_root = Path(str(generic_output))
        if output_root.suffix.lower() == ".parquet":
            raise ValueError(
                "When calibrating Vm/Va together with line_loading, --output must be a directory or omitted. "
                "Use outputs.calibrated_predictions_volt and outputs.calibrated_predictions_line for explicit file paths.",
            )
        return str(output_root / default_name)

    explicit_key = f"calibrated_predictions_{target_kind}"
    explicit_output = config.get(explicit_key)
    if explicit_output is not None:
        return str(explicit_output)

    shared_output = config.get("calibrated_predictions")
    if shared_output is not None:
        shared_path = Path(str(shared_output))
        if shared_path.suffix.lower() == ".parquet":
            raise ValueError(
                "A mixed-variable run needs separate parquet outputs. "
                "Provide outputs.calibrated_predictions_volt and outputs.calibrated_predictions_line, "
                "or set outputs.calibrated_predictions to a directory.",
            )
        return str(shared_path / default_name)

    return str(_default_output_dir(task_name, cp_type).joinpath(default_name))


def _sparse_strata_context(
    calibrators: Mapping[str, NaiveIntervalCalibrator | KernelIntervalCalibrator],
) -> list[str]:
    """A summary line on the strata whose elements use a fallback bound, if any."""

    sparse = _sparse_strata_listing(calibrators)
    if not sparse:
        return []
    if next(iter(calibrators.values())).sparse_strata == "pooled":
        outcome = (
            "They use their quantile over all calibration scenarios (`sparse_strata: "
            "pooled`), which carries no guarantee within the stratum; `n_pooled` "
            "counts the test values concerned and `<variable>_pooled` marks them."
        )
    else:
        outcome = "They get infinite bounds (`sparse_strata: infinite`)."
    return [f"- Too few calibration samples for a finite bound: {sparse}. {outcome}"]


def _default_calibrated_name(
    predictions_path: str,
    cp_type: str,
    channel_names: tuple[str, ...],
) -> str:
    base_name = Path(predictions_path).stem
    variable_suffix = "_".join(variable.lower() for variable in channel_names)
    return f"{base_name}_{variable_suffix}_with_intervals.parquet"


def _analysis_partials(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    *,
    entity_column: str,
) -> pd.DataFrame:
    """Additive per-element statistics of one block of calibrated rows.

    One row per variable, element and stratum (plus the overall scope), with
    sums, counts and extremes, so that blocks combine exactly in
    `_combine_analysis_partials`. Known values (the set-points marked by the
    `<variable>_known` columns) are counted but left out of every statistic.
    """

    parts: list[pd.DataFrame] = []
    static_columns = (
        ["from_bus", "to_bus", "rate_a"] if entity_column == "branch" else []
    )
    for variable in variables:
        target_column = f"{variable}_target"
        lower_column = f"{variable}_pi_lower"
        upper_column = f"{variable}_pi_upper"
        required = {entity_column, target_column, lower_column, upper_column}
        required.update(static_columns)
        if entity_column == "branch":
            required.add(f"{variable}_pred")
        if not required.issubset(frame.columns):
            raise ValueError(
                f"Missing required analysis columns for {variable}: "
                f"{sorted(required - set(frame.columns))}"
            )
        strat_columns = _analysis_strat_columns_for_variable(frame, variable)
        metadata_columns = _analysis_scenario_metadata_columns_from_frame(
            frame, variable=variable
        )
        work = frame.loc[
            :, ["scenario", entity_column, *static_columns, *metadata_columns]
        ].copy()
        target = frame[target_column].to_numpy(dtype=float)
        lower = frame[lower_column].to_numpy(dtype=float)
        upper = frame[upper_column].to_numpy(dtype=float)
        known = _known_mask(frame, variable)
        work["known"] = known.astype(np.int64)
        work["pooled"] = (_flag(frame, f"{variable}_pooled") & ~known).astype(np.int64)
        work["unknown"] = (~known).astype(np.int64)
        work["width"] = np.where(known, np.nan, upper - lower)
        # The finite widths, for means over values with finite bounds.
        work["finite_width"] = work["width"].where(np.isfinite(work["width"]))
        work["covered"] = np.where(
            known, 0.0, ((target >= lower) & (target <= upper)).astype(float)
        )
        work["scenario_unknown"] = np.where(
            known, np.nan, frame["scenario"].to_numpy(dtype=float)
        )
        # Coverage is measured where the true value is known. A missing prediction
        # (no bounds) counts as not covered: the interval failed there.
        if entity_column == "branch":
            # Out-of-service or unrated branches carry -1 and are not analysed.
            pred = frame[f"{variable}_pred"].to_numpy(dtype=float)
            work = work[~(pred < 0.0) & (target >= 0.0)]
        else:
            work = work[np.isfinite(target)]
        parts.append(
            _entity_statistics(
                work,
                [entity_column],
                static_columns,
                variable=variable,
                stratification="",
            )
        )
        for stratification in strat_columns:
            # Values without a stratum (a known set-point has no prediction
            # bin) count only in the overall rows.
            statistics = _entity_statistics(
                work[work[stratification].notna()],
                [stratification, entity_column],
                static_columns,
                variable=variable,
                stratification=stratification,
            )
            statistics["stratum_value"] = statistics.pop(stratification)
            statistics["stratum"] = [
                _stratum_label(work, stratification, value)
                for value in statistics["stratum_value"].tolist()
            ]
            parts.append(statistics)
    return pd.concat(parts, ignore_index=True)


def _entity_statistics(
    work: pd.DataFrame,
    group_columns: list[str],
    static_columns: list[str],
    *,
    variable: str,
    stratification: str,
) -> pd.DataFrame:
    statistics = (
        work.groupby(group_columns, sort=True, dropna=False)
        .agg(
            width_sum=("width", "sum"),
            width_count=("width", "count"),
            finite_width_sum=("finite_width", "sum"),
            finite_width_count=("finite_width", "count"),
            min_width=("width", "min"),
            max_width=("width", "max"),
            covered_sum=("covered", "sum"),
            n_values=("unknown", "sum"),
            n_known=("known", "sum"),
            n_pooled=("pooled", "sum"),
            n_scenarios=("scenario_unknown", "nunique"),
            **{column: (column, "first") for column in static_columns},
        )
        .reset_index()
    )
    statistics.insert(0, "stratification", stratification)
    statistics.insert(1, "variable", variable)
    if not stratification:
        statistics["stratum_value"] = float("nan")
        statistics["stratum"] = _format_stratum_label("", None)
    return statistics


def _combine_analysis_partials(
    partials: list[pd.DataFrame],
    *,
    entity_column: str,
    bus_types: dict[int, str],
) -> pd.DataFrame:
    """Per-element coverage and width in every stratum, from the blocks' statistics."""

    static_columns = (
        ["from_bus", "to_bus", "rate_a"] if entity_column == "branch" else []
    )
    keys = ["stratification", "stratum_value", "variable", entity_column]
    combined = (
        pd.concat(partials, ignore_index=True)
        .groupby(keys, sort=False, dropna=False)
        .agg(
            stratum=("stratum", "first"),
            width_sum=("width_sum", "sum"),
            width_count=("width_count", "sum"),
            finite_width_sum=("finite_width_sum", "sum"),
            finite_width_count=("finite_width_count", "sum"),
            min_width=("min_width", "min"),
            max_width=("max_width", "max"),
            covered_sum=("covered_sum", "sum"),
            n_values=("n_values", "sum"),
            n_known=("n_known", "sum"),
            n_pooled=("n_pooled", "sum"),
            n_scenarios=("n_scenarios", "sum"),
            **{column: (column, "first") for column in static_columns},
        )
        .reset_index()
    )
    width_count = combined["width_count"].to_numpy(dtype=float)
    n_values = combined["n_values"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_width = np.where(
            width_count > 0,
            combined["width_sum"].to_numpy(dtype=float) / width_count,
            np.nan,
        )
        coverage = np.where(
            n_values > 0,
            combined["covered_sum"].to_numpy(dtype=float) / n_values,
            np.nan,
        )
    elements = combined.loc[
        :, ["variable", "stratification", "stratum_value", "stratum", entity_column]
    ].copy()
    elements[entity_column] = elements[entity_column].astype(np.int64)
    if entity_column == "branch":
        elements["from_bus"] = combined["from_bus"].astype(np.int64)
        elements["to_bus"] = combined["to_bus"].astype(np.int64)
        elements["rate_a"] = combined["rate_a"].astype(float)
    else:
        elements["bus_type"] = [
            bus_types.get(int(bus), "Unknown") for bus in elements["bus"].tolist()
        ]
    elements["n_values"] = combined["n_values"].astype(np.int64)
    elements["n_known"] = combined["n_known"].astype(np.int64)
    elements["n_pooled"] = combined["n_pooled"].astype(np.int64)
    elements["n_scenarios"] = combined["n_scenarios"].astype(np.int64)
    elements["coverage"] = coverage
    elements["mean_width"] = mean_width
    elements["finite_width_sum"] = combined["finite_width_sum"].astype(float)
    elements["finite_width_count"] = combined["finite_width_count"].astype(np.int64)
    elements["min_width"] = combined["min_width"].astype(float)
    elements["max_width"] = combined["max_width"].astype(float)
    return elements.sort_values(
        ["variable", "stratification", "stratum_value", entity_column],
        na_position="first",
    ).reset_index(drop=True)


def _analysis_scenario_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    return _ordered_analysis_strat_columns(frame.columns)


def _analysis_strat_columns_for_variable(
    frame: pd.DataFrame,
    variable: str,
) -> list[str]:
    return [
        column
        for column in _analysis_scenario_columns_from_frame(frame)
        if _stratification_applies_to_variable(column, variable)
    ]
