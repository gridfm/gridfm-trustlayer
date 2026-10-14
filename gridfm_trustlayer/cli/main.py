"""Command-line interface: `compute` fits the layer on calibration data, `calibrate`
applies it to test data, and `screen` compares the intervals with operating limits."""

from __future__ import annotations

import argparse
import glob
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

from gridfm_trustlayer.cli.config import (
    N_EFF_TARGET_AUTO,
    coalesce_option,
    load_cli_config,
    normalize_variables,
    resolve_command_config,
    validate_bandwidth,
    validate_command_config,
    validate_cp_type,
    validate_distance,
    validate_k_nn,
    validate_n_eff_min,
    validate_n_eff_target,
    validate_pred_n_bins,
    validate_strat,
    validate_system_load_n_bins,
)
from gridfm_trustlayer.conformal import ChannelSet, NaiveIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import (
    DEFAULT_BANDWIDTH,
    DEFAULT_DISTANCE,
    DEFAULT_K_NN,
    DEFAULT_N_EFF_MIN,
    KernelIntervalCalibrator,
)
from gridfm_trustlayer.conformal.tuning import (
    BandwidthTuningResult,
    TuningOptions,
    tune_n_eff_target,
)
from gridfm_trustlayer.io import (
    GroundTruthSet,
    PredictionSet,
    compute_line_loading,
    load_ground_truth,
    load_predictions,
)
from gridfm_trustlayer.io.artifact import (
    load_interval_artifacts,
    save_interval_artifacts,
)
from gridfm_trustlayer.io.embeddings import (
    DEFAULT_POOLING,
    load_scenario_embeddings,
    validate_pooling,
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
    read_frame,
    read_scenario_ids,
    rows_per_scenario,
    scenario_blocks,
)

SCREEN_BOUND_TOLERANCE = 1e-6
# Rows (scenario x entity) that calibrate and screen hold in memory at once.
_BLOCK_ROWS = DEFAULT_BLOCK_ROWS
# Scenarios whose predictions enter the in-sample fingerprint (with every id).
_FINGERPRINT_SCENARIOS = 1000
BASE_ANALYSIS_STRAT_COLUMNS = ("n_outages", "outage_type", "system_load_bin")


def main(argv: list[str] | None = None) -> int:
    """Run the gridfm_trustlayer command-line interface."""

    # The commands read and free one block of scenarios at a time. Arrow's default
    # allocators keep freed blocks reserved; the system allocator returns them,
    # which keeps the peak memory of a long run close to that of one block. Each
    # decoding thread also keeps its own freed memory, and beyond a few threads
    # reading a block is no faster, so the thread count is capped.
    pa.set_memory_pool(pa.system_memory_pool())
    pa.set_cpu_count(min(4, pa.cpu_count()))
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compute":
        _run_compute(args)
        return 0
    if args.command == "calibrate":
        _run_calibrate(args)
        return 0
    if args.command == "screen":
        _run_screen(args)
        return 0
    raise ValueError(f"Unsupported command: {args.command!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(prog="gridfm_trustlayer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute_parser = subparsers.add_parser(
        "compute",
        help="Fit the layer on the calibration dataset (inputs.calibration) and save a .trustlayer artifact.",
    )
    compute_parser.add_argument(
        "--config",
        required=True,
        help="Required trustlayer YAML config file. CLI flags override config values.",
    )
    compute_parser.add_argument(
        "--predictions", help="Predictions parquet with *_pred columns."
    )
    compute_parser.add_argument(
        "--pf-node", help="Ground-truth bus parquet used to fit residual quantiles."
    )
    compute_parser.add_argument(
        "--pf-edge", help="Ground-truth branch parquet used to fit residual quantiles."
    )
    compute_parser.add_argument(
        "--output", help="Destination .trustlayer artifact path."
    )
    compute_parser.add_argument(
        "--alpha", type=float, help="Miscoverage level for conformal intervals."
    )
    compute_parser.add_argument(
        "--cp-type",
        help="Calibration type: 'element' (stratified) or 'kernel' (kernel-weighted).",
    )
    compute_parser.add_argument(
        "--strat",
        help="Optional residual stratification. Supported values: 'n_outages', 'outage_type', 'system_load_bin', 'pred_bins'.",
    )
    compute_parser.add_argument(
        "--system-load-n-bins",
        type=int,
        help="Requested number of percentile bins when --strat system_load_bin is used.",
    )
    compute_parser.add_argument(
        "--pred-n-bins",
        type=int,
        help="Requested number of percentile bins when --strat pred_bins is used.",
    )
    compute_parser.add_argument(
        "--variables",
        nargs="+",
        help="Variables to calibrate: any subset of Vm, Va, and line_loading.",
    )
    compute_parser.add_argument(
        "--embeddings",
        help="Scenario embeddings parquet (emb_* columns). Required for cp_type 'kernel'.",
    )
    compute_parser.add_argument(
        "--embedding-pooling",
        help="How per-entity embeddings are pooled into one vector per scenario: mean, max, meanmax, or concat.",
    )
    compute_parser.add_argument(
        "--distance",
        help="Distance over scenario representations: 'euclidean' (default) or 'cosine'.",
    )
    compute_parser.add_argument(
        "--bandwidth",
        help="How the kernel bandwidth is chosen: 'n_eff' (default, solve for a target effective sample size) or 'knn' (distance to the k_nn-th neighbour).",
    )
    compute_parser.add_argument(
        "--k-nn",
        type=int,
        help="Nearest-neighbour rank whose distance sets the kernel bandwidth (--bandwidth knn).",
    )
    compute_parser.add_argument(
        "--n-eff-target",
        help="Target effective sample size solved for per test scenario (--bandwidth n_eff). Defaults to 600. 'auto' chooses it by cross-validation on the calibration data (options under 'tuning' in the config).",
    )
    compute_parser.add_argument(
        "--n-eff-min",
        type=float,
        help="Minimum effective sample size before a scenario falls back to the stratum-wide quantile (cp_type 'kernel').",
    )
    compute_parser.add_argument(
        "--task-name", help="Task name used to interpret prediction schema."
    )
    compute_parser.add_argument(
        "--generator-predictions",
        help="Optional generator predictions parquet for OPF artifacts.",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Apply the artifact to the test dataset (inputs.test): write the predictions with their bounds, coverage reports and figures.",
    )
    calibrate_parser.add_argument(
        "--config",
        required=True,
        help="Required trustlayer YAML config file. CLI flags override config values.",
    )
    calibrate_parser.add_argument(
        "--predictions",
        help="Predictions parquet to augment or transform into branch outputs.",
    )
    calibrate_parser.add_argument(
        "--artifact", help="Previously computed .trustlayer artifact."
    )
    calibrate_parser.add_argument(
        "--output", help="Destination parquet for calibrated predictions."
    )
    calibrate_parser.add_argument(
        "--cp-type",
        help="Calibration type: 'element' (stratified) or 'kernel' (kernel-weighted).",
    )
    calibrate_parser.add_argument(
        "--strat",
        help="Optional residual stratification override. Supported values: 'n_outages', 'outage_type', 'system_load_bin', 'pred_bins'. Must match the saved artifact when provided.",
    )
    calibrate_parser.add_argument(
        "--pf-node",
        help="Ground-truth bus parquet; required for line_loading analysis when targets are not stored in predictions.",
    )
    calibrate_parser.add_argument(
        "--pf-edge",
        help="Ground-truth branch parquet; required for line_loading analysis when targets are not stored in predictions.",
    )
    calibrate_parser.add_argument(
        "--embeddings",
        help="Scenario embeddings parquet for the application scenarios. Required when the artifact is a kernel artifact.",
    )
    calibrate_parser.add_argument(
        "--task-name", help="Task name used to interpret prediction schema."
    )
    calibrate_parser.add_argument(
        "--generator-predictions",
        help="Optional generator predictions parquet for OPF artifacts.",
    )
    calibrate_figures_group = calibrate_parser.add_mutually_exclusive_group()
    calibrate_figures_group.add_argument(
        "--figures",
        dest="save_figures",
        action="store_true",
        default=None,
        help="Write analysis figures during calibrate (default behavior).",
    )
    calibrate_figures_group.add_argument(
        "--no-figures",
        dest="save_figures",
        action="store_false",
        help="Skip analysis figure generation and write only parquet/CSV outputs.",
    )

    screen_parser = subparsers.add_parser(
        "screen",
        help="Compare the test bounds with the operating limits (screen_bounds) and count caught, missed and false alarms.",
    )
    screen_parser.add_argument(
        "--config",
        required=True,
        help="Required trustlayer YAML config file. CLI flags override config values.",
    )
    screen_parser.add_argument(
        "--variables",
        nargs="+",
        help="Variables to screen. Defaults to config values for the screen command.",
    )
    screen_parser.add_argument(
        "--lower-bound",
        type=float,
        help="Override lower screening bound for a single selected variable. Not used for line_loading.",
    )
    screen_parser.add_argument(
        "--upper-bound",
        type=float,
        help="Override upper screening bound for a single selected variable.",
    )
    screen_parser.add_argument(
        "--output", help="Destination directory for screening CSVs and figures."
    )
    screen_parser.add_argument(
        "--calibrated-predictions",
        help="Single calibrated parquet input for one target family.",
    )
    screen_parser.add_argument(
        "--calibrated-predictions-volt",
        help="Calibrated voltage parquet input override.",
    )
    screen_parser.add_argument(
        "--calibrated-predictions-line",
        help="Calibrated line-loading parquet input override.",
    )
    screen_parser.add_argument(
        "--pf-node",
        help="Ground-truth bus parquet used when target columns are not stored in the calibrated voltage parquet.",
    )
    screen_parser.add_argument(
        "--pf-edge",
        help="Ground-truth branch parquet used when target columns are not stored in the calibrated outputs.",
    )
    screen_parser.add_argument(
        "--task-name", help="Task name used for default output-path resolution."
    )

    return parser


def _run_compute(args: argparse.Namespace) -> None:
    config = resolve_command_config(load_cli_config(args.config), "compute")
    validate_command_config("compute", config)
    cp_type = validate_cp_type(coalesce_option(args.cp_type, config, "cp_type"))
    strat = validate_strat(coalesce_option(args.strat, config, "strat"))
    system_load_n_bins = validate_system_load_n_bins(
        coalesce_option(args.system_load_n_bins, config, "system_load_n_bins")
    )
    pred_n_bins = validate_pred_n_bins(
        coalesce_option(args.pred_n_bins, config, "pred_n_bins")
    )
    if strat == "system_load_bin" and system_load_n_bins is None:
        raise ValueError(
            "system_load_n_bins is required when strat is 'system_load_bin'."
        )
    if strat == "pred_bins" and pred_n_bins is None:
        raise ValueError("pred_n_bins is required when strat is 'pred_bins'.")
    variables = normalize_variables(
        coalesce_option(args.variables, config, "variables")
    )
    variable_groups = _partition_variables(variables)
    embedding_pooling = validate_pooling(
        coalesce_option(args.embedding_pooling, config, "embedding_pooling")
        or DEFAULT_POOLING
    )
    k_nn = validate_k_nn(coalesce_option(args.k_nn, config, "k_nn")) or DEFAULT_K_NN
    distance = validate_distance(
        coalesce_option(args.distance, config, "distance") or DEFAULT_DISTANCE
    )
    bandwidth = validate_bandwidth(
        coalesce_option(args.bandwidth, config, "bandwidth") or DEFAULT_BANDWIDTH
    )
    n_eff_target = validate_n_eff_target(
        coalesce_option(args.n_eff_target, config, "n_eff_target")
    )
    n_eff_min = (
        validate_n_eff_min(coalesce_option(args.n_eff_min, config, "n_eff_min"))
        or DEFAULT_N_EFF_MIN
    )
    tuning_options = (
        TuningOptions.from_config(config.get("tuning"))
        if n_eff_target == N_EFF_TARGET_AUTO
        else None
    )
    if tuning_options is not None and (cp_type != "kernel" or bandwidth != "n_eff"):
        raise ValueError(
            "n_eff_target auto requires cp_type kernel and bandwidth n_eff."
        )
    fixed_n_eff_target = n_eff_target if isinstance(n_eff_target, float) else None
    predictions_path = coalesce_option(
        args.predictions, config, "predictions", required=True
    )
    predictions = load_predictions(
        predictions_path,
        task_name=coalesce_option(args.task_name, config, "task_name"),
        generator_parquet_path=coalesce_option(
            args.generator_predictions,
            config,
            "generator_predictions",
        ),
    )
    if cp_type == "kernel":
        predictions = _attach_scenario_embeddings(
            predictions,
            coalesce_option(args.embeddings, config, "embeddings", required=True),
            pooling=embedding_pooling,
        )
    pf_node_path = coalesce_option(args.pf_node, config, "pf_node", required=True)
    ground_truth = load_ground_truth(
        pf_node_path,
        coalesce_option(args.pf_edge, config, "pf_edge", required=True),
        scenario_ids=predictions.scenario_ids,
    )
    tuning_groups = (
        None
        if tuning_options is None
        else _load_tuning_groups(pf_node_path, tuning_options.group_by)
    )
    artifact_path = coalesce_option(args.output, config, "artifact")
    if artifact_path is None:
        artifact_path = str(
            _default_output_dir(predictions.task_name, cp_type).joinpath(
                _default_artifact_name(predictions_path, cp_type, variables),
            ),
        )
    # Every scenario of the calibration dataset is used for fitting; the layer is
    # evaluated by `calibrate` on a separate test dataset.
    fingerprint = _prediction_fingerprint(predictions.scenario_ids, predictions)
    calibrators: dict[str, NaiveIntervalCalibrator | KernelIntervalCalibrator] = {}
    tuning_results: dict[str, BandwidthTuningResult] = {}
    for target_kind, target_variables in variable_groups.items():
        working_predictions = predictions
        working_ground_truth = ground_truth
        base_predictions, _ = _build_target_channel_sets(
            working_predictions,
            working_ground_truth,
            target_variables,
        )
        system_load_bin_edges = None
        if strat == "system_load_bin":
            assert system_load_n_bins is not None
            system_load_bin_edges = _compute_system_load_bin_edges(
                working_ground_truth.system_load,
                system_load_n_bins,
            )
        pred_bin_edges = None
        if strat == "pred_bins":
            assert pred_n_bins is not None
            pred_bin_edges = _compute_prediction_bin_edges(
                base_predictions,
                pred_n_bins,
            )
        calibration_predictions, calibration_ground_truth = _build_calibration_sets(
            working_predictions,
            working_ground_truth,
            target_variables,
            strat=strat,
            system_load_n_bins=system_load_n_bins,
            system_load_bin_edges=system_load_bin_edges,
            pred_n_bins=pred_n_bins,
            pred_bin_edges=pred_bin_edges,
        )
        alpha = float(coalesce_option(args.alpha, config, "alpha", required=True))
        calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator
        if cp_type == "kernel":
            assert calibration_predictions.scenario_embeddings is not None
            family_n_eff_target = fixed_n_eff_target
            family_distance = distance
            if tuning_options is not None:
                # Cross-validated on the calibration data only; the test set is untouched.
                tuning_results[target_kind] = tune_n_eff_target(
                    calibration_predictions,
                    calibration_ground_truth,
                    alpha=alpha,
                    channel_names=target_variables,
                    strat=strat,
                    embedding_pooling=embedding_pooling,
                    distance=distance,
                    n_eff_min=n_eff_min,
                    options=tuning_options,
                    groups=None
                    if tuning_groups is None
                    else tuning_groups.reindex(
                        calibration_predictions.scenario_ids
                    ).to_numpy(),
                    system_load_n_bins=system_load_n_bins,
                    system_load_bin_edges=system_load_bin_edges,
                )
                family_n_eff_target = tuning_results[target_kind].selected_n_eff_target
                family_distance = tuning_results[target_kind].selected_distance
            calibrator = KernelIntervalCalibrator.fit(
                calibration_predictions,
                calibration_ground_truth,
                alpha=alpha,
                channel_names=target_variables,
                strat=strat,
                embeddings=calibration_predictions.scenario_embeddings,
                embedding_pooling=embedding_pooling,
                k_nn=k_nn,
                n_eff_min=n_eff_min,
                bandwidth=bandwidth,
                n_eff_target=family_n_eff_target,
                distance=family_distance,
                system_load_n_bins=system_load_n_bins,
                system_load_bin_edges=system_load_bin_edges,
            )
        else:
            calibrator = NaiveIntervalCalibrator.fit(
                calibration_predictions,
                calibration_ground_truth,
                alpha=alpha,
                channel_names=target_variables,
                strat=strat,
                system_load_n_bins=system_load_n_bins,
                system_load_bin_edges=system_load_bin_edges,
                pred_n_bins=pred_n_bins,
                pred_bin_edges=pred_bin_edges,
            )
        calibrator.calibration_fingerprint = fingerprint
        if (
            isinstance(calibrator, KernelIntervalCalibrator)
            and target_kind in tuning_results
        ):
            calibrator.tuning = tuning_results[target_kind].summary()
        calibrators[target_kind] = calibrator
    save_interval_artifacts(calibrators, artifact_path)
    if tuning_results:
        _write_tuning_reports(tuning_results, artifact_path)


def _load_tuning_groups(pf_node_path: str, group_by: str) -> pd.Series | None:
    """Fold groups per scenario for `n_eff_target: auto`, or None for scenario-level folds.

    Datakit writes `load_scenario_idx`: scenarios generated from the same load
    profile (different topologies) share it. Keeping them in one fold prevents
    near-duplicates on both sides of a split from flattering the kernel.
    """

    if group_by == "none":
        return None
    dataset = open_dataset(pf_node_path)
    if "load_scenario_idx" not in dataset.schema.names:
        if group_by == "load_scenario_idx":
            raise ValueError(
                "tuning.group_by is 'load_scenario_idx' but the calibration pf_node "
                "parquet has no load_scenario_idx column.",
            )
        return None
    frame = dataset.to_table(columns=["scenario", "load_scenario_idx"]).to_pandas()
    return frame.groupby("scenario")["load_scenario_idx"].first()


def _write_tuning_reports(
    results: dict[str, BandwidthTuningResult], artifact_path: str
) -> None:
    """Write the per-candidate and per-check tables next to the artifact."""

    stem = Path(artifact_path)
    candidates = pd.concat(
        [result.candidates.assign(target=kind) for kind, result in results.items()],
        ignore_index=True,
    )
    details = pd.concat(
        [result.details.assign(target=kind) for kind, result in results.items()],
        ignore_index=True,
    )
    summary_path = stem.with_name(f"{stem.stem}_kernel_tuning.csv")
    details_path = stem.with_name(f"{stem.stem}_kernel_tuning_details.csv")
    candidates.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    for kind, result in results.items():
        print(
            f"kernel tuning ({kind}): n_eff_target = {result.selected_n_eff_target:g}, "
            f"distance = {result.selected_distance} ({result.reason}; "
            f"{len(result.candidates)} candidates, {result.options.search} search, "
            f"{result.options.folds} {'grouped ' if result.grouped else ''}folds). "
            f"Report: {summary_path}"
        )


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
    artifact_path = coalesce_option(args.artifact, config, "artifact", required=True)
    calibrators = load_interval_artifacts(artifact_path)
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
    embeddings_path = None
    pooling = None
    kernel_calibrators = [
        calibrator
        for calibrator in calibrators.values()
        if isinstance(calibrator, KernelIntervalCalibrator)
    ]
    if kernel_calibrators:
        poolings = {calibrator.embedding_pooling for calibrator in kernel_calibrators}
        if len(poolings) != 1:
            raise ValueError(
                f"Kernel artifact mixes embedding_pooling strategies: {sorted(poolings)}."
            )
        pooling = next(iter(poolings))
        embeddings_path = coalesce_option(
            args.embeddings, config, "embeddings", required=True
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
        reference = scan_ground_truth_reference(*ground_truth_paths)
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
            task_name=sample.task_name,
        )
        for target_kind, calibrator in calibrators.items()
    }
    writers = {
        target_kind: PartitionedWriter(path)
        for target_kind, path in output_paths.items()
    }
    partials: dict[str, list[pd.DataFrame]] = {kind: [] for kind in calibrators}
    for block in scenario_blocks(
        scenario_ids,
        rows_per_scenario(predictions_path, len(scenario_ids)),
        _BLOCK_ROWS,
    ):
        predictions = load_predictions(
            predictions_path,
            task_name=task_name,
            generator_parquet_path=generator_path,
            scenario_ids=block,
        )
        if embeddings_path is not None:
            assert pooling is not None
            predictions = _attach_scenario_embeddings(
                predictions, embeddings_path, pooling=pooling
            )
        ground_truth = None
        if ground_truth_paths is not None:
            ground_truth = load_ground_truth(
                *ground_truth_paths,
                scenario_ids=predictions.scenario_ids,
                reference=reference,
            )
        original_frame = (
            read_frame(predictions_path, scenario_ids=block)
            if has_bus_targets
            else None
        )
        for target_kind, calibrator in calibrators.items():
            output_frame = _calibrated_block_frame(
                calibrator, predictions, ground_truth, original_frame
            )
            writers[target_kind].write(output_frame)
            partials[target_kind].append(
                _analysis_partials(
                    output_frame,
                    calibrator.channel_names,
                    entity_column=calibrator.entity_name,
                )
            )

    for target_kind, calibrator in calibrators.items():
        analysis_dir = Path(output_paths[target_kind]).parent
        figures_dir = analysis_dir / "figures"
        if save_figures:
            figures_dir.mkdir(parents=True, exist_ok=True)
        analysis_frame = _combine_analysis_partials(
            partials[target_kind],
            entity_column=calibrator.entity_name,
            bus_types=bus_types,
        )
        suffix = "line" if calibrator.entity_name == "branch" else "volt"
        analysis_frame.to_csv(
            analysis_dir / f"trust_analysis_{suffix}.csv", index=False
        )
        summary_frame = (
            _build_line_summary_frame(analysis_frame)
            if calibrator.entity_name == "branch"
            else _build_summary_frame(analysis_frame)
        )
        summary_frame.to_csv(analysis_dir / f"trust_summary_{suffix}.csv", index=False)
        _build_reliability_frame(
            analysis_frame,
            entity_column=calibrator.entity_name,
            target_coverage=1.0 - float(calibrator.alpha),
        ).to_csv(analysis_dir / f"trust_reliability_{suffix}.csv", index=False)
        if save_figures:
            if calibrator.entity_name == "branch":
                _save_line_analysis_figures(analysis_frame, figures_dir)
            else:
                _save_analysis_figures(analysis_frame, figures_dir)


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
        output_frame,
        calibrator,
        application_predictions.scenario_ids,
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


def _run_screen(args: argparse.Namespace) -> None:
    config = resolve_command_config(load_cli_config(args.config), "screen")
    validate_command_config("screen", config)
    cp_type = validate_cp_type(coalesce_option(None, config, "cp_type"))
    variables = normalize_variables(
        coalesce_option(args.variables, config, "variables")
    )
    bounds_by_variable = _resolve_screen_bounds(args, config, variables)
    target_groups = _partition_variables(variables)

    input_paths = {
        target_kind: _resolve_screen_input_path(
            args,
            config,
            target_kind=target_kind,
            channel_names=channel_names,
            cp_type=cp_type,
            task_name=str(
                coalesce_option(args.task_name, config, "task_name") or "unknown"
            ),
        )
        for target_kind, channel_names in target_groups.items()
    }
    screen_dir = _resolve_screen_output_dir(
        args,
        config,
        input_paths=input_paths,
        task_name=str(
            coalesce_option(args.task_name, config, "task_name") or "unknown"
        ),
        cp_type=cp_type,
    )
    screen_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = screen_dir / "screen_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for target_kind, channel_names in target_groups.items():
        input_path = input_paths[target_kind]
        if not Path(input_path).exists():
            raise FileNotFoundError(
                f"Calibrated predictions not found: {input_path}. Run calibrate first."
            )
        entity_column = "bus" if target_kind == "volt" else "branch"
        group_columns = (
            ["variable", "bus", "bus_type"]
            if target_kind == "volt"
            else ["variable", "branch", "from_bus", "to_bus", "rate_a"]
        )
        columns = set(column_names(input_path))
        needed = {
            f"{variable}_{suffix}"
            for variable in channel_names
            for suffix in ("target", "pi_lower", "pi_upper")
        }
        if target_kind == "line":
            needed.update({"from_bus", "to_bus", "rate_a"})
        reference = None
        if not needed.issubset(columns):
            reference = scan_ground_truth_reference(*_ground_truth_paths(args, config))
        bus_types = _scan_bus_types(input_path) if target_kind == "volt" else {}
        scenario_ids = read_scenario_ids(input_path)
        detail_writer = PartitionedWriter(
            screen_dir / f"trust_screen_detail_{target_kind}.parquet"
        )
        counts: list[pd.DataFrame] = []
        for block in scenario_blocks(
            scenario_ids,
            rows_per_scenario(input_path, len(scenario_ids)),
            _BLOCK_ROWS,
        ):
            frame = read_frame(
                input_path,
                ["scenario", entity_column, *sorted(needed)],
                scenario_ids=block,
            )
            if target_kind == "volt":
                frame = _ensure_voltage_screen_targets(
                    frame, channel_names, args, config, reference=reference
                )
                detail_frame = _build_voltage_screen_detail_frame(
                    frame, channel_names, bounds_by_variable, bus_types=bus_types
                )
            else:
                frame = _ensure_line_screen_targets(
                    frame, channel_names, args, config, reference=reference
                )
                detail_frame = _build_line_screen_detail_frame(
                    frame, channel_names, bounds_by_variable
                )
            detail_writer.write(detail_frame)
            counts.append(_screen_counts(detail_frame, group_columns))

        entity_summary = _combine_screen_counts(counts, group_columns)
        summary_frame = (
            _build_voltage_screen_summary(entity_summary)
            if target_kind == "volt"
            else _build_line_screen_summary(entity_summary)
        )
        entity_summary.to_csv(
            screen_dir / f"trust_screen_entity_{target_kind}.csv", index=False
        )
        summary_frame.to_csv(
            screen_dir / f"trust_screen_summary_{target_kind}.csv", index=False
        )
        _save_screen_figures(
            entity_summary, summary_frame, figures_dir, target_kind=target_kind
        )


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

    if args.output is not None:
        output_root = Path(args.output)
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


def _resolve_screen_bounds(
    args: argparse.Namespace,
    config: dict[str, object],
    variables: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    bounds_by_variable: dict[str, dict[str, float]] = {}
    raw_bounds = config.get("screen_bounds")
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

    if args.lower_bound is not None or args.upper_bound is not None:
        if len(variables) != 1:
            raise ValueError(
                "--lower-bound and --upper-bound can override bounds only when screening a single variable."
            )
        if variables[0] == "line_loading":
            if args.upper_bound is None:
                raise ValueError(
                    "Provide --upper-bound when overriding line_loading screening bounds from the CLI."
                )
            lower_value = (
                float("nan") if args.lower_bound is None else float(args.lower_bound)
            )
        else:
            if args.lower_bound is None or args.upper_bound is None:
                raise ValueError(
                    "Provide both --lower-bound and --upper-bound when overriding screening bounds from the CLI."
                )
            lower_value = float(args.lower_bound)
        bounds_by_variable[variables[0]] = {
            "lower": lower_value,
            "upper": float(args.upper_bound),
        }

    for variable in variables:
        if variable not in bounds_by_variable:
            raise ValueError(
                f"Missing screening bounds for {variable!r}. Add it under screen_bounds in the config or override it from the CLI.",
            )
    return bounds_by_variable


def _resolve_screen_input_path(
    args: argparse.Namespace,
    config: dict[str, object],
    *,
    target_kind: str,
    channel_names: tuple[str, ...],
    cp_type: str,
    task_name: str,
) -> str:
    if target_kind == "volt":
        cli_value = args.calibrated_predictions_volt
        explicit_key = "calibrated_predictions_volt"
    else:
        cli_value = args.calibrated_predictions_line
        explicit_key = "calibrated_predictions_line"

    if cli_value is not None:
        return str(cli_value)
    if explicit_key in config:
        return str(config[explicit_key])
    shared_value = coalesce_option(
        args.calibrated_predictions, config, "calibrated_predictions"
    )
    if shared_value is not None and not Path(str(shared_value)).is_dir():
        return str(shared_value)

    # A mixed `calibrate` run writes into a directory (the shared output, or the
    # default one), so look the family's parquet up there.
    directory = (
        Path(str(shared_value))
        if shared_value is not None
        else _default_output_dir(task_name, cp_type)
    )
    predictions_path = coalesce_option(None, config, "predictions", required=True)
    return str(
        _find_calibrated_parquet(
            directory, str(predictions_path), target_kind, channel_names
        )
    )


def _find_calibrated_parquet(
    directory: Path,
    predictions_path: str,
    target_kind: str,
    channel_names: tuple[str, ...],
) -> Path:
    """Locate the parquet `calibrate` wrote for one target family in `directory`.

    `calibrate` names the file after the calibrated variables (e.g. `_vm_va_`),
    which can be more than the screened ones (e.g. only `Vm`), so match on the
    target family and require the screened variables to be among them.
    """

    stem = Path(predictions_path).stem
    suffix = "_with_intervals.parquet"
    line_name = f"{stem}_line_loading{suffix}"
    screened = {name.lower() for name in channel_names}
    matches = []
    for path in sorted(directory.glob(f"{glob.escape(stem)}_*{suffix}")):
        if target_kind == "line":
            if path.name == line_name:
                matches.append(path)
            continue
        calibrated = set(path.name[len(stem) + 1 : -len(suffix)].split("_"))
        if path.name != line_name and screened <= calibrated:
            matches.append(path)

    family = "line-loading" if target_kind == "line" else "voltage"
    key = f"outputs.calibrated_predictions_{target_kind}"
    if not matches:
        raise FileNotFoundError(
            f"No calibrated {family} parquet for {stem!r} in {directory}. Run "
            f"calibrate first, or set {key} to the file to screen.",
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Several calibrated {family} parquets in {directory} match: {names}. "
            f"Set {key} to the file to screen.",
        )
    return matches[0]


def _resolve_screen_output_dir(
    args: argparse.Namespace,
    config: dict[str, object],
    *,
    input_paths: dict[str, str],
    task_name: str,
    cp_type: str,
) -> Path:
    output_value = coalesce_option(args.output, config, "output")
    if output_value is None:
        output_value = coalesce_option(args.output, config, "screen_dir")
    if output_value is not None:
        return Path(str(output_value))

    parent_dirs = {Path(path).parent for path in input_paths.values()}
    if len(parent_dirs) == 1:
        return next(iter(parent_dirs))
    return _default_output_dir(task_name, cp_type)


def _ensure_voltage_screen_targets(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    args: argparse.Namespace,
    config: dict[str, object],
    *,
    reference: GroundTruthReference | None = None,
) -> pd.DataFrame:
    missing_targets = [
        variable for variable in variables if f"{variable}_target" not in frame.columns
    ]
    if not missing_targets:
        return frame

    ground_truth = load_ground_truth(
        *_ground_truth_paths(args, config),
        scenario_ids=np.unique(frame["scenario"].to_numpy(dtype=np.int64)),
        reference=reference,
    )
    target_frame = _voltage_target_frame(ground_truth, tuple(missing_targets))
    return frame.merge(
        target_frame, on=["scenario", "bus"], how="left", validate="one_to_one"
    )


def _ensure_line_screen_targets(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    args: argparse.Namespace,
    config: dict[str, object],
    *,
    reference: GroundTruthReference | None = None,
) -> pd.DataFrame:
    missing_target = any(
        f"{variable}_target" not in frame.columns for variable in variables
    )
    missing_metadata = not {"from_bus", "to_bus", "rate_a"}.issubset(frame.columns)
    if not missing_target and not missing_metadata:
        return frame

    ground_truth = load_ground_truth(
        *_ground_truth_paths(args, config),
        scenario_ids=np.unique(frame["scenario"].to_numpy(dtype=np.int64)),
        reference=reference,
    )
    target_loading = compute_line_loading(
        ground_truth.Vm, ground_truth.Va, ground_truth
    )
    target_frame = pd.DataFrame(
        {
            "scenario": np.repeat(
                ground_truth.scenario_ids, len(ground_truth.branch_ids)
            ),
            "branch": np.tile(ground_truth.branch_ids, len(ground_truth.scenario_ids)),
            "from_bus": np.tile(ground_truth.f_bus, len(ground_truth.scenario_ids)),
            "to_bus": np.tile(ground_truth.t_bus, len(ground_truth.scenario_ids)),
            "rate_a": np.tile(ground_truth.rate_a, len(ground_truth.scenario_ids)),
            "line_loading_target": target_loading.reshape(-1),
        },
    )
    merge_columns = ["scenario", "branch"]
    extra_columns = [
        column
        for column in ["from_bus", "to_bus", "rate_a", "line_loading_target"]
        if column not in frame.columns
    ]
    return frame.merge(
        target_frame.loc[:, merge_columns + extra_columns],
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )


def _build_voltage_screen_detail_frame(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    bounds_by_variable: dict[str, dict[str, float]],
    *,
    bus_types: dict[int, str],
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for variable in variables:
        target_column = f"{variable}_target"
        lower_column = f"{variable}_pi_lower"
        upper_column = f"{variable}_pi_upper"
        required_columns = {
            "scenario",
            "bus",
            target_column,
            lower_column,
            upper_column,
        }
        if not required_columns.issubset(frame.columns):
            raise ValueError(
                f"Missing required voltage-screen columns for {variable}: {sorted(required_columns - set(frame.columns))}"
            )
        variable_frame = frame.loc[
            :, ["scenario", "bus", target_column, lower_column, upper_column]
        ].copy()
        variable_frame["variable"] = variable
        variable_frame["bus_type"] = (
            variable_frame["bus"].map(bus_types).fillna("Unknown").astype(object)
        )
        variable_frame = _append_screen_flag_columns(
            variable_frame, variable, bounds_by_variable[variable]
        )
        records.append(variable_frame)
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["variable", "scenario", "bus"])
        .reset_index(drop=True)
    )


def _build_line_screen_detail_frame(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    bounds_by_variable: dict[str, dict[str, float]],
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for variable in variables:
        target_column = f"{variable}_target"
        lower_column = f"{variable}_pi_lower"
        upper_column = f"{variable}_pi_upper"
        required_columns = {
            "scenario",
            "branch",
            "from_bus",
            "to_bus",
            "rate_a",
            target_column,
            lower_column,
            upper_column,
        }
        if not required_columns.issubset(frame.columns):
            raise ValueError(
                f"Missing required line-screen columns for {variable}: {sorted(required_columns - set(frame.columns))}"
            )
        variable_frame = frame.loc[
            :,
            [
                "scenario",
                "branch",
                "from_bus",
                "to_bus",
                "rate_a",
                target_column,
                lower_column,
                upper_column,
            ],
        ].copy()
        valid_mask = variable_frame[target_column] >= 0.0
        variable_frame = variable_frame[valid_mask].copy()
        variable_frame["variable"] = variable
        variable_frame = _append_screen_flag_columns(
            variable_frame, variable, bounds_by_variable[variable]
        )
        records.append(variable_frame)
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["variable", "scenario", "branch"])
        .reset_index(drop=True)
    )


def _append_screen_flag_columns(
    frame: pd.DataFrame,
    variable: str,
    bounds: dict[str, float],
) -> pd.DataFrame:
    target_column = f"{variable}_target"
    lower_column = f"{variable}_pi_lower"
    upper_column = f"{variable}_pi_upper"
    lower_bound = float(bounds["lower"])
    upper_bound = float(bounds["upper"])
    frame["target"] = frame[target_column].astype(float)
    frame["pi_lower"] = frame[lower_column].astype(float)
    frame["pi_upper"] = frame[upper_column].astype(float)
    frame["lower_bound"] = lower_bound
    frame["upper_bound"] = upper_bound
    if variable == "line_loading":
        frame["pred_lower_violation"] = False
        frame["true_lower_violation"] = False
    else:
        frame["pred_lower_violation"] = frame["pi_lower"] < (
            lower_bound - SCREEN_BOUND_TOLERANCE
        )
        frame["true_lower_violation"] = frame["target"] < (
            lower_bound - SCREEN_BOUND_TOLERANCE
        )
    frame["pred_upper_violation"] = frame["pi_upper"] > (
        upper_bound + SCREEN_BOUND_TOLERANCE
    )
    frame["true_upper_violation"] = frame["target"] > (
        upper_bound + SCREEN_BOUND_TOLERANCE
    )
    frame["pred_any_violation"] = (
        frame["pred_lower_violation"] | frame["pred_upper_violation"]
    )
    frame["true_any_violation"] = (
        frame["true_lower_violation"] | frame["true_upper_violation"]
    )
    for side in ("lower", "upper", "any"):
        pred = frame[f"pred_{side}_violation"]
        true = frame[f"true_{side}_violation"]
        frame[f"{side}_tp"] = (pred & true).astype(int)
        frame[f"{side}_fp"] = (pred & (~true)).astype(int)
        frame[f"{side}_fn"] = ((~pred) & true).astype(int)
        frame[f"{side}_tn"] = ((~pred) & (~true)).astype(int)
    return frame


def _screen_counts(
    detail_frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Confusion counts and sample counts per entity of one block."""

    return (
        detail_frame.groupby(group_columns, sort=False, dropna=False)
        .agg(
            n_samples=("scenario", "size"),
            **{column: (column, "sum") for column in _screen_count_columns()},
        )
        .reset_index()
    )


def _combine_screen_counts(
    counts: list[pd.DataFrame],
    group_columns: list[str],
) -> pd.DataFrame:
    """Per-entity screening table (`trust_screen_entity_*.csv`) from the blocks' counts."""

    if not counts:
        return pd.DataFrame(
            columns=[*group_columns, "n_samples", *_screen_count_columns()]
        )
    summary = (
        pd.concat(counts, ignore_index=True)
        .groupby(group_columns, sort=True, dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    summary = _add_screen_metric_columns(summary)
    return summary.sort_values(group_columns[:2]).reset_index(drop=True)


def _build_voltage_screen_summary(entity_summary: pd.DataFrame) -> pd.DataFrame:
    system = _build_grouped_screen_summary(
        entity_summary,
        group_columns=["variable"],
        scope="system",
        group_value_column="bus_type",
        group_value=float("nan"),
    )
    bus_type = _build_grouped_screen_summary(
        entity_summary,
        group_columns=["variable", "bus_type"],
        scope="bus_type",
        group_value_column="bus_type",
    )
    return pd.concat([system, bus_type], ignore_index=True)


def _build_line_screen_summary(entity_summary: pd.DataFrame) -> pd.DataFrame:
    system = _build_grouped_screen_summary(
        entity_summary,
        group_columns=["variable"],
        scope="system",
        group_value_column="rate_a",
        group_value=float("nan"),
    )
    rate_a = _build_grouped_screen_summary(
        entity_summary,
        group_columns=["variable", "rate_a"],
        scope="rate_a",
        group_value_column="rate_a",
    )
    if not system.empty:
        system["rate_a"] = system["rate_a"].astype(float)
    if not rate_a.empty:
        rate_a["rate_a"] = rate_a["rate_a"].astype(float)
    return pd.concat([system, rate_a], ignore_index=True)


def _build_grouped_screen_summary(
    entity_summary: pd.DataFrame,
    *,
    group_columns: list[str],
    scope: str,
    group_value_column: str,
    group_value: object | None = None,
) -> pd.DataFrame:
    """Screening counts summed over the entities of each group (one row per entity)."""

    records: list[dict[str, object]] = []
    grouped = entity_summary.groupby(group_columns, sort=True, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record: dict[str, object] = {
            "scope": scope,
            "variable": str(group_key[0]),
            "n_entities": len(group),
            "n_samples": int(group["n_samples"].sum()),
        }
        if len(group_columns) > 1:
            record[group_value_column] = group_key[1]
        else:
            record[group_value_column] = group_value
        for column in _screen_count_columns():
            record[column] = int(group[column].sum())
        records.append(record)
    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary
    summary = _add_screen_metric_columns(summary)
    sort_columns = ["scope", "variable"]
    if group_value_column in summary.columns:
        sort_columns.append(group_value_column)
    return summary.sort_values(sort_columns, na_position="first").reset_index(drop=True)


def _screen_count_columns() -> list[str]:
    columns: list[str] = []
    for side in ("lower", "upper", "any"):
        for suffix in ("tp", "fp", "fn", "tn"):
            columns.append(f"{side}_{suffix}")
    return columns


def _add_screen_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for side in ("lower", "upper", "any"):
        tp = frame[f"{side}_tp"].to_numpy(dtype=float, copy=False)
        fp = frame[f"{side}_fp"].to_numpy(dtype=float, copy=False)
        fn = frame[f"{side}_fn"].to_numpy(dtype=float, copy=False)
        frame[f"{side}_precision"] = _safe_ratio_array(tp, tp + fp)
        frame[f"{side}_recall"] = _safe_ratio_array(tp, tp + fn)
    return frame


def _safe_ratio_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=float)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _save_screen_figures(
    entity_summary: pd.DataFrame,
    summary_frame: pd.DataFrame,
    figures_dir: Path,
    *,
    target_kind: str,
) -> None:
    entity_column = "bus" if target_kind == "volt" else "branch"
    for variable, group in entity_summary.groupby("variable", sort=True):
        entity_values = group[entity_column].to_numpy(dtype=int, copy=True)
        for metric in ("precision", "recall"):
            fig, ax = plt.subplots(figsize=(10, 4))
            for side in ("lower", "upper", "any"):
                values = group[f"{side}_{metric}"].to_numpy(dtype=float, copy=True)
                ax.plot(entity_values, values, label=side)
            ax.set_ylim(0.0, 1.05)
            ax.set_title(f"{variable} {metric} by {entity_column}")
            ax.set_xlabel(entity_column)
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                figures_dir
                / f"{variable.lower()}_{metric}_{entity_column}_summary.png",
                dpi=150,
            )
            plt.close(fig)

            finite_values = (
                group[f"any_{metric}"].dropna().to_numpy(dtype=float, copy=True)
            )
            if finite_values.size > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                bins = min(max(len(finite_values) // 2, 5), 20)
                ax.hist(finite_values, bins=bins, density=True, alpha=0.75)
                ax.set_title(f"{variable} any-violation {metric} density")
                ax.set_xlabel(metric)
                ax.set_ylabel("density")
                fig.tight_layout()
                fig.savefig(
                    figures_dir / f"{variable.lower()}_any_{metric}_density.png",
                    dpi=150,
                )
                plt.close(fig)

        system_row = summary_frame[
            summary_frame["scope"].eq("system") & summary_frame["variable"].eq(variable)
        ]
        if system_row.empty:
            continue
        row = system_row.iloc[0]
        for side in ("lower", "upper", "any"):
            matrix = np.array(
                [
                    [float(row[f"{side}_tp"]), float(row[f"{side}_fp"])],
                    [float(row[f"{side}_fn"]), float(row[f"{side}_tn"])],
                ],
                dtype=float,
            )
            fig, ax = plt.subplots(figsize=(4, 4))
            image = ax.imshow(matrix, cmap="Blues")
            ax.set_xticks([0, 1], labels=["Pred +", "Pred -"])
            ax.set_yticks([0, 1], labels=["True +", "True -"])
            ax.set_title(f"{variable} {side} confusion")
            for row_index in range(2):
                for column_index in range(2):
                    ax.text(
                        column_index,
                        row_index,
                        str(int(matrix[row_index, column_index])),
                        ha="center",
                        va="center",
                    )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(
                figures_dir / f"{variable.lower()}_{side}_confusion.png", dpi=150
            )
            plt.close(fig)


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
    ground_truth: ChannelSet,
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
        frame[f"{channel_name}_target"] = ground_truth.channels[channel_name].reshape(
            -1
        )
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


def _finalize_analysis_frame(frame: pd.DataFrame, entity_kind: str) -> pd.DataFrame:
    sort_columns = ["variable", "analysis_scope", "stratification", "stratum_value"]
    sort_columns.append("bus" if entity_kind == "bus" else "branch")
    return frame.sort_values(sort_columns, na_position="first").reset_index(drop=True)


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


def _compute_prediction_bin_edges(
    predictions: ChannelSet,
    n_bins: int,
) -> dict[str, np.ndarray]:
    return {
        channel_name: _compute_system_load_bin_edges(
            _flatten_prediction_bin_values(
                channel_values,
                predictions.valid_masks.get(channel_name),
                channel_name=channel_name,
            ),
            n_bins,
        )
        for channel_name, channel_values in predictions.channels.items()
    }


def _flatten_prediction_bin_values(
    prediction_values: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    channel_name: str,
) -> np.ndarray:
    values = np.asarray(prediction_values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if valid_mask is not None:
        finite_mask &= valid_mask
    flattened = values[finite_mask]
    if flattened.size == 0:
        raise ValueError(
            f"pred_bins stratification requires at least one finite prediction value for {channel_name}."
        )
    return flattened.astype(np.float32, copy=False)


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
) -> np.ndarray:
    finite_load = np.asarray(system_load, dtype=np.float32)
    finite_load = finite_load[np.isfinite(finite_load)]
    if finite_load.size == 0:
        raise ValueError(
            "Unable to derive system_load_bin strata because bus demand column 'Pd' is missing or invalid."
        )
    quantiles = np.linspace(0.0, 1.0, num=int(n_bins) + 1)
    edges = np.quantile(finite_load, quantiles).astype(np.float32, copy=False)
    interior_edges = np.unique(edges[1:-1])
    assigned = _assign_system_load_bins(finite_load, interior_edges)
    if np.unique(assigned).size < 2:
        raise ValueError(
            "system_load_bin stratification requires at least two distinct percentile bins."
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


def _default_artifact_name(
    predictions_path: str,
    cp_type: str,
    variables: tuple[str, ...],
) -> str:
    base_name = Path(predictions_path).stem
    variable_suffix = "_".join(variable.lower() for variable in variables)
    return f"{base_name}_{variable_suffix}.trustlayer"


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
    """Additive per-entity statistics of one block of calibrated rows.

    One row per variable, entity and stratum (plus the overall scope), with
    sums, counts and extremes, so that blocks combine exactly in
    `_combine_analysis_partials`.
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
        work["interval_width"] = upper - lower
        work["covered"] = ((target >= lower) & (target <= upper)).astype(float)
        if entity_column == "branch":
            # Out-of-service or unrated branches carry -1 and are not analysed.
            valid = (frame[f"{variable}_pred"].to_numpy() >= 0.0) & (target >= 0.0)
            work = work[valid]
        parts.append(
            _entity_statistics(
                work,
                [entity_column],
                static_columns,
                variable=variable,
                analysis_scope="overall",
                stratification="",
            )
        )
        for stratification in strat_columns:
            statistics = _entity_statistics(
                work,
                [stratification, entity_column],
                static_columns,
                variable=variable,
                analysis_scope="stratified",
                stratification=stratification,
            )
            statistics["stratum_value"] = statistics.pop(stratification)
            statistics["stratum_label"] = [
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
    analysis_scope: str,
    stratification: str,
) -> pd.DataFrame:
    statistics = (
        work.groupby(group_columns, sort=True, dropna=False)
        .agg(
            width_sum=("interval_width", "sum"),
            width_count=("interval_width", "count"),
            width_min=("interval_width", "min"),
            width_max=("interval_width", "max"),
            covered_sum=("covered", "sum"),
            n_samples=("covered", "size"),
            n_scenarios=("scenario", "nunique"),
            **{column: (column, "first") for column in static_columns},
        )
        .reset_index()
    )
    statistics.insert(0, "analysis_scope", analysis_scope)
    statistics.insert(1, "stratification", stratification)
    statistics.insert(2, "variable", variable)
    if not stratification:
        statistics["stratum_value"] = float("nan")
        statistics["stratum_label"] = _format_stratum_label("", None)
    return statistics


def _stratum_label(work: pd.DataFrame, stratification: str, value: object) -> str:
    stratum_value = None if pd.isna(value) else int(cast(float, value))
    label_column = f"{stratification}_label"
    if stratum_value is not None and label_column in work.columns:
        labels = work.loc[work[stratification] == value, label_column].dropna()
        if not labels.empty:
            return f"{stratification}={labels.iloc[0]}"
    return _format_stratum_label(stratification, stratum_value)


def _combine_analysis_partials(
    partials: list[pd.DataFrame],
    *,
    entity_column: str,
    bus_types: dict[int, str],
) -> pd.DataFrame:
    """Per-entity analysis rows (`trust_analysis_*.csv`) from the blocks' statistics."""

    static_columns = (
        ["from_bus", "to_bus", "rate_a"] if entity_column == "branch" else []
    )
    keys = [
        "analysis_scope",
        "stratification",
        "stratum_value",
        "variable",
        entity_column,
    ]
    combined = (
        pd.concat(partials, ignore_index=True)
        .groupby(keys, sort=False, dropna=False)
        .agg(
            stratum_label=("stratum_label", "first"),
            width_sum=("width_sum", "sum"),
            width_count=("width_count", "sum"),
            width_min=("width_min", "min"),
            width_max=("width_max", "max"),
            covered_sum=("covered_sum", "sum"),
            n_samples=("n_samples", "sum"),
            n_scenarios=("n_scenarios", "sum"),
            **{column: (column, "first") for column in static_columns},
        )
        .reset_index()
    )
    width_count = combined["width_count"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        width_mean = np.where(
            width_count > 0,
            combined["width_sum"].to_numpy(dtype=float) / width_count,
            np.nan,
        )
    analysis = combined.loc[
        :,
        [
            "analysis_scope",
            "stratification",
            "stratum_value",
            "stratum_label",
            "variable",
            entity_column,
        ],
    ].copy()
    analysis[entity_column] = analysis[entity_column].astype(np.int64)
    if entity_column == "branch":
        analysis["from_bus"] = combined["from_bus"].astype(np.int64)
        analysis["to_bus"] = combined["to_bus"].astype(np.int64)
        analysis["rate_a"] = combined["rate_a"].astype(float)
    else:
        analysis["bus_type"] = [
            bus_types.get(int(bus), "Unknown") for bus in analysis["bus"].tolist()
        ]
    analysis["width_mean"] = width_mean
    analysis["width_min"] = combined["width_min"].astype(float)
    analysis["width_max"] = combined["width_max"].astype(float)
    if entity_column == "bus":
        is_angle = (analysis["variable"] == "Va").to_numpy()
        for name in ("width_mean", "width_min", "width_max"):
            analysis[f"{name}_degrees"] = np.where(
                is_angle, np.rad2deg(analysis[name].to_numpy(dtype=float)), np.nan
            )
    analysis["empirical_coverage"] = combined["covered_sum"].to_numpy(
        dtype=float
    ) / combined["n_samples"].to_numpy(dtype=float)
    analysis["n_samples"] = combined["n_samples"].astype(np.int64)
    analysis["n_scenarios"] = combined["n_scenarios"].astype(np.int64)
    return _finalize_analysis_frame(analysis, entity_column)


def _analysis_scenario_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    return _ordered_analysis_strat_columns(frame.columns)


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


def _analysis_strat_columns_for_variable(
    frame: pd.DataFrame,
    variable: str,
) -> list[str]:
    return [
        column
        for column in _analysis_scenario_columns_from_frame(frame)
        if _stratification_applies_to_variable(column, variable)
    ]


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


def _entity_column_map(channel_set: ChannelSet, column_name: str) -> dict[int, object]:
    if column_name not in channel_set.entity_columns:
        return {}
    values = channel_set.entity_columns[column_name]
    return {
        int(entity_id): values[index]
        for index, entity_id in enumerate(channel_set.entity_ids.tolist())
    }


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


def _maybe_width_degrees(variable: str, width_value: float) -> float:
    if variable != "Va":
        return float("nan")
    return float(np.rad2deg(float(width_value)))


def _save_analysis_figures(
    analysis_frame: pd.DataFrame,
    figures_dir: Path,
) -> None:
    overall_frame = analysis_frame[analysis_frame["analysis_scope"].eq("overall")]
    for variable, group in overall_frame.groupby("variable", sort=True):
        bus_values = group["bus"].to_numpy(dtype=int, copy=True)
        width_mean = group["width_mean"].to_numpy(dtype=float, copy=True)
        width_min = group["width_min"].to_numpy(dtype=float, copy=True)
        width_max = group["width_max"].to_numpy(dtype=float, copy=True)
        empirical_coverage = group["empirical_coverage"].to_numpy(
            dtype=float, copy=True
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(bus_values, width_mean, label="mean")
        ax.plot(bus_values, width_min, label="min")
        ax.plot(bus_values, width_max, label="max")
        ax.set_title(f"{variable} interval width by bus")
        ax.set_xlabel("bus")
        ax.set_ylabel("interval width")
        _style_axes(ax)
        _place_legend(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_width_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(bus_values, empirical_coverage, label="coverage")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{variable} empirical coverage by bus")
        ax.set_xlabel("bus")
        ax.set_ylabel("coverage")
        _style_axes(ax)
        _place_legend(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_coverage_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_density_curve(ax, width_mean, show_mean=True)
        ax.set_title(f"{variable} mean interval width density")
        ax.set_xlabel("mean interval width")
        ax.set_ylabel("density")
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_width_density.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_density_curve(ax, empirical_coverage, clip=(0.0, 1.0), show_mean=True)
        ax.set_title(f"{variable} empirical coverage density")
        ax.set_xlabel("empirical coverage")
        ax.set_ylabel("density")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, num=6))
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_coverage_density.png", dpi=150)
        plt.close(fig)

    _save_stratified_entity_figures(
        analysis_frame,
        figures_dir,
        entity_column="bus",
        entity_label="bus",
    )


def _save_line_analysis_figures(
    analysis_frame: pd.DataFrame,
    figures_dir: Path,
) -> None:
    overall_frame = analysis_frame[analysis_frame["analysis_scope"].eq("overall")]
    for variable, group in overall_frame.groupby("variable", sort=True):
        branch_values = group["branch"].to_numpy(dtype=int, copy=True)
        width_mean = group["width_mean"].to_numpy(dtype=float, copy=True)
        width_min = group["width_min"].to_numpy(dtype=float, copy=True)
        width_max = group["width_max"].to_numpy(dtype=float, copy=True)
        empirical_coverage = group["empirical_coverage"].to_numpy(
            dtype=float, copy=True
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(branch_values, width_mean, label="mean")
        ax.plot(branch_values, width_min, label="min")
        ax.plot(branch_values, width_max, label="max")
        ax.set_title(f"{variable} interval width by branch")
        ax.set_xlabel("branch")
        ax.set_ylabel("interval width")
        _style_axes(ax)
        _place_legend(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_width_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(branch_values, empirical_coverage, label="coverage")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{variable} empirical coverage by branch")
        ax.set_xlabel("branch")
        ax.set_ylabel("coverage")
        _style_axes(ax)
        _place_legend(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_coverage_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_density_curve(ax, width_mean, show_mean=True)
        ax.set_title(f"{variable} mean interval width density")
        ax.set_xlabel("mean interval width")
        ax.set_ylabel("density")
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_width_density.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_density_curve(ax, empirical_coverage, clip=(0.0, 1.0), show_mean=True)
        ax.set_title(f"{variable} empirical coverage density")
        ax.set_xlabel("empirical coverage")
        ax.set_ylabel("density")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, num=6))
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_coverage_density.png", dpi=150)
        plt.close(fig)

    _save_stratified_entity_figures(
        analysis_frame,
        figures_dir,
        entity_column="branch",
        entity_label="branch",
    )


def _save_stratified_entity_figures(
    analysis_frame: pd.DataFrame,
    figures_dir: Path,
    *,
    entity_column: str,
    entity_label: str,
) -> None:
    stratified_frame = analysis_frame[analysis_frame["analysis_scope"].eq("stratified")]
    if stratified_frame.empty:
        return

    for (variable, stratification), group in stratified_frame.groupby(
        ["variable", "stratification"], sort=True, dropna=False
    ):
        suffix = _normalize_output_segment(f"by_{stratification}")
        ordered_group = group.sort_values(
            ["stratum_value", entity_column], na_position="first"
        )

        legend_outside = ordered_group["stratum_value"].nunique(dropna=False) > 3
        fig, ax = plt.subplots(figsize=(10, 4))
        for stratum_value, stratum_group in ordered_group.groupby(
            "stratum_value", sort=True, dropna=False
        ):
            stratum_label = _format_figure_stratum_label(
                _resolve_stratum_label_from_frame(
                    stratum_group,
                    str(stratification),
                    None if pd.isna(stratum_value) else int(stratum_value),
                ),
                str(stratification),
            )
            ax.plot(
                stratum_group[entity_column].to_numpy(dtype=int, copy=True),
                stratum_group["width_mean"].to_numpy(dtype=float, copy=True),
                label=stratum_label,
            )
        ax.set_title(
            f"{variable} mean interval width by {entity_label} and {stratification}"
        )
        ax.set_xlabel(entity_label)
        ax.set_ylabel("mean interval width")
        _style_axes(ax)
        _place_legend(ax, outside=legend_outside)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_width_summary_{suffix}.png", dpi=150
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        for stratum_value, stratum_group in ordered_group.groupby(
            "stratum_value", sort=True, dropna=False
        ):
            stratum_label = _format_figure_stratum_label(
                _resolve_stratum_label_from_frame(
                    stratum_group,
                    str(stratification),
                    None if pd.isna(stratum_value) else int(stratum_value),
                ),
                str(stratification),
            )
            ax.plot(
                stratum_group[entity_column].to_numpy(dtype=int, copy=True),
                stratum_group["empirical_coverage"].to_numpy(dtype=float, copy=True),
                label=stratum_label,
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"{variable} empirical coverage by {entity_label} and {stratification}"
        )
        ax.set_xlabel(entity_label)
        ax.set_ylabel("coverage")
        _style_axes(ax)
        _place_legend(ax, outside=legend_outside)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_coverage_summary_{suffix}.png", dpi=150
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for stratum_value, stratum_group in ordered_group.groupby(
            "stratum_value", sort=True, dropna=False
        ):
            stratum_label = _format_figure_stratum_label(
                _resolve_stratum_label_from_frame(
                    stratum_group,
                    str(stratification),
                    None if pd.isna(stratum_value) else int(stratum_value),
                ),
                str(stratification),
            )
            _plot_density_curve(
                ax,
                stratum_group["width_mean"].to_numpy(dtype=float, copy=True),
                label=stratum_label,
                fill_alpha=0.22,
                show_mean=True,
            )
        ax.set_title(f"{variable} mean interval width density by {stratification}")
        ax.set_xlabel("mean interval width")
        ax.set_ylabel("density")
        _style_axes(ax)
        _place_legend(ax, outside=legend_outside)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_width_density_{suffix}.png", dpi=150
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for stratum_value, stratum_group in ordered_group.groupby(
            "stratum_value", sort=True, dropna=False
        ):
            stratum_label = _format_figure_stratum_label(
                _resolve_stratum_label_from_frame(
                    stratum_group,
                    str(stratification),
                    None if pd.isna(stratum_value) else int(stratum_value),
                ),
                str(stratification),
            )
            _plot_density_curve(
                ax,
                stratum_group["empirical_coverage"].to_numpy(dtype=float, copy=True),
                label=stratum_label,
                clip=(0.0, 1.0),
                fill_alpha=0.22,
                show_mean=True,
            )
        ax.set_title(f"{variable} empirical coverage density by {stratification}")
        ax.set_xlabel("empirical coverage")
        ax.set_ylabel("density")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, num=6))
        _style_axes(ax)
        _place_legend(ax, outside=legend_outside)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_coverage_density_{suffix}.png", dpi=150
        )
        plt.close(fig)

        per_stratum = (
            ordered_group.groupby("stratum_value", sort=True, dropna=False)
            .agg(
                stratum_label=("stratum_label", "first"),
                width_mean_across_entities=("width_mean", "mean"),
                empirical_coverage_across_entities=("empirical_coverage", "mean"),
                n_entities=(entity_column, "nunique"),
            )
            .reset_index()
        )
        labels = [
            _format_figure_stratum_label(str(value), str(stratification))
            for value in per_stratum["stratum_label"].astype(str).tolist()
        ]

        fig_height = max(4.5, 1.0 + 0.7 * len(labels))
        width_distributions = [
            ordered_group.loc[
                ordered_group["stratum_label"].eq(raw_label),
                "width_mean",
            ]
            .dropna()
            .to_numpy(dtype=float, copy=True)
            for raw_label in per_stratum["stratum_label"].astype(str).tolist()
        ]
        fig, ax = plt.subplots(figsize=(10, fig_height))
        positions = np.arange(1, len(labels) + 1, dtype=float)
        width_box = _horizontal_boxplot(
            ax,
            width_distributions,
            positions=positions,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#0f4c81",
                "markeredgecolor": "#0f4c81",
                "markersize": 4,
            },
        )
        for patch in width_box["boxes"]:
            patch.set_facecolor("#2f7db2")
            patch.set_alpha(0.45)
        ax.set_yticks(positions, labels=labels)
        ax.set_title(
            f"{variable} mean interval width distribution across {entity_label}s by {stratification}"
        )
        ax.set_xlabel("mean interval width")
        ax.set_ylabel(stratification)
        _style_axes(ax, axis="x")
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_width_mean_{suffix}.png", dpi=150
        )
        plt.close(fig)

        coverage_distributions = [
            ordered_group.loc[
                ordered_group["stratum_label"].eq(raw_label),
                "empirical_coverage",
            ]
            .dropna()
            .to_numpy(dtype=float, copy=True)
            for raw_label in per_stratum["stratum_label"].astype(str).tolist()
        ]
        fig, ax = plt.subplots(figsize=(10, fig_height))
        coverage_box = _horizontal_boxplot(
            ax,
            coverage_distributions,
            positions=positions,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#1f6f4a",
                "markeredgecolor": "#1f6f4a",
                "markersize": 4,
            },
        )
        for patch in coverage_box["boxes"]:
            patch.set_facecolor("#4f9f7a")
            patch.set_alpha(0.45)
        ax.set_yticks(positions, labels=labels)
        ax.set_title(
            f"{variable} empirical coverage distribution across {entity_label}s by {stratification}"
        )
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("coverage")
        ax.set_ylabel(stratification)
        ax.set_xticks(np.linspace(0.0, 1.0, num=6))
        _style_axes(ax, axis="x")
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_coverage_mean_{suffix}.png", dpi=150
        )
        plt.close(fig)


def _build_summary_frame(analysis_frame: pd.DataFrame) -> pd.DataFrame:
    system_summary = _aggregate_summary_rows(
        analysis_frame,
        group_columns=["analysis_scope", "stratification", "stratum_value", "variable"],
        scope="system",
        include_bus_type=False,
    )
    bus_type_summary = _aggregate_summary_rows(
        analysis_frame,
        group_columns=[
            "analysis_scope",
            "stratification",
            "stratum_value",
            "variable",
            "bus_type",
        ],
        scope="bus_type",
        include_bus_type=True,
    )
    return pd.concat([system_summary, bus_type_summary], ignore_index=True)


def _build_line_summary_frame(analysis_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = analysis_frame.groupby(
        ["analysis_scope", "stratification", "stratum_value", "variable"],
        sort=True,
        dropna=False,
    )
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        analysis_scope = str(group_key[0])
        stratification = str(group_key[1])
        stratum_value = group_key[2]
        variable = str(group_key[3])
        records.append(
            {
                "analysis_scope": analysis_scope,
                "stratification": stratification,
                "stratum_value": stratum_value,
                "stratum_label": _resolve_stratum_label_from_group(
                    group,
                    stratification,
                    stratum_value,
                ),
                "scope": "system",
                "variable": variable,
                "rate_a": float("nan"),
                "n_lines": int(group["branch"].nunique()),
                "n_samples_min": int(group["n_samples"].min()),
                "n_samples_max": int(group["n_samples"].max()),
                "n_scenarios_min": int(group["n_scenarios"].min()),
                "n_scenarios_max": int(group["n_scenarios"].max()),
                "width_mean_across_lines": float(group["width_mean"].mean()),
                "width_min_across_lines": float(group["width_mean"].min()),
                "width_max_across_lines": float(group["width_mean"].max()),
                "empirical_coverage_mean_across_lines": float(
                    group["empirical_coverage"].mean()
                ),
                "empirical_coverage_min_across_lines": float(
                    group["empirical_coverage"].min()
                ),
                "empirical_coverage_max_across_lines": float(
                    group["empirical_coverage"].max()
                ),
            },
        )
        grouped_by_rate = group.groupby("rate_a", sort=True, dropna=False)
        for rate_a, rate_group in grouped_by_rate:
            records.append(
                {
                    "analysis_scope": analysis_scope,
                    "stratification": stratification,
                    "stratum_value": stratum_value,
                    "stratum_label": _resolve_stratum_label_from_group(
                        rate_group,
                        stratification,
                        stratum_value,
                    ),
                    "scope": "rate_a",
                    "variable": variable,
                    "rate_a": float(rate_a),
                    "n_lines": int(rate_group["branch"].nunique()),
                    "n_samples_min": int(rate_group["n_samples"].min()),
                    "n_samples_max": int(rate_group["n_samples"].max()),
                    "n_scenarios_min": int(rate_group["n_scenarios"].min()),
                    "n_scenarios_max": int(rate_group["n_scenarios"].max()),
                    "width_mean_across_lines": float(rate_group["width_mean"].mean()),
                    "width_min_across_lines": float(rate_group["width_mean"].min()),
                    "width_max_across_lines": float(rate_group["width_mean"].max()),
                    "empirical_coverage_mean_across_lines": float(
                        rate_group["empirical_coverage"].mean()
                    ),
                    "empirical_coverage_min_across_lines": float(
                        rate_group["empirical_coverage"].min()
                    ),
                    "empirical_coverage_max_across_lines": float(
                        rate_group["empirical_coverage"].max()
                    ),
                },
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            [
                "analysis_scope",
                "stratification",
                "stratum_value",
                "scope",
                "variable",
                "rate_a",
            ],
            na_position="first",
        )
        .reset_index(drop=True)
    )


def _build_reliability_frame(
    analysis_frame: pd.DataFrame,
    *,
    entity_column: str,
    target_coverage: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = analysis_frame.groupby(
        ["analysis_scope", "stratification", "stratum_value", "variable"],
        sort=True,
        dropna=False,
    )
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        analysis_scope = str(group_key[0])
        stratification = str(group_key[1])
        stratum_value = group_key[2]
        variable = str(group_key[3])
        weights = group["n_samples"].to_numpy(dtype=float, copy=True)
        empirical = group["empirical_coverage"].to_numpy(dtype=float, copy=True)
        widths = group["width_mean"].to_numpy(dtype=float, copy=True)
        empirical_weighted = _weighted_average(empirical, weights)
        width_weighted = _weighted_average(widths, weights)
        records.append(
            {
                "analysis_scope": analysis_scope,
                "stratification": stratification,
                "stratum_value": stratum_value,
                "stratum_label": _resolve_stratum_label_from_group(
                    group,
                    stratification,
                    stratum_value,
                ),
                "variable": variable,
                "target_coverage": float(target_coverage),
                "n_entities": int(group[entity_column].nunique()),
                "n_samples_total": int(group["n_samples"].sum()),
                "n_scenarios_min": int(group["n_scenarios"].min()),
                "n_scenarios_max": int(group["n_scenarios"].max()),
                "empirical_coverage_mean_across_entities": float(
                    group["empirical_coverage"].mean()
                ),
                "empirical_coverage_weighted": empirical_weighted,
                "coverage_gap_weighted": empirical_weighted - float(target_coverage),
                "width_mean_across_entities": float(group["width_mean"].mean()),
                "width_mean_weighted": width_weighted,
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["variable", "analysis_scope", "stratification", "stratum_value"],
            na_position="first",
        )
        .reset_index(drop=True)
    )


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    valid_mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid_mask):
        return float("nan")
    return float(np.average(values[valid_mask], weights=weights[valid_mask]))


def _aggregate_summary_rows(
    analysis_frame: pd.DataFrame,
    *,
    group_columns: list[str],
    scope: str,
    include_bus_type: bool,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = analysis_frame.groupby(group_columns, sort=True, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        analysis_scope = str(group_key[0])
        stratification = str(group_key[1])
        stratum_value = group_key[2]
        variable = str(group_key[3])
        bus_type = str(group_key[4]) if include_bus_type else "All"
        records.append(
            {
                "analysis_scope": analysis_scope,
                "stratification": stratification,
                "stratum_value": stratum_value,
                "stratum_label": _resolve_stratum_label_from_group(
                    group,
                    stratification,
                    stratum_value,
                ),
                "scope": scope,
                "variable": variable,
                "bus_type": bus_type,
                "n_buses": int(group["bus"].nunique()),
                "n_samples_min": int(group["n_samples"].min()),
                "n_samples_max": int(group["n_samples"].max()),
                "n_scenarios_min": int(group["n_scenarios"].min()),
                "n_scenarios_max": int(group["n_scenarios"].max()),
                "width_mean_across_buses": float(group["width_mean"].mean()),
                "width_min_across_buses": float(group["width_mean"].min()),
                "width_max_across_buses": float(group["width_mean"].max()),
                "empirical_coverage_mean_across_buses": float(
                    group["empirical_coverage"].mean()
                ),
                "empirical_coverage_min_across_buses": float(
                    group["empirical_coverage"].min()
                ),
                "empirical_coverage_max_across_buses": float(
                    group["empirical_coverage"].max()
                ),
                "width_mean_degrees_across_buses": _summary_or_nan(
                    group["width_mean_degrees"]
                ),
                "width_min_degrees_across_buses": _summary_or_nan(
                    group["width_mean_degrees"], op="min"
                ),
                "width_max_degrees_across_buses": _summary_or_nan(
                    group["width_mean_degrees"], op="max"
                ),
            },
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            [
                "analysis_scope",
                "stratification",
                "stratum_value",
                "scope",
                "variable",
                "bus_type",
            ],
            na_position="first",
        )
        .reset_index(drop=True)
    )


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


def _resolve_stratum_label_from_frame(
    frame: pd.DataFrame,
    stratification: str,
    stratum_value: int | None,
) -> str:
    if not stratification:
        return _format_stratum_label(stratification, stratum_value)
    label_column = f"{stratification}_label"
    if label_column in frame.columns:
        labels = frame[label_column].dropna().astype(str).unique().tolist()
        if labels:
            return f"{stratification}={labels[0]}"
    if "stratum_label" in frame.columns:
        labels = frame["stratum_label"].dropna().astype(str).unique().tolist()
        if labels:
            return labels[0]
    return _format_stratum_label(stratification, stratum_value)


def _resolve_stratum_label_from_group(
    group: pd.DataFrame,
    stratification: str,
    stratum_value: object,
) -> str:
    if "stratum_label" in group.columns:
        labels = group["stratum_label"].dropna().astype(str).unique().tolist()
        if labels:
            return labels[0]
    return _format_stratum_label(stratification, stratum_value)


def _format_figure_stratum_label(label: str, stratification: str) -> str:
    if not stratification:
        return label
    prefix = f"{stratification}="
    if label.startswith(prefix):
        return label[len(prefix) :]
    return label


def _plot_density_curve(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    label: str | None = None,
    clip: tuple[float, float] | None = None,
    fill_alpha: float = 0.28,
    show_mean: bool = False,
) -> None:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if clip is not None:
        lower_clip, upper_clip = clip
        finite_values = finite_values[
            (finite_values >= lower_clip) & (finite_values <= upper_clip)
        ]
    if finite_values.size == 0:
        return

    unique_values = np.unique(finite_values)
    if unique_values.size < 2:
        line = ax.axvline(
            float(unique_values[0]),
            label=label,
            linewidth=2.0,
            alpha=0.95,
        )
        ax.axvspan(
            float(unique_values[0]),
            float(unique_values[0]),
            color=line.get_color(),
            alpha=fill_alpha,
        )
        if show_mean:
            ax.axvline(
                float(unique_values[0]),
                color=line.get_color(),
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
            )
        return

    value_min = float(finite_values.min())
    value_max = float(finite_values.max())
    span = max(value_max - value_min, 1e-6)
    if clip is None:
        grid_min = value_min - 0.1 * span
        grid_max = value_max + 0.1 * span
    else:
        clip_min, clip_max = clip
        padding = 0.02 * (clip_max - clip_min)
        grid_min = max(clip_min, value_min - padding)
        grid_max = min(clip_max, value_max + padding)

    if np.isclose(grid_min, grid_max):
        grid_min -= 0.5
        grid_max += 0.5

    grid = np.linspace(grid_min, grid_max, num=256)
    density = gaussian_kde(finite_values)(grid)
    (line,) = ax.plot(grid, density, linewidth=2.0, label=label)
    ax.fill_between(grid, 0.0, density, color=line.get_color(), alpha=fill_alpha)
    if show_mean:
        ax.axvline(
            float(finite_values.mean()),
            color=line.get_color(),
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )


def _style_axes(ax: plt.Axes, axis: Literal["both", "x", "y"] = "both") -> None:
    ax.grid(True, axis=axis, linestyle="--", linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _place_legend(ax: plt.Axes, *, outside: bool = False) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        return
    ax.legend(frameon=True)


def _annotate_horizontal_bars(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    *,
    fmt: str = "{:.4f}",
) -> None:
    max_value = float(np.nanmax(values)) if values.size else 0.0
    x_min, x_max = ax.get_xlim()
    x_span = max(float(x_max - x_min), 1e-12)
    # Keep label offsets proportional to the current axis scale.
    x_padding = max(0.01 * max_value, 0.01 * x_span)
    x_padding = min(x_padding, 0.05 * x_span)
    for bar, value in zip(bars, values.tolist(), strict=False):
        ax.text(
            float(bar.get_width()) + x_padding,
            float(bar.get_y()) + float(bar.get_height()) / 2.0,
            fmt.format(float(value)),
            va="center",
            ha="left",
            fontsize=9,
        )


def _horizontal_boxplot(
    ax: plt.Axes,
    data: list[np.ndarray],
    *,
    positions: np.ndarray,
    meanprops: dict[str, object],
):
    # A stratum with too few calibration samples has infinite bounds (and so
    # infinite widths), which a box plot cannot draw: its percentiles turn into
    # NaN. Plot the finite values; the CSV outputs keep the infinite entries.
    data = [np.asarray(values)[np.isfinite(values)] for values in data]
    try:
        return ax.boxplot(
            data,
            orientation="horizontal",
            positions=positions,
            patch_artist=True,
            showmeans=True,
            meanprops=meanprops,
        )
    except TypeError:
        return ax.boxplot(
            data,
            vert=False,
            positions=positions,
            patch_artist=True,
            showmeans=True,
            meanprops=meanprops,
        )


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


def _summary_or_nan(values: pd.Series, op: str = "mean") -> float:
    non_na = values.dropna()
    if non_na.empty:
        return float("nan")
    if op == "min":
        return float(non_na.min())
    if op == "max":
        return float(non_na.max())
    return float(non_na.mean())


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
