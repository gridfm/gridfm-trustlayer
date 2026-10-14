"""Command-line interface: `compute` fits the layer on calibration data, `calibrate`
applies it to test data, and `screen` compares the intervals with operating limits."""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]

from gridfm_trustlayer._version import __version__
from gridfm_trustlayer.cli import reports
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
from gridfm_trustlayer.conformal.calibrator import (
    DEFAULT_SPARSE_STRATA,
    SPARSE_STRATA_CHOICES,
    validate_sparse_strata,
)
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
# Test embeddings are read and pooled once when the file is small enough to
# decode at once (e.g. one row per scenario); larger ones, such as per-bus
# embeddings of a big grid, are read again for each block.
_EMBEDDINGS_PRELOAD_BYTES = 1 << 30
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
    if args.command == "apply":
        _run_apply(args)
        return 0
    if args.command == "screen":
        _run_screen(args)
        return 0
    raise ValueError(f"Unsupported command: {args.command!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="gridfm_trustlayer",
        description=(
            "Conformal trustworthiness layer for GridFM foundation models: "
            "compute, calibrate, then screen; apply gives bounds for new scenarios."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
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
        "--sparse-strata",
        choices=SPARSE_STRATA_CHOICES,
        help="Bounds of an element in a stratum with too few calibration samples: 'pooled' (default, its quantile over all strata) or 'infinite'.",
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

    apply_parser = subparsers.add_parser(
        "apply",
        help="Write bounds for new scenarios (inputs.apply), which need no known AC solution.",
    )
    apply_parser.add_argument(
        "--config",
        required=True,
        help="Required trustlayer YAML config file. CLI flags override config values.",
    )
    apply_parser.add_argument(
        "--predictions", help="Predictions parquet of the new scenarios."
    )
    apply_parser.add_argument(
        "--artifact", help="Previously computed .trustlayer artifact."
    )
    apply_parser.add_argument(
        "--output",
        help="Output dataset path for one target family, or directory for both.",
    )
    apply_parser.add_argument(
        "--pf-node",
        help="Bus file describing the new scenarios (loads, bus types); needed for "
        "strat n_outages, outage_type or system_load_bin, and for line_loading.",
    )
    apply_parser.add_argument(
        "--pf-edge",
        help="Branch file describing the new scenarios (status, parameters, ratings); "
        "needed with --pf-node.",
    )
    apply_parser.add_argument(
        "--embeddings",
        help="Scenario embeddings parquet of the new scenarios (kernel artifacts).",
    )
    apply_parser.add_argument(
        "--task-name", help="Task name used to interpret prediction schema."
    )
    apply_parser.add_argument(
        "--generator-predictions",
        help="Optional generator predictions parquet for OPF artifacts.",
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
    screen_figures_group = screen_parser.add_mutually_exclusive_group()
    screen_figures_group.add_argument(
        "--figures",
        dest="save_figures",
        action="store_true",
        default=None,
        help="Write screening figures (default behavior).",
    )
    screen_figures_group.add_argument(
        "--no-figures",
        dest="save_figures",
        action="store_false",
        help="Skip screening figures and write only the tables and the detail.",
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
    sparse_strata = validate_sparse_strata(
        str(
            coalesce_option(args.sparse_strata, config, "sparse_strata")
            or DEFAULT_SPARSE_STRATA
        )
    )
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
                    sparse_strata=sparse_strata,
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
                sparse_strata=sparse_strata,
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
                sparse_strata=sparse_strata,
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
    method = cp_type if strat is None else f"{cp_type}, strat {strat}"
    print(
        f"compute: fitted {method} for {', '.join(variables)} on "
        f"{len(predictions.scenario_ids)} calibration scenarios; "
        f"artifact: {artifact_path}",
        flush=True,
    )
    _report_sparse_strata(calibrators, sparse_strata)


def _report_sparse_strata(
    calibrators: Mapping[str, NaiveIntervalCalibrator | KernelIntervalCalibrator],
    sparse_strata: str,
) -> None:
    """Which strata have elements with too few calibration samples, and what they get."""

    sparse = _sparse_strata_listing(calibrators)
    if not sparse:
        return
    outcome = (
        "they use their quantile over all calibration scenarios (sparse_strata: pooled)"
        if sparse_strata == "pooled"
        else "they get infinite bounds (sparse_strata: infinite)"
    )
    print(
        f"compute: too few calibration samples for a finite bound: {sparse}; {outcome}",
        flush=True,
    )


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
    embeddings = _EmbeddingSource.for_artifact(calibrators, args, config, scenario_ids)
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
    blocks = scenario_blocks(
        scenario_ids,
        rows_per_scenario(predictions_path, len(scenario_ids)),
        _BLOCK_ROWS,
    )
    for index, block in enumerate(blocks, start=1):
        _report_progress("calibrate", index, blocks)
        predictions = load_predictions(
            predictions_path,
            task_name=task_name,
            generator_parquet_path=generator_path,
            scenario_ids=block,
        )
        if embeddings is not None:
            embeddings.attach(predictions)
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

    data_dir = Path(_common_parent(output_paths.values()))
    coverage_dir = Path(str(config.get("coverage_dir") or data_dir / "coverage"))
    strata: list[pd.DataFrame] = []
    groups: list[pd.DataFrame] = []
    figures: list[Path] = []
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
            figures += reports.save_coverage_figures(
                by_stratum,
                elements,
                coverage_dir / "figures",
                entity_column=entity_column,
            )
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
    print(f"calibrate: data in {data_dir}; reports in {coverage_dir}", flush=True)


class _EmbeddingSource:
    """Scenario embeddings for a kernel artifact, attached block by block.

    Small files are read and pooled once; larger ones (e.g. per-bus embeddings
    of a big grid) are read again for each block, which keeps memory bounded.
    """

    def __init__(self, path: str, pooling: str, scenario_ids: np.ndarray) -> None:
        self.path = path
        self.pooling = pooling
        self.preloaded = None
        if _decoded_size_estimate(path) <= _EMBEDDINGS_PRELOAD_BYTES:
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
        return cls(str(path), next(iter(poolings)), scenario_ids)

    def attach(self, predictions: PredictionSet) -> None:
        if self.preloaded is not None:
            predictions.scenario_embeddings = self.preloaded.align_to(
                predictions.scenario_ids
            )
        else:
            _attach_scenario_embeddings(predictions, self.path, pooling=self.pooling)


# Stratifications computed from the description of each scenario: its outages
# (branch and generator status) or its total load.
_DESCRIPTION_STRATA = {"n_outages", "outage_type", "system_load_bin"}


def _run_apply(args: argparse.Namespace) -> None:
    """Bounds for new scenarios, which need no known AC solution.

    Same computation as `calibrate`, without the coverage analysis: only the
    predictions are needed, plus the description of each scenario (topology,
    branch parameters, loads) when the layer is stratified by outages or load
    or calibrates line loading, and the embeddings for `cp_type: kernel`.
    """

    config = resolve_command_config(load_cli_config(args.config), "apply")
    validate_command_config("apply", config)
    predictions_path = str(
        coalesce_option(args.predictions, config, "predictions", required=True)
    )
    artifact_path = coalesce_option(args.artifact, config, "artifact", required=True)
    calibrators = load_interval_artifacts(artifact_path)
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
        reference = scan_ground_truth_reference(*description_paths)
    has_bus_targets = any(
        calibrator.entity_name == "bus" for calibrator in calibrators.values()
    )
    embeddings = _EmbeddingSource.for_artifact(calibrators, args, config, scenario_ids)
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
    writers = {
        target_kind: PartitionedWriter(path)
        for target_kind, path in output_paths.items()
    }
    blocks = scenario_blocks(
        scenario_ids,
        rows_per_scenario(predictions_path, len(scenario_ids)),
        _BLOCK_ROWS,
    )
    for index, block in enumerate(blocks, start=1):
        _report_progress("apply", index, blocks)
        predictions = load_predictions(
            predictions_path,
            task_name=task_name,
            generator_parquet_path=generator_path,
            scenario_ids=block,
        )
        _require_set_points(predictions)
        if embeddings is not None:
            embeddings.attach(predictions)
        description = None
        if description_paths is not None:
            description = load_ground_truth(
                *description_paths,
                scenario_ids=predictions.scenario_ids,
                reference=reference,
                solved=False,
            )
        original_frame = (
            read_frame(predictions_path, scenario_ids=block)
            if has_bus_targets
            else None
        )
        for target_kind, calibrator in calibrators.items():
            writers[target_kind].write(
                _bounds_block_frame(
                    calibrator, predictions, description, original_frame
                )
            )
    variables = [name for c in calibrators.values() for name in c.channel_names]
    print(
        f"apply: bounds for {', '.join(variables)} in {len(scenario_ids)} scenarios; "
        f"outputs in {_common_parent(output_paths.values())}",
        flush=True,
    )


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
_BUS_TYPE_WORDS = {"PV": "PV buses", "REF": "the reference bus"}
# `screen` sees only the `<variable>_known` columns, not the task.
_BUS_KNOWN_WHERE = {"Vm": "PV buses", "Va": "the reference bus"}


def _known_where(task_name: str) -> dict[str, str]:
    """Where each variable has known values, in words, for the reports."""

    return {
        variable: _BUS_TYPE_WORDS[bus_type]
        for variable, bus_type in _SET_POINTS.get(task_name, ())
    }


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


def _require_set_points(predictions: PredictionSet) -> None:
    for variable, bus_type in _SET_POINTS.get(predictions.task_name, ()):
        mask = predictions.bus_type_masks.get(bus_type)
        values = predictions.Vm if variable == "Vm" else predictions.Va
        if mask is not None and not np.isfinite(values[mask]).all():
            raise ValueError(
                f"{variable}_target is missing at some {bus_type} buses: these "
                f"set-points are inputs of each {predictions.task_name} scenario and "
                "are needed to apply the layer.",
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
        return str(output / default_name if multiple_targets else output)
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
    task_name = str(coalesce_option(args.task_name, config, "task_name") or "unknown")
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


def _report_progress(command: str, index: int, blocks: list[np.ndarray]) -> None:
    """One line per block, so that long runs show where they are."""

    if len(blocks) > 1:
        done = sum(len(block) for block in blocks[:index])
        total = sum(len(block) for block in blocks)
        print(
            f"{command}: block {index}/{len(blocks)} ({done}/{total} scenarios)",
            flush=True,
        )


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


def _report_screening(by_stratum: pd.DataFrame) -> None:
    """Caught, missed and false alarms per screened variable."""

    overall = by_stratum[by_stratum["stratification"].eq("")]
    for variable, rows in overall.groupby("variable", sort=False):
        side = "upper" if variable == "line_loading" else "any"
        row = rows[rows["side"].eq(side)].iloc[0]
        limit = "upper limit" if side == "upper" else "either limit"
        line = (
            f"screen: {variable} ({limit}): {int(row['caught']):,} of "
            f"{int(row['violations']):,} true violations flagged "
            f"({int(row['missed']):,} missed), {int(row['false_alarms']):,} false alarms "
            f"in {int(row['n_values']):,} values"
        )
        if int(row["n_known"]):
            line += f"; {int(row['n_known']):,} known values not counted"
        print(line, flush=True)


def _common_parent(paths: Iterable[str]) -> str:
    parents = [Path(path).parent for path in paths]
    return str(
        parents[0] if len(set(parents)) == 1 else Path(os.path.commonpath(parents))
    )


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

    strata: list[pd.DataFrame] = []
    groups: list[pd.DataFrame] = []
    files: list[tuple[str, str]] = [
        (
            "screening_by_stratum.csv",
            "alarms, misses and false alarms per variable and stratum",
        ),
        ("screening_by_group.csv", "the same per bus type (voltages) or line rating"),
    ]
    for target_kind, channel_names in target_groups.items():
        input_path = input_paths[target_kind]
        if not Path(input_path).exists():
            raise FileNotFoundError(
                f"Calibrated predictions not found: {input_path}. Run calibrate first."
            )
        entity_column = "bus" if target_kind == "volt" else "branch"
        family = reports.family_name(entity_column)
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
        strat_columns = [
            column
            for column in _ordered_analysis_strat_columns(columns)
            if any(
                _stratification_applies_to_variable(column, v) for v in channel_names
            )
        ]
        extra = [
            *strat_columns,
            *(
                f"{column}_label"
                for column in strat_columns
                if f"{column}_label" in columns
            ),
        ]
        known_columns = [f"{v}_known" for v in channel_names if f"{v}_known" in columns]
        bus_types = _scan_bus_types(input_path) if target_kind == "volt" else {}
        scenario_ids = read_scenario_ids(input_path)
        detail_name = f"screening_detail_{family}.parquet"
        detail_writer = PartitionedWriter(screen_dir / detail_name)
        counts: list[pd.DataFrame] = []
        stratum_counts: list[pd.DataFrame] = []
        blocks = scenario_blocks(
            scenario_ids,
            rows_per_scenario(input_path, len(scenario_ids)),
            _BLOCK_ROWS,
        )
        for index, block in enumerate(blocks, start=1):
            _report_progress(f"screen ({target_kind})", index, blocks)
            frame = read_frame(
                input_path,
                ["scenario", entity_column, *sorted(needed), *extra, *known_columns],
                scenario_ids=block,
            )
            if target_kind == "volt":
                frame = _ensure_voltage_screen_targets(
                    frame, channel_names, args, config, reference=reference
                )
                detail_frame = _build_voltage_screen_detail_frame(
                    frame,
                    channel_names,
                    bounds_by_variable,
                    bus_types=bus_types,
                    extra=extra,
                )
            else:
                frame = _ensure_line_screen_targets(
                    frame, channel_names, args, config, reference=reference
                )
                detail_frame = _build_line_screen_detail_frame(
                    frame, channel_names, bounds_by_variable, extra=extra
                )
            known = detail_frame["known"].to_numpy(dtype=bool)
            stratum_counts.append(_screen_stratum_counts(detail_frame, strat_columns))
            unknown = detail_frame[~known].drop(columns=["known"])
            detail_writer.write(unknown)
            counts.append(_screen_counts(unknown, group_columns))

        entity_summary = _combine_screen_counts(counts, group_columns)
        group_summary = (
            _build_voltage_screen_summary(entity_summary)
            if target_kind == "volt"
            else _build_line_screen_summary(entity_summary)
        )
        groups.append(_screening_groups_long(group_summary))
        element_file = f"screening_by_element_{family}.csv"
        reports.write_csv(
            reports.screening_long(
                entity_summary.rename(columns={"n_samples": "n_values"}),
                group_columns,
                keep=["n_values"],
            ),
            screen_dir / element_file,
        )
        files.append(
            (element_file, f"the same for every {_ELEMENT_NAMES[entity_column]}")
        )
        files.append(
            (
                detail_name,
                "one row per value screened: bounds, true value and the four flags",
            )
        )
        strata.append(_combine_screen_stratum_counts(stratum_counts))

    by_stratum = reports.sort_by_stratum(pd.concat(strata, ignore_index=True))
    reports.write_csv(
        by_stratum.drop(columns=["stratification", "stratum_value"]),
        screen_dir / "screening_by_stratum.csv",
    )
    reports.write_csv(
        pd.concat(groups, ignore_index=True), screen_dir / "screening_by_group.csv"
    )
    save_figures = coalesce_option(args.save_figures, config, "save_figures")
    figures = (
        reports.save_screening_figures(by_stratum, screen_dir / "figures")
        if save_figures is None or bool(save_figures)
        else []
    )
    limits = "; ".join(
        f"`{variable}` ≤ {bounds['upper']:g}"
        if variable == "line_loading"
        else f"`{variable}` in [{bounds['lower']:g}, {bounds['upper']:g}]"
        for variable, bounds in bounds_by_variable.items()
        if variable in variables
    )
    reports.write_screening_summary(
        screen_dir / "summary.md",
        by_stratum,
        figures,
        files,
        context=[
            f"- Limits: {limits}.",
            *(f"- Calibrated predictions: `{path}`." for path in input_paths.values()),
        ],
        known_where=_BUS_KNOWN_WHERE,
    )
    _report_screening(by_stratum)
    print(f"screen: reports in {screen_dir}", flush=True)


def _screen_stratum_counts(
    detail: pd.DataFrame, strat_columns: list[str]
) -> pd.DataFrame:
    """Confusion counts per variable and stratum of one block, known values apart."""

    counts = reports.side_counts(detail)
    count_columns = list(counts.columns)
    known = detail["known"].to_numpy(dtype=bool)
    counts.loc[known, count_columns] = 0
    counts["n_values"] = (~known).astype(np.int64)
    counts["n_known"] = known.astype(np.int64)
    sums = [*count_columns, "n_values", "n_known"]
    parts: list[pd.DataFrame] = []
    for variable, index in detail.groupby("variable", sort=False).groups.items():
        rows = detail.loc[index]
        work = counts.loc[index]
        overall = work[sums].sum().to_frame().T
        overall["stratification"] = ""
        overall["stratum_value"] = float("nan")
        overall["stratum"] = "all"
        overall["variable"] = variable
        parts.append(overall)
        for stratification in strat_columns:
            if not _stratification_applies_to_variable(stratification, str(variable)):
                continue
            grouped = (
                work[sums]
                .groupby(rows[stratification].rename("stratum_value"), dropna=False)
                .sum()
                .reset_index()
            )
            grouped["stratification"] = stratification
            grouped["stratum"] = [
                _stratum_label(rows, stratification, value)
                for value in grouped["stratum_value"].tolist()
            ]
            grouped["variable"] = variable
            parts.append(grouped)
    return pd.concat(parts, ignore_index=True)


def _combine_screen_stratum_counts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(parts, ignore_index=True)
    sums = [c for c in combined.columns if c.endswith(("_tp", "_fp", "_fn", "_tn"))]
    sums += ["n_values", "n_known"]
    totals = (
        combined.groupby(
            ["variable", "stratification", "stratum_value"], sort=False, dropna=False
        )
        .agg(stratum=("stratum", "first"), **{c: (c, "sum") for c in sums})
        .reset_index()
    )
    totals = reports.sort_by_stratum(totals)
    return reports.screening_long(
        totals,
        ["variable", "stratification", "stratum_value", "stratum"],
        keep=["n_values", "n_known"],
    )


def _screening_groups_long(group_summary: pd.DataFrame) -> pd.DataFrame:
    """Group rows (all, bus types or line ratings) in the long screening layout."""

    frame = group_summary.copy()
    value_column = "bus_type" if "bus_type" in frame.columns else "rate_a"
    frame["group"] = [
        "all"
        if scope == "system"
        else (
            f"{value_column}={value:g}"
            if value_column == "rate_a"
            else f"{value_column}={value}"
        )
        for scope, value in zip(frame["scope"], frame[value_column], strict=True)
    ]
    frame = frame.rename(columns={"n_entities": "n_elements", "n_samples": "n_values"})
    return reports.screening_long(
        frame, ["variable", "group"], keep=["n_elements", "n_values"]
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
    extra: Sequence[str] = (),
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for variable in variables:
        _require_screen_columns(frame, variable, ["scenario", "bus"])
        rows = frame.loc[:, ["scenario", "bus", *extra]].copy()
        rows["bus_type"] = rows["bus"].map(bus_types).fillna("Unknown").astype(object)
        records.append(
            _screen_detail_rows(rows, frame, variable, bounds_by_variable[variable])
        )
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["variable", "scenario", "bus"])
        .reset_index(drop=True)
    )


def _build_line_screen_detail_frame(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    bounds_by_variable: dict[str, dict[str, float]],
    *,
    extra: Sequence[str] = (),
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    columns = ["scenario", "branch", "from_bus", "to_bus", "rate_a"]
    for variable in variables:
        _require_screen_columns(frame, variable, columns)
        # Out-of-service or unrated branches carry -1 and are not screened.
        in_service = frame[frame[f"{variable}_target"] >= 0.0]
        records.append(
            _screen_detail_rows(
                in_service.loc[:, [*columns, *extra]].copy(),
                in_service,
                variable,
                bounds_by_variable[variable],
            )
        )
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["variable", "scenario", "branch"])
        .reset_index(drop=True)
    )


def _require_screen_columns(
    frame: pd.DataFrame, variable: str, columns: list[str]
) -> None:
    required = {
        *columns,
        f"{variable}_target",
        f"{variable}_pi_lower",
        f"{variable}_pi_upper",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Missing required screening columns for {variable}: "
            f"{sorted(required - set(frame.columns))}"
        )


def _screen_detail_rows(
    rows: pd.DataFrame,
    frame: pd.DataFrame,
    variable: str,
    bounds: dict[str, float],
) -> pd.DataFrame:
    """Screening rows of one variable: true value, bounds and violation flags.

    A violation is predicted when a bound crosses the limit, and true when the
    true value does. The counts of caught, missed and false alarms are derived
    from these four flags (see `_screen_counts`), so they are not stored.
    """

    target = frame[f"{variable}_target"]
    lower = frame[f"{variable}_pi_lower"]
    upper = frame[f"{variable}_pi_upper"]
    rows["variable"] = variable
    rows["target"] = target.to_numpy()
    rows["pi_lower"] = lower.to_numpy()
    rows["pi_upper"] = upper.to_numpy()
    lower_limit = float(bounds["lower"]) - SCREEN_BOUND_TOLERANCE
    upper_limit = float(bounds["upper"]) + SCREEN_BOUND_TOLERANCE
    true_values = target.to_numpy(dtype=float)
    if variable == "line_loading":
        rows["pred_lower_violation"] = False
        rows["true_lower_violation"] = False
    else:
        rows["pred_lower_violation"] = lower.to_numpy(dtype=float) < lower_limit
        rows["true_lower_violation"] = true_values < lower_limit
    rows["pred_upper_violation"] = upper.to_numpy(dtype=float) > upper_limit
    rows["true_upper_violation"] = true_values > upper_limit
    rows["known"] = _known_mask(frame, variable)
    return rows


def _screen_counts(
    detail_frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Confusion counts and sample counts per entity of one block.

    `tp`, `fp`, `fn` and `tn` count predicted against true violations on the
    lower side, the upper side, and either side (`any`).
    """

    counts = detail_frame.loc[:, group_columns].copy()
    flags = {
        side: (
            detail_frame[f"pred_{side}_violation"].to_numpy(dtype=bool),
            detail_frame[f"true_{side}_violation"].to_numpy(dtype=bool),
        )
        for side in ("lower", "upper")
    }
    flags["any"] = (
        flags["lower"][0] | flags["upper"][0],
        flags["lower"][1] | flags["upper"][1],
    )
    for side, (predicted, true) in flags.items():
        counts[f"{side}_tp"] = (predicted & true).astype(np.int64)
        counts[f"{side}_fp"] = (predicted & ~true).astype(np.int64)
        counts[f"{side}_fn"] = (~predicted & true).astype(np.int64)
        counts[f"{side}_tn"] = (~predicted & ~true).astype(np.int64)
    grouped = counts.groupby(group_columns, sort=False, dropna=False)
    summed = grouped.sum()
    summed.insert(0, "n_samples", grouped.size())
    return summed.reset_index()


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
        work["covered"] = np.where(
            known, 0.0, ((target >= lower) & (target <= upper)).astype(float)
        )
        work["scenario_unknown"] = np.where(
            known, np.nan, frame["scenario"].to_numpy(dtype=float)
        )
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
                stratification="",
            )
        )
        for stratification in strat_columns:
            statistics = _entity_statistics(
                work,
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


def _known_mask(frame: pd.DataFrame, variable: str) -> np.ndarray:
    """Rows whose value of `variable` is a known set-point (see `_SET_POINTS`)."""

    return _flag(frame, f"{variable}_known")


def _flag(frame: pd.DataFrame, column: str) -> np.ndarray:
    """A boolean column of the frame, or all False when it is absent."""

    if column not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    return frame[column].to_numpy(dtype=bool)


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
    elements["min_width"] = combined["min_width"].astype(float)
    elements["max_width"] = combined["max_width"].astype(float)
    return elements.sort_values(
        ["variable", "stratification", "stratum_value", entity_column],
        na_position="first",
    ).reset_index(drop=True)


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
