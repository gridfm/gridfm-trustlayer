"""CLI for naive interval computation and parquet augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

from gridfm_trustlayer.cli.config import (
    coalesce_option,
    load_cli_config,
    normalize_variables,
    resolve_command_config,
    validate_cp_type,
    validate_strat,
)
from gridfm_trustlayer.conformal import ChannelSet, NaiveIntervalCalibrator
from gridfm_trustlayer.io import (
    compute_line_loading,
    load_ground_truth,
    load_predictions,
)
from gridfm_trustlayer.io.artifact import (
    load_interval_artifacts,
    save_interval_artifacts,
)


SCREEN_BOUND_TOLERANCE = 1e-6
ANALYSIS_STRAT_COLUMNS = ("n_outages", "outage_type")


def main(argv: list[str] | None = None) -> int:
    """Run the gridfm_trustlayer command-line interface."""

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
        help="Compute naive intervals for Vm/Va or derived line_loading and save an artifact.",
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
        "--cp-type", help="Currently only 'element' is supported."
    )
    compute_parser.add_argument(
        "--strat",
        help="Optional residual stratification. Supported values: 'n_outages', 'outage_type'.",
    )
    compute_parser.add_argument(
        "--variables",
        nargs="+",
        help="Variables to calibrate: any subset of Vm, Va, and line_loading.",
    )
    compute_parser.add_argument(
        "--calibration-fraction",
        type=float,
        help="Optional scenario fraction reserved for calibration when fitting on a pool and applying on holdout scenarios.",
    )
    compute_parser.add_argument(
        "--random-seed", type=int, help="Random seed used with --calibration-fraction."
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
        help="Append fitted intervals to predictions and write voltage and/or line-loading analysis outputs.",
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
        "--cp-type", help="Currently only 'element' is supported."
    )
    calibrate_parser.add_argument(
        "--strat",
        help="Optional residual stratification override. Supported values: 'n_outages', 'outage_type'. Must match the saved artifact when provided.",
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
        "--task-name", help="Task name used to interpret prediction schema."
    )
    calibrate_parser.add_argument(
        "--generator-predictions",
        help="Optional generator predictions parquet for OPF artifacts.",
    )

    screen_parser = subparsers.add_parser(
        "screen",
        help="Evaluate screened lower/upper bounds against ground truth using calibrated prediction intervals.",
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
        help="Override lower screening bound for a single selected variable.",
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
    cp_type = validate_cp_type(coalesce_option(args.cp_type, config, "cp_type"))
    strat = validate_strat(coalesce_option(args.strat, config, "strat"))
    variables = normalize_variables(
        coalesce_option(args.variables, config, "variables")
    )
    variable_groups = _partition_variables(variables)
    calibration_fraction = coalesce_option(
        args.calibration_fraction,
        config,
        "calibration_fraction",
    )
    random_seed = coalesce_option(args.random_seed, config, "random_seed")
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
    ground_truth = load_ground_truth(
        coalesce_option(args.pf_node, config, "pf_node", required=True),
        coalesce_option(args.pf_edge, config, "pf_edge", required=True),
        scenario_ids=predictions.scenario_ids,
    )
    split = None
    if calibration_fraction is not None:
        split = _split_calibration_scenarios(
            predictions.scenario_ids,
            float(calibration_fraction),
            random_seed,
        )
    calibrators: dict[str, NaiveIntervalCalibrator] = {}
    for target_kind, target_variables in variable_groups.items():
        calibration_predictions, calibration_ground_truth = _build_calibration_sets(
            predictions,
            ground_truth,
            target_variables,
            strat=strat,
        )
        if split is not None:
            calibration_predictions = _subset_channel_set(
                calibration_predictions,
                split["calibration_scenario_ids"],
            )
            calibration_ground_truth = _subset_channel_set(
                calibration_ground_truth,
                split["calibration_scenario_ids"],
            )
        calibrator = NaiveIntervalCalibrator.fit(
            calibration_predictions,
            calibration_ground_truth,
            alpha=float(coalesce_option(args.alpha, config, "alpha", required=True)),
            channel_names=target_variables,
            strat=strat,
        )
        if split is not None and calibration_fraction is not None:
            _apply_split_metadata(
                calibrator, float(calibration_fraction), random_seed, split
            )
        calibrators[target_kind] = calibrator
    artifact_path = coalesce_option(args.output, config, "artifact")
    if artifact_path is None:
        artifact_path = str(
            _default_output_dir(predictions.task_name, cp_type).joinpath(
                _default_artifact_name(predictions_path, cp_type, variables),
            ),
        )
    save_interval_artifacts(calibrators, artifact_path)


def _run_calibrate(args: argparse.Namespace) -> None:
    config = resolve_command_config(load_cli_config(args.config), "calibrate")
    cp_type = validate_cp_type(coalesce_option(args.cp_type, config, "cp_type"))
    requested_strat = validate_strat(coalesce_option(args.strat, config, "strat"))
    predictions_path = coalesce_option(
        args.predictions, config, "predictions", required=True
    )
    artifact_path = coalesce_option(args.artifact, config, "artifact", required=True)
    calibrators = load_interval_artifacts(artifact_path)
    if requested_strat is not None:
        for calibrator in calibrators.values():
            if calibrator.strat != requested_strat:
                raise ValueError(
                    "Requested stratification does not match the saved interval artifact."
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
    original_frame = (
        None
        if not any(
            calibrator.entity_name == "bus" for calibrator in calibrators.values()
        )
        else pd.read_parquet(predictions_path)
    )
    application_scenario_ids = _resolve_application_scenario_ids(
        calibrators, predictions.scenario_ids
    )
    if application_scenario_ids is not None:
        predictions = _subset_prediction_set(predictions, application_scenario_ids)
        if original_frame is not None:
            original_frame = original_frame[
                original_frame["scenario"].isin(application_scenario_ids.tolist())
            ].copy()

    analysis_ground_truth = None
    if any(calibrator.strat is not None for calibrator in calibrators.values()):
        analysis_ground_truth = _load_analysis_ground_truth(
            args, config, predictions.scenario_ids
        )
    multiple_targets = len(calibrators) > 1
    for target_kind, calibrator in calibrators.items():
        if calibrator.entity_name == "branch":
            if analysis_ground_truth is None:
                analysis_ground_truth = _load_analysis_ground_truth(
                    args, config, predictions.scenario_ids
                )
            application_predictions, application_ground_truth = _build_calibration_sets(
                predictions,
                analysis_ground_truth,
                calibrator.channel_names,
                strat=calibrator.strat,
            )
            intervals = calibrator.predict_intervals(application_predictions)
            output_frame = _build_entity_output_frame(
                application_predictions,
                application_ground_truth,
                intervals,
            )
        else:
            application_predictions = _build_bus_channel_set(
                predictions,
                calibrator.channel_names,
                scenario_columns=(
                    {}
                    if analysis_ground_truth is None
                    else _build_scenario_columns(
                        analysis_ground_truth, calibrator.strat
                    )
                ),
            )
            intervals = calibrator.predict_intervals(application_predictions)
            interval_frame = _build_interval_frame(application_predictions, intervals)
            assert original_frame is not None
            output_frame = original_frame.merge(
                interval_frame,
                on=["scenario", "bus"],
                how="left",
                validate="one_to_one",
            )

        output_path = _resolve_output_path(
            args,
            config,
            predictions_path,
            cp_type,
            calibrator.channel_names,
            target_kind=target_kind,
            multiple_targets=multiple_targets,
            task_name=predictions.task_name,
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_parquet(output_path, index=False)

        analysis_dir = Path(output_path).parent
        figures_dir = analysis_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        if calibrator.entity_name == "branch":
            line_analysis = _build_line_analysis_frame(output_frame, tuple(intervals))
            line_analysis.to_csv(analysis_dir / "trust_analysis_line.csv", index=False)
            _build_line_summary_frame(line_analysis).to_csv(
                analysis_dir / "trust_summary_line.csv",
                index=False,
            )
            _save_line_analysis_figures(line_analysis, figures_dir)
        else:
            analysis_frame = _build_analysis_frame_from_targets(
                output_frame, tuple(intervals)
            )
            if analysis_frame is None:
                if analysis_ground_truth is None:
                    analysis_ground_truth = _load_analysis_ground_truth(
                        args, config, predictions.scenario_ids
                    )
                _, application_ground_truth = _build_calibration_sets(
                    predictions,
                    analysis_ground_truth,
                    calibrator.channel_names,
                )
                analysis_frame = _build_analysis_frame(
                    application_predictions, intervals, application_ground_truth
                )
            analysis_frame.to_csv(analysis_dir / "trust_analysis_volt.csv", index=False)
            _build_summary_frame(analysis_frame).to_csv(
                analysis_dir / "trust_summary_volt.csv", index=False
            )
            _save_analysis_figures(analysis_frame, figures_dir)


def _run_screen(args: argparse.Namespace) -> None:
    config = resolve_command_config(load_cli_config(args.config), "screen")
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
        frame = pd.read_parquet(input_paths[target_kind])
        if target_kind == "volt":
            frame = _ensure_voltage_screen_targets(frame, channel_names, args, config)
            detail_frame = _build_voltage_screen_detail_frame(
                frame, channel_names, bounds_by_variable
            )
            entity_summary = _build_voltage_screen_entity_summary(detail_frame)
            summary_frame = _build_voltage_screen_summary(detail_frame)
            detail_frame.to_csv(
                screen_dir / "trust_screen_detail_volt.csv", index=False
            )
            entity_summary.to_csv(
                screen_dir / "trust_screen_entity_volt.csv", index=False
            )
            summary_frame.to_csv(
                screen_dir / "trust_screen_summary_volt.csv", index=False
            )
            _save_screen_figures(
                entity_summary, summary_frame, figures_dir, target_kind="volt"
            )
        else:
            frame = _ensure_line_screen_targets(frame, channel_names, args, config)
            detail_frame = _build_line_screen_detail_frame(
                frame, channel_names, bounds_by_variable
            )
            entity_summary = _build_line_screen_entity_summary(detail_frame)
            summary_frame = _build_line_screen_summary(detail_frame)
            detail_frame.to_csv(
                screen_dir / "trust_screen_detail_line.csv", index=False
            )
            entity_summary.to_csv(
                screen_dir / "trust_screen_entity_line.csv", index=False
            )
            summary_frame.to_csv(
                screen_dir / "trust_screen_summary_line.csv", index=False
            )
            _save_screen_figures(
                entity_summary, summary_frame, figures_dir, target_kind="line"
            )


def _partition_variables(
    channel_names: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for channel_name in channel_names:
        target_kind = "line" if channel_name == "line_loading" else "volt"
        grouped.setdefault(target_kind, []).append(channel_name)
    return {target_kind: tuple(names) for target_kind, names in grouped.items()}


def _apply_split_metadata(
    calibrator: NaiveIntervalCalibrator,
    calibration_fraction: float,
    random_seed: int | None,
    split: dict[str, np.ndarray],
) -> None:
    calibrator.calibration_fraction = calibration_fraction
    calibrator.random_seed = None if random_seed is None else int(random_seed)
    calibrator.split_pool_scenario_ids = split["split_pool_scenario_ids"]
    calibrator.calibration_scenario_ids = split["calibration_scenario_ids"]
    calibrator.application_scenario_ids = split["application_scenario_ids"]


def _resolve_application_scenario_ids(
    calibrators: dict[str, NaiveIntervalCalibrator],
    scenario_ids: np.ndarray,
) -> np.ndarray | None:
    application_ids: np.ndarray | None = None
    for calibrator in calibrators.values():
        should_subset = _should_use_application_subset(calibrator, scenario_ids)
        if not should_subset:
            if application_ids is not None:
                raise ValueError(
                    "All calibrators in a mixed artifact must agree on the application-scenario subset."
                )
            continue
        assert calibrator.application_scenario_ids is not None
        candidate = calibrator.application_scenario_ids.astype(np.int64, copy=False)
        if application_ids is None:
            application_ids = candidate
            continue
        if not np.array_equal(
            np.sort(application_ids.astype(np.int64, copy=False)),
            np.sort(candidate.astype(np.int64, copy=False)),
        ):
            raise ValueError(
                "All calibrators in a mixed artifact must share the same application-scenario subset."
            )
    return None if application_ids is None else application_ids.copy()


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
                    f"screen_bounds.{raw_name} must be a mapping with 'lower' and 'upper'."
                )
            if "lower" not in raw_value or "upper" not in raw_value:
                raise ValueError(
                    f"screen_bounds.{raw_name} must define both 'lower' and 'upper'."
                )
            bounds_by_variable[variable] = {
                "lower": float(raw_value["lower"]),
                "upper": float(raw_value["upper"]),
            }

    if args.lower_bound is not None or args.upper_bound is not None:
        if len(variables) != 1:
            raise ValueError(
                "--lower-bound and --upper-bound can override bounds only when screening a single variable."
            )
        if args.lower_bound is None or args.upper_bound is None:
            raise ValueError(
                "Provide both --lower-bound and --upper-bound when overriding screening bounds from the CLI."
            )
        bounds_by_variable[variables[0]] = {
            "lower": float(args.lower_bound),
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
    if args.calibrated_predictions is not None:
        return str(args.calibrated_predictions)
    if "calibrated_predictions" in config:
        return str(config["calibrated_predictions"])

    predictions_path = coalesce_option(None, config, "predictions", required=True)
    return str(
        _default_output_dir(task_name, cp_type).joinpath(
            _default_calibrated_name(str(predictions_path), cp_type, channel_names),
        ),
    )


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
) -> pd.DataFrame:
    missing_targets = [
        variable for variable in variables if f"{variable}_target" not in frame.columns
    ]
    if not missing_targets:
        return frame

    ground_truth = _load_analysis_ground_truth(
        args,
        config,
        frame["scenario"].drop_duplicates().to_numpy(dtype=np.int64, copy=True),
    )
    target_frame = pd.DataFrame(
        {
            "scenario": np.repeat(ground_truth.scenario_ids, len(ground_truth.bus_ids)),
            "bus": np.tile(ground_truth.bus_ids, len(ground_truth.scenario_ids)),
        },
    )
    for variable in missing_targets:
        target_frame[f"{variable}_target"] = _ground_truth_channel(
            ground_truth, variable
        ).reshape(-1)
    return frame.merge(
        target_frame, on=["scenario", "bus"], how="left", validate="one_to_one"
    )


def _ensure_line_screen_targets(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    args: argparse.Namespace,
    config: dict[str, object],
) -> pd.DataFrame:
    missing_target = any(
        f"{variable}_target" not in frame.columns for variable in variables
    )
    missing_metadata = not {"from_bus", "to_bus", "rate_a"}.issubset(frame.columns)
    if not missing_target and not missing_metadata:
        return frame

    ground_truth = _load_analysis_ground_truth(
        args,
        config,
        frame["scenario"].drop_duplicates().to_numpy(dtype=np.int64, copy=True),
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
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    bus_type_by_bus = _resolve_bus_type_labels_from_frame(frame)
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
        variable_frame["bus_type"] = variable_frame["bus"].map(
            lambda bus: bus_type_by_bus.get(int(bus), "Unknown")
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


def _build_voltage_screen_entity_summary(detail_frame: pd.DataFrame) -> pd.DataFrame:
    return _build_screen_entity_summary(
        detail_frame,
        group_columns=["variable", "bus", "bus_type"],
        sort_columns=["variable", "bus"],
    )


def _build_line_screen_entity_summary(detail_frame: pd.DataFrame) -> pd.DataFrame:
    return _build_screen_entity_summary(
        detail_frame,
        group_columns=["variable", "branch", "from_bus", "to_bus", "rate_a"],
        sort_columns=["variable", "branch"],
    )


def _build_screen_entity_summary(
    detail_frame: pd.DataFrame,
    *,
    group_columns: list[str],
    sort_columns: list[str],
) -> pd.DataFrame:
    aggregate_map = {column: (column, "sum") for column in _screen_count_columns()}
    summary = (
        detail_frame.groupby(group_columns, sort=True, dropna=False)
        .agg(
            n_samples=("scenario", "size"),
            **aggregate_map,
        )
        .reset_index()
    )
    summary = _add_screen_metric_columns(summary)
    return summary.sort_values(sort_columns).reset_index(drop=True)


def _build_voltage_screen_summary(detail_frame: pd.DataFrame) -> pd.DataFrame:
    system = _build_grouped_screen_summary(
        detail_frame,
        group_columns=["variable"],
        scope="system",
        group_value_column="bus_type",
        group_value=float("nan"),
    )
    bus_type = _build_grouped_screen_summary(
        detail_frame,
        group_columns=["variable", "bus_type"],
        scope="bus_type",
        group_value_column="bus_type",
    )
    return pd.concat([system, bus_type], ignore_index=True)


def _build_line_screen_summary(detail_frame: pd.DataFrame) -> pd.DataFrame:
    system = _build_grouped_screen_summary(
        detail_frame,
        group_columns=["variable"],
        scope="system",
        group_value_column="rate_a",
        group_value=float("nan"),
    )
    rate_a = _build_grouped_screen_summary(
        detail_frame,
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
    detail_frame: pd.DataFrame,
    *,
    group_columns: list[str],
    scope: str,
    group_value_column: str,
    group_value: object | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = detail_frame.groupby(group_columns, sort=True, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record: dict[str, object] = {
            "scope": scope,
            "variable": str(group_key[0]),
            "n_entities": int(group["bus"].nunique())
            if "bus" in group.columns
            else int(group["branch"].nunique()),
            "n_samples": int(len(group)),
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
) -> tuple[ChannelSet, ChannelSet]:
    scenario_columns = _build_scenario_columns(ground_truth, strat)
    if _resolve_target_kind(channel_names) == "line":
        return _build_line_loading_sets(
            predictions,
            ground_truth,
            scenario_columns=scenario_columns,
        )
    return (
        _build_bus_channel_set(
            predictions,
            channel_names,
            scenario_columns=scenario_columns,
        ),
        _build_ground_truth_bus_channel_set(
            ground_truth,
            channel_names,
            scenario_columns=scenario_columns,
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
    )


def _append_voltage_analysis_records(
    records: list[dict[str, object]],
    *,
    variable: str,
    entity_ids: np.ndarray,
    bus_type_by_bus: dict[int, object],
    widths: np.ndarray,
    coverage: np.ndarray,
    analysis_scope: str,
    stratification: str = "",
    stratum_value: int | None = None,
) -> None:
    width_mean_per_bus = widths.mean(axis=0)
    width_min_per_bus = widths.min(axis=0)
    width_max_per_bus = widths.max(axis=0)
    coverage_mean_per_bus = coverage.mean(axis=0)
    n_samples = int(widths.shape[0])

    for index, bus_id in enumerate(entity_ids.tolist()):
        records.append(
            {
                "analysis_scope": analysis_scope,
                "stratification": stratification,
                "stratum_value": float("nan")
                if stratum_value is None
                else int(stratum_value),
                "stratum_label": _format_stratum_label(stratification, stratum_value),
                "variable": variable,
                "bus": int(bus_id),
                "bus_type": bus_type_by_bus.get(int(bus_id), "Unknown"),
                "width_mean": float(width_mean_per_bus[index]),
                "width_min": float(width_min_per_bus[index]),
                "width_max": float(width_max_per_bus[index]),
                "width_mean_degrees": _maybe_width_degrees(
                    variable, width_mean_per_bus[index]
                ),
                "width_min_degrees": _maybe_width_degrees(
                    variable, width_min_per_bus[index]
                ),
                "width_max_degrees": _maybe_width_degrees(
                    variable, width_max_per_bus[index]
                ),
                "empirical_coverage": float(coverage_mean_per_bus[index]),
                "n_samples": n_samples,
                "n_scenarios": n_samples,
            },
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
        ),
    )


def _build_scenario_columns(
    ground_truth,
    strat: str | None,
) -> dict[str, np.ndarray]:
    if strat is None:
        return {}
    if strat == "n_outages":
        return {"n_outages": ground_truth.n_outages.astype(np.int32, copy=True)}
    if strat == "outage_type":
        return {"outage_type": ground_truth.outage_type.astype(np.int32, copy=True)}
    raise ValueError(f"Unsupported stratification: {strat!r}")


def _subset_channel_set(
    channel_set: ChannelSet,
    scenario_ids: np.ndarray,
) -> ChannelSet:
    scenario_mask = np.isin(channel_set.scenario_ids, scenario_ids)
    return ChannelSet(
        scenario_ids=channel_set.scenario_ids[scenario_mask].copy(),
        entity_ids=channel_set.entity_ids.copy(),
        entity_name=channel_set.entity_name,
        channels={
            name: values[scenario_mask].copy()
            for name, values in channel_set.channels.items()
        },
        valid_masks={
            name: values[scenario_mask].copy()
            for name, values in channel_set.valid_masks.items()
        },
        entity_columns={
            name: values.copy() for name, values in channel_set.entity_columns.items()
        },
        scenario_columns={
            name: values[scenario_mask].copy()
            for name, values in channel_set.scenario_columns.items()
        },
    )


def _split_calibration_scenarios(
    scenario_ids: np.ndarray,
    calibration_fraction: float,
    random_seed: int | None,
) -> dict[str, np.ndarray]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            "calibration_fraction must be strictly between 0 and 1 when provided.",
        )
    if len(scenario_ids) < 2:
        raise ValueError(
            "calibration_fraction requires at least two scenarios so one can remain for application.",
        )
    n_calibration = int(np.floor(len(scenario_ids) * calibration_fraction))
    n_calibration = min(max(n_calibration, 1), len(scenario_ids) - 1)
    rng = np.random.default_rng(None if random_seed is None else int(random_seed))
    calibration_ids = np.sort(
        rng.choice(scenario_ids, size=n_calibration, replace=False).astype(
            np.int64, copy=False
        ),
    )
    application_ids = np.array(
        [
            scenario_id
            for scenario_id in scenario_ids
            if scenario_id not in set(calibration_ids.tolist())
        ],
        dtype=np.int64,
    )
    return {
        "split_pool_scenario_ids": scenario_ids.copy(),
        "calibration_scenario_ids": calibration_ids,
        "application_scenario_ids": application_ids,
    }


def _should_use_application_subset(
    calibrator: NaiveIntervalCalibrator,
    scenario_ids: np.ndarray,
) -> bool:
    if (
        calibrator.split_pool_scenario_ids is None
        or calibrator.application_scenario_ids is None
    ):
        return False
    return np.array_equal(
        np.sort(scenario_ids.astype(np.int64, copy=False)),
        np.sort(calibrator.split_pool_scenario_ids.astype(np.int64, copy=False)),
    )


def _subset_prediction_set(
    predictions,
    scenario_ids: np.ndarray,
):
    scenario_mask = np.isin(predictions.scenario_ids, scenario_ids)
    generator_predictions = predictions.generator_predictions
    if generator_predictions is not None:
        generator_mask = np.isin(generator_predictions.scenario_ids, scenario_ids)
        generator_predictions = type(generator_predictions)(
            channel_names=generator_predictions.channel_names,
            scenario_ids=generator_predictions.scenario_ids[generator_mask].copy(),
            generator_ids=generator_predictions.generator_ids[generator_mask].copy(),
            bus_ids=generator_predictions.bus_ids[generator_mask].copy(),
            channels={
                name: values[generator_mask].copy()
                for name, values in generator_predictions.channels.items()
            },
            target_channels={
                name: values[generator_mask].copy()
                for name, values in generator_predictions.target_channels.items()
            },
            static_columns={
                name: values[generator_mask].copy()
                for name, values in generator_predictions.static_columns.items()
            },
        )
    return type(predictions)(
        task_name=predictions.task_name,
        channel_names=predictions.channel_names,
        scenario_ids=predictions.scenario_ids[scenario_mask].copy(),
        bus_ids=predictions.bus_ids.copy(),
        Vm=predictions.Vm[scenario_mask].copy(),
        Va=predictions.Va[scenario_mask].copy(),
        channels={
            name: values[scenario_mask].copy()
            for name, values in predictions.channels.items()
        },
        bus_type_masks={
            name: values[scenario_mask].copy()
            for name, values in predictions.bus_type_masks.items()
        },
        generator_predictions=generator_predictions,
    )


def _subset_ground_truth_set(
    ground_truth,
    scenario_ids: np.ndarray,
):
    scenario_mask = np.isin(ground_truth.scenario_ids, scenario_ids)
    return type(ground_truth)(
        scenario_ids=ground_truth.scenario_ids[scenario_mask].copy(),
        bus_ids=ground_truth.bus_ids.copy(),
        branch_ids=ground_truth.branch_ids.copy(),
        n_outages=ground_truth.n_outages[scenario_mask].copy(),
        outage_type=ground_truth.outage_type[scenario_mask].copy(),
        removed_mask=ground_truth.removed_mask[scenario_mask].copy(),
        Vm=ground_truth.Vm[scenario_mask].copy(),
        Va=ground_truth.Va[scenario_mask].copy(),
        channels={
            name: values[scenario_mask].copy()
            for name, values in ground_truth.channels.items()
        },
        admittances={
            name: values[scenario_mask].copy()
            for name, values in ground_truth.admittances.items()
        },
        f_bus=ground_truth.f_bus.copy(),
        t_bus=ground_truth.t_bus.copy(),
        rate_a=ground_truth.rate_a.copy(),
        base_kv=ground_truth.base_kv.copy(),
        sn_mva=ground_truth.sn_mva,
        bus_type_masks={
            name: values[scenario_mask].copy()
            for name, values in ground_truth.bus_type_masks.items()
        },
    )


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


def _build_analysis_frame(
    predictions: ChannelSet,
    intervals: dict[str, tuple[np.ndarray, np.ndarray]],
    ground_truth: ChannelSet,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    bus_type_by_bus = _entity_column_map(predictions, "bus_type")
    for channel_name, (lower, upper) in intervals.items():
        target = ground_truth.channels[channel_name]
        widths = upper - lower
        coverage = (target >= lower) & (target <= upper)
        width_mean_per_bus = widths.mean(axis=0)
        width_min_per_bus = widths.min(axis=0)
        width_max_per_bus = widths.max(axis=0)
        coverage_mean_per_bus = coverage.mean(axis=0)
        n_samples = int(widths.shape[0])

        for index, bus_id in enumerate(predictions.entity_ids.tolist()):
            records.append(
                {
                    "variable": channel_name,
                    "bus": int(bus_id),
                    "bus_type": bus_type_by_bus.get(int(bus_id), "Unknown"),
                    "width_mean": float(width_mean_per_bus[index]),
                    "width_min": float(width_min_per_bus[index]),
                    "width_max": float(width_max_per_bus[index]),
                    "width_mean_degrees": _maybe_width_degrees(
                        channel_name, width_mean_per_bus[index]
                    ),
                    "width_min_degrees": _maybe_width_degrees(
                        channel_name, width_min_per_bus[index]
                    ),
                    "width_max_degrees": _maybe_width_degrees(
                        channel_name, width_max_per_bus[index]
                    ),
                    "empirical_coverage": float(coverage_mean_per_bus[index]),
                    "n_samples": n_samples,
                },
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["variable", "bus"])
        .reset_index(drop=True)
    )


def _build_line_analysis_frame(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    strat_columns = _analysis_scenario_columns_from_frame(frame)
    for variable in variables:
        target_column = f"{variable}_target"
        lower_column = f"{variable}_pi_lower"
        upper_column = f"{variable}_pi_upper"
        pred_column = f"{variable}_pred"
        required_columns = {
            "branch",
            "from_bus",
            "to_bus",
            "rate_a",
            pred_column,
            target_column,
            lower_column,
            upper_column,
        }
        if not required_columns.issubset(frame.columns):
            raise ValueError(
                f"Missing required line-analysis columns for {variable}: {sorted(required_columns - set(frame.columns))}"
            )

        variable_frame = frame.loc[
            :,
            [
                "scenario",
                "branch",
                "from_bus",
                "to_bus",
                "rate_a",
                pred_column,
                target_column,
                lower_column,
                upper_column,
                *strat_columns,
            ],
        ].copy()
        valid_mask = (variable_frame[pred_column] >= 0.0) & (
            variable_frame[target_column] >= 0.0
        )
        variable_frame = variable_frame[valid_mask].copy()
        variable_frame["interval_width"] = (
            variable_frame[upper_column] - variable_frame[lower_column]
        )
        variable_frame["covered"] = (
            (variable_frame[target_column] >= variable_frame[lower_column])
            & (variable_frame[target_column] <= variable_frame[upper_column])
        ).astype(float)
        _append_line_analysis_records(
            records,
            variable=variable,
            variable_frame=variable_frame,
            analysis_scope="overall",
        )
        for stratification in strat_columns:
            for stratum_value, stratum_group in variable_frame.groupby(
                stratification, sort=True, dropna=False
            ):
                _append_line_analysis_records(
                    records,
                    variable=variable,
                    variable_frame=stratum_group,
                    analysis_scope="stratified",
                    stratification=stratification,
                    stratum_value=None
                    if pd.isna(stratum_value)
                    else int(stratum_value),
                )
    return _finalize_analysis_frame(pd.DataFrame.from_records(records), "branch")


def _append_line_analysis_records(
    records: list[dict[str, object]],
    *,
    variable: str,
    variable_frame: pd.DataFrame,
    analysis_scope: str,
    stratification: str = "",
    stratum_value: int | None = None,
) -> None:
    grouped = variable_frame.groupby("branch", sort=True)
    summary = grouped.agg(
        from_bus=("from_bus", "first"),
        to_bus=("to_bus", "first"),
        rate_a=("rate_a", "first"),
        width_mean=("interval_width", "mean"),
        width_min=("interval_width", "min"),
        width_max=("interval_width", "max"),
        empirical_coverage=("covered", "mean"),
        n_samples=("covered", "size"),
        n_scenarios=("scenario", "nunique"),
    )
    for _, row in summary.reset_index().iterrows():
        records.append(
            {
                "analysis_scope": analysis_scope,
                "stratification": stratification,
                "stratum_value": float("nan")
                if stratum_value is None
                else int(stratum_value),
                "stratum_label": _format_stratum_label(stratification, stratum_value),
                "variable": variable,
                "branch": int(row["branch"]),
                "from_bus": int(row["from_bus"]),
                "to_bus": int(row["to_bus"]),
                "rate_a": float(row["rate_a"]),
                "width_mean": float(row["width_mean"]),
                "width_min": float(row["width_min"]),
                "width_max": float(row["width_max"]),
                "empirical_coverage": float(row["empirical_coverage"]),
                "n_samples": int(row["n_samples"]),
                "n_scenarios": int(row["n_scenarios"]),
            },
        )


def _build_analysis_frame_from_targets(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
) -> pd.DataFrame | None:
    records: list[dict[str, object]] = []
    bus_type_by_bus = _resolve_bus_type_labels_from_frame(frame)
    strat_columns = _analysis_scenario_columns_from_frame(frame)
    for variable in variables:
        target_column = f"{variable}_target"
        lower_column = f"{variable}_pi_lower"
        upper_column = f"{variable}_pi_upper"
        required_columns = {"bus", target_column, lower_column, upper_column}
        if not required_columns.issubset(frame.columns):
            return None

        variable_frame = frame.loc[
            :,
            [
                "scenario",
                "bus",
                target_column,
                lower_column,
                upper_column,
                *strat_columns,
            ],
        ].copy()
        variable_frame["interval_width"] = (
            variable_frame[upper_column] - variable_frame[lower_column]
        )
        variable_frame["covered"] = (
            (variable_frame[target_column] >= variable_frame[lower_column])
            & (variable_frame[target_column] <= variable_frame[upper_column])
        ).astype(float)
        _append_voltage_analysis_records_from_frame(
            records,
            variable=variable,
            variable_frame=variable_frame,
            bus_type_by_bus=bus_type_by_bus,
            analysis_scope="overall",
        )
        for stratification in strat_columns:
            for stratum_value, stratum_group in variable_frame.groupby(
                stratification, sort=True, dropna=False
            ):
                _append_voltage_analysis_records_from_frame(
                    records,
                    variable=variable,
                    variable_frame=stratum_group,
                    bus_type_by_bus=bus_type_by_bus,
                    analysis_scope="stratified",
                    stratification=stratification,
                    stratum_value=None
                    if pd.isna(stratum_value)
                    else int(stratum_value),
                )
    return _finalize_analysis_frame(pd.DataFrame.from_records(records), "bus")


def _append_voltage_analysis_records_from_frame(
    records: list[dict[str, object]],
    *,
    variable: str,
    variable_frame: pd.DataFrame,
    bus_type_by_bus: dict[int, str],
    analysis_scope: str,
    stratification: str = "",
    stratum_value: int | None = None,
) -> None:
    grouped = variable_frame.groupby("bus", sort=True)
    summary = grouped.agg(
        width_mean=("interval_width", "mean"),
        width_min=("interval_width", "min"),
        width_max=("interval_width", "max"),
        empirical_coverage=("covered", "mean"),
        n_samples=("covered", "size"),
        n_scenarios=("scenario", "nunique"),
    )
    for _, row in summary.reset_index().iterrows():
        records.append(
            {
                "analysis_scope": analysis_scope,
                "stratification": stratification,
                "stratum_value": float("nan")
                if stratum_value is None
                else int(stratum_value),
                "stratum_label": _format_stratum_label(stratification, stratum_value),
                "variable": variable,
                "bus": int(row["bus"]),
                "bus_type": bus_type_by_bus.get(int(row["bus"]), "Unknown"),
                "width_mean": float(row["width_mean"]),
                "width_min": float(row["width_min"]),
                "width_max": float(row["width_max"]),
                "width_mean_degrees": _maybe_width_degrees(variable, row["width_mean"]),
                "width_min_degrees": _maybe_width_degrees(variable, row["width_min"]),
                "width_max_degrees": _maybe_width_degrees(variable, row["width_max"]),
                "empirical_coverage": float(row["empirical_coverage"]),
                "n_samples": int(row["n_samples"]),
                "n_scenarios": int(row["n_scenarios"]),
            },
        )


def _analysis_scenario_columns_from_channel_set(
    channel_set: ChannelSet,
) -> dict[str, np.ndarray]:
    return {
        name: values.astype(np.int64, copy=False)
        for name, values in channel_set.scenario_columns.items()
        if name in ANALYSIS_STRAT_COLUMNS
    }


def _analysis_scenario_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    return [column for column in ANALYSIS_STRAT_COLUMNS if column in frame.columns]


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


def _resolve_bus_type_labels_from_frame(frame: pd.DataFrame) -> dict[int, str]:
    if "bus" not in frame.columns:
        return {}
    bus_type_frame = frame.loc[
        :, [column for column in ["bus", "REF", "PV", "PQ"] if column in frame.columns]
    ]
    if len(bus_type_frame.columns) == 1:
        return {}
    grouped = (
        bus_type_frame.groupby("bus", sort=True).max(numeric_only=True).reset_index()
    )
    labels: dict[int, str] = {}
    for _, row in grouped.iterrows():
        labels[int(row["bus"])] = _bus_type_label_from_flags(
            ref=row.get("REF", 0.0),
            pv=row.get("PV", 0.0),
            pq=row.get("PQ", 0.0),
        )
    return labels


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
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_width_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(bus_values, empirical_coverage, label="coverage")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{variable} empirical coverage by bus")
        ax.set_xlabel("bus")
        ax.set_ylabel("coverage")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_coverage_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_density_curve(ax, width_mean)
        ax.set_title(f"{variable} mean interval width density")
        ax.set_xlabel("mean interval width")
        ax.set_ylabel("density")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_width_density.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_density_curve(ax, empirical_coverage, clip=(0.0, 1.0))
        ax.set_title(f"{variable} empirical coverage density")
        ax.set_xlabel("empirical coverage")
        ax.set_ylabel("density")
        ax.set_xlim(0.0, 1.0)
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
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_width_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(branch_values, empirical_coverage, label="coverage")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{variable} empirical coverage by branch")
        ax.set_xlabel("branch")
        ax.set_ylabel("coverage")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_coverage_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_density_curve(ax, width_mean)
        ax.set_title(f"{variable} mean interval width density")
        ax.set_xlabel("mean interval width")
        ax.set_ylabel("density")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{variable.lower()}_width_density.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_density_curve(ax, empirical_coverage, clip=(0.0, 1.0))
        ax.set_title(f"{variable} empirical coverage density")
        ax.set_xlabel("empirical coverage")
        ax.set_ylabel("density")
        ax.set_xlim(0.0, 1.0)
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

        fig, ax = plt.subplots(figsize=(10, 4))
        for stratum_value, stratum_group in ordered_group.groupby(
            "stratum_value", sort=True, dropna=False
        ):
            ax.plot(
                stratum_group[entity_column].to_numpy(dtype=int, copy=True),
                stratum_group["width_mean"].to_numpy(dtype=float, copy=True),
                label=_format_stratum_label(stratification, stratum_value),
            )
        ax.set_title(
            f"{variable} mean interval width by {entity_label} and {stratification}"
        )
        ax.set_xlabel(entity_label)
        ax.set_ylabel("mean interval width")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_width_summary_{suffix}.png", dpi=150
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        for stratum_value, stratum_group in ordered_group.groupby(
            "stratum_value", sort=True, dropna=False
        ):
            ax.plot(
                stratum_group[entity_column].to_numpy(dtype=int, copy=True),
                stratum_group["empirical_coverage"].to_numpy(dtype=float, copy=True),
                label=_format_stratum_label(stratification, stratum_value),
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"{variable} empirical coverage by {entity_label} and {stratification}"
        )
        ax.set_xlabel(entity_label)
        ax.set_ylabel("coverage")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_coverage_summary_{suffix}.png", dpi=150
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for stratum_value, stratum_group in ordered_group.groupby(
            "stratum_value", sort=True, dropna=False
        ):
            _plot_density_curve(
                ax,
                stratum_group["width_mean"].to_numpy(dtype=float, copy=True),
                label=_format_stratum_label(stratification, stratum_value),
                fill_alpha=0.22,
            )
        ax.set_title(f"{variable} mean interval width density by {stratification}")
        ax.set_xlabel("mean interval width")
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_width_density_{suffix}.png", dpi=150
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for stratum_value, stratum_group in ordered_group.groupby(
            "stratum_value", sort=True, dropna=False
        ):
            _plot_density_curve(
                ax,
                stratum_group["empirical_coverage"].to_numpy(dtype=float, copy=True),
                label=_format_stratum_label(stratification, stratum_value),
                clip=(0.0, 1.0),
                fill_alpha=0.22,
            )
        ax.set_title(f"{variable} empirical coverage density by {stratification}")
        ax.set_xlabel("empirical coverage")
        ax.set_ylabel("density")
        ax.set_xlim(0.0, 1.0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_coverage_density_{suffix}.png", dpi=150
        )
        plt.close(fig)

        per_stratum = (
            ordered_group.groupby("stratum_value", sort=True, dropna=False)
            .agg(
                width_mean_across_entities=("width_mean", "mean"),
                empirical_coverage_across_entities=("empirical_coverage", "mean"),
                n_entities=(entity_column, "nunique"),
            )
            .reset_index()
        )
        labels = [
            _format_stratum_label(stratification, value)
            for value in per_stratum["stratum_value"].tolist()
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            labels,
            per_stratum["width_mean_across_entities"].to_numpy(dtype=float, copy=True),
        )
        ax.set_title(
            f"{variable} mean interval width across {entity_label}s by {stratification}"
        )
        ax.set_xlabel(stratification)
        ax.set_ylabel("mean interval width")
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"{variable.lower()}_width_mean_{suffix}.png", dpi=150
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            labels,
            per_stratum["empirical_coverage_across_entities"].to_numpy(
                dtype=float, copy=True
            ),
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"{variable} empirical coverage across {entity_label}s by {stratification}"
        )
        ax.set_xlabel(stratification)
        ax.set_ylabel("coverage")
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
                "stratum_label": _format_stratum_label(stratification, stratum_value),
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
                    "stratum_label": _format_stratum_label(
                        stratification, stratum_value
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
                "stratum_label": _format_stratum_label(stratification, stratum_value),
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
        return f"{stratification}={normalized_value}"
    return f"{stratification}={str(stratum_value)}"


def _format_outage_type_value(value: int) -> str:
    labels = {
        0: "none",
        1: "branch_only",
        2: "gen_only",
        3: "mixed",
    }
    return labels.get(int(value), str(int(value)))


def _plot_density_curve(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    label: str | None = None,
    clip: tuple[float, float] | None = None,
    fill_alpha: float = 0.28,
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


def _scenario_label_column(
    column_name: str,
    repeated_values: np.ndarray,
) -> np.ndarray | None:
    if column_name != "outage_type":
        return None
    return np.array(
        [_format_outage_type_value(int(value)) for value in repeated_values.tolist()],
        dtype=object,
    )


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


def _load_analysis_ground_truth(
    args: argparse.Namespace,
    config: dict[str, object],
    scenario_ids: np.ndarray,
):
    analysis_pf_node = coalesce_option(args.pf_node, config, "analysis_pf_node")
    if analysis_pf_node is None:
        analysis_pf_node = coalesce_option(args.pf_node, config, "pf_node")
    analysis_pf_edge = coalesce_option(args.pf_edge, config, "analysis_pf_edge")
    if analysis_pf_edge is None:
        analysis_pf_edge = coalesce_option(args.pf_edge, config, "pf_edge")
    if analysis_pf_node is None or analysis_pf_edge is None:
        raise ValueError(
            "calibrate requires analysis ground-truth inputs via --pf-node/--pf-edge when direct targets are not available in the predictions parquet.",
        )
    return load_ground_truth(
        analysis_pf_node,
        analysis_pf_edge,
        scenario_ids=scenario_ids,
    )
