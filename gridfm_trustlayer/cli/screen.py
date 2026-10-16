"""`screen`: statuses and alarms of the test set, checked against the truth."""

from __future__ import annotations

import argparse
import glob
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.cli import common, reports, status
from gridfm_trustlayer.cli.common import (
    _ELEMENT_NAMES,
    _SCENARIOS_FILE_TEXT,
    _STATUS_FILE_TEXT,
    SCREEN_BOUND_TOLERANCE,
    _artifact_path,
    _block_jobs,
    _default_output_dir,
    _ground_truth_paths,
    _known_mask,
    _ordered_analysis_strat_columns,
    _parse_screen_bounds,
    _partition_variables,
    _report_progress,
    _scan_bus_types,
    _scenario_strata,
    _stratification_applies_to_variable,
    _stratum_label,
    _voltage_target_frame,
    _write_block,
    _write_status_tables,
    screen_limits,
)
from gridfm_trustlayer.cli.config import (
    coalesce_option,
    load_cli_config,
    normalize_variables,
    resolve_command_config,
    validate_command_config,
    validate_cp_type,
)
from gridfm_trustlayer.io import (
    compute_line_loading,
    load_ground_truth,
)
from gridfm_trustlayer.io.artifact import load_ground_truth_reference
from gridfm_trustlayer.io.ground_truth import (
    GroundTruthReference,
    scan_ground_truth_reference,
)
from gridfm_trustlayer.io.parquet import (
    clear_partitioned_output,
    column_names,
    is_partitioned_dataset,
    read_frame,
    read_scenario_ids,
    rows_per_scenario,
    scenario_blocks,
)
from gridfm_trustlayer.parallel import block_rows, map_blocks

# `screen` sees only the `<variable>_known` columns, not the task.
_BUS_KNOWN_WHERE = {"Vm": "PV buses", "Va": "the reference bus"}


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
    reports.clear_report(screen_dir)
    for previous in screen_dir.glob("screening_detail_*.parquet"):
        clear_partitioned_output(previous)
    n_jobs = _block_jobs(args, config)

    strata: list[pd.DataFrame] = []
    groups: list[pd.DataFrame] = []
    status_counts: list[pd.DataFrame] = []
    scenario_parts: list[pd.DataFrame] = []
    files: list[tuple[str, str]] = [
        (
            "screening_by_stratum.csv",
            "alarms, misses and false alarms per variable and stratum",
        ),
        ("screening_by_group.csv", "the same per bus type (voltages) or line rating"),
        ("scenarios.csv", _SCENARIOS_FILE_TEXT + ", and the true status"),
        ("status_by_stratum.csv", _STATUS_FILE_TEXT + ", against the truth"),
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
        missing = [v for v in channel_names if f"{v}_pred" not in columns]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} not found in the calibrated predictions "
                f"{input_path}: calibrate {'them' if len(missing) > 1 else 'it'} "
                "first (add to `variables`, then run compute and calibrate), or "
                "leave it out of the screened variables."
            )
        needed = {
            f"{variable}_{suffix}"
            for variable in channel_names
            for suffix in ("pred", "target", "pi_lower", "pi_upper")
        }
        if target_kind == "line":
            needed.update({"from_bus", "to_bus", "rate_a"})
        reference = None
        ground_truth_paths = None
        if not needed.issubset(columns):
            ground_truth_paths = _ground_truth_paths(args, config)
            reference = _screen_reference(config, ground_truth_paths)
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
        detail_path = screen_dir / detail_name
        clear_partitioned_output(
            detail_path, inputs=(input_path, *(ground_truth_paths or ()))
        )
        rows = rows_per_scenario(input_path, len(scenario_ids))
        blocks = scenario_blocks(
            scenario_ids,
            rows,
            block_rows(common._BLOCK_ROWS, rows * len(scenario_ids), n_jobs),
        )
        job = _ScreenJob(
            target_kind=target_kind,
            input_path=str(input_path),
            entity_column=entity_column,
            columns=[
                "scenario",
                entity_column,
                *sorted(needed),
                *extra,
                *known_columns,
            ],
            channel_names=tuple(channel_names),
            bounds_by_variable=bounds_by_variable,
            bus_types=bus_types,
            extra=extra,
            strat_columns=strat_columns,
            group_columns=group_columns,
            detail_path=str(detail_path),
            ground_truth_paths=ground_truth_paths,
            reference=reference,
        )
        results = map_blocks(
            job,
            blocks,
            n_jobs=n_jobs,
            progress=partial(
                _report_progress, f"screen ({target_kind})", blocks=blocks
            ),
        )
        counts = [result.counts for result in results]
        stratum_counts = [result.stratum_counts for result in results]
        status_counts += [result.status_counts for result in results]
        scenario_parts += [result.scenario_part for result in results]

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
                _DETAIL_FILE_TEXT,
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
    _, status_by_stratum = _write_status_tables(
        scenario_parts, status_counts, screen_dir, truth=True
    )
    save_figures = coalesce_option(args.save_figures, config, "save_figures")
    figures = (
        reports.render_figures(
            reports.screening_figure_tasks(by_stratum, screen_dir / "figures")
        )
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
        sections=status.summary_sections(status_by_stratum, truth=True),
    )
    _report_screening(by_stratum)
    print(
        status.terminal_line(status_by_stratum, command="screen", truth=True),
        flush=True,
    )
    print(f"screen: reports in {screen_dir}", flush=True)


_DETAIL_FILE_TEXT = (
    "one row per value screened: prediction, bounds, true value, the four flags "
    "and the statuses"
)


@dataclass
class _ScreenBlock:
    """What one block contributes to the screening reports."""

    counts: pd.DataFrame
    stratum_counts: pd.DataFrame
    status_counts: pd.DataFrame
    scenario_part: pd.DataFrame


@dataclass
class _ScreenJob:
    """`screen` on one block of one target family (see `parallel.BlockJob`).

    Writes the block's screening detail and returns its counts.
    """

    target_kind: str
    input_path: str
    entity_column: str
    columns: list[str]
    channel_names: tuple[str, ...]
    bounds_by_variable: dict[str, dict[str, float]]
    bus_types: dict[int, str]
    extra: list[str]
    strat_columns: list[str]
    group_columns: list[str]
    detail_path: str
    ground_truth_paths: tuple[str, str] | None
    reference: GroundTruthReference | None
    schemas: dict[str, Any] = field(default_factory=dict)

    def setup(self, threads: int | None) -> None:
        pass

    def run(self, index: int, block: np.ndarray) -> _ScreenBlock:
        frame = read_frame(self.input_path, self.columns, scenario_ids=block)
        if self.target_kind == "volt":
            frame = _ensure_voltage_screen_targets(
                frame,
                self.channel_names,
                self.ground_truth_paths,
                reference=self.reference,
            )
            detail_frame = _build_voltage_screen_detail_frame(
                frame,
                self.channel_names,
                self.bounds_by_variable,
                bus_types=self.bus_types,
                extra=self.extra,
            )
        else:
            frame = _ensure_line_screen_targets(
                frame,
                self.channel_names,
                self.ground_truth_paths,
                reference=self.reference,
            )
            detail_frame = _build_line_screen_detail_frame(
                frame, self.channel_names, self.bounds_by_variable, extra=self.extra
            )
        known = detail_frame["known"].to_numpy(dtype=bool)
        stratum_counts = _screen_stratum_counts(detail_frame, self.strat_columns)
        unknown = detail_frame[~known].drop(columns=["known"])
        _write_block(self.detail_path, unknown, index, self.schemas, "detail")
        values = unknown.assign(
            truth=unknown["true_lower_violation"] | unknown["true_upper_violation"]
        )
        return _ScreenBlock(
            counts=_screen_counts(unknown, self.group_columns),
            stratum_counts=stratum_counts,
            status_counts=status.element_counts(
                values,
                self.strat_columns,
                label=_stratum_label,
                truth=True,
                applies=_stratification_applies_to_variable,
            ),
            scenario_part=status.scenario_rows(
                values, _scenario_strata(self.extra), truth=True
            ),
        )


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
            # Values without a stratum (a known set-point has no prediction
            # bin) count only in the overall row.
            in_stratum = rows[stratification].notna()
            grouped = (
                work.loc[in_stratum, sums]
                .groupby(rows.loc[in_stratum, stratification].rename("stratum_value"))
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


def _resolve_screen_bounds(
    args: argparse.Namespace,
    config: dict[str, object],
    variables: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    bounds_by_variable = _parse_screen_bounds(config.get("screen_bounds"))
    _apply_screen_bound_overrides(args, variables, bounds_by_variable)
    for variable in variables:
        if variable not in bounds_by_variable:
            raise ValueError(
                f"Missing screening bounds for {variable!r}. Add it under screen_bounds in the config or override it from the CLI.",
            )
    return bounds_by_variable


def _apply_screen_bound_overrides(
    args: argparse.Namespace,
    variables: tuple[str, ...],
    bounds_by_variable: dict[str, dict[str, float]],
) -> None:
    """`--lower-bound` / `--upper-bound` of `screen`, for a single variable."""

    if args.lower_bound is None and args.upper_bound is None:
        return
    if len(variables) != 1:
        raise ValueError(
            "--lower-bound and --upper-bound can override bounds only when screening a single variable."
        )
    variable = variables[0]
    # A side not given on the command line comes from the config.
    current = bounds_by_variable.get(variable, {})
    lower = (
        float(args.lower_bound)
        if args.lower_bound is not None
        else current.get("lower", float("nan"))
    )
    upper = args.upper_bound if args.upper_bound is not None else current.get("upper")
    if upper is None or (variable != "line_loading" and math.isnan(lower)):
        raise ValueError(
            f"Missing a screening bound for {variable}: set both under screen_bounds, "
            "or give the missing one with --lower-bound / --upper-bound."
        )
    bounds_by_variable[variable] = screen_limits(variable, lower, float(upper))


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
    if shared_value is not None and (
        not Path(str(shared_value)).is_dir()
        or is_partitioned_dataset(Path(str(shared_value)))
    ):
        # A single-family `calibrate` writes its dataset at this very path.
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


def _screen_reference(
    config: dict[str, Any], ground_truth_paths: tuple[str, str]
) -> GroundTruthReference:
    """The calibration reference of the configured artifact, else a scan of the data."""

    try:
        artifact: str | None = _artifact_path(
            argparse.Namespace(), config, explicit=None, for_reading=True
        )
    except (ValueError, FileNotFoundError):
        artifact = None
    stored = None if artifact is None else load_ground_truth_reference(artifact)
    if stored is not None:
        return GroundTruthReference.from_dict(stored)
    return scan_ground_truth_reference(*ground_truth_paths)


def _ensure_voltage_screen_targets(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    ground_truth_paths: tuple[str, str] | None,
    *,
    reference: GroundTruthReference | None = None,
) -> pd.DataFrame:
    missing_targets = [
        variable for variable in variables if f"{variable}_target" not in frame.columns
    ]
    if not missing_targets:
        return frame

    assert ground_truth_paths is not None
    ground_truth = load_ground_truth(
        *ground_truth_paths,
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
    ground_truth_paths: tuple[str, str] | None,
    *,
    reference: GroundTruthReference | None = None,
) -> pd.DataFrame:
    missing_target = any(
        f"{variable}_target" not in frame.columns for variable in variables
    )
    missing_metadata = not {"from_bus", "to_bus", "rate_a"}.issubset(frame.columns)
    if not missing_target and not missing_metadata:
        return frame

    assert ground_truth_paths is not None
    ground_truth = load_ground_truth(
        *ground_truth_paths,
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
        # A value without a true solution cannot be checked: left out, as for
        # line loading (and from the coverage of `calibrate`).
        solved = frame[np.isfinite(frame[f"{variable}_target"].to_numpy(dtype=float))]
        rows = solved.loc[:, ["scenario", "bus", *extra]].copy()
        rows["bus_type"] = rows["bus"].map(bus_types).fillna("Unknown").astype(object)
        records.append(
            _screen_detail_rows(rows, solved, variable, bounds_by_variable[variable])
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
        f"{variable}_pred",
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
    pred = frame[f"{variable}_pred"]
    rows["variable"] = variable
    rows["pred"] = pred.to_numpy()
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
    rows["status"], rows["point_status"] = status.classify(
        pred.to_numpy(dtype=float),
        lower.to_numpy(dtype=float),
        upper.to_numpy(dtype=float),
        float("nan") if variable == "line_loading" else lower_limit,
        upper_limit,
    )
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
    """Per-entity screening table (`screening_by_element_*.csv`) from the blocks' counts."""

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
