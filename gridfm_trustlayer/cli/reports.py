"""Reports of `calibrate` and `screen`: tables, figures and a summary page.

The per-element statistics are computed in `cli.calibrate` and `cli.screen`;
this module turns them into the files a user reads: CSV tables (rounded, one row per variable and
stratum, group or element), a few figures per variable, and `summary.md`, a
one-page overview that links to the rest.

Known values (set-points such as `Vm` at PV buses) are already excluded from
every statistic here; the tables report how many there were (`n_known`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# Units of the variables, and the factor that converts stored values to them.
UNITS: dict[str, tuple[str, float]] = {
    "Vm": ("p.u.", 1.0),
    "Va": ("degrees", 180.0 / math.pi),
    "line_loading": ("fraction of rating", 1.0),
}
# The limit sides screened per variable: `line_loading` has only an upper limit.
SCREEN_SIDES: dict[str, tuple[str, ...]] = {
    "line_loading": ("upper",),
}
DEFAULT_SCREEN_SIDES = ("lower", "upper", "any")
# 95% normal quantile, for approximate confidence intervals on coverage.
_Z95 = 1.96
# Elements with fewer values have too noisy a coverage to report as the worst.
MIN_ELEMENT_VALUES = 50
# Order of the variables in reports; others follow in order of appearance.
_VARIABLE_ORDER = {"Vm": 0, "Va": 1, "line_loading": 2}
_COLORS = {"main": "#2f6db3", "target": "#c0392b", "box": "#a9c6e8"}


# ------------------------------------------------------------------ helpers
# The files a report folder gets from `calibrate`, `screen` or `apply`.
_REPORT_FILES = (
    "summary.md",
    "coverage_by_*.csv",
    "screening_by_*.csv",
    "scenarios.csv",
    "status_by_*.csv",
    "figures/*.png",
)


def clear_report(directory: Path) -> None:
    """Remove the report files of a previous run from `directory`.

    A rerun with fewer variables would otherwise leave, e.g., the line-loading
    tables and figures of the previous run next to the new summary. Only the
    files trustlayer writes are removed.
    """

    for pattern in _REPORT_FILES:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a report table with floats rounded to 6 significant digits."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g")


# Horizontal box plots: `orientation` exists from matplotlib 3.10, and `vert`,
# its predecessor, is deprecated from 3.11 and removed in 3.13.
_HORIZONTAL: dict[str, Any] = (
    {"orientation": "horizontal"}
    if tuple(int(part) for part in matplotlib.__version__.split(".")[:2]) >= (3, 10)
    else {"vert": False}
)


def unit_of(variable: str) -> tuple[str, float]:
    return UNITS.get(variable, ("", 1.0))


def family_name(entity_column: str) -> str:
    """File-name suffix of a target family: `voltage` (buses) or `line_loading`."""

    return "line_loading" if entity_column == "branch" else "voltage"


def sort_by_stratum(frame: pd.DataFrame) -> pd.DataFrame:
    order = {
        name: _VARIABLE_ORDER.get(name, len(_VARIABLE_ORDER) + index)
        for index, name in enumerate(dict.fromkeys(frame["variable"]))
    }
    return (
        frame.assign(
            _variable_order=frame["variable"].map(order),
            _overall=frame["stratification"].ne(""),
        )
        .sort_values(
            ["_variable_order", "_overall", "stratification", "stratum_value"],
            na_position="first",
        )
        .drop(columns=["_variable_order", "_overall"])
    )


def sort_by_variable(frame: pd.DataFrame) -> pd.DataFrame:
    """Vm, Va, then line_loading; rows of each variable keep their order."""

    rank = frame["variable"].map(_VARIABLE_ORDER).fillna(len(_VARIABLE_ORDER))
    return frame.iloc[np.argsort(rank.to_numpy(), kind="stable")].reset_index(drop=True)


def _short_stratum(stratum: str) -> str:
    """`n_outages=2` -> `2`, for axis labels; `all` stays `all`."""

    return stratum.split("=", 1)[1] if "=" in stratum else stratum


def _stratification_name(frame: pd.DataFrame) -> str:
    names = [name for name in frame["stratification"].unique() if name]
    return names[0] if names else ""


# ------------------------------------------------------------------ coverage
def _finite_mean_width(group: pd.DataFrame) -> float:
    """Mean width over the values with finite bounds.

    Per value, not per element: an element with some infinite bounds (a sparse
    stratum with `sparse_strata: infinite`) still counts with its finite ones.
    """

    if "finite_width_count" in group:
        count = float(group["finite_width_count"].sum())
        return float(group["finite_width_sum"].sum()) / count if count else math.nan
    widths = group["mean_width"].to_numpy(dtype=float)
    finite = np.isfinite(widths)
    return (
        float(np.average(widths[finite], weights=group["n_values"][finite]))
        if finite.any()
        else math.nan
    )


def coverage_by_stratum(elements: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Coverage and width per variable and stratum, from per-element statistics.

    Coverage pools all values of the stratum. Its 95% interval is approximate:
    it treats the scenarios of the stratum as independent draws and the elements
    of one scenario as a single draw. A scenario's mean coverage varies no more
    than one indicator does, so the interval is conservative.
    """

    records: list[dict[str, object]] = []
    keys = ["variable", "stratification", "stratum_value"]
    for (variable, stratification, stratum_value), group in elements.groupby(
        keys, sort=False, dropna=False
    ):
        observed = group[group["n_values"] > 0]
        n_values = int(observed["n_values"].sum())
        coverage = (
            float((observed["coverage"] * observed["n_values"]).sum() / n_values)
            if n_values
            else math.nan
        )
        n_scenarios = int(group["n_scenarios"].max()) if len(group) else 0
        half_width = (
            _Z95 * math.sqrt(coverage * (1.0 - coverage) / n_scenarios)
            if n_scenarios and not math.isnan(coverage)
            else math.nan
        )
        widths = observed["mean_width"].to_numpy(dtype=float)
        finite = np.isfinite(widths)
        records.append(
            {
                "variable": variable,
                "stratification": stratification,
                "stratum_value": stratum_value,
                "stratum": group["stratum"].iloc[0],
                "n_scenarios": n_scenarios,
                "n_elements": len(observed),
                "n_values": n_values,
                "n_known": int(group["n_known"].sum()),
                "n_pooled": int(group["n_pooled"].sum()),
                "target_coverage": 1.0 - float(alpha),
                "coverage": coverage,
                "coverage_low": max(0.0, coverage - half_width),
                "coverage_high": min(1.0, coverage + half_width),
                "worst_element_coverage": _worst(observed),
                "mean_width": _finite_mean_width(observed),
                "n_elements_infinite": int((~finite).sum()),
            }
        )
    return sort_by_stratum(pd.DataFrame.from_records(records))


def _worst(elements: pd.DataFrame) -> float:
    """Lowest coverage among elements with at least `MIN_ELEMENT_VALUES` values."""

    reliable = elements[elements["n_values"] >= MIN_ELEMENT_VALUES]
    return float(reliable["coverage"].min()) if len(reliable) else math.nan


def coverage_by_group(elements: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    """Coverage and width per bus type (voltages) or line rating (line loading)."""

    group_column = "rate_a" if entity_column == "branch" else "bus_type"
    records: list[dict[str, object]] = []
    observed = elements[elements["n_values"] > 0]
    keys = ["variable", "stratification", "stratum_value", group_column]
    for (variable, stratification, stratum_value, value), group in observed.groupby(
        keys, sort=False, dropna=False
    ):
        n_values = int(group["n_values"].sum())
        records.append(
            {
                "variable": variable,
                "stratification": stratification,
                "stratum_value": stratum_value,
                "stratum": group["stratum"].iloc[0],
                "group": f"{group_column}={value:g}"
                if group_column == "rate_a"
                else f"{group_column}={value}",
                "_group_value": value,
                "n_elements": len(group),
                "n_values": n_values,
                "coverage": float(
                    (group["coverage"] * group["n_values"]).sum() / n_values
                ),
                "worst_element_coverage": _worst(group),
                "mean_width": _finite_mean_width(group),
            }
        )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    order = {
        name: _VARIABLE_ORDER.get(name, len(_VARIABLE_ORDER) + index)
        for index, name in enumerate(dict.fromkeys(frame["variable"]))
    }
    return (
        frame.assign(
            _variable_order=frame["variable"].map(order),
            _overall=frame["stratification"].ne(""),
        )
        .sort_values(
            [
                "_variable_order",
                "_overall",
                "stratification",
                "stratum_value",
                "_group_value",
            ],
            na_position="first",
        )
        .drop(columns=["_variable_order", "_overall", "_group_value"])
    )


def coverage_by_element(elements: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    attributes = (
        ["from_bus", "to_bus", "rate_a"] if entity_column == "branch" else ["bus_type"]
    )
    known = [] if entity_column == "branch" else ["n_known"]
    columns = [
        "variable",
        "stratum",
        entity_column,
        *attributes,
        "n_values",
        *known,
        "coverage",
        "mean_width",
        "min_width",
        "max_width",
    ]
    return sort_by_stratum(elements).loc[:, columns]


# A figure to draw: a function and its arguments, the last of which is the
# output path. The figures of a report take under a second together, less than
# starting worker processes, so they are drawn one after the other.
FigureTask = tuple[Callable[..., Path], tuple[Any, ...]]


def render_figures(tasks: Sequence[FigureTask]) -> list[Path]:
    """Draw the figures; returns their paths in task order."""

    for _, arguments in tasks:
        Path(arguments[-1]).parent.mkdir(parents=True, exist_ok=True)
    return [function(*arguments) for function, arguments in tasks]


def coverage_figure_tasks(
    by_stratum: pd.DataFrame,
    elements: pd.DataFrame,
    directory: Path,
    *,
    entity_column: str,
) -> list[FigureTask]:
    """Three figures per variable: coverage by stratum, per element, width by stratum."""

    tasks: list[FigureTask] = []
    entity_label = "line" if entity_column == "branch" else "bus"
    for variable, rows in by_stratum.groupby("variable", sort=False):
        variable_elements = elements[
            (elements["variable"] == variable) & (elements["n_values"] > 0)
        ]
        stem = variable.lower()
        tasks.append(
            (
                _coverage_by_stratum_figure,
                (rows, directory / f"{stem}_coverage_by_stratum.png"),
            )
        )
        tasks.append(
            (
                _coverage_by_element_figure,
                (
                    variable,
                    variable_elements[variable_elements["stratification"] == ""],
                    float(rows["target_coverage"].iloc[0]),
                    entity_label,
                    directory / f"{stem}_coverage_by_{entity_label}.png",
                ),
            )
        )
        tasks.append(
            (
                _width_by_stratum_figure,
                (
                    variable,
                    variable_elements,
                    directory / f"{stem}_width_by_stratum.png",
                ),
            )
        )
    return tasks


def _coverage_by_stratum_figure(rows: pd.DataFrame, path: Path) -> Path:
    variable = rows["variable"].iloc[0]
    target = float(rows["target_coverage"].iloc[0])
    labels = [_short_stratum(stratum) for stratum in rows["stratum"]]
    positions = np.arange(len(rows))[::-1]
    coverage = rows["coverage"].to_numpy(dtype=float)
    error = np.vstack(
        [
            coverage - rows["coverage_low"].to_numpy(dtype=float),
            rows["coverage_high"].to_numpy(dtype=float) - coverage,
        ]
    )
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.8))
    ax.axvline(
        target,
        color=_COLORS["target"],
        linestyle="--",
        linewidth=1.3,
        label=f"target {target:.0%}",
    )
    ax.errorbar(
        coverage,
        positions,
        xerr=error,
        fmt="o",
        color=_COLORS["main"],
        capsize=4,
        label="coverage (95% interval)",
    )
    for position, value in zip(positions, coverage, strict=True):
        ax.annotate(
            f"{value:.3f}",
            (value, position),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_yticks(positions, labels)
    stratification = _stratification_name(rows)
    ax.set_ylabel(stratification or "")
    ax.set_xlabel("coverage (share of true values inside the bounds)")
    low = min(float(np.nanmin(rows["coverage_low"])), target)
    high = max(float(np.nanmax(rows["coverage_high"])), target)
    margin = max(0.01, 0.15 * (high - low))
    ax.set_xlim(max(0.0, low - margin), min(1.0, high + margin))
    ax.set_ylim(-0.7, len(rows) - 0.3)
    title = f"{variable}: coverage" + (
        f" by {stratification}" if stratification else ""
    )
    ax.set_title(title)
    _finish(ax, legend=True)
    return _save(fig, path)


def _coverage_by_element_figure(
    variable: str,
    elements: pd.DataFrame,
    target: float,
    entity_label: str,
    path: Path,
) -> Path:
    coverage = elements["coverage"].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    if coverage.size:
        ax.hist(
            coverage,
            bins=min(40, max(10, coverage.size // 4)),
            color=_COLORS["box"],
            edgecolor="white",
        )
    ax.axvline(
        target,
        color=_COLORS["target"],
        linestyle="--",
        linewidth=1.3,
        label=f"target {target:.0%}",
    )
    ax.set_xlabel(f"coverage of each {entity_label}")
    ax.set_ylabel(
        f"number of {entity_label}es" if entity_label == "bus" else "number of lines"
    )
    ax.set_title(
        f"{variable}: coverage per {entity_label} ({coverage.size} {entity_label}"
        f"{'es' if entity_label == 'bus' else 's'})"
    )
    _finish(ax, legend=True)
    return _save(fig, path)


def _width_by_stratum_figure(variable: str, elements: pd.DataFrame, path: Path) -> Path:
    unit, factor = unit_of(variable)
    strata = elements.drop_duplicates(["stratification", "stratum_value"])
    strata = sort_by_stratum(strata)
    data, labels, infinite = [], [], 0
    for _, stratum in strata.iterrows():
        rows = elements[
            (elements["stratification"] == stratum["stratification"])
            & (
                elements["stratum_value"].fillna(-1)
                == (
                    -1
                    if pd.isna(stratum["stratum_value"])
                    else stratum["stratum_value"]
                )
            )
        ]
        widths = rows["mean_width"].to_numpy(dtype=float) * factor
        infinite += int((~np.isfinite(widths)).sum())
        data.append(widths[np.isfinite(widths)])
        labels.append(_short_stratum(str(stratum["stratum"])))
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(data) + 1.8))
    positions = np.arange(len(data))[::-1]
    box = ax.boxplot(
        data,
        positions=positions,
        **_HORIZONTAL,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": _COLORS["main"],
            "markeredgecolor": _COLORS["main"],
            "markersize": 4,
        },
        flierprops={"markersize": 3, "alpha": 0.5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(_COLORS["box"])
    ax.set_yticks(positions, labels)
    ax.set_ylabel(_stratification_name(elements) or "")
    ax.set_xlabel(
        f"mean interval width per element ({unit})" if unit else "mean interval width"
    )
    title = f"{variable}: interval width" + (
        f" by {_stratification_name(elements)}"
        if _stratification_name(elements)
        else ""
    )
    if infinite:
        title += f" ({infinite} element-strata with infinite bounds not shown)"
    ax.set_title(title)
    _finish(ax)
    return _save(fig, path)


# ----------------------------------------------------------------- screening
def side_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """0/1 indicators of caught, missed, false and cleared alarms, per side."""

    counts = pd.DataFrame(index=frame.index)
    flags = {
        side: (
            frame[f"pred_{side}_violation"].to_numpy(dtype=bool),
            frame[f"true_{side}_violation"].to_numpy(dtype=bool),
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
    return counts


def screening_long(
    counts: pd.DataFrame, id_columns: Sequence[str], *, keep: Sequence[str] = ()
) -> pd.DataFrame:
    """Wide confusion counts -> one row per side, with readable names and rates."""

    records: list[dict[str, object]] = []
    for _, row in counts.iterrows():
        for side in SCREEN_SIDES.get(str(row["variable"]), DEFAULT_SCREEN_SIDES):
            caught, false_alarms = int(row[f"{side}_tp"]), int(row[f"{side}_fp"])
            missed, cleared = int(row[f"{side}_fn"]), int(row[f"{side}_tn"])
            violations, alarms = caught + missed, caught + false_alarms
            record: dict[str, object] = {column: row[column] for column in id_columns}
            record.update({"side": side})
            record.update({column: row[column] for column in keep})
            record.update(
                {
                    "violations": violations,
                    "alarms": alarms,
                    "caught": caught,
                    "missed": missed,
                    "false_alarms": false_alarms,
                    "cleared": cleared,
                    "recall": caught / violations if violations else math.nan,
                    "precision": caught / alarms if alarms else math.nan,
                }
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def screening_figure_tasks(
    by_stratum: pd.DataFrame, directory: Path
) -> list[FigureTask]:
    """Per variable: the confusion matrix, and recall and precision by stratum."""

    tasks: list[FigureTask] = []
    for variable, rows in by_stratum.groupby("variable", sort=False):
        side = "upper" if variable == "line_loading" else "any"
        rows = rows[rows["side"] == side]
        overall = rows[rows["stratification"] == ""].iloc[0]
        tasks.append(
            (
                _confusion_figure,
                (
                    variable,
                    side,
                    overall,
                    directory / f"{variable.lower()}_confusion.png",
                ),
            )
        )
        if (rows["stratification"] != "").any():
            tasks.append(
                (
                    _screening_by_stratum_figure,
                    (
                        variable,
                        rows,
                        directory / f"{variable.lower()}_screening_by_stratum.png",
                    ),
                )
            )
    return tasks


CONFUSION_ROWS = ("true violation", "no violation")
CONFUSION_COLUMNS = ("alarm (bound crosses limit)", "no alarm")


def confusion_cells(row: pd.Series) -> list[list[tuple[str, int]]]:
    """The confusion matrix as drawn: rows `CONFUSION_ROWS`, columns `CONFUSION_COLUMNS`."""

    return [
        [("caught", int(row["caught"])), ("missed", int(row["missed"]))],
        [("false alarm", int(row["false_alarms"])), ("cleared", int(row["cleared"]))],
    ]


def _confusion_figure(variable: str, side: str, row: pd.Series, path: Path) -> Path:
    cells = confusion_cells(row)
    colors = [["#d5ecd4", "#f6d0cc"], ["#f6d0cc", "#d5ecd4"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i in range(2):
        row_total = sum(count for _, count in cells[i])
        for j in range(2):
            name, count = cells[i][j]
            ax.add_patch(
                plt.Rectangle(
                    (j, 1 - i),
                    1,
                    1,
                    facecolor=colors[i][j],
                    edgecolor="white",
                    linewidth=3,
                )
            )
            share = f"{count / row_total:.1%} of row" if row_total else ""
            ax.text(
                j + 0.5,
                1.5 - i + 0.12,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=15,
                weight="bold",
            )
            ax.text(
                j + 0.5,
                1.5 - i - 0.12,
                f"{name}\n{share}",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], list(CONFUSION_COLUMNS))
    ax.set_yticks([1.5, 0.5], list(CONFUSION_ROWS))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    limit = "upper limit" if side == "upper" else "either limit"
    recall = row["recall"]
    precision = row["precision"]
    ax.set_title(
        f"{variable}, {limit}: recall {recall:.3f}, precision {precision:.3f}"
        if not (pd.isna(recall) or pd.isna(precision))
        else f"{variable}, {limit}"
    )
    return _save(fig, path)


def _screening_by_stratum_figure(variable: str, rows: pd.DataFrame, path: Path) -> Path:
    labels = [_short_stratum(stratum) for stratum in rows["stratum"]]
    positions = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.45 * len(rows) + 1.8), sharey=True)
    for ax, metric, help_text in (
        (axes[0], "recall", "share of violations caught"),
        (axes[1], "precision", "share of alarms that are violations"),
    ):
        values = rows[metric].to_numpy(dtype=float)
        ax.barh(positions, np.nan_to_num(values), color=_COLORS["box"], height=0.6)
        for position, value in zip(positions, values, strict=True):
            ax.text(
                min(np.nan_to_num(value), 1.0) + 0.01,
                position,
                "n/a" if np.isnan(value) else f"{value:.3f}",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, 1.12)
        ax.set_xlabel(f"{metric} ({help_text})")
        _finish(ax)
    axes[0].set_yticks(positions, labels)
    axes[0].set_ylabel(_stratification_name(rows) or "")
    fig.suptitle(f"{variable}: screening by {_stratification_name(rows)}")
    return _save(fig, path)


# -------------------------------------------------------------- summary page
def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


def _fmt(value: object, digits: int = 3) -> str:
    """Numbers for the summary page: counts with separators, rates rounded."""

    if value is None:
        return "–"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    number = float(cast(float, value))
    if math.isnan(number):
        return "–"
    if math.isinf(number):
        return "∞"
    return f"{number:.{digits}g}" if 0 < abs(number) < 1e-3 else f"{number:.{digits}f}"


def _fmt_width(value: float) -> str:
    if math.isnan(value):
        return "–"
    return "∞" if math.isinf(value) else f"{value:.4g}"


def known_sentence(by_stratum: pd.DataFrame, known_where: Mapping[str, str]) -> str:
    """One sentence on the known values left out of the metrics, or ''."""

    overall = by_stratum[by_stratum["stratification"] == ""]
    overall = sort_by_stratum(overall)
    parts = [
        f"{int(row['n_known']):,} values of `{row['variable']}` at {known_where.get(row['variable'], 'known entries')}"
        for _, row in overall.drop_duplicates("variable").iterrows()
        if int(row["n_known"]) > 0
    ]
    if not parts:
        return ""
    return (
        "**Known values are not counted.** "
        + "; ".join(parts)
        + " are set-points: inputs of the problem, which the bounds contain by construction. "
        "They are left out of every coverage, width and screening figure, and counted in the "
        "`n_known` columns."
    )


def write_coverage_summary(
    path: Path,
    by_stratum: pd.DataFrame,
    figures: Sequence[Path],
    files: Sequence[tuple[str, str]],
    *,
    context: Sequence[str],
    known_where: Mapping[str, str],
) -> None:
    lines = ["# Coverage report", "", *context, ""]
    sentence = known_sentence(by_stratum, known_where)
    if sentence:
        lines += [sentence, ""]
    lines += ["## Coverage by stratum", ""]
    by_stratum = sort_by_stratum(by_stratum)
    for variable, rows in by_stratum.groupby("variable", sort=False):
        unit, factor = unit_of(variable)
        table = pd.DataFrame(
            {
                "stratum": rows["stratum"],
                "coverage (95% interval)": [
                    f"{_fmt(c)} ({_fmt(lo)}–{_fmt(hi)})"
                    for c, lo, hi in zip(
                        rows["coverage"],
                        rows["coverage_low"],
                        rows["coverage_high"],
                        strict=True,
                    )
                ],
                f"worst element (≥ {MIN_ELEMENT_VALUES} values)": [
                    _fmt(v) for v in rows["worst_element_coverage"]
                ],
                f"mean width ({unit})" if unit else "mean width": [
                    _fmt_width(v * factor) for v in rows["mean_width"]
                ],
                "elements with infinite bounds": [
                    _fmt(v) for v in rows["n_elements_infinite"]
                ],
                "values": [_fmt(v) for v in rows["n_values"]],
            }
        )
        if rows["n_pooled"].sum() > 0:
            table["with pooled bounds"] = [_fmt(v) for v in rows["n_pooled"]]
        target = float(rows["target_coverage"].iloc[0])
        lines += [
            f"### {variable}",
            "",
            f"Target coverage: {target:.0%}.",
            "",
            markdown_table(table),
            "",
        ]
    lines += figures_and_files(path, figures, files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_screening_summary(
    path: Path,
    by_stratum: pd.DataFrame,
    figures: Sequence[Path],
    files: Sequence[tuple[str, str]],
    *,
    context: Sequence[str],
    known_where: Mapping[str, str],
    sections: Sequence[str] = (),
) -> None:
    lines = ["# Screening report", "", *context, ""]
    sentence = known_sentence(by_stratum, known_where)
    if sentence:
        lines += [sentence, ""]
    lines += [
        "## Alarms per value",
        "",
        "An alarm is raised when a bound crosses a limit; it is checked against the true value.",
        "",
    ]
    by_stratum = sort_by_stratum(by_stratum)
    for variable, rows in by_stratum.groupby("variable", sort=False):
        side = "upper" if variable == "line_loading" else "any"
        rows = rows[rows["side"] == side]
        table = pd.DataFrame(
            {
                "stratum": rows["stratum"],
                "violations": [_fmt(v) for v in rows["violations"]],
                "caught": [_fmt(v) for v in rows["caught"]],
                "missed": [_fmt(v) for v in rows["missed"]],
                "false alarms": [_fmt(v) for v in rows["false_alarms"]],
                "recall": [_fmt(v) for v in rows["recall"]],
                "precision": [_fmt(v) for v in rows["precision"]],
                "values": [_fmt(v) for v in rows["n_values"]],
            }
        )
        limit = "upper limit" if side == "upper" else "either limit"
        lines += [f"### {variable} ({limit})", "", markdown_table(table), ""]
    lines += list(sections)
    lines += figures_and_files(path, figures, files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figures_and_files(
    path: Path, figures: Sequence[Path], files: Sequence[tuple[str, str]]
) -> list[str]:
    lines: list[str] = []
    if figures:
        lines += ["## Figures", ""]
        for figure in sorted(figures, key=lambda f: _variable_rank(f.stem)):
            relative = figure.relative_to(path.parent).as_posix()
            lines += [f"![{figure.stem}]({relative})", ""]
    lines += ["## Files", ""]
    # Tables first, then the datasets; voltages before lines within each.
    ordered = sorted(
        files,
        key=lambda f: (f[0].endswith(".parquet"), "line_loading" in f[0]),
    )
    lines += [f"- `{name}`: {description}" for name, description in ordered]
    return lines


def variable_rank(variable: str) -> int:
    """Report order of a variable: Vm, Va, line_loading, then any other."""

    return _VARIABLE_ORDER.get(variable, len(_VARIABLE_ORDER))


def _variable_rank(stem: str) -> int:
    """Position of a figure in the report, from its `<variable>_` prefix."""

    for variable, rank in _VARIABLE_ORDER.items():
        if stem.startswith(f"{variable.lower()}_"):
            return rank
    return len(_VARIABLE_ORDER)


# ------------------------------------------------------------------ plotting
def _finish(ax: plt.Axes, *, legend: bool = False) -> None:
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="best")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
