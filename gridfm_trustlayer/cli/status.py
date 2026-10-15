"""Status of each value and of each operating scenario against the screening limits.

The rule shared by `screen` (test scenarios, checked against the AC solution)
and `apply` (new scenarios, whose solution is unknown):

- a value is `within_limits` when its whole interval lies inside the limits,
  `violation` when its whole interval lies beyond a limit, and `uncertain` when
  the interval straddles a limit;
- an operating scenario takes the worst status of its values: `secure` when all
  are within limits, `insecure` when at least one is a violation, `uncertain`
  otherwise.

The point prediction is classified by the same rule, with the prediction in
place of both bounds, so it is only ever `within_limits` or `violation`
(`secure` or `insecure`). It is reported next to the interval for comparison.

The coverage guarantee holds per value; the scenario status combines many
values, so its error rates are measured by `screen`, not guaranteed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.cli import reports
from gridfm_trustlayer.screening import (
    ELEMENT_STATUSES,
    SCENARIO_STATUSES,
    status_codes,
)

METHODS = ("interval", "point")
_POINT_STATUSES = {
    "element": ("within_limits", "violation"),
    "scenario": ("secure", "insecure"),
}

# Stratum label of a value of a stratification column, from the frame's labels.
StratumLabel = Callable[[pd.DataFrame, str, object], str]


def classify(
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    lower_limit: float,
    upper_limit: float,
) -> tuple[pd.Categorical, pd.Categorical]:
    """Interval and point status of each value (`lower_limit` NaN: upper limit only)."""

    point_values = np.asarray(pred, float)
    return (
        pd.Categorical.from_codes(
            status_codes(lower, upper, lower_limit, upper_limit), ELEMENT_STATUSES
        ),
        pd.Categorical.from_codes(
            status_codes(point_values, point_values, lower_limit, upper_limit),
            ELEMENT_STATUSES,
        ),
    )


def _as_codes(statuses: pd.Series) -> np.ndarray:
    """Element statuses as 0, 1, 2 (within, uncertain, violation)."""

    return np.asarray(
        pd.Categorical(statuses, categories=ELEMENT_STATUSES).codes, dtype=np.int8
    )


# ------------------------------------------------------------------ counting
def element_counts(
    values: pd.DataFrame,
    strat_columns: Sequence[str],
    *,
    label: StratumLabel,
    truth: bool,
    applies: Callable[[str, str], bool] = lambda stratification, variable: True,
) -> pd.DataFrame:
    """Counts of one block of values per variable, stratum, method and status.

    `values` holds one row per screened value: `variable`, `status`,
    `point_status`, the stratification columns and, with `truth`, a boolean
    `truth` (the true value crosses a limit). Blocks add up in `combine_counts`.
    """

    parts: list[pd.DataFrame] = []
    for variable, rows in values.groupby("variable", sort=False, observed=True):
        work = pd.DataFrame(index=rows.index)
        work["interval"] = _as_codes(rows["status"])
        work["point"] = _as_codes(rows["point_status"])
        work["truth"] = rows["truth"].to_numpy(dtype=np.int64) if truth else 0
        columns = [
            c for c in strat_columns if c in rows.columns and applies(c, str(variable))
        ]
        for stratification in ["", *columns]:
            keys = (
                pd.Series(np.nan, index=rows.index, name="stratum_value")
                if not stratification
                else rows[stratification].rename("stratum_value")
            )
            for method in METHODS:
                grouped = (
                    work.groupby([keys, work[method].rename("code")], dropna=False)
                    .agg(n=("truth", "size"), truly_violated=("truth", "sum"))
                    .reset_index()
                )
                grouped["status"] = [ELEMENT_STATUSES[int(c)] for c in grouped["code"]]
                grouped["stratum"] = [
                    "all" if not stratification else label(rows, stratification, value)
                    for value in grouped["stratum_value"]
                ]
                grouped["variable"] = variable
                grouped["stratification"] = stratification
                grouped["method"] = method
                parts.append(grouped.drop(columns="code"))
    if not parts:
        return pd.DataFrame()
    counts = pd.concat(parts, ignore_index=True)
    counts.insert(0, "level", "element")
    return counts


def scenario_rows(
    values: pd.DataFrame, scenario_columns: Sequence[str], *, truth: bool
) -> pd.DataFrame:
    """One row per scenario and variable of a block: worst statuses and counts."""

    work = pd.DataFrame(
        {
            "scenario": values["scenario"].to_numpy(),
            "variable": values["variable"].to_numpy(),
            "interval": _as_codes(values["status"]),
            "point": _as_codes(values["point_status"]),
        }
    )
    work["n_uncertain"] = (work["interval"] == 1).astype(np.int64)
    work["n_violation"] = (work["interval"] == 2).astype(np.int64)
    aggregations: dict[str, tuple[str, str]] = {
        "interval": ("interval", "max"),
        "point": ("point", "max"),
        "n_uncertain": ("n_uncertain", "sum"),
        "n_violation": ("n_violation", "sum"),
    }
    if truth:
        work["truth"] = values["truth"].to_numpy(dtype=bool)
        aggregations["truth"] = ("truth", "any")
    present = [c for c in scenario_columns if c in values.columns]
    for column in present:
        work[column] = values[column].to_numpy()
        aggregations[column] = (column, "first")
    return (
        work.groupby(["scenario", "variable"], sort=False)
        .agg(**aggregations)
        .reset_index()
    )


def scenario_table(parts: Sequence[pd.DataFrame], *, truth: bool) -> pd.DataFrame:
    """One row per scenario: overall and per-variable statuses (as codes)."""

    rows = pd.concat(parts, ignore_index=True)
    variables = sorted(rows["variable"].unique(), key=reports.variable_rank)
    strata = [
        c
        for c in rows.columns
        if c
        not in {
            "scenario",
            "variable",
            "interval",
            "point",
            "n_uncertain",
            "n_violation",
            "truth",
        }
    ]
    table = rows.groupby("scenario", sort=True).agg(
        **{column: (column, "first") for column in strata},
        interval=("interval", "max"),
        point=("point", "max"),
        **({"truth": ("truth", "any")} if truth else {}),
    )
    for variable in variables:
        own = rows[rows["variable"] == variable].set_index("scenario")
        table[f"{variable}_interval"] = own["interval"]
        table[f"{variable}_point"] = own["point"]
        if truth:
            table[f"{variable}_truth"] = own["truth"]
        table[f"{variable}_n_uncertain"] = own["n_uncertain"]
        table[f"{variable}_n_violation"] = own["n_violation"]
    return table.reset_index()


def scenario_counts(
    table: pd.DataFrame,
    strat_columns: Sequence[str],
    *,
    label: StratumLabel,
    truth: bool,
) -> pd.DataFrame:
    """Scenario counts per variable (`all` for every variable), stratum, method and status."""

    variables = [
        c[: -len("_interval")] for c in table.columns if c.endswith("_interval")
    ]
    parts: list[pd.DataFrame] = []
    for variable in ["all", *variables]:
        prefix = "" if variable == "all" else f"{variable}_"
        scope = table[table[f"{prefix}interval"].notna()]
        columns = [c for c in strat_columns if c in scope.columns]
        for stratification in ["", *columns]:
            keys = (
                pd.Series(np.nan, index=scope.index, name="stratum_value")
                if not stratification
                else scope[stratification].rename("stratum_value")
            )
            truly = (
                scope[f"{prefix}truth"].astype(bool).astype(np.int64)
                if truth
                else pd.Series(0, index=scope.index)
            )
            for method in METHODS:
                codes = scope[f"{prefix}{method}"].astype(np.int64).rename("code")
                grouped = (
                    pd.DataFrame({"truly_violated": truly})
                    .groupby([keys, codes], dropna=False)
                    .agg(
                        n=("truly_violated", "size"),
                        truly_violated=("truly_violated", "sum"),
                    )
                    .reset_index()
                )
                grouped["status"] = [SCENARIO_STATUSES[int(c)] for c in grouped["code"]]
                grouped["stratum"] = [
                    "all" if not stratification else label(scope, stratification, value)
                    for value in grouped["stratum_value"]
                ]
                grouped["variable"] = variable
                grouped["stratification"] = stratification
                grouped["method"] = method
                parts.append(grouped.drop(columns="code"))
    counts = pd.concat(parts, ignore_index=True)
    counts.insert(0, "level", "scenario")
    return counts


def combine_counts(parts: Sequence[pd.DataFrame], *, truth: bool) -> pd.DataFrame:
    """Sum block counts, add the statuses that never occurred, and order the rows."""

    frames = [part for part in parts if len(part)]
    if not frames:
        return pd.DataFrame()
    counts = pd.concat(frames, ignore_index=True)
    keys = ["level", "variable", "stratification", "stratum_value", "method", "status"]
    counts = (
        counts.groupby(keys, sort=False, dropna=False)
        .agg(
            stratum=("stratum", "first"),
            n=("n", "sum"),
            truly_violated=("truly_violated", "sum"),
        )
        .reset_index()
    )
    # Every status of the level and method, zero when it never occurred.
    complete: list[dict[str, object]] = []
    for (level, variable, stratification, stratum_value), group in counts.groupby(
        ["level", "variable", "stratification", "stratum_value"],
        sort=False,
        dropna=False,
    ):
        statuses = {"element": ELEMENT_STATUSES, "scenario": SCENARIO_STATUSES}[
            str(level)
        ]
        for method in METHODS:
            names = statuses if method == "interval" else _POINT_STATUSES[str(level)]
            seen = set(group.loc[group["method"] == method, "status"])
            for status in names:
                if status not in seen:
                    complete.append(
                        {
                            "level": level,
                            "variable": variable,
                            "stratification": stratification,
                            "stratum_value": stratum_value,
                            "stratum": group["stratum"].iloc[0],
                            "method": method,
                            "status": status,
                            "n": 0,
                            "truly_violated": 0,
                        }
                    )
    if complete:
        counts = pd.concat([counts, pd.DataFrame(complete)], ignore_index=True)
    order = {
        name: index
        for index, name in enumerate((*ELEMENT_STATUSES, *SCENARIO_STATUSES))
    }
    counts = counts.assign(
        _level=counts["level"].map({"scenario": 0, "element": 1}),
        _variable=counts["variable"].map(
            lambda v: -1 if v == "all" else reports.variable_rank(v)
        ),
        _overall=counts["stratification"].ne(""),
        _method=counts["method"].map({"interval": 0, "point": 1}),
        _status=counts["status"].map(order),
    ).sort_values(
        [
            "_level",
            "_variable",
            "_overall",
            "stratification",
            "stratum_value",
            "_method",
            "_status",
        ],
        na_position="first",
    )
    counts = counts.drop(
        columns=["_level", "_variable", "_overall", "_method", "_status"]
    )
    columns = [
        "level",
        "variable",
        "stratification",
        "stratum_value",
        "stratum",
        "method",
        "status",
        "n",
    ]
    if truth:
        columns.append("truly_violated")
    return counts.loc[:, columns].reset_index(drop=True)


def labelled_scenarios(table: pd.DataFrame, *, truth: bool) -> pd.DataFrame:
    """The scenario table with status names instead of codes, for the CSV."""

    frame = table.copy()
    renames: dict[str, str] = {}
    for column in list(frame.columns):
        if column == "interval" or column.endswith("_interval"):
            frame[column] = _names(frame[column], SCENARIO_STATUSES)
            renames[column] = column.replace("interval", "status")
        elif column == "point" or column.endswith("_point"):
            frame[column] = _names(frame[column], SCENARIO_STATUSES)
            renames[column] = column.replace("point", "point_status")
        elif truth and (column == "truth" or column.endswith("_truth")):
            frame[column] = [
                None if pd.isna(v) else ("insecure" if bool(v) else "secure")
                for v in frame[column]
            ]
            renames[column] = column.replace("truth", "true_status")
        elif column.endswith(("_n_uncertain", "_n_violation")):
            frame[column] = frame[column].astype("Int64")
    return frame.rename(columns=renames)


def _names(codes: pd.Series, names: Sequence[str]) -> list[object]:
    return [None if pd.isna(code) else names[int(code)] for code in codes]


# ------------------------------------------------------------------ reporting
def _count(counts: pd.DataFrame, method: str, status: str) -> tuple[int, int]:
    rows = counts[(counts["method"] == method) & (counts["status"] == status)]
    n = int(rows["n"].sum())
    truly = int(rows["truly_violated"].sum()) if "truly_violated" in rows else 0
    return n, truly


def terminal_line(counts: pd.DataFrame, *, command: str, truth: bool) -> str:
    """E.g. `scenarios: 47 secure (3 truly insecure), 1,746 uncertain, ...`."""

    overall = counts[
        (counts["level"] == "scenario")
        & (counts["variable"] == "all")
        & (counts["stratification"] == "")
    ]
    parts: list[str] = []
    for method in METHODS:
        names = (
            SCENARIO_STATUSES if method == "interval" else _POINT_STATUSES["scenario"]
        )
        cells = []
        for status in names:
            n, truly = _count(overall, method, status)
            text = f"{n:,} {status}"
            if truth and status == "secure":
                text += f" ({truly:,} truly insecure)"
            elif truth and status == "insecure":
                text += f" ({n - truly:,} truly secure)"
            cells.append(text)
        lead = "scenarios" if method == "interval" else "point prediction"
        parts.append(f"{lead}: {', '.join(cells)}")
    return f"{command}: {'; '.join(parts)}"


def summary_sections(
    counts: pd.DataFrame, *, truth: bool, crosstab: pd.DataFrame | None = None
) -> list[str]:
    """Markdown sections: status per operating scenario, then per value."""

    lines = [
        "## Status per operating scenario",
        "",
        "A value is *within limits* when its whole interval lies inside the limits, a "
        "*violation* when its whole interval lies beyond a limit, and *uncertain* when "
        "the interval straddles a limit. An operating scenario is *secure* when all its "
        "values are within limits, *insecure* when at least one is a violation, and "
        "*uncertain* otherwise. The point prediction is classified the same way, using "
        "the predicted value alone. The coverage guarantee holds per value; per scenario, "
        + (
            "the rates below are measured on this test set, not guaranteed."
            if truth
            else "the rates are measured by `screen` on the test set, not guaranteed."
        ),
        "",
    ]
    scenarios = counts[counts["level"] == "scenario"]
    lines += _table_section(
        scenarios[scenarios["variable"] == "all"], "scenario", truth, by="stratum"
    )
    per_variable = scenarios[
        (scenarios["variable"] != "all") & (scenarios["stratification"] == "")
    ]
    if per_variable["variable"].nunique() > 1:
        lines += ["Per variable, over all scenarios:", ""]
        lines += _table_section(per_variable, "scenario", truth, by="variable")
    if crosstab is not None:
        lines += [
            "Interval status against point-prediction status, per scenario:",
            "",
            reports.markdown_table(crosstab),
            "",
        ]
    lines += ["## Status per value", ""]
    elements = counts[counts["level"] == "element"]
    for variable, rows in elements.groupby("variable", sort=False):
        lines += [f"### {variable}", ""]
        lines += _table_section(rows, "element", truth, by="stratum")
    return lines


def _table_section(
    rows: pd.DataFrame, level: str, truth: bool, *, by: str
) -> list[str]:
    if rows.empty:
        return []
    statuses = SCENARIO_STATUSES if level == "scenario" else ELEMENT_STATUSES
    point = _POINT_STATUSES[level]
    bad = "insecure" if level == "scenario" else "violated"
    good = "secure" if level == "scenario" else "not violated"
    records: list[dict[str, str]] = []
    for key, group in rows.groupby(by, sort=False):
        total = int(group.loc[group["method"] == "interval", "n"].sum())
        record = {
            by: str(key),
            "scenarios" if level == "scenario" else "values": f"{total:,}",
        }
        for method, names in (("interval", statuses), ("point", point)):
            for status in names:
                n, truly = _count(group, method, status)
                header = (
                    status.replace("_", " ")
                    if method == "interval"
                    else f"point: {status.replace('_', ' ')}"
                )
                if truth:
                    flagged = status in {"violation", "insecure"}
                    header += f" (truly {good})" if flagged else f" (truly {bad})"
                    record[header] = f"{n:,} ({(n - truly) if flagged else truly:,})"
                else:
                    share = f" ({n / total:.1%})" if total else ""
                    record[header] = f"{n:,}{share}"
        records.append(record)
    return [reports.markdown_table(pd.DataFrame.from_records(records)), ""]


def crosstab(table: pd.DataFrame) -> pd.DataFrame:
    """Scenarios by interval status (rows) and point-prediction status (columns)."""

    records = []
    for code, status in enumerate(SCENARIO_STATUSES):
        rows = table[table["interval"] == code]
        records.append(
            {
                "interval \\ point": status,
                "secure": f"{int((rows['point'] == 0).sum()):,}",
                "insecure": f"{int((rows['point'] == 2).sum()):,}",
            }
        )
    return pd.DataFrame.from_records(records)


def write_apply_summary(
    path: Path,
    counts: pd.DataFrame,
    table: pd.DataFrame,
    files: Sequence[tuple[str, str]],
    *,
    context: Sequence[str],
) -> None:
    """The `summary.md` of `apply`: statuses of the new scenarios, without truth."""

    lines = ["# Operating status", "", *context, ""]
    lines += summary_sections(counts, truth=False, crosstab=crosstab(table))
    lines += reports.figures_and_files(path, (), files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
