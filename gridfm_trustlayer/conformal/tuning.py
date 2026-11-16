"""Cross-validated choice of the kernel settings on the calibration set alone.

`cp_type: kernel` with `bandwidth: n_eff` localizes each test scenario to an
effective sample size `n_eff_target`, measured with a `distance` between scenario
embeddings. Kernel-weighted CP has no finite-sample guarantee, so whether a given
setting reaches the coverage goal has to be checked empirically.
`tune_n_eff_target` does that without touching the test data: it splits the
calibration scenarios into K folds, fits the kernel on K-1 folds and evaluates
coverage and interval width on the held-out fold, for every candidate.

A candidate is an (`n_eff_target`, `distance`) pair. By default only
`n_eff_target` varies (the config's `distance` is kept); listing several
`distances` adds the second dimension. Candidates come from:

* ``search="grid"``: every combination of ``grid`` and ``distances``;
* ``search="random"``: ``budget`` distinct combinations, with ``n_eff_target`` drawn
  log-uniformly from ``n_eff_range`` (so not limited to grid values) and the
  distance uniformly from ``distances``. Useful to bound the cost when both
  dimensions vary.

Selection rule (coverage first, then efficiency):

1. A candidate is *eligible* if its cross-validated coverage reaches
   ``1 - alpha - tolerance`` overall and in every stratum, for every channel.
   With ``coverage_tolerance="auto"`` the tolerance of each check is two
   standard errors of a coverage estimate from that many scenarios,
   ``2 * sqrt(alpha * (1 - alpha) / n_scenarios)``; a number sets a fixed tolerance.
2. Among eligible candidates, the one with the narrowest intervals is chosen.
   Widths are relative to the evaluated candidate with the largest
   ``n_eff_target`` (the one closest to uniform weights), so candidates with
   different distances are compared on one scale.
3. If no candidate is eligible, the kernel does not reach the target on this
   data. The target is then set to the largest calibration stratum, which makes
   the weights uniform within every stratum, i.e. equivalent to `cp_type: element`.

`n_eff_target` is an absolute effective sample size, so a value chosen on folds
holding (K-1)/K of the calibration data carries over to the final fit on all of it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.conformal.calibrator import DEFAULT_SPARSE_STRATA, ChannelSet
from gridfm_trustlayer.conformal.kernel import (
    FALLBACK_WARNING_SHARE,
    SUPPORTED_DISTANCES,
    KernelFallbackWarning,
    KernelIntervalCalibrator,
    _predict_targets,
)
from gridfm_trustlayer.parallel import resolve_threads

DEFAULT_TUNING_GRID = (100.0, 200.0, 400.0, 600.0, 800.0, 1200.0)
DEFAULT_TUNING_FOLDS = 5
DEFAULT_TUNING_BUDGET = 10
TUNING_GROUP_BY = ("auto", "load_scenario_idx", "none")
TUNING_SEARCHES = ("grid", "random")
REASON_SMALLEST_WIDTH = "narrowest eligible candidate"
REASON_NO_CANDIDATE = (
    "no candidate met the coverage target: uniform weights (equivalent to element)"
)


@dataclass(frozen=True)
class TuningOptions:
    """Options of the cross-validated bandwidth choice (the config's `tuning:` block).

    Attributes:
        folds: Number of cross-validation folds within the calibration set.
        grid: Candidate `n_eff_target` values for `search="grid"`.
        distances: Distances to compare; `None` keeps the given distance.
        search: `"grid"` or `"random"`.
        budget: Number of settings tried by `search="random"`.
        n_eff_range: `(low, high)` range for `search="random"`.
        group_by: `"auto"`, `"load_scenario_idx"` or `"none"`.
        seed: Seed of the fold assignment and the random search.
        coverage_tolerance: Allowed shortfall below `1 - alpha`, or `"auto"`.
    """

    folds: int = DEFAULT_TUNING_FOLDS
    grid: tuple[float, ...] = DEFAULT_TUNING_GRID
    distances: tuple[str, ...] | None = None
    search: str = "grid"
    budget: int = DEFAULT_TUNING_BUDGET
    n_eff_range: tuple[float, float] | None = None
    group_by: str = "auto"
    seed: int = 0
    coverage_tolerance: float | str = "auto"

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> TuningOptions:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("tuning must be a mapping when provided.")
        unknown = sorted(set(raw).difference(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unsupported keys under tuning: {unknown}.")
        folds = _positive_int(
            raw.get("folds", DEFAULT_TUNING_FOLDS), "folds", minimum=2
        )
        grid = tuple(
            sorted(set(_numbers(raw.get("grid", DEFAULT_TUNING_GRID), "grid")))
        )
        distances = None
        if raw.get("distances") is not None:
            values = raw["distances"]
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, Sequence) or not values:
                raise ValueError("tuning.distances must be a non-empty list.")
            normalized = [str(value).strip().lower() for value in values]
            unsupported = sorted(set(normalized).difference(SUPPORTED_DISTANCES))
            if unsupported:
                raise ValueError(
                    f"tuning.distances contains unsupported values {unsupported}; "
                    f"supported: {list(SUPPORTED_DISTANCES)}."
                )
            distances = tuple(dict.fromkeys(normalized))
        search = str(raw.get("search", "grid")).strip().lower()
        if search not in TUNING_SEARCHES:
            raise ValueError(
                f"tuning.search must be one of {list(TUNING_SEARCHES)}; got {search!r}."
            )
        budget = _positive_int(
            raw.get("budget", DEFAULT_TUNING_BUDGET), "budget", minimum=1
        )
        n_eff_range = None
        if raw.get("n_eff_range") is not None:
            bounds = _numbers(raw["n_eff_range"], "n_eff_range")
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(
                    "tuning.n_eff_range must be [low, high] with low <= high."
                )
            # Candidates are whole numbers of scenarios.
            if math.ceil(bounds[0]) > math.floor(bounds[1]):
                raise ValueError(
                    f"tuning.n_eff_range must contain a whole number; got {bounds}."
                )
            n_eff_range = (bounds[0], bounds[1])
        group_by = str(raw.get("group_by", "auto")).strip().lower()
        if group_by not in TUNING_GROUP_BY:
            raise ValueError(
                f"tuning.group_by must be one of {list(TUNING_GROUP_BY)}; got {group_by!r}."
            )
        seed = raw.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"tuning.seed must be an integer; got {seed!r}.")
        tolerance = raw.get("coverage_tolerance", "auto")
        if tolerance != "auto":
            try:
                tolerance = float(tolerance)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "tuning.coverage_tolerance must be 'auto' or a number."
                ) from exc
            if not 0.0 <= tolerance < 1.0:
                raise ValueError("tuning.coverage_tolerance must be in [0, 1).")
        return cls(
            folds=folds,
            grid=grid,
            distances=distances,
            search=search,
            budget=budget,
            n_eff_range=n_eff_range,
            group_by=group_by,
            seed=seed,
            coverage_tolerance=tolerance,
        )

    def smallest_n_eff_target(self) -> float:
        """Lowest target any candidate can have (checked against `n_eff_min`)."""

        if self.search == "random":
            return self.resolved_range()[0]
        return min(self.grid)

    def resolved_range(self) -> tuple[float, float]:
        """`n_eff_range`, or the span of `grid` when not given."""

        return self.n_eff_range or (min(self.grid), max(self.grid))


@dataclass
class BandwidthTuningResult:
    """Outcome of `tune_n_eff_target`."""

    selected_n_eff_target: float
    selected_distance: str
    reason: str
    options: TuningOptions
    grouped: bool
    candidates: pd.DataFrame
    details: pd.DataFrame = field(repr=False)

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-serializable record stored in the artifact."""

        return {
            "selected_n_eff_target": self.selected_n_eff_target,
            "selected_distance": self.selected_distance,
            "reason": self.reason,
            "search": self.options.search,
            "folds": self.options.folds,
            "grid": list(self.options.grid) if self.options.search == "grid" else None,
            "budget": self.options.budget if self.options.search == "random" else None,
            "n_eff_range": list(self.options.resolved_range())
            if self.options.search == "random"
            else None,
            "distances": sorted(set(self.candidates["distance"])),
            "grouped_folds": self.grouped,
            "seed": self.options.seed,
            "coverage_tolerance": self.options.coverage_tolerance,
            "candidates": self.candidates.to_dict(orient="records"),
        }

    @property
    def selected_fallback_share(self) -> float:
        """Share of held-out scenarios that fell back at the chosen setting.

        NaN when no candidate was chosen (uniform weights).
        """

        chosen = self.candidates[self.candidates["selected"]]
        return float(chosen["fallback_share"].iloc[0]) if len(chosen) else math.nan

    def fallback_note(self, name: str = "") -> str | None:
        """Why the chosen setting is nearly the element method, when it is."""

        share = self.selected_fallback_share
        if not share >= FALLBACK_WARNING_SHARE:
            return None
        return (
            f"kernel tuning{f' ({name})' if name else ''}: at the chosen "
            f"n_eff_target = {self.selected_n_eff_target:g}, {share:.0%} of the "
            "held-out scenarios fall back to the stratum quantiles, so the bounds "
            "are nearly those of the element method: the kernel does not localize "
            "usefully on this embedding (see fallback_share per candidate)."
        )


def candidate_settings(
    options: TuningOptions, default_distance: str
) -> list[tuple[float, str]]:
    """The (`n_eff_target`, `distance`) pairs to evaluate, sorted."""

    distances = options.distances or (default_distance,)
    if options.search == "grid":
        return sorted((target, d) for d in distances for target in options.grid)
    low, high = options.resolved_range()
    space = len(distances) * (int(high) - int(low) + 1)
    budget = min(options.budget, space)
    rng = np.random.default_rng(options.seed + 1)
    chosen: set[tuple[float, str]] = set()
    for _ in range(100 * budget):
        if len(chosen) == budget:
            break
        target = float(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
        target = min(max(target, math.ceil(low)), math.floor(high))
        chosen.add((float(target), str(distances[rng.integers(len(distances))])))
    return sorted(chosen)


def assign_folds(
    n_scenarios: int,
    folds: int,
    *,
    groups: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Fold index (0..folds-1) per scenario; scenarios sharing a group stay together."""

    rng = np.random.default_rng(seed)
    if groups is None:
        units = np.arange(n_scenarios)
        unit_of_row = units
    else:
        groups = np.asarray(groups)
        if groups.shape != (n_scenarios,):
            raise ValueError("groups must hold one value per scenario.")
        units, unit_of_row = np.unique(groups, return_inverse=True)
    if len(units) < folds:
        raise ValueError(
            f"Cannot split {len(units)} {'groups' if groups is not None else 'scenarios'} "
            f"into {folds} folds."
        )
    fold_of_unit = np.empty(len(units), dtype=np.int64)
    fold_of_unit[rng.permutation(len(units))] = np.arange(len(units)) % folds
    return fold_of_unit[unit_of_row]


def select_candidate(
    candidates: pd.DataFrame, fallback_n_eff_target: float, fallback_distance: str
) -> tuple[float, str, str]:
    """Apply the selection rule to a per-candidate table (see the module docstring).

    Returns (`n_eff_target`, `distance`, reason). Ties in width go to the larger
    target (more calibration scenarios per bound), then to the distance listed first
    alphabetically, so the result is deterministic.
    """

    eligible = candidates[candidates["eligible"]]
    if eligible.empty:
        return float(fallback_n_eff_target), fallback_distance, REASON_NO_CANDIDATE
    best = eligible.sort_values(
        ["relative_width", "n_eff_target", "distance"], ascending=[True, False, True]
    ).iloc[0]
    return float(best["n_eff_target"]), str(best["distance"]), REASON_SMALLEST_WIDTH


def tune_n_eff_target(
    predictions: ChannelSet,
    ground_truth: ChannelSet,
    *,
    alpha: float,
    channel_names: Sequence[str],
    strat: str | None = None,
    embedding_pooling: str = "mean",
    distance: str = "euclidean",
    n_eff_min: float = 5.0,
    options: TuningOptions | None = None,
    groups: np.ndarray | None = None,
    system_load_n_bins: int | None = None,
    system_load_bin_edges: np.ndarray | None = None,
    sparse_strata: str = DEFAULT_SPARSE_STRATA,
    n_jobs: int | None = None,
) -> BandwidthTuningResult:
    """Choose `n_eff_target` (and optionally `distance`) by K-fold cross-validation.

    `predictions` and `ground_truth` are the calibration ChannelSets, with
    scenario embeddings attached to `predictions`. `distance` is used unless
    `options.distances` lists the distances to compare. `groups` (one value per
    scenario, e.g. the datakit `load_scenario_idx`) keeps related scenarios in
    the same fold; without it, near-duplicate scenarios on both sides of a fold
    make localization look better than it is. `sparse_strata` is passed to every
    fold's calibrator, so the candidates are compared as they will be applied.

    The kernel runs in `n_jobs` threads (`None` or `0`: every CPU, a negative
    number: all but `-n_jobs - 1`); the result does not depend on it. To tune
    several calibrated families on the same scenarios, `tune_n_eff_targets`
    shares the kernel work between them.
    """

    result = _tune_n_eff_targets(
        {"": (predictions, ground_truth, channel_names)},
        alpha=alpha,
        strat=strat,
        embedding_pooling=embedding_pooling,
        distance=distance,
        n_eff_min=n_eff_min,
        options=options,
        groups=groups,
        system_load_n_bins=system_load_n_bins,
        system_load_bin_edges=system_load_bin_edges,
        sparse_strata=sparse_strata,
        n_jobs=n_jobs,
    )[""]
    _warn_fallback(result, "")
    return result


def tune_n_eff_targets(
    families: Mapping[str, tuple[ChannelSet, ChannelSet, Sequence[str]]],
    *,
    alpha: float,
    strat: str | None = None,
    embedding_pooling: str = "mean",
    distance: str = "euclidean",
    n_eff_min: float = 5.0,
    options: TuningOptions | None = None,
    groups: np.ndarray | None = None,
    system_load_n_bins: int | None = None,
    system_load_bin_edges: np.ndarray | None = None,
    sparse_strata: str = DEFAULT_SPARSE_STRATA,
    n_jobs: int | None = None,
) -> dict[str, BandwidthTuningResult]:
    """`tune_n_eff_target` for several families of variables at once.

    `families` maps a name (e.g. `"volt"`, `"line"`) to `(predictions,
    ground_truth, channel_names)`. Families on the same calibration scenarios
    with the same embeddings, like the voltage and line-loading families of
    one config, get the same folds; each fold's distances are then computed
    once, and each candidate's bandwidths once, for all of them. Every result
    equals that of `tune_n_eff_target` on its family alone. The other
    arguments are those of `tune_n_eff_target`.

    Warns:
        KernelFallbackWarning: For a family whose chosen setting falls back to
            the stratum quantiles in at least half of the held-out scenarios.
    """

    results = _tune_n_eff_targets(
        families,
        alpha=alpha,
        strat=strat,
        embedding_pooling=embedding_pooling,
        distance=distance,
        n_eff_min=n_eff_min,
        options=options,
        groups=groups,
        system_load_n_bins=system_load_n_bins,
        system_load_bin_edges=system_load_bin_edges,
        sparse_strata=sparse_strata,
        n_jobs=n_jobs,
    )
    for name, result in results.items():
        _warn_fallback(result, name)
    return results


def _warn_fallback(result: BandwidthTuningResult, name: str) -> None:
    note = result.fallback_note(name)
    if note is not None:
        # Two frames up: the caller of tune_n_eff_target(s).
        warnings.warn(note[0].upper() + note[1:], KernelFallbackWarning, stacklevel=3)


def _tune_n_eff_targets(
    families: Mapping[str, tuple[ChannelSet, ChannelSet, Sequence[str]]],
    *,
    alpha: float,
    strat: str | None = None,
    embedding_pooling: str = "mean",
    distance: str = "euclidean",
    n_eff_min: float = 5.0,
    options: TuningOptions | None = None,
    groups: np.ndarray | None = None,
    system_load_n_bins: int | None = None,
    system_load_bin_edges: np.ndarray | None = None,
    sparse_strata: str = DEFAULT_SPARSE_STRATA,
    n_jobs: int | None = None,
) -> dict[str, BandwidthTuningResult]:
    """`tune_n_eff_targets` without the warning (the CLI prints the note instead)."""

    options = options or TuningOptions()
    for predictions, _, _ in families.values():
        if predictions.scenario_embeddings is None:
            raise ValueError(
                "tune_n_eff_target needs scenario embeddings on the predictions."
            )
    shared: dict[str, Any] = {
        "alpha": alpha,
        "strat": strat,
        "embedding_pooling": embedding_pooling,
        "distance": distance,
        "n_eff_min": n_eff_min,
        "options": options,
        "groups": groups,
        "system_load_n_bins": system_load_n_bins,
        "system_load_bin_edges": system_load_bin_edges,
        "sparse_strata": sparse_strata,
        "n_jobs": n_jobs,
    }
    first = next(iter(families.values()))[0]
    if any(
        not _same_scenarios(first, predictions, strat)
        for predictions, _, _ in families.values()
    ):
        return {
            name: _tune_n_eff_targets({name: family}, **shared)[name]
            for name, family in families.items()
        }

    settings = candidate_settings(options, distance)
    n = len(first.scenario_ids)
    fold = assign_folds(n, options.folds, groups=groups, seed=options.seed)
    strata = (
        np.zeros(n, dtype=np.int64)
        if strat is None
        else np.asarray(first.scenario_columns[strat]).astype(np.int64)
    )
    stratum_values = np.unique(strata)
    threads = resolve_threads(n_jobs)

    # Every fold: one fit per family and distance (the fitted state does not
    # depend on the target), then all targets of that distance in one pass.
    results: dict[tuple[str, int, tuple[float, str]], _FoldResult] = {}
    for f in range(options.folds):
        train = np.flatnonzero(fold != f)
        held = np.flatnonzero(fold == f)
        for setting_distance in dict.fromkeys(d for _, d in settings):
            targets = [target for target, d in settings if d == setting_distance]
            calibrators = []
            held_sets = []
            for predictions, ground_truth, channel_names in families.values():
                calibrator = KernelIntervalCalibrator.fit(
                    _rows(predictions, train),
                    _rows(ground_truth, train),
                    alpha=alpha,
                    channel_names=tuple(channel_names),
                    strat=strat,
                    embeddings=np.asarray(predictions.scenario_embeddings)[train],
                    embedding_pooling=embedding_pooling,
                    n_eff_min=n_eff_min,
                    bandwidth="n_eff",
                    n_eff_target=targets[0],
                    distance=setting_distance,
                    system_load_n_bins=system_load_n_bins,
                    system_load_bin_edges=system_load_bin_edges,
                    sparse_strata=sparse_strata,
                )
                calibrator.n_jobs = threads
                calibrators.append(calibrator)
                held_sets.append(_rows(predictions, held))
            by_family = _predict_targets(calibrators, held_sets, targets)
            for (
                name,
                (_, ground_truth, channel_names),
            ), held_predictions, by_target in zip(
                families.items(), held_sets, by_family, strict=True
            ):
                held_truth = _rows(ground_truth, held)
                for target in targets:
                    intervals, diagnostics = by_target[float(target)]
                    results[(name, f, (target, setting_distance))] = _fold_result(
                        intervals,
                        diagnostics,
                        held_predictions,
                        held_truth,
                        strata[held],
                        stratum_values,
                        tuple(channel_names),
                    )
    return {
        name: _summarize(
            [
                results[(name, f, setting)]
                for f in range(options.folds)
                for setting in settings
            ],
            [setting for _ in range(options.folds) for setting in settings],
            settings,
            options,
            alpha=alpha,
            strat=strat,
            distance=distance,
            n_eff_min=n_eff_min,
            channel_names=tuple(channel_names),
            strata=strata,
            stratum_values=stratum_values,
            grouped=groups is not None,
        )
        for name, (_, _, channel_names) in families.items()
    }


def _same_scenarios(first: ChannelSet, other: ChannelSet, strat: str | None) -> bool:
    """Whether two families can share folds and kernel work."""

    if other is first:
        return True
    if not np.array_equal(first.scenario_ids, other.scenario_ids):
        return False
    if not np.array_equal(
        np.asarray(first.scenario_embeddings), np.asarray(other.scenario_embeddings)
    ):
        return False
    return strat is None or np.array_equal(
        np.asarray(first.scenario_columns[strat]),
        np.asarray(other.scenario_columns[strat]),
    )


def _summarize(
    fold_results: Sequence[_FoldResult],
    fold_settings: Sequence[tuple[float, str]],
    settings: Sequence[tuple[float, str]],
    options: TuningOptions,
    *,
    alpha: float,
    strat: str | None,
    distance: str,
    n_eff_min: float,
    channel_names: tuple[str, ...],
    strata: np.ndarray,
    stratum_values: np.ndarray,
    grouped: bool,
) -> BandwidthTuningResult:
    """Coverage and width per candidate, the selection, and the result tables."""

    n = len(strata)
    # Accumulators per (candidate, channel, scope); scope None is the overall check.
    Setting = tuple[float, str]
    covered: dict[tuple[Setting, str, int | None], int] = {}
    counted: dict[tuple[Setting, str, int | None], int] = {}
    # Widths per (candidate, channel, scope), like coverage.
    width_sum: dict[tuple[Setting, str, int | None], float] = {}
    width_n: dict[tuple[Setting, str, int | None], int] = {}
    infinite: dict[tuple[Setting, str, int | None], int] = {}
    fallbacks: dict[Setting, int] = {}
    # Added up fold by fold, in candidate order, so that sums are reproducible.
    for setting, result in zip(fold_settings, fold_results, strict=True):
        fallbacks[setting] = fallbacks.get(setting, 0) + result.fallbacks
        for (channel, scope), value in result.covered.items():
            key = (setting, channel, scope)
            covered[key] = covered.get(key, 0) + value
            counted[key] = counted.get(key, 0) + result.counted[(channel, scope)]
        for (channel, scope), width_total in result.width_sum.items():
            wkey = (setting, channel, scope)
            width_sum[wkey] = width_sum.get(wkey, 0.0) + width_total
            width_n[wkey] = width_n.get(wkey, 0) + result.width_n[(channel, scope)]
            infinite[wkey] = infinite.get(wkey, 0) + result.infinite[(channel, scope)]

    def mean_width(setting: Setting, channel: str, scope: int | None = None) -> float:
        key = (setting, channel, scope)
        return width_sum.get(key, 0.0) / max(width_n.get(key, 0), 1)

    scenarios_in: dict[int | None, int] = {None: n}
    scenarios_in.update({int(v): int((strata == v).sum()) for v in stratum_values})
    # One common reference for every candidate: the largest target (closest to
    # uniform weights), preferring the config's own distance on a tie.
    reference = max(settings, key=lambda s: (s[0], s[1] == distance))
    detail_rows, candidate_rows = [], []
    for setting in settings:
        target, setting_distance = setting
        margins, relative = [], []
        for channel in channel_names:
            ref_width = mean_width(reference, channel)
            relative.append(
                mean_width(setting, channel) / ref_width if ref_width > 0 else 1.0
            )
            for scope in (None, *stratum_values.tolist()):
                key = (setting, channel, scope)
                if counted[key] == 0:
                    continue
                coverage = covered[key] / counted[key]
                threshold = (1.0 - alpha) - _tolerance(
                    options.coverage_tolerance, alpha, scenarios_in[scope]
                )
                margins.append(coverage - threshold)
                detail_rows.append(
                    {
                        "n_eff_target": target,
                        "distance": setting_distance,
                        "channel": channel,
                        "scope": "overall" if scope is None else f"{strat}={scope}",
                        "n_scenarios": scenarios_in[scope],
                        "coverage": coverage,
                        "threshold": threshold,
                        "meets_target": coverage >= threshold,
                        "mean_width": mean_width(setting, channel, scope),
                        "infinite_width_share": infinite.get(
                            (setting, channel, scope), 0
                        )
                        / max(
                            width_n.get((setting, channel, scope), 0)
                            + infinite.get((setting, channel, scope), 0),
                            1,
                        ),
                    }
                )
        overall = [
            covered[(setting, c, None)] / counted[(setting, c, None)]
            for c in channel_names
            if counted[(setting, c, None)]
        ]
        candidate_rows.append(
            {
                "n_eff_target": target,
                "distance": setting_distance,
                "coverage": min(overall) if overall else float("nan"),
                "worst_margin": min(margins) if margins else float("nan"),
                "relative_width": float(np.mean(relative)),
                "fallback_share": fallbacks[setting] / n,
                "eligible": bool(margins) and min(margins) >= 0.0,
            }
        )

    candidates = pd.DataFrame(candidate_rows)
    largest_stratum = float(max(scenarios_in[int(v)] for v in stratum_values))
    selected, selected_distance, reason = select_candidate(
        candidates,
        fallback_n_eff_target=max(largest_stratum, n_eff_min),
        fallback_distance=distance,
    )
    candidates["selected"] = (
        (candidates["n_eff_target"] == selected)
        & (candidates["distance"] == selected_distance)
        & (reason == REASON_SMALLEST_WIDTH)
    )
    return BandwidthTuningResult(
        selected_n_eff_target=selected,
        selected_distance=selected_distance,
        reason=reason,
        options=options,
        grouped=grouped,
        candidates=candidates,
        details=pd.DataFrame(detail_rows),
    )


def _positive_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"tuning.{name} must be an integer >= {minimum}; got {value!r}."
        )
    return value


def _numbers(values: Any, name: str) -> list[float]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise ValueError(f"tuning.{name} must be a non-empty list of numbers.")
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tuning.{name} values must be numbers.") from exc
    if min(numbers) < 1.0:
        raise ValueError(f"tuning.{name} values must be >= 1.")
    return numbers


def _tolerance(setting: float | str, alpha: float, n_scenarios: int) -> float:
    if setting == "auto":
        return 2.0 * math.sqrt(alpha * (1.0 - alpha) / max(n_scenarios, 1))
    return float(setting)


@dataclass
class _FoldResult:
    """Coverage and width counts of one candidate on one held-out fold."""

    covered: dict[tuple[str, int | None], int]
    counted: dict[tuple[str, int | None], int]
    width_sum: dict[tuple[str, int | None], float]
    width_n: dict[tuple[str, int | None], int]
    infinite: dict[tuple[str, int | None], int]
    fallbacks: int


def _fold_result(
    intervals: dict[str, tuple[np.ndarray, np.ndarray]],
    diagnostics: dict[str, np.ndarray],
    held_predictions: ChannelSet,
    held_truth: ChannelSet,
    held_strata: np.ndarray,
    stratum_values: np.ndarray,
    channel_names: tuple[str, ...],
) -> _FoldResult:
    """Coverage and width counts of one candidate on one held-out fold."""

    result = _FoldResult(
        covered={},
        counted={},
        width_sum={},
        width_n={},
        infinite={},
        fallbacks=int(np.asarray(diagnostics["kernel_fallback"]).sum()),
    )
    for channel in channel_names:
        lower, upper = intervals[channel]
        truth = held_truth.channels[channel]
        valid = _valid(held_predictions, held_truth, channel)
        hit = (lower <= truth) & (truth <= upper) & valid
        width = upper - lower
        for scope in (None, *stratum_values.tolist()):
            rows = slice(None) if scope is None else held_strata == scope
            result.covered[(channel, scope)] = int(hit[rows].sum())
            result.counted[(channel, scope)] = int(valid[rows].sum())
            scope_width = width[rows][valid[rows]]
            finite = np.isfinite(scope_width)
            result.width_sum[(channel, scope)] = float(scope_width[finite].sum())
            result.width_n[(channel, scope)] = int(finite.sum())
            result.infinite[(channel, scope)] = int((~finite).sum())
    return result


def _valid(
    predictions: ChannelSet, ground_truth: ChannelSet, channel: str
) -> np.ndarray:
    shape = predictions.channels[channel].shape
    valid = np.ones(shape, dtype=bool)
    for channel_set in (predictions, ground_truth):
        mask = channel_set.valid_masks.get(channel)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
    return valid


def _rows(channel_set: ChannelSet, rows: np.ndarray) -> ChannelSet:
    return ChannelSet(
        scenario_ids=channel_set.scenario_ids[rows],
        entity_ids=channel_set.entity_ids,
        entity_name=channel_set.entity_name,
        channels={name: values[rows] for name, values in channel_set.channels.items()},
        valid_masks={
            name: values[rows] for name, values in channel_set.valid_masks.items()
        },
        entity_columns=dict(channel_set.entity_columns),
        scenario_columns={
            name: values[rows] for name, values in channel_set.scenario_columns.items()
        },
        scenario_embeddings=None
        if channel_set.scenario_embeddings is None
        else channel_set.scenario_embeddings[rows],
    )
