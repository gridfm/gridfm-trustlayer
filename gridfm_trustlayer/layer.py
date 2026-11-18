"""Several calibrators fitted, applied and saved as one: `TrustLayer`.

The calibrators work on one `ChannelSet` each, i.e. one kind of element: bus
voltages and line loadings need two. `TrustLayer` fits one per set with the
same settings (and, for the kernel method, chooses `n_eff_target` for all of
them at once), sends new data to the calibrator of its variables, and saves
them in one `.trustlayer` file together with the definitions of the strata
(`OutageStrata`, `LoadBins`), so that new scenarios can be stratified the same
way after loading.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.conformal.calibrator import (
    DEFAULT_SPARSE_STRATA,
    ChannelSet,
    NaiveIntervalCalibrator,
    _integer_ids,
)
from gridfm_trustlayer.conformal.inputs import check_inputs
from gridfm_trustlayer.conformal.kernel import (
    DEFAULT_DISTANCE,
    DEFAULT_N_EFF_MIN,
    KernelIntervalCalibrator,
    _predict_jointly,
    _warn_if_fallback,
)
from gridfm_trustlayer.conformal.tuning import (
    BandwidthTuningResult,
    TuningOptions,
    _tune_n_eff_targets,
    _warn_fallback,
)
from gridfm_trustlayer.io.artifact import _read_artifact, _write_artifact
from gridfm_trustlayer.strata import LoadBins, OutageStrata, strata_from_dict

Calibrator = NaiveIntervalCalibrator | KernelIntervalCalibrator
Strata = OutageStrata | LoadBins
Intervals = dict[str, tuple[np.ndarray, np.ndarray]]

METHODS = ("element", "kernel")


@dataclass(eq=False)
class TrustLayer:
    """Calibrators of several sets of variables, fitted, applied and saved as one.

    Create it with `fit`, or `load` a saved one (also a `.trustlayer` file of
    the command-line tool). `predict_intervals` takes the new data as one
    `ChannelSet` or a list (e.g. `[buses, lines]`) and sends each to the
    calibrator of its variables.

    Attributes:
        calibrators: One calibrator per set of variables, keyed by its
            variables joined with `+` (e.g. `"Vm"`, `"line_loading"`); `"volt"`
            and `"line"` in a file of the command-line tool.
        strata: The definitions behind the strata columns, by stratum name
            (e.g. `{"n_outages": OutageStrata}`), saved with the layer: compute
            the strata of new scenarios with them.
        tuning: Per calibrator, the cross-validated choice of `n_eff_target`
            when fitted with `n_eff_target="auto"` (a summary is also saved in
            each kernel calibrator's `tuning`).
    """

    calibrators: dict[str, Calibrator]
    strata: dict[str, Strata] = field(default_factory=dict)
    tuning: dict[str, BandwidthTuningResult] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        predictions: ChannelSet | Sequence[ChannelSet],
        true_values: ChannelSet | Sequence[ChannelSet],
        *,
        alpha: float,
        method: str = "element",
        strat: str | None = None,
        strata: Mapping[str, Strata] | None = None,
        n_eff_target: float | str | None = None,
        tuning_options: TuningOptions | None = None,
        groups: np.ndarray | pd.Series | None = None,
        distance: str = DEFAULT_DISTANCE,
        n_eff_min: float = DEFAULT_N_EFF_MIN,
        sparse_strata: str = DEFAULT_SPARSE_STRATA,
        n_jobs: int | None = None,
    ) -> TrustLayer:
        """Fit one calibrator per set of predictions, with the same settings.

        The data are checked with `check_inputs` first; its errors are raised
        (its warnings are not: run `check_inputs` yourself to read them).

        Args:
            predictions: The calibration predictions, one `ChannelSet` or a
                list such as `[buses, lines]`. Every variable of a set is
                calibrated.
            true_values: The true values, a set (or list of sets, in the same
                order) with the same scenarios and elements.
            alpha: Two-sided miscoverage level; `0.1` gives 90% intervals.
            method: `"element"` (`NaiveIntervalCalibrator`) or `"kernel"`
                (`KernelIntervalCalibrator`; the predictions need
                `scenario_embeddings`).
            strat: A stratum column of the sets to fit per stratum, e.g.
                `"n_outages"`.
            strata: The definitions behind the strata columns, saved with the
                layer, e.g. `{"n_outages": outage_strata}`. For
                `strat="system_load_bin"`, a `LoadBins` under that name is
                required (its edges are stored in the calibrators, as the
                command-line tool does).
            n_eff_target: Kernel only: the effective number of calibration
                scenarios per bound (600 when not given), or `"auto"` to choose
                it by cross-validation on the calibration set, for all sets at
                once (`tune_n_eff_targets`, which also chooses among
                `tuning_options.distances` when given).
            tuning_options: Options of the cross-validation (`"auto"` only).
            groups: For `"auto"`: a group per calibration scenario (e.g. its
                load scenario) to keep related scenarios in the same fold, as
                a Series by scenario id or an array in scenario-id order.
            distance: Kernel only: `"euclidean"` or `"cosine"`.
            n_eff_min: Kernel only: below this effective sample size a scenario
                falls back to the stratum quantiles.
            sparse_strata: `"pooled"` (default) or `"infinite"`: the bounds of
                an element in a stratum with too few calibration samples.
            n_jobs: Kernel only: threads of the tuning (every CPU by default).

        Returns:
            The fitted layer.

        Raises:
            ValueError: If `check_inputs` finds errors, the sets do not pair up,
                a variable is in two sets, or an option is invalid.

        Warns:
            KernelFallbackWarning: When the chosen `n_eff_target` makes most
                held-out scenarios fall back to the stratum quantiles.
        """

        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}; got {method!r}.")
        if strat == "pred_bins":
            raise ValueError(
                "strat='pred_bins' is not supported here; use a stratum column "
                "of the sets (e.g. bins of your own) or NaiveIntervalCalibrator."
            )
        kernel = method == "kernel"
        if not kernel and (n_eff_target is not None or tuning_options is not None):
            raise ValueError("n_eff_target and tuning_options need method='kernel'.")
        auto = n_eff_target == "auto"
        if isinstance(n_eff_target, str) and not auto:
            raise ValueError(
                f"n_eff_target must be a number or 'auto'; got {n_eff_target!r}."
            )
        definitions = dict(strata or {})
        for name, definition in definitions.items():
            if not isinstance(definition, (OutageStrata, LoadBins)):
                raise ValueError(
                    f"strata[{name!r}] must be an OutageStrata or LoadBins; got "
                    f"{type(definition).__name__}."
                )
        edges = None
        if strat == "system_load_bin":
            bins = definitions.get(strat)
            if not isinstance(bins, LoadBins):
                raise ValueError(
                    "strat='system_load_bin' needs its LoadBins: pass "
                    "strata={'system_load_bin': LoadBins.fit(...)}."
                )
            edges = bins.edges
        pairs = _pairs(predictions, true_values)
        for prediction_set, truth in pairs.values():
            check_inputs(
                prediction_set,
                truth,
                strat=strat,
                alpha=alpha,
                kernel=kernel,
            ).raise_for_errors()

        tuning: dict[str, BandwidthTuningResult] = {}
        if auto:
            first = next(iter(pairs.values()))[0]
            tuning = _tune_n_eff_targets(
                {
                    key: (prediction_set, truth, tuple(prediction_set.channels))
                    for key, (prediction_set, truth) in pairs.items()
                },
                alpha=alpha,
                strat=strat,
                distance=distance,
                n_eff_min=n_eff_min,
                options=tuning_options,
                groups=None if groups is None else _groups(groups, first.scenario_ids),
                system_load_bin_edges=edges,
                sparse_strata=sparse_strata,
                n_jobs=n_jobs,
            )
            for key, result in tuning.items():
                _warn_fallback(result, key)

        calibrators: dict[str, Calibrator] = {}
        for key, (prediction_set, truth) in pairs.items():
            channel_names = tuple(prediction_set.channels)
            if not kernel:
                calibrators[key] = NaiveIntervalCalibrator.fit(
                    prediction_set,
                    truth,
                    alpha=alpha,
                    channel_names=channel_names,
                    strat=strat,
                    system_load_bin_edges=edges,
                    sparse_strata=sparse_strata,
                )
                continue
            assert prediction_set.scenario_embeddings is not None
            target = None if auto else n_eff_target
            family_distance = distance
            if key in tuning:
                target = tuning[key].selected_n_eff_target
                family_distance = tuning[key].selected_distance
            calibrator = KernelIntervalCalibrator.fit(
                prediction_set,
                truth,
                alpha=alpha,
                channel_names=channel_names,
                strat=strat,
                embeddings=prediction_set.scenario_embeddings,
                n_eff_min=n_eff_min,
                n_eff_target=None if target is None else float(target),
                distance=family_distance,
                system_load_bin_edges=edges,
                sparse_strata=sparse_strata,
            )
            if key in tuning:
                calibrator.tuning = tuning[key].summary()
            calibrators[key] = calibrator
        return cls(calibrators=calibrators, strata=definitions, tuning=tuning)

    @property
    def variables(self) -> tuple[str, ...]:
        """Every calibrated variable."""

        return tuple(
            name
            for calibrator in self.calibrators.values()
            for name in calibrator.channel_names
        )

    def predict_intervals(
        self, predictions: ChannelSet | Sequence[ChannelSet]
    ) -> Intervals:
        """Bounds for new predictions.

        Args:
            predictions: One `ChannelSet` or a list (e.g. `[buses, lines]`),
                with the same elements, strata columns and, for the kernel
                method, scenario vectors as at fit. Each set goes to the
                calibrator of its variables; calibrators whose variables are
                not passed are skipped.

        Returns:
            `{variable: (lower, upper)}` for the variables passed.

        Raises:
            ValueError: If a set has no calibrator, two sets hold the variables
                of one calibrator, or `check_inputs` finds errors.

        Warns:
            KernelFallbackWarning: For a kernel calibrator whose bounds fall
                back to the stratum quantiles in most scenarios.
        """

        routed = self._route(predictions)
        for key, prediction_set in routed.items():
            check_inputs(
                prediction_set, calibrator=self.calibrators[key]
            ).raise_for_errors()
        # Kernel calibrators share their kernel work (same scenarios).
        kernel = {
            key: calibrator
            for key, calibrator in self.calibrators.items()
            if key in routed and isinstance(calibrator, KernelIntervalCalibrator)
        }
        results: dict[str, Intervals] = {}
        if kernel:
            joint = _predict_jointly(
                list(kernel.values()), [routed[key] for key in kernel]
            )
            results.update(zip(kernel, joint, strict=True))
            for key, calibrator in kernel.items():
                _warn_if_fallback(calibrator, routed[key], stacklevel=2)
        for key in routed:
            if key not in results:
                results[key] = self.calibrators[key].predict_intervals(routed[key])
        return {
            name: bounds
            for key in self.calibrators
            if key in results
            for name, bounds in results[key].items()
        }

    @property
    def diagnostics(self) -> dict[str, dict[str, np.ndarray]]:
        """Per kernel calibrator, the diagnostics of the last `predict_intervals`.

        `{key: {"n_eff": ..., "kernel_bandwidth_sq": ..., "kernel_fallback": ...}}`,
        one value per scenario (see `KernelIntervalCalibrator.diagnostics`).
        """

        return {
            key: calibrator.diagnostics
            for key, calibrator in self.calibrators.items()
            if isinstance(calibrator, KernelIntervalCalibrator)
            and calibrator.diagnostics
        }

    def save(self, path: str | Path) -> None:
        """Write the calibrators and the strata definitions to one file.

        The format of `save_interval_artifact`: the command-line tool and
        `load_interval_artifacts` read the calibrators of it.
        """

        extras: dict[str, Any] = {}
        if self.strata:
            extras["strata"] = {
                name: definition.to_dict() for name, definition in self.strata.items()
            }
        _write_artifact(self.calibrators, path, extras)

    @classmethod
    def load(cls, path: str | Path) -> TrustLayer:
        """Read a layer saved by `save`, or the artifact of a command-line run."""

        calibrators, extras = _read_artifact(path)
        raw_strata = extras.get("strata") or {}
        if not isinstance(raw_strata, dict):
            raise ValueError(f"{path}: 'strata' must be a mapping.")
        return cls(
            calibrators=dict(calibrators),
            strata={
                str(name): strata_from_dict(dict(data))
                for name, data in raw_strata.items()
            },
        )

    def _route(
        self, predictions: ChannelSet | Sequence[ChannelSet]
    ) -> dict[str, ChannelSet]:
        """The set of each calibrator whose variables are passed."""

        sets = (
            [predictions] if isinstance(predictions, ChannelSet) else list(predictions)
        )
        routed: dict[str, ChannelSet] = {}
        for prediction_set in sets:
            if not isinstance(prediction_set, ChannelSet):
                raise ValueError(
                    "Pass ChannelSets (e.g. from ChannelSet.from_frame); got "
                    f"{type(prediction_set).__name__}."
                )
            keys = [
                key
                for key, calibrator in self.calibrators.items()
                if set(calibrator.channel_names) <= set(prediction_set.channels)
            ]
            if not keys:
                raise ValueError(
                    f"No calibrator for the variables {sorted(prediction_set.channels)}"
                    f"; the layer calibrates {list(self.variables)}, one set per "
                    f"calibrator: {[list(c.channel_names) for c in self.calibrators.values()]}."
                )
            for key in keys:
                if key in routed:
                    raise ValueError(
                        f"Two sets hold the variables of the calibrator {key!r}."
                    )
                routed[key] = prediction_set
        return routed


def _pairs(
    predictions: ChannelSet | Sequence[ChannelSet],
    true_values: ChannelSet | Sequence[ChannelSet],
) -> dict[str, tuple[ChannelSet, ChannelSet]]:
    """Predictions and true values in pairs, keyed by their variables."""

    sets = [predictions] if isinstance(predictions, ChannelSet) else list(predictions)
    truths = [true_values] if isinstance(true_values, ChannelSet) else list(true_values)
    if not sets or len(sets) != len(truths):
        raise ValueError(
            f"Got {len(sets)} prediction sets and {len(truths)} true-value sets; "
            "pass them in pairs, in the same order."
        )
    pairs: dict[str, tuple[ChannelSet, ChannelSet]] = {}
    seen: set[str] = set()
    for prediction_set, truth in zip(sets, truths, strict=True):
        if not isinstance(prediction_set, ChannelSet) or not isinstance(
            truth, ChannelSet
        ):
            raise ValueError("Pass ChannelSets (e.g. from ChannelSet.from_frame).")
        names = tuple(prediction_set.channels)
        repeated = sorted(seen & set(names))
        if repeated:
            raise ValueError(f"The variables {repeated} are in more than one set.")
        seen.update(names)
        pairs["+".join(names)] = (prediction_set, truth)
    return pairs


def _groups(groups: np.ndarray | pd.Series, scenario_ids: np.ndarray) -> np.ndarray:
    """One group per calibration scenario, in the order of `scenario_ids`."""

    if isinstance(groups, pd.Series):
        ids = _integer_ids(groups.index, "the index of groups", "Scenario")
        position = pd.Index(ids).get_indexer(scenario_ids)
        if (position < 0).any():
            raise ValueError(
                f"groups has no value for {(position < 0).sum():,} calibration "
                "scenario(s)."
            )
        return groups.to_numpy()[position]
    array = np.asarray(groups)
    if array.shape != (len(scenario_ids),):
        raise ValueError(
            f"groups has shape {array.shape}; expected one value per calibration "
            f"scenario ({len(scenario_ids)}), or a Series by scenario id."
        )
    return array
