"""Naive residual-quantile interval calibration for direct bus outputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from gridfm_trustlayer.io.ground_truth import GroundTruthSet
from gridfm_trustlayer.io.predictions import PredictionSet

FLOAT32_DTYPE = np.dtype(np.float32)

# What an element gets in a stratum with too few calibration samples for a
# finite bound (fewer than ceil(2 / alpha) - 1 valid residuals, or a stratum
# absent from the calibration data): its quantile over all calibration
# scenarios (default), which is finite but carries no guarantee within that
# stratum, or infinite bounds, which keep the per-stratum guarantee.
SPARSE_STRATA_CHOICES = ("pooled", "infinite")
DEFAULT_SPARSE_STRATA = "pooled"


@dataclass
class ChannelSet:
    """Variables of one dataset as aligned (scenario x entity) arrays.

    The input of both calibrators: predictions and true values are two
    `ChannelSet`s with the same `scenario_ids` and `entity_ids`, in the same order.

    Attributes:
        scenario_ids: Scenario identifiers, shape `(n_scenarios,)`.
        entity_ids: Bus or branch identifiers, shape `(n_entities,)`.
        entity_name: `"bus"` or `"branch"`.
        channels: One `(n_scenarios, n_entities)` array per variable, e.g.
            `{"Vm": ..., "Va": ...}`.
        valid_masks: Optional boolean array per variable; `False` entries are
            ignored when fitting and get no interval.
        entity_columns: Optional per-entity metadata, e.g. `rate_a`.
        scenario_columns: Per-scenario values used for stratification, e.g.
            `{"n_outages": ...}`. Required for the chosen `strat`, both when
            fitting and on the predictions passed to `predict_intervals`.
        scenario_embeddings: `(n_scenarios, d)` scenario representations, for
            `KernelIntervalCalibrator` only.
    """

    scenario_ids: np.ndarray
    entity_ids: np.ndarray
    entity_name: str
    channels: dict[str, np.ndarray]
    valid_masks: dict[str, np.ndarray] = field(default_factory=dict)
    entity_columns: dict[str, np.ndarray] = field(default_factory=dict)
    scenario_columns: dict[str, np.ndarray] = field(default_factory=dict)
    # Scenario-level representations z(x) used by kernel-weighted calibration.
    # None for every cp_type other than 'kernel'.
    scenario_embeddings: np.ndarray | None = None

    @property
    def bus_ids(self) -> np.ndarray:
        """Compatibility alias for older bus-only call sites."""

        return self.entity_ids


# (n + 1) * q is computed in floating point; absorb rounding so that an exact
# integer rank (e.g. n = 9, q = 0.9) is not pushed to the next order statistic.
_RANK_TOLERANCE = 1e-9


@dataclass
class NaiveIntervalCalibrator:
    """Stratified split-conformal intervals (`cp_type: element`).

    Each variable is calibrated independently per entity, and per stratum when
    `strat` is set. Each tail uses the split-conformal order statistic at level
    `alpha / 2`, so the interval contains the true value with probability at
    least `1 - alpha` (within each stratum) when calibration and test scenarios
    are exchangeable, for any number of calibration samples. An element with
    too few samples in a stratum for that guarantee gets its quantile over all
    strata there (`sparse_strata="pooled"`, the default), or infinite bounds
    (`sparse_strata="infinite"`).

    Create it with `fit`; the class is named "naive" for historical reasons.
    """

    alpha: float
    entity_ids: np.ndarray
    entity_name: str
    channel_names: tuple[str, ...]
    lower_residual_quantiles: dict[str, np.ndarray]
    upper_residual_quantiles: dict[str, np.ndarray]
    strat: str | None = None
    stratification_values: np.ndarray | None = None
    stratified_lower_residual_quantiles: dict[str, np.ndarray] = field(
        default_factory=dict
    )
    stratified_upper_residual_quantiles: dict[str, np.ndarray] = field(
        default_factory=dict
    )
    system_load_n_bins: int | None = None
    system_load_bin_edges: np.ndarray | None = None
    pred_n_bins: int | None = None
    pred_bin_edges: dict[str, np.ndarray] = field(default_factory=dict)
    # SHA-256 of the calibration predictions, so that applying the artifact to the
    # very same data (in-sample evaluation) can be detected and refused.
    calibration_fingerprint: str | None = None
    sparse_strata: str = DEFAULT_SPARSE_STRATA
    _pooled: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def fit(
        cls,
        predictions: PredictionSet | ChannelSet,
        ground_truth: GroundTruthSet | ChannelSet,
        *,
        alpha: float,
        channel_names: Iterable[str] = ("Vm", "Va"),
        strat: str | None = None,
        system_load_n_bins: int | None = None,
        system_load_bin_edges: np.ndarray | None = None,
        pred_n_bins: int | None = None,
        pred_bin_edges: dict[str, np.ndarray] | None = None,
        sparse_strata: str = DEFAULT_SPARSE_STRATA,
    ) -> NaiveIntervalCalibrator:
        """Fit per-entity (and per-stratum) residual quantiles on calibration data.

        Args:
            predictions: Calibration predictions.
            ground_truth: True values for the same scenarios and entities.
            alpha: Two-sided miscoverage level; `0.1` gives 90% intervals.
            channel_names: Variables to calibrate.
            strat: Optional stratification: `"n_outages"`, `"outage_type"`,
                `"system_load_bin"` or `"pred_bins"`. Scenario strata are read
                from `ground_truth.scenario_columns` (or the `GroundTruthSet`).
            system_load_n_bins: Number of load bins, for `"system_load_bin"`.
            system_load_bin_edges: Stored load-bin edges, for `"system_load_bin"`.
            pred_n_bins: Number of prediction bins, for `"pred_bins"`.
            pred_bin_edges: Bin edges per variable, for `"pred_bins"`.
            sparse_strata: `"pooled"` (default) or `"infinite"`: the bounds of
                an element in a stratum with too few calibration samples.

        Returns:
            The fitted calibrator.

        Raises:
            ValueError: If the scenarios or entities of the two inputs are not
                aligned, `alpha` is outside (0, 1), or a stratification input
                is missing.
        """

        _validate_alpha(alpha)
        validate_sparse_strata(sparse_strata)
        _validate_entity_alignment(
            _get_entity_ids(predictions), _get_entity_ids(ground_truth)
        )
        if not np.array_equal(predictions.scenario_ids, ground_truth.scenario_ids):
            raise ValueError(
                "Prediction and ground-truth scenario_ids must be aligned before fitting.",
            )

        selected_channels = tuple(channel_names)
        lower_residual_quantiles: dict[str, np.ndarray] = {}
        upper_residual_quantiles: dict[str, np.ndarray] = {}
        stratified_lower_residual_quantiles: dict[str, np.ndarray] = {}
        stratified_upper_residual_quantiles: dict[str, np.ndarray] = {}
        stratification_values: np.ndarray | None = None
        shared_scenario_strata: np.ndarray | None = None
        if strat == "pred_bins":
            if pred_n_bins is None:
                raise ValueError(
                    "pred_bins stratification requires a configured pred_n_bins value."
                )
            if pred_bin_edges is None:
                raise ValueError(
                    "pred_bins stratification requires stored prediction-bin edges for each channel."
                )
        elif strat is not None:
            shared_scenario_strata = _get_scenario_column(ground_truth, strat)
            stratification_values = np.sort(
                np.unique(shared_scenario_strata.astype(np.int64, copy=False))
            )
            if strat == "system_load_bin" and system_load_bin_edges is None:
                raise ValueError(
                    "system_load_bin stratification requires stored percentile bin edges.",
                )
        for channel_name in selected_channels:
            prediction_values = _get_prediction_channel(predictions, channel_name)
            target_values = _get_ground_truth_channel(ground_truth, channel_name)
            residuals = (target_values - prediction_values).astype(
                FLOAT32_DTYPE, copy=False
            )
            valid_mask = _combine_valid_masks(
                _get_valid_mask(predictions, channel_name),
                _get_valid_mask(ground_truth, channel_name),
            )
            lower_residual_quantiles[channel_name] = _masked_column_quantile(
                residuals,
                valid_mask,
                alpha / 2.0,
            )
            upper_residual_quantiles[channel_name] = _masked_column_quantile(
                residuals,
                valid_mask,
                1.0 - (alpha / 2.0),
            )
            if strat == "pred_bins":
                if pred_bin_edges is None or channel_name not in pred_bin_edges:
                    raise ValueError(
                        f"pred_bins stratification is missing stored bin edges for channel {channel_name!r}."
                    )
                prediction_bins = _assign_prediction_bins(
                    prediction_values,
                    pred_bin_edges[channel_name],
                    valid_mask=valid_mask,
                )
                n_channel_bins = _prediction_bin_count(pred_bin_edges[channel_name])
                lower_by_stratum = np.zeros(
                    (n_channel_bins, residuals.shape[1]),
                    dtype=FLOAT32_DTYPE,
                )
                upper_by_stratum = np.zeros_like(lower_by_stratum)
                for index in range(n_channel_bins):
                    stratum_valid_mask = prediction_bins == index
                    if valid_mask is not None:
                        stratum_valid_mask &= valid_mask
                    lower_by_stratum[index] = _masked_column_quantile(
                        residuals,
                        stratum_valid_mask,
                        alpha / 2.0,
                    )
                    upper_by_stratum[index] = _masked_column_quantile(
                        residuals,
                        stratum_valid_mask,
                        1.0 - (alpha / 2.0),
                    )
                stratified_lower_residual_quantiles[channel_name] = lower_by_stratum
                stratified_upper_residual_quantiles[channel_name] = upper_by_stratum
                continue

            channel_strata = shared_scenario_strata
            channel_stratification_values = stratification_values
            if channel_strata is not None and channel_stratification_values is not None:
                lower_by_stratum = np.zeros(
                    (len(channel_stratification_values), residuals.shape[1]),
                    dtype=FLOAT32_DTYPE,
                )
                upper_by_stratum = np.zeros_like(lower_by_stratum)
                for index, stratum_value in enumerate(channel_stratification_values):
                    stratum_mask = channel_strata == stratum_value
                    stratum_valid_mask = (
                        None if valid_mask is None else valid_mask[stratum_mask]
                    )
                    lower_by_stratum[index] = _masked_column_quantile(
                        residuals[stratum_mask],
                        stratum_valid_mask,
                        alpha / 2.0,
                    )
                    upper_by_stratum[index] = _masked_column_quantile(
                        residuals[stratum_mask],
                        stratum_valid_mask,
                        1.0 - (alpha / 2.0),
                    )
                stratified_lower_residual_quantiles[channel_name] = lower_by_stratum
                stratified_upper_residual_quantiles[channel_name] = upper_by_stratum

        return cls(
            alpha=float(alpha),
            entity_ids=_get_entity_ids(predictions).copy(),
            entity_name=_get_entity_name(predictions),
            channel_names=selected_channels,
            lower_residual_quantiles=lower_residual_quantiles,
            upper_residual_quantiles=upper_residual_quantiles,
            strat=strat,
            stratification_values=None
            if stratification_values is None
            else stratification_values.copy(),
            stratified_lower_residual_quantiles=stratified_lower_residual_quantiles,
            stratified_upper_residual_quantiles=stratified_upper_residual_quantiles,
            system_load_n_bins=system_load_n_bins,
            system_load_bin_edges=None
            if system_load_bin_edges is None
            else system_load_bin_edges.astype(FLOAT32_DTYPE, copy=True),
            pred_n_bins=pred_n_bins,
            pred_bin_edges={
                name: values.astype(FLOAT32_DTYPE, copy=True)
                for name, values in (pred_bin_edges or {}).items()
            },
            sparse_strata=sparse_strata,
        )

    def predict_intervals(
        self,
        predictions: PredictionSet | ChannelSet,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Intervals for new predictions.

        Args:
            predictions: Predictions with the same entities as at fit time. With
                scenario stratification, they must carry the stratum column.

        Returns:
            `{variable: (lower, upper)}`, two arrays of the predictions' shape.
            Where an element's stratum had too few samples or none, the bounds
            use its quantile over all strata (marked in `pooled_masks`), or are
            `inf` / `-inf` with `sparse_strata="infinite"` or when even all
            strata together are too few.
        """

        _validate_entity_alignment(self.entity_ids, _get_entity_ids(predictions))
        self._pooled = {}
        intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        shared_scenario_strata = (
            None
            if self.strat is None or self.strat == "pred_bins"
            else _get_scenario_column(predictions, self.strat)
        )
        for channel_name in self.channel_names:
            prediction_values = _get_prediction_channel(predictions, channel_name)
            lower_offsets = np.repeat(
                self.lower_residual_quantiles[channel_name][None, :],
                len(prediction_values),
                axis=0,
            )
            upper_offsets = np.repeat(
                self.upper_residual_quantiles[channel_name][None, :],
                len(prediction_values),
                axis=0,
            )
            stratified = (
                channel_name in self.stratified_lower_residual_quantiles
                and channel_name in self.stratified_upper_residual_quantiles
            )
            if stratified:
                # Scenarios of a stratum the calibration data lacks keep these.
                lower_offsets = np.full(lower_offsets.shape, -np.inf, FLOAT32_DTYPE)
                upper_offsets = np.full(upper_offsets.shape, np.inf, FLOAT32_DTYPE)
                if self.strat == "pred_bins":
                    if channel_name not in self.pred_bin_edges:
                        raise ValueError(
                            f"pred_bins stratification is missing stored bin edges for channel {channel_name!r}."
                        )
                    prediction_bins = _assign_prediction_bins(
                        prediction_values,
                        self.pred_bin_edges[channel_name],
                        valid_mask=_get_valid_mask(predictions, channel_name),
                    )
                    for index in range(
                        self.stratified_lower_residual_quantiles[channel_name].shape[0]
                    ):
                        stratum_mask = prediction_bins == index
                        if not np.any(stratum_mask):
                            continue
                        lower_offsets = np.where(
                            stratum_mask,
                            self.stratified_lower_residual_quantiles[channel_name][
                                index
                            ][None, :],
                            lower_offsets,
                        )
                        upper_offsets = np.where(
                            stratum_mask,
                            self.stratified_upper_residual_quantiles[channel_name][
                                index
                            ][None, :],
                            upper_offsets,
                        )
                elif (
                    shared_scenario_strata is not None
                    and self.stratification_values is not None
                ):
                    for index, stratum_value in enumerate(self.stratification_values):
                        stratum_mask = shared_scenario_strata == stratum_value
                        if not np.any(stratum_mask):
                            continue
                        lower_offsets[stratum_mask] = (
                            self.stratified_lower_residual_quantiles[channel_name][
                                index
                            ][None, :]
                        )
                        upper_offsets[stratum_mask] = (
                            self.stratified_upper_residual_quantiles[channel_name][
                                index
                            ][None, :]
                        )
            valid_mask = _get_valid_mask(predictions, channel_name)
            if stratified and self.sparse_strata == "pooled":
                lower_offsets, upper_offsets, pooled = pool_sparse_strata(
                    lower_offsets,
                    upper_offsets,
                    self.lower_residual_quantiles[channel_name],
                    self.upper_residual_quantiles[channel_name],
                )
                self._pooled[channel_name] = (
                    pooled if valid_mask is None else pooled & valid_mask
                )
            lower = prediction_values + lower_offsets
            upper = prediction_values + upper_offsets
            if valid_mask is not None:
                lower = np.where(valid_mask, lower, prediction_values)
                upper = np.where(valid_mask, upper, prediction_values)
            intervals[channel_name] = (
                lower.astype(FLOAT32_DTYPE, copy=False),
                upper.astype(FLOAT32_DTYPE, copy=False),
            )
        return intervals

    @property
    def pooled_masks(self) -> dict[str, np.ndarray]:
        """Entries of the last `predict_intervals` call that used the pooled quantile.

        One boolean array per variable, of the predictions' shape; empty unless
        `sparse_strata` is `"pooled"` and the calibrator is stratified.
        """

        return self._pooled

    def sparse_strata_counts(self) -> dict[str, dict[int, int]]:
        """Per variable and stratum, the elements too sparse for a finite bound.

        `{variable: {stratum: n_elements}}`, strata with none left out; the
        stratum is the value of `strat` (the bin index for `"pred_bins"`).
        Only elements with a finite quantile over all strata are counted: those
        are the ones `sparse_strata="pooled"` makes finite.
        """

        counts: dict[str, dict[int, int]] = {}
        for name in self.stratified_lower_residual_quantiles:
            per_stratum = _count_sparse(
                self.stratified_lower_residual_quantiles[name],
                self.stratified_upper_residual_quantiles[name],
                self.lower_residual_quantiles[name],
                self.upper_residual_quantiles[name],
            )
            values = (
                np.arange(len(per_stratum))
                if self.stratification_values is None
                else self.stratification_values
            )
            counts[name] = {
                int(value): int(count)
                for value, count in zip(values, per_stratum, strict=True)
                if count
            }
        return counts

    def to_dict(self) -> dict[str, object]:
        """Serialize the calibrator into a JSON-friendly dictionary."""

        return {
            "version": "trustlayer_naive_interval_v2",
            "method": "naive_residual_quantiles",
            "alpha": self.alpha,
            "entity_ids": self.entity_ids.tolist(),
            "entity_name": self.entity_name,
            "bus_ids": self.entity_ids.tolist() if self.entity_name == "bus" else None,
            "channel_names": list(self.channel_names),
            "strat": self.strat,
            "stratification_values": None
            if self.stratification_values is None
            else self.stratification_values.tolist(),
            "calibration_fingerprint": self.calibration_fingerprint,
            "lower_residual_quantiles": {
                key: values.tolist()
                for key, values in self.lower_residual_quantiles.items()
            },
            "upper_residual_quantiles": {
                key: values.tolist()
                for key, values in self.upper_residual_quantiles.items()
            },
            "stratified_lower_residual_quantiles": {
                key: values.tolist()
                for key, values in self.stratified_lower_residual_quantiles.items()
            },
            "stratified_upper_residual_quantiles": {
                key: values.tolist()
                for key, values in self.stratified_upper_residual_quantiles.items()
            },
            "system_load_n_bins": self.system_load_n_bins,
            "system_load_bin_edges": None
            if self.system_load_bin_edges is None
            else self.system_load_bin_edges.tolist(),
            "pred_n_bins": self.pred_n_bins,
            "pred_bin_edges": {
                key: values.tolist() for key, values in self.pred_bin_edges.items()
            },
            "sparse_strata": self.sparse_strata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NaiveIntervalCalibrator:
        """Deserialize a calibrator previously produced by :meth:`to_dict`."""

        version = data.get("version")
        if version not in {
            "trustlayer_naive_interval_v1",
            "trustlayer_naive_interval_v2",
        }:
            raise ValueError(f"Unsupported interval artifact version: {version!r}")

        raw_channel_names = data.get("channel_names")
        if not isinstance(raw_channel_names, list):
            raise ValueError("Interval artifact field 'channel_names' must be a list.")
        raw_lower_residual_quantiles = data.get("lower_residual_quantiles")
        if not isinstance(raw_lower_residual_quantiles, dict):
            raise ValueError(
                "Interval artifact field 'lower_residual_quantiles' must be a mapping."
            )
        raw_upper_residual_quantiles = data.get("upper_residual_quantiles")
        if not isinstance(raw_upper_residual_quantiles, dict):
            raise ValueError(
                "Interval artifact field 'upper_residual_quantiles' must be a mapping."
            )
        raw_stratified_lower_residual_quantiles = data.get(
            "stratified_lower_residual_quantiles", {}
        )
        if not isinstance(raw_stratified_lower_residual_quantiles, dict):
            raise ValueError(
                "Interval artifact field 'stratified_lower_residual_quantiles' must be a mapping."
            )
        raw_stratified_upper_residual_quantiles = data.get(
            "stratified_upper_residual_quantiles", {}
        )
        if not isinstance(raw_stratified_upper_residual_quantiles, dict):
            raise ValueError(
                "Interval artifact field 'stratified_upper_residual_quantiles' must be a mapping."
            )
        raw_pred_bin_edges = data.get("pred_bin_edges", {})
        if not isinstance(raw_pred_bin_edges, dict):
            raise ValueError(
                "Interval artifact field 'pred_bin_edges' must be a mapping."
            )

        channel_names = tuple(str(name) for name in raw_channel_names)
        lower_residual_quantiles = {
            str(key): np.asarray(values, dtype=FLOAT32_DTYPE)
            for key, values in raw_lower_residual_quantiles.items()
        }
        upper_residual_quantiles = {
            str(key): np.asarray(values, dtype=FLOAT32_DTYPE)
            for key, values in raw_upper_residual_quantiles.items()
        }
        stratified_lower_residual_quantiles = {
            str(key): np.asarray(values, dtype=FLOAT32_DTYPE)
            for key, values in raw_stratified_lower_residual_quantiles.items()
        }
        stratified_upper_residual_quantiles = {
            str(key): np.asarray(values, dtype=FLOAT32_DTYPE)
            for key, values in raw_stratified_upper_residual_quantiles.items()
        }
        pred_bin_edges = {
            str(key): np.asarray(values, dtype=FLOAT32_DTYPE)
            for key, values in raw_pred_bin_edges.items()
        }
        return cls(
            alpha=float(cast(Any, data["alpha"])),
            entity_ids=np.asarray(
                data.get("entity_ids", data.get("bus_ids")), dtype=np.int64
            ),
            entity_name=str(data.get("entity_name", "bus")),
            channel_names=channel_names,
            lower_residual_quantiles=lower_residual_quantiles,
            upper_residual_quantiles=upper_residual_quantiles,
            strat=cast(str | None, data.get("strat")),
            stratification_values=_optional_int_array(
                data.get("stratification_values")
            ),
            stratified_lower_residual_quantiles=stratified_lower_residual_quantiles,
            stratified_upper_residual_quantiles=stratified_upper_residual_quantiles,
            system_load_n_bins=_optional_int(data.get("system_load_n_bins")),
            system_load_bin_edges=_optional_float_array(
                data.get("system_load_bin_edges")
            ),
            pred_n_bins=_optional_int(data.get("pred_n_bins")),
            pred_bin_edges=pred_bin_edges,
            calibration_fingerprint=_optional_str(data.get("calibration_fingerprint")),
            sparse_strata=validate_sparse_strata(
                str(data.get("sparse_strata", DEFAULT_SPARSE_STRATA))
            ),
        )

    @property
    def bus_ids(self) -> np.ndarray:
        """Compatibility alias for older bus-only call sites."""

        return self.entity_ids


def _get_prediction_channel(
    predictions: PredictionSet | ChannelSet,
    channel_name: str,
) -> np.ndarray:
    if isinstance(predictions, ChannelSet):
        if channel_name not in predictions.channels:
            raise ValueError(f"Prediction channel not available: {channel_name}")
        return predictions.channels[channel_name]
    if channel_name == "Vm":
        return predictions.Vm
    if channel_name == "Va":
        return predictions.Va
    if channel_name not in predictions.channels:
        raise ValueError(f"Prediction channel not available: {channel_name}")
    return predictions.channels[channel_name]


def _get_ground_truth_channel(
    ground_truth: GroundTruthSet | ChannelSet,
    channel_name: str,
) -> np.ndarray:
    if isinstance(ground_truth, ChannelSet):
        if channel_name not in ground_truth.channels:
            raise ValueError(f"Ground-truth channel not available: {channel_name}")
        return ground_truth.channels[channel_name]
    if channel_name == "Vm":
        return ground_truth.Vm
    if channel_name == "Va":
        return ground_truth.Va
    if channel_name not in ground_truth.channels:
        raise ValueError(f"Ground-truth channel not available: {channel_name}")
    return ground_truth.channels[channel_name]


def validate_sparse_strata(value: str) -> str:
    if value not in SPARSE_STRATA_CHOICES:
        raise ValueError(
            f"sparse_strata must be one of {', '.join(SPARSE_STRATA_CHOICES)}; "
            f"got {value!r}."
        )
    return value


def pool_sparse_strata(
    lower: np.ndarray,
    upper: np.ndarray,
    pooled_lower: np.ndarray,
    pooled_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace infinite (scenario x element) offsets by the element's pooled quantiles.

    Each tail is replaced where it is infinite and the pooled quantile is
    finite. Returns the new offsets and the mask of entries changed.
    """

    use_lower = ~np.isfinite(lower) & np.isfinite(pooled_lower)[None, :]
    use_upper = ~np.isfinite(upper) & np.isfinite(pooled_upper)[None, :]
    return (
        np.where(use_lower, pooled_lower[None, :], lower).astype(FLOAT32_DTYPE),
        np.where(use_upper, pooled_upper[None, :], upper).astype(FLOAT32_DTYPE),
        use_lower | use_upper,
    )


def _count_sparse(
    lower: np.ndarray,
    upper: np.ndarray,
    pooled_lower: np.ndarray,
    pooled_upper: np.ndarray,
) -> np.ndarray:
    """Per stratum (row), the elements with an infinite bound but a finite pooled one."""

    sparse = ~np.isfinite(lower) | ~np.isfinite(upper)
    pooled_finite = np.isfinite(pooled_lower) & np.isfinite(pooled_upper)
    return np.asarray((sparse & pooled_finite[None, :]).sum(axis=1), dtype=np.int64)


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1; got {alpha!r}")


def _validate_entity_alignment(
    left_entity_ids: np.ndarray, right_entity_ids: np.ndarray
) -> None:
    if not np.array_equal(left_entity_ids, right_entity_ids):
        raise ValueError(
            "Prediction and calibration artifacts must share the same entity ordering."
        )


def _get_entity_ids(values: PredictionSet | GroundTruthSet | ChannelSet) -> np.ndarray:
    if isinstance(values, ChannelSet):
        return values.entity_ids
    return values.bus_ids


def _get_entity_name(values: PredictionSet | GroundTruthSet | ChannelSet) -> str:
    if isinstance(values, ChannelSet):
        return values.entity_name
    return "bus"


def _get_scenario_column(
    values: PredictionSet | GroundTruthSet | ChannelSet,
    column_name: str,
) -> np.ndarray:
    if isinstance(values, ChannelSet):
        if column_name not in values.scenario_columns:
            raise ValueError(f"Scenario column not available: {column_name}")
        return values.scenario_columns[column_name]
    if isinstance(values, GroundTruthSet):
        if column_name == "n_outages":
            return values.n_outages
        if column_name == "outage_type":
            return values.outage_type
        if column_name == "system_load_bin":
            raise ValueError(
                "system_load_bin must be derived before fitting or prediction application.",
            )
    raise ValueError(
        f"Scenario column {column_name!r} is not available on this prediction artifact."
    )


def _get_valid_mask(
    values: PredictionSet | GroundTruthSet | ChannelSet,
    channel_name: str,
) -> np.ndarray | None:
    if isinstance(values, ChannelSet):
        return values.valid_masks.get(channel_name)
    return None


def _combine_valid_masks(
    left: np.ndarray | None, right: np.ndarray | None
) -> np.ndarray | None:
    if left is None:
        return None if right is None else right
    if right is None:
        return left
    return left & right


def _masked_column_quantile(
    residuals: np.ndarray,
    valid_mask: np.ndarray | None,
    quantile: float,
) -> np.ndarray:
    """Per-column split-conformal residual quantile over the valid entries.

    With `n` valid residuals in a column, an upper level `q >= 0.5` returns the
    `ceil((n + 1) q)`-th smallest residual and a lower level `q < 0.5` the
    `floor((n + 1) q)`-th smallest. This is the finite-sample correction of
    split conformal prediction: each tail then holds with probability at least
    its nominal level under exchangeability, which the plain empirical quantile
    does not guarantee. When the rank falls outside `1..n` (too few residuals
    for the requested level, including `n = 0`) the bound is `+inf` or `-inf`:
    without enough calibration data no finite margin is valid.
    """

    upper_tail = quantile >= 0.5
    output = np.empty(residuals.shape[1], dtype=FLOAT32_DTYPE)
    for index in range(residuals.shape[1]):
        column = residuals[:, index]
        if valid_mask is not None:
            column = column[valid_mask[:, index]]
        n = column.size
        if upper_tail:
            rank = int(np.ceil((n + 1) * quantile - _RANK_TOLERANCE))
            in_range = rank <= n
            fallback = np.inf
        else:
            rank = int(np.floor((n + 1) * quantile + _RANK_TOLERANCE))
            in_range = rank >= 1
            fallback = -np.inf
        if in_range:
            output[index] = np.partition(column, rank - 1)[rank - 1]
        else:
            output[index] = fallback
    return output


def _assign_prediction_bins(
    prediction_values: np.ndarray,
    pred_bin_edges: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(prediction_values, dtype=np.float32)
    edges = np.asarray(pred_bin_edges, dtype=np.float32)
    if valid_mask is None:
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "pred_bins stratification requires finite prediction values."
            )
    else:
        if not np.all(np.isfinite(values[valid_mask])):
            raise ValueError(
                "pred_bins stratification requires finite prediction values on valid entries."
            )
    bins = np.searchsorted(edges, values, side="right").astype(np.int32, copy=False)
    if valid_mask is not None:
        bins = np.where(valid_mask, bins, 0).astype(np.int32, copy=False)
    return bins


def _prediction_bin_count(pred_bin_edges: np.ndarray) -> int:
    return len(np.asarray(pred_bin_edges, dtype=np.float32)) + 1


def _optional_int_array(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.int64)


def _optional_str(value: object | None) -> str | None:
    return None if value is None else str(value)


def _optional_float_array(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=FLOAT32_DTYPE)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(cast(Any, value))
