"""Kernel-weighted conformal prediction (KCP) interval calibration.

Unlike :class:`~gridfm_trustlayer.conformal.calibrator.NaiveIntervalCalibrator`,
which reduces calibration residuals to a fixed quantile table, KCP produces a
*test-point-specific* quantile. The fitted object therefore stores the
calibration residual bank plus scenario embeddings, and the quantile is
estimated at application time by weighting calibration residuals with a
Gaussian kernel over embedding distance.

Following Tibshirani et al. (2019), the weighted quantile carries a point mass
at ``+inf`` representing the unseen test residual, so the estimator degrades to
an explicit fallback rather than silently returning an over-optimistic bound.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, cast

import numpy as np

from gridfm_trustlayer.conformal.calibrator import (
    DEFAULT_SPARSE_STRATA,
    ChannelSet,
    _combine_valid_masks,
    _count_sparse,
    _get_entity_ids,
    _get_entity_name,
    _get_ground_truth_channel,
    _get_prediction_channel,
    _get_scenario_column,
    _get_valid_mask,
    _masked_column_quantile,
    _validate_alpha,
    _validate_entity_alignment,
    _with_finite,
    pool_sparse_strata,
    validate_sparse_strata,
)
from gridfm_trustlayer.io.ground_truth import GroundTruthSet
from gridfm_trustlayer.io.predictions import PredictionSet

FLOAT32_DTYPE = np.dtype(np.float32)
INT32_DTYPE = np.dtype(np.int32)

ARTIFACT_VERSION = "trustlayer_kernel_interval_v1"

DEFAULT_K_NN = 20

# Warn when at least this share of the scenarios falls back to the stratum
# quantiles: the kernel then barely changes the bounds. The kernel runs on
# case24 fall back in 0 to 5% of the scenarios (strata absent from the
# calibration data, or a few scenarios far from every calibration scenario).
FALLBACK_WARNING_SHARE = 0.5


class KernelFallbackWarning(UserWarning):
    """Most scenarios got the stratum quantiles instead of kernel-weighted bounds.

    Raised by `KernelIntervalCalibrator.predict_intervals`,
    `predict_intervals_jointly` and the bandwidth tuning when at least half of
    the scenarios fall back, which usually means the calibration scenarios are
    too far from them in the embedding. The bounds stay valid: they are those
    of the stratified (element) method. Silence it with
    `warnings.simplefilter("ignore", KernelFallbackWarning)`.
    """


DEFAULT_N_EFF_MIN = 5.0

DISTANCE_EUCLIDEAN = "euclidean"
DISTANCE_COSINE = "cosine"
SUPPORTED_DISTANCES = (DISTANCE_EUCLIDEAN, DISTANCE_COSINE)
DEFAULT_DISTANCE = DISTANCE_EUCLIDEAN

BANDWIDTH_KNN = "knn"
BANDWIDTH_N_EFF = "n_eff"
SUPPORTED_BANDWIDTHS = (BANDWIDTH_KNN, BANDWIDTH_N_EFF)
DEFAULT_BANDWIDTH = BANDWIDTH_N_EFF

# Default target effective sample size for `bandwidth: n_eff`. It is
# data-dependent: in a stratum with fewer calibration scenarios the weights
# become uniform there (equivalent to cp_type element); `n_eff_target: auto`
# chooses the target from the calibration data instead.
DEFAULT_N_EFF_TARGET = 600.0

# Artifacts written before the bandwidth option existed carry no `bandwidth`
# key; they used the k-nearest-neighbour rule.
_LEGACY_ARTIFACT_BANDWIDTH = BANDWIDTH_KNN

# Settings of the n_eff bandwidth solve (Newton steps in log space, see
# `_solve_bandwidth_for_n_eff`). A tight tolerance makes each bound independent
# of how the solve gets there: a bandwidth anywhere within a loose tolerance can
# move a weighted quantile to a neighbouring residual. Rows that have not
# converged after `_N_EFF_SOLVE_NEWTON_ITER` iterations continue by bisection,
# which needs well under `_N_EFF_SOLVE_MAX_ITER` in total.
_N_EFF_SOLVE_NEWTON_ITER = 30
_N_EFF_SOLVE_MAX_ITER = 90
_N_EFF_SOLVE_RTOL = 1e-6
_N_EFF_BRACKET_SPAN = 1e3

# Target working-set size for one block of test scenarios, in bytes, and the
# largest block: small blocks keep every thread busy when a stratum is small.
# Neither changes any result.
_TARGET_BLOCK_BYTES = 48 * 1024 * 1024
_MAX_BLOCK_ROWS = 128
# Skip the quantile search only when the weight sum is clearly below the
# minimum, so float rounding can never skip a reachable case.
_UNREACHABLE_MARGIN = 1.0 - 1e-6
# Columns (elements) argsorted together when a calibration bank is built.
_SORT_CHUNK = 64

# Embedding dimensions with (near) zero spread carry no distance information
# and would blow up the standardized values.
_MIN_EMBEDDING_SCALE = 1e-6

# A standardized scenario sitting exactly on the mean has zero norm and cannot
# be projected onto the unit sphere.
_MIN_EMBEDDING_NORM = 1e-12


@dataclass
class _StratumBank:
    """Sorted calibration residuals for one stratum and one channel."""

    rows: np.ndarray
    # Element-major, `(n_elements, n_calibration)`: per element, the residuals
    # in increasing order, and the calibration index of each sorted position
    # with invalid entries pointing at `n_calibration` (a zero weight).
    sorted_residuals: np.ndarray
    order: np.ndarray
    # Validity in calibration order, float64: kernel weights @ valid gives the
    # total weight per element in one matrix product.
    valid: np.ndarray
    fallback_lower: np.ndarray
    fallback_upper: np.ndarray
    # Per element: whether the stratum holds at least one valid residual.
    has_data: np.ndarray

    @classmethod
    def build(
        cls,
        rows: np.ndarray,
        residuals: np.ndarray,
        valid: np.ndarray,
        fallback_lower: np.ndarray,
        fallback_upper: np.ndarray,
        executor: ThreadPoolExecutor | None = None,
    ) -> _StratumBank:
        """Sort `(n_calibration, n_elements)` residuals per element (stable).

        Columns are sorted in chunks, on `executor` when given; a stable sort
        gives the same order whichever way the columns are split.
        """

        n_calibration, n_elements = residuals.shape
        by_element = np.ascontiguousarray(residuals.T, dtype=FLOAT32_DTYPE)
        valid_by_element = np.ascontiguousarray(np.asarray(valid, dtype=bool).T)
        sorted_residuals = np.empty_like(by_element)
        order = np.empty(by_element.shape, dtype=INT32_DTYPE)

        def sort(start: int) -> None:
            part = slice(start, start + _SORT_CHUNK)
            part_order = np.argsort(by_element[part], axis=1, kind="stable")
            sorted_residuals[part] = np.take_along_axis(
                by_element[part], part_order, axis=1
            )
            order[part] = np.where(
                np.take_along_axis(valid_by_element[part], part_order, axis=1),
                part_order,
                n_calibration,
            )

        starts = range(0, n_elements, _SORT_CHUNK) if n_calibration else range(0)
        if executor is None:
            for start in starts:
                sort(start)
        else:
            list(executor.map(sort, starts))
        return cls(
            rows=rows,
            sorted_residuals=sorted_residuals,
            order=order,
            valid=np.ascontiguousarray(valid, dtype=np.float64),
            fallback_lower=fallback_lower,
            fallback_upper=fallback_upper,
            has_data=np.asarray(np.asarray(valid, dtype=bool).any(axis=0)),
        )


@dataclass
class KernelIntervalCalibrator:
    """Kernel-weighted conformal intervals (`cp_type: kernel`).

    For each test scenario, calibration residuals are weighted by a Gaussian
    kernel on the distance between scenario embeddings, within the scenario's
    stratum, and the bound is a weighted quantile with a point mass for the
    unseen test residual. The margin therefore adapts to each scenario.
    Coverage is checked empirically, not guaranteed in finite samples.

    Create it with `fit`; the artifact keeps all calibration residuals and
    standardized embeddings.
    """

    alpha: float
    entity_ids: np.ndarray
    entity_name: str
    channel_names: tuple[str, ...]
    strat: str | None
    stratification_values: np.ndarray
    k_nn: int
    n_eff_min: float
    bandwidth: str
    n_eff_target: float
    distance: str
    embedding_pooling: str
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    calibration_embeddings: np.ndarray
    calibration_strata: np.ndarray
    residuals: dict[str, np.ndarray]
    valid_masks: dict[str, np.ndarray]
    fallback_lower_quantiles: dict[str, np.ndarray] = field(default_factory=dict)
    fallback_upper_quantiles: dict[str, np.ndarray] = field(default_factory=dict)
    # Bin definitions are unused by the kernel weighting itself, but the CLI
    # rebuilds scenario strata from them at application time, so they must
    # survive a round trip exactly as they do for the naive calibrator.
    system_load_n_bins: int | None = None
    system_load_bin_edges: np.ndarray | None = None
    pred_n_bins: int | None = None
    pred_bin_edges: dict[str, np.ndarray] = field(default_factory=dict)
    # SHA-256 of the calibration predictions, so that applying the artifact to the
    # very same data (in-sample evaluation) can be detected and refused.
    calibration_fingerprint: str | None = None
    # Which branches and generators are in service in the calibration set, and
    # the units of its voltage angles (`GroundTruthReference.to_dict`), set by
    # the CLI: the other commands count outages and convert angles the same way.
    ground_truth_reference: dict[str, Any] | None = None
    # How n_eff_target was chosen when `n_eff_target: auto` (cross-validation
    # summary from gridfm_trustlayer.conformal.tuning); None when set by hand.
    tuning: dict[str, Any] | None = None
    sparse_strata: str = DEFAULT_SPARSE_STRATA
    n_jobs: int | None = None
    _banks: dict[str, list[_StratumBank]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _diagnostics: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )
    _pooled: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )
    _pooled_quantiles: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(
        cls,
        predictions: PredictionSet | ChannelSet,
        ground_truth: GroundTruthSet | ChannelSet,
        *,
        alpha: float,
        channel_names: Iterable[str] = ("Vm", "Va"),
        strat: str | None = None,
        embeddings: np.ndarray,
        embedding_pooling: str = "mean",
        k_nn: int = DEFAULT_K_NN,
        n_eff_min: float = DEFAULT_N_EFF_MIN,
        bandwidth: str = DEFAULT_BANDWIDTH,
        n_eff_target: float | None = None,
        distance: str = DEFAULT_DISTANCE,
        system_load_n_bins: int | None = None,
        system_load_bin_edges: np.ndarray | None = None,
        sparse_strata: str = DEFAULT_SPARSE_STRATA,
    ) -> KernelIntervalCalibrator:
        """Store the calibration residuals and embeddings that `predict_intervals` weights.

        Args:
            predictions: Calibration predictions.
            ground_truth: True values for the same scenarios and entities.
            alpha: Two-sided miscoverage level; `0.1` gives 90% intervals.
            channel_names: Variables to calibrate.
            strat: Optional scenario stratification (`"n_outages"`,
                `"outage_type"` or `"system_load_bin"`); the kernel localizes
                within each stratum. `"pred_bins"` is not supported.
            embeddings: `(n_scenarios, d)` scenario embeddings of the
                calibration scenarios, in the order of `predictions`.
            embedding_pooling: Name of the pooling that produced `embeddings`
                (default `"mean"`), stored so that the CLI pools test
                embeddings the same way; the calibrator itself does not use it.
            k_nn: Neighbour rank that sets the bandwidth, for `bandwidth="knn"`.
            n_eff_min: Effective sample size below which a scenario falls back
                to the stratum-wide quantile.
            bandwidth: `"n_eff"` (solve for `n_eff_target`) or `"knn"`.
            n_eff_target: Target effective sample size for `bandwidth="n_eff"`;
                600 when not given.
            distance: `"euclidean"` or `"cosine"`.
            system_load_n_bins: Number of load bins, for `"system_load_bin"`.
            system_load_bin_edges: Stored load-bin edges, for `"system_load_bin"`.
            sparse_strata: `"pooled"` (default) or `"infinite"`: the fallback
                bound of an element in a stratum with too few calibration
                samples, or of a stratum absent from the calibration data.

        Returns:
            The fitted calibrator.

        Raises:
            ValueError: If inputs are misaligned, an option is invalid, or
                `strat="pred_bins"`.
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
        if strat == "pred_bins":
            raise ValueError(
                "cp_type 'kernel' does not support strat 'pred_bins': prediction bins are "
                "element-wise while kernel localization is scenario-wise. Use 'n_outages', "
                "'outage_type', 'system_load_bin', or omit strat.",
            )
        embeddings = _finite_embeddings(
            np.ascontiguousarray(embeddings, dtype=FLOAT32_DTYPE), "calibration"
        )
        if embeddings.shape[0] != len(predictions.scenario_ids):
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) do not match calibration "
                f"scenarios ({len(predictions.scenario_ids)}).",
            )
        if k_nn < 1:
            raise ValueError(f"k_nn must be >= 1; got {k_nn!r}")
        if n_eff_min < 1.0:
            raise ValueError(f"n_eff_min must be >= 1; got {n_eff_min!r}")
        if distance not in SUPPORTED_DISTANCES:
            supported = ", ".join(repr(name) for name in SUPPORTED_DISTANCES)
            raise ValueError(
                f"Unsupported distance: {distance!r}. Supported values: {supported}.",
            )
        if bandwidth not in SUPPORTED_BANDWIDTHS:
            supported = ", ".join(repr(name) for name in SUPPORTED_BANDWIDTHS)
            raise ValueError(
                f"Unsupported bandwidth: {bandwidth!r}. Supported values: {supported}.",
            )
        resolved_n_eff_target = (
            DEFAULT_N_EFF_TARGET if n_eff_target is None else float(n_eff_target)
        )
        if resolved_n_eff_target < 1.0:
            raise ValueError(
                f"n_eff_target must be >= 1; got {resolved_n_eff_target!r}",
            )
        if bandwidth == BANDWIDTH_N_EFF and resolved_n_eff_target < n_eff_min:
            raise ValueError(
                f"n_eff_target ({resolved_n_eff_target}) is below n_eff_min ({n_eff_min}), "
                "so every scenario would immediately fall back to the stratum quantile.",
            )

        embedding_mean = embeddings.mean(axis=0)
        embedding_scale = embeddings.std(axis=0)
        embedding_scale = np.where(
            embedding_scale < _MIN_EMBEDDING_SCALE, 1.0, embedding_scale
        ).astype(FLOAT32_DTYPE, copy=False)
        standardized = _prepare_embeddings(
            embeddings, embedding_mean, embedding_scale, distance
        )

        strata: np.ndarray
        stratification_values: np.ndarray
        if strat is None:
            strata = np.zeros(len(predictions.scenario_ids), dtype=np.int64)
            stratification_values = np.zeros(1, dtype=np.int64)
        else:
            strata = _get_scenario_column(ground_truth, strat).astype(
                np.int64, copy=False
            )
            stratification_values = np.sort(np.unique(strata))

        selected_channels = tuple(channel_names)
        residuals: dict[str, np.ndarray] = {}
        valid_masks: dict[str, np.ndarray] = {}
        fallback_lower: dict[str, np.ndarray] = {}
        fallback_upper: dict[str, np.ndarray] = {}
        for channel_name in selected_channels:
            prediction_values = _get_prediction_channel(predictions, channel_name)
            target_values = _get_ground_truth_channel(ground_truth, channel_name)
            channel_residuals = (target_values - prediction_values).astype(
                FLOAT32_DTYPE, copy=False
            )
            valid_mask = _with_finite(
                _combine_valid_masks(
                    _get_valid_mask(predictions, channel_name),
                    _get_valid_mask(ground_truth, channel_name),
                ),
                channel_residuals,
            )
            if valid_mask is None:
                valid_mask = np.ones(channel_residuals.shape, dtype=bool)
            # Masked out above; zeroed so that they cannot reach any sum.
            channel_residuals = np.nan_to_num(
                channel_residuals, nan=0.0, posinf=0.0, neginf=0.0
            )
            residuals[channel_name] = channel_residuals
            valid_masks[channel_name] = valid_mask

            lower_by_stratum = np.zeros(
                (len(stratification_values), channel_residuals.shape[1]),
                dtype=FLOAT32_DTYPE,
            )
            upper_by_stratum = np.zeros_like(lower_by_stratum)
            for index, stratum_value in enumerate(stratification_values):
                stratum_mask = strata == stratum_value
                lower_by_stratum[index] = _masked_column_quantile(
                    channel_residuals[stratum_mask],
                    valid_mask[stratum_mask],
                    alpha / 2.0,
                )
                upper_by_stratum[index] = _masked_column_quantile(
                    channel_residuals[stratum_mask],
                    valid_mask[stratum_mask],
                    1.0 - (alpha / 2.0),
                )
            fallback_lower[channel_name] = lower_by_stratum
            fallback_upper[channel_name] = upper_by_stratum

        return cls(
            alpha=float(alpha),
            entity_ids=_get_entity_ids(predictions).copy(),
            entity_name=_get_entity_name(predictions),
            channel_names=selected_channels,
            strat=strat,
            stratification_values=stratification_values,
            k_nn=int(k_nn),
            n_eff_min=float(n_eff_min),
            bandwidth=bandwidth,
            n_eff_target=resolved_n_eff_target,
            distance=distance,
            embedding_pooling=str(embedding_pooling),
            embedding_mean=embedding_mean.astype(FLOAT32_DTYPE, copy=False),
            embedding_scale=embedding_scale,
            calibration_embeddings=np.ascontiguousarray(
                standardized, dtype=FLOAT32_DTYPE
            ),
            calibration_strata=strata,
            residuals=residuals,
            valid_masks=valid_masks,
            fallback_lower_quantiles=fallback_lower,
            fallback_upper_quantiles=fallback_upper,
            system_load_n_bins=system_load_n_bins,
            system_load_bin_edges=None
            if system_load_bin_edges is None
            else np.asarray(system_load_bin_edges, dtype=FLOAT32_DTYPE),
            sparse_strata=sparse_strata,
        )

    # -------------------------------------------------------------- predict

    def predict_intervals(
        self,
        predictions: PredictionSet | ChannelSet,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Scenario-specific intervals for new predictions.

        Args:
            predictions: Predictions with the same entities as at fit time, with
                `scenario_embeddings` pooled like the calibration embeddings
                and, with stratification, the stratum column.

        Returns:
            `{variable: (lower, upper)}`, two arrays of the predictions' shape.
            Per-scenario diagnostics are available afterwards in `diagnostics`.
        """

        intervals = _predict_jointly([self], [predictions])[0]
        _warn_if_fallback(self, predictions, stacklevel=2)
        return intervals

    @property
    def diagnostics(self) -> dict[str, np.ndarray]:
        """Per-scenario diagnostics of the last `predict_intervals` call.

        `{"n_eff": ..., "kernel_bandwidth_sq": ..., "kernel_fallback": ...}`, one
        value per test scenario: the effective sample size of its weights, the
        squared bandwidth used, and whether it fell back to the stratum-wide
        quantile for at least one entity.
        """

        return self._diagnostics

    @property
    def pooled_masks(self) -> dict[str, np.ndarray]:
        """Entries of the last `predict_intervals` call that used the pooled quantile.

        One boolean array per variable, of the predictions' shape; empty unless
        `sparse_strata` is `"pooled"`.
        """

        return self._pooled

    def sparse_strata_counts(self) -> dict[str, dict[int, int]]:
        """Per variable and stratum, the elements too sparse for a finite fallback.

        `{variable: {stratum: n_elements}}`, strata with none left out. Only
        elements with a finite quantile over all calibration scenarios are
        counted: those are the ones `sparse_strata="pooled"` makes finite.
        Without stratification there is nothing to count.
        """

        if self.strat is None:
            return {}
        counts: dict[str, dict[int, int]] = {}
        for name in self.channel_names:
            pooled_lower, pooled_upper = self._pooled_quantiles_of(name)
            per_stratum = _count_sparse(
                self.fallback_lower_quantiles[name],
                self.fallback_upper_quantiles[name],
                pooled_lower,
                pooled_upper,
            )
            counts[name] = {
                int(value): int(count)
                for value, count in zip(
                    self.stratification_values, per_stratum, strict=True
                )
                if count
            }
        return counts

    # ------------------------------------------------------------ internals

    def _pooled_quantiles_of(self, channel_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Each element's quantiles over all calibration scenarios, computed once."""

        if channel_name not in self._pooled_quantiles:
            residuals = self.residuals[channel_name]
            valid = self.valid_masks[channel_name]
            self._pooled_quantiles[channel_name] = (
                _masked_column_quantile(residuals, valid, self.alpha / 2.0),
                _masked_column_quantile(residuals, valid, 1.0 - self.alpha / 2.0),
            )
        return self._pooled_quantiles[channel_name]

    def _ensure_banks(self, executor: ThreadPoolExecutor | None = None) -> None:
        """Sort the calibration residuals of every channel and stratum, once."""

        if self._banks:
            return
        banks: dict[str, list[_StratumBank]] = {}
        for channel_name in self.channel_names:
            channel_residuals = self.residuals[channel_name]
            channel_valid = self.valid_masks[channel_name]
            per_stratum: list[_StratumBank] = []
            for index, stratum_value in enumerate(self.stratification_values):
                rows = np.flatnonzero(self.calibration_strata == stratum_value)
                per_stratum.append(
                    _StratumBank.build(
                        rows,
                        channel_residuals[rows],
                        channel_valid[rows],
                        self.fallback_lower_quantiles[channel_name][index],
                        self.fallback_upper_quantiles[channel_name][index],
                        executor,
                    )
                )
            banks[channel_name] = per_stratum
        self._banks = banks

    # ------------------------------------------------------------ artifact

    def to_dict(self) -> dict[str, object]:
        """Serialize scalar metadata; large arrays go through `array_payload`."""

        return {
            "version": ARTIFACT_VERSION,
            "method": "kernel_weighted_conformal",
            "cp_type": "kernel",
            "alpha": self.alpha,
            "entity_ids": self.entity_ids.tolist(),
            "entity_name": self.entity_name,
            "channel_names": list(self.channel_names),
            "strat": self.strat,
            "stratification_values": self.stratification_values.tolist(),
            "k_nn": self.k_nn,
            "n_eff_min": self.n_eff_min,
            "bandwidth": self.bandwidth,
            "n_eff_target": self.n_eff_target,
            "distance": self.distance,
            "embedding_pooling": self.embedding_pooling,
            "system_load_n_bins": self.system_load_n_bins,
            "system_load_bin_edges": None
            if self.system_load_bin_edges is None
            else self.system_load_bin_edges.tolist(),
            "pred_n_bins": self.pred_n_bins,
            "pred_bin_edges": {
                key: values.tolist() for key, values in self.pred_bin_edges.items()
            },
            "calibration_fingerprint": self.calibration_fingerprint,
            "ground_truth_reference": self.ground_truth_reference,
            "tuning": self.tuning,
            "sparse_strata": self.sparse_strata,
        }

    def array_payload(self) -> dict[str, np.ndarray]:
        """Return the large float arrays that belong in the binary artifact side."""

        payload: dict[str, np.ndarray] = {
            "embedding_mean": self.embedding_mean,
            "embedding_scale": self.embedding_scale,
            "calibration_embeddings": self.calibration_embeddings,
            "calibration_strata": self.calibration_strata,
        }
        for channel_name in self.channel_names:
            payload[f"residuals::{channel_name}"] = self.residuals[channel_name]
            payload[f"valid_masks::{channel_name}"] = self.valid_masks[
                channel_name
            ].astype(np.bool_, copy=False)
            payload[f"fallback_lower::{channel_name}"] = self.fallback_lower_quantiles[
                channel_name
            ]
            payload[f"fallback_upper::{channel_name}"] = self.fallback_upper_quantiles[
                channel_name
            ]
        return payload

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        arrays: dict[str, np.ndarray],
    ) -> KernelIntervalCalibrator:
        """Rebuild a calibrator from its metadata mapping and array payload."""

        version = data.get("version")
        if version != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported kernel artifact version: {version!r}")
        raw_channel_names = data.get("channel_names")
        if not isinstance(raw_channel_names, list):
            raise ValueError("Kernel artifact field 'channel_names' must be a list.")
        channel_names = tuple(str(name) for name in raw_channel_names)

        residuals = {
            name: np.asarray(arrays[f"residuals::{name}"], dtype=FLOAT32_DTYPE)
            for name in channel_names
        }
        valid_masks = {
            name: np.asarray(arrays[f"valid_masks::{name}"], dtype=bool)
            for name in channel_names
        }
        fallback_lower = {
            name: np.asarray(arrays[f"fallback_lower::{name}"], dtype=FLOAT32_DTYPE)
            for name in channel_names
        }
        fallback_upper = {
            name: np.asarray(arrays[f"fallback_upper::{name}"], dtype=FLOAT32_DTYPE)
            for name in channel_names
        }
        return cls(
            alpha=float(cast(Any, data["alpha"])),
            entity_ids=np.asarray(data.get("entity_ids"), dtype=np.int64),
            entity_name=str(data.get("entity_name", "bus")),
            channel_names=channel_names,
            strat=cast(str | None, data.get("strat")),
            stratification_values=np.asarray(
                data.get("stratification_values"), dtype=np.int64
            ),
            k_nn=int(cast(Any, data.get("k_nn", DEFAULT_K_NN))),
            n_eff_min=float(cast(Any, data.get("n_eff_min", DEFAULT_N_EFF_MIN))),
            bandwidth=str(data.get("bandwidth", _LEGACY_ARTIFACT_BANDWIDTH)),
            distance=str(data.get("distance", DEFAULT_DISTANCE)),
            n_eff_target=float(
                cast(Any, data.get("n_eff_target", DEFAULT_N_EFF_TARGET))
            ),
            embedding_pooling=str(data.get("embedding_pooling", "mean")),
            embedding_mean=np.asarray(arrays["embedding_mean"], dtype=FLOAT32_DTYPE),
            embedding_scale=np.asarray(arrays["embedding_scale"], dtype=FLOAT32_DTYPE),
            calibration_embeddings=np.ascontiguousarray(
                arrays["calibration_embeddings"], dtype=FLOAT32_DTYPE
            ),
            calibration_strata=np.asarray(arrays["calibration_strata"], dtype=np.int64),
            residuals=residuals,
            valid_masks=valid_masks,
            fallback_lower_quantiles=fallback_lower,
            fallback_upper_quantiles=fallback_upper,
            system_load_n_bins=_optional_int(data.get("system_load_n_bins")),
            system_load_bin_edges=_optional_float_array(
                data.get("system_load_bin_edges")
            ),
            pred_n_bins=_optional_int(data.get("pred_n_bins")),
            pred_bin_edges={
                str(key): np.asarray(values, dtype=FLOAT32_DTYPE)
                for key, values in cast(dict, data.get("pred_bin_edges") or {}).items()
            },
            calibration_fingerprint=_optional_str(data.get("calibration_fingerprint")),
            ground_truth_reference=cast(
                dict[str, Any] | None, data.get("ground_truth_reference")
            ),
            tuning=cast(dict[str, Any] | None, data.get("tuning")),
            sparse_strata=validate_sparse_strata(
                str(data.get("sparse_strata", DEFAULT_SPARSE_STRATA))
            ),
        )

    @property
    def bus_ids(self) -> np.ndarray:
        """Compatibility alias for bus-only call sites."""

        return self.entity_ids


def predict_intervals_jointly(
    calibrators: Sequence[KernelIntervalCalibrator],
    predictions: Sequence[PredictionSet | ChannelSet],
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """`predict_intervals` of several calibrators, sharing the kernel work.

    Calibrators fitted on the same calibration scenarios and embeddings (e.g.
    the voltage and line-loading families of one artifact), applied to the
    same test scenarios, need the same distances and, with the same settings,
    the same bandwidths and weights: these are computed once and shared.
    Others are processed separately. Each result, and each calibrator's
    `diagnostics` and `pooled_masks`, equal those of its own
    `predict_intervals` call.

    Args:
        calibrators: The calibrators.
        predictions: One prediction set per calibrator, as for
            `predict_intervals`.

    Returns:
        One `{variable: (lower, upper)}` per calibrator, in order.

    Warns:
        KernelFallbackWarning: For a calibrator whose bounds fall back to the
            stratum quantiles in at least half of the scenarios.
    """

    results = _predict_jointly(calibrators, predictions)
    for calibrator, prediction_set in zip(calibrators, predictions, strict=True):
        _warn_if_fallback(calibrator, prediction_set, stacklevel=2)
    return results


def _predict_jointly(
    calibrators: Sequence[KernelIntervalCalibrator],
    predictions: Sequence[PredictionSet | ChannelSet],
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """`predict_intervals_jointly` without the warning (the CLI notes it instead)."""

    if len(calibrators) != len(predictions):
        raise ValueError("Pass one prediction set per calibrator.")
    units = [
        _Unit.start(calibrator, prediction_set, calibrator.n_eff_target)
        for calibrator, prediction_set in zip(calibrators, predictions, strict=True)
    ]
    _run_units(units)
    results = []
    for unit in units:
        intervals, diagnostics, pooled = unit.finish()
        unit.calibrator._diagnostics = diagnostics
        unit.calibrator._pooled = pooled
        results.append(intervals)
    return results


def _warn_if_fallback(
    calibrator: KernelIntervalCalibrator,
    predictions: PredictionSet | ChannelSet,
    *,
    stacklevel: int,
) -> None:
    """Warn when most scenarios of the last prediction fell back (see the class)."""

    flags = np.asarray(calibrator.diagnostics.get("kernel_fallback", ()), dtype=bool)
    if flags.size == 0 or flags.mean() < FALLBACK_WARNING_SHARE:
        return
    absent = 0
    if calibrator.strat is not None:
        strata = _get_scenario_column(predictions, calibrator.strat)
        absent = int((~np.isin(strata, calibrator.stratification_values)).sum())
    message = (
        f"Kernel bounds of {', '.join(calibrator.channel_names)}: "
        f"{flags.mean():.0%} of the {flags.size:,} "
        f"scenario{'s' if flags.size != 1 else ''} fall back to the "
        "stratum quantiles (those of NaiveIntervalCalibrator) for at least one "
        "value, so the kernel barely changes the bounds."
    )
    if absent:
        message += (
            f" {absent:,} of them are in strata absent from the calibration data."
        )
    if int(flags.sum()) > absent:
        setting, larger = (
            (f"n_eff_target={calibrator.n_eff_target:g}", "a larger n_eff_target")
            if calibrator.bandwidth == BANDWIDTH_N_EFF
            else (f"k_nn={calibrator.k_nn}", "a larger k_nn")
        )
        level = 1.0 - calibrator.alpha / 2.0
        message += (
            f" {'Otherwise, the' if absent else 'The'} calibration scenarios are "
            f"too far from them in the embedding to weigh in at {setting}: their "
            "kernel weights must sum to "
            f"at least {level / (1.0 - level):.0f}, the scenario itself weighing 1. "
            f"Try {larger}, or a scenario vector with fewer dimensions."
        )
    message += " Per scenario: diagnostics['kernel_fallback']."
    warnings.warn(message, KernelFallbackWarning, stacklevel=stacklevel + 1)


def _predict_targets(
    calibrators: Sequence[KernelIntervalCalibrator],
    predictions: Sequence[PredictionSet | ChannelSet],
    targets: Sequence[float],
) -> list[dict[float, tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]]]:
    """Intervals and diagnostics of each calibrator at several `n_eff_target`s.

    The same as fitting with each target and calling `predict_intervals`; the
    distances are computed once for all targets (cross-validated tuning).
    """

    units = [
        [_Unit.start(calibrator, prediction_set, float(target)) for target in targets]
        for calibrator, prediction_set in zip(calibrators, predictions, strict=True)
    ]
    _run_units([unit for row in units for unit in row])
    return [
        {
            float(target): unit.finish()[:2]
            for target, unit in zip(targets, row, strict=True)
        }
        for row in units
    ]


@dataclass
class _Unit:
    """One calibrator applied to one test set at one `n_eff_target`.

    The offsets and diagnostics are filled in place by `_run_units`.
    """

    calibrator: KernelIntervalCalibrator
    predictions: PredictionSet | ChannelSet
    n_eff_target: float
    test_embeddings: np.ndarray
    test_strata: np.ndarray
    prediction_values: dict[str, np.ndarray]
    test_valid: dict[str, np.ndarray | None]
    lower_offsets: dict[str, np.ndarray]
    upper_offsets: dict[str, np.ndarray]
    n_eff: np.ndarray
    bandwidth: np.ndarray
    fallback: np.ndarray

    @classmethod
    def start(
        cls,
        calibrator: KernelIntervalCalibrator,
        predictions: PredictionSet | ChannelSet,
        n_eff_target: float,
    ) -> _Unit:
        _validate_entity_alignment(calibrator.entity_ids, _get_entity_ids(predictions))
        test_embeddings = _require_scenario_embeddings(predictions)
        if test_embeddings.shape[1] != calibrator.calibration_embeddings.shape[1]:
            raise ValueError(
                "Test embedding width "
                f"({test_embeddings.shape[1]}) does not match the calibration bank "
                f"({calibrator.calibration_embeddings.shape[1]}). The same "
                "embedding_pooling must be used for compute and calibrate.",
            )
        n_test = test_embeddings.shape[0]
        test_strata: np.ndarray
        if calibrator.strat is None:
            test_strata = np.zeros(n_test, dtype=np.int64)
        else:
            test_strata = _get_scenario_column(predictions, calibrator.strat).astype(
                np.int64, copy=False
            )
        prediction_values = {
            name: _get_prediction_channel(predictions, name)
            for name in calibrator.channel_names
        }
        fallback = np.zeros(n_test, dtype=bool)
        fallback[~np.isin(test_strata, calibrator.stratification_values)] = True
        return cls(
            calibrator=calibrator,
            predictions=predictions,
            n_eff_target=float(n_eff_target),
            test_embeddings=test_embeddings,
            test_strata=test_strata,
            prediction_values=prediction_values,
            test_valid={
                name: _get_valid_mask(predictions, name)
                for name in calibrator.channel_names
            },
            # Scenarios of a stratum the calibration data lacks keep infinite offsets.
            lower_offsets={
                name: np.full(values.shape, -np.inf, dtype=FLOAT32_DTYPE)
                for name, values in prediction_values.items()
            },
            upper_offsets={
                name: np.full(values.shape, np.inf, dtype=FLOAT32_DTYPE)
                for name, values in prediction_values.items()
            },
            n_eff=np.zeros(n_test, dtype=FLOAT32_DTYPE),
            bandwidth=np.zeros(n_test, dtype=FLOAT32_DTYPE),
            fallback=fallback,
        )

    def shares_kernel_inputs(self, other: _Unit) -> bool:
        """Same calibration scenarios, embedding geometry and test scenarios."""

        mine, theirs = self.calibrator, other.calibrator
        return (
            mine.distance == theirs.distance
            and _same_array(mine.stratification_values, theirs.stratification_values)
            and _same_array(mine.calibration_strata, theirs.calibration_strata)
            and _same_array(mine.embedding_mean, theirs.embedding_mean)
            and _same_array(mine.embedding_scale, theirs.embedding_scale)
            and _same_array(mine.calibration_embeddings, theirs.calibration_embeddings)
            and _same_array(self.test_strata, other.test_strata)
            and _same_array(self.test_embeddings, other.test_embeddings)
        )

    def finish(
        self,
    ) -> tuple[
        dict[str, tuple[np.ndarray, np.ndarray]],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        """`(intervals, diagnostics, pooled masks)` from the filled offsets."""

        calibrator = self.calibrator
        intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pooled_masks: dict[str, np.ndarray] = {}
        for channel_name in calibrator.channel_names:
            prediction = self.prediction_values[channel_name]
            valid_mask = self.test_valid[channel_name]
            lower_offsets = self.lower_offsets[channel_name]
            upper_offsets = self.upper_offsets[channel_name]
            if calibrator.sparse_strata == "pooled":
                pooled_lower, pooled_upper = calibrator._pooled_quantiles_of(
                    channel_name
                )
                lower_offsets, upper_offsets, pooled = pool_sparse_strata(
                    lower_offsets, upper_offsets, pooled_lower, pooled_upper
                )
                pooled_masks[channel_name] = (
                    pooled if valid_mask is None else pooled & valid_mask
                )
            lower = prediction + lower_offsets
            upper = prediction + upper_offsets
            if valid_mask is not None:
                lower = np.where(valid_mask, lower, prediction)
                upper = np.where(valid_mask, upper, prediction)
            intervals[channel_name] = (
                lower.astype(FLOAT32_DTYPE, copy=False),
                upper.astype(FLOAT32_DTYPE, copy=False),
            )
        diagnostics = {
            "n_eff": self.n_eff,
            "kernel_bandwidth_sq": self.bandwidth,
            "kernel_fallback": self.fallback,
        }
        return intervals, diagnostics, pooled_masks


def _same_array(first: np.ndarray, second: np.ndarray) -> bool:
    return first is second or (
        first.shape == second.shape and bool(np.array_equal(first, second))
    )


def _run_units(units: Sequence[_Unit]) -> None:
    """Fill the offsets and diagnostics of every unit, sharing what they share."""

    groups: list[list[_Unit]] = []
    for unit in units:
        for group in groups:
            if group[0].shares_kernel_inputs(unit):
                group.append(unit)
                break
        else:
            groups.append([unit])
    for group in groups:
        _run_group(group)


@dataclass
class _StratumWork:
    """What every block of one stratum needs."""

    index: int
    calibration: np.ndarray
    calibration_sq: np.ndarray


def _run_group(group: Sequence[_Unit]) -> None:
    """Units with the same calibration and test scenarios: one pass over the blocks.

    Distances, bandwidths and weights depend on the scenario only, so each
    block computes them once, per bandwidth setting, for every unit and
    calibrated variable. Blocks of all strata are spread over one thread pool.
    """

    first_unit = group[0]
    first = first_unit.calibrator
    test_embeddings = _prepare_embeddings(
        first_unit.test_embeddings,
        first.embedding_mean,
        first.embedding_scale,
        first.distance,
    )
    test_strata = first_unit.test_strata
    threads = _resolve_worker_count(first.n_jobs, os.cpu_count() or 1)
    executor = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
    try:
        for unit in group:
            unit.calibrator._ensure_banks(executor)
        n_elements = max(
            bank.sorted_residuals.shape[0]
            for unit in group
            for banks in unit.calibrator._banks.values()
            for bank in banks[:1]
        )
        work: list[tuple[_StratumWork, np.ndarray]] = []
        for stratum_index, stratum_value in enumerate(first.stratification_values):
            test_rows = np.flatnonzero(test_strata == stratum_value)
            if test_rows.size == 0:
                continue
            rows = first._banks[first.channel_names[0]][stratum_index].rows
            if rows.size == 0:
                # No calibration data in this stratum: its fallback quantiles
                # are infinite (pooled ones replace them with sparse_strata).
                for unit in group:
                    calibrator = unit.calibrator
                    for name in calibrator.channel_names:
                        unit.lower_offsets[name][test_rows] = (
                            calibrator.fallback_lower_quantiles[name][stratum_index]
                        )
                        unit.upper_offsets[name][test_rows] = (
                            calibrator.fallback_upper_quantiles[name][stratum_index]
                        )
                    unit.fallback[test_rows] = True
                continue
            calibration = first.calibration_embeddings[rows]
            stratum = _StratumWork(
                index=stratum_index,
                calibration=calibration,
                calibration_sq=np.einsum("ij,ij->i", calibration, calibration),
            )
            chunk = _resolve_chunk_size(rows.size, n_elements)
            work += [
                (stratum, test_rows[start : start + chunk])
                for start in range(0, test_rows.size, chunk)
            ]

        def process(item: tuple[_StratumWork, np.ndarray]) -> None:
            _process_block(group, test_embeddings, *item)

        if executor is None or len(work) <= 1:
            for item in work:
                process(item)
        else:
            list(executor.map(process, work))
    finally:
        if executor is not None:
            executor.shutdown()


def _process_block(
    group: Sequence[_Unit],
    test_embeddings: np.ndarray,
    stratum: _StratumWork,
    block_rows: np.ndarray,
) -> None:
    """Bounds of one block of test scenarios, for every unit of a group."""

    distances = _squared_distances(
        test_embeddings[block_rows], stratum.calibration, stratum.calibration_sq
    )
    n_calibration = distances.shape[1]
    # Units with the same bandwidth rule share bandwidths and weights.
    by_rule: dict[tuple[str, float], list[_Unit]] = {}
    for unit in group:
        calibrator = unit.calibrator
        if calibrator.bandwidth == BANDWIDTH_N_EFF:
            # A stratum can be smaller than the requested target, in which case
            # the solve saturates at "all calibration scenarios".
            rule = (BANDWIDTH_N_EFF, min(unit.n_eff_target, float(n_calibration)))
        else:
            rule = (BANDWIDTH_KNN, float(min(calibrator.k_nn, n_calibration) - 1))
        by_rule.setdefault(rule, []).append(unit)
    for (name, parameter), units in by_rule.items():
        solved_n_eff: np.ndarray | None = None
        if name == BANDWIDTH_N_EFF:
            solved, solved_n_eff = _solve_bandwidth_for_n_eff(distances, parameter)
        else:
            neighbor_index = int(parameter)
            solved = np.partition(distances, neighbor_index, axis=1)[:, neighbor_index]
        bandwidth = np.maximum(solved, np.float32(1e-12))
        # Unshifted: the total weight is compared with the test point's own
        # unit weight, so its scale matters (see `_weighted_tail_offsets`).
        weights = np.exp(-distances / bandwidth[:, None]).astype(
            FLOAT32_DTYPE, copy=False
        )
        # n_eff does not depend on the scale of the weights, and is computed
        # on shifted ones: far from every calibration scenario, the squares
        # of these weights underflow in float32.
        if solved_n_eff is None or not np.array_equal(bandwidth, solved):
            solved_n_eff = _effective_sample_size_and_slope(
                distances, bandwidth.astype(np.float64)
            )[0]
        n_eff = solved_n_eff.astype(FLOAT32_DTYPE, copy=False)
        weight_sum = weights.sum(axis=1, dtype=np.float64)
        for unit in units:
            _unit_block(
                unit, stratum.index, block_rows, weights, weight_sum, n_eff, bandwidth
            )


def _unit_block(
    unit: _Unit,
    stratum_index: int,
    block_rows: np.ndarray,
    weights: np.ndarray,
    weight_sum: np.ndarray,
    n_eff: np.ndarray,
    bandwidth: np.ndarray,
) -> None:
    """One unit's bounds and diagnostics for one block, from shared weights."""

    calibrator = unit.calibrator
    level = 1.0 - (calibrator.alpha / 2.0)
    # Weights at least this large in total are needed to reach `level`.
    min_weight_sum = level / (1.0 - level)
    unit.n_eff[block_rows] = n_eff
    unit.bandwidth[block_rows] = bandwidth
    # Too few effective neighbours: the stratum quantile. With too little
    # total weight the level is unreachable for every element, so the
    # quantile search is skipped too (the margin absorbs rounding).
    degenerate = n_eff < calibrator.n_eff_min
    unreachable = weight_sum < min_weight_sum * _UNREACHABLE_MARGIN
    kernel_rows = ~(degenerate | unreachable)
    kernel_weights: np.ndarray | None = None
    kernel_weights64: np.ndarray | None = None
    if kernel_rows.any():
        kernel_weights = weights[kernel_rows]
        kernel_weights64 = kernel_weights.astype(np.float64)
    for name in calibrator.channel_names:
        bank = calibrator._banks[name][stratum_index]
        lower = np.broadcast_to(
            bank.fallback_lower, (len(block_rows), bank.fallback_lower.size)
        ).copy()
        upper = np.broadcast_to(
            bank.fallback_upper, (len(block_rows), bank.fallback_upper.size)
        ).copy()
        reached = np.zeros(lower.shape, dtype=bool)
        if kernel_weights is not None and kernel_weights64 is not None:
            kernel_lower, kernel_upper, kernel_reached = _tail_offsets(
                kernel_weights, kernel_weights64, bank, level
            )
            lower[kernel_rows] = np.where(
                kernel_reached, kernel_lower, bank.fallback_lower
            )
            upper[kernel_rows] = np.where(
                kernel_reached, kernel_upper, bank.fallback_upper
            )
            reached[kernel_rows] = kernel_reached
        # An element with no valid calibration residual in this stratum
        # (e.g. a branch out of service in every calibration scenario)
        # always takes the fallback bound. That is a property of the data,
        # not of the kernel, so it must not flag every test scenario;
        # nor must an element that has no interval in this test scenario
        # (a known set-point, an out-of-service line).
        usable = reached | ~bank.has_data[None, :]
        valid = unit.test_valid[name]
        if valid is not None:
            usable |= ~np.asarray(valid[block_rows], dtype=bool)
        unit.fallback[block_rows] |= degenerate | ~usable.all(axis=1)
        unit.lower_offsets[name][block_rows] = lower
        unit.upper_offsets[name][block_rows] = upper


def _jit() -> ModuleType:
    """The compiled kernels, imported on first use (numba takes a while to load)."""

    from gridfm_trustlayer.conformal import _kernel_jit

    return _kernel_jit


def _squared_distances(
    block: np.ndarray,
    calibration: np.ndarray,
    calibration_sq: np.ndarray,
) -> np.ndarray:
    """Blocked ``||a - b||^2`` via one GEMM, which BLAS runs multi-threaded."""

    block_sq = np.einsum("ij,ij->i", block, block)
    # Apple's Accelerate BLAS (e.g. under numpy 2.2 on macOS) can leave
    # floating-point flags set after a product of finite values, which numpy
    # reports as "divide by zero encountered in matmul". The embeddings are
    # checked to be finite, so the flags mean nothing here.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        distances = calibration @ block.T
    distances *= -2.0
    distances += calibration_sq[:, None]
    distances = distances.T
    distances += block_sq[:, None]
    np.maximum(distances, 0.0, out=distances)
    return np.ascontiguousarray(distances, dtype=FLOAT32_DTYPE)


def _prepare_embeddings(
    embeddings: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    distance: str,
) -> np.ndarray:
    """Standardize embeddings, then apply the geometry the distance implies.

    Cosine distance is squared Euclidean distance on unit-norm vectors
    (`d^2 = 2 - 2*cos`), so it is realized by projecting onto the unit sphere
    here and leaving every downstream computation untouched. Note that this
    happens *after* centering, so what survives is the direction in which a
    scenario deviates from the mean scenario, not the size of that deviation.
    """

    standardized = (
        (np.asarray(embeddings, dtype=FLOAT32_DTYPE) - mean) / scale
    ).astype(FLOAT32_DTYPE, copy=False)
    if distance == DISTANCE_COSINE:
        norms = np.linalg.norm(standardized, axis=1, keepdims=True)
        standardized = standardized / np.maximum(norms, _MIN_EMBEDDING_NORM)
    return np.ascontiguousarray(standardized, dtype=FLOAT32_DTYPE)


def _solve_bandwidth_for_n_eff(
    distances: np.ndarray,
    target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-row bandwidth whose kernel weights hit `target` n_eff.

    Also returns the `n_eff` of each row at that bandwidth, computed on the
    shifted weights of `_effective_sample_size_and_slope`, which do not
    underflow.

    `n_eff` increases monotonically from 1 (all weight on the nearest
    calibration scenario) to the stratum size (uniform weights). It behaves
    close to a power of the bandwidth, so Newton steps on `log n_eff` against
    `log h` converge in a few iterations; a bracket around the root is kept,
    and a step that would leave it is replaced by a bisection step. Where
    `n_eff` is flat at one end of the bracket and steep at the other, Newton
    steps can alternate between the two ends; a row still unconverged after
    `_N_EFF_SOLVE_NEWTON_ITER` iterations continues by bisection only. Each
    row is solved on its own (compiled, see `_kernel_jit.solve_bandwidths`),
    so its bandwidth does not depend on the other rows of the block.
    """

    distances = np.ascontiguousarray(distances, dtype=FLOAT32_DTYPE)
    n_rows = distances.shape[0]
    minimum = np.empty(n_rows, dtype=FLOAT32_DTYPE)
    nearest: np.ndarray = np.empty(n_rows, dtype=FLOAT32_DTYPE)
    farthest: np.ndarray = np.empty(n_rows, dtype=FLOAT32_DTYPE)
    bandwidth = np.zeros(n_rows, dtype=np.float64)
    reached = np.zeros(n_rows, dtype=np.float64)
    if n_rows == 0 or distances.shape[1] == 0:
        return bandwidth.astype(FLOAT32_DTYPE), reached
    jit = _jit()
    jit.row_extremes(distances, minimum, nearest, farthest)
    nearest = np.where(np.isfinite(nearest), nearest, 1.0)
    farthest = np.maximum(farthest, nearest)
    # The bracket is computed in float32, as by every earlier version.
    log_lower = np.log(nearest / _N_EFF_BRACKET_SPAN).astype(np.float64)
    log_upper = np.log(farthest * _N_EFF_BRACKET_SPAN).astype(np.float64)
    jit.solve_bandwidths(
        distances,
        minimum,
        log_lower,
        log_upper,
        float(target),
        _N_EFF_SOLVE_RTOL,
        _N_EFF_SOLVE_NEWTON_ITER,
        _N_EFF_SOLVE_MAX_ITER,
        bandwidth,
        reached,
    )
    return bandwidth.astype(FLOAT32_DTYPE, copy=False), reached


def _effective_sample_size_and_slope(
    distances: np.ndarray,
    bandwidth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`n_eff` of the kernel weights, and the derivative of `log n_eff` in `log h`.

    With `a_j = d_j / h`, the weights are `exp(-a_j)` and
    `d log n_eff / d log h = 2 sum(w a) / sum(w) - 2 sum(w^2 a) / sum(w^2)`.
    Both are invariant to a common rescaling of the weights, so the exponent
    is shifted by each row's minimum distance, which avoids underflow at small
    bandwidths without changing the result.
    """

    scaled = (distances - distances.min(axis=1, keepdims=True)) / bandwidth[:, None]
    weights = np.exp(-scaled)
    squared = weights * weights
    weight_sum = weights.sum(axis=1)
    squared_sum = squared.sum(axis=1)
    n_eff = (weight_sum**2) / squared_sum
    slope = 2.0 * np.einsum("ij,ij->i", weights, scaled) / weight_sum - 2.0 * (
        np.einsum("ij,ij->i", squared, scaled) / squared_sum
    )
    return n_eff, slope


def _resolve_chunk_size(n_calibration: int, n_elements: int) -> int:
    """Test rows per block: distances, weights and temporaries, per-element totals."""

    per_row_bytes = max(n_calibration * 4 * 6 + max(n_elements, 1) * 8 * 4, 1)
    return int(max(1, min(_MAX_BLOCK_ROWS, _TARGET_BLOCK_BYTES // per_row_bytes)))


def _weighted_tail_offsets(
    weights: np.ndarray,
    bank: _StratumBank,
    level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted conformal quantiles of each test row, for every element.

    With `p` the kernel weights of the valid calibration residuals in sorted
    order and `T` their total, the upper bound is the first sorted residual
    whose cumulative weight reaches `level * (T + 1)` (the `+1` is the unseen
    test residual), and the lower bound mirrors it from the top. Both depend
    only on the weight in the tails: the upper bound is where the weight above
    it stops exceeding `T - level * (T + 1)`. So `T` comes from one matrix
    product, and each bound is found by accumulating weight from its end of
    the sorted order until it exceeds that slack (compiled, see
    `_kernel_jit.tail_bounds`). The result equals the full scan over all
    residuals.

    Returns:
        `(lower, upper, reached)` of shape `(n_rows, n_elements)`; `reached` is
        False where the level cannot be attained, and the bounds there are
        meaningless.
    """

    # Weights are gathered in float32 and summed in float64: the conversion is
    # exact, so the sums equal a float64 computation at half the memory traffic.
    weights = np.ascontiguousarray(weights, dtype=FLOAT32_DTYPE)
    return _tail_offsets(weights, weights.astype(np.float64), bank, level)


def _tail_offsets(
    weights: np.ndarray,
    weights64: np.ndarray,
    bank: _StratumBank,
    level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`_weighted_tail_offsets` with the float64 copy of the weights given."""

    # Finite weights in [0, 1]: any floating-point flag comes from the BLAS
    # (see `_squared_distances`).
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = weights64 @ bank.valid
    slack = total - level * (total + 1.0)
    reached = slack >= 0.0
    n_elements = bank.sorted_residuals.shape[0]
    lower = np.zeros((weights.shape[0], n_elements), dtype=FLOAT32_DTYPE)
    upper = np.zeros_like(lower)
    if reached.any():
        _jit().tail_bounds(
            weights,
            np.ascontiguousarray(slack),
            np.ascontiguousarray(reached),
            bank.order,
            bank.sorted_residuals,
            lower,
            upper,
        )
    return lower, upper, reached


def _resolve_worker_count(n_jobs: int | None, n_blocks: int) -> int:
    if n_blocks <= 1:
        return 1
    if n_jobs is not None and n_jobs > 0:
        return min(n_jobs, n_blocks)
    return min(max(1, (os.cpu_count() or 1)), n_blocks)


def _require_scenario_embeddings(
    predictions: PredictionSet | ChannelSet,
) -> np.ndarray:
    embeddings = getattr(predictions, "scenario_embeddings", None)
    if embeddings is None:
        raise ValueError(
            "cp_type 'kernel' requires scenario embeddings on the prediction set. "
            "Provide inputs.embeddings in the trustlayer config.",
        )
    return _finite_embeddings(np.asarray(embeddings, dtype=FLOAT32_DTYPE), "test")


def _finite_embeddings(embeddings: np.ndarray, which: str) -> np.ndarray:
    """`embeddings`, refused if any value is NaN or infinite.

    One such value makes every distance to it NaN; in the calibration bank, it
    would make every test scenario fall back without a word.
    """

    broken = ~np.isfinite(embeddings).all(axis=1)
    if broken.any():
        raise ValueError(
            f"The {which} embeddings have NaN or infinite values in "
            f"{int(broken.sum()):,} scenario(s) (rows {np.flatnonzero(broken)[:5].tolist()})."
        )
    return embeddings


def _optional_float_array(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=FLOAT32_DTYPE)


def _optional_str(value: object | None) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(cast(Any, value))
