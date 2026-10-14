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
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    pool_sparse_strata,
    validate_sparse_strata,
)
from gridfm_trustlayer.io.ground_truth import GroundTruthSet
from gridfm_trustlayer.io.predictions import PredictionSet

FLOAT32_DTYPE = np.dtype(np.float32)
INT32_DTYPE = np.dtype(np.int32)

ARTIFACT_VERSION = "trustlayer_kernel_interval_v1"

DEFAULT_K_NN = 20
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
# move a weighted quantile to a neighbouring residual.
_N_EFF_SOLVE_MAX_ITER = 60
_N_EFF_SOLVE_RTOL = 1e-6
_N_EFF_BRACKET_SPAN = 1e3

# Target working-set size for one distance/quantile block, in bytes. Chunk
# sizes are derived from this so memory stays bounded regardless of how many
# calibration scenarios or grid elements a run carries.
_TARGET_BLOCK_BYTES = 48 * 1024 * 1024
# Weighted quantile: the first scan from each end covers this many times the
# tail share `1 - level` of the sorted positions (at least `_MIN_TAIL_WINDOW`).
# `_TAIL_SCAN_ARRAYS` counts the (rows x window x elements) float64 arrays a
# scan holds at once, for the memory budget.
_TAIL_WINDOW_FACTOR = 3.0
_MIN_TAIL_WINDOW = 32
_TAIL_SCAN_ARRAYS = 2
# Positions summed together in the first level of the two-level tail scan.
_SCAN_BLOCK = 16
# Skip the quantile search only when the weight sum is clearly below the
# minimum, so float rounding can never skip a reachable case.
_UNREACHABLE_MARGIN = 1.0 - 1e-6

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
    sorted_residuals: np.ndarray
    sort_order: np.ndarray
    sorted_valid: np.ndarray
    # Validity in calibration order, float64: kernel weights @ valid gives the
    # total weight per element in one matrix product.
    valid: np.ndarray
    # `sort_order` with invalid entries pointing at index `n_calibration`, a
    # zero-weight column appended to the weights before gathering.
    sort_order_masked: np.ndarray
    fallback_lower: np.ndarray
    fallback_upper: np.ndarray
    # Per element: whether the stratum holds at least one valid residual.
    has_data: np.ndarray


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
        embedding_pooling: str,
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
            embedding_pooling: Name of the pooling that produced `embeddings`,
                stored so that test embeddings can be pooled the same way.
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
        embeddings = np.ascontiguousarray(embeddings, dtype=FLOAT32_DTYPE)
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
            valid_mask = _combine_valid_masks(
                _get_valid_mask(predictions, channel_name),
                _get_valid_mask(ground_truth, channel_name),
            )
            if valid_mask is None:
                valid_mask = np.ones(channel_residuals.shape, dtype=bool)
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

        _validate_entity_alignment(self.entity_ids, _get_entity_ids(predictions))
        test_embeddings = _require_scenario_embeddings(predictions)
        if test_embeddings.shape[1] != self.calibration_embeddings.shape[1]:
            raise ValueError(
                "Test embedding width "
                f"({test_embeddings.shape[1]}) does not match the calibration bank "
                f"({self.calibration_embeddings.shape[1]}). The same embedding_pooling "
                "must be used for compute and calibrate.",
            )
        standardized_test = _prepare_embeddings(
            test_embeddings,
            self.embedding_mean,
            self.embedding_scale,
            self.distance,
        )

        n_test = standardized_test.shape[0]
        test_strata: np.ndarray
        if self.strat is None:
            test_strata = np.zeros(n_test, dtype=np.int64)
        else:
            test_strata = _get_scenario_column(predictions, self.strat).astype(
                np.int64, copy=False
            )

        self._ensure_banks()
        self._pooled = {}
        n_eff_out = np.zeros(n_test, dtype=FLOAT32_DTYPE)
        bandwidth_out = np.zeros(n_test, dtype=FLOAT32_DTYPE)
        fallback_out = np.zeros(n_test, dtype=bool)
        prediction_values = {
            name: _get_prediction_channel(predictions, name)
            for name in self.channel_names
        }
        # Scenarios of a stratum the calibration data lacks keep infinite offsets.
        lower_offsets = {
            name: np.full(values.shape, -np.inf, dtype=FLOAT32_DTYPE)
            for name, values in prediction_values.items()
        }
        upper_offsets = {
            name: np.full(values.shape, np.inf, dtype=FLOAT32_DTYPE)
            for name, values in prediction_values.items()
        }
        fallback_out[~np.isin(test_strata, self.stratification_values)] = True

        for stratum_index, stratum_value in enumerate(self.stratification_values):
            test_rows = np.flatnonzero(test_strata == stratum_value)
            if test_rows.size == 0:
                continue
            banks = {
                name: self._banks[name][stratum_index] for name in self.channel_names
            }
            if next(iter(banks.values())).rows.size == 0:
                # No calibration data in this stratum: its fallback quantiles
                # are infinite (pooled ones replace them with sparse_strata).
                for name in self.channel_names:
                    lower_offsets[name][test_rows] = self.fallback_lower_quantiles[
                        name
                    ][stratum_index]
                    upper_offsets[name][test_rows] = self.fallback_upper_quantiles[
                        name
                    ][stratum_index]
                fallback_out[test_rows] = True
                continue
            self._apply_stratum(
                banks=banks,
                test_rows=test_rows,
                test_embeddings=standardized_test,
                lower_offsets=lower_offsets,
                upper_offsets=upper_offsets,
                n_eff_out=n_eff_out,
                bandwidth_out=bandwidth_out,
                fallback_out=fallback_out,
            )

        intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for channel_name in self.channel_names:
            prediction = prediction_values[channel_name]
            valid_mask = _get_valid_mask(predictions, channel_name)
            if self.sparse_strata == "pooled":
                pooled_lower, pooled_upper = self._pooled_quantiles_of(channel_name)
                (
                    lower_offsets[channel_name],
                    upper_offsets[channel_name],
                    pooled,
                ) = pool_sparse_strata(
                    lower_offsets[channel_name],
                    upper_offsets[channel_name],
                    pooled_lower,
                    pooled_upper,
                )
                self._pooled[channel_name] = (
                    pooled if valid_mask is None else pooled & valid_mask
                )
            lower = prediction + lower_offsets[channel_name]
            upper = prediction + upper_offsets[channel_name]
            if valid_mask is not None:
                lower = np.where(valid_mask, lower, prediction)
                upper = np.where(valid_mask, upper, prediction)
            intervals[channel_name] = (
                lower.astype(FLOAT32_DTYPE, copy=False),
                upper.astype(FLOAT32_DTYPE, copy=False),
            )

        self._diagnostics = {
            "n_eff": n_eff_out,
            "kernel_bandwidth_sq": bandwidth_out,
            "kernel_fallback": fallback_out,
        }
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

    def _ensure_banks(self) -> None:
        if self._banks:
            return
        banks: dict[str, list[_StratumBank]] = {}
        for channel_name in self.channel_names:
            channel_residuals = self.residuals[channel_name]
            channel_valid = self.valid_masks[channel_name]
            per_stratum: list[_StratumBank] = []
            for index, stratum_value in enumerate(self.stratification_values):
                rows = np.flatnonzero(self.calibration_strata == stratum_value)
                stratum_residuals = channel_residuals[rows]
                stratum_valid = channel_valid[rows]
                if rows.size == 0:
                    order = np.zeros((0, channel_residuals.shape[1]), dtype=INT32_DTYPE)
                    sorted_residuals = np.zeros(
                        (0, channel_residuals.shape[1]), dtype=FLOAT32_DTYPE
                    )
                    sorted_valid = np.zeros(
                        (0, channel_residuals.shape[1]), dtype=FLOAT32_DTYPE
                    )
                else:
                    order = np.argsort(stratum_residuals, axis=0, kind="stable").astype(
                        INT32_DTYPE, copy=False
                    )
                    sorted_residuals = np.take_along_axis(
                        stratum_residuals, order, axis=0
                    ).astype(FLOAT32_DTYPE, copy=False)
                    sorted_valid = np.take_along_axis(
                        stratum_valid, order, axis=0
                    ).astype(FLOAT32_DTYPE, copy=False)
                per_stratum.append(
                    _StratumBank(
                        rows=rows,
                        sorted_residuals=np.ascontiguousarray(sorted_residuals),
                        sort_order=np.ascontiguousarray(order),
                        sorted_valid=np.ascontiguousarray(sorted_valid),
                        valid=np.ascontiguousarray(stratum_valid, dtype=np.float64),
                        sort_order_masked=np.ascontiguousarray(
                            np.where(sorted_valid > 0, order, rows.size),
                            dtype=INT32_DTYPE,
                        ),
                        has_data=np.asarray(
                            np.asarray(stratum_valid, dtype=bool).any(axis=0)
                        ),
                        fallback_lower=self.fallback_lower_quantiles[channel_name][
                            index
                        ],
                        fallback_upper=self.fallback_upper_quantiles[channel_name][
                            index
                        ],
                    )
                )
            banks[channel_name] = per_stratum
        self._banks = banks

    def _apply_stratum(
        self,
        *,
        banks: dict[str, _StratumBank],
        test_rows: np.ndarray,
        test_embeddings: np.ndarray,
        lower_offsets: dict[str, np.ndarray],
        upper_offsets: dict[str, np.ndarray],
        n_eff_out: np.ndarray,
        bandwidth_out: np.ndarray,
        fallback_out: np.ndarray,
    ) -> None:
        """Kernel-weighted bounds of the test scenarios of one stratum.

        Distances, bandwidth and weights depend on the scenario only, so they
        are computed once and shared by every calibrated variable.
        """

        rows = next(iter(banks.values())).rows
        calibration = self.calibration_embeddings[rows]
        calibration_sq = np.einsum("ij,ij->i", calibration, calibration)
        n_calibration = calibration.shape[0]
        n_elements = max(bank.sorted_residuals.shape[1] for bank in banks.values())
        level = 1.0 - (self.alpha / 2.0)
        neighbor_index = min(self.k_nn, n_calibration) - 1
        # A stratum can be smaller than the requested target, in which case the
        # solve saturates at "all calibration scenarios" rather than failing.
        n_eff_target = min(self.n_eff_target, float(n_calibration))
        # Weights at least this large in total are needed to reach `level`.
        min_weight_sum = level / (1.0 - level)

        chunk = _resolve_chunk_size(
            n_calibration, n_elements, _initial_tail_window(n_calibration, level)
        )
        blocks = [
            test_rows[start : start + chunk]
            for start in range(0, test_rows.size, chunk)
        ]

        def _process(block_rows: np.ndarray) -> None:
            block = test_embeddings[block_rows]
            distances = _squared_distances(block, calibration, calibration_sq)
            if self.bandwidth == BANDWIDTH_N_EFF:
                bandwidth = _solve_bandwidth_for_n_eff(distances, n_eff_target)
            else:
                bandwidth = np.partition(distances, neighbor_index, axis=1)[
                    :, neighbor_index
                ]
            bandwidth = np.maximum(bandwidth, np.float32(1e-12))
            weights = np.exp(-distances / bandwidth[:, None]).astype(
                FLOAT32_DTYPE, copy=False
            )

            weight_sum = weights.sum(axis=1)
            weight_sq_sum = np.einsum("ij,ij->i", weights, weights)
            n_eff = np.where(
                weight_sq_sum > 0.0, (weight_sum**2) / weight_sq_sum, 0.0
            ).astype(FLOAT32_DTYPE, copy=False)
            n_eff_out[block_rows] = n_eff
            bandwidth_out[block_rows] = bandwidth
            # Too few effective neighbours: the stratum quantile. With too little
            # total weight the level is unreachable for every element, so the
            # quantile search is skipped too (the margin absorbs rounding).
            degenerate = n_eff < self.n_eff_min
            unreachable = (
                weights.sum(axis=1, dtype=np.float64)
                < min_weight_sum * _UNREACHABLE_MARGIN
            )
            kernel_rows = ~(degenerate | unreachable)

            for name, bank in banks.items():
                lower = np.broadcast_to(
                    bank.fallback_lower, (len(block_rows), bank.fallback_lower.size)
                ).copy()
                upper = np.broadcast_to(
                    bank.fallback_upper, (len(block_rows), bank.fallback_upper.size)
                ).copy()
                reached = np.zeros(lower.shape, dtype=bool)
                if kernel_rows.any():
                    kernel_lower, kernel_upper, kernel_reached = _weighted_tail_offsets(
                        weights[kernel_rows], bank, level
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
                # not of the kernel, so it must not flag every test scenario.
                usable = reached | ~bank.has_data[None, :]
                fallback_out[block_rows] |= degenerate | ~usable.all(axis=1)
                lower_offsets[name][block_rows] = lower
                upper_offsets[name][block_rows] = upper

        workers = _resolve_worker_count(self.n_jobs, len(blocks))
        if workers <= 1:
            for block_rows in blocks:
                _process(block_rows)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_process, blocks))

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
            tuning=cast(dict[str, Any] | None, data.get("tuning")),
            sparse_strata=validate_sparse_strata(
                str(data.get("sparse_strata", DEFAULT_SPARSE_STRATA))
            ),
        )

    @property
    def bus_ids(self) -> np.ndarray:
        """Compatibility alias for bus-only call sites."""

        return self.entity_ids


def _squared_distances(
    block: np.ndarray,
    calibration: np.ndarray,
    calibration_sq: np.ndarray,
) -> np.ndarray:
    """Blocked ``||a - b||^2`` via one GEMM, which BLAS runs multi-threaded."""

    block_sq = np.einsum("ij,ij->i", block, block)
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
) -> np.ndarray:
    """Return the per-row bandwidth whose kernel weights hit `target` n_eff.

    `n_eff` increases monotonically from 1 (all weight on the nearest
    calibration scenario) to the stratum size (uniform weights). It behaves
    close to a power of the bandwidth, so Newton steps on `log n_eff` against
    `log h` converge in a few iterations; a bracket around the root is kept,
    and a step that would leave it is replaced by a bisection step. The rows of
    a block are solved together so the work stays vectorized, but each row
    stops as soon as it converges, so its bandwidth does not depend on the
    other rows of the block.
    """

    positive = np.where(distances > 0.0, distances, np.inf)
    nearest = positive.min(axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, 1.0)
    farthest = np.maximum(distances.max(axis=1), nearest)

    log_lower = np.log(nearest / _N_EFF_BRACKET_SPAN).astype(np.float64)
    log_upper = np.log(farthest * _N_EFF_BRACKET_SPAN).astype(np.float64)
    log_target = np.log(target)
    log_bandwidth = 0.5 * (log_lower + log_upper)
    bandwidth = np.exp(log_bandwidth)
    active = np.arange(distances.shape[0])
    step = log_bandwidth.copy()
    for _ in range(_N_EFF_SOLVE_MAX_ITER):
        point = step[active]
        bandwidth[active] = np.exp(point)
        n_eff, slope = _effective_sample_size_and_slope(
            distances[active], np.exp(point)
        )
        converged = np.abs(n_eff - target) <= _N_EFF_SOLVE_RTOL * target
        below_target = n_eff < target
        lower = np.where(below_target, point, log_lower[active])
        upper = np.where(below_target, log_upper[active], point)
        log_lower[active] = lower
        log_upper[active] = upper
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = point - (np.log(n_eff) - log_target) / slope
        inside = np.isfinite(newton) & (newton > lower) & (newton < upper)
        step[active] = np.where(inside, newton, 0.5 * (lower + upper))
        active = active[~converged]
        if active.size == 0:
            break
    return bandwidth.astype(FLOAT32_DTYPE, copy=False)


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


def _resolve_chunk_size(n_calibration: int, n_elements: int, window: int) -> int:
    """Test rows per block: distances and weights, plus the tail scans."""

    per_row_bytes = max(
        n_calibration * 4 * 3,
        window * max(n_elements, 1) * 8 * _TAIL_SCAN_ARRAYS,
        1,
    )
    return int(max(1, min(4096, _TARGET_BLOCK_BYTES // per_row_bytes)))


def _first_crossing(
    weights: np.ndarray,
    positions: np.ndarray,
    slack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First scan position at which the accumulated weight exceeds `slack`.

    `weights` is `(n_rows, n_calibration + 1)` with a final zero column,
    `positions` is `(window, n_elements)` calibration indices in scan order and
    `slack` is `(n_rows, n_elements)`. Returns, per row and element, whether
    the weight of the first `k + 1` positions exceeds the slack for some `k`,
    and the first such `k`.

    Two levels: sums over blocks of `_SCAN_BLOCK` positions locate the block
    of the crossing, and a running sum inside that block finds the position.
    Both levels add the float64 values in the same sequential order, so the
    block that crosses always contains the crossing.
    """

    n_rows = weights.shape[0]
    window, n_elements = positions.shape
    n_blocks = -(-window // _SCAN_BLOCK)
    padded = np.full(
        (n_blocks * _SCAN_BLOCK, n_elements), weights.shape[1] - 1, dtype=INT32_DTYPE
    )
    padded[:window] = positions
    gathered = weights[:, padded].reshape(n_rows, n_blocks, _SCAN_BLOCK, n_elements)
    block_sums = gathered[:, :, 0, :].astype(np.float64)
    for offset in range(1, _SCAN_BLOCK):
        block_sums += gathered[:, :, offset, :]
    cumulative = np.cumsum(block_sums, axis=1)
    crossed = cumulative > slack[:, None, :]
    found = np.asarray(crossed.any(axis=1))
    block = np.argmax(crossed, axis=1)
    before = np.where(
        block > 0,
        np.take_along_axis(cumulative, np.maximum(block - 1, 0)[:, None, :], axis=1)[
            :, 0
        ],
        0.0,
    )
    inside = np.take_along_axis(gathered, block[:, None, None, :], axis=1)[:, 0]
    running = before[:, None, :] + np.cumsum(inside, axis=1, dtype=np.float64)
    offset_in_block = np.argmax(running > slack[:, None, :], axis=1)
    return found, block * _SCAN_BLOCK + offset_in_block


def _initial_tail_window(n_calibration: int, level: float) -> int:
    """Sorted positions scanned first from each end for the weighted quantile.

    The bound sits where the tail holds a share `1 - level` of the weight; a few
    times that share of positions settles almost every case, and the rest are
    scanned again with a doubled window.
    """

    window = int(np.ceil(_TAIL_WINDOW_FACTOR * (1.0 - level) * n_calibration))
    return max(1, min(n_calibration, max(_MIN_TAIL_WINDOW, window)))


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
    product and only the ends of the sorted order are scanned, widening the
    scan where it does not settle the bound. The result equals the full scan
    over all residuals.

    Returns:
        `(lower, upper, reached)` of shape `(n_rows, n_elements)`; `reached` is
        False where the level cannot be attained, and the bounds there are
        meaningless.
    """

    # Weights are gathered in float32 and summed in float64: the conversion is
    # exact, so the sums equal a float64 computation at half the memory traffic.
    weights = np.asarray(weights, dtype=FLOAT32_DTYPE)
    n_rows = weights.shape[0]
    n_calibration, n_elements = bank.sorted_residuals.shape
    total = weights.astype(np.float64) @ bank.valid
    slack = total - level * (total + 1.0)
    reached = slack >= 0.0

    upper_index = np.zeros((n_rows, n_elements), dtype=np.int64)
    lower_index = np.zeros((n_rows, n_elements), dtype=np.int64)
    pending = reached.copy()
    window = _initial_tail_window(n_calibration, level)
    while True:
        pending_rows = np.flatnonzero(pending.any(axis=1))
        if pending_rows.size == 0:
            break
        window = min(window, n_calibration)
        top = bank.sort_order_masked[n_calibration - window :][::-1]
        bottom = bank.sort_order_masked[:window]
        rows_per_step = max(
            1,
            _TARGET_BLOCK_BYTES // (window * n_elements * 8 * _TAIL_SCAN_ARRAYS),
        )
        for start in range(0, pending_rows.size, rows_per_step):
            rows = pending_rows[start : start + rows_per_step]
            # A zero column for the invalid entries, which index it.
            row_weights = np.concatenate(
                [weights[rows], np.zeros((rows.size, 1), dtype=FLOAT32_DTYPE)], axis=1
            )
            row_slack = slack[rows]
            # First k at which the weight of the k + 1 largest (smallest)
            # residuals exceeds the slack.
            above_found, above_k = _first_crossing(row_weights, top, row_slack)
            below_found, below_k = _first_crossing(row_weights, bottom, row_slack)
            settle = pending[rows] & above_found & below_found
            upper_index[rows] = np.where(
                settle, n_calibration - 1 - above_k, upper_index[rows]
            )
            lower_index[rows] = np.where(settle, below_k, lower_index[rows])
            pending[rows] &= ~(above_found & below_found)
        if window == n_calibration:
            # Over all residuals the weight is T > slack, so every reachable
            # bound settles at the latest here.
            break
        window *= 2

    element = np.arange(n_elements)
    upper = bank.sorted_residuals[upper_index, element]
    lower = bank.sorted_residuals[lower_index, element]
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
    return np.asarray(embeddings, dtype=FLOAT32_DTYPE)


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
