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
    ChannelSet,
    _combine_valid_masks,
    _get_entity_ids,
    _get_entity_name,
    _get_ground_truth_channel,
    _get_prediction_channel,
    _get_scenario_column,
    _get_valid_mask,
    _masked_column_quantile,
    _validate_alpha,
    _validate_entity_alignment,
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
DEFAULT_BANDWIDTH = BANDWIDTH_KNN

# When `bandwidth: n_eff` is requested without an explicit target, aim for this
# many effective calibration scenarios per unit of alpha. A (1 - alpha/2)
# quantile puts roughly n_eff * alpha/2 effective observations in the tail, so
# n_eff = 20/alpha leaves about ten of them -- enough for a stable tail estimate.
# At alpha = 0.1 this resolves to 200.
DEFAULT_N_EFF_TARGET_SCALE = 20.0

# Bisection settings for the n_eff bandwidth solve. n_eff is monotonically
# increasing in the bandwidth, so a log-space bisection converges quickly.
_N_EFF_SOLVE_MAX_ITER = 60
_N_EFF_SOLVE_RTOL = 1e-3
_N_EFF_BRACKET_SPAN = 1e3

# Target working-set size for one distance/quantile block, in bytes. Chunk
# sizes are derived from this so memory stays bounded regardless of how many
# calibration scenarios or grid elements a run carries.
_TARGET_BLOCK_BYTES = 48 * 1024 * 1024

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
    fallback_lower: np.ndarray
    fallback_upper: np.ndarray


@dataclass
class KernelIntervalCalibrator:
    """Kernel-weighted per-entity conformal intervals localized in embedding space."""

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
    calibration_fraction: float | None = None
    random_seed: int | None = None
    split_pool_scenario_ids: np.ndarray | None = None
    calibration_scenario_ids: np.ndarray | None = None
    application_scenario_ids: np.ndarray | None = None
    n_jobs: int | None = None
    _banks: dict[str, list[_StratumBank]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _diagnostics: dict[str, np.ndarray] = field(
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
    ) -> KernelIntervalCalibrator:
        """Build the calibration bank used for kernel-weighted interval estimation."""

        _validate_alpha(alpha)
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
            DEFAULT_N_EFF_TARGET_SCALE / alpha
            if n_eff_target is None
            else float(n_eff_target)
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
        )

    # -------------------------------------------------------------- predict

    def predict_intervals(
        self,
        predictions: PredictionSet | ChannelSet,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return kernel-weighted lower/upper interval arrays for each channel."""

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
        if self.strat is None:
            test_strata = np.zeros(n_test, dtype=np.int64)
        else:
            test_strata = _get_scenario_column(predictions, self.strat).astype(
                np.int64, copy=False
            )

        self._ensure_banks()
        n_eff_out = np.zeros(n_test, dtype=FLOAT32_DTYPE)
        bandwidth_out = np.zeros(n_test, dtype=FLOAT32_DTYPE)
        fallback_out = np.zeros(n_test, dtype=bool)

        intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for channel_name in self.channel_names:
            prediction_values = _get_prediction_channel(predictions, channel_name)
            n_elements = prediction_values.shape[1]
            lower_offsets = np.zeros((n_test, n_elements), dtype=FLOAT32_DTYPE)
            upper_offsets = np.zeros((n_test, n_elements), dtype=FLOAT32_DTYPE)

            for stratum_index, stratum_value in enumerate(self.stratification_values):
                test_rows = np.flatnonzero(test_strata == stratum_value)
                if test_rows.size == 0:
                    continue
                bank = self._banks[channel_name][stratum_index]
                if bank.rows.size == 0:
                    # No calibration data in this stratum: fall back to the
                    # pooled quantiles rather than emitting a zero-width band.
                    lower_offsets[test_rows] = self.fallback_lower_quantiles[
                        channel_name
                    ][stratum_index]
                    upper_offsets[test_rows] = self.fallback_upper_quantiles[
                        channel_name
                    ][stratum_index]
                    fallback_out[test_rows] = True
                    continue
                self._apply_stratum(
                    channel_name=channel_name,
                    bank=bank,
                    stratum_index=stratum_index,
                    test_rows=test_rows,
                    test_embeddings=standardized_test,
                    lower_offsets=lower_offsets,
                    upper_offsets=upper_offsets,
                    n_eff_out=n_eff_out,
                    bandwidth_out=bandwidth_out,
                    fallback_out=fallback_out,
                )

            lower = prediction_values + lower_offsets
            upper = prediction_values + upper_offsets
            valid_mask = _get_valid_mask(predictions, channel_name)
            if valid_mask is not None:
                lower = np.where(valid_mask, lower, prediction_values)
                upper = np.where(valid_mask, upper, prediction_values)
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
        """Per-test-scenario KCP diagnostics from the last `predict_intervals` call."""

        return self._diagnostics

    # ------------------------------------------------------------ internals

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
        channel_name: str,
        bank: _StratumBank,
        stratum_index: int,
        test_rows: np.ndarray,
        test_embeddings: np.ndarray,
        lower_offsets: np.ndarray,
        upper_offsets: np.ndarray,
        n_eff_out: np.ndarray,
        bandwidth_out: np.ndarray,
        fallback_out: np.ndarray,
    ) -> None:
        calibration = self.calibration_embeddings[bank.rows]
        calibration_sq = np.einsum("ij,ij->i", calibration, calibration)
        n_calibration = calibration.shape[0]
        n_elements = bank.sorted_residuals.shape[1]
        level = 1.0 - (self.alpha / 2.0)
        neighbor_index = min(self.k_nn, n_calibration) - 1
        # A stratum can be smaller than the requested target, in which case the
        # solve saturates at "all calibration scenarios" rather than failing.
        n_eff_target = min(self.n_eff_target, float(n_calibration))

        chunk = _resolve_chunk_size(n_calibration, n_elements)
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
            degenerate = n_eff < self.n_eff_min

            n_eff_out[block_rows] = n_eff
            bandwidth_out[block_rows] = bandwidth
            fallback_out[block_rows] |= degenerate

            if np.all(degenerate):
                lower_offsets[block_rows] = bank.fallback_lower
                upper_offsets[block_rows] = bank.fallback_upper
                return

            # Kernel weights permuted into per-element residual-sorted order,
            # then masked so invalid calibration entries carry no mass.
            permuted = weights[:, bank.sort_order] * bank.sorted_valid
            cumulative = np.cumsum(permuted, axis=1)
            total = cumulative[:, -1, :]
            threshold = (level * (total + 1.0)).astype(FLOAT32_DTYPE, copy=False)

            upper_reached = total >= threshold
            upper_index = np.argmax(cumulative >= threshold[:, None, :], axis=1)

            # Suffix sums give the mirrored tail: the atom now sits at -inf,
            # which is the conservative direction for the lower bound.
            suffix = total[:, None, :] - cumulative + permuted
            lower_count = (suffix >= threshold[:, None, :]).sum(axis=1)
            lower_reached = lower_count > 0
            lower_index = np.clip(lower_count - 1, 0, n_calibration - 1)

            element_index = np.arange(n_elements)
            block_upper = bank.sorted_residuals[upper_index, element_index]
            block_lower = bank.sorted_residuals[lower_index, element_index]

            block_upper = np.where(upper_reached, block_upper, bank.fallback_upper)
            block_lower = np.where(lower_reached, block_lower, bank.fallback_lower)
            unusable = (total <= 0.0) | degenerate[:, None]
            block_upper = np.where(unusable, bank.fallback_upper, block_upper)
            block_lower = np.where(unusable, bank.fallback_lower, block_lower)
            fallback_out[block_rows] |= ~(upper_reached & lower_reached).all(axis=1)

            lower_offsets[block_rows] = block_lower
            upper_offsets[block_rows] = block_upper

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
            "calibration_fraction": self.calibration_fraction,
            "random_seed": self.random_seed,
            "split_pool_scenario_ids": None
            if self.split_pool_scenario_ids is None
            else self.split_pool_scenario_ids.tolist(),
            "calibration_scenario_ids": None
            if self.calibration_scenario_ids is None
            else self.calibration_scenario_ids.tolist(),
            "application_scenario_ids": None
            if self.application_scenario_ids is None
            else self.application_scenario_ids.tolist(),
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
            bandwidth=str(data.get("bandwidth", DEFAULT_BANDWIDTH)),
            distance=str(data.get("distance", DEFAULT_DISTANCE)),
            n_eff_target=float(
                cast(
                    Any,
                    data.get(
                        "n_eff_target",
                        DEFAULT_N_EFF_TARGET_SCALE / float(cast(Any, data["alpha"])),
                    ),
                )
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
            calibration_fraction=_optional_float(data.get("calibration_fraction")),
            random_seed=_optional_int(data.get("random_seed")),
            split_pool_scenario_ids=_optional_int_array(
                data.get("split_pool_scenario_ids")
            ),
            calibration_scenario_ids=_optional_int_array(
                data.get("calibration_scenario_ids")
            ),
            application_scenario_ids=_optional_int_array(
                data.get("application_scenario_ids")
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


def _effective_sample_size(
    distances: np.ndarray,
    bandwidth: np.ndarray,
) -> np.ndarray:
    """Effective sample size of the Gaussian kernel weights, per test row.

    `n_eff` is invariant to a common rescaling of the weights, so the exponent
    is shifted by each row's minimum distance. That keeps the computation from
    underflowing at small bandwidths without changing the result.
    """

    shifted = distances - distances.min(axis=1, keepdims=True)
    weights = np.exp(-shifted / bandwidth[:, None])
    weight_sum = weights.sum(axis=1)
    weight_sq_sum = np.einsum("ij,ij->i", weights, weights)
    return np.where(weight_sq_sum > 0.0, (weight_sum**2) / weight_sq_sum, 1.0)


def _solve_bandwidth_for_n_eff(
    distances: np.ndarray,
    target: float,
) -> np.ndarray:
    """Return the per-row bandwidth whose kernel weights hit `target` n_eff.

    `n_eff` increases monotonically from 1 (all weight on the nearest
    calibration scenario) to the stratum size (uniform weights), so a
    log-space bisection is well posed. All rows in the block are solved in
    lockstep so the work stays vectorized.
    """

    positive = np.where(distances > 0.0, distances, np.inf)
    nearest = positive.min(axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, 1.0)
    farthest = np.maximum(distances.max(axis=1), nearest)

    lower = nearest / _N_EFF_BRACKET_SPAN
    upper = farthest * _N_EFF_BRACKET_SPAN
    midpoint = np.sqrt(lower * upper)
    for _ in range(_N_EFF_SOLVE_MAX_ITER):
        midpoint = np.sqrt(lower * upper)
        n_eff = _effective_sample_size(distances, midpoint)
        if np.all(np.abs(n_eff - target) <= _N_EFF_SOLVE_RTOL * target):
            break
        below_target = n_eff < target
        lower = np.where(below_target, midpoint, lower)
        upper = np.where(below_target, upper, midpoint)
    return midpoint.astype(FLOAT32_DTYPE, copy=False)


def _resolve_chunk_size(n_calibration: int, n_elements: int) -> int:
    per_row_bytes = max(n_calibration * max(n_elements, 1) * 4, 1)
    return int(max(1, min(4096, _TARGET_BLOCK_BYTES // per_row_bytes)))


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


def _optional_int_array(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.int64)


def _optional_float_array(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=FLOAT32_DTYPE)


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    return float(cast(Any, value))


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(cast(Any, value))
