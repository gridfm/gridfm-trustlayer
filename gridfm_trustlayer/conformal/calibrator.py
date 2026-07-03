"""Naive residual-quantile interval calibration for direct bus outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from gridfm_trustlayer.io.ground_truth import GroundTruthSet
from gridfm_trustlayer.io.predictions import PredictionSet


FLOAT32_DTYPE = np.dtype(np.float32)


@dataclass
class ChannelSet:
    """Aligned scenario-by-entity channel arrays for calibration/application."""

    scenario_ids: np.ndarray
    entity_ids: np.ndarray
    entity_name: str
    channels: dict[str, np.ndarray]
    valid_masks: dict[str, np.ndarray] = field(default_factory=dict)
    entity_columns: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def bus_ids(self) -> np.ndarray:
        """Compatibility alias for older bus-only call sites."""

        return self.entity_ids


@dataclass
class NaiveIntervalCalibrator:
    """Per-entity residual-quantile intervals for aligned prediction channels.

    This initial Phase 3 implementation is intentionally simple: it calibrates
    each requested variable independently per entity index, ignoring contingency
    structure and using empirical residual quantiles at `alpha / 2` and
    `1 - alpha / 2`.
    """

    alpha: float
    entity_ids: np.ndarray
    entity_name: str
    channel_names: tuple[str, ...]
    lower_residual_quantiles: dict[str, np.ndarray]
    upper_residual_quantiles: dict[str, np.ndarray]
    calibration_fraction: float | None = None
    random_seed: int | None = None
    split_pool_scenario_ids: np.ndarray | None = None
    calibration_scenario_ids: np.ndarray | None = None
    application_scenario_ids: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        predictions: PredictionSet | ChannelSet,
        ground_truth: GroundTruthSet | ChannelSet,
        *,
        alpha: float,
        channel_names: Iterable[str] = ("Vm", "Va"),
    ) -> "NaiveIntervalCalibrator":
        """Fit naive residual-quantile intervals from aligned predictions and targets."""

        _validate_alpha(alpha)
        _validate_entity_alignment(_get_entity_ids(predictions), _get_entity_ids(ground_truth))
        if not np.array_equal(predictions.scenario_ids, ground_truth.scenario_ids):
            raise ValueError(
                "Prediction and ground-truth scenario_ids must be aligned before fitting.",
            )

        selected_channels = tuple(channel_names)
        lower_residual_quantiles: dict[str, np.ndarray] = {}
        upper_residual_quantiles: dict[str, np.ndarray] = {}
        for channel_name in selected_channels:
            prediction_values = _get_prediction_channel(predictions, channel_name)
            target_values = _get_ground_truth_channel(ground_truth, channel_name)
            residuals = (target_values - prediction_values).astype(FLOAT32_DTYPE, copy=False)
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

        return cls(
            alpha=float(alpha),
            entity_ids=_get_entity_ids(predictions).copy(),
            entity_name=_get_entity_name(predictions),
            channel_names=selected_channels,
            lower_residual_quantiles=lower_residual_quantiles,
            upper_residual_quantiles=upper_residual_quantiles,
        )

    def predict_intervals(
        self,
        predictions: PredictionSet | ChannelSet,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return lower and upper prediction-interval arrays for each channel."""

        _validate_entity_alignment(self.entity_ids, _get_entity_ids(predictions))
        intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for channel_name in self.channel_names:
            prediction_values = _get_prediction_channel(predictions, channel_name)
            lower = prediction_values + self.lower_residual_quantiles[channel_name][None, :]
            upper = prediction_values + self.upper_residual_quantiles[channel_name][None, :]
            valid_mask = _get_valid_mask(predictions, channel_name)
            if valid_mask is not None:
                lower = np.where(valid_mask, lower, prediction_values)
                upper = np.where(valid_mask, upper, prediction_values)
            intervals[channel_name] = (
                lower.astype(FLOAT32_DTYPE, copy=False),
                upper.astype(FLOAT32_DTYPE, copy=False),
            )
        return intervals

    def to_dict(self) -> dict[str, object]:
        """Serialize the calibrator into a JSON-friendly dictionary."""

        return {
            "version": "trustlayer_naive_interval_v1",
            "method": "naive_residual_quantiles",
            "alpha": self.alpha,
            "entity_ids": self.entity_ids.tolist(),
            "entity_name": self.entity_name,
            "bus_ids": self.entity_ids.tolist() if self.entity_name == "bus" else None,
            "channel_names": list(self.channel_names),
            "calibration_fraction": self.calibration_fraction,
            "random_seed": self.random_seed,
            "split_pool_scenario_ids": None if self.split_pool_scenario_ids is None else self.split_pool_scenario_ids.tolist(),
            "calibration_scenario_ids": None if self.calibration_scenario_ids is None else self.calibration_scenario_ids.tolist(),
            "application_scenario_ids": None if self.application_scenario_ids is None else self.application_scenario_ids.tolist(),
            "lower_residual_quantiles": {
                key: values.tolist()
                for key, values in self.lower_residual_quantiles.items()
            },
            "upper_residual_quantiles": {
                key: values.tolist()
                for key, values in self.upper_residual_quantiles.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NaiveIntervalCalibrator":
        """Deserialize a calibrator previously produced by :meth:`to_dict`."""

        version = data.get("version")
        if version != "trustlayer_naive_interval_v1":
            raise ValueError(f"Unsupported interval artifact version: {version!r}")

        channel_names = tuple(str(name) for name in data["channel_names"])
        lower_residual_quantiles = {
            str(key): np.asarray(values, dtype=FLOAT32_DTYPE)
            for key, values in dict(data["lower_residual_quantiles"]).items()
        }
        upper_residual_quantiles = {
            str(key): np.asarray(values, dtype=FLOAT32_DTYPE)
            for key, values in dict(data["upper_residual_quantiles"]).items()
        }
        return cls(
            alpha=float(data["alpha"]),
            entity_ids=np.asarray(data.get("entity_ids", data.get("bus_ids")), dtype=np.int64),
            entity_name=str(data.get("entity_name", "bus")),
            channel_names=channel_names,
            lower_residual_quantiles=lower_residual_quantiles,
            upper_residual_quantiles=upper_residual_quantiles,
            calibration_fraction=_optional_float(data.get("calibration_fraction")),
            random_seed=_optional_int(data.get("random_seed")),
            split_pool_scenario_ids=_optional_int_array(data.get("split_pool_scenario_ids")),
            calibration_scenario_ids=_optional_int_array(data.get("calibration_scenario_ids")),
            application_scenario_ids=_optional_int_array(data.get("application_scenario_ids")),
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


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1; got {alpha!r}")


def _validate_entity_alignment(left_entity_ids: np.ndarray, right_entity_ids: np.ndarray) -> None:
    if not np.array_equal(left_entity_ids, right_entity_ids):
        raise ValueError("Prediction and calibration artifacts must share the same entity ordering.")


def _get_entity_ids(values: PredictionSet | GroundTruthSet | ChannelSet) -> np.ndarray:
    if isinstance(values, ChannelSet):
        return values.entity_ids
    return values.bus_ids


def _get_entity_name(values: PredictionSet | GroundTruthSet | ChannelSet) -> str:
    if isinstance(values, ChannelSet):
        return values.entity_name
    return "bus"


def _get_valid_mask(
    values: PredictionSet | GroundTruthSet | ChannelSet,
    channel_name: str,
) -> np.ndarray | None:
    if isinstance(values, ChannelSet):
        return values.valid_masks.get(channel_name)
    return None


def _combine_valid_masks(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
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
    if valid_mask is None:
        return np.quantile(residuals, quantile, axis=0).astype(FLOAT32_DTYPE, copy=False)

    output = np.zeros(residuals.shape[1], dtype=FLOAT32_DTYPE)
    for index in range(residuals.shape[1]):
        column_mask = valid_mask[:, index]
        if np.any(column_mask):
            output[index] = np.quantile(residuals[column_mask, index], quantile)
        else:
            output[index] = np.float32(0.0)
    return output.astype(FLOAT32_DTYPE, copy=False)


def _optional_int_array(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.int64)


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)