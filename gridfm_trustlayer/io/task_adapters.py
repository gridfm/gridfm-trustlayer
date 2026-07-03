"""Task-specific prediction semantics and correction rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass
class BusPredictionBundle:
    """Rectangular bus prediction arrays before task-specific correction."""

    Vm: np.ndarray
    Va: np.ndarray
    channels: dict[str, np.ndarray]


class TaskAdapter:
    """Apply task-aware correction rules and resolve auxiliary artifacts."""

    task_name = "Unknown"
    requires_generator_artifact = False

    def resolve_generator_artifact_path(
        self,
        bus_parquet_path: str,
        generator_parquet_path: str | None,
    ) -> str | None:
        if generator_parquet_path is not None:
            candidate_path = Path(generator_parquet_path)
            if not candidate_path.exists():
                raise ValueError(f"Missing generator predictions parquet: {candidate_path}")
            return str(candidate_path)

        if not self.requires_generator_artifact:
            return None

        sibling_generator_path = default_generator_artifact_path(bus_parquet_path)
        if sibling_generator_path is not None:
            return sibling_generator_path
        raise ValueError(
            "OptimalPowerFlow predictions require a sibling *_gen_predictions.parquet artifact.",
        )

    def apply_bus_corrections(
        self,
        predictions: BusPredictionBundle,
        target_channels: Mapping[str, np.ndarray],
        bus_type_masks: Mapping[str, np.ndarray],
    ) -> BusPredictionBundle:
        return predictions


class PowerFlowTaskAdapter(TaskAdapter):
    """PF-specific known-variable correction semantics."""

    task_name = "PowerFlow"

    def apply_bus_corrections(
        self,
        predictions: BusPredictionBundle,
        target_channels: Mapping[str, np.ndarray],
        bus_type_masks: Mapping[str, np.ndarray],
    ) -> BusPredictionBundle:
        corrected_channels = {
            name: values.copy() for name, values in predictions.channels.items()
        }
        pq_mask = bus_type_masks.get("PQ")
        pv_mask = bus_type_masks.get("PV")
        ref_mask = bus_type_masks.get("REF")

        if "Pg" in corrected_channels:
            corrected_channels["Pg"] = _replace_where(
                corrected_channels["Pg"],
                target_channels.get("Pg"),
                _combine_masks(pq_mask, pv_mask),
            )
        if "Qg" in corrected_channels:
            corrected_channels["Qg"] = _replace_where(
                corrected_channels["Qg"],
                target_channels.get("Qg"),
                pq_mask,
            )

        return BusPredictionBundle(
            Vm=_replace_where(predictions.Vm, target_channels.get("Vm"), pv_mask),
            Va=_replace_where(predictions.Va, target_channels.get("Va"), ref_mask),
            channels=corrected_channels,
        )


class OptimalPowerFlowTaskAdapter(TaskAdapter):
    """OPF-specific correction semantics and generator-artifact handling.

    The sibling generator parquet is required as a first-class artifact, but the
    current export must be treated as long-form scenario records rather than a
    stable scenario-by-generator tensor.
    """

    task_name = "OptimalPowerFlow"
    requires_generator_artifact = True

    def apply_bus_corrections(
        self,
        predictions: BusPredictionBundle,
        target_channels: Mapping[str, np.ndarray],
        bus_type_masks: Mapping[str, np.ndarray],
    ) -> BusPredictionBundle:
        return BusPredictionBundle(
            Vm=predictions.Vm.copy(),
            Va=_replace_where(predictions.Va, target_channels.get("Va"), bus_type_masks.get("REF")),
            channels={name: values.copy() for name, values in predictions.channels.items()},
        )


class PassthroughTaskAdapter(TaskAdapter):
    """Schema-preserving adapter for tasks without explicit correction rules yet."""

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name


def resolve_task_adapter(
    task_name: str | None,
    bus_parquet_path: str,
    generator_parquet_path: str | None,
    infer_power_flow: bool,
) -> TaskAdapter:
    """Resolve the task adapter from explicit task metadata or artifact layout."""

    normalized = _normalize_task_name(task_name)
    if normalized in {"powerflow", "pf"}:
        return PowerFlowTaskAdapter()
    if normalized in {"optimalpowerflow", "opf"}:
        return OptimalPowerFlowTaskAdapter()
    if task_name is not None:
        return PassthroughTaskAdapter(task_name)
    if generator_parquet_path is not None or default_generator_artifact_path(bus_parquet_path) is not None:
        return OptimalPowerFlowTaskAdapter()
    if infer_power_flow:
        return PowerFlowTaskAdapter()
    return PassthroughTaskAdapter("Unknown")


def default_generator_artifact_path(bus_parquet_path: str) -> str | None:
    """Return the default sibling generator parquet for OPF exports when present."""

    bus_path = Path(bus_parquet_path)
    if not bus_path.name.endswith("_predictions.parquet"):
        return None
    candidate = bus_path.with_name(
        bus_path.name.replace("_predictions.parquet", "_gen_predictions.parquet"),
    )
    if candidate.exists():
        return str(candidate)
    return None


def _combine_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    active_masks = [mask for mask in masks if mask is not None]
    if not active_masks:
        return None
    combined = active_masks[0].copy()
    for mask in active_masks[1:]:
        combined |= mask
    return combined


def _replace_where(
    prediction: np.ndarray,
    target: np.ndarray | None,
    mask: np.ndarray | None,
) -> np.ndarray:
    if target is None or mask is None:
        return prediction.copy()
    return np.where(mask, target, prediction).astype(np.float32, copy=False)


def _normalize_task_name(task_name: str | None) -> str | None:
    if task_name is None:
        return None
    return task_name.replace("_", "").replace("-", "").strip().lower()