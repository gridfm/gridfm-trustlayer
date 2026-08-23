"""Load per-entity model embeddings and pool them into scenario representations.

Kernel-weighted conformal prediction (KCP) needs one vector ``z(x)`` per
scenario. Graphkit exports embeddings per bus, so this module owns the pooling
step that reduces ``(n_scenarios, n_buses, d)`` to ``(n_scenarios, d')``.
Pooling strategies live in :data:`POOLING_REGISTRY` so new ones can be added
without touching the calibration code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


FLOAT32_DTYPE = np.dtype(np.float32)

EMBEDDING_COLUMN_PREFIX = "emb_"


@dataclass
class ScenarioEmbeddingSet:
    """Scenario-level representations aligned to sorted scenario ids."""

    scenario_ids: np.ndarray
    embeddings: np.ndarray
    pooling: str
    source_dim: int

    @property
    def dim(self) -> int:
        """Width of the pooled representation."""

        return int(self.embeddings.shape[1])

    def subset(self, scenario_ids: Iterable[int]) -> "ScenarioEmbeddingSet":
        """Return the rows matching `scenario_ids`, preserving stored order."""

        wanted = np.asarray(list(scenario_ids), dtype=np.int64)
        mask = np.isin(self.scenario_ids, wanted)
        return ScenarioEmbeddingSet(
            scenario_ids=self.scenario_ids[mask].copy(),
            embeddings=self.embeddings[mask].copy(),
            pooling=self.pooling,
            source_dim=self.source_dim,
        )

    def align_to(self, scenario_ids: np.ndarray) -> np.ndarray:
        """Return embeddings ordered to match `scenario_ids` exactly."""

        wanted = np.asarray(scenario_ids, dtype=np.int64)
        order = np.searchsorted(self.scenario_ids, wanted)
        if np.any(order >= len(self.scenario_ids)) or not np.array_equal(
            self.scenario_ids[np.clip(order, 0, len(self.scenario_ids) - 1)],
            wanted,
        ):
            missing = np.setdiff1d(wanted, self.scenario_ids)
            raise ValueError(
                "Embedding parquet is missing "
                f"{len(missing)} scenario(s) present in the predictions, "
                f"for example {missing[:5].tolist()}.",
            )
        return self.embeddings[order]


def _pool_mean(stacked: np.ndarray) -> np.ndarray:
    return stacked.mean(axis=1)


def _pool_max(stacked: np.ndarray) -> np.ndarray:
    return stacked.max(axis=1)


def _pool_meanmax(stacked: np.ndarray) -> np.ndarray:
    return np.concatenate([stacked.mean(axis=1), stacked.max(axis=1)], axis=1)


def _pool_concat(stacked: np.ndarray) -> np.ndarray:
    return stacked.reshape(stacked.shape[0], -1)


POOLING_REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": _pool_mean,
    "max": _pool_max,
    "meanmax": _pool_meanmax,
    "concat": _pool_concat,
}

DEFAULT_POOLING = "mean"


def validate_pooling(pooling: str | None) -> str:
    """Validate and normalize a pooling-strategy name."""

    normalized = DEFAULT_POOLING if pooling is None else str(pooling).strip().lower()
    if normalized not in POOLING_REGISTRY:
        supported = ", ".join(sorted(POOLING_REGISTRY))
        raise ValueError(
            f"Unsupported embedding_pooling: {pooling!r}. Supported values: {supported}.",
        )
    return normalized


def load_scenario_embeddings(
    parquet_path: str,
    *,
    pooling: str = DEFAULT_POOLING,
    scenario_ids: np.ndarray | None = None,
) -> ScenarioEmbeddingSet:
    """Load a graphkit embeddings parquet and pool it to one vector per scenario.

    The parquet is expected to be rectangular (every scenario carries the same
    entity ids), which lets the pooling run as a single reshape plus reduction
    instead of a per-scenario groupby.
    """

    pooling = validate_pooling(pooling)
    columns_needed = None
    frame = pd.read_parquet(parquet_path, columns=columns_needed)
    if "scenario" not in frame.columns:
        raise ValueError(
            f"Embedding parquet {parquet_path!r} must contain a 'scenario' column.",
        )
    entity_column = "bus" if "bus" in frame.columns else None
    if entity_column is None:
        raise ValueError(
            f"Embedding parquet {parquet_path!r} must contain a 'bus' column.",
        )

    embedding_columns = [
        column
        for column in frame.columns
        if str(column).startswith(EMBEDDING_COLUMN_PREFIX)
    ]
    if not embedding_columns:
        raise ValueError(
            f"Embedding parquet {parquet_path!r} contains no {EMBEDDING_COLUMN_PREFIX}* columns.",
        )
    embedding_columns.sort(key=lambda name: str(name))

    frame = frame[["scenario", entity_column, *embedding_columns]]
    if scenario_ids is not None:
        wanted = np.asarray(scenario_ids, dtype=np.int64)
        frame = frame[frame["scenario"].isin(wanted)]
        if frame.empty:
            raise ValueError(
                "No embedding rows remain after filtering to the requested scenarios.",
            )

    frame = frame.sort_values(["scenario", entity_column], kind="stable")
    scenario_values = frame["scenario"].to_numpy(dtype=np.int64, copy=False)
    unique_scenarios, counts = np.unique(scenario_values, return_counts=True)
    if len(np.unique(counts)) != 1:
        raise ValueError(
            "Embedding parquet is not rectangular: scenarios carry differing "
            f"entity counts (min={counts.min()}, max={counts.max()}).",
        )
    entities_per_scenario = int(counts[0])

    values = frame[embedding_columns].to_numpy(dtype=FLOAT32_DTYPE, copy=False)
    stacked = values.reshape(
        len(unique_scenarios), entities_per_scenario, len(embedding_columns)
    )
    pooled = np.ascontiguousarray(
        POOLING_REGISTRY[pooling](stacked), dtype=FLOAT32_DTYPE
    )
    return ScenarioEmbeddingSet(
        scenario_ids=unique_scenarios,
        embeddings=pooled,
        pooling=pooling,
        source_dim=len(embedding_columns),
    )
