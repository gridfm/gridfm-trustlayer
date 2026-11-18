"""Load per-entity model embeddings and pool them into scenario representations.

Kernel-weighted conformal prediction (KCP) needs one vector ``z(x)`` per
scenario. Graphkit exports embeddings per bus, so this module owns the pooling
step that reduces ``(n_scenarios, n_buses, d)`` to ``(n_scenarios, d')``.
Pooling strategies live in :data:`POOLING_REGISTRY` so new ones can be added
without touching the calibration code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from gridfm_trustlayer.io.parquet import column_names, open_dataset

FLOAT32_DTYPE = np.dtype(np.float32)

EMBEDDING_COLUMN_PREFIX = "emb_"

# Embedding values decoded at once while pooling, in bytes: columns are read in
# groups of this size, so that memory does not grow with the file.
_COLUMN_BYTES = 64 * 1024 * 1024


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

    def subset(self, scenario_ids: Iterable[int]) -> ScenarioEmbeddingSet:
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


# Each strategy applied to an `(n_scenarios, n_buses, d)` array. The loader
# below computes the same values one embedding dimension at a time.
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
    """Read an embeddings parquet and pool it to one vector per scenario.

    The file is read one embedding column at a time, so memory stays bounded by
    a few columns of the file and the pooled result, however large the file
    (graphkit exports one row per scenario and bus).

    Args:
        parquet_path: Embeddings file with `scenario`, `bus` and `emb_*`
            columns, e.g. graphkit's `<grid>_bus_embeddings.parquet`. Every
            scenario must have the same buses.
        pooling: `"mean"`, `"max"`, `"meanmax"` or `"concat"`.
        scenario_ids: Scenarios to return, in this order; all when not given.

    Returns:
        A `ScenarioEmbeddingSet` with `scenario_ids` and an
        `(n_scenarios, d)` array `embeddings`.
    """

    pooling = validate_pooling(pooling)
    available = column_names(parquet_path)
    if "scenario" not in available:
        raise ValueError(
            f"Embedding parquet {parquet_path!r} must contain a 'scenario' column.",
        )
    if "bus" not in available:
        raise ValueError(
            f"Embedding parquet {parquet_path!r} must contain a 'bus' column.",
        )
    embedding_columns = sorted(
        (name for name in available if str(name).startswith(EMBEDDING_COLUMN_PREFIX)),
        key=lambda name: str(name),
    )
    if not embedding_columns:
        raise ValueError(
            f"Embedding parquet {parquet_path!r} contains no {EMBEDDING_COLUMN_PREFIX}* columns.",
        )

    files = [pq.ParquetFile(path) for path in _parquet_files(parquet_path)]
    scenario = _read_columns(files, ["scenario"])[0].astype(np.int64, copy=False)
    bus = _read_columns(files, ["bus"])[0].astype(np.int64, copy=False)
    # Rows in (scenario, bus) order; scenario-major as a stable sort of both.
    rows: np.ndarray = np.arange(scenario.size)
    if scenario_ids is not None:
        rows = rows[np.isin(scenario, np.asarray(scenario_ids, dtype=np.int64))]
        if rows.size == 0:
            raise ValueError(
                "No embedding rows remain after filtering to the requested scenarios.",
            )
    rows = rows[np.lexsort((bus[rows], scenario[rows]))]
    unique_scenarios, counts = np.unique(scenario[rows], return_counts=True)
    if len(np.unique(counts)) != 1:
        raise ValueError(
            "Embedding parquet is not rectangular: scenarios carry differing "
            f"entity counts (min={counts.min()}, max={counts.max()}).",
        )
    n_scenarios, n_buses = len(unique_scenarios), int(counts[0])
    width = len(embedding_columns)
    group = max(1, min(64, _COLUMN_BYTES // max(4 * scenario.size, 1)))
    del scenario, bus

    def columns() -> Iterable[tuple[int, np.ndarray]]:
        """Each embedding column as `(n_scenarios, n_buses)`, in column order."""

        for start in range(0, width, group):
            names = embedding_columns[start : start + group]
            for offset, values in enumerate(_read_columns(files, names)):
                yield (
                    start + offset,
                    values.astype(FLOAT32_DTYPE, copy=False)[rows].reshape(
                        n_scenarios, n_buses
                    ),
                )

    pooled: np.ndarray
    if pooling in _STREAMED_POOLINGS:
        pooled = np.empty(
            (n_scenarios, _STREAMED_POOLINGS[pooling](n_buses, width)),
            dtype=FLOAT32_DTYPE,
        )
        for index, values in columns():
            # The bus axis is reduced innermost: the order in which earlier
            # versions (pandas frame -> F-ordered array -> mean) added it up.
            if pooling in ("mean", "meanmax"):
                pooled[:, index] = np.add.reduce(values, axis=1) / FLOAT32_DTYPE.type(
                    n_buses
                )
            if pooling in ("max", "meanmax"):
                pooled[:, index + (width if pooling == "meanmax" else 0)] = values.max(
                    axis=1
                )
            if pooling == "concat":
                pooled[:, index::width] = values
    else:
        # A strategy added to POOLING_REGISTRY: it gets the whole array.
        stacked = np.empty((n_scenarios, n_buses, width), dtype=FLOAT32_DTYPE)
        for index, values in columns():
            stacked[:, :, index] = values
        pooled = np.ascontiguousarray(
            POOLING_REGISTRY[pooling](stacked), dtype=FLOAT32_DTYPE
        )

    # One NaN would make every distance NaN and switch the kernel off silently.
    broken = ~np.isfinite(pooled).all(axis=1)
    if broken.any():
        examples = ", ".join(str(s) for s in unique_scenarios[broken][:5].tolist())
        raise ValueError(
            f"Embedding parquet {parquet_path!r} has NaN or infinite values in "
            f"{int(broken.sum()):,} scenario(s) (e.g. {examples})."
        )
    return ScenarioEmbeddingSet(
        scenario_ids=unique_scenarios,
        embeddings=pooled,
        pooling=pooling,
        source_dim=width,
    )


# Width of the pooled vector for the strategies pooled one column at a time.
_STREAMED_POOLINGS: dict[str, Callable[[int, int], int]] = {
    "mean": lambda n_buses, width: width,
    "max": lambda n_buses, width: width,
    "meanmax": lambda n_buses, width: 2 * width,
    "concat": lambda n_buses, width: n_buses * width,
}


def _parquet_files(path: str) -> list[str]:
    """The files of a parquet file or partitioned dataset, in a fixed order."""

    return list(getattr(open_dataset(path), "files", [path]))


def _read_columns(files: list[pq.ParquetFile], names: list[str]) -> list[np.ndarray]:
    """Whole columns of a dataset, files concatenated in order."""

    parts = [file.read(columns=names) for file in files]
    return [
        np.concatenate([np.asarray(part.column(name).to_numpy()) for part in parts])
        for name in names
    ]
