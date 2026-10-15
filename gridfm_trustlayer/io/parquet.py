"""Parquet datasets: one file or a Hive-partitioned directory.

Inputs may be single parquet files (graphkit writes these) or Hive-partitioned
directories such as datakit's `raw/bus_data.parquet/scenario_partition=k/`.
Reads select only the columns and scenarios they need. Large outputs are
written partitioned the same way, `SCENARIOS_PER_PARTITION` scenarios per
partition, one block of scenarios at a time, so that neither reading nor
writing needs the whole dataset in memory.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

PARTITION_COLUMN = "scenario_partition"
# Same partition size as gridfm-datakit's raw outputs.
SCENARIOS_PER_PARTITION = 200
# Rows (scenario x entity) processed at once by `calibrate` and `screen`.
DEFAULT_BLOCK_ROWS = 500_000


def open_dataset(path: str | Path) -> ds.Dataset:
    """Open a parquet file or a Hive-partitioned parquet directory.

    Opened datasets are cached until the path changes on disk, so that reading
    one block at a time does not list a partitioned directory again per block.
    """

    resolved = Path(path).resolve()
    return _open_dataset(str(resolved), resolved.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def _open_dataset(path: str, modified_ns: int) -> ds.Dataset:
    del modified_ns  # part of the cache key only
    return ds.dataset(path, format="parquet", partitioning="hive")


def column_names(path: str | Path) -> list[str]:
    """Data columns of a dataset, without the partition column."""

    return [
        name for name in open_dataset(path).schema.names if name != PARTITION_COLUMN
    ]


def read_frame(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    scenario_ids: Iterable[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Read selected columns and scenarios of a dataset into a DataFrame.

    Args:
        path: Parquet file or partitioned directory.
        columns: Columns to read; names that do not exist are skipped. All data
            columns when `None`.
        scenario_ids: Only rows of these scenarios; all rows when `None`.
    """

    dataset = open_dataset(path)
    available = dataset.schema.names
    if columns is None:
        selected = [name for name in available if name != PARTITION_COLUMN]
    else:
        selected = [name for name in dict.fromkeys(columns) if name in available]
    expression = None
    if scenario_ids is not None:
        ids = np.unique(np.asarray(list(scenario_ids), dtype=np.int64))
        scenario_type = dataset.schema.field("scenario").type
        expression = ds.field("scenario").isin(pa.array(ids).cast(scenario_type))
    return dataset.to_table(columns=selected, filter=expression).to_pandas()


def read_scenario_ids(path: str | Path) -> np.ndarray:
    """Sorted unique scenario ids, read one batch at a time."""

    unique_per_batch = [
        np.unique(batch.column(0).to_numpy(zero_copy_only=False))
        for batch in open_dataset(path).to_batches(columns=["scenario"])
    ]
    if not unique_per_batch:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(unique_per_batch)).astype(np.int64, copy=False)


def rows_per_scenario(path: str | Path, n_scenarios: int) -> int:
    """Average number of rows per scenario, from the file metadata."""

    if n_scenarios == 0:
        return 1
    return max(1, -(-open_dataset(path).count_rows() // n_scenarios))


def scenario_blocks(
    scenario_ids: np.ndarray,
    rows_per_scenario: int,
    block_rows: int = DEFAULT_BLOCK_ROWS,
) -> list[np.ndarray]:
    """Split sorted scenario ids into blocks of about `block_rows` rows.

    Blocks consist of whole output partitions (`scenario // SCENARIOS_PER_PARTITION`),
    so each partition of a partitioned output is written by exactly one block.
    """

    ids = np.asarray(scenario_ids, dtype=np.int64)
    if ids.size == 0:
        return []
    partitions = ids // SCENARIOS_PER_PARTITION
    starts = np.concatenate([[0], np.flatnonzero(np.diff(partitions)) + 1])
    per_block = max(1, int(block_rows) // max(1, int(rows_per_scenario)))
    blocks: list[np.ndarray] = []
    block_start = 0
    for start in starts.tolist()[1:]:
        if start - block_start >= per_block:
            blocks.append(ids[block_start:start])
            block_start = start
    blocks.append(ids[block_start:])
    return blocks


class PartitionedWriter:
    """Write a DataFrame block by block as a Hive-partitioned parquet dataset.

    Rows are partitioned by `scenario // SCENARIOS_PER_PARTITION`. Floats use
    zstd with byte-stream-split encoding, like gridfm-datakit's outputs. The
    target is replaced: an existing file is removed, and so is an existing
    directory if it holds only partitions written by a previous run.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clear: bool = True,
        schema: pa.Schema | None = None,
    ) -> None:
        self.path = Path(path)
        if clear:
            clear_partitioned_output(self.path)
        self._blocks_written = 0
        self._schema = schema

    @property
    def schema(self) -> pa.Schema | None:
        """Column types of the dataset, fixed by the first block written."""

        return self._schema

    def write(self, frame: pd.DataFrame, *, block_index: int | None = None) -> None:
        """Write one block of rows.

        Several writers (e.g. in parallel processes) can add blocks to the same
        dataset when each passes the dataset's `schema` and its own
        `block_index`, and their blocks hold different partitions.
        """

        if frame.empty:
            return
        partition = (
            frame["scenario"].to_numpy(dtype=np.int64) // SCENARIOS_PER_PARTITION
        )
        table = _plain_types(pa.Table.from_pandas(frame, preserve_index=False))
        table = table.append_column(PARTITION_COLUMN, pa.array(partition))
        if self._schema is None:
            self._schema = table.schema
        else:
            # Same columns in every block; e.g. a label column that holds only
            # nulls in one block still gets the string type of the others.
            table = table.cast(self._schema)
        floats = [
            field.name
            for field in table.schema
            if pa.types.is_floating(field.type) and field.name != PARTITION_COLUMN
        ]
        pq.write_to_dataset(
            table,
            str(self.path),
            partition_cols=[PARTITION_COLUMN],
            basename_template=(
                f"part-{self._blocks_written if block_index is None else block_index:05d}"
                "-{i}.parquet"
            ),
            existing_data_behavior="overwrite_or_ignore",
            compression="zstd",
            use_dictionary=[name for name in table.column_names if name not in floats],
            column_encoding={name: "BYTE_STREAM_SPLIT" for name in floats},
        )
        self._blocks_written += 1


def _plain_types(table: pa.Table) -> pa.Table:
    """Plain column types: categoricals as their values, text and all-null columns as `string`.

    The pandas metadata is dropped too, so the dataset reads back with the
    same types whichever block a reader sees first.
    """

    fields = []
    for field in table.schema:
        if pa.types.is_dictionary(field.type):
            field = field.with_type(field.type.value_type)
        if pa.types.is_null(field.type) or pa.types.is_large_string(field.type):
            field = field.with_type(pa.string())
        fields.append(field)
    return table.cast(pa.schema(fields))


def is_partitioned_dataset(path: str | Path) -> bool:
    """Whether `path` is a partitioned dataset (a folder of `scenario_partition=k`)."""

    path = Path(path)
    return path.is_dir() and any(
        child.is_dir() and child.name.startswith(f"{PARTITION_COLUMN}=")
        for child in path.iterdir()
    )


def clear_partitioned_output(
    path: str | Path, *, inputs: Iterable[str | Path | None] = ()
) -> None:
    """Remove a previous trustlayer output at `path`; refuse anything else.

    Only a file or partitioned dataset with interval bounds among its columns
    (`*pi_lower`, in every trustlayer output) is removed, and never one of the
    command's `inputs`. The layout alone does not tell an output apart: datakit
    writes its raw data in the same partitioned layout.
    """

    path = Path(path)
    if not path.exists():
        return
    resolved = path.resolve()
    if any(
        source is not None and Path(source).resolve() == resolved for source in inputs
    ):
        raise ValueError(
            f"The output path {path} is also an input of this command; choose "
            "another output path.",
        )
    if path.is_dir():
        children = list(path.iterdir())
        if not children:
            path.rmdir()
            return
        partitioned = all(
            child.is_dir() and child.name.startswith(f"{PARTITION_COLUMN}=")
            for child in children
        )
        if partitioned and _has_interval_columns(path):
            shutil.rmtree(path)
            return
    elif _has_interval_columns(path):
        path.unlink()
        return
    raise ValueError(
        f"{path} exists and is not a trustlayer output: it would be replaced. "
        "Choose another output path or remove it.",
    )


def _has_interval_columns(path: Path) -> bool:
    """Whether `path` is a parquet file or dataset with interval bounds."""

    try:
        names = (
            pq.read_schema(path).names
            if path.is_file()
            else ds.dataset(path, format="parquet", partitioning="hive").schema.names
        )
    except (OSError, pa.ArrowException):
        return False
    return any(name.endswith("pi_lower") for name in names)
