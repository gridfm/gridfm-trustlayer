"""Reading and writing parquet datasets, one file or Hive-partitioned."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer.io.parquet import (
    PARTITION_COLUMN,
    SCENARIOS_PER_PARTITION,
    PartitionedWriter,
    column_names,
    read_frame,
    read_scenario_ids,
    rows_per_scenario,
    scenario_blocks,
)


def _frame(scenarios: np.ndarray, entities: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": np.repeat(scenarios, entities),
            "bus": np.tile(np.arange(entities), len(scenarios)),
            "value": np.arange(len(scenarios) * entities, dtype=np.float32),
            "label": "x",
        }
    )


@pytest.mark.parametrize("layout", ["file", "partitioned"])
def test_read_frame_selects_columns_and_scenarios_in_both_layouts(
    tmp_path, layout
) -> None:
    frame = _frame(np.arange(450))
    path = tmp_path / "data.parquet"
    if layout == "file":
        frame.to_parquet(path, index=False)
    else:
        frame.assign(**{PARTITION_COLUMN: frame["scenario"] // 200}).to_parquet(
            path, partition_cols=[PARTITION_COLUMN]
        )

    assert column_names(path) == ["scenario", "bus", "value", "label"]
    selected = read_frame(path, ["scenario", "value", "missing"], scenario_ids=[3, 401])
    assert list(selected.columns) == ["scenario", "value"]
    selected = selected.sort_values(["scenario", "value"]).reset_index(drop=True)
    expected = frame[frame["scenario"].isin([3, 401])][["scenario", "value"]]
    np.testing.assert_array_equal(selected["value"], expected["value"])
    np.testing.assert_array_equal(read_scenario_ids(path), np.arange(450))
    assert rows_per_scenario(path, 450) == 3


def test_scenario_blocks_hold_whole_partitions_of_about_the_requested_rows() -> None:
    ids = np.arange(1000)
    blocks = scenario_blocks(ids, rows_per_scenario=10, block_rows=4000)

    np.testing.assert_array_equal(np.concatenate(blocks), ids)
    for block in blocks:
        assert block[0] % SCENARIOS_PER_PARTITION == 0
        assert len(block) % SCENARIOS_PER_PARTITION == 0 or block is blocks[-1]
    assert [len(block) for block in blocks] == [400, 400, 200]
    # A block is never empty, even when one partition alone exceeds the budget.
    assert [len(block) for block in scenario_blocks(ids, 10, block_rows=1)] == [200] * 5
    assert scenario_blocks(np.array([], dtype=np.int64), 10) == []


def test_partitioned_writer_writes_blocks_that_read_back_as_one_dataset(
    tmp_path,
) -> None:
    path = tmp_path / "out.parquet"
    writer = PartitionedWriter(path)
    first = _frame(np.arange(0, 200))
    second = _frame(np.arange(200, 400)).assign(label=pd.NA)  # only nulls here
    writer.write(first.assign(label=pd.Categorical(first["label"])))
    writer.write(second)

    partitions = sorted(child.name for child in path.iterdir())
    assert partitions == [f"{PARTITION_COLUMN}=0", f"{PARTITION_COLUMN}=1"]
    data = pd.read_parquet(path).sort_values(["scenario", "bus"]).reset_index(drop=True)
    assert len(data) == 1200
    assert data["label"].iloc[0] == "x"
    assert data["label"].iloc[-1] is None or pd.isna(data["label"].iloc[-1])
    schema = pq.read_schema(next((path / f"{PARTITION_COLUMN}=0").iterdir()))
    assert schema.field("label").type == pa.string()
    assert schema.field("value").type == pa.float32()


def test_partitioned_writer_replaces_previous_outputs_but_nothing_else(
    tmp_path,
) -> None:
    old_file = tmp_path / "old.parquet"
    old_file.write_text("previous single-file output")
    PartitionedWriter(old_file).write(_frame(np.arange(3)))
    assert old_file.is_dir()

    PartitionedWriter(old_file).write(_frame(np.arange(5)))
    assert len(pd.read_parquet(old_file)) == 15

    other = tmp_path / "results"
    other.mkdir()
    (other / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="not a partitioned trustlayer output"):
        PartitionedWriter(other)
    assert (other / "notes.txt").exists()
