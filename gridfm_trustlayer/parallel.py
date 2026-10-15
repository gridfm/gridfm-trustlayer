"""Process pools: the bandwidth-tuning folds and, on request, blocks of scenarios.

Work runs in separate processes started with `spawn`, which is safe whatever
threads the parent runs (Arrow starts some) and works on every platform. The
number of processes comes from `n_jobs`:

- `None` or `0`: `DEFAULT_MAX_JOBS`, at most the number of CPUs;
- a positive number: that many processes;
- a negative number: all CPUs but `-n_jobs - 1` (`-1` uses them all);
- `1`: no pool, everything runs in the calling process.

Each worker caps its native thread pools (BLAS, OpenMP, Arrow) at its share of
the CPUs, so that processes and threads do not oversubscribe the machine.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any, Protocol

DEFAULT_MAX_JOBS = 4
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def resolve_jobs(n_jobs: int | None) -> int:
    """The number of processes `n_jobs` asks for (see the module docstring)."""

    cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return max(1, min(DEFAULT_MAX_JOBS, cpus))
    if n_jobs < 0:
        return max(1, cpus + 1 + n_jobs)
    return int(n_jobs)


def threads_per_worker(workers: int) -> int:
    return max(1, (os.cpu_count() or 1) // max(1, workers))


@contextmanager
def process_pool(
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: Sequence[Any] = (),
) -> Iterator[ProcessPoolExecutor]:
    """A pool of `workers` spawned processes, each with its share of the CPUs."""

    threads = threads_per_worker(workers)
    # Spawned processes read these when their native libraries load.
    saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    os.environ.update({name: str(threads) for name in _THREAD_VARIABLES})
    # Workers draw figures off-screen.
    saved["MPLBACKEND"] = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(threads, initializer, tuple(initargs)),
        ) as pool:
            yield pool
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize(
    threads: int,
    initializer: Callable[..., None] | None,
    initargs: tuple[Any, ...],
) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]

    pa.set_memory_pool(pa.system_memory_pool())
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    if initializer is not None:
        initializer(*initargs)


# ------------------------------------------------------------------ blocks
class BlockJob(Protocol):
    """The work of one command on one block of scenarios.

    The job is pickled to every worker, which calls `setup` once before its
    first block: heavy state (calibrators, embeddings) is built there, not
    pickled. `run` returns the block's partial results; it may also update
    picklable attributes of the job (e.g. the output schemas, fixed by the
    first block) before the job is sent to the workers.
    """

    def setup(self, threads: int | None) -> None: ...

    def run(self, index: int, block: Any) -> Any: ...


_JOB: BlockJob | None = None


def _initialize_job(job: BlockJob, threads: int, settings: dict[str, Any]) -> None:
    global _JOB
    apply_settings(settings)
    job.setup(threads)
    _JOB = job


def _run_block(index: int, block: Any) -> Any:
    assert _JOB is not None
    return _JOB.run(index, block)


def map_blocks(
    job: BlockJob,
    blocks: Sequence[Any],
    *,
    n_jobs: int | None,
    progress: Callable[[int], None] | None = None,
) -> list[Any]:
    """`job.run` on every block, the first in this process, the rest in parallel.

    Results come back in block order. Running the first block here lets it fix
    whatever the others must match (e.g. the column types of a partitioned
    output) before the job is copied to the workers.
    """

    if not blocks:
        return []
    workers = min(resolve_jobs(n_jobs), len(blocks) - 1)
    job.setup(None if workers <= 1 else threads_per_worker(workers))
    results = [job.run(0, blocks[0])]
    if progress is not None:
        progress(1)
    rest = list(range(1, len(blocks)))
    if workers <= 1:
        for index in rest:
            results.append(job.run(index, blocks[index]))
            if progress is not None:
                progress(index + 1)
        return results
    threads = threads_per_worker(workers)
    with process_pool(
        workers,
        initializer=_initialize_job,
        initargs=(job, threads, current_settings()),
    ) as pool:
        for index, result in zip(
            rest, pool.map(_run_block, rest, [blocks[i] for i in rest]), strict=True
        ):
            results.append(result)
            if progress is not None:
                progress(index + 1)
    return results


def block_rows(base_rows: int, total_rows: int, n_jobs: int | None) -> int:
    """Rows per block: `base_rows`, divided among the workers when it pays.

    A test set that fits in one block runs in one block (no pool). A larger one
    is cut into blocks `workers` times smaller, so that `workers` blocks in
    memory at once take about as much as one block in a serial run.
    """

    workers = resolve_jobs(n_jobs)
    if workers <= 1 or total_rows <= base_rows:
        return base_rows
    return max(1, base_rows // workers)


# --------------------------------------------------------------- settings
def current_settings() -> dict[str, Any]:
    """Module settings a spawned worker must share with this process."""

    from gridfm_trustlayer.io import parquet

    return {"scenarios_per_partition": parquet.SCENARIOS_PER_PARTITION}


def apply_settings(settings: dict[str, Any]) -> None:
    from gridfm_trustlayer.io import parquet

    parquet.SCENARIOS_PER_PARTITION = settings["scenarios_per_partition"]
