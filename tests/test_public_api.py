"""The top-level Python API and the README example that documents it."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

import gridfm_trustlayer
from gridfm_trustlayer import (
    ChannelSet,
    KernelIntervalCalibrator,
    NaiveIntervalCalibrator,
    load_interval_artifact,
    load_interval_artifacts,
    save_interval_artifact,
)

README = Path(__file__).resolve().parents[1] / "README.md"


def _readme_python_example() -> str:
    section = README.read_text(encoding="utf-8").split("## Python API", 1)[1]
    match = re.search(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert match is not None, "README has no python block under '## Python API'"
    return match.group(1)


def test_public_names_are_importable_from_the_top_level() -> None:
    for name in gridfm_trustlayer.__all__:
        assert hasattr(gridfm_trustlayer, name), name


def test_readme_python_example_runs_and_reaches_target_coverage(capsys) -> None:
    namespace: dict[str, object] = {}
    # Running the documented example verbatim is the point of this test.
    exec(compile(_readme_python_example(), str(README), "exec"), namespace)  # noqa: S102

    printed = capsys.readouterr().out
    coverage = float(printed.split("coverage:")[1])
    assert abs(coverage - 0.9) < 0.02
    assert isinstance(namespace["calibrator"], NaiveIntervalCalibrator)


def test_fitted_calibrator_survives_an_artifact_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(1)
    predicted = rng.normal(size=(200, 3)).astype(np.float32)
    true = predicted + rng.normal(scale=0.1, size=predicted.shape).astype(np.float32)
    common: dict[str, Any] = {
        "scenario_ids": np.arange(200, dtype=np.int64),
        "entity_ids": np.arange(3, dtype=np.int64),
        "entity_name": "bus",
    }
    calibrator = NaiveIntervalCalibrator.fit(
        ChannelSet(channels={"Vm": predicted}, **common),
        ChannelSet(channels={"Vm": true}, **common),
        alpha=0.1,
        channel_names=("Vm",),
    )

    path = tmp_path / "vm.trustlayer"
    save_interval_artifact(calibrator, str(path))
    loaded = load_interval_artifact(str(path))

    assert isinstance(loaded, NaiveIntervalCalibrator)
    assert not isinstance(loaded, KernelIntervalCalibrator)
    np.testing.assert_array_equal(
        loaded.upper_residual_quantiles["Vm"], calibrator.upper_residual_quantiles["Vm"]
    )
    by_family = load_interval_artifacts(str(path))
    assert list(by_family) == ["volt"]
    volt = by_family["volt"]
    assert isinstance(volt, NaiveIntervalCalibrator)
    np.testing.assert_array_equal(
        volt.upper_residual_quantiles["Vm"],
        calibrator.upper_residual_quantiles["Vm"],
    )


PYTHON_API_PAGE = Path(__file__).resolve().parents[1] / "docs/user-guide/python-api.md"


def _runnable_doc_examples() -> list[str]:
    # Plain ```python blocks are self-contained examples on synthetic data and are
    # run here; blocks with a title (```python title="...") need the user's own
    # files or are fragments, and are only rendered.
    text = PYTHON_API_PAGE.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def test_python_api_page_examples_run_and_print_what_the_page_shows(capsys) -> None:
    examples = _runnable_doc_examples()
    assert len(examples) == 2

    for example in examples:
        exec(compile(example, str(PYTHON_API_PAGE), "exec"), {})  # noqa: S102
    printed = capsys.readouterr().out

    coverage = float(printed.split("coverage:")[1].split()[0])
    assert abs(coverage - 0.9) < 0.02
    # The kernel comparison output is quoted verbatim on the page.
    page = PYTHON_API_PAGE.read_text(encoding="utf-8")
    for line in printed.splitlines()[1:]:
        assert line in page, line
