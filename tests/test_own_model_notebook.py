"""The own-model notebook runs from top to bottom, and what its text says holds."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_own_model_notebook_runs_and_its_claims_hold(tmp_path) -> None:
    # The notebook reads ../demo/data and writes nothing; run it from a copy of
    # that layout anyway, so that the repository stays untouched.
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "data").symlink_to(
        EXAMPLES / "demo" / "data", target_is_directory=True
    )
    (tmp_path / "notebooks").mkdir()
    notebook = nbformat.read(EXAMPLES / "notebooks" / "own_model.ipynb", as_version=4)
    nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path / "notebooks")}},
    ).execute()
    printed = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )

    # check_inputs: only the expected warning (the reference bus is never PQ).
    assert printed.count("No problems found.") == 3
    warnings = re.findall(r"^warning: .*$", printed, flags=re.MULTILINE)
    expected = (
        "warning: Vm: 1 elements have no valid calibration value (e.g. ids [0]): "
        "they get no finite bound."
    )
    assert warnings == [expected]

    # Section 6: SCP covers 90%; KCP is close to it, with narrower bounds.
    for variable in ("Vm", "line_loading"):
        match = re.search(
            rf"^{variable}: SCP covers ([\d.]+)% with a mean width of ([\d.]+); "
            rf"KCP ([\d.]+)% with ([\d.]+) \((\d+)% narrower\)\.$",
            printed,
            flags=re.MULTILINE,
        )
        assert match is not None, variable
        scp_coverage, scp_width, kcp_coverage, kcp_width, _ = map(float, match.groups())
        assert 89.0 <= scp_coverage <= 92.0
        assert 89.0 <= kcp_coverage <= 93.0
        assert kcp_width < scp_width

    # Section 7: the bounds clear fewer scenarios, and fewer insecure ones.
    cleared = {
        label: (int(n.replace(",", "")), int(insecure.replace(",", "")))
        for label, n, insecure in re.findall(
            r"^The (point predictions alone|SCP bounds) clear ([\d,]+) of 4,848 "
            r"scenarios as secure; ([\d,]+) of them",
            printed,
            flags=re.MULTILINE,
        )
    }
    point, bounds = cleared["point predictions alone"], cleared["SCP bounds"]
    assert bounds[0] < point[0] / 2
    assert bounds[1] / bounds[0] < point[1] / point[0]
