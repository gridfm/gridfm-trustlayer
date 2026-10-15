"""The demo notebook runs from top to bottom, and the results the docs quote hold."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def test_demo_notebook_runs_and_the_quoted_results_hold(tmp_path) -> None:
    (tmp_path / "data").symlink_to(DEMO / "data", target_is_directory=True)
    notebook = nbformat.read(DEMO / "demo.ipynb", as_version=4)
    nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    printed = "\n".join(
        output.get("text", "") + output.get("data", {}).get("text/markdown", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    # The bandwidth solve used to print an overflow warning here.
    assert "overflow encountered in divide" not in printed

    # Quoted in docs/getting-started/demo.md and the README of the demo.
    assert "coverage is 69% at N-5" in printed
    assert "100% at N-1, with bounds 250 times wider" in printed
    assert "lets 110 of the 806 line overloads through; the bounds catch 85" in printed
    assert "`n_eff_target` = 400" in printed
    assert "(1,212 to 2,530 per level)" in printed
    assert "`element` catches 85 and the kernel 103" in printed
    assert "(330 against 473)" in printed
    assert "exceeded 21% of the time with `element`" in printed
    assert "barely help (18%)" in printed
    assert "coverage drops to 72% at N-5" in printed
    assert "brings it down to 5%" in printed
    status = pd.read_csv(tmp_path / "outputs" / "screen" / "status_by_stratum.csv")
    overall = status[
        status["level"].eq("scenario")
        & status["variable"].eq("all")
        & status["stratum"].eq("all")
    ]
    counts = {
        (row.method, row.status): (int(row.n), int(row.truly_violated))
        for row in overall.itertuples()
    }
    assert counts[("interval", "secure")] == (3710, 7)
    assert counts[("interval", "uncertain")][0] == 473
    assert counts[("interval", "insecure")] == (665, 665 - 14)
    assert counts[("point", "secure")] == (4135, 74)
    assert counts[("point", "insecure")] == (713, 713 - 32)
    applied = pd.read_csv(tmp_path / "outputs" / "applied" / "status" / "scenarios.csv")
    assert applied["status"].value_counts().to_dict() == {
        "secure": 221,
        "insecure": 48,
        "uncertain": 25,
    }
