"""Known values: set-points are inputs, not predictions.

In PowerFlow, Vm at PV buses and Va at the reference bus are given to the model.
They are left out of the fit (a residual of zero would shrink the bounds of
buses that are PV only in some scenarios), out of every coverage, width and
screening metric, and counted in `n_known` instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.cli import main, reports
from tests.conformal_reference import conformal_quantile
from tests.test_streaming import N_SCENARIOS, _config

ELEMENT = ["cp_type: element"]


def _run(config: Path, *commands: str) -> None:
    for command in commands:
        assert main([command, "--config", str(config)]) == 0


def test_known_values_are_counted_apart_in_every_report(tmp_path) -> None:
    # Bus 0 is the reference bus (Va known), bus 1 is PV (Vm known), bus 2 is PQ.
    config = _config(tmp_path, "run", ELEMENT)
    _run(config, "compute", "calibrate", "screen")
    coverage_dir = tmp_path / "run" / "calibrated" / "coverage"
    screen_dir = tmp_path / "run" / "screen"

    elements = pd.read_csv(coverage_dir / "coverage_by_element_voltage.csv")
    elements = elements.set_index(["variable", "bus"])
    assert elements.loc[("Vm", 1), "n_known"] == N_SCENARIOS
    assert elements.loc[("Vm", 1), "n_values"] == 0
    assert np.isnan(elements.loc[("Vm", 1), "coverage"])
    assert elements.loc[("Va", 0), "n_known"] == N_SCENARIOS
    assert elements.loc[("Va", 0), "n_values"] == 0
    unknown = elements.drop([("Vm", 1), ("Va", 0)])
    assert (unknown["n_known"] == 0).all()
    assert (unknown["n_values"] == N_SCENARIOS).all()

    by_stratum = pd.read_csv(coverage_dir / "coverage_by_stratum.csv")
    overall = by_stratum[by_stratum["stratum"].eq("all")].set_index("variable")
    assert overall.loc["Vm", "n_known"] == N_SCENARIOS
    assert overall.loc["Vm", "n_values"] == 2 * N_SCENARIOS
    assert overall.loc["Va", "n_known"] == N_SCENARIOS
    assert overall.loc["line_loading", "n_known"] == 0
    summary = (coverage_dir / "summary.md").read_text(encoding="utf-8")
    assert "Known values are not counted" in summary
    assert "values of `Vm` at PV buses" in summary
    assert "values of `Va` at the reference bus" in summary

    detail = pd.read_parquet(screen_dir / "screening_detail_voltage.parquet")
    assert 1 not in set(detail["bus"].tolist())
    screening = pd.read_csv(screen_dir / "screening_by_stratum.csv")
    vm = screening[screening["variable"].eq("Vm") & screening["side"].eq("any")]
    assert vm["n_known"].tolist() == [N_SCENARIOS]
    assert vm["n_values"].tolist() == [2 * N_SCENARIOS]
    cells = vm[["caught", "missed", "false_alarms", "cleared"]]
    assert int(cells.to_numpy().sum()) == 2 * N_SCENARIOS
    assert "Known values are not counted" in (screen_dir / "summary.md").read_text(
        encoding="utf-8"
    )


def test_known_values_are_left_out_of_the_fit(tmp_path) -> None:
    config = _config(tmp_path, "run", ELEMENT)
    # Bus 2 is PV in the even calibration scenarios only: its Vm there is a set-point.
    predictions_path = tmp_path / "calibration" / "predictions.parquet"
    predictions = pd.read_parquet(predictions_path)
    sometimes_pv = predictions["bus"].eq(2) & predictions["scenario"].mod(2).eq(0)
    predictions.loc[sometimes_pv, ["PV", "PQ"]] = [1.0, 0.0]
    predictions.to_parquet(predictions_path, index=False)
    _run(config, "compute", "calibrate")

    bus_2 = predictions[predictions["bus"].eq(2)].sort_values("scenario")
    residuals = (bus_2["Vm_target"] - bus_2["Vm_pred"]).to_numpy()
    odd = bus_2["scenario"].mod(2).eq(1).to_numpy()
    expected = conformal_quantile(residuals[odd][:, None], 0.9)[0]
    # With the set-points (zero residuals) in, the quantile would be another.
    with_known = np.where(odd, residuals, 0.0)
    assert not np.isclose(conformal_quantile(with_known[:, None], 0.9)[0], expected)

    calibrated = tmp_path / "run" / "calibrated"
    frame = pd.read_parquet(next(calibrated.glob("*vm_va*")))
    rows = frame[frame["bus"].eq(2)]
    offsets = (rows["Vm_pi_upper"] - rows["Vm_pred"]).to_numpy(dtype=float)
    np.testing.assert_allclose(offsets, expected, rtol=1e-5, atol=1e-6)


def test_confusion_figure_puts_each_count_in_its_cell() -> None:
    # (alarm, true violation) on the upper side: 1 caught, 2 false alarms,
    # 3 missed and 4 cleared.
    pairs = [(1, 1)] + [(1, 0)] * 2 + [(0, 1)] * 3 + [(0, 0)] * 4
    detail = pd.DataFrame(
        {
            "variable": "line_loading",
            "pred_lower_violation": False,
            "true_lower_violation": False,
            "pred_upper_violation": [bool(p) for p, _ in pairs],
            "true_upper_violation": [bool(t) for _, t in pairs],
        }
    )
    counts = reports.side_counts(detail).sum().to_frame().T
    counts["variable"] = "line_loading"
    (row,) = [r for _, r in reports.screening_long(counts, ["variable"]).iterrows()]

    assert reports.CONFUSION_ROWS == ("true violation", "no violation")
    assert reports.CONFUSION_COLUMNS[0].startswith("alarm")
    assert reports.CONFUSION_COLUMNS[1] == "no alarm"
    assert reports.confusion_cells(row) == [
        [("caught", 1), ("missed", 3)],
        [("false alarm", 2), ("cleared", 4)],
    ]
