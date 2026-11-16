"""`TrustLayer`, and the `KernelFallbackWarning` of the kernel method and its tuning."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer import (
    ChannelSet,
    KernelFallbackWarning,
    KernelIntervalCalibrator,
    LoadBins,
    NaiveIntervalCalibrator,
    OutageStrata,
    TrustLayer,
    load_interval_artifacts,
    save_interval_artifact,
)
from gridfm_trustlayer.cli.common import _predict_all
from gridfm_trustlayer.conformal.kernel import predict_intervals_jointly
from gridfm_trustlayer.conformal.tuning import (
    TuningOptions,
    _tune_n_eff_targets,
    tune_n_eff_targets,
)


def _data(
    n: int = 400, *, seed: int = 0, dims: int = 2, first_id: int = 0
) -> dict[str, Any]:
    """Buses (`Vm`, `Va`) and lines (`line_loading`) of `n` scenarios, two strata.

    Errors are three times larger where the first embedding dimension is
    positive, so the kernel has something to localize.
    """

    rng = np.random.default_rng(seed)
    scenario_ids = np.arange(first_id, first_id + n)
    n_outages = rng.integers(1, 3, n)
    embeddings = rng.normal(size=(n, dims)).astype(np.float32)
    scale = np.where(embeddings[:, 0] > 0, 3.0, 1.0)[:, None] * n_outages[:, None]
    common: dict[str, Any] = {
        "scenario_ids": scenario_ids,
        "scenario_columns": {"n_outages": n_outages},
    }

    def pair(names: tuple[str, ...], width: int, entity: str) -> tuple[ChannelSet, ...]:
        predicted = {
            name: rng.normal(size=(n, width)).astype(np.float32) for name in names
        }
        true = {
            name: values + 0.01 * scale * rng.normal(size=values.shape)
            for name, values in predicted.items()
        }
        return tuple(
            ChannelSet(
                entity_ids=np.arange(width),
                entity_name=entity,
                channels={
                    k: np.asarray(v, dtype=np.float32) for k, v in values.items()
                },
                scenario_embeddings=embeddings,
                **common,
            )
            for values in (predicted, true)
        )

    buses, lines = pair(("Vm", "Va"), 4, "bus"), pair(("line_loading",), 3, "branch")
    return {
        "pred": [buses[0], lines[0]],
        "true": [buses[1], lines[1]],
        "n_outages": n_outages,
    }


def _manual(
    data: dict[str, Any], test: dict[str, Any], kernel: dict[str, Any] | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for predictions, truth, new in zip(
        data["pred"], data["true"], test["pred"], strict=True
    ):
        common: dict[str, Any] = {
            "alpha": 0.1,
            "channel_names": tuple(predictions.channels),
            "strat": "n_outages",
        }
        calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator = (
            NaiveIntervalCalibrator.fit(predictions, truth, **common)
            if kernel is None
            else KernelIntervalCalibrator.fit(
                predictions,
                truth,
                embeddings=predictions.scenario_embeddings,
                **kernel,
                **common,
            )
        )
        bounds.update(calibrator.predict_intervals(new))
    return bounds


def _equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first.keys() == second.keys() and all(
        np.array_equal(first[name][side], second[name][side])
        for name in first
        for side in (0, 1)
    )


# ------------------------------------------------------------------ fit / predict


def test_trust_layer_equals_one_calibrator_per_set() -> None:
    data, test = _data(), _data(seed=1, first_id=1000)

    layer = TrustLayer.fit(data["pred"], data["true"], alpha=0.1, strat="n_outages")

    assert list(layer.calibrators) == ["Vm+Va", "line_loading"]
    assert layer.variables == ("Vm", "Va", "line_loading")
    bounds = layer.predict_intervals(test["pred"][::-1])  # any order of the sets
    assert list(bounds) == ["Vm", "Va", "line_loading"]
    assert _equal(bounds, _manual(data, test))
    # Only the lines: only their bounds.
    assert list(layer.predict_intervals(test["pred"][1])) == ["line_loading"]
    assert layer.diagnostics == {}


def test_kernel_trust_layer_equals_one_calibrator_per_set() -> None:
    data, test = _data(), _data(seed=1, first_id=1000)

    layer = TrustLayer.fit(
        data["pred"],
        data["true"],
        alpha=0.1,
        strat="n_outages",
        method="kernel",
        n_eff_target=60,
    )

    bounds = layer.predict_intervals(test["pred"])
    assert _equal(bounds, _manual(data, test, kernel={"n_eff_target": 60}))
    assert set(layer.diagnostics) == {"Vm+Va", "line_loading"}
    assert layer.diagnostics["Vm+Va"]["n_eff"].shape == (400,)


def test_auto_n_eff_target_is_tuned_for_all_sets_at_once() -> None:
    data = _data()
    options = TuningOptions(grid=(40.0, 80.0, 160.0), folds=3)
    groups = pd.Series(np.arange(400) // 2, index=data["pred"][0].scenario_ids)

    layer = TrustLayer.fit(
        data["pred"],
        data["true"],
        alpha=0.1,
        strat="n_outages",
        method="kernel",
        n_eff_target="auto",
        tuning_options=options,
        groups=groups.iloc[::-1],  # aligned by scenario id
    )

    expected = tune_n_eff_targets(
        {
            "Vm+Va": (data["pred"][0], data["true"][0], ("Vm", "Va")),
            "line_loading": (data["pred"][1], data["true"][1], ("line_loading",)),
        },
        alpha=0.1,
        strat="n_outages",
        options=options,
        groups=groups.to_numpy(),
    )
    for key, result in expected.items():
        chosen = layer.tuning[key]
        assert chosen.selected_n_eff_target == result.selected_n_eff_target
        calibrator = layer.calibrators[key]
        assert isinstance(calibrator, KernelIntervalCalibrator)
        assert calibrator.n_eff_target == result.selected_n_eff_target
        assert calibrator.tuning == result.summary()


# ------------------------------------------------------------------ save / load


@pytest.mark.parametrize("method", ["element", "kernel"])
def test_save_and_load_keep_the_bounds_and_the_strata(tmp_path, method) -> None:
    data, test = _data(), _data(seed=1, first_id=1000)
    in_service = pd.DataFrame(
        {"l7": [True, False, True], "l3": [True, True, False]}, index=[10, 11, 12]
    )
    loads = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=range(5))
    bins = LoadBins.fit(loads, 3)
    strata: dict[str, OutageStrata | LoadBins] = {
        "n_outages": OutageStrata.fit(in_service),
        "load": bins,
    }
    kernel_options: dict[str, Any] = {"n_eff_target": 60} if method == "kernel" else {}
    layer = TrustLayer.fit(
        data["pred"],
        data["true"],
        alpha=0.1,
        strat="n_outages",
        strata=strata,
        method=method,
        **kernel_options,
    )
    path = tmp_path / "layer.trustlayer"

    layer.save(path)
    loaded = TrustLayer.load(path)

    assert _equal(
        loaded.predict_intervals(test["pred"]), layer.predict_intervals(test["pred"])
    )
    outages = loaded.strata["n_outages"]
    assert isinstance(outages, OutageStrata)
    assert outages.branch_ids is not None
    assert outages.branch_ids.tolist() == ["l7", "l3"]
    reordered = in_service[["l3", "l7"]]  # matched by label, not position
    assert outages.n_outages(reordered).tolist() == [0, 1, 1]
    loaded_bins = loaded.strata["load"]
    assert isinstance(loaded_bins, LoadBins)
    np.testing.assert_array_equal(loaded_bins.edges, bins.edges)
    # The calibrators are an ordinary artifact.
    assert list(load_interval_artifacts(str(path))) == ["Vm+Va", "line_loading"]


def test_load_reads_a_single_calibrator_artifact(tmp_path) -> None:
    data, test = _data(), _data(seed=1, first_id=1000)
    calibrator = NaiveIntervalCalibrator.fit(
        data["pred"][1], data["true"][1], alpha=0.1, channel_names=("line_loading",)
    )
    path = tmp_path / "lines.trustlayer"
    save_interval_artifact(calibrator, str(path))
    assert "artifacts" not in json.loads(path.read_text())  # the plain format

    layer = TrustLayer.load(path)

    assert list(layer.calibrators) == ["line"]
    assert layer.strata == {}
    expected = calibrator.predict_intervals(test["pred"][1])
    assert _equal(layer.predict_intervals(test["pred"][1]), expected)


def test_load_bins_edges_go_to_the_calibrators_for_system_load_bin() -> None:
    data = _data()
    bins = LoadBins(np.array([1.5], dtype=np.float32))
    for sets in (data["pred"], data["true"]):
        for channel_set in sets:
            channel_set.scenario_columns["system_load_bin"] = data["n_outages"] - 1

    layer = TrustLayer.fit(
        data["pred"],
        data["true"],
        alpha=0.1,
        strat="system_load_bin",
        strata={"system_load_bin": bins},
    )

    for calibrator in layer.calibrators.values():
        assert calibrator.system_load_bin_edges is not None
        np.testing.assert_array_equal(calibrator.system_load_bin_edges, bins.edges)
    with pytest.raises(ValueError, match="needs its LoadBins"):
        TrustLayer.fit(data["pred"], data["true"], alpha=0.1, strat="system_load_bin")


# ------------------------------------------------------------------ refusals


def _refusal_cases() -> list[tuple[Any, str]]:
    def shifted(data: dict[str, Any]) -> dict[str, Any]:
        truth = data["true"][1]
        moved = ChannelSet(
            scenario_ids=truth.scenario_ids + 1,
            entity_ids=truth.entity_ids,
            entity_name=truth.entity_name,
            channels=truth.channels,
        )
        return {**data, "true": [data["true"][0], moved]}

    return [
        (lambda d: ({**d, "method": "bayes"}), "method must be one of"),
        (lambda d: ({**d, "n_eff_target": 60}), "need method='kernel'"),
        (
            lambda d: ({**d, "method": "kernel", "n_eff_target": "big"}),
            "a number or 'auto'",
        ),
        (lambda d: ({**d, "true": d["true"][:1]}), "pass them in pairs"),
        (
            lambda d: ({**d, "pred": [d["pred"][0], d["pred"][0]]}),
            "more than one set",
        ),
        (shifted, "scenario"),
        (lambda d: ({**d, "strata": {"n_outages": "yes"}}), "OutageStrata or LoadBins"),
        (lambda d: ({**d, "strat": "pred_bins"}), "not supported"),
    ]


@pytest.mark.parametrize(("change", "message"), _refusal_cases())
def test_fit_refuses_inputs_that_do_not_fit(change, message) -> None:
    data = _data(n=60)
    settings = change(
        {"pred": data["pred"], "true": data["true"], "strat": "n_outages"}
    )
    predictions, truth = settings.pop("pred"), settings.pop("true")

    with pytest.raises(ValueError, match=message):
        TrustLayer.fit(predictions, truth, alpha=0.1, **settings)


def test_predict_refuses_sets_it_cannot_route() -> None:
    data = _data(n=60)
    layer = TrustLayer.fit(data["pred"], data["true"], alpha=0.1, strat="n_outages")
    other = ChannelSet(
        scenario_ids=np.arange(3),
        entity_ids=np.arange(2),
        entity_name="bus",
        channels={"Pg": np.zeros((3, 2), dtype=np.float32)},
        scenario_columns={"n_outages": np.ones(3, dtype=np.int64)},
    )

    with pytest.raises(ValueError, match="No calibrator for the variables"):
        layer.predict_intervals(other)
    with pytest.raises(ValueError, match="Two sets hold"):
        layer.predict_intervals([data["pred"][1], data["pred"][1]])


# ------------------------------------------------------------------ fallback warning


def _far_apart(n: int, seed: int, first_id: int = 0) -> dict[str, Any]:
    """50-dimensional embeddings: every calibration scenario is far from a new one."""

    return _data(n, seed=seed, dims=50, first_id=first_id)


def test_predict_warns_when_most_scenarios_fall_back() -> None:
    data, test = _far_apart(300, 0), _far_apart(100, 1, first_id=1000)
    calibrator = KernelIntervalCalibrator.fit(
        data["pred"][1],
        data["true"][1],
        alpha=0.1,
        channel_names=("line_loading",),
        strat="n_outages",
        embeddings=data["pred"][1].scenario_embeddings,
        n_eff_target=20,
    )

    with pytest.warns(KernelFallbackWarning) as caught:
        calibrator.predict_intervals(test["pred"][1])

    message = str(caught[0].message)
    assert "100% of the 100 scenarios fall back" in message
    assert "weights must sum to at least 19" in message
    assert "Try a larger n_eff_target" in message
    assert Path(caught[0].filename).name == "test_layer.py"  # the caller's line
    assert calibrator.diagnostics["kernel_fallback"].all()

    with pytest.warns(KernelFallbackWarning):
        predict_intervals_jointly([calibrator], [test["pred"][1]])
    # The CLI reports it in its notes instead.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _predict_all({"line": calibrator}, {"line": test["pred"][1]})


def test_predict_names_strata_absent_from_the_calibration_data() -> None:
    data, test = _data(), _data(seed=1, first_id=1000)
    for channel_set in test["pred"]:
        channel_set.scenario_columns["n_outages"] = np.full(400, 7)
    calibrator = KernelIntervalCalibrator.fit(
        data["pred"][1],
        data["true"][1],
        alpha=0.1,
        channel_names=("line_loading",),
        strat="n_outages",
        embeddings=data["pred"][1].scenario_embeddings,
        n_eff_target=60,
    )

    with pytest.warns(KernelFallbackWarning, match="400 of them are in strata absent"):
        calibrator.predict_intervals(test["pred"][1])


def test_no_warning_when_the_kernel_localizes() -> None:
    data, test = _data(), _data(seed=1, first_id=1000)
    layer = TrustLayer.fit(
        data["pred"],
        data["true"],
        alpha=0.1,
        strat="n_outages",
        method="kernel",
        n_eff_target=60,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layer.predict_intervals(test["pred"])
    shares = [d["kernel_fallback"].mean() for d in layer.diagnostics.values()]
    assert max(shares) < 0.5


def test_tuning_warns_when_the_chosen_setting_falls_back() -> None:
    data = _far_apart(300, 0)
    families = {"lines": (data["pred"][1], data["true"][1], ("line_loading",))}
    options = TuningOptions(grid=(10.0, 20.0), folds=3)

    with pytest.warns(
        KernelFallbackWarning, match=r"Kernel tuning \(lines\)"
    ) as caught:
        results = tune_n_eff_targets(
            families, alpha=0.1, strat="n_outages", options=options
        )
    assert Path(caught[0].filename).name == "test_layer.py"
    result = results["lines"]
    assert result.selected_fallback_share >= 0.5
    note = result.fallback_note("lines")
    assert note is not None
    assert note.startswith("kernel tuning (lines): at the chosen n_eff_target")
    # The CLI prints the note instead.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _tune_n_eff_targets(families, alpha=0.1, strat="n_outages", options=options)
