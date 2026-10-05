"""The config builder: its logic against the CLI validation, and the Jupyter form."""

from __future__ import annotations

from typing import Any

import pytest

from gridfm_trustlayer.cli.config import load_cli_config, resolve_command_config
from gridfm_trustlayer.interactive import (
    build_config,
    save_config,
    to_yaml,
    validate_config,
)


def _dataset(name: str, *, kernel: bool = False) -> dict[str, str]:
    block = {
        "predictions": f"{name}/predictions.parquet",
        "pf_node": f"{name}/bus_data.parquet",
        "pf_edge": f"{name}/branch_data.parquet",
    }
    if kernel:
        block["embeddings"] = f"{name}/bus_embeddings.parquet"
    return block


def _config(**kwargs: Any) -> dict[str, Any]:
    kernel = kwargs.get("cp_type") == "kernel"
    kwargs.setdefault("calibration", _dataset("cal", kernel=kernel))
    kwargs.setdefault("test", _dataset("test", kernel=kernel))
    return build_config(**kwargs)


BOUNDS = {"Vm": {"lower": 0.95, "upper": 1.05}, "line_loading": {"upper": 1.0}}


def test_saved_config_loads_and_each_command_reads_its_dataset(tmp_path) -> None:
    path = save_config(
        _config(
            variables=["Vm", "line_loading"], strat="n_outages", screen_bounds=BOUNDS
        ),
        str(tmp_path / "trust_config.yaml"),
    )
    config = load_cli_config(str(path))

    assert (
        resolve_command_config(config, "compute")["predictions"]
        == "cal/predictions.parquet"
    )
    assert (
        resolve_command_config(config, "calibrate")["predictions"]
        == "test/predictions.parquet"
    )
    assert (
        resolve_command_config(config, "screen")["pf_node"] == "test/bus_data.parquet"
    )
    assert "gridfm_trustlayer screen" in path.read_text(encoding="utf-8")


def test_element_config_drops_kernel_options_and_embeddings() -> None:
    config = _config(
        calibration=_dataset("cal", kernel=True),
        test=_dataset("test", kernel=True),
        k_nn=7,
    )
    validate_config(config)

    assert not {"k_nn", "bandwidth", "distance", "embedding_pooling"} & set(config)
    assert "embeddings" not in config["inputs"]["calibration"]


@pytest.mark.parametrize(
    ("bandwidth", "present", "absent"),
    [("knn", "k_nn", "n_eff_target"), ("n_eff", "n_eff_target", "k_nn")],
)
def test_kernel_writes_only_the_knob_of_the_chosen_bandwidth(
    bandwidth: str, present: str, absent: str
) -> None:
    config = _config(cp_type="kernel", bandwidth=bandwidth, n_eff_target=300.0)
    validate_config(config)

    assert present in config
    assert absent not in config


def test_kernel_with_prediction_bins_is_rejected() -> None:
    with pytest.raises(ValueError, match="pred_bins"):
        validate_config(_config(cp_type="kernel", strat="pred_bins", pred_n_bins=5))


def test_kernel_requires_embeddings_in_both_datasets() -> None:
    config = _config(cp_type="kernel", test=_dataset("test"))
    with pytest.raises(ValueError, match="inputs.test is missing: embeddings"):
        validate_config(config)


def test_bin_counts_follow_the_stratification() -> None:
    config = _config(strat="system_load_bin", system_load_n_bins=4, pred_n_bins=9)
    validate_config(config)

    assert config["system_load_n_bins"] == 4
    assert "pred_n_bins" not in config


def test_output_keys_follow_the_number_of_target_families() -> None:
    single = _config(variables=["Vm", "Va"])["outputs"]
    mixed = _config(variables=["Vm", "line_loading"])["outputs"]

    assert "calibrated_predictions" in single
    assert {"calibrated_predictions_volt", "calibrated_predictions_line"} <= set(mixed)


def test_va_is_calibrated_but_never_screened() -> None:
    config = _config(variables=["Vm", "Va"], screen_bounds=BOUNDS)
    validate_config(config)

    assert set(config["screen_bounds"]) == {"Vm"}
    assert config["commands"]["screen"]["variables"] == ["Vm"]


def test_without_screenable_variables_there_is_no_screen_step() -> None:
    config = _config(variables=["Va"], screen_bounds=BOUNDS)
    validate_config(config)

    assert "screen_bounds" not in config
    assert "screen_dir" not in config["outputs"]
    assert "gridfm_trustlayer screen" not in to_yaml(config)


def test_missing_paths_are_reported() -> None:
    with pytest.raises(ValueError, match="inputs.calibration is missing: pf_node"):
        validate_config(
            _config(calibration={"predictions": "cal/p.parquet", "pf_edge": "e"})
        )


def test_same_file_for_calibration_and_test_is_rejected() -> None:
    with pytest.raises(ValueError, match="same file"):
        validate_config(_config(test=_dataset("cal")))


# ------------------------------------------------------------------ Jupyter form
def _all_widgets(widget: Any) -> list[Any]:
    found = [widget]
    for child in getattr(widget, "children", ()):
        found.extend(_all_widgets(child))
    return found


def test_form_builds_a_valid_config_and_saves_it(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ipywidgets")
    from gridfm_trustlayer.interactive import config_builder

    monkeypatch.chdir(tmp_path)
    ui = config_builder()
    widgets = _all_widgets(ui)

    def field(description: str, index: int = 0) -> Any:
        matches = [x for x in widgets if getattr(x, "description", None) == description]
        return matches[index]

    status = [x for x in widgets if "Not valid yet" in str(getattr(x, "value", ""))]
    assert status, "an empty form must not be valid"
    save_button = field("Save config")
    assert save_button.disabled

    for index, name in enumerate(["cal", "test"]):
        for key, value in _dataset(name).items():
            field(key, index).value = value
    field("Stratify by").value = "n_outages"

    assert not save_button.disabled
    save_button.click()
    config = load_cli_config(str(tmp_path / "trust_config.yaml"))
    assert config["strat"] == "n_outages"
    assert config["inputs"]["test"]["predictions"] == "test/predictions.parquet"


def test_form_hides_prediction_bins_for_kernel() -> None:
    pytest.importorskip("ipywidgets")
    from gridfm_trustlayer.interactive import config_builder

    widgets = _all_widgets(config_builder())
    strat = next(x for x in widgets if getattr(x, "description", None) == "Stratify by")
    method = next(x for x in widgets if getattr(x, "description", None) == "Method")

    strat.value = "pred_bins"
    method.value = "kernel"

    assert "pred_bins" not in [value for _, value in strat.options]
    assert strat.value is None


def test_without_bounds_the_config_only_produces_intervals(tmp_path) -> None:
    config = _config(variables=["Vm", "line_loading"], screen_bounds=None)
    path = save_config(config, str(tmp_path / "intervals_only.yaml"))

    assert "screen_bounds" not in config
    assert "screen_dir" not in config["outputs"]
    assert "commands" not in config
    text = path.read_text(encoding="utf-8")
    assert "gridfm_trustlayer calibrate" in text
    assert "gridfm_trustlayer screen" not in text


def test_form_can_turn_screening_off(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ipywidgets")
    from gridfm_trustlayer.interactive import config_builder

    monkeypatch.chdir(tmp_path)
    widgets = _all_widgets(config_builder())

    def field(description: str, index: int = 0) -> Any:
        return [x for x in widgets if getattr(x, "description", None) == description][
            index
        ]

    for index, name in enumerate(["cal", "test"]):
        for key, value in _dataset(name).items():
            field(key, index).value = value
    field("Screen against operating limits").value = False

    assert field("Vm lower (pu)").layout.display == "none"
    field("Save config").click()
    config = load_cli_config(str(tmp_path / "trust_config.yaml"))
    assert "screen_bounds" not in config
    assert "screen_dir" not in config["outputs"]
    saved = [x for x in widgets if "no screening step" in str(getattr(x, "value", ""))]
    assert saved, "the confirmation should not suggest running screen"


def _form() -> tuple[list[Any], Any]:
    pytest.importorskip("ipywidgets")
    from gridfm_trustlayer.interactive import config_builder

    widgets = _all_widgets(config_builder())

    def field(description: str, index: int = 0) -> Any:
        return [x for x in widgets if getattr(x, "description", None) == description][
            index
        ]

    return widgets, field


def test_form_widgets_do_not_share_a_layout() -> None:
    # Fields are hidden through layout.display: a Layout shared by two widgets
    # would show or hide both together.
    widgets, _ = _form()
    layouts = [id(x.layout) for x in widgets if hasattr(x, "layout")]
    assert len(layouts) == len(set(layouts))


@pytest.mark.parametrize(
    ("strat", "load_bins_shown", "pred_bins_shown"),
    [
        (None, False, False),
        ("n_outages", False, False),
        ("outage_type", False, False),
        ("system_load_bin", True, False),
        ("pred_bins", False, True),
    ],
)
def test_bin_counts_appear_only_for_their_stratification(
    strat: str | None, load_bins_shown: bool, pred_bins_shown: bool
) -> None:
    _, field = _form()
    field("Stratify by").value = "system_load_bin"  # start from a different state
    field("Stratify by").value = strat

    assert (field("Load bins").layout.display != "none") is load_bins_shown
    assert (field("Prediction bins").layout.display != "none") is pred_bins_shown
    assert field("predictions").layout.display != "none"


def test_turning_screening_off_hides_only_the_bounds() -> None:
    _, field = _form()
    field("Screen against operating limits").value = False

    assert field("Vm lower (pu)").layout.display == "none"
    assert field("Line loading upper").layout.display == "none"
    for description in ["predictions", "alpha", "Output directory", "Config file"]:
        assert field(description).layout.display != "none", description


def test_tuning_writes_auto_and_the_tuning_block() -> None:
    config = _config(
        cp_type="kernel",
        bandwidth="n_eff",
        n_eff_target=300.0,
        tune_n_eff_target=True,
        tuning={"folds": 4, "grid": [200, 100]},
    )
    validate_config(config)

    assert config["n_eff_target"] == "auto"
    assert config["tuning"]["folds"] == 4
    assert config["tuning"]["grid"] == [100.0, 200.0]
    assert config["tuning"]["group_by"] == "auto"


@pytest.mark.parametrize(
    ("cp_type", "bandwidth"), [("element", "n_eff"), ("kernel", "knn")]
)
def test_tuning_is_ignored_where_it_does_not_apply(
    cp_type: str, bandwidth: str
) -> None:
    config = _config(cp_type=cp_type, bandwidth=bandwidth, tune_n_eff_target=True)
    validate_config(config)

    assert "tuning" not in config
    assert config.get("n_eff_target") != "auto"


def test_form_tuning_toggle_swaps_the_target_for_the_tuning_options(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    widgets, field = _form()
    field("Method").value = "kernel"
    field("Bandwidth").value = "n_eff"
    tune = field("Tune n_eff_target by cross-validation on the calibration set")

    assert not tune.value  # off by default
    assert _shown(widgets, "n_eff_target")
    assert not _shown(widgets, "Folds")

    tune.value = True
    assert not _shown(widgets, "n_eff_target")
    assert _shown(widgets, "Folds")

    for index, name in enumerate(["cal", "test"]):
        for key, value in _dataset(name, kernel=True).items():
            field(key, index).value = value
    field("Candidates").value = "150, 300"
    field("Save config").click()
    config = load_cli_config(str(tmp_path / "trust_config.yaml"))
    assert config["n_eff_target"] == "auto"
    assert config["tuning"]["grid"] == [150.0, 300.0]


def _shown(widgets: list[Any], description: str) -> bool:
    """A field is shown when neither it nor any enclosing box is hidden."""

    target = next(x for x in widgets if getattr(x, "description", None) == description)

    def visible(node: Any) -> bool:
        if node is target:
            return node.layout.display != "none"
        for child in getattr(node, "children", ()):
            if visible(child):
                return node.layout.display != "none"
        return False

    return visible(widgets[0])


def test_random_search_writes_budget_and_range_instead_of_the_grid() -> None:
    config = _config(
        cp_type="kernel",
        bandwidth="n_eff",
        tune_n_eff_target=True,
        tuning={
            "search": "random",
            "budget": 7,
            "n_eff_range": [80, 900],
            "distances": ["euclidean", "cosine"],
        },
    )
    validate_config(config)

    tuning = config["tuning"]
    assert tuning["search"] == "random"
    assert tuning["budget"] == 7
    assert tuning["n_eff_range"] == [80.0, 900.0]
    assert tuning["distances"] == ["euclidean", "cosine"]
    assert "grid" not in tuning


def test_form_random_search_with_both_distances(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    widgets, field = _form()
    field("Method").value = "kernel"
    field("Bandwidth").value = "n_eff"
    field("Tune n_eff_target by cross-validation on the calibration set").value = True

    assert _shown(widgets, "Candidates")
    assert not _shown(widgets, "Budget (tries)")
    field("Search").value = "random"
    assert not _shown(widgets, "Candidates")
    assert _shown(widgets, "Budget (tries)")
    assert _shown(widgets, "n_eff range")

    field("Also compare distances (euclidean and cosine)").value = True
    field("Budget (tries)").value = 6
    field("n_eff range").value = "150, 1500"
    for index, name in enumerate(["cal", "test"]):
        for key, value in _dataset(name, kernel=True).items():
            field(key, index).value = value
    field("Save config").click()

    tuning = load_cli_config(str(tmp_path / "trust_config.yaml"))["tuning"]
    assert tuning["search"] == "random"
    assert tuning["budget"] == 6
    assert tuning["n_eff_range"] == [150.0, 1500.0]
    assert set(tuning["distances"]) == {"euclidean", "cosine"}


def test_a_blank_n_eff_target_is_written_as_the_default() -> None:
    config = _config(cp_type="kernel")
    validate_config(config)

    assert config["bandwidth"] == "n_eff"
    assert config["n_eff_target"] == 600.0


def test_form_preselects_and_lists_the_default_bandwidth_first() -> None:
    widgets, field = _form()
    field("Method").value = "kernel"
    bandwidth = field("Bandwidth")

    assert bandwidth.value == "n_eff"
    assert list(bandwidth.options)[0] == "n_eff"
    preview = " ".join(str(getattr(x, "value", "")) for x in widgets)
    assert "n_eff_target: 600.0" in preview
