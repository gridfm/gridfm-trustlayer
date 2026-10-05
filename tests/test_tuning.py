"""Cross-validated choice of n_eff_target on the calibration set (`n_eff_target: auto`)."""

from __future__ import annotations

import json
import textwrap

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer import (
    ChannelSet,
    KernelIntervalCalibrator,
    load_interval_artifact,
)
from gridfm_trustlayer.cli import main
from gridfm_trustlayer.cli.config import (
    load_cli_config,
    resolve_command_config,
    validate_command_config,
)
from gridfm_trustlayer.conformal.tuning import (
    REASON_NO_CANDIDATE,
    REASON_SMALLEST_WIDTH,
    TuningOptions,
    assign_folds,
    candidate_settings,
    select_candidate,
    tune_n_eff_target,
)

# ------------------------------------------------------------------ options


def test_default_options() -> None:
    options = TuningOptions.from_config(None)
    assert options.folds == 5
    assert options.group_by == "auto"
    assert options.coverage_tolerance == "auto"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"folds": 1}, "folds must be an integer >= 2"),
        ({"grid": []}, "non-empty list"),
        ({"grid": [0.5, 100]}, ">= 1"),
        ({"group_by": "region"}, "group_by must be one of"),
        ({"coverage_tolerance": 1.5}, r"\[0, 1\)"),
        ({"mystery": 1}, "Unsupported keys under tuning"),
    ],
)
def test_invalid_options_are_rejected(raw: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TuningOptions.from_config(raw)


def test_grid_is_sorted_and_deduplicated() -> None:
    assert TuningOptions.from_config({"grid": [400, 100, 400]}).grid == (100.0, 400.0)


# ------------------------------------------------------------------ folds and selection


def test_folds_keep_groups_together_and_are_balanced() -> None:
    groups = np.repeat(np.arange(50), 4)  # 50 load profiles x 4 topologies
    fold = assign_folds(len(groups), 5, groups=groups, seed=3)

    for group in np.unique(groups):
        assert len(set(fold[groups == group])) == 1
    assert sorted(np.bincount(fold).tolist()) == [40] * 5
    np.testing.assert_array_equal(
        fold, assign_folds(len(groups), 5, groups=groups, seed=3)
    )


def test_too_few_groups_for_the_folds_is_an_error() -> None:
    with pytest.raises(ValueError, match="Cannot split 3 groups into 5 folds"):
        assign_folds(6, 5, groups=np.array([0, 0, 1, 1, 2, 2]))


def _candidates(rows: list[tuple[float, str, float, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["n_eff_target", "distance", "relative_width", "eligible"]
    )


def test_selection_picks_the_narrowest_eligible_candidate() -> None:
    table = _candidates(
        [
            (100, "euclidean", 0.9, False),
            (200, "cosine", 0.95, True),
            (400, "euclidean", 0.97, True),
        ]
    )
    assert select_candidate(table, 5000, "euclidean") == (
        200.0,
        "cosine",
        REASON_SMALLEST_WIDTH,
    )


def test_selection_breaks_ties_towards_the_larger_target() -> None:
    table = _candidates(
        [(200, "euclidean", 0.95, True), (400, "euclidean", 0.95, True)]
    )
    assert select_candidate(table, 5000, "euclidean")[0] == 400.0


def test_selection_falls_back_to_uniform_weights_when_nothing_qualifies() -> None:
    table = _candidates([(100, "cosine", 0.9, False), (200, "cosine", 0.95, False)])
    assert select_candidate(table, 5000, "euclidean") == (
        5000.0,
        "euclidean",
        REASON_NO_CANDIDATE,
    )


# ------------------------------------------------------------------ candidates


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"distances": ["manhattan"]}, "unsupported values"),
        ({"distances": []}, "non-empty list"),
        ({"search": "bayesian"}, "search must be one of"),
        ({"budget": 0}, "budget must be an integer >= 1"),
        ({"n_eff_range": [500, 100]}, "low <= high"),
        ({"n_eff_range": [100]}, "low <= high"),
    ],
)
def test_invalid_search_options_are_rejected(raw: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TuningOptions.from_config(raw)


def test_grid_search_tries_every_combination() -> None:
    options = TuningOptions(grid=(100.0, 400.0), distances=("euclidean", "cosine"))
    assert candidate_settings(options, "euclidean") == [
        (100.0, "cosine"),
        (100.0, "euclidean"),
        (400.0, "cosine"),
        (400.0, "euclidean"),
    ]


def test_without_distances_only_the_config_distance_is_used() -> None:
    options = TuningOptions(grid=(100.0, 400.0))
    assert {d for _, d in candidate_settings(options, "cosine")} == {"cosine"}


def test_random_search_draws_the_budget_within_range() -> None:
    options = TuningOptions(
        search="random",
        budget=8,
        n_eff_range=(50.0, 2000.0),
        distances=("euclidean", "cosine"),
        seed=4,
    )
    settings = candidate_settings(options, "euclidean")

    assert len(settings) == len(set(settings)) == 8
    assert all(50 <= target <= 2000 for target, _ in settings)
    assert {d for _, d in settings} <= {"euclidean", "cosine"}
    assert settings == candidate_settings(options, "euclidean")  # seeded
    # Log-uniform draws are not restricted to grid values.
    assert not {t for t, _ in settings} <= set(options.grid)


def test_random_search_range_defaults_to_the_grid_span() -> None:
    options = TuningOptions(search="random", budget=5, grid=(150.0, 900.0))
    assert all(150 <= t <= 900 for t, _ in candidate_settings(options, "euclidean"))


def test_random_budget_is_capped_by_the_size_of_the_space() -> None:
    options = TuningOptions(search="random", budget=50, n_eff_range=(10.0, 12.0))
    assert len(candidate_settings(options, "euclidean")) == 3  # 10, 11, 12


# ------------------------------------------------------------------ cross-validation


def _synthetic(n: int = 600, seed: int = 0) -> tuple[ChannelSet, ChannelSet]:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, 3)).astype(np.float32)
    strata = rng.integers(0, 2, n)
    predicted = rng.normal(size=(n, 4)).astype(np.float32)
    scale = (0.1 + np.abs(embeddings[:, :1])).astype(np.float32)
    true = predicted + rng.normal(size=predicted.shape).astype(np.float32) * scale

    def channel_set(values: np.ndarray) -> ChannelSet:
        return ChannelSet(
            scenario_ids=np.arange(n),
            entity_ids=np.arange(4),
            entity_name="bus",
            channels={"Vm": values},
            scenario_columns={"n_outages": strata},
            scenario_embeddings=embeddings,
        )

    return channel_set(predicted), channel_set(true)


def test_cross_validation_reports_every_candidate_and_check() -> None:
    predictions, truth = _synthetic()
    options = TuningOptions(folds=3, grid=(20.0, 50.0, 100.0, 200.0))
    result = tune_n_eff_target(
        predictions,
        truth,
        alpha=0.2,
        channel_names=("Vm",),
        strat="n_outages",
        options=options,
    )

    assert result.candidates["n_eff_target"].tolist() == list(options.grid)
    assert result.candidates["selected"].sum() == (
        1 if result.reason == REASON_SMALLEST_WIDTH else 0
    )
    assert set(result.details["scope"]) == {"overall", "n_outages=0", "n_outages=1"}
    assert (result.details["coverage"].between(0, 1)).all()
    if result.reason == REASON_SMALLEST_WIDTH:
        assert result.selected_n_eff_target in options.grid
    json.dumps(result.summary())  # stored in the artifact as JSON


def test_cross_validation_compares_both_distances() -> None:
    predictions, truth = _synthetic()
    result = tune_n_eff_target(
        predictions,
        truth,
        alpha=0.2,
        channel_names=("Vm",),
        strat="n_outages",
        options=TuningOptions(
            folds=3, grid=(50.0, 200.0), distances=("euclidean", "cosine")
        ),
    )

    assert set(result.candidates["distance"]) == {"euclidean", "cosine"}
    assert len(result.candidates) == 4
    assert set(result.details["distance"]) == {"euclidean", "cosine"}
    assert result.selected_distance in {"euclidean", "cosine"}
    assert result.summary()["selected_distance"] == result.selected_distance


def test_cross_validation_is_deterministic_for_a_seed() -> None:
    predictions, truth = _synthetic()
    options = TuningOptions(folds=3, grid=(50.0, 200.0), seed=7)
    first, second = (
        tune_n_eff_target(
            predictions, truth, alpha=0.2, channel_names=("Vm",), options=options
        )
        for _ in range(2)
    )
    pd.testing.assert_frame_equal(first.candidates, second.candidates)


def test_a_strict_tolerance_can_reject_every_candidate() -> None:
    # With zero tolerance, any stratum whose cross-validated coverage lands even
    # slightly below 1 - alpha disqualifies the candidate. On this data both
    # candidates miss in some stratum, so the weights become uniform: the target
    # is set to the largest stratum (equivalent to cp_type element).
    predictions, truth = _synthetic(seed=0)
    result = tune_n_eff_target(
        predictions,
        truth,
        alpha=0.2,
        channel_names=("Vm",),
        strat="n_outages",
        options=TuningOptions(folds=3, grid=(100.0, 200.0), coverage_tolerance=0.0),
    )

    assert result.reason == REASON_NO_CANDIDATE
    assert not result.candidates["eligible"].any()
    assert not result.candidates["selected"].any()
    largest_stratum = int(np.bincount(predictions.scenario_columns["n_outages"]).max())
    assert result.selected_n_eff_target == largest_stratum


# ------------------------------------------------------------------ config


def _kernel_config(tmp_path, extra: str) -> dict:
    path = tmp_path / "trust_config.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            version: 1
            task_name: PowerFlow
            cp_type: kernel
            alpha: 0.1
            variables: [Vm]
            inputs:
              calibration:
                predictions: cal.parquet
                embeddings: cal_emb.parquet
                pf_node: cal_bus.parquet
                pf_edge: cal_branch.parquet
              test:
                predictions: test.parquet
                embeddings: test_emb.parquet
                pf_node: test_bus.parquet
                pf_edge: test_branch.parquet
            """
        )
        + textwrap.dedent(extra),
        encoding="utf-8",
    )
    return load_cli_config(str(path))


def _validate(config: dict) -> None:
    for command in ("compute", "calibrate", "screen"):
        validate_command_config(command, resolve_command_config(config, command))


def test_auto_is_accepted_with_kernel_and_n_eff(tmp_path) -> None:
    _validate(
        _kernel_config(
            tmp_path,
            """\
            bandwidth: n_eff
            n_eff_target: auto
            tuning:
              folds: 4
              grid: [100, 300]
            """,
        )
    )


def test_auto_requires_the_n_eff_bandwidth(tmp_path) -> None:
    with pytest.raises(
        ValueError, match="requires cp_type: kernel and bandwidth: n_eff"
    ):
        _validate(_kernel_config(tmp_path, "bandwidth: knn\nn_eff_target: auto\n"))


def test_tuning_block_without_auto_is_rejected(tmp_path) -> None:
    config = _kernel_config(
        tmp_path, "bandwidth: n_eff\nn_eff_target: 200\ntuning:\n  folds: 3\n"
    )
    with pytest.raises(ValueError, match="only applies with n_eff_target: auto"):
        _validate(config)


def test_candidates_below_n_eff_min_are_rejected(tmp_path) -> None:
    config = _kernel_config(
        tmp_path,
        "bandwidth: n_eff\nn_eff_target: auto\nn_eff_min: 50\ntuning:\n  grid: [20, 100]\n",
    )
    with pytest.raises(ValueError, match="below n_eff_min"):
        _validate(config)


def test_random_range_below_n_eff_min_is_rejected(tmp_path) -> None:
    config = _kernel_config(
        tmp_path,
        "bandwidth: n_eff\nn_eff_target: auto\nn_eff_min: 50\n"
        "tuning:\n  search: random\n  n_eff_range: [20, 400]\n",
    )
    with pytest.raises(ValueError, match="tuning.n_eff_range contains values below"):
        _validate(config)


# ------------------------------------------------------------------ CLI end to end


def _write_kernel_dataset(directory, n: int, seed: int) -> dict[str, str]:
    """Two buses, one branch; errors scale with the first embedding coordinate.

    Pairs of scenarios share a load profile (load_scenario_idx), as datakit
    writes when it generates several topologies per load scenario.
    """

    rng = np.random.default_rng(seed)
    scenario = np.repeat(np.arange(n), 2)
    bus = np.tile([0, 1], n)
    embedding = rng.normal(size=(n, 2))
    vm_pred = 1.0 + 0.02 * rng.normal(size=2 * n)
    scale = 0.002 + 0.01 * np.abs(np.repeat(embedding[:, 0], 2))
    vm_true = vm_pred + scale * rng.normal(size=2 * n)
    va = 0.05 * rng.normal(size=2 * n)
    paths = {
        name: str(directory / f"{name}.parquet")
        for name in ("pred", "emb", "bus", "branch")
    }
    pd.DataFrame(
        {
            "scenario": scenario,
            "bus": bus,
            "Vm_target": vm_true,
            "Va_target": va,
            "Vm_pred": vm_pred,
            "Va_pred": va,
            "PQ": np.tile([0, 1], n),
            "PV": np.tile([1, 0], n),
            "REF": 0,
        }
    ).to_parquet(paths["pred"], index=False)
    pd.DataFrame(
        {
            "scenario": scenario,
            "bus": bus,
            "emb_0": np.repeat(embedding[:, 0], 2),
            "emb_1": np.repeat(embedding[:, 1], 2),
        }
    ).to_parquet(paths["emb"], index=False)
    pd.DataFrame(
        {
            "scenario": scenario,
            "load_scenario_idx": scenario // 2,
            "bus": bus,
            "Pd": 10.0,
            "Qd": 0.0,
            "Vm": vm_true,
            "Va": va,
            "Pg": np.tile([1.0, 0.0], n),
            "Qg": 0.0,
            "PQ": np.tile([0, 1], n),
            "PV": np.tile([1, 0], n),
            "REF": 0,
            "vn_kv": 100.0,
        }
    ).to_parquet(paths["bus"], index=False)
    pd.DataFrame(
        {
            "scenario": np.arange(n),
            "idx": 0,
            "from_bus": 0,
            "to_bus": 1,
            "rate_a": 10.0,
            "br_status": 1.0,
            **{
                key: value
                for key, value in [
                    ("Yff_r", 1.0),
                    ("Yff_i", 0.0),
                    ("Yft_r", -1.0),
                    ("Yft_i", 0.0),
                    ("Ytf_r", -1.0),
                    ("Ytf_i", 0.0),
                    ("Ytt_r", 1.0),
                    ("Ytt_i", 0.0),
                ]
            },
        }
    ).to_parquet(paths["branch"], index=False)
    return paths


def test_compute_tunes_n_eff_target_and_records_it(tmp_path, capsys) -> None:
    (tmp_path / "cal").mkdir()
    (tmp_path / "test").mkdir()
    cal = _write_kernel_dataset(tmp_path / "cal", n=300, seed=1)
    test = _write_kernel_dataset(tmp_path / "test", n=200, seed=2)
    artifact = tmp_path / "out" / "kernel.trustlayer"
    config_path = tmp_path / "trust_config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            version: 1
            task_name: PowerFlow
            cp_type: kernel
            alpha: 0.2
            variables: [Vm]
            bandwidth: n_eff
            n_eff_target: auto
            n_eff_min: 5
            tuning:
              folds: 3
              search: random
              budget: 4
              n_eff_range: [20, 100]
              distances: [euclidean, cosine]
            inputs:
              calibration:
                predictions: {cal["pred"]}
                embeddings: {cal["emb"]}
                pf_node: {cal["bus"]}
                pf_edge: {cal["branch"]}
              test:
                predictions: {test["pred"]}
                embeddings: {test["emb"]}
                pf_node: {test["bus"]}
                pf_edge: {test["branch"]}
            outputs:
              artifact: {artifact}
              calibrated_predictions: {tmp_path / "out" / "calibrated.parquet"}
            save_figures: false
            """
        ),
        encoding="utf-8",
    )

    assert main(["compute", "--config", str(config_path)]) == 0
    assert "kernel tuning (volt)" in capsys.readouterr().out

    calibrator = load_interval_artifact(str(artifact))
    assert isinstance(calibrator, KernelIntervalCalibrator)
    assert calibrator.tuning is not None
    assert calibrator.tuning["grouped_folds"] is True  # load_scenario_idx was found
    assert calibrator.n_eff_target == calibrator.tuning["selected_n_eff_target"]
    assert calibrator.distance == calibrator.tuning["selected_distance"]
    report = pd.read_csv(tmp_path / "out" / "kernel_kernel_tuning.csv")
    assert len(report) == 4  # the random-search budget
    assert report["n_eff_target"].between(20, 100).all()
    assert (tmp_path / "out" / "kernel_kernel_tuning_details.csv").exists()

    assert main(["calibrate", "--config", str(config_path)]) == 0
    assert (tmp_path / "out" / "calibrated.parquet").exists()
