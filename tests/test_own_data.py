"""Using trustlayer with your own data: `ChannelSet.from_frame`, `check_inputs` and the guide."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from gridfm_trustlayer import (
    ChannelSet,
    KernelIntervalCalibrator,
    NaiveIntervalCalibrator,
    check_inputs,
)
from gridfm_trustlayer.cli import main

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "gridfm_trustlayer" / "guide" / "own_data.md"


# ------------------------------------------------------------------ the guide


def test_guide_example_runs_and_prints_its_documented_output(capsys) -> None:
    text = GUIDE.read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    # Running the documented example verbatim is the point of this test.
    for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL):
        exec(compile(block, str(GUIDE), "exec"), namespace)  # noqa: S102
    printed = capsys.readouterr().out.splitlines()
    documented = re.findall(r'```text title="Output"\n(.*?)```', text, flags=re.DOTALL)
    assert documented
    # The Output blocks, in order, hold everything the code blocks print.
    expected = [line for block in documented for line in block.rstrip().splitlines()]

    assert len(printed) == len(expected)
    for got, want in zip(printed, expected, strict=True):
        # Same words; numbers equal up to the last printed digit.
        assert re.sub(r"[\d.]+", "#", got) == re.sub(r"[\d.]+", "#", want)
        for a, b in zip(
            re.findall(r"\d+\.\d+", got), re.findall(r"\d+\.\d+", want), strict=True
        ):
            assert abs(float(a) - float(b)) <= 0.002, (got, want)


def test_guide_command_prints_the_guide_and_its_path(capsys) -> None:
    assert main(["guide"]) == 0
    assert capsys.readouterr().out == GUIDE.read_text(encoding="utf-8")

    assert main(["guide", "--path"]) == 0
    assert Path(capsys.readouterr().out.strip()).resolve() == GUIDE.resolve()


def test_docs_page_includes_the_packaged_guide() -> None:
    page = (ROOT / "docs" / "user-guide" / "own-data.md").read_text(encoding="utf-8")
    assert '--8<-- "gridfm_trustlayer/guide/own_data.md"' in page


# ------------------------------------------------------------------ from_frame


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": [3, 3, 1, 1, 2],
            "line": [7, 5, 5, 7, 5],
            "pred": [1.0, 2.0, 3.0, np.nan, 5.0],
            "true": [1.5, 2.5, 3.5, 4.5, 5.5],
            "k": [2, 2, 1, 1, 1],
            "ok": [True, False, True, True, True],
        }
    )


def test_from_frame_sorts_by_id_and_marks_missing_values_invalid() -> None:
    channel_set = ChannelSet.from_frame(
        _table(), {"x": "pred"}, entity="line", valid="ok", strata=["k"]
    )

    np.testing.assert_array_equal(channel_set.scenario_ids, [1, 2, 3])
    np.testing.assert_array_equal(channel_set.entity_ids, [5, 7])
    np.testing.assert_array_equal(
        channel_set.channels["x"], [[3.0, np.nan], [5.0, np.nan], [2.0, 1.0]]
    )
    # NaN (scenario 1, line 7), absent row (2, 7) and flagged row (3, 5).
    np.testing.assert_array_equal(
        channel_set.valid_masks["x"], [[True, False], [True, False], [False, True]]
    )
    np.testing.assert_array_equal(channel_set.scenario_columns["k"], [1, 1, 2])
    assert channel_set.entity_name == "bus"


def test_from_frame_takes_a_list_of_columns_and_per_variable_validity() -> None:
    channel_set = ChannelSet.from_frame(
        _table().assign(scenario=lambda t: t["scenario"].astype(float)),
        ["true", "pred"],
        entity="line",
        valid={"true": "ok"},
    )

    assert set(channel_set.channels) == {"true", "pred"}
    np.testing.assert_array_equal(channel_set.scenario_ids, [1, 2, 3])
    assert channel_set.valid_masks["true"][2, 0] == np.False_
    # `pred` has no validity column: only its absent and NaN cells are invalid.
    assert channel_set.valid_masks["pred"][2, 0] == np.True_


def test_from_frame_aligns_embeddings_by_scenario_id() -> None:
    by_index = pd.DataFrame(
        {"a": [0.3, 0.1, 9.0, 0.2], "b": [3.0, 1.0, 9.0, 2.0]},
        index=pd.Index([3, 1, 99, 2], name="scenario"),
    )
    with_column = by_index.reset_index()

    for embeddings in (by_index, with_column):
        channel_set = ChannelSet.from_frame(
            _table(), {"x": "pred"}, entity="line", embeddings=embeddings
        )
        assert channel_set.scenario_embeddings is not None
        np.testing.assert_allclose(
            channel_set.scenario_embeddings,
            np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]]),
        )
    array = np.arange(6, dtype=np.float32).reshape(3, 2)
    channel_set = ChannelSet.from_frame(
        _table(), {"x": "pred"}, entity="line", embeddings=array
    )
    np.testing.assert_array_equal(channel_set.scenario_embeddings, array)


@pytest.mark.parametrize(
    ("change", "kwargs", "message"),
    [
        (lambda t: pd.concat([t, t.iloc[:1]]), {}, "appear more than once"),
        (lambda t: t.assign(line=t["line"].astype(str)), {}, "must be integers"),
        (lambda t: t.assign(scenario=t["scenario"] + 0.5), {}, "must be integers"),
        (lambda t: t, {"strata": ["missing"]}, "not in the table"),
        (lambda t: t.assign(k=[2, 1, 1, 1, 1]), {"strata": ["k"]}, "different values"),
        (lambda t: t.assign(k=t["k"] + 0.5), {"strata": ["k"]}, "must hold integers"),
        (lambda t: t, {"valid": {"y": "ok"}}, "not in channels"),
        (
            lambda t: t,
            {"embeddings": pd.DataFrame({"a": [1.0]}, index=pd.Index([1], name="s"))},
            "no row for 2",
        ),
        (
            lambda t: t,
            {"embeddings": pd.DataFrame({"scenario": [1, 1, 2, 3], "a": [0.0] * 4})},
            "Pool them",
        ),
        (lambda t: t, {"embeddings": np.zeros((2, 4))}, "one row per scenario"),
    ],
)
def test_from_frame_refuses_bad_tables_with_a_clear_message(
    change, kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        ChannelSet.from_frame(change(_table()), {"x": "pred"}, entity="line", **kwargs)


# ------------------------------------------------------------------ check_inputs


def _sets(
    n: int = 300, seed: int = 0, n_lines: int = 4
) -> tuple[ChannelSet, ChannelSet]:
    rng = np.random.default_rng(seed)
    scenario = np.repeat(np.arange(n), n_lines)
    predicted = rng.normal(size=scenario.size)
    table = pd.DataFrame(
        {
            "scenario": scenario,
            "line": np.tile(np.arange(n_lines), n),
            "k": rng.integers(1, 3, n)[scenario],
            "pred": predicted,
            "true": predicted + rng.normal(0.0, 0.1, scenario.size),
        }
    )
    embeddings = pd.DataFrame(rng.normal(size=(n, 3)))
    common = {"entity": "line", "strata": ["k"], "embeddings": embeddings}
    return (
        ChannelSet.from_frame(table, {"x": "pred"}, **common),
        ChannelSet.from_frame(table, {"x": "true"}, **common),
    )


def _fit(predictions: ChannelSet, true_values: ChannelSet, kernel: bool):
    settings: dict[str, Any] = {"alpha": 0.2, "channel_names": ("x",), "strat": "k"}
    if kernel:
        assert predictions.scenario_embeddings is not None
        return KernelIntervalCalibrator.fit(
            predictions,
            true_values,
            embeddings=predictions.scenario_embeddings,
            n_eff_target=50,
            **settings,
        )
    return NaiveIntervalCalibrator.fit(predictions, true_values, **settings)


def test_check_inputs_passes_good_data_before_and_after_fitting() -> None:
    predictions, true_values = _sets()
    check = check_inputs(predictions, true_values, strat="k", alpha=0.2, kernel=True)
    assert check.ok
    assert str(check) == "No problems found."
    for kernel in (False, True):
        calibrator = _fit(predictions, true_values, kernel)
        new, _ = _sets(seed=1)
        assert str(check_inputs(new, calibrator=calibrator)) == "No problems found."


def _with(channel_set: ChannelSet, **changes) -> ChannelSet:
    fields = {**channel_set.__dict__, **changes}
    return ChannelSet(**fields)


@pytest.mark.parametrize(
    ("make", "kwargs", "message"),
    [
        (
            lambda p, t: (p, _with(t, scenario_ids=t.scenario_ids[::-1].copy())),
            {},
            "different order",
        ),
        (
            lambda p, t: (p, _with(t, scenario_ids=t.scenario_ids + 1000)),
            {},
            "different scenarios",
        ),
        (lambda p, t: (p, t), {"channel_names": ["y"]}, "no channel"),
        (lambda p, t: (p, t), {"strat": "n_outages"}, "no stratum column"),
        (
            lambda p, t: (_with(p, scenario_embeddings=None), t),
            {"kernel": True},
            "one embedding vector per scenario",
        ),
        (
            lambda p, t: (
                _with(
                    p,
                    scenario_embeddings=np.where(
                        np.arange(len(p.scenario_ids))[:, None] == 3,
                        np.nan,
                        p.scenario_embeddings,
                    ),
                ),
                t,
            ),
            {"kernel": True},
            "NaN or infinite",
        ),
        (
            lambda p, t: (
                _with(p, scenario_embeddings=np.ones_like(p.scenario_embeddings)),
                t,
            ),
            {"kernel": True},
            "same embedding vector",
        ),
        (
            lambda p, t: (_with(p, entity_ids=p.entity_ids.astype(float)), t),
            {},
            "must be integers",
        ),
    ],
)
def test_check_inputs_reports_errors_before_fitting(make, kwargs, message) -> None:
    predictions, true_values = make(*_sets())

    check = check_inputs(predictions, true_values, **kwargs)

    assert not check.ok
    assert message in str(check)
    with pytest.raises(ValueError, match=message):
        check.raise_for_errors()


def test_check_inputs_warns_about_values_without_a_finite_bound() -> None:
    predictions, true_values = _sets(n=40)
    channels = predictions.channels["x"].copy()
    channels[0, 0] = np.nan
    predictions = _with(predictions, channels={"x": channels})

    check = check_inputs(predictions, true_values, strat="k", alpha=0.1)

    assert check.ok
    text = str(check)
    assert "NaN or infinite but marked valid" in text
    # 40 scenarios in two strata: fewer than 19 per stratum, enough overall.
    assert "get their quantile over all strata" in text
    small, small_true = _sets(n=10)
    assert "their bounds will be infinite" in str(
        check_inputs(small, small_true, alpha=0.1)
    )


def test_check_inputs_compares_new_data_with_the_fitted_calibrator() -> None:
    predictions, true_values = _sets()
    calibrator = _fit(predictions, true_values, kernel=True)
    new, _ = _sets(seed=1)

    fewer_lines, _ = _sets(seed=1, n_lines=3)
    assert "differ from those the calibrator was fitted on" in str(
        check_inputs(fewer_lines, calibrator=calibrator)
    )
    assert new.scenario_embeddings is not None
    narrow = _with(new, scenario_embeddings=new.scenario_embeddings[:, :2])
    assert "was fitted on 3" in str(check_inputs(narrow, calibrator=calibrator))
    other_stratum = _with(
        new, scenario_columns={"k": np.full(len(new.scenario_ids), 7)}
    )
    check = check_inputs(other_stratum, calibrator=calibrator)
    assert check.ok
    assert "strata absent from the calibration data" in str(check)
    assert "differs from the calibrator's" in str(
        check_inputs(new, calibrator=calibrator, strat="other")
    )


def test_check_inputs_refuses_objects_that_are_not_channel_sets() -> None:
    check = check_inputs(np.zeros((3, 2)))  # type: ignore[arg-type]
    assert not check.ok
    assert "must be a ChannelSet" in str(check)
