"""The shipped example configs must stay loadable and internally consistent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from gridfm_trustlayer.cli.config import (
    load_cli_config,
    resolve_command_config,
    validate_command_config,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "config"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yaml"))


def test_examples_directory_is_populated() -> None:
    assert EXAMPLE_CONFIGS, f"no example configs found under {EXAMPLES_DIR}"


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_validates_for_every_verb(path: Path) -> None:
    config = load_cli_config(str(path))
    for command in ("compute", "calibrate", "screen"):
        validate_command_config(command, resolve_command_config(config, command))


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_screened_variables_all_have_bounds(path: Path) -> None:
    """`screen` fails at runtime, not at validation, if a variable lacks bounds."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    screen_config = resolve_command_config(config, "screen")
    variables = screen_config.get("variables") or ["Vm", "Va"]
    bounds = screen_config.get("screen_bounds") or {}
    missing = [name for name in variables if name not in bounds]
    assert not missing, f"{path.name} screens {missing} without screen_bounds"


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_output_keys_match_the_number_of_target_families(path: Path) -> None:
    """Single-family runs use `calibrated_predictions`; mixed runs use the split keys."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    variables = config.get("variables") or ["Vm", "Va"]
    outputs = config.get("outputs") or {}
    families = {"line" if name == "line_loading" else "volt" for name in variables}
    if len(families) > 1:
        assert "calibrated_predictions_volt" in outputs
        assert "calibrated_predictions_line" in outputs
    else:
        assert "calibrated_predictions" in outputs, (
            f"{path.name} requests one target family, so calibrate resolves "
            "outputs.calibrated_predictions rather than the split keys"
        )


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_kernel_examples_declare_their_required_inputs(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("cp_type") != "kernel":
        pytest.skip("not a kernel example")
    assert config.get("calibration_fraction") is not None
    assert (config.get("inputs") or {}).get("embeddings") is not None
