"""Configuration helpers for the trustlayer CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_cli_config(config_path: str | None) -> dict[str, Any]:
    """Load the required trustlayer YAML config file."""

    if config_path is None:
        raise ValueError("Missing required option 'config'. Provide it via --config.")
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("trust_config.yaml must decode to a mapping at the top level.")
    return data


def resolve_command_config(
    config: dict[str, Any],
    command_name: str,
) -> dict[str, Any]:
    """Resolve shared and command-specific config values for a CLI verb."""

    resolved: dict[str, Any] = {}
    for key in (
        "version",
        "task_name",
        "alpha",
        "cp_type",
        "variables",
        "calibration_fraction",
        "random_seed",
        "screen_bounds",
    ):
        if key in config:
            resolved[key] = config[key]

    _merge_mapping(
        resolved,
        config.get("inputs"),
        allowed_keys={
            "predictions",
            "generator_predictions",
            "pf_node",
            "pf_edge",
            "artifact",
            "analysis_pf_node",
            "analysis_pf_edge",
        },
    )
    _merge_mapping(
        resolved,
        config.get("outputs"),
        allowed_keys={
            "output",
            "artifact",
            "calibrated_predictions",
            "calibrated_predictions_volt",
            "calibrated_predictions_line",
            "screen_dir",
            "screened_predictions",
        },
    )

    command_overrides = {}
    commands = config.get("commands")
    if isinstance(commands, dict):
        command_overrides = commands.get(command_name, {})
    if command_overrides is None:
        command_overrides = {}
    if not isinstance(command_overrides, dict):
        raise ValueError(f"commands.{command_name} must be a mapping when provided.")

    for key in (
        "task_name",
        "alpha",
        "cp_type",
        "variables",
        "output",
        "artifact",
        "calibration_fraction",
        "random_seed",
        "screen_bounds",
    ):
        if key in command_overrides:
            resolved[key] = command_overrides[key]
    _merge_mapping(
        resolved,
        command_overrides.get("inputs"),
        allowed_keys={
            "predictions",
            "generator_predictions",
            "pf_node",
            "pf_edge",
            "artifact",
            "analysis_pf_node",
            "analysis_pf_edge",
        },
    )
    _merge_mapping(
        resolved,
        command_overrides.get("outputs"),
        allowed_keys={
            "output",
            "artifact",
            "calibrated_predictions",
            "calibrated_predictions_volt",
            "calibrated_predictions_line",
            "screen_dir",
            "screened_predictions",
        },
    )
    return resolved


def coalesce_option(
    cli_value: Any,
    config: dict[str, Any],
    key: str,
    *,
    required: bool = False,
) -> Any:
    """Return the CLI value when present, otherwise fall back to config."""

    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    if required:
        raise ValueError(
            f"Missing required option '{key}'. Provide it via CLI or config."
        )
    return None


def normalize_variables(raw_value: Any) -> tuple[str, ...]:
    """Normalize a variable selection from CLI or YAML config."""

    if raw_value is None:
        raw_values = ["Vm", "Va"]
    elif isinstance(raw_value, str):
        raw_values = [part.strip() for part in raw_value.split(",") if part.strip()]
    elif isinstance(raw_value, (list, tuple)):
        raw_values = [str(part).strip() for part in raw_value if str(part).strip()]
    else:
        raise ValueError("variables must be a string or a list of strings.")

    aliases = {
        "vm": "Vm",
        "va": "Va",
        "line_loading": "line_loading",
        "line-loading": "line_loading",
        "lineloading": "line_loading",
    }
    values: list[str] = []
    for value in raw_values:
        normalized = aliases.get(value.strip().lower())
        if normalized is None:
            raise ValueError(
                "Unsupported variable selection "
                f"{value!r}. Supported values are Vm, Va, and line_loading.",
            )
        values.append(normalized)

    deduplicated = tuple(dict.fromkeys(values))
    if not deduplicated:
        raise ValueError("variables must not be empty.")
    return deduplicated


def validate_cp_type(cp_type: str | None) -> str:
    """Validate the currently supported CP type selection."""

    normalized = "base" if cp_type is None else str(cp_type).strip().lower()
    if normalized == "naive":
        return "base"
    if normalized != "base":
        raise ValueError(
            f"Unsupported cp_type: {cp_type!r}. The current implementation supports 'base' (legacy alias: 'naive').",
        )
    return normalized


def _merge_mapping(
    target: dict[str, Any],
    value: Any,
    *,
    allowed_keys: set[str],
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("Expected a mapping in trust_config.yaml.")
    for key, item in value.items():
        if key in allowed_keys:
            target[key] = item
