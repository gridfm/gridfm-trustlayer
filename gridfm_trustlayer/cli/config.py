"""Configuration helpers for the trustlayer CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from gridfm_trustlayer.conformal.kernel import (
    SUPPORTED_BANDWIDTHS,
    SUPPORTED_DISTANCES,
)
from gridfm_trustlayer.io.embeddings import validate_pooling


_ALLOWED_TOP_LEVEL_KEYS = {
    "version",
    "task_name",
    "alpha",
    "cp_type",
    "strat",
    "system_load_n_bins",
    "pred_n_bins",
    "k_nn",
    "n_eff_min",
    "bandwidth",
    "n_eff_target",
    "distance",
    "embedding_pooling",
    "variables",
    "calibration_fraction",
    "random_seed",
    "screen_bounds",
    "save_figures",
    "inputs",
    "outputs",
    "commands",
}

_ALLOWED_COMMAND_NAMES = {"compute", "calibrate", "screen"}

_ALLOWED_COMMAND_OVERRIDE_KEYS = {
    "task_name",
    "alpha",
    "cp_type",
    "strat",
    "system_load_n_bins",
    "pred_n_bins",
    "k_nn",
    "n_eff_min",
    "bandwidth",
    "n_eff_target",
    "distance",
    "embedding_pooling",
    "variables",
    "output",
    "artifact",
    "calibration_fraction",
    "random_seed",
    "screen_bounds",
    "save_figures",
    "inputs",
    "outputs",
}


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
    validate_config_schema(data)
    return data


def validate_config_schema(config: dict[str, Any]) -> None:
    """Validate trustlayer YAML structure before command-level resolution."""

    unknown_top_level = sorted(set(config).difference(_ALLOWED_TOP_LEVEL_KEYS))
    if unknown_top_level:
        raise ValueError(
            f"Unsupported top-level trustlayer config keys: {unknown_top_level}."
        )

    _validate_mapping_keys(
        config,
        key="inputs",
        allowed_keys={
            "predictions",
            "generator_predictions",
            "pf_node",
            "pf_edge",
            "embeddings",
            "artifact",
            "analysis_pf_node",
            "analysis_pf_edge",
        },
    )
    _validate_mapping_keys(
        config,
        key="outputs",
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

    commands = config.get("commands")
    if commands is None:
        return
    if not isinstance(commands, dict):
        raise ValueError("commands must be a mapping when provided.")

    unknown_commands = sorted(set(commands).difference(_ALLOWED_COMMAND_NAMES))
    if unknown_commands:
        raise ValueError(
            f"Unsupported commands.* sections in trustlayer config: {unknown_commands}."
        )

    for command_name, overrides in commands.items():
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            raise ValueError(
                f"commands.{command_name} must be a mapping when provided."
            )
        unknown_override_keys = sorted(
            set(overrides).difference(_ALLOWED_COMMAND_OVERRIDE_KEYS)
        )
        if unknown_override_keys:
            raise ValueError(
                f"Unsupported keys under commands.{command_name}: "
                f"{unknown_override_keys}."
            )
        _validate_mapping_keys(
            overrides,
            key="inputs",
            allowed_keys={
                "predictions",
                "generator_predictions",
                "pf_node",
                "pf_edge",
                "embeddings",
                "artifact",
                "analysis_pf_node",
                "analysis_pf_edge",
            },
            context=f"commands.{command_name}",
        )
        _validate_mapping_keys(
            overrides,
            key="outputs",
            allowed_keys={
                "output",
                "artifact",
                "calibrated_predictions",
                "calibrated_predictions_volt",
                "calibrated_predictions_line",
                "screen_dir",
                "screened_predictions",
            },
            context=f"commands.{command_name}",
        )


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
        "strat",
        "system_load_n_bins",
        "pred_n_bins",
        "k_nn",
        "n_eff_min",
        "bandwidth",
        "n_eff_target",
        "distance",
        "embedding_pooling",
        "variables",
        "calibration_fraction",
        "random_seed",
        "screen_bounds",
        "save_figures",
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
            "embeddings",
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
        "strat",
        "system_load_n_bins",
        "pred_n_bins",
        "k_nn",
        "n_eff_min",
        "bandwidth",
        "n_eff_target",
        "distance",
        "embedding_pooling",
        "variables",
        "output",
        "artifact",
        "calibration_fraction",
        "random_seed",
        "screen_bounds",
        "save_figures",
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
            "embeddings",
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


def validate_command_config(command_name: str, config: dict[str, Any]) -> None:
    """Validate resolved command-level config values with clear messages."""

    cp_type = validate_cp_type(config.get("cp_type"))
    strat = validate_strat(config.get("strat"))
    validate_k_nn(config.get("k_nn"))
    validate_n_eff_min(config.get("n_eff_min"))
    validate_distance(config.get("distance"))
    bandwidth = validate_bandwidth(config.get("bandwidth"))
    n_eff_target = validate_n_eff_target(config.get("n_eff_target"))
    if bandwidth == "n_eff" and n_eff_target is not None:
        n_eff_min = config.get("n_eff_min")
        if n_eff_min is not None and n_eff_target < float(n_eff_min):
            raise ValueError(
                f"n_eff_target ({n_eff_target}) is below n_eff_min ({n_eff_min}), "
                "so every scenario would immediately fall back to the stratum quantile.",
            )
    if "embedding_pooling" in config:
        validate_pooling(config.get("embedding_pooling"))

    if cp_type == "kernel":
        if strat == "pred_bins":
            raise ValueError(
                "cp_type 'kernel' does not support strat 'pred_bins': prediction bins are "
                "element-wise while kernel localization is scenario-wise.",
            )
        if command_name == "compute":
            if config.get("embeddings") is None:
                raise ValueError(
                    "cp_type 'kernel' requires inputs.embeddings pointing at a scenario "
                    "embeddings parquet.",
                )
            if config.get("calibration_fraction") is None:
                raise ValueError(
                    "cp_type 'kernel' requires a calibration_fraction. Without a split the "
                    "kernel would weight every test scenario against itself at distance "
                    "zero, which invalidates the coverage guarantee.",
                )

    if "system_load_n_bins" in config:
        validate_system_load_n_bins(config.get("system_load_n_bins"))
    if "pred_n_bins" in config:
        validate_pred_n_bins(config.get("pred_n_bins"))
    if "variables" in config:
        normalize_variables(config.get("variables"))

    if strat == "system_load_bin" and config.get("system_load_n_bins") is None:
        raise ValueError(
            "system_load_n_bins is required when strat is 'system_load_bin'."
        )
    if strat == "pred_bins" and config.get("pred_n_bins") is None:
        raise ValueError("pred_n_bins is required when strat is 'pred_bins'.")

    calibration_fraction = config.get("calibration_fraction")
    if calibration_fraction is not None:
        try:
            normalized_fraction = float(calibration_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "calibration_fraction must be numeric when provided."
            ) from exc
        if not 0.0 < normalized_fraction < 1.0:
            raise ValueError("calibration_fraction must be strictly between 0 and 1.")

    random_seed = config.get("random_seed")
    if random_seed is not None:
        try:
            int(random_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("random_seed must be an integer when provided.") from exc

    if "screen_bounds" in config and config.get("screen_bounds") is not None:
        screen_bounds = config["screen_bounds"]
        if not isinstance(screen_bounds, dict):
            raise ValueError("screen_bounds must be a mapping when provided.")

    if "save_figures" in config and config.get("save_figures") is not None:
        _validate_boolean_option(config.get("save_figures"), option_name="save_figures")


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


SUPPORTED_CP_TYPES = ("element", "kernel")


def validate_cp_type(cp_type: str | None) -> str:
    """Validate the currently supported CP type selection."""

    normalized = "element" if cp_type is None else str(cp_type).strip().lower()
    if normalized not in SUPPORTED_CP_TYPES:
        supported = ", ".join(repr(name) for name in SUPPORTED_CP_TYPES)
        raise ValueError(
            f"Unsupported cp_type: {cp_type!r}. The current implementation supports {supported}.",
        )
    return normalized


def validate_k_nn(value: Any) -> int | None:
    """Validate the kernel nearest-neighbour count used to set the bandwidth."""

    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_nn must be an integer when provided.") from exc
    if normalized < 1:
        raise ValueError(f"k_nn must be >= 1; got {value!r}.")
    return normalized


def validate_distance(value: str | None) -> str:
    """Validate the kernel distance metric used over scenario representations."""

    normalized = "euclidean" if value is None else str(value).strip().lower()
    if normalized not in SUPPORTED_DISTANCES:
        supported = ", ".join(repr(name) for name in SUPPORTED_DISTANCES)
        raise ValueError(
            f"Unsupported distance: {value!r}. Supported values: {supported}.",
        )
    return normalized


def validate_bandwidth(value: str | None) -> str:
    """Validate the kernel bandwidth-selection mode."""

    normalized = "knn" if value is None else str(value).strip().lower()
    if normalized not in SUPPORTED_BANDWIDTHS:
        supported = ", ".join(repr(name) for name in SUPPORTED_BANDWIDTHS)
        raise ValueError(
            f"Unsupported bandwidth: {value!r}. Supported values: {supported}.",
        )
    return normalized


def validate_n_eff_target(value: Any) -> float | None:
    """Validate the target effective sample size for `bandwidth: n_eff`."""

    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_eff_target must be numeric when provided.") from exc
    if normalized < 1.0:
        raise ValueError(f"n_eff_target must be >= 1; got {value!r}.")
    return normalized


def validate_n_eff_min(value: Any) -> float | None:
    """Validate the minimum effective sample size before KCP falls back to SCP."""

    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_eff_min must be numeric when provided.") from exc
    if normalized < 1.0:
        raise ValueError(f"n_eff_min must be >= 1; got {value!r}.")
    return normalized


def validate_strat(strat: str | None) -> str | None:
    """Validate the supported residual stratification modes."""

    if strat is None:
        return None
    normalized = str(strat).strip().lower()
    if normalized not in {"n_outages", "outage_type", "system_load_bin", "pred_bins"}:
        raise ValueError(
            f"Unsupported strat: {strat!r}. The current implementation supports only 'n_outages', 'outage_type', 'system_load_bin', and 'pred_bins'."
        )
    return normalized


def validate_system_load_n_bins(value: Any) -> int | None:
    """Validate the requested percentile-bin count for system_load_bin."""

    return _validate_bin_count(value, option_name="system_load_n_bins")


def validate_pred_n_bins(value: Any) -> int | None:
    """Validate the requested percentile-bin count for pred_bins."""

    return _validate_bin_count(value, option_name="pred_n_bins")


def _validate_bin_count(value: Any, *, option_name: str) -> int | None:
    """Validate a requested percentile-bin count option."""

    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{option_name} must be an integer when provided; got {value!r}."
        ) from exc
    if normalized < 2:
        raise ValueError(
            f"{option_name} must be at least 2 when provided; got {normalized!r}."
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


def _validate_mapping_keys(
    container: dict[str, Any],
    *,
    key: str,
    allowed_keys: set[str],
    context: str = "top-level",
) -> None:
    mapping = container.get(key)
    if mapping is None:
        return
    if not isinstance(mapping, dict):
        raise ValueError(f"{context}.{key} must be a mapping when provided.")
    unknown_keys = sorted(set(mapping).difference(allowed_keys))
    if unknown_keys:
        raise ValueError(f"Unsupported keys under {context}.{key}: {unknown_keys}.")


def _validate_boolean_option(value: Any, *, option_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{option_name} must be a boolean when provided.")
