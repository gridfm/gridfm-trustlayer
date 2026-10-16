"""Artifact helpers for saved interval-calibration state.

Two on-disk encodings are supported:

* ``element`` (naive quantile-table) artifacts stay plain JSON, unchanged.
* ``kernel`` artifacts carry a calibration residual bank plus scenario
  embeddings, which are far too large for pretty-printed JSON, so they are
  written as a compressed ``npz`` container with a ``__meta__`` JSON entry.

Both encodings are read back through the same two loader functions; the format
is detected from the file's magic bytes rather than its extension.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from gridfm_trustlayer.conformal.calibrator import NaiveIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import KernelIntervalCalibrator

BUNDLE_VERSION = "trustlayer_interval_bundle_v1"
KERNEL_BUNDLE_VERSION = "trustlayer_kernel_bundle_v1"

_NPZ_MAGIC = b"PK\x03\x04"
_META_KEY = "__meta__"

IntervalCalibrator = NaiveIntervalCalibrator | KernelIntervalCalibrator


def save_interval_artifact(
    calibrator: IntervalCalibrator,
    output_path: str,
) -> None:
    """Write a fitted calibrator to a `.trustlayer` file.

    JSON for `NaiveIntervalCalibrator`, a compressed NumPy archive for
    `KernelIntervalCalibrator`; the format the CLI writes.

    Args:
        calibrator: The fitted calibrator.
        output_path: Destination file; parent directories are created.
    """

    save_interval_artifacts({_artifact_key(calibrator): calibrator}, output_path)


def save_interval_artifacts(
    calibrators: dict[str, IntervalCalibrator],
    output_path: str,
) -> None:
    """Persist one or more interval calibrators to a single artifact file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not calibrators:
        raise ValueError(
            "At least one calibrator must be provided when saving an artifact."
        )
    if any(
        isinstance(calibrator, KernelIntervalCalibrator)
        for calibrator in calibrators.values()
    ):
        _save_kernel_artifacts(calibrators, path)
        return

    if len(calibrators) == 1:
        payload: dict[str, object] = next(iter(calibrators.values())).to_dict()
    else:
        payload = {
            "version": BUNDLE_VERSION,
            "artifacts": {
                name: calibrator.to_dict() for name, calibrator in calibrators.items()
            },
        }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_interval_artifact(artifact_path: str) -> IntervalCalibrator:
    """Read a calibrator from a `.trustlayer` file.

    Args:
        artifact_path: A file written by `save_interval_artifact` or `compute`.

    Returns:
        The calibrator, of the class it was saved from.

    Raises:
        ValueError: If the file holds several calibrators (a mixed CLI run).
    """

    calibrators = load_interval_artifacts(artifact_path)
    if len(calibrators) != 1:
        raise ValueError(
            "Artifact contains multiple calibrators; use load_interval_artifacts() or the CLI mixed-variable flow.",
        )
    return next(iter(calibrators.values()))


def load_interval_artifacts(artifact_path: str) -> dict[str, IntervalCalibrator]:
    """Read every calibrator of a `.trustlayer` file.

    A CLI run with both bus voltages and line loading stores one calibrator
    per family in the same file; a single-family file holds one.

    Args:
        artifact_path: A file written by `save_interval_artifact` or `compute`.

    Returns:
        The calibrators by family: `"volt"` (bus variables) and `"line"`
        (`line_loading`).
    """

    path = Path(artifact_path)
    raw = path.read_bytes()
    if raw[: len(_NPZ_MAGIC)] == _NPZ_MAGIC:
        return _load_kernel_artifacts(raw)

    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"{artifact_path} is not a trustlayer artifact (a .trustlayer file "
            "written by compute)."
        ) from exc
    if not isinstance(data, dict) or not str(data.get("version", "")).startswith(
        "trustlayer_"
    ):
        raise ValueError(
            f"{artifact_path} is not a trustlayer artifact (a .trustlayer file "
            "written by compute)."
        )
    if data.get("version") == BUNDLE_VERSION:
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(
                "Interval artifact bundle must contain a non-empty 'artifacts' mapping."
            )
        return {
            str(name): NaiveIntervalCalibrator.from_dict(dict(calibrator_data))
            for name, calibrator_data in artifacts.items()
        }
    calibrator = NaiveIntervalCalibrator.from_dict(data)
    return {_artifact_key(calibrator): calibrator}


def is_interval_artifact(path: str | Path) -> bool:
    """Whether `path` is a `.trustlayer` file (either encoding)."""

    path = Path(path)
    if not path.is_file():
        return False
    try:
        raw = path.read_bytes()
        if raw[: len(_NPZ_MAGIC)] == _NPZ_MAGIC:
            with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
                return _META_KEY in payload
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return isinstance(data, dict) and str(data.get("version", "")).startswith(
        "trustlayer_"
    )


def load_ground_truth_reference(artifact_path: str) -> dict[str, Any] | None:
    """The calibration set's ground-truth reference stored in an artifact, if any.

    Reads only the metadata, not the residual bank of a kernel artifact.
    """

    raw = Path(artifact_path).read_bytes()
    if raw[: len(_NPZ_MAGIC)] == _NPZ_MAGIC:
        with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
            if _META_KEY not in payload:
                return None
            meta = json.loads(bytes(payload[_META_KEY]).decode("utf-8"))
        entries = list((meta.get("artifacts") or {}).values())
    else:
        data = json.loads(raw.decode("utf-8"))
        entries = (
            list((data.get("artifacts") or {}).values())
            if data.get("version") == BUNDLE_VERSION
            else [data]
        )
    for entry in entries:
        if isinstance(entry, dict) and entry.get("ground_truth_reference"):
            return cast(dict[str, Any], entry["ground_truth_reference"])
    return None


def _save_kernel_artifacts(
    calibrators: dict[str, IntervalCalibrator],
    path: Path,
) -> None:
    meta: dict[str, object] = {
        "version": KERNEL_BUNDLE_VERSION,
        "artifacts": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for name, calibrator in calibrators.items():
        entry = calibrator.to_dict()
        if isinstance(calibrator, KernelIntervalCalibrator):
            entry["_encoding"] = "kernel"
            for array_name, values in calibrator.array_payload().items():
                arrays[f"{name}::{array_name}"] = values
        else:
            entry["_encoding"] = "naive"
        meta["artifacts"][name] = entry  # type: ignore[index]

    arrays[_META_KEY] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    with path.open("wb") as handle:
        # The numpy stub types the second positional as `allow_pickle`, so the
        # **kwargs array payload has to be passed through explicitly typed.
        np.savez_compressed(handle, **cast(dict[str, Any], arrays))


def _load_kernel_artifacts(raw: bytes) -> dict[str, IntervalCalibrator]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
        if _META_KEY not in payload:
            raise ValueError(
                "Binary interval artifact is missing its '__meta__' entry.",
            )
        meta = json.loads(bytes(payload[_META_KEY]).decode("utf-8"))
        if meta.get("version") != KERNEL_BUNDLE_VERSION:
            raise ValueError(
                f"Unsupported binary artifact version: {meta.get('version')!r}",
            )
        artifacts = meta.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(
                "Binary interval artifact must contain a non-empty 'artifacts' mapping."
            )
        loaded: dict[str, IntervalCalibrator] = {}
        for name, entry in artifacts.items():
            encoding = entry.get("_encoding", "naive")
            entry = {key: value for key, value in entry.items() if key != "_encoding"}
            if encoding == "kernel":
                prefix = f"{name}::"
                arrays = {
                    key[len(prefix) :]: payload[key]
                    for key in payload.files
                    if key.startswith(prefix)
                }
                loaded[str(name)] = KernelIntervalCalibrator.from_dict(entry, arrays)
            else:
                loaded[str(name)] = NaiveIntervalCalibrator.from_dict(entry)
    return loaded


def _artifact_key(calibrator: IntervalCalibrator) -> str:
    if calibrator.entity_name == "branch" or calibrator.channel_names == (
        "line_loading",
    ):
        return "line"
    return "volt"
