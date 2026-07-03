"""Artifact helpers for saved interval-calibration state."""

from __future__ import annotations

import json
from pathlib import Path

from gridfm_trustlayer.conformal.calibrator import NaiveIntervalCalibrator


BUNDLE_VERSION = "trustlayer_interval_bundle_v1"


def save_interval_artifact(
    calibrator: NaiveIntervalCalibrator,
    output_path: str,
) -> None:
    """Persist a naive interval calibrator to a single JSON artifact file."""

    save_interval_artifacts({_artifact_key(calibrator): calibrator}, output_path)


def save_interval_artifacts(
    calibrators: dict[str, NaiveIntervalCalibrator],
    output_path: str,
) -> None:
    """Persist one or more naive interval calibrators to a JSON artifact file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not calibrators:
        raise ValueError("At least one calibrator must be provided when saving an artifact.")
    if len(calibrators) == 1:
        payload: dict[str, object] = next(iter(calibrators.values())).to_dict()
    else:
        payload = {
            "version": BUNDLE_VERSION,
            "artifacts": {
                name: calibrator.to_dict()
                for name, calibrator in calibrators.items()
            },
        }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_interval_artifact(artifact_path: str) -> NaiveIntervalCalibrator:
    """Load a naive interval calibrator from a JSON artifact file."""

    calibrators = load_interval_artifacts(artifact_path)
    if len(calibrators) != 1:
        raise ValueError(
            "Artifact contains multiple calibrators; use load_interval_artifacts() or the CLI mixed-variable flow.",
        )
    return next(iter(calibrators.values()))


def load_interval_artifacts(artifact_path: str) -> dict[str, NaiveIntervalCalibrator]:
    """Load one or more naive interval calibrators from a JSON artifact file."""

    path = Path(artifact_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Interval artifact must decode to a JSON object.")
    if data.get("version") == BUNDLE_VERSION:
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("Interval artifact bundle must contain a non-empty 'artifacts' mapping.")
        return {
            str(name): NaiveIntervalCalibrator.from_dict(dict(calibrator_data))
            for name, calibrator_data in artifacts.items()
        }
    calibrator = NaiveIntervalCalibrator.from_dict(data)
    return {_artifact_key(calibrator): calibrator}


def _artifact_key(calibrator: NaiveIntervalCalibrator) -> str:
    if calibrator.entity_name == "branch" or calibrator.channel_names == ("line_loading",):
        return "line"
    return "volt"