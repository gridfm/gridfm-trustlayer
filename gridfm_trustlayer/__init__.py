"""Conformal trustworthiness layer for GridFM foundation models.

The public Python API is re-exported here; import from the top-level package
rather than from submodules, whose layout may change.

Calibrators:
    NaiveIntervalCalibrator   stratified split-conformal intervals (``cp_type: element``)
    KernelIntervalCalibrator  kernel-weighted conformal intervals (``cp_type: kernel``)
    ChannelSet                aligned (scenario x entity) arrays to calibrate

Data loading:
    load_predictions, load_ground_truth, load_scenario_embeddings, compute_line_loading

Artifacts:
    save_interval_artifact, load_interval_artifact, load_interval_artifacts

Screening:
    classify_status, scenario_status  status of values and operating scenarios
"""

from gridfm_trustlayer._version import __version__
from gridfm_trustlayer.conformal import (
    ChannelSet,
    KernelIntervalCalibrator,
    NaiveIntervalCalibrator,
)
from gridfm_trustlayer.io import (
    GroundTruthSet,
    PredictionSet,
    compute_line_loading,
    load_ground_truth,
    load_predictions,
)
from gridfm_trustlayer.io.artifact import (
    load_interval_artifact,
    load_interval_artifacts,
    save_interval_artifact,
)
from gridfm_trustlayer.io.embeddings import load_scenario_embeddings
from gridfm_trustlayer.screening import classify_status, scenario_status

__all__ = [
    "ChannelSet",
    "GroundTruthSet",
    "KernelIntervalCalibrator",
    "NaiveIntervalCalibrator",
    "PredictionSet",
    "__version__",
    "classify_status",
    "compute_line_loading",
    "load_ground_truth",
    "load_interval_artifact",
    "load_interval_artifacts",
    "load_predictions",
    "load_scenario_embeddings",
    "save_interval_artifact",
    "scenario_status",
]
