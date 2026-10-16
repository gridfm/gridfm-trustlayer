"""`compute`: fit the layer on the calibration set and write the artifact."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.cli import common
from gridfm_trustlayer.cli.common import (
    _FINGERPRINT_SCENARIOS,
    _SET_POINTS,
    _artifact_path,
    _block_jobs,
    _build_scenario_columns,
    _build_target_channel_sets,
    _compute_system_load_bin_edges,
    _EmbeddingSource,
    _n_jobs,
    _partition_variables,
    _prediction_fingerprint,
    _report_progress,
    _sparse_strata_listing,
    _with_scenario_columns,
)
from gridfm_trustlayer.cli.config import (
    N_EFF_TARGET_AUTO,
    coalesce_option,
    load_cli_config,
    normalize_variables,
    resolve_command_config,
    validate_bandwidth,
    validate_command_config,
    validate_cp_type,
    validate_distance,
    validate_k_nn,
    validate_n_eff_min,
    validate_n_eff_target,
    validate_pred_n_bins,
    validate_strat,
    validate_system_load_n_bins,
)
from gridfm_trustlayer.conformal import ChannelSet, NaiveIntervalCalibrator
from gridfm_trustlayer.conformal.calibrator import (
    DEFAULT_SPARSE_STRATA,
    validate_sparse_strata,
)
from gridfm_trustlayer.conformal.kernel import (
    DEFAULT_BANDWIDTH,
    DEFAULT_DISTANCE,
    DEFAULT_K_NN,
    DEFAULT_N_EFF_MIN,
    KernelIntervalCalibrator,
)
from gridfm_trustlayer.conformal.tuning import (
    BandwidthTuningResult,
    TuningOptions,
    tune_n_eff_target,
)
from gridfm_trustlayer.io import (
    load_ground_truth,
    load_predictions,
)
from gridfm_trustlayer.io.artifact import (
    is_interval_artifact,
    save_interval_artifacts,
)
from gridfm_trustlayer.io.embeddings import (
    DEFAULT_POOLING,
    validate_pooling,
)
from gridfm_trustlayer.io.ground_truth import (
    GroundTruthReference,
    scan_ground_truth_reference,
)
from gridfm_trustlayer.io.parquet import (
    open_dataset,
    read_scenario_ids,
    rows_per_scenario,
    scenario_blocks,
)
from gridfm_trustlayer.parallel import block_rows, map_blocks


def _run_compute(args: argparse.Namespace) -> None:
    config = resolve_command_config(load_cli_config(args.config), "compute")
    validate_command_config("compute", config)
    cp_type = validate_cp_type(coalesce_option(args.cp_type, config, "cp_type"))
    strat = validate_strat(coalesce_option(args.strat, config, "strat"))
    system_load_n_bins = validate_system_load_n_bins(
        coalesce_option(args.system_load_n_bins, config, "system_load_n_bins")
    )
    pred_n_bins = validate_pred_n_bins(
        coalesce_option(args.pred_n_bins, config, "pred_n_bins")
    )
    if strat == "system_load_bin" and system_load_n_bins is None:
        raise ValueError(
            "system_load_n_bins is required when strat is 'system_load_bin'."
        )
    if strat == "pred_bins" and pred_n_bins is None:
        raise ValueError("pred_n_bins is required when strat is 'pred_bins'.")
    sparse_strata = validate_sparse_strata(
        str(
            coalesce_option(args.sparse_strata, config, "sparse_strata")
            or DEFAULT_SPARSE_STRATA
        )
    )
    variables = normalize_variables(
        coalesce_option(args.variables, config, "variables")
    )
    variable_groups = _partition_variables(variables)
    embedding_pooling = validate_pooling(
        coalesce_option(args.embedding_pooling, config, "embedding_pooling")
        or DEFAULT_POOLING
    )
    k_nn = validate_k_nn(coalesce_option(args.k_nn, config, "k_nn")) or DEFAULT_K_NN
    distance = validate_distance(
        coalesce_option(args.distance, config, "distance") or DEFAULT_DISTANCE
    )
    bandwidth = validate_bandwidth(
        coalesce_option(args.bandwidth, config, "bandwidth") or DEFAULT_BANDWIDTH
    )
    n_eff_target = validate_n_eff_target(
        coalesce_option(args.n_eff_target, config, "n_eff_target")
    )
    n_eff_min = (
        validate_n_eff_min(coalesce_option(args.n_eff_min, config, "n_eff_min"))
        or DEFAULT_N_EFF_MIN
    )
    tuning_options = (
        TuningOptions.from_config(config.get("tuning"))
        if n_eff_target == N_EFF_TARGET_AUTO
        else None
    )
    if tuning_options is not None and (cp_type != "kernel" or bandwidth != "n_eff"):
        raise ValueError(
            "n_eff_target auto requires cp_type kernel and bandwidth n_eff."
        )
    fixed_n_eff_target = n_eff_target if isinstance(n_eff_target, float) else None
    predictions_path = coalesce_option(
        args.predictions, config, "predictions", required=True
    )
    task_name = coalesce_option(args.task_name, config, "task_name")
    generator_path = coalesce_option(
        args.generator_predictions, config, "generator_predictions"
    )
    pf_node_path = coalesce_option(args.pf_node, config, "pf_node", required=True)
    pf_edge_path = coalesce_option(args.pf_edge, config, "pf_edge", required=True)
    scenario_ids = read_scenario_ids(predictions_path)
    if scenario_ids.size == 0:
        raise ValueError(f"{predictions_path} contains no scenarios.")
    embeddings = (
        (
            str(coalesce_option(args.embeddings, config, "embeddings", required=True)),
            embedding_pooling,
        )
        if cp_type == "kernel"
        else None
    )
    # Checked before the data is read, so that a wrong path fails at once.
    artifact_path = _artifact_path(
        args, config, explicit=args.output, for_reading=False
    )
    _check_artifact_target(
        artifact_path,
        inputs=(
            predictions_path,
            pf_node_path,
            pf_edge_path,
            generator_path,
            None if embeddings is None else embeddings[0],
        ),
    )
    data = _load_calibration_data(
        predictions_path,
        pf_node_path,
        pf_edge_path,
        variable_groups,
        scenario_ids,
        task_name=task_name,
        generator_path=generator_path,
        embeddings=embeddings,
        n_jobs=_block_jobs(args, config),
    )
    # The first scenarios identify the dataset, so that `calibrate` can refuse it.
    sample = load_predictions(
        predictions_path,
        task_name=task_name,
        generator_parquet_path=generator_path,
        scenario_ids=scenario_ids[:_FINGERPRINT_SCENARIOS],
    )
    if sample.task_name not in _SET_POINTS:
        # A misspelled PowerFlow would otherwise fit the set-points silently.
        print(
            f"compute: task_name {sample.task_name!r} is not PowerFlow or "
            "OptimalPowerFlow: the predictions are read as they are, with no "
            "known set-points set aside. Check the spelling if you meant one of them.",
            flush=True,
        )
    tuning_groups = (
        None
        if tuning_options is None
        else _load_tuning_groups(pf_node_path, tuning_options.group_by)
    )
    # Every scenario of the calibration dataset is used for fitting; the layer is
    # evaluated by `calibrate` on a separate test dataset.
    fingerprint = _prediction_fingerprint(scenario_ids, sample)
    calibrators: dict[str, NaiveIntervalCalibrator | KernelIntervalCalibrator] = {}
    tuning_results: dict[str, BandwidthTuningResult] = {}
    for target_kind, target_variables in variable_groups.items():
        base_predictions = data.predictions[target_kind]
        system_load_bin_edges = None
        if strat == "system_load_bin":
            assert system_load_n_bins is not None
            system_load_bin_edges = _compute_system_load_bin_edges(
                data.scenarios.system_load,
                system_load_n_bins,
            )
        pred_bin_edges = None
        if strat == "pred_bins":
            assert pred_n_bins is not None
            pred_bin_edges = _compute_prediction_bin_edges(
                base_predictions,
                pred_n_bins,
            )
        scenario_columns = _build_scenario_columns(
            base_predictions,
            data.scenarios,
            strat,
            system_load_n_bins=system_load_n_bins,
            system_load_bin_edges=system_load_bin_edges,
            pred_n_bins=pred_n_bins,
            pred_bin_edges=pred_bin_edges,
        )
        calibration_predictions = _with_scenario_columns(
            base_predictions, scenario_columns
        )
        calibration_ground_truth = _with_scenario_columns(
            data.ground_truth[target_kind], scenario_columns
        )
        alpha = float(coalesce_option(args.alpha, config, "alpha", required=True))
        calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator
        if cp_type == "kernel":
            assert calibration_predictions.scenario_embeddings is not None
            family_n_eff_target = fixed_n_eff_target
            family_distance = distance
            if tuning_options is not None:
                # Cross-validated on the calibration data only; the test set is untouched.
                tuning_results[target_kind] = tune_n_eff_target(
                    calibration_predictions,
                    calibration_ground_truth,
                    alpha=alpha,
                    channel_names=target_variables,
                    strat=strat,
                    embedding_pooling=embedding_pooling,
                    distance=distance,
                    n_eff_min=n_eff_min,
                    options=tuning_options,
                    groups=None
                    if tuning_groups is None
                    else tuning_groups.reindex(
                        calibration_predictions.scenario_ids
                    ).to_numpy(),
                    system_load_n_bins=system_load_n_bins,
                    system_load_bin_edges=system_load_bin_edges,
                    sparse_strata=sparse_strata,
                    n_jobs=_n_jobs(args, config),
                )
                family_n_eff_target = tuning_results[target_kind].selected_n_eff_target
                family_distance = tuning_results[target_kind].selected_distance
            calibrator = KernelIntervalCalibrator.fit(
                calibration_predictions,
                calibration_ground_truth,
                alpha=alpha,
                channel_names=target_variables,
                strat=strat,
                embeddings=calibration_predictions.scenario_embeddings,
                embedding_pooling=embedding_pooling,
                k_nn=k_nn,
                n_eff_min=n_eff_min,
                bandwidth=bandwidth,
                n_eff_target=family_n_eff_target,
                distance=family_distance,
                system_load_n_bins=system_load_n_bins,
                system_load_bin_edges=system_load_bin_edges,
                sparse_strata=sparse_strata,
            )
        else:
            calibrator = NaiveIntervalCalibrator.fit(
                calibration_predictions,
                calibration_ground_truth,
                alpha=alpha,
                channel_names=target_variables,
                strat=strat,
                system_load_n_bins=system_load_n_bins,
                system_load_bin_edges=system_load_bin_edges,
                pred_n_bins=pred_n_bins,
                pred_bin_edges=pred_bin_edges,
                sparse_strata=sparse_strata,
            )
        calibrator.calibration_fingerprint = fingerprint
        calibrator.ground_truth_reference = data.reference.to_dict()
        if (
            isinstance(calibrator, KernelIntervalCalibrator)
            and target_kind in tuning_results
        ):
            calibrator.tuning = tuning_results[target_kind].summary()
        calibrators[target_kind] = calibrator
    save_interval_artifacts(calibrators, artifact_path)
    if tuning_results:
        _write_tuning_reports(tuning_results, artifact_path)
    method = cp_type if strat is None else f"{cp_type}, strat {strat}"
    print(
        f"compute: fitted {method} for {', '.join(variables)} on "
        f"{len(scenario_ids):,} calibration scenarios; "
        f"artifact: {artifact_path}",
        flush=True,
    )
    _report_sparse_strata(calibrators, sparse_strata)


@dataclass
class _CalibrationData:
    """The calibration set, per target family, in (scenario x element) arrays."""

    predictions: dict[str, ChannelSet]
    ground_truth: dict[str, ChannelSet]
    # Per-scenario quantities the strata are built from (outages, total load).
    scenarios: SimpleNamespace
    # Elements in service and angle units of the calibration set, stored in the
    # artifact for the other commands.
    reference: GroundTruthReference


def _load_calibration_data(
    predictions_path: str,
    pf_node_path: str,
    pf_edge_path: str,
    variable_groups: Mapping[str, tuple[str, ...]],
    scenario_ids: np.ndarray,
    *,
    task_name: str | None,
    generator_path: str | None,
    embeddings: tuple[str, str] | None,
    n_jobs: int | None,
) -> _CalibrationData:
    """Read the calibration set one block of scenarios at a time, in parallel.

    Only the arrays the calibrators need are kept: predictions, true values and
    valid masks per variable, embeddings, and the per-scenario quantities of the
    strata. Strata and bin edges are built afterwards, over the whole set.
    """

    rows = rows_per_scenario(predictions_path, len(scenario_ids))
    blocks = scenario_blocks(
        scenario_ids,
        rows,
        block_rows(common._BLOCK_ROWS, rows * len(scenario_ids), n_jobs),
    )
    # Outage counts and angle units are judged over the whole dataset.
    reference = scan_ground_truth_reference(pf_node_path, pf_edge_path)
    job = _ComputeJob(
        predictions_path=predictions_path,
        pf_node_path=pf_node_path,
        pf_edge_path=pf_edge_path,
        variable_groups=dict(variable_groups),
        task_name=task_name,
        generator_path=generator_path,
        embeddings=embeddings,
        scenario_ids=scenario_ids,
        reference=reference,
    )
    results = map_blocks(
        job,
        blocks,
        n_jobs=n_jobs,
        progress=partial(_report_progress, "compute", blocks=blocks),
    )
    return _CalibrationData(
        predictions={
            kind: _concatenate([result[0][kind] for result in results])
            for kind in variable_groups
        },
        ground_truth={
            kind: _concatenate([result[1][kind] for result in results])
            for kind in variable_groups
        },
        scenarios=SimpleNamespace(
            **{
                name: np.concatenate([result[2][name] for result in results])
                for name in _SCENARIO_QUANTITIES
            }
        ),
        reference=reference,
    )


_SCENARIO_QUANTITIES = ("n_outages", "outage_type", "system_load")


@dataclass
class _ComputeJob:
    """`compute` reading one block of calibration scenarios (see `parallel.BlockJob`).

    Returns the block's predictions and true values per target family, and its
    per-scenario quantities.
    """

    predictions_path: str
    pf_node_path: str
    pf_edge_path: str
    variable_groups: dict[str, tuple[str, ...]]
    task_name: str | None
    generator_path: str | None
    embeddings: tuple[str, str] | None
    scenario_ids: np.ndarray
    reference: GroundTruthReference
    _embeddings: _EmbeddingSource | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_embeddings": None}

    def setup(self, threads: int | None) -> None:
        if self.embeddings is not None:
            path, pooling = self.embeddings
            self._embeddings = _EmbeddingSource(
                path, pooling, self.scenario_ids, preload=threads is None
            )

    def run(
        self, index: int, block: np.ndarray
    ) -> tuple[dict[str, ChannelSet], dict[str, ChannelSet], dict[str, np.ndarray]]:
        predictions = load_predictions(
            self.predictions_path,
            task_name=self.task_name,
            generator_parquet_path=self.generator_path,
            scenario_ids=block,
        )
        if self._embeddings is not None:
            self._embeddings.attach(predictions)
        ground_truth = load_ground_truth(
            self.pf_node_path,
            self.pf_edge_path,
            scenario_ids=predictions.scenario_ids,
            reference=self.reference,
        )
        predicted: dict[str, ChannelSet] = {}
        true: dict[str, ChannelSet] = {}
        for kind, names in self.variable_groups.items():
            predicted[kind], true[kind] = _build_target_channel_sets(
                predictions, ground_truth, names
            )
        quantities = {
            name: np.asarray(getattr(ground_truth, name))
            for name in _SCENARIO_QUANTITIES
        }
        return predicted, true, quantities


def _concatenate(parts: list[ChannelSet]) -> ChannelSet:
    """Blocks of scenarios with the same elements, as one `ChannelSet`."""

    first = parts[0]
    for part in parts[1:]:
        if not np.array_equal(part.entity_ids, first.entity_ids):
            raise ValueError(
                f"The {first.entity_name}es differ between scenarios of the calibration set."
            )
    return ChannelSet(
        scenario_ids=np.concatenate([part.scenario_ids for part in parts]),
        entity_ids=first.entity_ids,
        entity_name=first.entity_name,
        channels={
            name: np.concatenate([part.channels[name] for part in parts])
            for name in first.channels
        },
        valid_masks={
            name: np.concatenate([part.valid_masks[name] for part in parts])
            for name in first.valid_masks
        },
        entity_columns=first.entity_columns,
        scenario_embeddings=None
        if first.scenario_embeddings is None
        else np.concatenate([part.scenario_embeddings for part in parts]),
    )


def _report_sparse_strata(
    calibrators: Mapping[str, NaiveIntervalCalibrator | KernelIntervalCalibrator],
    sparse_strata: str,
) -> None:
    """Which strata have elements with too few calibration samples, and what they get."""

    sparse = _sparse_strata_listing(calibrators)
    if not sparse:
        return
    outcome = (
        "they use their quantile over all calibration scenarios (sparse_strata: pooled)"
        if sparse_strata == "pooled"
        else "they get infinite bounds (sparse_strata: infinite)"
    )
    print(
        f"compute: too few calibration samples for a finite bound: {sparse}; {outcome}",
        flush=True,
    )


def _load_tuning_groups(pf_node_path: str, group_by: str) -> pd.Series | None:
    """Fold groups per scenario for `n_eff_target: auto`, or None for scenario-level folds.

    Datakit writes `load_scenario_idx`: scenarios generated from the same load
    profile (different topologies) share it. Keeping them in one fold prevents
    near-duplicates on both sides of a split from flattering the kernel.
    """

    if group_by == "none":
        return None
    dataset = open_dataset(pf_node_path)
    if "load_scenario_idx" not in dataset.schema.names:
        if group_by == "load_scenario_idx":
            raise ValueError(
                "tuning.group_by is 'load_scenario_idx' but the calibration pf_node "
                "parquet has no load_scenario_idx column.",
            )
        return None
    frame = dataset.to_table(columns=["scenario", "load_scenario_idx"]).to_pandas()
    return frame.groupby("scenario")["load_scenario_idx"].first()


def _write_tuning_reports(
    results: dict[str, BandwidthTuningResult], artifact_path: str
) -> None:
    """Write the per-candidate and per-check tables next to the artifact."""

    stem = Path(artifact_path)
    candidates = pd.concat(
        [result.candidates.assign(target=kind) for kind, result in results.items()],
        ignore_index=True,
    )
    details = pd.concat(
        [result.details.assign(target=kind) for kind, result in results.items()],
        ignore_index=True,
    )
    summary_path = stem.with_name(f"{stem.stem}_kernel_tuning.csv")
    details_path = stem.with_name(f"{stem.stem}_kernel_tuning_details.csv")
    candidates.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    for kind, result in results.items():
        print(
            f"kernel tuning ({kind}): n_eff_target = {result.selected_n_eff_target:g}, "
            f"distance = {result.selected_distance} ({result.reason}; "
            f"{len(result.candidates)} candidates, {result.options.search} search, "
            f"{result.options.folds} {'grouped ' if result.grouped else ''}folds). "
            f"Report: {summary_path}"
        )


def _compute_prediction_bin_edges(
    predictions: ChannelSet,
    n_bins: int,
) -> dict[str, np.ndarray]:
    return {
        channel_name: _compute_system_load_bin_edges(
            _flatten_prediction_bin_values(
                channel_values,
                predictions.valid_masks.get(channel_name),
                channel_name=channel_name,
            ),
            n_bins,
            stratification=f"pred_bins ({channel_name})",
        )
        for channel_name, channel_values in predictions.channels.items()
    }


def _flatten_prediction_bin_values(
    prediction_values: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    channel_name: str,
) -> np.ndarray:
    values = np.asarray(prediction_values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if valid_mask is not None:
        finite_mask &= valid_mask
    flattened = values[finite_mask]
    if flattened.size == 0:
        raise ValueError(
            f"pred_bins stratification requires at least one finite prediction value for {channel_name}."
        )
    return flattened.astype(np.float32, copy=False)


def _check_artifact_target(path: str, *, inputs: tuple[object, ...]) -> None:
    """Refuse to write the artifact over an input, a directory or another file."""

    target = Path(path)
    if any(
        source is not None and Path(str(source)).resolve() == target.resolve()
        for source in inputs
    ):
        raise ValueError(
            f"The artifact path {path} is also an input of compute; choose another path."
        )
    if target.is_dir():
        raise ValueError(
            f"The artifact path {path} is a directory; give a file name, e.g. "
            f"{target / 'layer.trustlayer'}."
        )
    if target.exists() and not is_interval_artifact(target):
        raise ValueError(
            f"{path} exists and is not a trustlayer artifact: it would be replaced. "
            "Choose another path or remove it."
        )
