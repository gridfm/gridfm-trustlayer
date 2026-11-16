"""Checks of the data passed to the calibrators, with messages that say what to fix.

`check_inputs` looks at `ChannelSet`s before fitting (predictions and true
values of the calibration set) or before predicting (predictions of new
scenarios, against a fitted calibrator). The calibrators refuse some of these
problems themselves; others (misaligned scenarios, NaN values, strata too small
for a finite bound) would pass silently and give misleading intervals.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from gridfm_trustlayer.conformal.calibrator import ChannelSet, NaiveIntervalCalibrator
from gridfm_trustlayer.conformal.kernel import KernelIntervalCalibrator

_EXAMPLES = 5


@dataclass
class InputCheck:
    """What `check_inputs` found.

    Attributes:
        errors: Problems to fix before fitting or predicting: the calibrators
            would refuse the data or give wrong intervals.
        warnings: Things that work but deserve a look, e.g. values that will
            get no interval.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings allowed)."""

        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise `ValueError` listing the errors, if there are any."""

        if self.errors:
            raise ValueError("\n".join(self.errors))

    def __str__(self) -> str:
        if not self.errors and not self.warnings:
            return "No problems found."
        return "\n".join(
            [f"error: {message}" for message in self.errors]
            + [f"warning: {message}" for message in self.warnings]
        )


def check_inputs(
    predictions: ChannelSet,
    true_values: ChannelSet | None = None,
    *,
    channel_names: Iterable[str] | None = None,
    strat: str | None = None,
    alpha: float | None = None,
    kernel: bool = False,
    calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator | None = None,
) -> InputCheck:
    """Check data for the calibrators and say what to fix.

    Before fitting, pass the calibration predictions and true values with the
    settings you will fit with; before predicting, pass the predictions of the
    new scenarios and the fitted `calibrator`, whose settings are then used.

    Args:
        predictions: Predictions, e.g. from `ChannelSet.from_frame`.
        true_values: True values of the same scenarios and elements
            (calibration data).
        channel_names: Variables to calibrate; all channels of `predictions`
            (or the calibrator's) when not given.
        strat: The stratification column you will fit with, if any.
        alpha: The miscoverage level you will fit with (0.1 when not given),
            to check that every stratum has enough samples for a finite bound.
        kernel: Whether you will fit `KernelIntervalCalibrator`, which needs
            one embedding vector per scenario.
        calibrator: A fitted calibrator, when checking new scenarios.

    Returns:
        An `InputCheck`: `ok` when there are no errors; `print` it to read the
        messages, or call `raise_for_errors()`.
    """

    check = InputCheck()
    if calibrator is not None:
        if strat is not None and strat != calibrator.strat:
            check.errors.append(
                f"strat {strat!r} differs from the calibrator's ({calibrator.strat!r}); "
                "leave strat out when passing a calibrator."
            )
        strat = calibrator.strat
        alpha = calibrator.alpha
        kernel = isinstance(calibrator, KernelIntervalCalibrator)
    level = 0.1 if alpha is None else float(alpha)
    for name, value in (("predictions", predictions), ("true_values", true_values)):
        if value is not None and not isinstance(value, ChannelSet):
            check.errors.append(
                f"{name} must be a ChannelSet (got {type(value).__name__}); build one "
                "with ChannelSet.from_frame."
            )
    if not check.ok:
        return check
    names = list(
        channel_names
        if channel_names is not None
        else calibrator.channel_names
        if calibrator is not None
        else predictions.channels
    )
    for which, data in (("predictions", predictions), ("true_values", true_values)):
        if data is None:
            continue
        missing = [name for name in names if name not in data.channels]
        if missing:
            check.errors.append(
                f"{which} has no channel {missing} (it has {sorted(data.channels)})."
            )
    _check_ids(check, predictions, "predictions")
    if true_values is not None:
        _check_ids(check, true_values, "true_values")
        _check_alignment(check, predictions, true_values)
    if not check.ok:
        return check

    fitting = true_values is not None and calibrator is None
    for name in names:
        _check_values(check, predictions, name, "predicted", fitting)
        if true_values is not None:
            _check_values(check, true_values, name, "true", fitting)
    if fitting:
        assert true_values is not None
        _check_calibration_counts(check, predictions, true_values, names, strat, level)
    if strat is not None:
        _check_strata(check, predictions, true_values, strat)
    if calibrator is not None:
        _check_against_calibrator(check, predictions, calibrator, names)
    if kernel:
        _check_embeddings(check, predictions, calibrator)
    return check


def _check_ids(check: InputCheck, data: ChannelSet, which: str) -> None:
    for kind, ids in (("scenario", data.scenario_ids), ("element", data.entity_ids)):
        array = np.asarray(ids)
        if not np.issubdtype(array.dtype, np.integer):
            check.errors.append(
                f"{which}: {kind} ids must be integers (they are {array.dtype}); "
                "number them, e.g. with pd.factorize, the same way in every dataset."
            )
        elif len(np.unique(array)) != len(array):
            check.errors.append(f"{which}: some {kind} ids appear more than once.")


def _check_alignment(
    check: InputCheck, predictions: ChannelSet, true_values: ChannelSet
) -> None:
    """Predictions and true values must describe the same cells, in the same order."""

    for kind, mine, theirs in (
        ("scenario", predictions.scenario_ids, true_values.scenario_ids),
        ("element", predictions.entity_ids, true_values.entity_ids),
    ):
        if np.array_equal(mine, theirs):
            continue
        if len(mine) == len(theirs) and np.array_equal(np.sort(mine), np.sort(theirs)):
            check.errors.append(
                f"predictions and true_values have the same {kind}s in a different "
                "order; build both from the same table (ChannelSet.from_frame sorts "
                "by id)."
            )
        else:
            only_predicted = np.setdiff1d(mine, theirs)
            only_true = np.setdiff1d(theirs, mine)
            check.errors.append(
                f"predictions and true_values have different {kind}s: "
                f"{len(only_predicted):,} only in predictions "
                f"(e.g. {only_predicted[:_EXAMPLES].tolist()}), {len(only_true):,} "
                f"only in true_values (e.g. {only_true[:_EXAMPLES].tolist()})."
            )
    if predictions.entity_name != true_values.entity_name:
        check.errors.append(
            f"entity_name differs: {predictions.entity_name!r} in predictions, "
            f"{true_values.entity_name!r} in true_values."
        )


def _valid(data: ChannelSet, name: str) -> np.ndarray:
    mask = data.valid_masks.get(name)
    shape = np.shape(data.channels[name])
    return np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


def _check_values(
    check: InputCheck, data: ChannelSet, name: str, kind: str, fitting: bool
) -> None:
    values = np.asarray(data.channels[name], dtype=float)
    broken = ~np.isfinite(values) & _valid(data, name)
    if broken.any():
        consequence = (
            "they are left out of the fit"
            if fitting
            else "their bounds will not be finite"
        )
        check.warnings.append(
            f"{name}: {int(broken.sum()):,} {kind} values are NaN or infinite but "
            f"marked valid; {consequence}. Mark them invalid (valid_masks, or the "
            "valid column of from_frame) if they are not meant to be calibrated."
        )


def _check_calibration_counts(
    check: InputCheck,
    predictions: ChannelSet,
    true_values: ChannelSet,
    names: list[str],
    strat: str | None,
    alpha: float,
) -> None:
    """Enough valid calibration values per element (and stratum) for a finite bound."""

    needed = math.ceil(2.0 / alpha) - 1
    strata = (
        np.zeros(len(predictions.scenario_ids), dtype=np.int64)
        if strat is None or strat not in predictions.scenario_columns
        else np.asarray(predictions.scenario_columns[strat])
    )
    for name in names:
        valid = (
            _valid(predictions, name)
            & _valid(true_values, name)
            & np.isfinite(np.asarray(predictions.channels[name], dtype=float))
            & np.isfinite(np.asarray(true_values.channels[name], dtype=float))
        )
        never = np.flatnonzero(~valid.any(axis=0))
        if never.size:
            check.warnings.append(
                f"{name}: {never.size:,} elements have no valid calibration value "
                f"(e.g. ids {np.asarray(predictions.entity_ids)[never][:_EXAMPLES].tolist()}): "
                "they get no finite bound."
            )
        overall = valid.sum(axis=0)
        short_overall = int(((overall > 0) & (overall < needed)).sum())
        if short_overall:
            check.warnings.append(
                f"{name}: {short_overall:,} elements have fewer than {needed} valid "
                f"calibration values in total, the minimum for a finite bound at alpha "
                f"{alpha:g}: their bounds will be infinite. Add calibration scenarios."
            )
        if strat is None or strat not in predictions.scenario_columns:
            continue
        sparse = []
        for value in np.unique(strata):
            counts = valid[strata == value].sum(axis=0)
            short = int(((counts < needed) & (overall >= needed)).sum())
            if short:
                sparse.append(f"{strat}={value}: {short:,}")
        if sparse:
            check.warnings.append(
                f"{name}: elements with fewer than {needed} valid calibration values "
                f"in a stratum ({'; '.join(sparse)}) get their quantile over all "
                "strata there (sparse_strata='pooled'), without a guarantee within "
                "that stratum."
            )


def _check_strata(
    check: InputCheck,
    predictions: ChannelSet,
    true_values: ChannelSet | None,
    strat: str,
) -> None:
    for which, data in (("predictions", predictions), ("true_values", true_values)):
        if data is None:
            continue
        if strat not in data.scenario_columns:
            check.errors.append(
                f"{which} has no stratum column {strat!r} (it has "
                f"{sorted(data.scenario_columns)}); pass strata=[{strat!r}] to "
                "ChannelSet.from_frame, or set scenario_columns."
            )
            continue
        values = np.asarray(data.scenario_columns[strat])
        if not np.issubdtype(values.dtype, np.integer):
            check.errors.append(
                f"{which}: the stratum column {strat!r} must hold integers "
                f"(it holds {values.dtype})."
            )
    if (
        true_values is not None
        and strat in predictions.scenario_columns
        and strat in true_values.scenario_columns
        and not np.array_equal(
            np.asarray(predictions.scenario_columns[strat]),
            np.asarray(true_values.scenario_columns[strat]),
        )
    ):
        check.errors.append(
            f"The stratum column {strat!r} differs between predictions and true_values."
        )


def _check_against_calibrator(
    check: InputCheck,
    predictions: ChannelSet,
    calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator,
    names: list[str],
) -> None:
    fitted = np.asarray(calibrator.entity_ids)
    given = np.asarray(predictions.entity_ids)
    if not np.array_equal(fitted, given):
        if len(fitted) == len(given) and np.array_equal(
            np.sort(fitted), np.sort(given)
        ):
            check.errors.append(
                "The elements are those the calibrator was fitted on, in a different "
                "order; build the new data the same way (from_frame sorts by id)."
            )
        else:
            check.errors.append(
                "The elements differ from those the calibrator was fitted on: "
                f"{len(np.setdiff1d(fitted, given)):,} missing, "
                f"{len(np.setdiff1d(given, fitted)):,} unknown. Every scenario must "
                "list the same elements; mark absent ones invalid instead of "
                "dropping them."
            )
    if predictions.entity_name != calibrator.entity_name:
        check.errors.append(
            f"entity_name is {predictions.entity_name!r}; the calibrator was fitted "
            f"on {calibrator.entity_name!r}."
        )
    unknown = [name for name in names if name not in calibrator.channel_names]
    if unknown:
        check.errors.append(
            f"The calibrator was not fitted for {unknown} (it calibrates "
            f"{list(calibrator.channel_names)})."
        )
    strat = calibrator.strat
    if strat is None or strat not in predictions.scenario_columns:
        return
    known = getattr(calibrator, "stratification_values", None)
    if known is None:
        return
    values = np.asarray(predictions.scenario_columns[strat])
    absent = np.setdiff1d(np.unique(values), np.asarray(known))
    if absent.size:
        count = int(np.isin(values, absent).sum())
        check.warnings.append(
            f"{count:,} scenarios are in strata absent from the calibration data "
            f"({strat} = {absent[:_EXAMPLES].tolist()}): their bounds come from all "
            "strata pooled (sparse_strata='pooled') or are infinite."
        )


def _check_embeddings(
    check: InputCheck,
    predictions: ChannelSet,
    calibrator: NaiveIntervalCalibrator | KernelIntervalCalibrator | None,
) -> None:
    embeddings = predictions.scenario_embeddings
    if embeddings is None:
        check.errors.append(
            "cp_type kernel needs one embedding vector per scenario: pass "
            "embeddings= to ChannelSet.from_frame (or set scenario_embeddings)."
        )
        return
    array = np.asarray(embeddings)
    if array.ndim != 2:
        check.errors.append(
            f"The embeddings must be a (n_scenarios, d) array; they have shape "
            f"{array.shape}."
        )
        return
    if not np.issubdtype(array.dtype, np.number):
        check.errors.append(f"The embeddings must be numeric (they are {array.dtype}).")
        return
    broken = np.flatnonzero(~np.isfinite(array).all(axis=1))
    if broken.size:
        check.errors.append(
            f"The embeddings of {broken.size:,} scenarios contain NaN or infinite "
            f"values (e.g. ids {np.asarray(predictions.scenario_ids)[broken][:_EXAMPLES].tolist()})."
        )
    elif len(array) > 1 and np.all(array == array[0]):
        check.errors.append(
            "Every scenario has the same embedding vector: the kernel cannot tell "
            "scenarios apart."
        )
    if isinstance(calibrator, KernelIntervalCalibrator):
        width = calibrator.calibration_embeddings.shape[1]
        if array.shape[1] != width:
            check.errors.append(
                f"The embeddings have {array.shape[1]} dimensions; the calibrator "
                f"was fitted on {width}. Build them the same way (same model, same "
                "pooling) for calibration and new scenarios."
            )
