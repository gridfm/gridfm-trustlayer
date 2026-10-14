"""Derived metric computation from aligned prediction and ground-truth arrays."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from gridfm_trustlayer.io.ground_truth import GroundTruthSet


def compute_line_loading(
    Vm: np.ndarray,
    Va_rad: np.ndarray,
    gt: GroundTruthSet,
) -> np.ndarray:
    """Line loading implied by bus voltages: `max(|S_from|, |S_to|) / rate_a`.

    The AC branch flows are computed by datakit's branch-flow routine, so
    `gridfm_datakit` must be installed (the `graphkit` extra).

    Args:
        Vm: Voltage magnitudes, `(n_scenarios, n_buses)`, predicted or true.
        Va_rad: Voltage angles in radians, same shape.
        gt: Ground truth of the same scenarios, for the branch parameters.

    Returns:
        `(n_scenarios, n_branches)` loading as a fraction of `rate_a`; `-1` for
        out-of-service or unrated branches.
    """

    try:
        power_balance = import_module("gridfm_datakit.utils.power_balance")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gridfm_datakit is required for compute_line_loading(). Install the "
            'graphkit extra: pip install "gridfm-trustlayer[graphkit]".',
        ) from exc
    compute_branch_powers_vectorized = cast(
        Any,
        power_balance.compute_branch_powers_vectorized,
    )

    if Vm.shape != (len(gt.scenario_ids), len(gt.bus_ids)):
        raise ValueError(
            f"Vm shape {Vm.shape} does not match ground-truth layout "
            f"({len(gt.scenario_ids)}, {len(gt.bus_ids)}).",
        )
    if Va_rad.shape != Vm.shape:
        raise ValueError(
            f"Va_rad shape {Va_rad.shape} does not match Vm shape {Vm.shape}."
        )

    bus_df = _build_bus_voltage_frame(gt, Vm, Va_rad)
    branch_df = _build_branch_parameter_frame(gt)

    pf, qf, pt, qt = compute_branch_powers_vectorized(
        branch_df,
        bus_df,
        dc=False,
        sn_mva=gt.sn_mva,
    )

    s_from = np.sqrt(pf**2 + qf**2)
    s_to = np.sqrt(pt**2 + qt**2)
    rate = branch_df["rate_a"].to_numpy(dtype=np.float64, copy=False)
    status = branch_df["br_status"].to_numpy(dtype=np.float64, copy=False)

    loading = np.maximum(s_from, s_to) / rate
    loading[(status != 1.0) | (rate <= 0.0)] = -1.0
    return loading.reshape(len(gt.scenario_ids), len(gt.branch_ids)).astype(np.float32)


def _build_bus_voltage_frame(
    gt: GroundTruthSet,
    Vm: np.ndarray,
    Va_rad: np.ndarray,
) -> pd.DataFrame:
    scenario_ids = np.repeat(
        gt.scenario_ids.astype(np.int64, copy=False), len(gt.bus_ids)
    )
    bus_ids = np.tile(gt.bus_ids.astype(np.int64, copy=False), len(gt.scenario_ids))
    return pd.DataFrame(
        {
            "scenario": scenario_ids,
            "bus": bus_ids,
            "Vm": Vm.reshape(-1).astype(np.float64, copy=False),
            "Va": np.rad2deg(Va_rad.reshape(-1).astype(np.float64, copy=False)),
        },
    )


def _build_branch_parameter_frame(gt: GroundTruthSet) -> pd.DataFrame:
    scenario_ids = np.repeat(
        gt.scenario_ids.astype(np.int64, copy=False),
        len(gt.branch_ids),
    )
    branch_ids = np.tile(
        gt.branch_ids.astype(np.int64, copy=False),
        len(gt.scenario_ids),
    )
    f_bus = np.tile(gt.f_bus.astype(np.int64, copy=False), len(gt.scenario_ids))
    t_bus = np.tile(gt.t_bus.astype(np.int64, copy=False), len(gt.scenario_ids))
    rate_a = np.tile(gt.rate_a.astype(np.float64, copy=False), len(gt.scenario_ids))
    br_status = (~gt.removed_mask).reshape(-1).astype(np.float64, copy=False)

    frame = pd.DataFrame(
        {
            "scenario": scenario_ids,
            "idx": branch_ids,
            "from_bus": f_bus,
            "to_bus": t_bus,
            "rate_a": rate_a,
            "br_status": br_status,
        },
    )
    for key, values in gt.admittances.items():
        frame[key] = values.reshape(-1).astype(np.float64, copy=False)
    return frame
