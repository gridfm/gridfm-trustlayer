"""Derived metric computation from aligned prediction and ground-truth arrays."""

from __future__ import annotations

import numpy as np

from gridfm_trustlayer.io.ground_truth import GroundTruthSet

_ADMITTANCES = ("Yff", "Yft", "Ytf", "Ytt")


def compute_line_loading(
    Vm: np.ndarray,
    Va_rad: np.ndarray,
    gt: GroundTruthSet,
) -> np.ndarray:
    """Line loading implied by bus voltages: `max(|S_from|, |S_to|) / rate_a`.

    The AC branch flows use the formula of datakit's branch-flow routine
    (`gridfm_datakit.utils.power_balance.compute_branch_powers_vectorized`),
    operation by operation, so the result is the same; datakit itself is not
    needed.

    Args:
        Vm: Voltage magnitudes, `(n_scenarios, n_buses)`, predicted or true.
        Va_rad: Voltage angles in radians, same shape.
        gt: Ground truth of the same scenarios, for the branch parameters.

    Returns:
        `(n_scenarios, n_branches)` loading as a fraction of `rate_a`; `-1` for
        out-of-service or unrated branches.
    """

    if Vm.shape != (len(gt.scenario_ids), len(gt.bus_ids)):
        raise ValueError(
            f"Vm shape {Vm.shape} does not match ground-truth layout "
            f"({len(gt.scenario_ids)}, {len(gt.bus_ids)}).",
        )
    if Va_rad.shape != Vm.shape:
        raise ValueError(
            f"Va_rad shape {Va_rad.shape} does not match Vm shape {Vm.shape}."
        )

    from_column = _bus_columns(gt, gt.f_bus, "from")
    to_column = _bus_columns(gt, gt.t_bus, "to")
    # Angles go through degrees and back, as in datakit's tables.
    va = np.radians(np.rad2deg(Va_rad.astype(np.float64, copy=False)))
    voltage = Vm.astype(np.float64, copy=False) * (np.cos(va) + 1j * np.sin(va))
    v_from = voltage[:, from_column]
    v_to = voltage[:, to_column]
    y = {
        name: gt.admittances[f"{name}_r"].astype(np.float64, copy=False)
        + 1j * gt.admittances[f"{name}_i"].astype(np.float64, copy=False)
        for name in _ADMITTANCES
    }
    current_from = y["Yff"] * v_from + y["Yft"] * v_to
    current_to = y["Ytt"] * v_to + y["Ytf"] * v_from
    s_from_complex = v_from * np.conj(current_from) * gt.sn_mva
    s_to_complex = v_to * np.conj(current_to) * gt.sn_mva

    s_from = np.sqrt(np.real(s_from_complex) ** 2 + np.imag(s_from_complex) ** 2)
    s_to = np.sqrt(np.real(s_to_complex) ** 2 + np.imag(s_to_complex) ** 2)
    rate = np.broadcast_to(gt.rate_a.astype(np.float64, copy=False), s_from.shape)

    # Unrated branches (rate 0) divide by zero here and are marked just below.
    with np.errstate(divide="ignore", invalid="ignore"):
        loading = np.maximum(s_from, s_to) / rate
    loading[np.asarray(gt.removed_mask, dtype=bool) | (rate <= 0.0)] = -1.0
    return loading.astype(np.float32)


def _bus_columns(gt: GroundTruthSet, buses: np.ndarray, end: str) -> np.ndarray:
    """Column of each branch end's bus in the `(n_scenarios, n_buses)` arrays."""

    bus_ids = np.asarray(gt.bus_ids, dtype=np.int64)
    wanted = np.asarray(buses, dtype=np.int64)
    if wanted.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(bus_ids, kind="stable")
    position = np.searchsorted(bus_ids, wanted, sorter=order)
    position = np.clip(position, 0, max(len(bus_ids) - 1, 0))
    columns = order[position] if len(bus_ids) else position
    found = len(bus_ids) > 0 and np.array_equal(bus_ids[columns], wanted)
    if not found:
        missing = np.setdiff1d(wanted, bus_ids)
        raise ValueError(
            f"Branch {end}-buses {missing[:5].tolist()} are not among the buses "
            "of the ground truth.",
        )
    return np.asarray(columns, dtype=np.int64)
