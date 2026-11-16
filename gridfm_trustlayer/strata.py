"""Strata from your own data: outage counts and load bins, computed as the CLI does.

Both are learned on the calibration set and then applied unchanged to new
scenarios, so that a scenario gets the same stratum whatever data it comes with:

- `OutageStrata` counts the branches (and generators) out of service among
  those in service in at least one calibration scenario, and classifies the
  outage type (none, branch only, generator only, mixed);
- `LoadBins` cuts the total load into percentile bins of the calibration set.

Inputs are `(n_scenarios, n_elements)` in-service arrays, or wide DataFrames
indexed by scenario id (then the results are Series with the same index, ready
for `ChannelSet.from_frame(..., strata={...})`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# Outage types, in the order of their codes (as in the CLI's `outage_type`).
OUTAGE_TYPES = ("none", "branch_only", "gen_only", "mixed")


@dataclass(frozen=True, eq=False)
class OutageStrata:
    """Number of outages and outage type per scenario (`n_outages`, `outage_type`).

    Create it with `fit` on the calibration set's in-service flags, then use it
    on any dataset. An element counts as an outage when it is out of service in
    the scenario but in service in at least one calibration scenario, so
    elements never in service (e.g. a line not built yet) are not outages.

    Attributes:
        branches: Which branches count, `(n_branches,)`.
        generators: Which generators count, `(n_generators,)`, or None when
            fitted on branches only.
    """

    branches: np.ndarray
    generators: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        branch_in_service: np.ndarray | pd.DataFrame,
        generator_in_service: np.ndarray | pd.DataFrame | None = None,
    ) -> OutageStrata:
        """Learn which elements count, from the calibration set.

        Args:
            branch_in_service: `True` where a branch is in service, one row per
                calibration scenario and one column per branch.
            generator_in_service: The same for generators, optional.
        """

        return cls(
            branches=_ever(branch_in_service, "branch_in_service"),
            generators=None
            if generator_in_service is None
            else _ever(generator_in_service, "generator_in_service"),
        )

    def n_outages(
        self,
        branch_in_service: np.ndarray | pd.DataFrame,
        generator_in_service: np.ndarray | pd.DataFrame | None = None,
    ) -> np.ndarray | pd.Series:
        """Branches plus generators out of service, per scenario."""

        branch, generator = self._counts(branch_in_service, generator_in_service)
        return _like(branch_in_service, branch + generator, "n_outages")

    def outage_type(
        self,
        branch_in_service: np.ndarray | pd.DataFrame,
        generator_in_service: np.ndarray | pd.DataFrame | None = None,
    ) -> np.ndarray | pd.Series:
        """Outage type code per scenario: 0 none, 1 branch only, 2 generator only, 3 mixed.

        The names are `OUTAGE_TYPES[code]`.
        """

        branch, generator = self._counts(branch_in_service, generator_in_service)
        codes = np.where(
            branch > 0, np.where(generator > 0, 3, 1), np.where(generator > 0, 2, 0)
        ).astype(np.int64)
        return _like(branch_in_service, codes, "outage_type")

    def _counts(
        self,
        branch_in_service: np.ndarray | pd.DataFrame,
        generator_in_service: np.ndarray | pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        branch = _flags(branch_in_service, "branch_in_service")
        _check_width(branch, self.branches, "branches")
        branch_out = (~branch & self.branches).sum(axis=1).astype(np.int64)
        if generator_in_service is None:
            if self.generators is not None:
                raise ValueError(
                    "Fitted with generators: pass generator_in_service as well."
                )
            return branch_out, np.zeros_like(branch_out)
        if self.generators is None:
            raise ValueError(
                "Fitted without generators: fit again with generator_in_service "
                "to count generator outages."
            )
        generator = _flags(generator_in_service, "generator_in_service")
        _check_width(generator, self.generators, "generators")
        if generator.shape[0] != branch.shape[0]:
            raise ValueError(
                f"branch_in_service has {branch.shape[0]} scenarios and "
                f"generator_in_service {generator.shape[0]}."
            )
        return branch_out, (~generator & self.generators).sum(axis=1).astype(np.int64)


@dataclass(frozen=True, eq=False)
class LoadBins:
    """Percentile bins of the total load of a scenario (`system_load_bin`).

    Create it with `fit` on the calibration set's total loads, then `assign`
    bins to any dataset. Store `edges` with the calibrator
    (`fit(..., system_load_bin_edges=bins.edges)`) to rebuild it later with
    `LoadBins(calibrator.system_load_bin_edges)`.

    Attributes:
        edges: The interior bin edges, increasing.
    """

    edges: np.ndarray

    @classmethod
    def fit(cls, system_load: np.ndarray | pd.Series, n_bins: int) -> LoadBins:
        """Edges at the percentiles of the calibration loads (e.g. quintiles for 5)."""

        return cls(edges=percentile_edges(np.asarray(system_load), n_bins))

    def assign(self, system_load: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
        """The bin of each scenario, 0 to `len(edges)`."""

        bins = assign_bins(np.asarray(system_load), self.edges).astype(np.int64)
        if isinstance(system_load, pd.Series):
            return pd.Series(bins, index=system_load.index, name="system_load_bin")
        return bins


def percentile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior percentile edges of `n_bins` bins of the finite `values` (float32).

    Repeated edges are merged, so heavily tied values can give fewer bins; at
    least two distinct bins are required.
    """

    if int(n_bins) < 2:
        raise ValueError(f"n_bins must be at least 2; got {n_bins!r}.")
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values to place bin edges on.")
    quantiles = np.linspace(0.0, 1.0, num=int(n_bins) + 1)
    edges = np.quantile(finite, quantiles).astype(np.float32, copy=False)
    interior = np.unique(edges[1:-1])
    if np.unique(assign_bins(finite, interior)).size < 2:
        raise ValueError("The values do not spread over at least two distinct bins.")
    return interior.astype(np.float32, copy=True)


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of each value: the number of edges at or below it (float32 comparison)."""

    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("Bins need finite values; some are NaN or infinite.")
    return np.searchsorted(np.asarray(edges, dtype=np.float32), array, side="right")


def _flags(values: np.ndarray | pd.DataFrame, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D, one row per scenario and one column per element; "
            f"it has shape {array.shape}."
        )
    if array.dtype != bool:
        numeric = np.asarray(array, dtype=float)
        if not np.all(np.isin(numeric, (0.0, 1.0))):
            raise ValueError(f"{name} must hold True/False (or 1/0) values only.")
        array = numeric.astype(bool)
    return array


def _ever(values: np.ndarray | pd.DataFrame, name: str) -> np.ndarray:
    """Elements in service in at least one scenario."""

    return np.asarray(_flags(values, name).any(axis=0), dtype=bool)


def _check_width(flags: np.ndarray, fitted: np.ndarray, what: str) -> None:
    if flags.shape[1] != fitted.size:
        raise ValueError(
            f"Got {flags.shape[1]} {what}; the strata were fitted on {fitted.size}. "
            "Use the same elements, in the same order, for every dataset."
        )


def _like(
    source: np.ndarray | pd.DataFrame, values: np.ndarray, name: str
) -> np.ndarray | pd.Series:
    if isinstance(source, pd.DataFrame):
        return pd.Series(values, index=source.index, name=name)
    return values
