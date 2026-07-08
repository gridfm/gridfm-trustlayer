# Methodology

This page summarizes the current trustlayer workflow.

## Goal

`gridfm-trustlayer` wraps existing GridFM prediction artifacts with calibrated uncertainty sets.

The core idea is:

1. start from model predictions and matching ground truth
2. measure calibration residuals on a held-out calibration set
3. convert those residuals into conformal interval adjustments
4. apply the resulting intervals to new predictions
5. optionally screen those intervals against operating limits

The repository is intentionally post-hoc and model-agnostic: it does not train GridFM models and it does not depend on Lightning or GPU-specific runtime in the statistical core.

## Target types

The current implementation supports two target families.

### Direct prediction targets

These are channels already predicted by the model, such as:

- `Vm`
- `Va`

Trustlayer reads these directly from graphkit prediction parquets and calibrates them against matching target values or external ground truth.

### Derived physical targets

These are not direct model heads. They are computed from predictions plus topology and physical metadata.

The current derived target is:

- `line_loading`

Trustlayer computes line loading through the datakit-backed branch-power path and then calibrates the derived quantity in the same conformal framework.

## Current method: `cp_type: element`

The current shipped method is exposed as:

- `cp_type: element`

This is the current residual-quantile conformal workflow used by the repository.

Current characteristics:

- separate calibration per target dimension
- optional stratification by `n_outages`, where the outage count includes inactive branches and inactive generators per scenario
- optional stratification by `outage_type`, which groups scenarios into `none`, `branch_only`, `gen_only`, and `mixed`
- optional stratification by `system_load_bin`, which groups scenarios by percentile bins of total active load derived from raw bus `Pd` and exposes human-readable labels like `p20-p40 [1450, 1620)`
- mixed bus-level and branch-level runs supported through separate target families

`element` is the canonical name used by the current workflow.

## Implementation details

This section describes the math the current code actually applies. It is intentionally limited to the quantities needed to understand the artifact and the outputs.

### Notation

For one variable such as `Vm`, `Va`, or `line_loading`:

- let $s$ index scenarios
- let $e$ index entities, either buses or branches
- let $\hat{y}_{s,e}$ be the model prediction
- let $y_{s,e}$ be the matching ground-truth value

Trustlayer calibrates each variable independently and, within a variable, each entity position independently.

### Residuals used during `compute`

For every aligned scenario-entity pair in the calibration slice, trustlayer computes the signed residual

$$
r_{s,e} = y_{s,e} - \hat{y}_{s,e}.
$$

The sign matters. A positive residual means the model underpredicted the target, and a negative residual means it overpredicted it.

For each entity $e$, trustlayer then stores two empirical residual quantiles:

$$
q^-_e = Q_{\alpha/2}(\{r_{s,e}\}_s), \qquad
q^+_e = Q_{1-\alpha/2}(\{r_{s,e}\}_s).
$$

So with `alpha = 0.1`, the implementation uses the empirical 5th and 95th percentiles of the residuals for each entity.

This is why the current method is described in the code as a naive residual-quantile method rather than a more elaborate conformal variant.

### Stratified residual quantiles

If `strat` is enabled, trustlayer first assigns each scenario to a stratum $g_s$ and then fits separate quantiles inside each stratum:

$$
q^-_{e,g} = Q_{\alpha/2}(\{r_{s,e} : g_s = g\}), \qquad
q^+_{e,g} = Q_{1-\alpha/2}(\{r_{s,e} : g_s = g\}).
$$

The currently supported strata are:

- `n_outages`
- `outage_type`
- `system_load_bin`

If no stratification is used, the artifact only stores the global per-entity quantiles $q^-_e$ and $q^+_e$.

### How `system_load_bin` is constructed

When `strat: system_load_bin` is requested, trustlayer derives one total active-load value per scenario:

$$
L_s = \sum_b Pd_{s,b},
$$

where the sum runs over buses in the ground-truth input.

If the requested number of bins is $B$, the code builds percentile cut points from the calibration scenarios only:

$$
E_j = Q_{j/B}(\{L_s\}_s), \qquad j = 1, \dots, B-1.
$$

Those interior cut points are stored in the `.trustlayer` artifact. Each scenario is then assigned with a right-sided search over the stored edges:

$$
g_s = \mathrm{searchsorted}(E, L_s).
$$

In practical terms:

- `compute` learns the load-bin edges from the calibration slice
- `calibrate` reuses those same edges to assign new scenarios to bins
- the human-readable labels such as `p20-p40 [1450, 1620)` are derived from those stored percentile edges

### Interval construction during `calibrate`

For a new prediction $\hat{y}_{s,e}$, trustlayer adds the saved residual quantiles back to the prediction.

Without stratification:

$$
\ell_{s,e} = \hat{y}_{s,e} + q^-_e, \qquad
u_{s,e} = \hat{y}_{s,e} + q^+_e.
$$

With stratification:

$$
\ell_{s,e} = \hat{y}_{s,e} + q^-_{e,g_s}, \qquad
u_{s,e} = \hat{y}_{s,e} + q^+_{e,g_s}.
$$

The stored output columns are therefore prediction intervals of the form:

$$
[\ell_{s,e}, u_{s,e}].
$$

For invalid or masked entries, the implementation does not expand an interval and leaves the lower and upper endpoints equal to the original prediction.

### Analysis quantities written to CSV

The trust-analysis outputs are derived directly from those interval endpoints.

For each scenario-entity pair, the interval width is $u_{s,e} - \ell_{s,e}$.

Coverage is recorded as a binary indicator: it is `1` when $\ell_{s,e} \le y_{s,e} \le u_{s,e}$ and `0` otherwise.

The entity-level `empirical_coverage` reported in the analysis CSV is the average of that indicator over scenarios for each bus or branch, and the reported width summaries are the mean, min, and max interval widths over scenarios.

## Workflow

### `compute`

`compute` fits interval calibrators from aligned prediction and ground-truth artifacts.

Inputs:

- prediction parquet
- bus ground-truth parquet
- branch ground-truth parquet when derived loading is requested
- conformal settings such as `alpha`, `cp_type`, `strat`, and `variables`

Outputs:

- one `.trustlayer` artifact containing one or more fitted calibrators

### `calibrate`

`calibrate` applies the fitted calibrators to predictions.

Outputs:

- calibrated bus parquet for voltage variables
- calibrated branch parquet for `line_loading`
- trust-analysis CSVs
- trust-summary CSVs
- figures

This step is intentionally split by target family because bus-level and branch-level results live on different entity axes.

### `screen`

`screen` evaluates calibrated intervals against operating bounds defined in the YAML config.

Example bounds:

- `Vm` in `[0.95, 1.05]`
- `line_loading` <= `1.0`

Outputs:

- detail CSVs with row-level predicted and true violation flags
- entity-level summaries by bus or branch
- aggregate summaries by bus type or `rate_a`
- screening figures

## Screening interpretation

The current screen implementation uses a possible-violation interpretation.

For an upper limit:

- the interval raises an alarm when `pi_upper` crosses the limit

For a lower limit:

- the interval raises an alarm when `pi_lower` crosses the limit

This is conservative by design. It answers:

- could this operating point violate the bound, given the calibrated uncertainty?

It does not require the true target to violate the bound.

That distinction matters especially for OPF data:

- PF data often contains realized violations, so precision/recall style screening is naturally informative
- OPF data is often feasible by construction, so the same screen behaves more like a near-limit risk alarm

## Task semantics

Trustlayer does not assume that all prediction parquets mean the same thing.

Current task-aware behavior includes:

- PF-specific clamping/correction rules applied behind task adapters
- OPF support for separate bus and generator prediction artifacts
- preservation of task-specific direct channels instead of forcing a PF-only interpretation

This is why the repository keeps schema handling inside the I/O layer rather than in the conformal core.
