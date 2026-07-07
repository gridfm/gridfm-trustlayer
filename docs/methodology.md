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
- mixed bus-level and branch-level runs supported through separate target families

`element` is the canonical name used by the current workflow.

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
- `line_loading` in `[0.0, 1.0]`

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