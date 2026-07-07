# CLI Reference

This page documents the current `gridfm_trustlayer` command-line interface.

## Command overview

Entrypoint:

```bash
gridfm_trustlayer <command> [OPTIONS]
```

Available commands:

- `compute`
- `calibrate`
- `screen`

All commands require `--config`. CLI flags are overrides on top of the YAML config.

## Shared workflow

The current workflow is:

1. `compute` fits interval calibrators and writes a `.trustlayer` artifact.
2. `calibrate` applies the artifact and writes calibrated prediction outputs plus analysis CSVs and figures.
3. `screen` evaluates those calibrated intervals against operating bounds.

Example:

```bash
gridfm_trustlayer compute --config path/to/trust_config.yaml
gridfm_trustlayer calibrate --config path/to/trust_config.yaml
gridfm_trustlayer screen --config path/to/trust_config.yaml
```

## `compute`

Purpose:

- load prediction and ground-truth artifacts
- fit one or more `element` conformal interval calibrators
- optionally fit only on a calibration subset of scenarios
- save the result as a `.trustlayer` artifact

Syntax:

```bash
gridfm_trustlayer compute --config path/to/trust_config.yaml
```

| Option | Type | Description |
| --- | --- | --- |
| `--config` | `str` | Required YAML config. |
| `--predictions` | `str` | Predictions parquet with `*_pred` columns. |
| `--pf-node` | `str` | Ground-truth bus parquet used during fitting. |
| `--pf-edge` | `str` | Ground-truth branch parquet used during fitting and loading computation. |
| `--output` | `str` | Artifact output path. |
| `--alpha` | `float` | Miscoverage level. |
| `--cp-type` | `str` | Current supported value is `element`. |
| `--strat` | `str` | Optional residual stratification. Current supported value is `n_outages`. |
| `--variables` | `list[str]` | Any subset of `Vm`, `Va`, `line_loading`. |
| `--calibration-fraction` | `float` | Reserve a scenario fraction for fitting and apply the learned intervals on the remaining holdout scenarios. |
| `--random-seed` | `int` | Random seed used for the calibration/holdout split. |
| `--task-name` | `str` | Task adapter name such as `PowerFlow` or `OptimalPowerFlow`. |
| `--generator-predictions` | `str` | Optional generator parquet for OPF prediction artifacts. |

Notes:

- A mixed run such as `variables: [Vm, Va, line_loading]` still writes one artifact bundle.
- The saved artifact may contain one calibrator or multiple calibrators keyed by target family.
- With `--strat n_outages`, residual quantiles are fitted separately per scenario outage count.

## `calibrate`

Purpose:

- load a `.trustlayer` artifact
- apply the fitted intervals to raw predictions
- write calibrated prediction outputs
- emit analysis CSVs and figures

Syntax:

```bash
gridfm_trustlayer calibrate --config path/to/trust_config.yaml
```

| Option | Type | Description |
| --- | --- | --- |
| `--config` | `str` | Required YAML config. |
| `--predictions` | `str` | Input predictions parquet. |
| `--artifact` | `str` | Saved `.trustlayer` artifact from `compute`. |
| `--output` | `str` | Output parquet path for a single target family, or output directory for mixed voltage + loading runs. |
| `--cp-type` | `str` | Current supported value is `element`. |
| `--strat` | `str` | Optional stratification override. When provided, it must match the saved artifact. |
| `--pf-node` | `str` | Bus ground truth used for analysis or target reconstruction when needed. |
| `--pf-edge` | `str` | Branch ground truth used for line-loading analysis or target reconstruction when needed. |
| `--task-name` | `str` | Task adapter name used while loading predictions. |
| `--generator-predictions` | `str` | Optional OPF generator prediction parquet. |

Output behavior:

- Bus targets produce a calibrated bus parquet with interval columns merged onto the original prediction rows.
- Branch targets produce a branch-level output parquet derived from aligned bus predictions and branch topology.
- Mixed runs write separate voltage and line-loading parquet outputs.
- When the artifact uses scenario stratification such as `strat: n_outages` or `strat: outage_type`, `calibrate` reloads raw ground truth for the application scenarios so it can apply the correct scenario stratum.

Analysis side outputs:

- `trust_analysis_volt.csv`
- `trust_summary_volt.csv`
- `trust_analysis_line.csv`
- `trust_summary_line.csv`
- `figures/`

Mixed-run rule:

- If both voltage and line-loading targets are present, `--output` must be a directory or omitted.
- To force exact parquet names in a mixed run, use `outputs.calibrated_predictions_volt` and `outputs.calibrated_predictions_line` in the YAML.

## `screen`

Purpose:

- read calibrated interval outputs
- compare interval bounds against configured operating limits
- save detailed screening flags, aggregate confusion counts, and figures

Syntax:

```bash
gridfm_trustlayer screen --config path/to/trust_config.yaml
```

| Option | Type | Description |
| --- | --- | --- |
| `--config` | `str` | Required YAML config. |
| `--variables` | `list[str]` | Variables to screen. Defaults to the `commands.screen.variables` or shared `variables` config. |
| `--lower-bound` | `float` | Override lower bound for one screened variable. |
| `--upper-bound` | `float` | Override upper bound for one screened variable. |
| `--output` | `str` | Output directory for screening artifacts. |
| `--calibrated-predictions` | `str` | Shared calibrated parquet input for a single target family. |
| `--calibrated-predictions-volt` | `str` | Voltage calibrated parquet override. |
| `--calibrated-predictions-line` | `str` | Line-loading calibrated parquet override. |
| `--pf-node` | `str` | Bus ground truth used when voltage targets are missing from the calibrated parquet. |
| `--pf-edge` | `str` | Branch ground truth used when line targets are missing from the calibrated parquet. |
| `--task-name` | `str` | Task name used for default path resolution. |

Current semantics:

- lower violation alarm when `pi_lower < lower_bound`
- upper violation alarm when `pi_upper > upper_bound`
- a numeric tolerance is applied internally to avoid floating-point boundary drift on exact limits

Saved outputs:

- `trust_screen_detail_volt.csv`
- `trust_screen_detail_line.csv`
- `trust_screen_entity_volt.csv`
- `trust_screen_entity_line.csv`
- `trust_screen_summary_volt.csv`
- `trust_screen_summary_line.csv`
- `screen_figures/`

## Current supported values

### `cp_type`

- `element`

### `strat`

- `n_outages`

### Variables

- `Vm`
- `Va`
- `line_loading`

### Task names in active use

- `PowerFlow`
- `OptimalPowerFlow`

## See also

- [Configuration](configuration.md)
- [Schemas](schemas.md)