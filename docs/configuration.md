# Configuration Reference

This page documents the current trustlayer YAML structure used by `compute`, `calibrate`, and `screen`.

## Config resolution order

The CLI resolves values in this order:

1. shared top-level config values
2. shared `inputs` and `outputs`
3. command-specific overrides under `commands.<verb>`
4. explicit CLI flags

CLI flags always override the YAML.

## Example config

```yaml
version: 1

task_name: PowerFlow
cp_type: base
alpha: 0.1
calibration_fraction: 0.25
random_seed: 7
variables:
  - Vm
  - Va
  - line_loading

screen_bounds:
  Vm:
    lower: 0.95
    upper: 1.05
  line_loading:
    lower: 0.0
    upper: 1.0

inputs:
  predictions: path/to/predictions.parquet
  generator_predictions: null
  pf_node: path/to/bus_data.parquet
  pf_edge: path/to/branch_data.parquet
  analysis_pf_node: null
  analysis_pf_edge: null

outputs:
  artifact: trustlayer_outputs/powerflow/base/grid_predictions_vm_va_line_loading.trustlayer
  calibrated_predictions_volt: trustlayer_outputs/powerflow/base/grid_predictions_vm_va_with_intervals.parquet
  calibrated_predictions_line: trustlayer_outputs/powerflow/base/grid_predictions_line_loading_with_intervals.parquet
  screen_dir: trustlayer_outputs/powerflow/base/screen

commands:
  screen:
    variables:
      - Vm
      - line_loading
```

## Top-level keys

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `version` | `int` | No | Config schema marker. |
| `task_name` | `str` | Usually yes | Task adapter used when reading prediction artifacts. |
| `cp_type` | `str` | Yes | Current supported value: `base`. Legacy alias: `naive`. |
| `alpha` | `float` | Required for `compute` | Miscoverage level used while fitting intervals. |
| `calibration_fraction` | `float or null` | No | Optional fraction of scenarios reserved for fitting before applying on holdout scenarios. |
| `random_seed` | `int or null` | No | Random seed used with `calibration_fraction`. |
| `variables` | `list[str]` | Usually yes | Any subset of `Vm`, `Va`, and `line_loading`. Defaults to `Vm, Va` if omitted. |
| `screen_bounds` | `mapping` | Required for `screen` on each screened variable | Per-variable lower and upper screening bounds. |

## `inputs`

| Key | Type | Used by | Description |
| --- | --- | --- | --- |
| `predictions` | `str` | `compute`, `calibrate`, default `screen` path resolution | Main bus-level prediction parquet. |
| `generator_predictions` | `str or null` | `compute`, `calibrate` | Optional OPF generator prediction parquet. |
| `pf_node` | `str` | `compute`, `calibrate`, `screen` | Raw bus ground truth. |
| `pf_edge` | `str` | `compute`, `calibrate`, `screen` | Raw branch ground truth. |
| `analysis_pf_node` | `str or null` | `calibrate` | Optional alternate bus ground truth for analysis when the prediction artifact does not already contain target columns. |
| `analysis_pf_edge` | `str or null` | `calibrate` | Optional alternate branch ground truth for line-loading analysis when needed. |

## `outputs`

| Key | Type | Used by | Description |
| --- | --- | --- | --- |
| `artifact` | `str` | `compute`, `calibrate` | `.trustlayer` path written by `compute` and read by `calibrate`. |
| `output` | `str` | `calibrate`, `screen` | Shared output override. For mixed `calibrate` runs it must be a directory. |
| `calibrated_predictions` | `str` | `calibrate`, `screen` | Shared calibrated parquet path for a single target family, or directory root for mixed runs. |
| `calibrated_predictions_volt` | `str` | `calibrate`, `screen` | Explicit voltage calibrated parquet path. |
| `calibrated_predictions_line` | `str` | `calibrate`, `screen` | Explicit line-loading calibrated parquet path. |
| `screen_dir` | `str` | `screen` | Output directory for screening CSVs and figures. |

## `screen_bounds`

Each screened variable must define both `lower` and `upper`.

Example:

```yaml
screen_bounds:
  Vm:
    lower: 0.95
    upper: 1.05
  line_loading:
    lower: 0.0
    upper: 1.0
```

Notes:

- Bounds are interpreted in the variable's natural units.
- `Vm` uses per-unit voltage magnitude.
- `line_loading` uses a ratio relative to `rate_a`, so `1.0` means 100% of the branch rating.
- CLI `--lower-bound` and `--upper-bound` can override bounds only when screening exactly one variable.

## `commands`

Each command can override shared values under `commands.compute`, `commands.calibrate`, or `commands.screen`.

Supported override keys:

- `task_name`
- `alpha`
- `cp_type`
- `variables`
- `output`
- `artifact`
- `calibration_fraction`
- `random_seed`
- `screen_bounds`
- nested `inputs`
- nested `outputs`

Example:

```yaml
commands:
  screen:
    variables:
      - Vm
      - line_loading
    outputs:
      screen_dir: trustlayer_outputs/powerflow/base/screen
```

## Mixed-variable behavior

When `variables` includes both voltage channels and `line_loading`:

- `compute` still writes one combined artifact bundle
- `calibrate` writes split bus-level and branch-level parquet outputs
- `screen` reads split calibrated outputs if both target families are requested

Path rules:

- use `outputs.calibrated_predictions_volt` and `outputs.calibrated_predictions_line` for explicit mixed-run file paths
- or set `outputs.calibrated_predictions` to a directory root and let trustlayer derive file names
- do not point `outputs.calibrated_predictions` to a single parquet file for a mixed run

## Recommended patterns

### PowerFlow screening for actual violations

Use PF prediction and ground-truth artifacts, then set operating bounds to your true acceptance region. This is the most natural setting for confusion-matrix style violation detection.

### OPF screening for near-limit alarms

OPF solutions are often feasible by construction. If you screen against the same solver limits, you may see few or no true violations. In that case, tighter operational margins such as `Vm in [0.97, 1.03]` or `line_loading <= 0.95` are often more informative.

## See also

- [CLI Reference](cli.md)
- [Schemas](schemas.md)