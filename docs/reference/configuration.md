# Configuration reference

`compute`, `calibrate` and `screen` all read one YAML file, passed with `--config`. This page lists every key. To write the file, use the [config builder](../getting-started/config-builder.md) or start from an [example config](https://github.com/gridfm/gridfm-trustlayer/tree/main/examples/config); unknown keys are rejected.

## Structure

```yaml
version: 1

# What to calibrate and how
task_name: PowerFlow
cp_type: element            # or kernel
strat: n_outages            # optional
alpha: 0.1
variables: [Vm, Va, line_loading]

# Kernel options (cp_type: kernel only)
embedding_pooling: mean
distance: euclidean
bandwidth: n_eff
n_eff_target: 600           # or auto, with an optional tuning: block
n_eff_min: 5

# Screening limits (screen only)
screen_bounds:
  Vm: {lower: 0.95, upper: 1.05}
  line_loading: {upper: 1.0}

inputs:
  calibration:              # read by compute
    predictions: path/to/calibration/predictions.parquet
    pf_node: path/to/calibration/raw/bus_data.parquet
    pf_edge: path/to/calibration/raw/branch_data.parquet
    embeddings: path/to/calibration/bus_embeddings.parquet   # kernel only
  test:                     # read by calibrate and screen
    predictions: path/to/test/predictions.parquet
    pf_node: path/to/test/raw/bus_data.parquet
    pf_edge: path/to/test/raw/branch_data.parquet
    embeddings: path/to/test/bus_embeddings.parquet          # kernel only

outputs:
  artifact: trustlayer_outputs/run.trustlayer
  calibrated_predictions: trustlayer_outputs/calibrated
  screen_dir: trustlayer_outputs/screen

commands:                   # per-command overrides
  screen:
    variables: [Vm, line_loading]
```

## How values are resolved

For each command, a value is taken from the first of these that sets it:

1. a command-line flag, e.g. `--alpha 0.05`;
2. `commands.<command>`, e.g. `commands.screen.variables`;
3. the command's dataset under `inputs` (`inputs.calibration` for `compute`, `inputs.test` for `calibrate` and `screen`) and `outputs`;
4. the top-level keys.

## Method keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `version` | int | – | Config format marker; use `1`. |
| `task_name` | str | – | How to read the predictions: `PowerFlow` or `OptimalPowerFlow` (case, spaces, `_` and `-` do not matter; `PF` and `OPF` work too). For `PowerFlow`, the set-points (`Vm` at PV buses, `Va` at the reference bus) are set aside as known values. Any other name reads the predictions as they are, and `compute` says so. Also names the default output directory (`unknown` when not set). |
| `cp_type` | str | `element` | `element` (stratified split-conformal) or `kernel` (kernel-weighted). See [Choosing a setup](../user-guide/choosing-a-setup.md#method-cp_type). |
| `alpha` | float | required by `compute` | Miscoverage level, two-sided: `0.1` gives 90% intervals, 5% per tail. |
| `variables` | list | `[Vm, Va]` | Any of `Vm`, `Va`, `line_loading`. |
| `strat` | str or null | null | Group scenarios and fit bounds per group: `n_outages`, `outage_type`, `system_load_bin` or `pred_bins` (`element` only). See [Choosing a setup](../user-guide/choosing-a-setup.md#grouping-strat). |
| `system_load_n_bins` | int | – | Number of percentile bins of total load. Required with `strat: system_load_bin`. |
| `pred_n_bins` | int | – | Number of percentile bins of the predicted values, per variable. Required with `strat: pred_bins`. |
| `sparse_strata` | str | `pooled` | With `strat`: the bounds of a bus or line in a stratum with fewer than $\lceil 2/\alpha \rceil - 1$ valid calibration samples, or in a stratum absent from the calibration data. `pooled` uses the element's quantile over all calibration scenarios: finite, but without a guarantee within that stratum. `infinite` keeps the per-stratum guarantee with infinite bounds. See [Sparse strata](../user-guide/choosing-a-setup.md#sparse-strata-sparse_strata). |
| `save_figures` | bool | `true` | `false` makes `calibrate` and `screen` skip their figures, like `--no-figures`. |
| `n_jobs` | int | – | Processes for the cross-validated bandwidth choice (`n_eff_target: auto`), which runs in 4 processes by default (at most the CPU count); `-1` uses every CPU, `1` one process. When set, blocks of scenarios are also processed in parallel by every command, which is off by default: on IEEE-118 it made runs slower and larger, as reading parquet dominates. Each process holds its own copy of the data it works on. Same as `--n-jobs`. |

The stratifications in detail:

- `n_outages`: number of out-of-service branches plus out-of-service generators in the scenario, from the datakit ground truth (generators from `gen_data.parquet` next to `pf_node`, when present).
- `outage_type`: `none`, `branch_only`, `gen_only` or `mixed`.
- `system_load_bin`: percentile bins of the total active load $\sum_b P_{d,b}$. The bin edges are learned from the calibration set and stored in the artifact. Labels read like `p20-p40 [1450, 1620)`.
- `pred_bins`: percentile bins of each predicted value, learned per variable from the calibration predictions. One scenario can fall into different bins at different buses or lines.

## Kernel keys

Used only with `cp_type: kernel`; see [Kernel-weighted calibration](../user-guide/kernel.md).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `embedding_pooling` | str | `mean` | How per-bus embeddings become one vector per scenario: `mean`, `max`, `meanmax` or `concat`. |
| `distance` | str | `euclidean` | `euclidean` or `cosine`, on the standardized scenario vectors. |
| `bandwidth` | str | `n_eff` | `n_eff`: solve the bandwidth for a target effective sample size. `knn`: distance to the `k_nn`-th nearest calibration scenario. |
| `n_eff_target` | number or `auto` | `600` | Target effective sample size for `bandwidth: n_eff`, at least `n_eff_min`. `auto` chooses it by cross-validation on the calibration set (see [`tuning`](#tuning)). |
| `k_nn` | int | `20` | Neighbour rank for `bandwidth: knn`. |
| `n_eff_min` | number | `5` | Below this effective sample size, a scenario falls back to the stratum-wide quantile. |
| `tuning` | mapping | – | Options of `n_eff_target: auto`; rejected otherwise. |

### `tuning`

Only with `cp_type: kernel`, `bandwidth: n_eff` and `n_eff_target: auto`. All keys are optional. The procedure is described in [Kernel-weighted calibration](../user-guide/kernel.md#choosing-the-bandwidth-automatically).

| Key | Default | Description |
| --- | --- | --- |
| `folds` | `5` | Number of cross-validation folds within the calibration set (at least 2). |
| `search` | `grid` | `grid`: every combination of `grid` and `distances`. `random`: `budget` settings drawn at random. |
| `grid` | `[100, 200, 400, 600, 800, 1200]` | Candidate targets for `search: grid`; each at least `n_eff_min`. |
| `distances` | the config's `distance` | List of distances to compare, e.g. `[euclidean, cosine]`. |
| `budget` | `10` | Number of (target, distance) settings tried by `search: random`. |
| `n_eff_range` | span of `grid` | `[low, high]` range for `search: random`; targets are drawn log-uniformly. |
| `group_by` | `auto` | Keep related scenarios in one fold. `auto` uses datakit's `load_scenario_idx` column of the calibration `pf_node` when present; `load_scenario_idx` requires it; `none` assigns scenarios at random. |
| `seed` | `0` | Seed of the fold assignment and of the random search. |
| `coverage_tolerance` | `auto` | How far cross-validated coverage may fall below $1-\alpha$ and still count as reaching it. `auto`: two standard errors, $2\sqrt{\alpha(1-\alpha)/n}$, per check; or a fixed number such as `0.005`. |

## `inputs`

Two blocks, one per dataset, and an optional third. The first two must be different data; see [Preparing the data](../user-guide/data.md#two-separate-datasets).

- `inputs.calibration` is read by `compute`.
- `inputs.test` is read by `calibrate` and `screen`.
- `inputs.apply` (optional) is read by `apply`: new scenarios without a known solution. Only `predictions` is required; `pf_node` and `pf_edge` describe the scenarios when the layer is stratified by outages or load or calibrates `line_loading`, and need no solved values. See the [CLI reference](cli.md#apply).

| Key | Required | Description |
| --- | --- | --- |
| `predictions` | yes | Predictions parquet, one row per scenario and bus (graphkit `<grid>_predictions.parquet`). In the test block, it also sets the default names of the calibrated outputs. |
| `pf_node` | yes | Bus ground truth (datakit `raw/bus_data.parquet`). |
| `pf_edge` | yes | Branch ground truth (datakit `raw/branch_data.parquet`): line parameters and ratings for `line_loading`. |
| `embeddings` | `kernel` only | Embeddings parquet with `scenario`, `bus` and `emb_*` columns (graphkit `predict --get_embeddings`). |
| `generator_predictions` | no | Generator predictions parquet of OPF models. |

The config is rejected if both blocks name the same `predictions` file, and `calibrate` refuses test predictions identical to the calibration ones. `inputs.artifact` may be set next to the two blocks as an alternative to `outputs.artifact`. The file layouts are described in [Data schemas](schemas.md).

## `outputs`

| Key | Used by | Description |
| --- | --- | --- |
| `artifact` | `compute` writes, `calibrate` reads | The `.trustlayer` file. |
| `calibrated_predictions` | `calibrate` writes, `screen` reads | One target family (voltages or `line_loading`): the path of the calibrated dataset. Both families: a directory, in which the two datasets are named after the test predictions and the variables, e.g. `<predictions>_vm_va_with_intervals.parquet`. |
| `calibrated_predictions_volt`, `calibrated_predictions_line` | `calibrate`, `screen` | Explicit paths for the voltage and line-loading datasets. Take precedence over `calibrated_predictions`. |
| `applied_predictions` | `apply` | One target family: the path of the bounds dataset. Both families: a directory, in which the datasets are named `<predictions>_<variables>_bounds.parquet`. |
| `applied_predictions_volt`, `applied_predictions_line` | `apply` | Explicit paths for the two bounds datasets. Take precedence over `applied_predictions`. |
| `coverage_dir` | `calibrate` | Directory for the coverage report: `summary.md`, the `coverage_by_*.csv` tables and `figures/`. Default: a `coverage/` folder next to the calibrated datasets. |
| `screen_dir` | `screen` | Directory for the screening report, tables, figures and detail datasets. |
| `status_dir` | `apply` | Directory for the status report of the new scenarios (`summary.md`, `scenarios.csv`, `status_by_stratum.csv`), written when `screen_bounds` is set. Default: a `status/` folder next to the applied datasets. |
| `output` | `calibrate`, `screen` | Generic override, like the `--output` flag. |

The calibrated predictions and the screening detail are written as partitioned parquet datasets: directories with one `scenario_partition=k/` folder per 200 scenarios, like gridfm-datakit's outputs ([Reading the results](../user-guide/results.md)). A path that already holds a previous trustlayer output is replaced; anything else that exists there (an input, datakit's raw data, any other file or directory) is left alone and the run stops. The coverage report of `calibrate` goes to `coverage_dir`, by default a `coverage/` folder next to the calibrated datasets. Without any output keys, everything goes to `trustlayer_outputs/<task_name>/<cp_type>/` in the working directory (`<task_name>` is `unknown` when not set); the artifact is then `<calibration predictions>_<variables>.trustlayer` there, which every command finds by the same rule. Relative paths in the config, inputs and outputs alike, are relative to the directory the command runs in, not to the config file.

## `screen_bounds`

The operating limits that `screen` and `apply` compare the intervals with, in the variable's own units:

| Variable | Keys | Units |
| --- | --- | --- |
| `Vm` | `lower` and `upper` | per unit |
| `line_loading` | `upper` | fraction of the branch rating `rate_a`; `1.0` is 100% |

For `screen`, every screened variable needs bounds. `Va` has no natural limit, so restrict screening with `commands.screen.variables` when you calibrate it. Each entry takes only `lower` and `upper`, with `lower` below `upper`; a `lower` given for `line_loading` is not used, by `screen` and `apply` alike, since a loading is screened against its rating only. The `--lower-bound` and `--upper-bound` flags override the bounds when exactly one variable is screened; a side not given on the command line comes from the config. `apply` screens every variable of the layer that has bounds here and skips the others; without `screen_bounds`, it writes bounds only.

## `commands`

`commands.compute`, `commands.calibrate`, `commands.screen` and `commands.apply` override shared keys for one command. Allowed keys: `task_name`, `alpha`, `cp_type`, `strat`, `system_load_n_bins`, `pred_n_bins`, `embedding_pooling`, `distance`, `bandwidth`, `n_eff_target`, `k_nn`, `n_eff_min`, `sparse_strata`, `n_jobs`, `variables`, `screen_bounds`, `save_figures`, `artifact`, `output` and a nested `outputs` mapping. Inputs cannot be overridden per command, since each command already reads its own dataset.

The most common use is restricting screening to the variables that have limits:

```yaml
commands:
  screen:
    variables: [Vm, line_loading]
```

## Mixed runs

When `variables` contains both voltages and `line_loading`:

- `compute` writes one artifact holding both calibrators;
- `calibrate` writes two parquets, per bus and per branch, so `outputs.calibrated_predictions` must be a directory, or `calibrated_predictions_volt` and `calibrated_predictions_line` must both be set;
- `screen` reads both.
