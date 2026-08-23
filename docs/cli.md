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
- fit one or more conformal interval calibrators (`element` or `kernel`)
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
| `--cp-type` | `str` | `element` or `kernel`. |
| `--strat` | `str` | Optional residual stratification. Current supported values are `n_outages`, `outage_type`, `system_load_bin`, and `pred_bins`. |
| `--system-load-n-bins` | `int` | Requested number of percentile bins when `--strat system_load_bin` is used. |
| `--pred-n-bins` | `int` | Requested number of percentile bins when `--strat pred_bins` is used. |
| `--variables` | `list[str]` | Any subset of `Vm`, `Va`, `line_loading`. |
| `--calibration-fraction` | `float` | Reserve a scenario fraction for fitting and apply the learned intervals on the remaining holdout scenarios. Required for `--cp-type kernel`. |
| `--random-seed` | `int` | Random seed used for the calibration/holdout split. |
| `--embeddings` | `str` | Scenario embeddings parquet with `emb_*` columns. Required for `--cp-type kernel`. |
| `--embedding-pooling` | `str` | How per-entity embeddings are pooled into one vector per scenario: `mean` (default), `max`, `meanmax`, or `concat`. |
| `--distance` | `str` | Distance over scenario representations: `euclidean` (default) or `cosine`. |
| `--bandwidth` | `str` | How the kernel bandwidth is chosen: `knn` (default) or `n_eff`. |
| `--k-nn` | `int` | Nearest-neighbour rank whose distance sets the kernel bandwidth. Used by `--bandwidth knn`. Default `20`. |
| `--n-eff-target` | `float` | Target effective sample size solved for per test scenario. Used by `--bandwidth n_eff`. Defaults to `20/alpha`. |
| `--n-eff-min` | `float` | Minimum effective sample size before kernel weights revert to the stratum-wide quantile. Default `5`. |
| `--task-name` | `str` | Task adapter name such as `PowerFlow` or `OptimalPowerFlow`. |
| `--generator-predictions` | `str` | Optional generator parquet for OPF prediction artifacts. |

Notes:

- A mixed run such as `variables: [Vm, Va, line_loading]` still writes one artifact bundle.
- The saved artifact may contain one calibrator or multiple calibrators keyed by target family.
- With `--strat n_outages`, residual quantiles are fitted separately per scenario outage count.
- With `--strat system_load_bin`, residual quantiles are fitted separately per percentile bin of total active load derived from the raw bus `Pd` column.
- With `--strat pred_bins`, residual quantiles are fitted separately per percentile bin of the model's own predictions, using one learned edge set per calibrated variable and assigning bins per prediction value.
- The resulting labels in calibrated outputs and analysis artifacts are percentile ranges with cut-point hints, such as `p00-p20 (<1450)` or `p20-p40 [1450, 1620)`.

Kernel-specific notes:

- `--cp-type kernel` requires `--embeddings` and a non-null `--calibration-fraction`. Without a split, every test scenario would sit at distance zero from itself in the calibration bank, which invalidates the coverage guarantee, so the CLI rejects it rather than producing an optimistic result.
- `--cp-type kernel` is incompatible with `--strat pred_bins`; use `n_outages`, `outage_type`, `system_load_bin`, or omit `strat`.
- Kernel and stratification compose. With `--cp-type kernel --strat n_outages`, the nearest-neighbour search, the bandwidth, and the weighted quantile are all restricted to calibration scenarios in the same outage stratum.
- Kernel artifacts store the calibration residual bank and scenario embeddings, so they are substantially larger than `element` artifacts and are written as a compressed binary container rather than JSON. Both encodings load through the same `--artifact` path.
- The embeddings parquet must cover every scenario present in the predictions parquet, and the same `--embedding-pooling` must be used for `compute` and `calibrate`. The pooling name is stored in the artifact and mismatches are rejected.

Distance metric:

- `--distance euclidean` (default) uses squared Euclidean distance on the standardized scenario representations.
- `--distance cosine` L2-normalizes those representations first, which makes squared Euclidean distance equal to `2 - 2*cos`. Because normalization happens after centering, cosine compares the *direction* in which scenarios deviate from the mean scenario and ignores the magnitude of that deviation.
- The metric is stored in the artifact and reapplied automatically during `calibrate`; there is no `--distance` flag on `calibrate`.
- Cosine is not a remedy for distance concentration. Angular distances concentrate in high dimension just as Euclidean distances do, so it will not restore localization that a wide embedding has lost.

Bandwidth selection:

- `--bandwidth knn` sets `h^2` to the squared distance to the `k_nn`-th nearest calibration scenario in the same stratum. This is the rule the paper specifies and remains the default.
- `--bandwidth n_eff` instead solves, per test scenario, for the `h^2` whose kernel weights reach `--n-eff-target` effective calibration scenarios. `n_eff` is monotone in `h^2`, so this is a log-space bisection, vectorized across a block of test scenarios.
- Prefer `n_eff` when the same configuration has to work across different embedding widths, stratum sizes, or networks: the `k_nn` that localizes well at one embedding width can be effectively uniform at another, whereas the `n_eff` target means the same thing everywhere.
- Neither mode removes the need to choose a number. `k_nn` and `n_eff_target` are both hyperparameters, and picking one by looking at calibrated results on the application set invalidates the coverage guarantee. Use a tuning partition disjoint from the set you report on.

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
| `--cp-type` | `str` | `element` or `kernel`. |
| `--strat` | `str` | Optional stratification override. When provided, it must match the saved artifact. |
| `--pf-node` | `str` | Bus ground truth used for analysis or target reconstruction when needed. |
| `--pf-edge` | `str` | Branch ground truth used for line-loading analysis or target reconstruction when needed. |
| `--embeddings` | `str` | Scenario embeddings parquet for the application scenarios. Required when the artifact is a kernel artifact. |
| `--task-name` | `str` | Task adapter name used while loading predictions. |
| `--generator-predictions` | `str` | Optional OPF generator prediction parquet. |
| `--no-figures` | flag | Skip analysis figure generation and write only calibrated parquet plus CSV outputs. |

Output behavior:

- Bus targets produce a calibrated bus parquet with interval columns merged onto the original prediction rows.
- Branch targets produce a branch-level output parquet derived from aligned bus predictions and branch topology.
- Mixed runs write separate voltage and line-loading parquet outputs.
- When the artifact uses scenario stratification such as `strat: n_outages`, `strat: outage_type`, or `strat: system_load_bin`, `calibrate` rebuilds the application-time scenario strata before applying the calibrated intervals. For `strat: pred_bins`, `calibrate` instead assigns each prediction value to a stored percentile bin for its variable.
- For kernel artifacts, `calibrate` is where the localization happens: it loads and pools the test embeddings, computes distances to the stored calibration bank within each stratum, derives the per-scenario bandwidth and kernel weights, and evaluates the weighted quantile. The interval offsets therefore vary per test scenario rather than per stratum.
- Kernel runs add three per-scenario diagnostic columns to the calibrated parquet: `kcp_n_eff`, `kcp_bandwidth_sq`, and `kcp_fallback`. See the notes below.

Analysis side outputs:

- `trust_analysis_volt.csv`
- `trust_summary_volt.csv`
- `trust_reliability_volt.csv`
- `trust_analysis_line.csv`
- `trust_summary_line.csv`
- `trust_reliability_line.csv`
- `figures/`

Reliability side outputs summarize calibration quality per variable and stratum with expected coverage (`1 - alpha`), weighted empirical coverage, coverage gap, and mean-width aggregates.

Kernel diagnostic columns:

| Column | Meaning |
| --- | --- |
| `kcp_n_eff` | Effective sample size of the kernel weights for that scenario. Close to the stratum size means the kernel is barely localizing (behaving like `element`); close to `1` means weight has collapsed onto a handful of neighbours. |
| `kcp_bandwidth_sq` | The squared bandwidth used. Under `bandwidth: knn` this is the squared distance to the `k_nn`-th nearest calibration scenario in the same stratum; under `bandwidth: n_eff` it is the solved value, and becomes the interesting diagnostic since `kcp_n_eff` is then near-constant by construction. |
| `kcp_fallback` | `True` when the scenario reverted to the stratum-wide quantile, either because `kcp_n_eff` fell below `n_eff_min` or because the weighted quantile could not reach its target level. |

A high fallback rate or an `kcp_n_eff` near the stratum size means the kernel is not adding value at the current `k_nn`; lowering `k_nn` localizes harder, raising it moves behavior toward `element`. Under `bandwidth: n_eff`, `kcp_n_eff` is pinned to the target, so read `kcp_bandwidth_sq` and `kcp_fallback` instead.

One fallback mode is specific to concentrated embedding spaces. The unobserved test residual enters the weighted quantile as a point mass of weight 1, since the test scenario sits at distance zero from itself. When no calibration scenario is close in the kernel's own metric, every calibration weight is far below 1, that point mass dominates, and the target level becomes unreachable, so the scenario falls back. This is the guarantee refusing to extrapolate rather than a defect, but a rising `kcp_fallback` rate is a signal that the bandwidth has been driven too small for the geometry.

Figure aggregation semantics:

- Summary and density figures are generated from entity-level analysis rows.
- `*_width_density*.png` plots are densities of `width_mean` (per-entity mean interval width over scenarios), not densities of raw per-row `interval_width` samples.

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
| `--lower-bound` | `float` | Override lower bound for one screened variable. Not used for `line_loading`. |
| `--upper-bound` | `float` | Override upper bound for one screened variable. |
| `--output` | `str` | Output directory for screening artifacts. |
| `--calibrated-predictions` | `str` | Shared calibrated parquet input for a single target family. |
| `--calibrated-predictions-volt` | `str` | Voltage calibrated parquet override. |
| `--calibrated-predictions-line` | `str` | Line-loading calibrated parquet override. |
| `--pf-node` | `str` | Bus ground truth used when voltage targets are missing from the calibrated parquet. |
| `--pf-edge` | `str` | Branch ground truth used when line targets are missing from the calibrated parquet. |
| `--task-name` | `str` | Task name used for default path resolution. |

Current semantics:

- lower violation alarm when `pi_lower < lower_bound` for variables that use a lower bound, such as `Vm`
- upper violation alarm when `pi_upper > upper_bound`
- `line_loading` uses only the upper-bound screening rule
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
- `kernel`

### `strat`

- `n_outages`
- `outage_type`
- `system_load_bin`
- `pred_bins`

### `distance` (kernel only)

- `euclidean` (default)
- `cosine`

### `bandwidth` (kernel only)

- `knn` (default)
- `n_eff`

### `embedding_pooling` (kernel only)

- `mean` (default)
- `max`
- `meanmax`
- `concat`

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
