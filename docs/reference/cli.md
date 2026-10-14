# CLI reference

The `gridfm_trustlayer` command has three subcommands. Each reads the YAML config given with `--config`; flags override individual config values for one run. `gridfm_trustlayer <command> --help` prints the same options.

## Command overview

Entrypoint:

```bash
gridfm_trustlayer <command> [OPTIONS]
python -m gridfm_trustlayer <command> [OPTIONS]   # equivalent
gridfm_trustlayer --version
```

Available commands:

- `compute`
- `calibrate`
- `screen`
- `apply`

The three run in this order:

1. `compute` fits the calibrators on the calibration dataset (`inputs.calibration`) and writes a `.trustlayer` artifact.
2. `calibrate` applies the artifact to the test dataset (`inputs.test`) and writes the predictions with their bounds, coverage reports and figures.
3. `screen` compares the test bounds with the operating limits in `screen_bounds`.

`apply` uses the artifact on new scenarios without a known solution (`inputs.apply`), at any point after `compute`.

Each command ends with a short summary: what `compute` fitted and where the artifact is; the coverage and mean width per variable, with the weakest stratum, after `calibrate`; the violations caught and missed and the false alarms after `screen`. Both also say how many known values (set-points) were left out. On large test sets, `calibrate` and `screen` also print one line per block of scenarios as they go.

```bash
gridfm_trustlayer compute --config path/to/trust_config.yaml
gridfm_trustlayer calibrate --config path/to/trust_config.yaml
gridfm_trustlayer screen --config path/to/trust_config.yaml
```

## `compute`

Purpose:

- load the calibration predictions and ground truth (`inputs.calibration`)
- fit one conformal calibrator per target family (`element` or `kernel`)
- with `n_eff_target: auto`, choose the kernel bandwidth by cross-validation first
- save the result as a `.trustlayer` artifact

Syntax:

```bash
gridfm_trustlayer compute --config path/to/trust_config.yaml
```

| Option | Type | Description |
| --- | --- | --- |
| `--config` | `str` | Required YAML config. |
| `--predictions` | `str` | Calibration predictions parquet with `*_pred` columns (overrides `inputs.calibration.predictions`). |
| `--pf-node` | `str` | Calibration bus ground truth (overrides `inputs.calibration.pf_node`). |
| `--pf-edge` | `str` | Calibration branch ground truth, also used for loading computation (overrides `inputs.calibration.pf_edge`). |
| `--output` | `str` | Artifact output path. |
| `--alpha` | `float` | Miscoverage level. |
| `--cp-type` | `str` | `element` or `kernel`. |
| `--strat` | `str` | Optional stratification: `n_outages`, `outage_type`, `system_load_bin`, or `pred_bins`. |
| `--system-load-n-bins` | `int` | Requested number of percentile bins when `--strat system_load_bin` is used. |
| `--pred-n-bins` | `int` | Requested number of percentile bins when `--strat pred_bins` is used. |
| `--sparse-strata` | `str` | `pooled` (default) or `infinite`: the bounds of an element in a stratum with too few calibration samples. See [Sparse strata](../user-guide/choosing-a-setup.md#sparse-strata-sparse_strata). |
| `--variables` | `list[str]` | Any subset of `Vm`, `Va`, `line_loading`. |
| `--embeddings` | `str` | Scenario embeddings parquet with `emb_*` columns. Required for `--cp-type kernel`. |
| `--embedding-pooling` | `str` | How per-entity embeddings are pooled into one vector per scenario: `mean` (default), `max`, `meanmax`, or `concat`. |
| `--distance` | `str` | Distance over scenario representations: `euclidean` (default) or `cosine`. |
| `--bandwidth` | `str` | How the kernel bandwidth is chosen: `n_eff` (default) or `knn`. |
| `--k-nn` | `int` | Nearest-neighbour rank whose distance sets the kernel bandwidth. Used by `--bandwidth knn`. Default `20`. |
| `--n-eff-target` | `float` or `auto` | Target effective sample size solved for per test scenario. Used by `--bandwidth n_eff`. Defaults to `600`. `auto` chooses it by cross-validation on the calibration data, with the options of the config's `tuning` block. |
| `--n-eff-min` | `float` | Minimum effective sample size before kernel weights revert to the stratum-wide quantile. Default `5`. |
| `--task-name` | `str` | Task adapter name such as `PowerFlow` or `OptimalPowerFlow`. |
| `--generator-predictions` | `str` | Optional generator parquet for OPF prediction artifacts. |

Notes:

- `compute` fits on every scenario of the calibration dataset; trustlayer never splits a dataset internally. Evaluate the result with `calibrate` on a separate test dataset (see [Preparing the data](../user-guide/data.md#two-separate-datasets)).
- A mixed run such as `variables: [Vm, Va, line_loading]` still writes one artifact bundle.
- The saved artifact may contain one calibrator or multiple calibrators keyed by target family.
- With `--strat n_outages`, residual quantiles are fitted separately per scenario outage count.
- With `--strat system_load_bin`, residual quantiles are fitted separately per percentile bin of total active load derived from the raw bus `Pd` column.
- With `--strat pred_bins`, residual quantiles are fitted separately per percentile bin of the model's own predictions, using one learned edge set per calibrated variable and assigning bins per prediction value.
- The resulting labels in calibrated outputs and analysis artifacts are percentile ranges with cut-point hints, such as `p00-p20 (<1450)` or `p20-p40 [1450, 1620)`.

Kernel-specific notes:

- `--cp-type kernel` requires `--embeddings` for the calibration dataset at `compute` time and for the test dataset at `calibrate` time.
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

- `--bandwidth n_eff` (default) solves, per test scenario, for the `h^2` whose kernel weights reach `--n-eff-target` effective calibration scenarios (600 by default). `n_eff` is monotone in `h^2`, so this is a one-dimensional root find, solved with safeguarded Newton steps in log space, vectorized across a block of test scenarios.
- `--bandwidth knn` instead sets `h^2` to the squared distance to the `k_nn`-th nearest calibration scenario in the same stratum.
- Prefer `n_eff` when the same configuration has to work across different embedding widths, stratum sizes, or networks: the `k_nn` that localizes well at one embedding width can be effectively uniform at another, whereas the `n_eff` target means the same thing everywhere.
- Neither mode removes the need to choose a number. `k_nn` and `n_eff_target` are both hyperparameters, and picking one by looking at calibrated results on the application set invalidates the coverage guarantee. Use a tuning partition disjoint from the set you report on, or let `--n-eff-target auto` choose it by cross-validation inside the calibration data.
- With `--n-eff-target auto`, `compute` writes `<artifact>_kernel_tuning.csv` and `<artifact>_kernel_tuning_details.csv` next to the artifact and prints the chosen value per target family. See [Kernel-weighted calibration](../user-guide/kernel.md#choosing-the-bandwidth-automatically).

## `calibrate`

Purpose:

- load a `.trustlayer` artifact
- apply the fitted intervals to the test predictions (`inputs.test`)
- write calibrated prediction outputs
- emit analysis CSVs and figures

Syntax:

```bash
gridfm_trustlayer calibrate --config path/to/trust_config.yaml
```

| Option | Type | Description |
| --- | --- | --- |
| `--config` | `str` | Required YAML config. |
| `--predictions` | `str` | Test predictions parquet (overrides `inputs.test.predictions`). Rejected if identical to the calibration predictions the artifact was fitted on. |
| `--artifact` | `str` | Saved `.trustlayer` artifact from `compute`. |
| `--output` | `str` | Output dataset path for a single target family, or output directory for mixed voltage + loading runs. |
| `--cp-type` | `str` | `element` or `kernel`. |
| `--strat` | `str` | Optional stratification override. When provided, it must match the saved artifact. |
| `--pf-node` | `str` | Test bus ground truth, used for analysis or target reconstruction (overrides `inputs.test.pf_node`). |
| `--pf-edge` | `str` | Test branch ground truth, used for line loading and its analysis (overrides `inputs.test.pf_edge`). |
| `--embeddings` | `str` | Scenario embeddings parquet for the test scenarios. Required when the artifact is a kernel artifact. |
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

Reports, in `outputs.coverage_dir` (default: a `coverage/` folder next to the calibrated parquets):

- `summary.md`: method, data, known values set aside, coverage tables by stratum, figures and files
- `coverage_by_stratum.csv`: coverage (with an approximate 95% interval), worst element and mean width per variable and stratum
- `coverage_by_group.csv`: the same per bus type or line rating
- `coverage_by_element_voltage.csv`, `coverage_by_element_line_loading.csv`: the same per bus or line
- `figures/`: coverage by stratum, coverage per element, width by stratum

Known values (`Vm` at PV buses and `Va` at the reference bus, which are set-points) are left out of the fit and of every metric, and counted in `n_known`. The calibrated parquet marks them in `Vm_known` and `Va_known`. See [Reading the results](../user-guide/results.md#outputs-of-calibrate).

Kernel diagnostic columns:

| Column | Meaning |
| --- | --- |
| `kcp_n_eff` | Effective sample size of the kernel weights for that scenario. Close to the stratum size means the kernel is barely localizing (behaving like `element`); close to `1` means weight has collapsed onto a handful of neighbours. |
| `kcp_bandwidth_sq` | The squared bandwidth used. Under `bandwidth: knn` this is the squared distance to the `k_nn`-th nearest calibration scenario in the same stratum; under `bandwidth: n_eff` it is the solved value, and becomes the interesting diagnostic since `kcp_n_eff` is then near-constant by construction. |
| `kcp_fallback` | `True` when the scenario reverted to the stratum-wide quantile, either because `kcp_n_eff` fell below `n_eff_min` or because the weighted quantile could not reach its target level. Elements with no valid calibration residual in the stratum (e.g. a branch out of service in every calibration scenario) always use the fallback bound and do not set this flag. |

A `kcp_n_eff` near the stratum size means the kernel is barely localizing. A high fallback rate means it localizes too hard: raise `n_eff_target` (or `k_nn`). Under `bandwidth: n_eff`, `kcp_n_eff` is pinned to the target, so read `kcp_bandwidth_sq` and `kcp_fallback` instead. See [Kernel-weighted calibration](../user-guide/kernel.md#checking-that-it-works).

One fallback mode is specific to concentrated embedding spaces. The unobserved test residual enters the weighted quantile as a point mass of weight 1, since the test scenario sits at distance zero from itself. When no calibration scenario is close in the kernel's own metric, every calibration weight is far below 1, that point mass dominates, and the target level becomes unreachable, so the scenario falls back. This is the guarantee refusing to extrapolate rather than a defect, but a rising `kcp_fallback` rate is a signal that the bandwidth has been driven too small for the geometry.

Figure semantics:

- `*_coverage_by_stratum.png` pools all counted values of a stratum; `*_coverage_by_bus.png` / `*_coverage_by_line.png` show the coverage of each element over the whole test set.
- `*_width_by_stratum.png` shows the distribution, across elements, of each element's mean interval width, not raw per-row widths.

Mixed-run rule:

- If both voltage and line-loading targets are present, `--output` must be a directory or omitted.
- To force exact parquet names in a mixed run, use `outputs.calibrated_predictions_volt` and `outputs.calibrated_predictions_line` in the YAML.

## `apply`

Purpose:

- load a `.trustlayer` artifact
- write bounds for new scenarios (`inputs.apply`), which need no known AC solution
- no coverage analysis, figures or reports: there is no true value to compare with

Syntax:

```bash
gridfm_trustlayer apply --config path/to/trust_config.yaml
```

| Option | Type | Description |
| --- | --- | --- |
| `--config` | `str` | Required YAML config. |
| `--predictions` | `str` | Predictions parquet of the new scenarios (overrides `inputs.apply.predictions`). |
| `--artifact` | `str` | Saved `.trustlayer` artifact from `compute`. |
| `--output` | `str` | Output dataset path for one target family, or directory for both. |
| `--pf-node` | `str` | Bus file describing the new scenarios (loads, bus types). |
| `--pf-edge` | `str` | Branch file describing the new scenarios (status, parameters, ratings). |
| `--embeddings` | `str` | Embeddings parquet of the new scenarios, for kernel artifacts. |
| `--task-name` | `str` | Task adapter name used while loading predictions. |
| `--generator-predictions` | `str` | Optional OPF generator prediction parquet. |

What it reads:

- the predictions. For `PowerFlow`, they must carry the set-points of each scenario: `Vm_target` at PV buses and `Va_target` at the reference bus (`Va_target` only, for `OptimalPowerFlow`). These are inputs of the problem, and the model's predictions there are replaced by them, as when the layer was fitted. Values at other buses are not read and may be missing;
- `pf_node` and `pf_edge`, only when the layer needs the description of each scenario: its outages (`strat: n_outages`, `outage_type`), total load (`strat: system_load_bin`) or branch parameters (`line_loading`). Only topology, loads and parameters are read; the files need no solved `Vm` or `Va`;
- the embeddings, for kernel artifacts.

Outputs: one partitioned dataset per target family, `<predictions>_<variables>_bounds.parquet`, with the columns of the outputs of `calibrate` except the true values: the `*_target` columns of the predictions file are dropped, since for new scenarios they hold placeholders except at the set-points. They contain the bounds `*_pi_lower` and `*_pi_upper`, the strata, and the kernel diagnostics. The `_bounds` suffix keeps it apart from `calibrate`'s `_with_intervals` outputs, so that the two never overwrite each other. On labeled data, `apply` gives exactly the bounds of `calibrate`.

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
| `--figures` / `--no-figures` | flag | Write (default) or skip the screening figures. |

Alarm rule (see [Reading the results](../user-guide/results.md#how-an-alarm-is-defined)):

- lower violation alarm when `pi_lower < lower_bound` for variables that use a lower bound, such as `Vm`
- upper violation alarm when `pi_upper > upper_bound`
- `line_loading` uses only the upper-bound screening rule
- a numeric tolerance is applied internally to avoid floating-point boundary drift on exact limits

Saved outputs, in `outputs.screen_dir`:

- `summary.md`: limits, known values set aside, results by stratum, figures and files
- `screening_by_stratum.csv`: violations, alarms, caught, missed, false alarms, cleared, recall and precision per variable, stratum and side (`lower`, `upper`, `any`)
- `screening_by_group.csv`: the same per bus type or line rating
- `screening_by_element_voltage.csv`, `screening_by_element_line_loading.csv`: the same per bus or line
- `screening_detail_voltage.parquet/`, `screening_detail_line_loading.parquet/`: one row per screened value (partitioned datasets)
- `figures/`: confusion matrix and, with `strat`, recall and precision by stratum

Known values are not screened: they are left out of every count and of the detail, and counted in `n_known`.

## Supported values

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

- `n_eff` (default)
- `knn`

### `embedding_pooling` (kernel only)

- `mean` (default)
- `max`
- `meanmax`
- `concat`

### Variables

- `Vm`
- `Va`
- `line_loading`

### Task names

- `PowerFlow`
- `OptimalPowerFlow`

## See also

- [Configuration reference](configuration.md)
- [Data schemas](schemas.md)
- [Reading the results](../user-guide/results.md)
