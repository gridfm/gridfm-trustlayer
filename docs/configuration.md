# Configuration Reference

This page documents the current trustlayer YAML structure used by `compute`, `calibrate`, and `screen`.

To build a config without editing YAML by hand, use the Jupyter form `gridfm_trustlayer.interactive.config_builder()` (`pip install "gridfm-trustlayer[interactive]"`, or open `examples/notebooks/config_builder.ipynb`). It shows only the options that apply to your choices and validates the result before saving.

## Config resolution order

The CLI resolves values in this order:

1. shared top-level config values
2. the command's dataset under `inputs` (`inputs.calibration` for `compute`, `inputs.test` for `calibrate` and `screen`) and the shared `outputs`
3. command-specific overrides under `commands.<verb>`
4. explicit CLI flags

CLI flags always override the YAML.

## Example config

```yaml
version: 1

task_name: PowerFlow
cp_type: element
strat: pred_bins
pred_n_bins: 5
alpha: 0.1
variables:
  - Vm
  - Va
  - line_loading

screen_bounds:
  Vm:
    lower: 0.95
    upper: 1.05
  line_loading:
    upper: 1.0

inputs:
  calibration:            # fits the layer (`compute`)
    predictions: path/to/calibration/predictions.parquet
    pf_node: path/to/calibration/bus_data.parquet
    pf_edge: path/to/calibration/branch_data.parquet
  test:                   # applies and evaluates it (`calibrate`, `screen`)
    predictions: path/to/test/predictions.parquet
    pf_node: path/to/test/bus_data.parquet
    pf_edge: path/to/test/branch_data.parquet

outputs:
  artifact: trustlayer_outputs/powerflow/element/grid_predictions_vm_va_line_loading.trustlayer
  calibrated_predictions_volt: trustlayer_outputs/powerflow/element/grid_predictions_vm_va_with_intervals.parquet
  calibrated_predictions_line: trustlayer_outputs/powerflow/element/grid_predictions_line_loading_with_intervals.parquet
  screen_dir: trustlayer_outputs/powerflow/element/screen

commands:
  screen:
    variables:
      - Vm
      - line_loading
```

## Example kernel config

`cp_type: kernel` adds an embeddings input to each dataset and three kernel knobs:

```yaml
version: 1

task_name: PowerFlow
cp_type: kernel
strat: n_outages
alpha: 0.1

# Kernel knobs
embedding_pooling: mean
distance: euclidean # or cosine
bandwidth: n_eff    # default; or knn
n_eff_target: 600   # default; or auto (see below)
# k_nn: 20          # used when bandwidth: knn
n_eff_min: 5

variables:
  - Vm
  - Va
  - line_loading

screen_bounds:
  Vm:
    lower: 0.95
    upper: 1.05
  line_loading:
    upper: 1.0

inputs:
  calibration:
    predictions: path/to/calibration/predictions.parquet
    embeddings: path/to/calibration/bus_embeddings.parquet
    pf_node: path/to/calibration/bus_data.parquet
    pf_edge: path/to/calibration/branch_data.parquet
  test:
    predictions: path/to/test/predictions.parquet
    embeddings: path/to/test/bus_embeddings.parquet
    pf_node: path/to/test/bus_data.parquet
    pf_edge: path/to/test/branch_data.parquet

outputs:
  artifact: trustlayer_outputs/powerflow/kernel/predictions_vm_va_line_loading.trustlayer
  calibrated_predictions_volt: trustlayer_outputs/powerflow/kernel/predictions_vm_va_with_intervals.parquet
  calibrated_predictions_line: trustlayer_outputs/powerflow/kernel/predictions_line_loading_with_intervals.parquet
  screen_dir: trustlayer_outputs/powerflow/kernel/screen
```

## Top-level keys

| Key                      | Type              | Required                                         | Description                                                                                                     |
| ------------------------ | ----------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `version`              | `int`           | No                                               | Config schema marker.                                                                                           |
| `task_name`            | `str`           | Usually yes                                      | Task adapter used when reading prediction artifacts.                                                            |
| `cp_type`              | `str`           | Yes                                              | Current supported values:`element`, `kernel`.                                                                 |
| `strat`                | `str or null`   | No                                               | Optional residual stratification. Current supported values:`n_outages`, `outage_type`, `system_load_bin`, `pred_bins`. |
| `system_load_n_bins`   | `int or null`   | Required when`strat: system_load_bin`          | Requested number of percentile bins learned from per-scenario total active load.                                |
| `pred_n_bins`          | `int or null`   | Required when`strat: pred_bins`                | Requested number of percentile bins learned from the calibration predictions for each calibrated variable. |
| `alpha`                | `float`         | Required for`compute`                          | Miscoverage level used while fitting intervals.                                                                 |
| `embedding_pooling`    | `str or null`   | No (kernel only)                                 | How per-entity embeddings are pooled into one vector per scenario:`mean` (default), `max`, `meanmax`, `concat`. Stored in the artifact; `compute` and `calibrate` must agree. |
| `distance`             | `str or null`   | No (kernel only)                                 | Distance over scenario representations:`euclidean` (default) or `cosine`.                                     |
| `bandwidth`            | `str or null`   | No (kernel only)                                 | How the kernel bandwidth is chosen: `n_eff` (default) or `knn`.                                                |
| `k_nn`                 | `int or null`   | No (`bandwidth: knn`)                            | Nearest-neighbour rank whose squared distance becomes the kernel bandwidth `h^2`. Default `20`. Lower localizes harder; higher moves behavior toward `element`. |
| `n_eff_target`         | `float, auto or null` | No (`bandwidth: n_eff`)                    | Target effective sample size solved for per test scenario. Defaults to `600`. Must be at least `n_eff_min`. `auto` chooses it by cross-validation on the calibration data (see [below](#choosing-n_eff_target-automatically-n_eff_target-auto)). |
| `tuning`               | `mapping or null` | No (`n_eff_target: auto` only)                 | Options of the cross-validated choice of `n_eff_target`: `folds`, `grid`, `group_by`, `seed`, `coverage_tolerance`. Rejected unless `n_eff_target: auto`. |
| `n_eff_min`            | `float or null` | No (kernel only)                                 | Minimum effective sample size before kernel weights revert to the stratum-wide quantile. Default `5`.          |
| `variables`            | `list[str]`     | Usually yes                                      | Any subset of`Vm`, `Va`, and `line_loading`. Defaults to `Vm, Va` if omitted.                           |
| `screen_bounds`        | `mapping`       | Required for`screen` on each screened variable | Per-variable screening bounds.`Vm` uses lower and upper bounds; `line_loading` uses only `upper`.         |
| `save_figures`         | `bool or null`  | No                                               | Optional default for `calibrate` figure generation. Set `false` to skip writing analysis figures while keeping parquet and CSV outputs. |

`strat: n_outages` means residual quantiles are fitted and applied per scenario outage count. The current `n_outages` value includes both inactive branches and inactive generators detected in the raw datakit scenario files.

`strat: outage_type` means residual quantiles are fitted and applied per scenario outage regime. The current implementation uses four labels derived from the raw datakit scenario: `none`, `branch_only`, `gen_only`, and `mixed`.

`strat: system_load_bin` means residual quantiles are fitted and applied per percentile bin of total active demand. Trustlayer derives total load from the raw bus `Pd` column, learns percentile cut points during `compute`, stores them in the artifact, and reapplies the same cut points during `calibrate`.

`cp_type: element` means residual quantiles are reduced to a fixed per-entity table during `compute` and looked up during `calibrate`.

`cp_type: kernel` means the artifact instead stores the calibration residual bank plus one embedding per calibration scenario, and the quantile is estimated per test scenario at `calibrate` time by weighting calibration residuals with a Gaussian kernel over embedding distance.

`cp_type` and `strat` compose rather than competing. With `cp_type: kernel` and `strat: n_outages`, the nearest-neighbour search, the bandwidth, and the weighted quantile all draw exclusively from calibration scenarios in the same outage stratum. The one excluded combination is `cp_type: kernel` with `strat: pred_bins`, because prediction bins are assigned per element while kernel localization is per scenario.

`strat: pred_bins` means residual quantiles are fitted and applied per percentile bin of the model's own predictions. Trustlayer pools all calibration predictions for each variable, learns percentile cut points from those values during `compute`, stores them in the artifact, and then assigns each prediction value to a bin during `calibrate`.

The produced label columns and summary labels use percentile-range notation with cut-point hints, for example `p00-p20 (<1450)`, `p20-p40 [1450, 1620)`, or `p80-p100 (>=1920)`. For `pred_bins`, the same label pattern is emitted on variable-specific columns such as `pred_bins_vm_label`, but the bins are assigned per prediction row rather than per scenario.

## `inputs`

The layer is always fitted and evaluated on two separate datasets, and `inputs` holds one block for each:

- `inputs.calibration`: the data `compute` fits the layer on.
- `inputs.test`: the data `calibrate` applies the layer to and `screen` evaluates it on.

Coverage is a statement about new data, so it is only meaningful on scenarios the layer was not fitted on. Trustlayer therefore never splits one dataset internally. Generate the two sets independently, for example two GridFM-datakit runs with the same settings and different seeds. If your data has structure, for example several topologies per load profile, keep related scenarios on the same side, or near-duplicate scenarios will make coverage and kernel localization look better than they are.

Two checks enforce the separation. The config is rejected if both blocks point at the same `predictions` file. In addition, the artifact stores a fingerprint of the calibration predictions, and `calibrate` refuses test predictions identical to them, for instance a renamed copy.

Each block accepts:

| Key                       | Type            | Description                                                                                                            |
| ------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `predictions`           | `str`         | Bus-level prediction parquet. The test block also sets the default output names of `calibrate` and `screen`.         |
| `pf_node`               | `str`         | Raw bus ground truth of that dataset.                                                                                  |
| `pf_edge`               | `str`         | Raw branch ground truth of that dataset (also provides the branch parameters for `line_loading`).                    |
| `embeddings`            | `str or null` | Per-entity embeddings parquet with `scenario`, `bus`, and `emb_*` columns. Required in both blocks for `cp_type: kernel`, ignored otherwise. |
| `generator_predictions` | `str or null` | Optional OPF generator prediction parquet.                                                                             |

`inputs.artifact` may also be set at the top of `inputs` (next to the two blocks) as an alternative to `outputs.artifact` for the path `calibrate` reads.

## `outputs`

| Key                             | Type    | Used by                    | Description                                                                                  |
| ------------------------------- | ------- | -------------------------- | -------------------------------------------------------------------------------------------- |
| `artifact`                    | `str` | `compute`, `calibrate` | `.trustlayer` path written by `compute` and read by `calibrate`.                       |
| `output`                      | `str` | `calibrate`, `screen`  | Shared output override. For mixed`calibrate` runs it must be a directory.                  |
| `calibrated_predictions`      | `str` | `calibrate`, `screen`  | Shared calibrated parquet path for a single target family, or directory root for mixed runs. |
| `calibrated_predictions_volt` | `str` | `calibrate`, `screen`  | Explicit voltage calibrated parquet path.                                                    |
| `calibrated_predictions_line` | `str` | `calibrate`, `screen`  | Explicit line-loading calibrated parquet path.                                               |
| `screen_dir`                  | `str` | `screen`                 | Output directory for screening CSVs and figures.                                             |

## `screen_bounds`

Each screened variable must define the bounds relevant to that variable. `Vm` uses both `lower` and `upper`, while `line_loading` uses only `upper`.

Example:

```yaml
screen_bounds:
  Vm:
    lower: 0.95
    upper: 1.05
  line_loading:
    upper: 1.0
```

Notes:

- Bounds are interpreted in the variable's natural units.
- `Vm` uses per-unit voltage magnitude.
- `line_loading` uses a ratio relative to `rate_a`, so `1.0` means 100% of the branch rating.
- `line_loading` screening is upper-bound-only; the lower side is ignored.
- CLI `--lower-bound` and `--upper-bound` can override bounds only when screening exactly one variable. For `line_loading`, only `--upper-bound` is required.

## `commands`

Each command can override shared values under `commands.compute`, `commands.calibrate`, or `commands.screen`.

Supported override keys:

- `task_name`
- `alpha`
- `cp_type`
- `strat`
- `system_load_n_bins`
- `pred_n_bins`
- `embedding_pooling`
- `k_nn`
- `n_eff_min`
- `variables`
- `output`
- `artifact`
- `screen_bounds`
- `save_figures` (calibrate only)
- nested `outputs`

Inputs are not overridden per command: each command already reads its own dataset block.

Example:

```yaml
commands:
  screen:
    variables:
      - Vm
      - line_loading
    outputs:
      screen_dir: trustlayer_outputs/powerflow/element/screen
```

You can also override stratification per command, for example:

```yaml
commands:
  compute:
    strat: outage_type
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

## Kernel embeddings

`cp_type: kernel` needs a scenario representation `z(x)`. Trustlayer reads a per-entity embeddings parquet and pools it to one vector per scenario.

Expected columns:

- `scenario`
- `bus`
- `emb_000`, `emb_001`, ... (any number of columns with the `emb_` prefix, sorted by name)

Graphkit writes exactly this layout with `gridfm_graphkit predict --get_embeddings`.

Requirements:

- The parquet must be rectangular: every scenario must carry the same number of entities.
- It must cover every scenario present in the predictions parquet.
- Pooling must match between `compute` and `calibrate`; the strategy name is stored in the artifact and mismatches are rejected.

Pooling strategies, for a scenario with `n` entities and `d`-wide per-entity embeddings:

| `embedding_pooling` | Output width | Notes |
| --- | --- | --- |
| `mean` | `d` | Default. Topology-invariant and the recommended starting point. |
| `max` | `d` | Emphasizes the most-activated entity per dimension. |
| `meanmax` | `2d` | Concatenation of mean and max pooling. |
| `concat` | `n * d` | Only valid when the entity count is fixed. Very high dimensional, so Euclidean distance concentrates and the kernel may stop localizing. |

## Distance metric

The kernel needs a notion of "similar scenario". Two metrics are implemented over the pooled, standardized scenario representations.

### `distance: euclidean` (default)

Squared Euclidean distance, `||z_a - z_b||^2`.

### `distance: cosine`

The representations are L2-normalized first, which makes squared Euclidean distance exactly `2 - 2*cos`. Implementation-wise it is one row normalization followed by the same computation, so it costs nothing extra.

The geometry it implies is worth being deliberate about. Standardization centers the representations, so each vector is already a *deviation from the mean scenario*. Normalizing that keeps only the direction of the deviation and discards its magnitude — cosine asks "in which way is this scenario unusual", never "how unusual is it".

Whether that helps depends on whether deviation magnitude carries the error signal. On IEEE-24 line loading it does not help: at `k_nn: 75`, cosine widened the mean correction from 0.00636 to 0.00855 and pushed marginal coverage from 0.922 to 0.948, ending up wider than plain SCP. Its higher recall (0.974 vs 0.966) is only the mechanical consequence of wider bounds, and precision fell slightly, so it is a move along the precision-recall curve rather than an improvement over it. Treat cosine as a cheap experiment to run, not a default to adopt.

Cosine is also **not** a remedy for distance concentration. Angular distances concentrate in high dimension just as Euclidean distances do; reducing dimensionality is the lever for that, not changing metric.

### Choosing between them

The right metric is the one under which the conditional residual distribution varies smoothly, which is an empirical question per dataset. Because both metrics are functions of the covariates alone, switching between them cannot affect the coverage guarantee — only efficiency. Selecting between them by inspecting the application set, however, has the same problem as tuning `k_nn` that way.

## Bandwidth selection

The bandwidth `h^2` controls how sharply the kernel localizes, and is the main lever on KCP behavior in either mode:

- Too small: weights concentrate on a few neighbours, the tail quantile becomes noisy, intervals turn conservative and marginal coverage overshoots `1 - alpha`.
- Too large: weights approach uniform and results converge to `cp_type: element`.

Two selection modes are implemented.

### `bandwidth: knn`

`h^2` is the squared distance to the `k_nn`-th nearest calibration scenario within the stratum.

Its weakness is that `k_nn` is not a portable number. Squared Euclidean distances concentrate as embedding width grows, so the same `k_nn` can localize sharply in one representation and be effectively uniform in another. On synthetic data with 1500 calibration scenarios per stratum, `k_nn: 20` yields a median `kcp_n_eff` of about 40 at 2 dimensions, 1045 at 16, and 1498 at 384 — at the widest, `k_nn: 20` and `k_nn: 50` become indistinguishable and the knob has stopped working.

### `bandwidth: n_eff` (default)

`h^2` is instead solved per test scenario so that the kernel weights reach a target effective sample size:

```
n_eff = (sum_j u_j)^2 / sum_j u_j^2  ==  n_eff_target
```

`n_eff` increases monotonically with `h^2`, so the solve is a log-space bisection, vectorized across a block of test scenarios. It costs roughly 70% more application time than `knn` in practice, which is small in absolute terms.

Why prefer it:

- **Portable.** The same `n_eff_target` means the same thing at any embedding width, stratum size, or network. The example above resolves to exactly 200 at 2, 16, and 384 dimensions.
- **Interpretable.** "200 calibration scenarios effectively contribute" is a statement about estimator variance; "the 20th nearest neighbour sets the scale" is not, until you know the geometry.
- **Default.** The default target is 600. A stratum with fewer calibration scenarios than the target gets uniform weights (equivalent to `cp_type: element`), and `n_eff_target: auto` chooses the target from the calibration data.

Note that `n_eff_target` is still a hyperparameter, just a better-conditioned one. Choosing it by inspecting calibrated results on the application set breaks the coverage guarantee exactly as tuning `k_nn` that way would. To choose it from the calibration data instead, use `n_eff_target: auto` (next section).

### Choosing `n_eff_target` automatically (`n_eff_target: auto`)

Instead of a number, `n_eff_target: auto` lets `compute` choose the target by cross-validation **on the calibration dataset alone**. The test dataset is never used, so the coverage that `calibrate` reports on it stays an honest out-of-sample check. Tuning is off by default; it requires `cp_type: kernel` and `bandwidth: n_eff`.

```yaml
cp_type: kernel
bandwidth: n_eff
n_eff_target: auto
tuning:                       # optional; these are the defaults
  folds: 5
  search: grid
  grid: [100, 200, 400, 600, 800, 1200]
  group_by: auto
  seed: 0
  coverage_tolerance: auto
```

To also choose the distance, and to bound the cost when both dimensions vary, use random search:

```yaml
tuning:
  distances: [euclidean, cosine]   # second dimension; omit to keep the config's `distance`
  search: random
  budget: 12                       # number of (n_eff_target, distance) settings to evaluate
  n_eff_range: [100, 1500]         # n_eff_target drawn log-uniformly from this range
```

**How it works.** A candidate is an (`n_eff_target`, `distance`) pair. By default only `n_eff_target` varies and the config's `distance` is kept. Listing `distances` adds the distance as a second dimension. The candidates are either every combination of `grid` and `distances` (`search: grid`), or `budget` distinct settings drawn at random (`search: random`). The calibration scenarios are split into `folds` folds. For every candidate and every fold, the kernel is fitted on the other folds and applied to the held-out one. That gives cross-validated coverage, overall and in every stratum, and the mean interval width. Then:

1. A candidate is **eligible** if its coverage reaches `1 - alpha - tolerance` overall and in every stratum, for every calibrated variable.
2. Among eligible candidates, the one with the **narrowest intervals** is chosen. Width is measured relative to the evaluated candidate with the largest `n_eff_target` (the one closest to uniform weights), so candidates with different distances are compared on one scale.
3. If **no candidate is eligible**, kernel localization does not reach the target on this data, and the target is set to the size of the largest stratum. This makes the weights uniform within every stratum, which is equivalent to `cp_type: element`. The report says so explicitly.

Coverage comes first and width second, because kernel-weighted CP has no finite-sample guarantee: a narrow interval is only useful if its coverage holds. Each target family (voltages, line loading) is tuned separately.

| Key | Default | Meaning |
| --- | --- | --- |
| `folds` | `5` | Number of cross-validation folds (at least 2). More folds give each fit more data but take proportionally longer. |
| `grid` | `[100, 200, 400, 600, 800, 1200]` | Candidate `n_eff_target` values for `search: grid`. A value above a stratum's size gives uniform weights in that stratum. All values must be at least `n_eff_min`. |
| `distances` | not set | Distances to compare (`euclidean`, `cosine`). Not set: only the config's `distance`. Both distances support `bandwidth: n_eff`; cosine only changes how the embeddings are prepared (projected onto the unit sphere). |
| `search` | `grid` | `grid`: every combination of `grid` and `distances`. `random`: `budget` distinct settings, with `n_eff_target` drawn log-uniformly from `n_eff_range` (so not limited to grid values) and the distance uniformly from `distances`. |
| `budget` | `10` | Number of settings evaluated by `search: random`. The cost grows with the budget, not with the size of the search space. A small budget spread over two distances tries only a few targets per distance and can miss the best one, which gives valid but wider intervals than a finer search. Prefer `search: grid` when only `n_eff_target` varies. |
| `n_eff_range` | span of `grid` | `[low, high]` for `search: random`. `low` must be at least `n_eff_min`. |
| `group_by` | `auto` | Keeps related scenarios in one fold. `auto` uses datakit's `load_scenario_idx` column of the calibration `pf_node` file when it exists; `load_scenario_idx` requires it; `none` splits scenarios at random. |
| `seed` | `0` | Seed of the fold assignment, so a run can be reproduced. |
| `coverage_tolerance` | `auto` | How far below `1 - alpha` cross-validated coverage may fall and still count as meeting the target. `auto` uses two standard errors for the number of scenarios behind each check, `2 * sqrt(alpha * (1 - alpha) / n_scenarios)`, so small strata get more room; a number (e.g. `0.005`) applies the same tolerance to every check. |

**Why group the folds.** When datakit generates several topologies per load profile, those scenarios are near-duplicates. If they land on both sides of a fold, every held-out scenario has a near-twin in the bank, and localization looks better than it will on new data. Grouping by `load_scenario_idx` prevents that. With one topology per load profile, the grouping changes nothing.

**Why `n_eff_target` and not `k_nn`.** An effective sample size is absolute: a target chosen on folds holding 80% of the calibration data means the same on all of it, so the value transfers to the final fit. A good `k_nn` shifts with the size of the bank.

**Outputs.**
- The chosen values are stored in the artifact (`n_eff_target` and `distance`), together with a `tuning` record: the candidates, the selection reason, the search, the folds and whether they were grouped.
- `compute` writes `<artifact>_kernel_tuning.csv` (one row per candidate: coverage, worst margin to the target, relative width, fallback share, eligible, selected) and `<artifact>_kernel_tuning_details.csv` (every coverage check).
- It also prints a one-line summary per target family.

**Cost.** About one kernel run per fold and candidate: `folds x len(grid) x len(distances)` runs for a grid search, `folds x budget` for a random search, each on `(folds - 1) / folds` of the calibration data. With about 20k calibration scenarios and 186 branches, the default grid takes about two minutes on a laptop CPU. Most of the time goes into the per-element weighted quantiles, which depend on each candidate's weights, so it cannot be shared across candidates.

**Limits.** Cross-validation gives an honest empirical estimate of coverage, not a formal guarantee: the chosen kernel is still verified empirically, as any KCP configuration is. A grid search cannot pick a value between two candidates; refine the grid around the chosen value, or use `search: random` over a range, if the precise target matters.


### Interaction with `n_eff_min`

Under `bandwidth: knn`, `n_eff_min` is the guard that catches scenarios whose weights collapsed onto too few neighbours.

Under `bandwidth: n_eff` the target is hit by construction, so `n_eff_min` only binds when a stratum is too small to reach the target at all. The config rejects `n_eff_target < n_eff_min`, since every scenario would fall back immediately.

Worth knowing: for the `1 - alpha/2` level to be expressible at all, the summed kernel weight must satisfy `sum_j u_j >= 2/alpha - 1`, which is 19 at `alpha: 0.1`. Since weights are bounded by 1, this implies `n_eff >= 19` is necessary though not sufficient. The default `n_eff_min: 5` therefore never binds before unreachability does; if you want the guard to do real work, set it to at least `2/alpha`.

## Recommended patterns

### PowerFlow screening for actual violations

Use PF prediction and ground-truth artifacts, then set operating bounds to your true acceptance region. This is the most natural setting for confusion-matrix style violation detection.

### OPF screening for near-limit alarms

OPF solutions are often feasible by construction. If you screen against the same solver limits, you may see few or no true violations. In that case, tighter operational margins such as `Vm in [0.97, 1.03]` or `line_loading <= 0.95` are often more informative.

## Runnable examples

`examples/config/` ships one complete config per option combination — element and kernel,
each stratification mode, both bandwidth modes, both distance metrics, and an OPF variant.
`examples/config/README.md` maps each file to what it demonstrates.

Two conventions those examples encode, both easy to get wrong:

- **Restrict `commands.screen.variables` to variables that have bounds.** `screen` otherwise
  defaults to the shared `variables` list, which usually includes `Va`, and voltage angle has no
  operating limit to screen against. This passes config validation and then fails at run time.
- **Output keys depend on how many target families you request.** A single-family run (voltages
  only, or `line_loading` only) resolves `outputs.calibrated_predictions`. Only a mixed run reads
  `outputs.calibrated_predictions_volt` and `outputs.calibrated_predictions_line`.

## See also

- [CLI Reference](cli.md)
- [Schemas](schemas.md)
