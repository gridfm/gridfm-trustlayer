# Analysis and Figures Reference

This page explains how trustlayer computes each analysis metric and each figure, including stratification behavior and how to interpret calibrated versus non-calibrated errors.

## Scope

This page covers outputs produced by:

- `gridfm_trustlayer calibrate`
- `gridfm_trustlayer screen`

Main artifacts discussed here:

- `trust_analysis_volt.csv`
- `trust_summary_volt.csv`
- `trust_reliability_volt.csv`
- `trust_analysis_line.csv`
- `trust_summary_line.csv`
- `trust_reliability_line.csv`
- `figures/` and `screen_figures/`

## Key principle: what calibration changes and what it does not

Trustlayer interval calibration adds interval columns around existing model predictions.

It does not change the original point prediction values (`*_pred`).

So point-prediction error before and after calibrate is the same:

$$
\hat{y}^{\text{after}}_{s,e} = \hat{y}^{\text{before}}_{s,e}.
$$

Any change in quality after calibration should be read through interval metrics (coverage, width, screening behavior), not through point-error metrics.

## Levels of aggregation

Most confusion comes from mixing these three levels:

1. Sample level: one `(scenario, entity)` pair.
2. Entity level: per bus or per branch, averaged over scenarios.
3. Stratum level: per stratification group (for example one `pred_bins_va` bin), often aggregating many entities.

## Core interval metrics

For each sample:

- lower endpoint: $\ell_{s,e}$
- upper endpoint: $u_{s,e}$
- target: $y_{s,e}$

### Interval width

$$
\text{interval\_width}_{s,e} = u_{s,e} - \ell_{s,e}.
$$

### Coverage indicator

$$
\text{covered}_{s,e} = \mathbf{1}[\ell_{s,e} \le y_{s,e} \le u_{s,e}].
$$

### Empirical coverage at entity level

For one entity $e$ in a group $G$:

$$
\text{empirical\_coverage}_e = \frac{1}{|G_e|} \sum_{(s,e) \in G_e} \text{covered}_{s,e}.
$$

### Mean interval width at entity level

$$
\text{width\_mean}_e = \frac{1}{|G_e|} \sum_{(s,e) \in G_e} \text{interval\_width}_{s,e}.
$$

## `trust_analysis_*.csv`

This is the entity-level table.

Each row corresponds to one entity (bus or branch) for one variable and one grouping context.

Important columns:

- `analysis_scope`: `overall` or `stratified`
- `stratification`: grouping key (empty for overall)
- `stratum_value`, `stratum_label`: bin id and readable label
- `width_mean`, `width_min`, `width_max`
- `empirical_coverage`
- `n_samples`, `n_scenarios`

Interpretation:

- `width_*` columns summarize interval size for that entity.
- `empirical_coverage` is that entity's coverage rate.
- For `stratified` rows, the same entity can appear in multiple strata (with different counts and metrics).

## `trust_summary_*.csv`

This is a higher-level rollup of `trust_analysis_*.csv`.

For voltage summaries:

- `scope=system`: aggregates across buses
- `scope=bus_type`: aggregates within `REF`, `PV`, `PQ`

For line summaries:

- `scope=system`: aggregates across branches
- `scope=rate_a`: aggregates by branch rating

These are mostly unweighted means across entities (plus min/max across entities).

## `trust_reliability_*.csv`

This table is designed to evaluate calibration quality by variable and stratum.

Each row groups entity-level rows by:

- `analysis_scope`
- `stratification`
- `stratum_value`
- `variable`

Main columns and formulas:

- `target_coverage = 1 - alpha`
- `empirical_coverage_mean_across_entities`: simple mean across entity coverages
- `empirical_coverage_weighted`: weighted by each entity `n_samples`

$$
\text{empirical\_coverage\_weighted}
= \frac{\sum_e n_e \cdot c_e}{\sum_e n_e}
$$

where $n_e$ is entity sample count and $c_e$ is entity empirical coverage.

- `coverage_gap_weighted`

$$
\text{coverage\_gap\_weighted} = \text{empirical\_coverage\_weighted} - (1-\alpha)
$$

- `width_mean_across_entities`: unweighted mean entity width
- `width_mean_weighted`: sample-weighted mean entity width

Why both weighted and unweighted coverage are shown:

- unweighted answers fairness across entities
- weighted answers global sample-level behavior

When strata are imbalanced (`n_scenarios_min` very small), these two can differ a lot.

## Stratification semantics

### Kernel diagnostics (`cp_type: kernel`)

Kernel runs add three per-scenario columns to the calibrated parquet. They are diagnostics, not stratification keys, so they do not create analysis strata and no figures are generated from them.

- `kcp_n_eff`: effective sample size of the kernel weights, $\left(\sum_j u_j\right)^2 / \sum_j u_j^2$. Near the stratum size means the kernel is barely localizing; near `1` means weight collapsed onto a few neighbours.
- `kcp_bandwidth_sq`: the squared bandwidth used. Under `bandwidth: knn` this is the squared distance to the `k_nn`-th nearest calibration scenario in the same stratum; under `bandwidth: n_eff` it is the solved value.
- `kcp_fallback`: `True` when that scenario reverted to the stratum-wide quantile.

Read these alongside the reliability CSV. If empirical coverage sits well above `1 - alpha` while `kcp_n_eff` is low, the kernel is localizing too aggressively and the bandwidth should be widened (raise `k_nn`, or raise `n_eff_target`). If `kcp_n_eff` is close to the stratum size, the run is effectively equivalent to `cp_type: element`.

Under `bandwidth: n_eff`, `kcp_n_eff` is pinned to the target by construction and carries no information, so `kcp_bandwidth_sq` becomes the diagnostic worth reading: it shows where the embedding space is locally sparse. A rising `kcp_fallback` rate under that mode usually means the target is small enough that the solved bandwidth is driving the summed kernel weight below the level needed to express the `1 - alpha/2` quantile.

### Scenario-level stratification

These assign one stratum per scenario:

- `n_outages`
- `outage_type`
- `system_load_bin`

Rows from many entities share the same scenario stratum when the scenario matches.

### Prediction-level stratification (`pred_bins`)

This assigns stratum per prediction value, per variable.

Examples:

- `pred_bins_vm`
- `pred_bins_va`
- `pred_bins_line_loading`

A single scenario can map to different bins across buses/branches because values differ by entity.

## Figure guide (`figures/`)

For each variable, trustlayer writes overall and stratified figures.

### Overall figures (no strat suffix)

- `*_width_summary.png`: line plot of entity index vs `width_mean`
- `*_coverage_summary.png`: line plot of entity index vs `empirical_coverage`
- `*_width_density.png`: density of entity-level `width_mean` values
- `*_coverage_density.png`: density of entity-level `empirical_coverage` values

Important: density plots are over entity-level metrics, not raw sample-level `interval_width`.

### Stratified figures (`*_by-<strat>.png`)

- `*_width_summary_by-*.png`: entity trajectories per stratum
- `*_coverage_summary_by-*.png`: entity trajectories per stratum
- `*_width_density_by-*.png`: density per stratum of entity `width_mean`
- `*_coverage_density_by-*.png`: density per stratum of entity `empirical_coverage`
- `*_width_mean_by-*.png`: boxplot distribution across entities of `width_mean` per stratum
- `*_coverage_mean_by-*.png`: boxplot distribution across entities of `empirical_coverage` per stratum

The boxplots are the best place to inspect cross-entity spread and outliers within each stratum.

## Screening figures (`screen_figures/`)

Screening outputs use interval endpoints against operating limits.

Types:

- any-violation metric density plots
- confusion-matrix style plots per side (`lower` or `upper`)

These evaluate risk detection behavior under uncertainty bounds, not point-prediction regression error.

## Calibrated vs non-calibrated error interpretation

Because point predictions are unchanged by calibrate, non-calibrated and calibrated point errors are identical.

If you want explicit point-error tracking, compute it from prediction and target columns:

- absolute error: $|y_{s,e} - \hat{y}_{s,e}|$
- squared error: $(y_{s,e} - \hat{y}_{s,e})^2$

Trustlayer's built-in analysis tables focus on interval quality instead:

- coverage quality
- interval width
- coverage-width tradeoff by stratum

## Practical reading checklist

When reviewing one variable and one stratification:

1. Check `trust_reliability_*.csv` weighted coverage and weighted gap against target.
2. Check boxplots (`*_coverage_mean_by-*.png`) for spread across entities.
3. Check `n_scenarios_min` and `n_scenarios_max` to detect sparse strata.
4. Use unweighted coverage for fairness view and weighted coverage for global behavior.
