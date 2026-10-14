# Reading the results

`calibrate` and `screen` write parquet datasets, CSV tables and figures. This page explains what each contains, how the numbers are computed, and in which order to read them.

The per-row outputs (the calibrated predictions and the screening detail) can have tens of millions of rows. They are written as **partitioned parquet datasets**, the layout gridfm-datakit uses: a directory with one sub-directory per 200 scenarios, `scenario_partition=0/`, `scenario_partition=1/`, and so on. Read one whole with `pd.read_parquet(path)`, or only some scenarios and columns without loading the rest:

```python title="Reading part of a partitioned output"
import pyarrow.dataset as ds

dataset = ds.dataset(
    "trustlayer_outputs/calibrated/case118_ieee_predictions_line_loading_with_intervals.parquet",
    format="parquet",
    partitioning="hive",
)
table = dataset.to_table(
    columns=["scenario", "branch", "line_loading_pi_upper"],
    filter=ds.field("scenario") < 1000,
)
```

The summary tables are small and stay CSV.

!!! tip "In short"
    1. `trust_reliability_*.csv`: is coverage close to $1-\alpha$ in every stratum?
    2. `trust_screen_summary_*.csv`, `scope = system`: how many violations were caught (`tp`), missed (`fn`) and falsely flagged (`fp`)?
    3. The figures and the per-entity tables: which buses or lines behave differently?

Calibration never changes the point predictions (`*_pred`). It only adds bounds around them, so its effect is read through coverage, width and screening, not through prediction error.

## Outputs of `calibrate`

The reports and figures go next to the calibrated datasets: in the directory that contains `outputs.calibrated_predictions_volt` / `_line`, or in `outputs.calibrated_predictions` when it is a directory for both.

### Calibrated predictions

Two partitioned datasets, one per target family:

| Dataset | One row per | Added columns |
| --- | --- | --- |
| `<predictions>_vm_va_with_intervals.parquet/` (name follows the calibrated variables) | scenario and bus | `Vm_pi_lower`, `Vm_pi_upper`, `Va_pi_lower`, `Va_pi_upper`, next to every column of the predictions file; `Vm_target` / `Va_target` from the ground truth if the predictions file has none |
| `<predictions>_line_loading_with_intervals.parquet/` | scenario and branch | `line_loading_pred`, `line_loading_target`, `line_loading_pi_lower`, `line_loading_pi_upper`, plus `from_bus`, `to_bus`, `rate_a` |

Both also carry the stratum of each row when `strat` is set (`n_outages`, `outage_type`, `system_load_bin`, or `pred_bins_<variable>`, each with a readable `_label` column), and the [kernel diagnostics](kernel.md#checking-that-it-works) `kcp_n_eff`, `kcp_bandwidth_sq` and `kcp_fallback` for `cp_type: kernel`.

Conventions:

- `line_loading` is a fraction of the branch rating `rate_a`: 1.0 is 100%. Out-of-service or unrated branches are set to `-1`.
- A bound of `inf` or `-inf` means the stratum had too few calibration samples for a finite bound at this `alpha` (fewer than $\lceil 2/\alpha \rceil - 1$).
- For entries marked invalid in the inputs (masked), no interval is formed: both bounds equal the prediction.

### Reliability: `trust_reliability_volt.csv`, `trust_reliability_line.csv`

The first table to open. One row per variable and stratum, plus an `overall` row per variable.

| Column | Meaning |
| --- | --- |
| `analysis_scope`, `stratification`, `stratum_value`, `stratum_label` | `overall`, or `stratified` with the stratum, e.g. `n_outages=2` |
| `target_coverage` | $1-\alpha$ |
| `empirical_coverage_weighted` | share of all test samples (scenario × entity) whose true value lies inside its bounds |
| `coverage_gap_weighted` | `empirical_coverage_weighted - target_coverage` |
| `empirical_coverage_mean_across_entities` | the same, averaged per bus or line first, so every entity counts equally |
| `width_mean_weighted`, `width_mean_across_entities` | mean interval width $u - \ell$, weighted the same two ways |
| `n_entities`, `n_samples_total` | buses or lines, and samples, behind the row |
| `n_scenarios_min`, `n_scenarios_max` | fewest and most test scenarios that any entity has in this stratum |

**How to read it.** With `cp_type: element`, `coverage_gap_weighted` should be close to zero or positive in every row; small negative values are sampling noise, of the order of $\sqrt{\alpha(1-\alpha)/n}$ for $n$ test scenarios. A clearly negative gap in one stratum means calibration and test scenarios differ there, for example because they were generated with different settings. With `cp_type: kernel`, coverage is checked here rather than guaranteed, so this table is the verification.

Among setups that reach the target, the one with the smaller `width_mean_weighted` gives the sharper screen.

### Summary: `trust_summary_volt.csv`, `trust_summary_line.csv`

Per-entity results aggregated by group: `scope = system` over all buses or lines, `scope = bus_type` for `REF`, `PV`, `PQ` buses, and `scope = rate_a` for lines with the same rating. Each row gives the mean, minimum and maximum, across entities, of the mean width and of the coverage, so you can spot groups with poorly covered members.

### Per entity: `trust_analysis_volt.csv`, `trust_analysis_line.csv`

One row per bus or line (and per stratum): `width_mean`, `width_min`, `width_max`, `empirical_coverage`, `n_samples`, `n_scenarios`. Use it to find the specific entities with low coverage or very wide bounds.

### Figures: `figures/`

For each variable (`vm`, `va`, `line_loading`):

| File | Shows |
| --- | --- |
| `<var>_width_summary.png`, `<var>_coverage_summary.png` | mean width and coverage of every entity, by entity index |
| `<var>_width_density.png`, `<var>_coverage_density.png` | distribution of those per-entity values |

With `strat`, the same plots per stratum (`..._by-<strat>.png`), plus box plots of the per-entity values per stratum (`<var>_width_mean_by-<strat>.png`, `<var>_coverage_mean_by-<strat>.png`), which are the quickest way to see how widths grow with the stratum and how spread the coverage is.

The density and box plots summarize per-entity means, not individual samples. Infinite widths are left out of the plots but kept in the CSVs. `--no-figures` (or `save_figures: false`) skips this folder.

## Outputs of `screen`

`screen` compares the test intervals with `screen_bounds` and checks each alarm against the true AC value. Files go to `outputs.screen_dir`.

### How an alarm is defined

For each scenario and bus or line:

- **predicted violation (alarm):** `pi_upper > upper_bound`, or, for variables with a lower limit (`Vm`), `pi_lower < lower_bound`;
- **true violation:** the same test on the true value.

This is a *possible-violation* rule: it asks whether the bounds allow a violation, not whether the prediction itself violates. It is conservative by design. A real violation of the upper limit can only be missed if the true value lies above the upper bound, which happens for at most a share $\alpha/2$ of samples when coverage holds; the same applies to the lower side. `line_loading` has only an upper limit. Each alarm is then classified:

| | True violation | No true violation |
| --- | --- | --- |
| **Alarm** | `tp`: caught | `fp`: false alarm (an unnecessary AC power flow) |
| **No alarm** | `fn`: missed | `tn`: correctly cleared |

with `precision = tp / (tp + fp)` and `recall = tp / (tp + fn)`. Counts are reported for the `lower` side, the `upper` side and `any` (either side).

### `trust_screen_summary_*.csv`

The table to read first. `scope = system` aggregates all samples; `scope = bus_type` (voltage) and `scope = rate_a` (lines) break it down. Columns: `n_entities`, `n_samples`, `{lower,upper,any}_{tp,fp,fn,tn}`, `{lower,upper,any}_{precision,recall}`.

For N-k line screening, the key numbers are `upper_fn` (missed overloads) and `upper_fp` (false alarms). Lowering `alpha` moves cases from `fn` to `tp` at the cost of more `fp`.

### `trust_screen_entity_*.csv`

The same counts and rates per bus or line, to find the elements where the screen misses or over-flags.

### `trust_screen_detail_*.parquet/`

A partitioned dataset with one row per scenario and bus or line: the bounds, the limits, the true value and every flag (`pred_*_violation`, `true_*_violation`, and the 0/1 indicators `*_tp`, `*_fp`, `*_fn`, `*_tn`). Use it to inspect individual cases, e.g. the missed overloads: `upper_fn == 1`.

### Figures: `screen_figures/`

- `<var>_precision_<entity>_summary.png`, `<var>_recall_<entity>_summary.png`: precision and recall per bus or line;
- `<var>_any_precision_density.png`, `<var>_any_recall_density.png`: their distributions across entities;
- `<var>_{lower,upper,any}_confusion.png`: confusion matrices.

## Checklist

1. **Coverage per stratum** in `trust_reliability_*.csv`: gaps near zero or positive. If one stratum falls short, check that calibration and test data were generated the same way, and that the stratum has enough calibration scenarios.
2. **Sparse strata:** `n_scenarios_min` small, or `inf` widths in the CSVs. Merge strata or add calibration data.
3. **Spread across entities:** the box plots in `figures/` and `empirical_coverage_mean_across_entities`. A few badly covered buses or lines can hide behind a good average.
4. **Screening:** `upper_fn` and `upper_fp` in `trust_screen_summary_line.csv`, then `trust_screen_entity_line.csv` for where they occur.
5. **Kernel runs:** the share of `kcp_fallback` in the calibrated parquet, and the comparison with an `element` run.
