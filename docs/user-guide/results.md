# Reading the results

`calibrate` and `screen` each write data, CSV tables, figures and a `summary.md` report. This page explains what each file contains, how the numbers are computed, and in which order to read them.

With `outputs.calibrated_predictions: trustlayer_outputs/calibrated` and `outputs.screen_dir: trustlayer_outputs/screen`, a run of all three commands leaves:

```text
trustlayer_outputs/
├── case118.trustlayer                       the fitted layer (compute)
├── calibrated/                              calibrate: data
│   ├── case118_ieee_predictions_vm_va_with_intervals.parquet/
│   ├── case118_ieee_predictions_line_loading_with_intervals.parquet/
│   └── coverage/                            calibrate: reports
│       ├── summary.md                       <- start here
│       ├── coverage_by_stratum.csv
│       ├── coverage_by_group.csv
│       ├── coverage_by_element_voltage.csv
│       ├── coverage_by_element_line_loading.csv
│       └── figures/
└── screen/                                  screen: data and reports
    ├── summary.md                           <- start here
    ├── screening_by_stratum.csv
    ├── screening_by_group.csv
    ├── screening_by_element_voltage.csv
    ├── screening_by_element_line_loading.csv
    ├── screening_detail_voltage.parquet/
    ├── screening_detail_line_loading.parquet/
    └── figures/
```

The two `summary.md` files are separate reports: `calibrate` writes `coverage/summary.md`, `screen` writes `screen/summary.md`, and neither touches the other's. Running a command again replaces its own files only. `apply` writes data only, with no report.

!!! tip "In short"
    1. `coverage/summary.md`: is coverage close to $1-\alpha$ in every stratum?
    2. `screen/summary.md`: how many violations were caught, how many missed, and how many false alarms?
    3. The per-element tables and the figures: which buses or lines behave differently?

Calibration never changes the point predictions (`*_pred`). It only adds bounds around them, so its effect is read through coverage, width and screening, not through prediction error.

## Known values are not counted

In a power flow, some values are inputs rather than predictions: the voltage magnitude `Vm` at PV buses and the angle `Va` at the reference bus are set-points. The model receives them, trustlayer replaces the predicted value with the set-point, and both bounds equal it. Counting them would inflate coverage (they are always inside their bounds) and screening (they never surprise), so:

- they are **left out of every coverage, width and screening number**, and of the calibration fit itself;
- they are **counted in the `n_known` columns**, and the reports and terminal lines say how many were set aside, e.g. `261,580 known values not counted`;
- they are **marked** in the calibrated and applied predictions by the boolean columns `Vm_known` and `Va_known`.

A bus that is PV in some scenarios and PQ in others counts only its PQ scenarios. For the `PowerFlow` task this concerns `Vm` at PV buses and `Va` at the reference bus; for `OptimalPowerFlow`, `Va` at the reference bus. `line_loading` has no known values.

## Outputs of `calibrate`

### Calibrated predictions

Two partitioned datasets, one per target family, in `outputs.calibrated_predictions` (or at the paths of `outputs.calibrated_predictions_volt` / `_line`):

| Dataset | One row per | Added columns |
| --- | --- | --- |
| `<predictions>_vm_va_with_intervals.parquet/` (name follows the calibrated variables) | scenario and bus | `Vm_pi_lower`, `Vm_pi_upper`, `Va_pi_lower`, `Va_pi_upper`, `Vm_known`, `Va_known`, next to every column of the predictions file; `Vm_target` / `Va_target` from the ground truth if the predictions file has none |
| `<predictions>_line_loading_with_intervals.parquet/` | scenario and branch | `line_loading_pred`, `line_loading_target`, `line_loading_pi_lower`, `line_loading_pi_upper`, plus `from_bus`, `to_bus`, `rate_a` |

Both also carry the stratum of each row when `strat` is set (`n_outages`, `outage_type`, `system_load_bin`, or `pred_bins_<variable>`, each with a readable `_label` column), and the [kernel diagnostics](kernel.md#checking-that-it-works) `kcp_n_eff`, `kcp_bandwidth_sq` and `kcp_fallback` for `cp_type: kernel`.

Conventions:

- Units are those of the model: `Vm` in p.u., `Va` in radians, `line_loading` as a fraction of the branch rating `rate_a` (1.0 is 100%). Out-of-service or unrated branches are set to `-1`.
- With `strat`, a bus or line with too few calibration samples in its stratum (fewer than $\lceil 2/\alpha \rceil - 1$, or none) uses its quantile over all calibration scenarios; the boolean `<variable>_pooled` column marks those rows ([Sparse strata](choosing-a-setup.md#sparse-strata-sparse_strata)). A bound of `inf` or `-inf` means too few samples even over all strata, or `sparse_strata: infinite`.
- For known values and entries marked invalid in the inputs, no interval is formed: both bounds equal the prediction.

These are **partitioned parquet datasets**, the layout gridfm-datakit uses: a directory with one sub-directory per 200 scenarios, `scenario_partition=0/`, `scenario_partition=1/`, and so on. Read one whole with `pd.read_parquet(path)`, or only some scenarios and columns without loading the rest:

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

### Reports: `coverage/`

The reports go to `outputs.coverage_dir`, by default a `coverage/` folder next to the calibrated datasets.

#### `summary.md`

The page to open first. It states the method, the test set and the artifact, how many known values were set aside, and then, per variable, one table of coverage and width by stratum, followed by the figures and a list of the files. It renders on GitHub and in any Markdown viewer.

Coverage is printed with an approximate 95% interval. It treats the test scenarios of a stratum as independent draws and all values of one scenario as a single draw. That makes it conservative: wider than needed when the buses or lines of a scenario err independently, about right when they err together, as they often do. A coverage below target whose interval still contains the target is within sampling noise.

#### `coverage_by_stratum.csv`

One row per variable and stratum, starting with `all` (every test scenario).

| Column | Meaning |
| --- | --- |
| `variable`, `stratum` | e.g. `Vm`, `n_outages=2` |
| `n_scenarios` | test scenarios in the stratum |
| `n_elements` | buses or lines with at least one counted value |
| `n_values` | values counted (scenario × element), known values excluded |
| `n_known` | known values set aside |
| `n_pooled` | counted values whose bounds use the element's quantile over all strata, because it had too few calibration samples in this one |
| `target_coverage` | $1-\alpha$ |
| `coverage` | share of the counted values whose true value lies inside its bounds |
| `coverage_low`, `coverage_high` | the approximate 95% interval of `coverage` |
| `worst_element_coverage` | lowest coverage of a single bus or line, among those with at least 50 values in the stratum |
| `mean_width` | mean interval width $u - \ell$ over the counted values with finite bounds, in the model's units (radians for `Va`) |
| `n_elements_infinite` | buses or lines whose bounds are infinite in this stratum |

**How to read it.** With `cp_type: element`, `coverage` should be close to `target_coverage` or above in every row. A clear shortfall in one stratum means calibration and test scenarios differ there, for example because they were generated with different settings. With `cp_type: kernel`, coverage is checked here rather than guaranteed, so this table is the verification. Among setups that reach the target, the one with the smaller `mean_width` gives the sharper screen.

#### `coverage_by_group.csv`

The same numbers per group of elements, in every stratum: `group` is `bus_type=PQ`, `bus_type=PV`, `bus_type=REF` for voltages and `rate_a=<rating>` for lines. Columns: `variable`, `stratum`, `group`, `n_elements`, `n_values`, `coverage`, `worst_element_coverage`, `mean_width`. A bus takes the highest type it has in any test scenario (REF, then PV, then PQ), so a bus that is PV in some scenarios is in `bus_type=PV`; its `Vm` values counted there are those of its PQ scenarios.

#### `coverage_by_element_voltage.csv`, `coverage_by_element_line_loading.csv`

One row per bus or line and stratum: `variable`, `stratum`, the element (`bus`, `bus_type`, or `branch`, `from_bus`, `to_bus`, `rate_a`), `n_values`, `n_known` (voltages), `coverage`, `mean_width`, `min_width`, `max_width`. Use them to find the elements with low coverage or very wide bounds. A bus that is PV in every scenario has `n_values = 0` and an empty `coverage` for `Vm`.

#### `figures/`

Three figures per variable (`vm`, `va`, `line_loading`):

| File | Shows |
| --- | --- |
| `<var>_coverage_by_stratum.png` | coverage per stratum with its 95% interval, against the target |
| `<var>_coverage_by_bus.png`, `line_loading_coverage_by_line.png` | histogram of the coverage of every bus or line, over all test scenarios |
| `<var>_width_by_stratum.png` | box plots of the mean width of every element, per stratum (`Va` in degrees) |

Infinite widths are left out of the box plots, and the title says how many. `--no-figures` (or `save_figures: false`) skips the folder.

The terminal shows the headline of each variable when `calibrate` ends:

```text
calibrate: Vm: coverage 0.901 (target 0.900), mean width 0.001109 p.u.; lowest stratum n_outages=4 0.893; 261,580 known values not counted
calibrate: line_loading: coverage 0.903 (target 0.900), mean width 0.01364 fraction of rating; lowest stratum n_outages=4 0.893
calibrate: data in trustlayer_outputs/calibrated; reports in trustlayer_outputs/calibrated/coverage
```

## Outputs of `apply`

`apply` writes only the bounds of the new scenarios: one partitioned dataset per target family, `<predictions>_<variables>_bounds.parquet`, with the columns of the calibrated predictions above except the true values: the `*_target` columns are not written. For new scenarios they would hold real values only at the set-points (PV and reference buses) and placeholders elsewhere. The `Vm_known` and `Va_known` columns mark the set-points. There is no report, since the true values are unknown. To flag possible violations, compare the bounds with your limits directly, e.g. `line_loading_pi_upper > 1.0`.

## Outputs of `screen`

`screen` compares the test bounds with `screen_bounds` and checks each alarm against the true AC value. Everything goes to `outputs.screen_dir`.

### How an alarm is defined

For each scenario and bus or line:

- **alarm:** `pi_upper > upper_bound`, or, for variables with a lower limit (`Vm`), `pi_lower < lower_bound`;
- **true violation:** the same test on the true value.

This is a *possible-violation* rule: it asks whether the bounds allow a violation, not whether the prediction itself violates. It is conservative by design. A real violation of the upper limit can only be missed if the true value lies above the upper bound, which happens for at most a share $\alpha/2$ of values when coverage holds; the same applies to the lower side. `line_loading` has only an upper limit. Each value then falls in one cell:

| | Alarm | No alarm |
| --- | --- | --- |
| **True violation** | caught | missed |
| **No violation** | false alarm (an unnecessary AC power flow) | cleared |

with `recall = caught / violations` and `precision = caught / alarms`. Counts are reported for the `lower` side, the `upper` side and `any` (either side); the reports and figures use `any` for `Vm` and `upper` for `line_loading`.

### `summary.md`

The page to open first: the limits, how many known values were set aside, and per variable a table of violations, caught, missed, false alarms, recall and precision by stratum, followed by the figures and the files.

### `screening_by_stratum.csv`

One row per variable, stratum and side.

| Column | Meaning |
| --- | --- |
| `variable`, `stratum`, `side` | e.g. `Vm`, `n_outages=2`, `any` |
| `n_values`, `n_known` | values screened, and known values set aside |
| `violations` | true violations: `caught + missed` |
| `alarms` | alarms raised: `caught + false_alarms` |
| `caught`, `missed`, `false_alarms`, `cleared` | the four cells above; they add up to `n_values` |
| `recall`, `precision` | `caught / violations`, `caught / alarms`; empty when the denominator is zero |

For N-k line screening, the key numbers are `missed` (overloads the screen let through) and `false_alarms` (AC power flows run for nothing). Lowering `alpha` turns misses into catches at the cost of more false alarms.

### `screening_by_group.csv`, `screening_by_element_voltage.csv`, `screening_by_element_line_loading.csv`

The same counts per group (`all`, `bus_type=...`, `rate_a=...`, with `n_elements`) and per bus or line (with `bus`, `bus_type`, or `branch`, `from_bus`, `to_bus`, `rate_a`), to find where the screen misses or over-flags.

### `screening_detail_voltage.parquet/`, `screening_detail_line_loading.parquet/`

Partitioned datasets with one row per screened value (known values excluded):

| Column | Content |
| --- | --- |
| `scenario`, `bus` / `branch` | the row; plus `bus_type`, or `from_bus`, `to_bus`, `rate_a` |
| `variable` | `Vm` or `line_loading` |
| `target`, `pi_lower`, `pi_upper` | the true value and the bounds |
| `pred_lower_violation`, `pred_upper_violation` | the bound crosses the lower (upper) limit: an alarm |
| `true_lower_violation`, `true_upper_violation` | the true value crosses it |

The stratum columns of the calibrated predictions are carried over when `strat` is set. `line_loading` rows only carry upper-side flags. Use the detail to inspect individual cases, for example the missed overloads:

```python title="Missed overloads"
import pandas as pd

detail = pd.read_parquet(
    "trustlayer_outputs/screen/screening_detail_line_loading.parquet"
)
missed = detail[detail["true_upper_violation"] & ~detail["pred_upper_violation"]]
```

The counts of the tables follow from the flags: `caught` counts rows with both flags of a side set, `missed` rows with only the true one, and so on; `any` combines the two sides.

### `figures/`

- `<var>_confusion.png`: the four cells, laid out as in the table above, with each count's share of its row;
- `<var>_screening_by_stratum.png` (with `strat`): recall and precision per stratum.

`--no-figures` (or `save_figures: false`) skips the folder.

The terminal shows the headline of each variable:

```text
screen: Vm (either limit): 36,850 of 36,954 true violations flagged (104 missed), 1,818 false alarms in 328,420 values; 261,580 known values not counted
screen: line_loading (upper limit): 15,512 of 15,724 true violations flagged (212 missed), 6,255 false alarms in 918,842 values
screen: reports in trustlayer_outputs/screen
```

## Checklist

1. **Coverage per stratum** in `coverage/summary.md`: close to the target or above. If one stratum falls short beyond its 95% interval, check that calibration and test data were generated the same way, and that the stratum has enough calibration scenarios.
2. **Sparse strata:** `n_pooled` or `n_elements_infinite` above zero. The coverage of those strata is not guaranteed; merge strata or add calibration data ([Sparse strata](choosing-a-setup.md#sparse-strata-sparse_strata)).
3. **Spread across elements:** `worst_element_coverage` and the `coverage_by_bus` / `coverage_by_line` histograms. A few badly covered buses or lines can hide behind a good average.
4. **Screening:** `missed` and `false_alarms` in `screen/summary.md`, then `screening_by_element_*.csv` for where they occur.
5. **Kernel runs:** the share of `kcp_fallback` in the calibrated predictions, and the comparison with an `element` run.
