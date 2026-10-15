# Quick start

This page takes you from a trained GridFM model to calibrated intervals and screening results for an N-k contingency study. It uses the command-line interface; the [Python API](../user-guide/python-api.md) offers the same methods on plain arrays. The [demo notebook](demo.md) runs the same steps on ready-made IEEE 14-bus data, if you want to see them work first.

You need:

- `gridfm-trustlayer` installed with the `datakit` extra, since this example calibrates line loading ([Installation](installation.md));
- [gridfm-datakit](https://github.com/gridfm/gridfm-datakit) to generate scenarios;
- [gridfm-graphkit](https://github.com/gridfm/gridfm-graphkit) and a trained power-flow model to predict them.

## 1. Generate a calibration set and a test set

Trustlayer always needs **two separately generated datasets** with known AC solutions:

- the **calibration set**, from which `compute` learns how large the model's errors are;
- the **test set**, on which `calibrate` checks that the intervals reach the coverage you asked for.

Generate them as two datakit runs with the same settings but different output directories. Leave `settings.seed` at `null`, its default, so that each run draws its own random seed; if you set seeds, use a different one for each run.

```yaml title="datakit_calibration.yaml (only the lines that differ between the two runs)"
settings:
  data_dir: "./data_calibration"   # "./data_test" in datakit_test.yaml
  seed: null
```

```bash
gridfm_datakit generate datakit_calibration.yaml
gridfm_datakit generate datakit_test.yaml
```

Each run writes its ground truth to `<data_dir>/<grid>/raw/`, including `bus_data.parquet` and `branch_data.parquet`.

!!! warning "Why not split one dataset?"
    Datakit usually solves several topologies for each load profile. A random split of one dataset puts near-identical scenarios on both sides, which makes the measured coverage look better than it will be on new data. Trustlayer therefore never splits a dataset itself, and it refuses to evaluate on the data it was fitted on. See [Preparing the data](../user-guide/data.md).

How many scenarios? Coverage is checked per bus or line and per group of scenarios, so each group needs enough of them. A few thousand calibration scenarios is a reasonable start. A bus or line with fewer than $\lceil 2/\alpha \rceil - 1$ samples in a group (19 at $\alpha = 0.1$) gets bounds learned from all groups together there, and the reports say where ([Sparse strata](../user-guide/choosing-a-setup.md#sparse-strata-sparse_strata)).

## 2. Predict both sets with your model

Run graphkit's `predict` once per dataset:

```bash
gridfm_graphkit predict --config model_config.yaml --model_path model.pt \
    --data_path data_calibration --output_path predictions/calibration
gridfm_graphkit predict --config model_config.yaml --model_path model.pt \
    --data_path data_test --output_path predictions/test
```

Each writes `<grid>_predictions.parquet`, with one row per scenario and bus. Add `--get_embeddings` if you plan to use kernel-weighted calibration (`cp_type: kernel`); it also writes `<grid>_bus_embeddings.parquet`. The [graphkit documentation](https://github.com/gridfm/gridfm-graphkit) lists the remaining options, such as `--normalizer_stats`.

## 3. Write the config

Everything trustlayer does is described in one YAML file. Save this as `trust_config.yaml` and replace the paths and the grid name (`case118_ieee` here):

```yaml title="trust_config.yaml"
version: 1

task_name: PowerFlow
cp_type: element          # stratified split-conformal prediction
strat: n_outages          # one set of bounds per number of outages (N-1, N-2, ...)
alpha: 0.1                # 90% intervals

variables: [Vm, Va, line_loading]

screen_bounds:
  Vm: {lower: 0.95, upper: 1.05}   # p.u.
  line_loading: {upper: 1.0}       # fraction of the rating rate_a

inputs:
  calibration:            # read by `compute`
    predictions: predictions/calibration/case118_ieee_predictions.parquet
    pf_node: data_calibration/case118_ieee/raw/bus_data.parquet
    pf_edge: data_calibration/case118_ieee/raw/branch_data.parquet
  test:                   # read by `calibrate` and `screen`
    predictions: predictions/test/case118_ieee_predictions.parquet
    pf_node: data_test/case118_ieee/raw/bus_data.parquet
    pf_edge: data_test/case118_ieee/raw/branch_data.parquet

outputs:
  artifact: trustlayer_outputs/case118.trustlayer
  calibrated_predictions: trustlayer_outputs/calibrated
  screen_dir: trustlayer_outputs/screen

commands:
  screen:
    variables: [Vm, line_loading]  # Va has no operating limit to screen against
```

What the main keys mean:

- **`alpha`** is the share of true values allowed to fall outside the interval, split equally between the two tails. `alpha: 0.1` gives 90% intervals, and each one-sided bound holds with probability 95%.
- **`strat: n_outages`** fits separate bounds for each contingency level, so an N-3 scenario is not given the narrow bounds learned mostly from N-1 scenarios.
- **`variables`** lists what to calibrate. `Vm` and `Va` are read from the predictions; `line_loading` is computed from the predicted voltages and the branch data.

Rather than editing YAML by hand, you can also generate this file with the [config builder](config-builder.md), or start from one of the [example configs](https://github.com/gridfm/gridfm-trustlayer/tree/main/examples/config).

## 4. Fit the layer: `compute`

```bash
gridfm_trustlayer compute --config trust_config.yaml
```

`compute` reads the calibration set and writes `trustlayer_outputs/case118.trustlayer`. This small file is the layer: everything after it, on test or new scenarios, uses it. You always run `compute` first, then either or both of the next two steps.

| Step | Commands | Needs the AC solution | Gives |
| --- | --- | --- | --- |
| Validate | `calibrate`, `screen` | yes (test set) | how well the layer works: coverage, caught and missed violations |
| Operate | `apply` | **no** | bounds and statuses for new scenarios |

## 5. Validate on the test set: `calibrate` and `screen`

```bash
gridfm_trustlayer calibrate --config trust_config.yaml
gridfm_trustlayer screen    --config trust_config.yaml
```

**`calibrate`** applies the layer to the test set. It writes one partitioned parquet per target family with the bounds appended (`Vm_pi_lower`, `Vm_pi_upper`, ..., `line_loading_pi_upper`), and a coverage report:

```text
trustlayer_outputs/calibrated/
├── case118_ieee_predictions_vm_va_with_intervals.parquet/         one row per scenario and bus
├── case118_ieee_predictions_line_loading_with_intervals.parquet/  one row per scenario and branch
└── coverage/
    ├── summary.md                  <- start here
    ├── coverage_by_stratum.csv     coverage_by_group.csv
    ├── coverage_by_element_voltage.csv   coverage_by_element_line_loading.csv
    └── figures/                    coverage and width by stratum, coverage per bus and line
```

**`screen`** compares the test intervals with `screen_bounds`, gives every value and every operating scenario a status, and checks both against the true AC solution:

```text
trustlayer_outputs/screen/
├── summary.md                      <- start here
├── scenarios.csv                   status of every operating scenario
├── status_by_stratum.csv           scenarios and values per status
├── screening_by_stratum.csv        screening_by_group.csv
├── screening_by_element_voltage.csv   screening_by_element_line_loading.csv
├── screening_detail_voltage.parquet/  screening_detail_line_loading.parquet/   one row per value
└── figures/                        confusion matrix, recall and precision by stratum
```

Each command also prints its headline numbers when it ends.

!!! tip "Large test sets"
    `calibrate`, `screen` and `apply` read, process and write the scenarios one block at a time, so their memory use is set mainly by the block size, not by the number of scenarios. The per-row outputs are partitioned parquet datasets, like datakit's: directories with one `scenario_partition=k/` folder per 200 scenarios, which `pd.read_parquet(path)` reads as one table. See [Reading the results](../user-guide/results.md).

### Check the results

Open `calibrated/coverage/summary.md`. For each variable, a table gives one row per stratum:

- **coverage** is the share of true test values inside their bounds, with an approximate 95% interval. It should be close to the target, $1-\alpha$ = 0.9, or above, in every stratum;
- **worst element** is the lowest coverage of a single bus or line;
- **mean width** is the average size of the bounds: narrower bounds at the same coverage mean a sharper screen.

Then open `screen/summary.md`. It reports the screen in two ways.

**Per operating scenario and per value**, with three statuses:

| Status of a value | of a scenario | When |
| --- | --- | --- |
| `within_limits` | `secure` | the whole interval is inside the limits (for a scenario: every value is) |
| `uncertain` | `uncertain` | the interval straddles a limit (for a scenario: no violation, but at least one uncertain value) |
| `violation` | `insecure` | the whole interval is beyond a limit (for a scenario: at least one value is) |

Each count comes with how many were wrong according to the true solution: secure scenarios that were actually insecure, insecure ones that were actually secure. The same statuses computed from the **point prediction** alone are shown next to them, so you can see what the intervals add.

**As alarms per value**, the classic screening view: a value raises an alarm when its bound crosses the limit, e.g. `line_loading_pi_upper > 1.0`. Alarms are counted as **caught** (real violations flagged), **missed** (real violations not flagged; keeping these rare is the point of the layer) and **false alarms**, with recall and precision.

!!! note "Known values are not counted"
    In a power flow, `Vm` at PV buses and `Va` at the reference bus are set-points: inputs of the model, not predictions. They are left out of every number above, and the reports say how many were set aside. See [Known values](../user-guide/results.md#known-values-are-not-counted).

[Reading the results](../user-guide/results.md) explains every column and figure.

## 6. Operate on new scenarios: `apply`

The coverage you measured on the test set carries over to new scenarios drawn the same way. For scenarios **without a known AC solution**, add their files under `inputs.apply` and run `apply`:

```yaml title="trust_config.yaml (addition)"
inputs:
  apply:                  # read by `apply`; no AC solution needed
    predictions: predictions/new/case118_ieee_predictions.parquet
    pf_node: data_new/case118_ieee/raw/bus_data.parquet
    pf_edge: data_new/case118_ieee/raw/branch_data.parquet
```

```bash
gridfm_trustlayer apply --config trust_config.yaml
```

`apply` computes the same bounds as `calibrate`, and, when `screen_bounds` is set, the same statuses as `screen`, with no truth to check them against. With `outputs.applied_predictions: trustlayer_outputs/applied`:

```text
trustlayer_outputs/applied/
├── case118_ieee_predictions_vm_va_bounds.parquet/         bounds, Vm_status, Vm_point_status
├── case118_ieee_predictions_line_loading_bounds.parquet/  bounds, line_loading_status, ...
└── status/
    ├── summary.md                  <- start here: secure, uncertain, insecure scenarios
    ├── scenarios.csv               status of every operating scenario
    └── status_by_stratum.csv       scenarios and values per status
```

The report goes to `outputs.status_dir`, by default `status/` next to the data; without `outputs.applied_predictions`, the data go to `trustlayer_outputs/<task>/<cp_type>/`. Since the rule is the one `screen` uses, the error rates `screen` measured on the test set are the ones to expect here. Without `screen_bounds`, `apply` writes the bounds only.

`apply` reads only what the layer needs:

- the predictions, which for power flow must carry the **set-points** of each scenario: `Vm_target` at PV buses and `Va_target` at the reference bus, as graphkit writes them. They are inputs of a power flow, not results; other buses may have no target;
- the bus and branch files, for the outages (`strat: n_outages`, `outage_type`), the total load (`strat: system_load_bin`) and the branch parameters (`line_loading`); only the topology, loads and parameters are read, never the solved voltages;
- the embeddings, for `cp_type: kernel`.

A voltage layer without stratification needs only the predictions.

## Next steps

- [Choosing a setup](../user-guide/choosing-a-setup.md): which `cp_type`, `strat` and `alpha` to use.
- [Kernel-weighted calibration](../user-guide/kernel.md): narrower bounds on easy scenarios, using the model's embeddings.
- [Troubleshooting](../user-guide/troubleshooting.md): common error messages and what to do about them.
