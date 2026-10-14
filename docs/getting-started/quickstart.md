# Quick start

This page takes you from a trained GridFM model to calibrated intervals and screening results for an N-k contingency study. It uses the command-line interface; the [Python API](../user-guide/python-api.md) offers the same methods on plain arrays.

You need:

- `gridfm-trustlayer` installed with the `graphkit` extra, since this example calibrates line loading ([Installation](installation.md));
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
gridfm-datakit generate datakit_calibration.yaml
gridfm-datakit generate datakit_test.yaml
```

Each run writes its ground truth to `<data_dir>/<grid>/raw/`, including `bus_data.parquet` and `branch_data.parquet`.

!!! warning "Why not split one dataset?"
    Datakit usually solves several topologies for each load profile. A random split of one dataset puts near-identical scenarios on both sides, which makes the measured coverage look better than it will be on new data. Trustlayer therefore never splits a dataset itself, and it refuses to evaluate on the data it was fitted on. See [Preparing the data](../user-guide/data.md).

How many scenarios? Coverage is checked per bus or line and per group of scenarios, so each group needs enough of them. A few thousand calibration scenarios is a reasonable start. A group with fewer than $2/\alpha - 1$ scenarios (19 at $\alpha = 0.1$) gets infinite bounds instead of misleadingly narrow ones.

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

## 4. Run the three steps

```bash
gridfm_trustlayer compute   --config trust_config.yaml
gridfm_trustlayer calibrate --config trust_config.yaml
gridfm_trustlayer screen    --config trust_config.yaml
```

**`compute`** reads the calibration set and writes `trustlayer_outputs/case118.trustlayer`. This small file is all you need to apply the layer later.

**`calibrate`** applies it to the test set. It writes one parquet per target family with the bounds appended (`Vm_pi_lower`, `Vm_pi_upper`, ..., `line_loading_pi_upper`), plus coverage reports and figures:

```text
trustlayer_outputs/calibrated/
├── case118_ieee_predictions_vm_va_with_intervals.parquet/       partitioned, one row per scenario and bus
├── case118_ieee_predictions_line_loading_with_intervals.parquet/  one row per scenario and branch
├── trust_reliability_volt.csv   trust_reliability_line.csv    <- start here
├── trust_summary_volt.csv       trust_summary_line.csv
├── trust_analysis_volt.csv      trust_analysis_line.csv
└── figures/                     coverage and width plots, overall and by stratum
```

**`screen`** compares the test intervals with `screen_bounds` and counts true and false alarms against the true AC solution:

```text
trustlayer_outputs/screen/
├── trust_screen_summary_volt.csv   trust_screen_summary_line.csv   <- start here
├── trust_screen_entity_volt.csv    trust_screen_entity_line.csv
├── trust_screen_detail_volt.parquet/  trust_screen_detail_line.parquet/   per row, partitioned
└── screen_figures/                 precision, recall and confusion plots
```

!!! tip "Large test sets"
    `calibrate` and `screen` read, process and write the test set one block of scenarios at a time, so their memory use is set mainly by the block size, not by the number of scenarios. The per-row outputs are partitioned parquet datasets, like datakit's: directories with one `scenario_partition=k/` folder per 200 scenarios, which `pd.read_parquet(path)` reads as one table. See [Reading the results](../user-guide/results.md).

## 5. Check the results

Open `trust_reliability_line.csv`. Each row is one variable in one stratum:

- `target_coverage` is $1-\alpha$, here 0.9;
- `empirical_coverage_weighted` is the share of true test values inside their interval;
- `coverage_gap_weighted` is the difference. It should be close to zero or positive in every stratum.

If coverage holds, look at `width_mean_weighted`: narrower intervals at the same coverage mean a sharper screen.

Then open `trust_screen_summary_line.csv` and read the `scope = system` row. With an upper limit, a line is flagged when `line_loading_pi_upper > 1.0`:

- `upper_tp`: real overloads that were flagged;
- `upper_fn`: real overloads that were missed. Keeping these rare is the point of the layer;
- `upper_fp`: flagged lines that were not overloaded, which cost an unnecessary AC power flow;
- `upper_recall` and `upper_precision` summarize the three.

[Reading the results](../user-guide/results.md) explains every column and figure.

## 6. Use it on new scenarios

The coverage you measured on the test set carries over to new scenarios drawn the same way. To apply the artifact to scenarios without a known solution, point `inputs.test.predictions` at their predictions and run `calibrate` again. The bounds are written as before; coverage figures are only meaningful where ground truth exists.

## Next steps

- [Choosing a setup](../user-guide/choosing-a-setup.md): which `cp_type`, `strat` and `alpha` to use.
- [Kernel-weighted calibration](../user-guide/kernel.md): narrower bounds on easy scenarios, using the model's embeddings.
- [Troubleshooting](../user-guide/troubleshooting.md): common error messages and what to do about them.
