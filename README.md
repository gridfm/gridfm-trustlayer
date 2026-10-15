<p align="center">
    <img src="https://github.com/gridfm/gridfm-trustlayer/raw/main/docs/figs/gridfm-trustlayer-logo-1024.png" alt="gridfm-trustlayer logo" style="width: 40%; height: auto;"/>
  <br/>
</p>

<h1 align="center">gridfm-trustlayer</h1>


[![CI Build](https://github.com/gridfm/gridfm-trustlayer/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/gridfm/gridfm-trustlayer/actions/workflows/ci-build.yaml)
[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](https://gridfm.github.io/gridfm-trustlayer/)
![Python](<https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue>)
![License](<https://img.shields.io/badge/license-Apache%202.0-blue>)

**Statistically valid error bars on grid-state predictions, without retraining the model.**

`gridfm-trustlayer` wraps the point predictions of a power-system foundation model, such as a GridFM power-flow model, in intervals that contain the true AC value with a probability you choose, for example 90%. It learns how large the model's errors are on scenarios with a known solution, and turns them into bounds on bus voltages and line loadings for new scenarios, including contingencies the model never saw in training.

In N-k contingency screening, a line is flagged when the upper bound of its predicted loading exceeds its rating. Only flagged contingencies need a full AC power flow, and the share of real overloads that slip through is controlled by the coverage level.

📖 **Documentation: [https://gridfm.github.io/gridfm-trustlayer/](https://gridfm.github.io/gridfm-trustlayer/)**

## Features

- **Two conformal methods.** Stratified split-conformal prediction (`cp_type: element`), with a finite-sample coverage guarantee when calibration and test scenarios are exchangeable. Kernel-weighted conformal prediction (`cp_type: kernel`), which adapts the margin to each scenario using the model's embeddings and whose coverage is verified empirically.
- **Stratification** by number of outages, outage type, system load or predicted value, so bounds widen where the model is less accurate.
- **Voltages and line loading.** `Vm` and `Va` are calibrated directly; `line_loading` is derived from the predicted voltages with datakit's branch-flow routine.
- **Screening** against operating limits: every value is *within limits*, *uncertain* or a *violation*, and every operating scenario *secure*, *uncertain* or *insecure*, compared with the point prediction alone. On test scenarios, checked against the true AC solution; on new scenarios, with no solution needed.
- **Works with any model** whose predictions follow the GridFM parquet layout, or with plain NumPy arrays through the Python API. CPU only.

## Installation

Requires Python 3.10–3.12.

```bash
python -m pip install gridfm-trustlayer
```

Optional extras: `datakit` (needed for `line_loading`) and `interactive` (the Jupyter config builder):

```bash
python -m pip install "gridfm-trustlayer[datakit,interactive]"
```

The development version: `python -m pip install "git+https://github.com/gridfm/gridfm-trustlayer"`.

For development: `python -m pip install -e ".[dev,test]"` in a clone, then `pre-commit install`.

## Quick start

**To see it work first**, open the [demo notebook](https://github.com/gridfm/gridfm-trustlayer/blob/main/examples/demo/demo.ipynb): the whole workflow on IEEE 14-bus contingencies with a released GridFM model, data included, in under a minute.

You need two **separately generated** datasets with known AC solutions, for example two [gridfm-datakit](https://github.com/gridfm/gridfm-datakit) runs with different seeds, each predicted with your model through [gridfm-graphkit](https://github.com/gridfm/gridfm-graphkit). Trustlayer is fitted on the first (calibration) and evaluated on the second (test); it never splits one dataset itself.

Describe the run in a YAML config:

```yaml
task_name: PowerFlow
cp_type: element
strat: n_outages          # separate bounds per contingency level
alpha: 0.1                # 90% intervals
variables: [Vm, Va, line_loading]
screen_bounds:
  Vm: {lower: 0.95, upper: 1.05}
  line_loading: {upper: 1.0}

inputs:
  calibration:
    predictions: predictions/calibration/case118_ieee_predictions.parquet
    pf_node: data_calibration/case118_ieee/raw/bus_data.parquet
    pf_edge: data_calibration/case118_ieee/raw/branch_data.parquet
  test:
    predictions: predictions/test/case118_ieee_predictions.parquet
    pf_node: data_test/case118_ieee/raw/bus_data.parquet
    pf_edge: data_test/case118_ieee/raw/branch_data.parquet

outputs:
  artifact: trustlayer_outputs/case118.trustlayer
  calibrated_predictions: trustlayer_outputs/calibrated
  screen_dir: trustlayer_outputs/screen

commands:
  screen:
    variables: [Vm, line_loading]
```

and run:

```bash
gridfm_trustlayer compute   --config trust_config.yaml   # fit: always, on the calibration set

gridfm_trustlayer calibrate --config trust_config.yaml   # validate: bounds + coverage on the test set
gridfm_trustlayer screen    --config trust_config.yaml   # validate: caught / missed violations

gridfm_trustlayer apply     --config trust_config.yaml   # operate: new scenarios, no AC solution needed
```

`calibrate` and `screen` measure how well the layer works on scenarios with a known solution. `apply` is the operational step: for new scenarios (`inputs.apply` in the config) it gives the bounds and, with `screen_bounds`, the status of every value (within limits, uncertain, violation) and of every operating scenario (secure, uncertain, insecure), next to what the point prediction alone would say.

To write the config, use the Jupyter **config builder**, which shows only the options that apply to your choices and validates the result ([`examples/notebooks/config_builder.ipynb`](examples/notebooks/config_builder.ipynb)), or start from one of the ten [example configs](examples/config/). The [Quick start guide](https://gridfm.github.io/gridfm-trustlayer/getting-started/quickstart/) walks through every step, including data generation and how to read the results.

## Python API

The calibrators also work directly on NumPy arrays, with any model's predictions arranged as (scenario × entity) arrays:

```python
import numpy as np
from gridfm_trustlayer import (
    ChannelSet,
    NaiveIntervalCalibrator,
    classify_status,
    scenario_status,
)

# Synthetic stand-in for model outputs: 2,000 contingency scenarios x 50 lines,
# with prediction errors that grow with the contingency level k = 1..5.
rng = np.random.default_rng(0)
n_scenarios, n_lines = 2000, 50
k = rng.integers(1, 6, n_scenarios)
predicted = rng.uniform(0.2, 1.2, (n_scenarios, n_lines))
true = predicted + rng.normal(0.0, 0.01 * k[:, None], predicted.shape)


def channel_set(values, rows):
    return ChannelSet(
        scenario_ids=np.flatnonzero(rows),
        entity_ids=np.arange(n_lines),
        entity_name="branch",
        channels={"line_loading": values[rows]},
        scenario_columns={"n_outages": k[rows]},
    )


# Independent synthetic draws, so halving them is a valid calibration/test split.
# With real data, use separately generated datasets (see Quick start).
calibration = np.arange(n_scenarios) < 1000
calibrator = NaiveIntervalCalibrator.fit(
    channel_set(predicted, calibration),
    channel_set(true, calibration),
    alpha=0.1,  # two-sided: 90% intervals, 5% per tail
    channel_names=("line_loading",),
    strat="n_outages",  # one quantile per (contingency level, line)
)
test_predictions = channel_set(predicted, ~calibration)
lower, upper = calibrator.predict_intervals(test_predictions)["line_loading"]

test_true = true[~calibration]
coverage = ((lower <= test_true) & (test_true <= upper)).mean()
print(f"coverage: {coverage:.3f}")  # about 0.90

# Status against the rating (1.0 = 100%): per line and test scenario, then per
# scenario (the worst of its lines). Same rule as `screen` and `apply`.
line_status = classify_status(lower, upper, upper_limit=1.0)
scenario = scenario_status(line_status, axis=1)  # secure, uncertain or insecure
```

See the [Python API guide](https://gridfm.github.io/gridfm-trustlayer/user-guide/python-api/) for the kernel method and for loading GridFM files.

## Documentation

- [Getting started](https://gridfm.github.io/gridfm-trustlayer/getting-started/installation/): installation, demo notebook, quick start, config builder
- [User guide](https://gridfm.github.io/gridfm-trustlayer/user-guide/data/): preparing the data, choosing a setup, kernel-weighted calibration, reading the results, troubleshooting
- [Reference](https://gridfm.github.io/gridfm-trustlayer/reference/cli/): CLI, configuration keys, data schemas, Python API
- [Methodology](https://gridfm.github.io/gridfm-trustlayer/methodology/): the statistics behind both methods

The sources are in [`docs/`](https://github.com/gridfm/gridfm-trustlayer/tree/main/docs); `mkdocs serve` builds the site locally.

> [!NOTE]
> This is a pre-1.0 release. The statistical core is tested, but the Python API, configuration keys and artifact formats may still change between minor versions; check [`CHANGELOG.md`](https://github.com/gridfm/gridfm-trustlayer/blob/main/CHANGELOG.md) before upgrading.

## About

`gridfm-trustlayer` was initially developed by [Antonio Alcántara](https://antonioalcantara.com) for the [OpenGridFM](https://gridfm.org/) community, as a common trustworthiness layer for its foundation models to build on.

Contributions are welcome (and will be acknowledged in the repo), from bug reports to new methods: see [CONTRIBUTING.md](https://github.com/gridfm/gridfm-trustlayer/blob/main/CONTRIBUTING.md) for how to get started, or open an issue to discuss an idea.

## Citation

If you use `gridfm-trustlayer` in your research, please cite:

```bibtex
@article{alcantara2026trustworthiness,
  title={Trustworthiness Layer for Foundation Models in Power Systems: Application to N-k Contingency Screening},
  author={Antonio Alcántara and Spyros Chatzivasileiadis},
  journal={arXiv preprint arXiv:2602.07995},
  year={2026},
  url={https://arxiv.org/abs/2602.07995}
}
```

GitHub's "Cite this repository" button uses [`CITATION.cff`](https://github.com/gridfm/gridfm-trustlayer/blob/main/CITATION.cff).

## License

Apache License 2.0; see [`LICENSE`](https://github.com/gridfm/gridfm-trustlayer/blob/main/LICENSE).
