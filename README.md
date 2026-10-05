# gridfm-trustlayer

Conformal trustworthiness layer for GridFM foundation models.

**Statistically valid error bars on grid-state predictions, without retraining the model.** Give it a model's predictions on held-out scenarios with a known AC solution, and it returns calibrated bounds on bus voltages and line loadings that contain the true value with a chosen probability, for example 90%. It works on top of any trained model, including contingencies the model never saw in training. In N-k screening, a line is flagged when its upper bound exceeds its rating, which turns a point prediction into a conservative screen with a controlled miss rate.

The layer offers two methods: stratified split-conformal prediction (`cp_type: element`), with a finite-sample coverage guarantee under exchangeability of calibration and test scenarios, and kernel-weighted conformal prediction (`cp_type: kernel`), which adapts the margin to each scenario and whose coverage is verified empirically. See [`docs/methodology.md`](docs/methodology.md) and the paper listed under [Reference](#reference).

> [!NOTE]
> This is a pre-1.0 release. The statistical core is tested, but the Python API, configuration keys and artifact formats may still change between minor versions; check [`CHANGELOG.md`](CHANGELOG.md) before upgrading.

## Scope

- Consume prediction parquet artifacts produced by GridFM model workflows.
- Support direct calibration targets from task outputs, including quantities such as `Vm`, `Va`, and other task-specific continuous channels.
- Support derived calibration targets computed from predictions and topology, including quantities such as line loading.
- Use the established GridFM validation pipeline as the source of truth for line-loading computation.
- Keep the statistical core independent from Lightning, PyG, and GPU-bound dependencies.

## Current Status

The current implementation exposes two conformal method families:

- `cp_type: element`
- `cp_type: kernel`

`element` is the canonical label for the residual-quantile interval workflow: residuals are reduced to a fixed quantile table at `compute` time and looked up at `calibrate` time.

`kernel` is kernel-weighted conformal prediction (KCP). Instead of a fixed table, the artifact stores the calibration residual bank plus a scenario representation `z(x)`, and the quantile is estimated per test scenario by weighting calibration residuals with a Gaussian kernel over embedding distance. This yields tighter, approximately conditional bounds at the cost of a larger artifact and a small amount of work at application time. See `docs/methodology.md` for the math and `docs/configuration.md` for the keys.

The distance over scenario representations is `distance: euclidean` (default) or `distance: cosine`. Cosine is squared Euclidean distance on unit-norm vectors, so it keeps only the *direction* in which a scenario deviates from the mean scenario and discards the size of that deviation.

The kernel bandwidth can be selected two ways:

- `bandwidth: n_eff` (default): `h^2` is solved per test scenario so the kernel weights reach a target effective sample size `n_eff_target`, 600 by default. This is invariant to embedding width and stratum size, which `k_nn` is not. `n_eff_target: auto` (off by default) chooses the target by cross-validation on the calibration set instead, requiring coverage first and then the narrowest intervals (see `docs/configuration.md`).
- `bandwidth: knn`: `h^2` is the squared distance to the `k_nn`-th nearest calibration scenario.

The two families compose with stratification rather than replacing it: with `cp_type: kernel` and `strat: n_outages`, the kernel localizes *within* each outage stratum.

The current implementation also exposes optional residual-stratification modes:

- `strat: n_outages`
- `strat: outage_type`
- `strat: system_load_bin`
- `strat: pred_bins`

When enabled, residual quantiles are fitted and applied per scenario outage count, where `n_outages` includes both inactive branches and inactive generators in the raw datakit scenario.

`outage_type` groups scenarios into four coarse outage regimes: `none`, `branch_only`, `gen_only`, and `mixed`.

`system_load_bin` groups scenarios by percentile bins of total active load, where `system_load_n_bins` controls the requested number of bins learned from the compute/calibration scenarios. Output labels are written as percentile ranges with stored cut-point hints, for example `p00-p20 (<1450)` or `p80-p100 (>=1920)`.

`pred_bins` groups individual predictions by percentile bins of the model's own outputs. Trustlayer learns percentile cut points per calibrated variable from all calibration predictions during `compute`, stores them in the artifact, and reapplies them to each prediction value during `calibrate`. The calibrated outputs expose variable-specific columns such as `pred_bins_vm`, `pred_bins_va`, and `pred_bins_line_loading`.

Today the repository supports three CLI verbs:

- `compute`: fit interval calibrators and save a `.trustlayer` artifact
- `calibrate`: apply saved intervals to prediction artifacts and write calibrated parquet outputs plus analysis CSVs and figures
- `screen`: compare calibrated intervals against operating bounds and write confusion-matrix style summaries and plots

The current implementation can run on:

- direct voltage variables such as `Vm` and `Va`
- derived branch loading via `line_loading`
- mixed runs that request both voltage and loading variables from one YAML config

### Kernel-weighted calibration inputs

`cp_type: kernel` needs one scenario representation per scenario. Trustlayer reads a per-entity embeddings parquet (columns `scenario`, `bus`, and `emb_*`) such as the one graphkit writes with `predict --get_embeddings`, and pools it to one vector per scenario. `embedding_pooling` selects the strategy (`mean` by default, plus `max`, `meanmax`, and `concat`).

## Installation

Requires Python 3.10-3.12. To use the package:

```bash
python -m pip install "git+https://github.com/gridfm/gridfm-trustlayer"
```

For development, clone the repository and install it in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"
```

If you also want the datakit-backed loading path available in the same environment, install the optional extras:

```bash
python -m pip install -e ".[graphkit,dev,test]"
```

## Quick Start

The CLI is config-first: every verb requires `--config`, and individual CLI flags act only as overrides of specific YAML values. To write the config, start from one of the [example configs](#example-configs) or generate it with the config-builder notebook, [`examples/notebooks/config_builder.ipynb`](examples/notebooks/config_builder.ipynb), which shows only the options that apply to your choices and validates the result (see [Building the config interactively](#building-the-config-interactively)).

Minimal workflow:

```bash
gridfm_trustlayer compute --config path/to/trust_config.yaml
gridfm_trustlayer calibrate --config path/to/trust_config.yaml
gridfm_trustlayer screen --config path/to/trust_config.yaml
```

The layer is always fitted on one dataset and evaluated on another, so the config names two:

```yaml
inputs:
  calibration:          # fitted on by `compute`
    predictions: path/to/calibration/predictions.parquet
    pf_node: path/to/calibration/raw/bus_data.parquet
    pf_edge: path/to/calibration/raw/branch_data.parquet
  test:                 # applied to and evaluated on by `calibrate` and `screen`
    predictions: path/to/test/predictions.parquet
    pf_node: path/to/test/raw/bus_data.parquet
    pf_edge: path/to/test/raw/branch_data.parquet
```

Generate them independently, for example two GridFM-datakit runs with different seeds. Trustlayer never splits one dataset internally, and it refuses to evaluate on the data it was fitted on, because coverage measured there says nothing about new scenarios.

Typical run structure:

1. `compute` reads the calibration predictions and ground truth and fits one or more interval calibrators.
2. `calibrate` loads the saved `.trustlayer` artifact, applies it to the test predictions, and reports coverage against the test ground truth.
3. `screen` evaluates the calibrated test intervals against operating bounds such as `Vm in [0.95, 1.05]` or `line_loading <= 1.0`.

### Building the config interactively

The config has many options, and which ones apply depends on earlier choices (kernel options, bin counts, screening bounds, OPF generator files). A Jupyter form builds it for you:

```bash
python -m pip install "gridfm-trustlayer[interactive]"
```

```python
from gridfm_trustlayer.interactive import config_builder

config_builder()
```

It is also available as [`examples/notebooks/config_builder.ipynb`](examples/notebooks/config_builder.ipynb). Options appear only when they apply, each choice has a one-line explanation, a live preview shows the YAML, and the config is checked with the CLI's own validation before it can be saved. The same logic is available without Jupyter as `gridfm_trustlayer.interactive.build_config(...)`.

## Python API

The calibrators can also be used directly from Python, with any model's predictions arranged as (scenario x entity) arrays:

```python
import numpy as np
from gridfm_trustlayer import ChannelSet, NaiveIntervalCalibrator

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
# With real data, use separately generated datasets (see Quick Start).
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

# Conservative screen: flag a line if its upper bound exceeds its rating.
overloaded = upper > 1.0
```

Two conventions worth knowing:

- **`alpha` is two-sided.** Each tail gets `alpha / 2`, so `alpha=0.1` gives 90% intervals whose upper bound alone holds with probability at least 95%. For a one-sided screen at level `1 - a` (for example, an upper bound on line loading), pass `alpha = 2 * a` and use only the upper bound.
- **Too few calibration samples give infinite bounds.** A stratum needs at least `ceil(1 / (alpha / 2)) - 1` valid residuals (19 at `alpha=0.1`) for a finite bound. With fewer, the bound is `+inf` / `-inf` rather than a falsely narrow interval; merge strata or add calibration scenarios if that happens.

`load_predictions`, `load_ground_truth`, `load_scenario_embeddings` and `compute_line_loading` read GridFM parquet artifacts into these arrays (`compute_line_loading` uses datakit's branch-flow routine and needs the `graphkit` extra), and `save_interval_artifact` / `load_interval_artifact` persist a fitted calibrator.

## CLI Summary

Command entrypoint:

```bash
gridfm_trustlayer <command> [OPTIONS]
```

Available commands:

- `compute`: fit one or more interval calibrators and save a `.trustlayer` artifact
- `calibrate`: apply a saved artifact and write calibrated parquet outputs plus analysis CSVs and figures
- `screen`: evaluate calibrated intervals against operating bounds and write screening summaries and figures

For full option tables and command behavior, see:

- `docs/cli.md`

## Example Configs

The repository ships a matrix of runnable example configs, one per option combination:

- `examples/config/` — see [`examples/config/README.md`](examples/config/README.md) for the full table
- `examples/notebooks/config_builder.ipynb` — a Jupyter form that builds and validates a config for your own choices (see [Building the config interactively](#building-the-config-interactively))

| Config | Demonstrates |
| --- | --- |
| `element_global_pf.yaml` | Smallest useful run: one global quantile table |
| `element_n_outages_pf.yaml` | Canonical N-k screening, stratified by contingency severity |
| `element_outage_type_pf.yaml` | Stratified by outage regime instead of count |
| `element_system_load_bin_pf.yaml` | Stratified by percentile bins of total demand |
| `element_pred_bins_pf.yaml` | Stratified by percentile bins of the model's own predictions |
| `kernel_knn_pf.yaml` | Kernel-weighted CP with the k-nearest-neighbour bandwidth |
| `kernel_n_eff_pf.yaml` | Kernel-weighted CP with a dimension-portable bandwidth |
| `kernel_n_eff_tuned_pf.yaml` | Same, with the target chosen by cross-validation on the calibration set |
| `kernel_cosine_pf.yaml` | Kernel-weighted CP with cosine distance |
| `element_n_outages_opf.yaml` | OPF artifacts with near-limit screening bounds |

The YAML carries:

- shared run metadata such as `task_name`, `cp_type`, `strat`, and `alpha`
- input paths for predictions and ground truth
- output paths for artifacts, calibrated parquets, and screening outputs
- `screen_bounds` for bounded variables such as `Vm` and `line_loading`
- optional command-specific overrides under `commands`

For the full YAML reference, example structure, and mixed-run path rules, see:

- `docs/configuration.md`

## Generated Outputs

The current workflow produces:

- `.trustlayer` artifacts from `compute` (JSON for `element`, a compressed binary container for `kernel`)
- calibrated voltage and/or line-loading parquet outputs from `calibrate`
- voltage and line analysis CSVs plus figures from `calibrate`
- voltage and line screening CSVs plus figures from `screen`

## Current Limits

- Only `cp_type: element` and `cp_type: kernel` are implemented today.
- Only `strat: n_outages`, `strat: outage_type`, `strat: system_load_bin`, and `strat: pred_bins` are implemented today when stratification is requested.
- `cp_type: kernel` requires `embeddings` in both datasets, and it does not support `strat: pred_bins` (prediction bins are element-wise while kernel localization is scenario-wise).
- `bandwidth: knn` and `bandwidth: n_eff` are the two implemented bandwidth-selection modes. `n_eff_target` can be chosen automatically by cross-validation on the calibration set (`n_eff_target: auto`); `k_nn` cannot, and choosing it by hand requires a tuning partition disjoint from the evaluation set.
- Mixed bus-level and branch-level workflows are supported, but their outputs remain intentionally split because they live on different entity axes.

## Documentation

- `docs/index.md`: project overview
- `docs/methodology.md`: current method, workflow, implementation details, and screening semantics
- `docs/cli.md`: detailed CLI reference
- `docs/configuration.md`: YAML configuration reference
- `docs/schemas.md`: current schema and loading-path notes
- `examples/notebooks/config_builder.ipynb`: interactive config builder

## Reference

- Paper: https://arxiv.org/pdf/2602.07995
- If you use this software, please cite it as described in [`CITATION.cff`](CITATION.cff).
