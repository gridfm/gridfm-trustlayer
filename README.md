# gridfm-trustlayer

Conformal trustworthiness layer for GridFM foundation models.

This repository implements a post-hoc conformal prediction layer for continuous grid-state predictions and derived physical quantities. It is intended to provide statistically valid uncertainty sets for direct model outputs such as `Vm` and `Va`, and for computed metrics such as `line_loading`.

> [!WARNING]
> This project is under active development.
> Do not rely on it yet for production use, operational decisions, published benchmarks, or stable downstream integrations.
> Interfaces, file formats, and implementation details may change without notice while the package is being built out and validated. ⚠️

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

- `bandwidth: knn` (default): `h^2` is the squared distance to the `k_nn`-th nearest calibration scenario, as specified in the paper.
- `bandwidth: n_eff`: `h^2` is solved per test scenario so the kernel weights reach a target effective sample size `n_eff_target`. This is invariant to embedding width and stratum size, which `k_nn` is not, and its default is derived from `alpha` rather than tuned.

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

Create or activate a Python 3.10-3.12 environment, then install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"
```

If you also want the datakit-backed loading path available in the same environment, install the optional extras:

```bash
python -m pip install -e ".[graphkit,dev,test]"
```

## Quick Start

The CLI is config-first: every verb requires `--config`, and individual CLI flags act only as overrides of specific YAML values.

Minimal workflow:

```bash
gridfm_trustlayer compute --config path/to/trust_config.yaml
gridfm_trustlayer calibrate --config path/to/trust_config.yaml
gridfm_trustlayer screen --config path/to/trust_config.yaml
```

Typical run structure:

1. `compute` reads raw prediction and ground-truth artifacts and fits one or more interval calibrators.
2. `calibrate` loads the saved `.trustlayer` artifact and appends interval columns to bus-level predictions or builds branch-level loading outputs.
3. `screen` evaluates calibrated intervals against operating bounds such as `Vm in [0.95, 1.05]` or `line_loading <= 1.0`.

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

| Config | Demonstrates |
| --- | --- |
| `element_global_pf.yaml` | Smallest useful run: one global quantile table, no split |
| `element_holdout_pf.yaml` | Adds `calibration_fraction` so coverage is measured honestly |
| `element_n_outages_pf.yaml` | Canonical N-k screening, stratified by contingency severity |
| `element_outage_type_pf.yaml` | Stratified by outage regime instead of count |
| `element_system_load_bin_pf.yaml` | Stratified by percentile bins of total demand |
| `element_pred_bins_pf.yaml` | Stratified by percentile bins of the model's own predictions |
| `kernel_knn_pf.yaml` | Kernel-weighted CP with the paper's bandwidth rule |
| `kernel_n_eff_pf.yaml` | Kernel-weighted CP with a dimension-portable bandwidth |
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
- `cp_type: kernel` requires `inputs.embeddings` and a non-null `calibration_fraction`, and it does not support `strat: pred_bins` (prediction bins are element-wise while kernel localization is scenario-wise).
- `bandwidth: knn` and `bandwidth: n_eff` are the two implemented bandwidth-selection modes. Selecting a good `k_nn` or `n_eff_target` for a given dataset still requires a tuning partition disjoint from the evaluation set.
- Mixed bus-level and branch-level workflows are supported, but their outputs remain intentionally split because they live on different entity axes.

## Documentation

- `docs/index.md`: project overview
- `docs/methodology.md`: current method, workflow, implementation details, and screening semantics
- `docs/cli.md`: detailed CLI reference
- `docs/configuration.md`: YAML configuration reference
- `docs/schemas.md`: current schema and loading-path notes

## Reference

- Paper: https://arxiv.org/pdf/2602.07995
