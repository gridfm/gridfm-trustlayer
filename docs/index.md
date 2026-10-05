# gridfm-trustlayer

`gridfm-trustlayer` is a conformal trustworthiness layer for GridFM foundation models.

The package provides:

- parquet-based data ingestion for predictions and ground truth
- calibration support for direct prediction targets such as `Vm`, `Va`, and other task-specific continuous outputs
- calibration support for derived physical metrics such as line loading
- stratified conformal prediction (`cp_type: element`) and kernel-weighted conformal prediction (`cp_type: kernel`)
- a config-first CLI for calibration and screening workflows, a Jupyter form that builds the config, and a Python API

## Current workflow

The layer is fitted on one dataset and evaluated on another, separately generated one (`inputs.calibration` and `inputs.test` in the config; see [configuration.md](configuration.md#inputs)):

1. `compute` fits interval calibrators on the calibration dataset and writes a `.trustlayer` artifact
2. `calibrate` applies them to the test dataset and writes calibrated parquet outputs plus coverage analyses
3. `screen` compares the test intervals against operating limits and writes aggregate summaries

All verbs require `--config`, and CLI flags act as overrides of the YAML.

## Getting started

- **Build a config interactively:** `examples/notebooks/config_builder.ipynb` (install with `pip install "gridfm-trustlayer[interactive]"`) shows only the options that apply to your choices and validates the result.
- **Start from an example:** `examples/config/` has one runnable config per option combination.
- **Use it from Python:** `from gridfm_trustlayer import NaiveIntervalCalibrator, KernelIntervalCalibrator, ChannelSet`; a worked example is in the repository README, under "Python API".

## Current supported calibration types

- `element`: fixed residual-quantile table, optionally stratified, with a finite-sample coverage guarantee under exchangeability of calibration and test scenarios
- `kernel`: kernel-weighted quantile localized per test scenario in embedding space

## Current supported targets

- direct voltage variables such as `Vm` and `Va`
- derived branch loading via `line_loading`
- mixed runs that combine voltage and line-loading targets in one config

## Documentation map

- [methodology.md](methodology.md): current trustlayer method, workflow, implementation details, and screening interpretation
- [analysis_and_figures.md](analysis_and_figures.md): exact definitions for analysis metrics, reliability metrics, figure computations, and stratification aggregation behavior
- [cli.md](cli.md): CLI verbs, options, and workflow behavior
- [configuration.md](configuration.md): YAML config structure, merge rules, and mixed-run path rules
- [schemas.md](schemas.md): current prediction, ground-truth, and produced-artifact schema reference
- `examples/config/`: runnable configs, one per option combination, with a README mapping each file to what it shows

## Current implementation notes

- `cp_type: element` is the canonical name for the residual-quantile interval workflow
- `cp_type: kernel` is kernel-weighted conformal prediction, which localizes the quantile to each test scenario using a scenario embedding instead of applying one quantile table per stratum
- `bandwidth: n_eff` (default: solve for a target effective sample size, 600 unless set or chosen by cross-validation with `n_eff_target: auto`) and `bandwidth: knn` (distance to the `k_nn`-th nearest neighbour) are the two kernel bandwidth-selection modes
- `distance: euclidean` (default) and `distance: cosine` are the two kernel distance metrics over scenario representations
- the two `cp_type` values compose with stratification: with `cp_type: kernel` and `strat: n_outages`, kernel weights are computed only among calibration scenarios in the same outage stratum
- `strat: n_outages`, `strat: outage_type`, `strat: system_load_bin`, and `strat: pred_bins` are the current supported residual-stratification modes
- `system_load_bin` labels are exported as percentile ranges with cut-point hints instead of opaque integer-only names
- `pred_bins` exports variable-specific prediction-bin columns such as `pred_bins_vm`, `pred_bins_va`, and `pred_bins_line_loading`, each with matching percentile-range labels
- screening currently uses a possible-violation interpretation based on interval endpoints crossing the configured bounds
- `cp_type: kernel` writes per-scenario diagnostics (`kcp_n_eff`, `kcp_bandwidth_sq`, `kcp_fallback`) into the calibrated parquet so localization behavior can be audited
