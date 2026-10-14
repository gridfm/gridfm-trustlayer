# Troubleshooting

Error messages are quoted as the CLI prints them, with the values in your run in place of `...`.

## Config errors

??? failure "`inputs.predictions is no longer supported: the calibration and test data must be separate datasets.`"
    The config uses the old flat `inputs` block of version 0.0.1. Move the keys under `inputs.calibration` and add an `inputs.test` block for a separately generated dataset:

    ```yaml
    inputs:
      calibration:
        predictions: ...
        pf_node: ...
        pf_edge: ...
      test:
        predictions: ...
        pf_node: ...
        pf_edge: ...
    ```

    `calibration_fraction` and `random_seed` were removed for the same reason; see [Preparing the data](data.md#two-separate-datasets).

??? failure "`inputs.calibration.predictions and inputs.test.predictions point to the same file.`"
    Coverage can only be measured on scenarios the layer was not fitted on. Generate a second dataset with datakit (same settings, different output directory and seed), predict it with your model, and use it as `inputs.test`.

??? failure "`The test predictions are identical to the calibration predictions this artifact was fitted on.`"
    Same cause as above, detected by `calibrate` from the content rather than the path, for example a renamed copy of the calibration file.

??? failure "`Unsupported top-level trustlayer config keys: [...]`"
    A key is misspelled or does not exist. All keys are listed in the [configuration reference](../reference/configuration.md). The [config builder](../getting-started/config-builder.md) only writes valid keys.

??? failure "`cp_type 'kernel' requires inputs.calibration.embeddings pointing at a scenario embeddings parquet.`"
    The kernel method needs the model's embeddings for both datasets. Run `gridfm_graphkit predict` with `--get_embeddings` and set `embeddings` in both `inputs.calibration` and `inputs.test`.

??? failure "`cp_type 'kernel' does not support strat 'pred_bins'`"
    Prediction bins are assigned per bus or line, but the kernel compares whole scenarios. Use `n_outages`, `outage_type`, `system_load_bin`, or no stratification.

??? failure "`system_load_n_bins is required when strat is 'system_load_bin'.` / `pred_n_bins is required when strat is 'pred_bins'.`"
    Binned stratifications need the number of bins, for example `system_load_n_bins: 5`.

??? failure "`n_eff_target: auto tunes the kernel bandwidth, so it requires cp_type: kernel and bandwidth: n_eff.`"
    Automatic tuning only exists for the effective-sample-size bandwidth. Set `bandwidth: n_eff` (the default), or give a number.

??? failure "`The tuning block only applies with n_eff_target: auto`"
    A `tuning:` block is present but the target is fixed. Remove the block, or set `n_eff_target: auto`.

??? failure "`n_eff_target (...) is below n_eff_min (...)` / `tuning.grid contains values below n_eff_min`"
    Every scenario would immediately fall back to the stratum quantile. Raise the target (or the grid values), or lower `n_eff_min`.

## Screening errors

??? failure "`Missing screening bounds for 'Va'.`"
    `screen` screens the shared `variables` list unless told otherwise, and voltage angle has no operating limit. Restrict it:

    ```yaml
    commands:
      screen:
        variables: [Vm, line_loading]
    ```

??? failure "`screen_bounds.Vm must define both 'lower' and 'upper'.` / `screen_bounds.line_loading must define 'upper'.`"
    `Vm` is screened against both limits, `line_loading` only against `upper` (a fraction of `rate_a`).

??? failure "`No calibrated voltage parquet for '...' in .... Run calibrate first, or set outputs.calibrated_predictions_volt to the file to screen.`"
    `screen` reads the parquet written by `calibrate`. Run `calibrate` with the same config first. If you moved or renamed the file, point `outputs.calibrated_predictions_volt` (or `_line`) at it.

??? failure "`A mixed-variable run needs separate parquet outputs.`"
    Calibrating voltages and `line_loading` together writes two files, one per bus and one per branch. Set `outputs.calibrated_predictions` to a **directory**, or set `outputs.calibrated_predictions_volt` and `outputs.calibrated_predictions_line` to two file paths.

## Results that look wrong

??? question "Some bounds are `inf` or `-inf`"
    The stratum had fewer than $\lceil 2/\alpha \rceil - 1$ calibration samples for that bus or line (19 at $\alpha = 0.1$), too few for a finite bound with a coverage guarantee. Add calibration scenarios, use fewer strata (or none), or a larger `alpha`. Every such case is flagged by `screen`, which is the conservative outcome.

??? question "Coverage is clearly below $1-\alpha$ in one stratum"
    With `cp_type: element`, the guarantee assumes calibration and test scenarios are drawn the same way. Check that both datasets were generated with the same datakit settings and the same model, and that the stratum is not tiny on either side (`n_scenarios_min` in `trust_reliability_*.csv`). Small shortfalls, of the order of $\sqrt{\alpha(1-\alpha)/n}$ for $n$ test scenarios, are sampling noise.

??? question "Coverage is well above $1-\alpha$ with `cp_type: kernel`"
    The kernel is localizing too hard: the bounds are estimated from few neighbours and come out conservative. Raise `n_eff_target` (or `k_nn`), or let `n_eff_target: auto` choose it. A high share of `kcp_fallback` points the same way.

??? question "The kernel gives the same bounds as `element`"
    The weights are close to uniform. Under `bandwidth: n_eff`, the strata may be smaller than `n_eff_target`; under `bandwidth: knn`, a wide embedding makes all distances similar. Check `kcp_n_eff` in the calibrated parquet: values near the stratum size mean no localization. Try a lower target, `embedding_pooling: mean`, or `n_eff_target: auto`.

??? question "`screen` finds no true violations"
    Typical for OPF data, which satisfies its limits by construction. Screen against tighter, near-limit bounds, for example `Vm: {lower: 0.97, upper: 1.03}` and `line_loading: {upper: 0.95}`.

## Environment

??? question "The config builder notebook does not list my virtual environment"
    Register the environment as a Jupyter kernel, then reload the notebook and select it:

    ```bash
    python -m pip install ipykernel
    python -m ipykernel install --user --name trustlayer --display-name "Python (trustlayer)"
    ```

??? failure "`gridfm_datakit is required for compute_line_loading().`"
    Line loading is computed with datakit's branch-flow routine. Install the extra: `pip install "gridfm-trustlayer[graphkit] @ git+https://github.com/gridfm/gridfm-trustlayer"`.

??? question "`compute`, `calibrate` or `screen` needs a lot of memory"
    `calibrate` and `screen` process the test set one block of scenarios at a time, so their memory use is set mainly by the block size rather than by the number of test scenarios. `compute` needs the whole calibration set in memory, since every residual enters the quantiles: its memory grows with the number of calibration scenarios times the number of buses and lines. If it is too large, use fewer calibration scenarios; a few thousand per stratum already give stable bounds.

??? failure "`... exists and is not a partitioned trustlayer output: it would be replaced.`"
    The per-row outputs are written as partitioned directories, and an existing directory at that path is only replaced if it holds a previous trustlayer output. Point `outputs.calibrated_predictions*` (or `screen_dir`) at a new path, or remove the directory.
