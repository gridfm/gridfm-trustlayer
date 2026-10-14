# Example trustlayer configs

Each file is a complete, runnable config that demonstrates one combination of options.
Every config names two separately generated datasets, `inputs.calibration` (read by `compute`)
and `inputs.test` (read by `calibrate` and `screen`). They all point at `path/to/...`
placeholders — substitute your own artifacts, then:

```bash
gridfm_trustlayer compute   --config examples/config/<file>.yaml
gridfm_trustlayer calibrate --config examples/config/<file>.yaml
gridfm_trustlayer screen    --config examples/config/<file>.yaml
```

## What each one shows

| Config | `cp_type` | `strat` | Demonstrates |
| --- | --- | --- | --- |
| `element_global_pf.yaml` | element | none | Smallest useful run. One global quantile per bus, single target family. Start here. |
| `element_n_outages_pf.yaml` | element | `n_outages` | The canonical N-k screening setup: bounds widen with contingency severity. |
| `element_outage_type_pf.yaml` | element | `outage_type` | Groups by outage regime (`none`, `branch_only`, `gen_only`, `mixed`) rather than count. |
| `element_system_load_bin_pf.yaml` | element | `system_load_bin` | Percentile bins of total active demand, with `system_load_n_bins`. |
| `element_pred_bins_pf.yaml` | element | `pred_bins` | Bins individual predictions rather than whole scenarios, with `pred_n_bins`. |
| `kernel_knn_pf.yaml` | kernel | `n_outages` | Kernel-weighted CP with the k-nearest-neighbour bandwidth (`bandwidth: knn`). |
| `kernel_n_eff_pf.yaml` | kernel | `n_outages` | Kernel-weighted CP with the default bandwidth (`bandwidth: n_eff`, target 600). |
| `kernel_n_eff_tuned_pf.yaml` | kernel | `n_outages` | Same, with `n_eff_target: auto`: the target is chosen by cross-validation on the calibration set. |
| `kernel_cosine_pf.yaml` | kernel | `n_outages` | Kernel-weighted CP with `distance: cosine`. |
| `element_n_outages_opf.yaml` | element | `n_outages` | OPF artifacts: generator predictions plus tighter near-limit screening bounds. |

## Choosing a starting point

- **New to the tool** — `element_global_pf.yaml`, then `element_n_outages_pf.yaml`.
- **N-k contingency screening** — `element_n_outages_pf.yaml`, then `kernel_n_eff_pf.yaml` to compare.
- **Kernel calibration on a new network or embedding width** — `kernel_n_eff_pf.yaml`, since
  `n_eff_target` transfers where a fixed `k_nn` does not; `kernel_n_eff_tuned_pf.yaml` to let
  `compute` choose the target from the calibration data.
- **OPF** — `element_n_outages_opf.yaml`.

## Two conventions worth knowing

**`commands.screen.variables`.** Every example restricts screening to the variables that have
operating bounds. `screen` otherwise defaults to the shared `variables` list, which includes `Va`,
and voltage angle has no natural limit to screen against.

**Mixed runs write two parquets.** Voltages are per bus and line loading per branch, so a run
calibrating both writes one file for each. Name them with `outputs.calibrated_predictions_volt` and
`outputs.calibrated_predictions_line` (as these examples do), or set `outputs.calibrated_predictions`
to a directory. A run with one family can use either `outputs.calibrated_predictions` or its
family-specific key.

## Reference

Full key-by-key documentation is in the [configuration reference](https://gridfm.github.io/gridfm-trustlayer/reference/configuration/);
the CLI flags that override them are in the [CLI reference](https://gridfm.github.io/gridfm-trustlayer/reference/cli/).
To build a config for your own choices instead, use the [config builder](../notebooks/config_builder.ipynb).
