# Data schemas

This page describes the files `gridfm-trustlayer` reads and writes: their row layout, the columns it uses, and how it interprets them. Predictions from a model other than GridFM work if they follow the same layout.

## Scope

This page covers:

- graphkit prediction artifacts used as trustlayer inputs
- datakit raw ground-truth parquets used as trustlayer inputs
- the derived `line_loading` computation path
- the calibrated parquet and CSV artifacts produced by trustlayer

Before version 1.0, these layouts may change between minor versions; changes are listed in the [Changelog](../changelog.md).

## Artifact families

Trustlayer works with five kinds of files.

### Prediction artifacts

Produced by graphkit.

Examples:

- PF-style bus parquet: `<grid>_predictions.parquet`
- OPF bus parquet: `<grid>_predictions.parquet`
- OPF generator parquet: `<grid>_gen_predictions.parquet`

### Scenario embedding artifacts

Consumed by `compute` and `calibrate` when `cp_type: kernel` is requested.

Example:

- `*_bus_embeddings.parquet`

### Ground-truth artifacts

Produced by datakit.

Examples:

- `raw/bus_data.parquet`
- `raw/branch_data.parquet`

### Trustlayer calibration artifacts

Produced by `gridfm_trustlayer compute`.

Example:

- `*.trustlayer`

### Trustlayer calibrated and screening outputs

Produced by `gridfm_trustlayer calibrate` and `gridfm_trustlayer screen`.

Examples:

- calibrated voltage parquet
- calibrated line-loading parquet
- trust-analysis CSVs
- trust-summary CSVs
- trust-screen CSVs

## Prediction artifacts

### Common row layout

The bus prediction parquet is long-form: one row per `(scenario, bus)`, with every scenario carrying the same buses.

### Common bus-level fields

Typical fields used by trustlayer include:

- `scenario`
- `bus`
- `Vm_target`
- `Va_target`
- `Pg_target`
- `Qg_target`
- `Vm_pred`
- `Va_pred`
- `Pg_pred`
- `Qg_pred`
- `PQ`
- `PV`
- `REF`

Additional task-specific or evaluation columns may also be present, such as:

- `Pd`
- `Qd`
- `Vm_min`
- `Vm_max`
- `Qg_min`
- `Qg_max`
- `active res. (MW)`
- `reactive res. (MVar)`
- `PBE`

### PowerFlow semantics

Interpretation:

- `Vm_target`, `Va_target`, `Pg_target`, and `Qg_target` are direct targets exposed in the prediction parquet
- `Vm_pred`, `Va_pred`, `Pg_pred`, and `Qg_pred` are raw graphkit prediction outputs
- `PQ`, `PV`, and `REF` encode bus-type masks

Important note:

- graphkit computes PF residual metrics from a clamped evaluation tensor
- the exported `*_pred` columns remain the raw model outputs
- trustlayer applies PF-specific correction rules behind the task adapter rather than assuming the exported predictions are already clamped

When loading PowerFlow predictions:

- `Vm` is clamped on `PV` and `REF` buses
- `Va` is clamped on `REF` buses
- `Pg_pred` and `Qg_pred` remain available as raw direct-output channels in `PredictionSet.channels`

### OptimalPowerFlow semantics

OPF predictions differ from PF ones and are not clamped like them.

Files:

- one bus parquet: `<grid>_predictions.parquet`
- one generator parquet: `<grid>_gen_predictions.parquet`

Interpretation:

- bus `Pg_pred` is generator-derived, not an independent raw bus target
- bus `Qg_pred` is not a free PF-style direct head
- OPF generator prediction rows are long-form per scenario
- generator `idx` is not assumed to be a stable cross-scenario physical generator identifier

Trustlayer therefore preserves the OPF generator artifact as a first-class input instead of forcing it into a rectangular bus-only schema.

## Scenario embedding artifacts

Used only by `cp_type: kernel`, and read from `inputs.calibration.embeddings` (`compute`) and `inputs.test.embeddings` (`calibrate`).

Required columns:

| Column | Type | Description |
| --- | --- | --- |
| `scenario` | integer | Scenario id, matching the predictions parquet. |
| `bus` | integer | Entity id within the scenario. |
| `emb_*` | float | Embedding dimensions. Any number of columns carrying the `emb_` prefix; they are read in sorted name order. |

Graphkit produces this layout with `gridfm_graphkit predict --get_embeddings`, where the embedding width is the model's hidden size times its attention-head count.

Requirements:

- The table must be rectangular: every scenario must carry the same number of entities. A ragged table is rejected rather than silently pooled over differing entity counts.
- It must cover every scenario present in the predictions parquet.

Trustlayer reduces this to one vector per scenario using `embedding_pooling`, then standardizes it per dimension with statistics fitted on the calibration set, and finally projects onto the unit sphere when `distance: cosine` is requested. Only the pooled, standardized vectors are stored in the artifact; the source parquet is not embedded.

## Ground-truth artifacts

Ground truth is read from the raw parquets written by datakit.

Examples:

- `data_out_pf/<grid>/raw/bus_data.parquet`
- `data_out_pf/<grid>/raw/branch_data.parquet`
- `data_out_opf/<grid>/raw/bus_data.parquet`
- `data_out_opf/<grid>/raw/branch_data.parquet`

### Bus data

One row per `(scenario, bus)`. Columns used:

- `scenario`
- `bus`
- `Pd`
- `Qd`
- `Pg`
- `Qg`
- `Vm`
- `Va`
- `PQ`
- `PV`
- `REF`
- `vn_kv`

Interpretation:

- `Vm` and `Va` are loaded as rectangular `(n_scenarios, n_buses)` arrays
- `Pd` is summed per scenario to derive `system_load` for `system_load_bin` stratification
- `Pg` and `Qg` are preserved as direct ground-truth channels where relevant
- bus-type masks are preserved as rectangular arrays
- `vn_kv` is used to derive base voltage for loading computation

Angle convention:

- datakit raw `Va` may be stored in degrees
- graphkit prediction artifacts use radians
- trustlayer normalizes degree-scale raw `Va` to radians inside `load_ground_truth()`

### Branch data

One row per `(scenario, idx)`. Columns used:

- `scenario`
- `idx`
- `from_bus`
- `to_bus`
- `rate_a`
- `br_status`
- `Yff_r`
- `Yff_i`
- `Yft_r`
- `Yft_i`
- `Ytf_r`
- `Ytf_i`
- `Ytt_r`
- `Ytt_i`

Interpretation:

- `idx` defines the rectangular branch axis
- `br_status != 1` marks inactive branches
- `n_outages` is derived per scenario from both inactive-branch counts and inactive-generator counts when a sibling `gen_data.parquet` is available
- `from_bus`, `to_bus`, and `rate_a` are taken from the first scenario as topology and rating reference columns
- the complex admittance coefficients are kept as aligned rectangular arrays

## Derived `line_loading`

`line_loading` is computed by `gridfm_trustlayer.compute_line_loading`, which delegates the AC branch flows to datakit's `gridfm_datakit.utils.power_balance.compute_branch_powers_vectorized`:

1. build a bus dataframe from aligned `Vm` and `Va`
2. build a branch dataframe from topology, status, ratings, and admittances
3. delegate AC branch-flow computation to datakit
4. compute loading as `max(|S_from|, |S_to|) / rate_a`
5. set inactive or unrated branches to `-1`

Scale convention:

- `0.5` means 50% of the branch rating
- `1.0` means 100% of the branch rating
- `> 1.0` means overload relative to `rate_a`

Behavioral reference:

- `gridfm_datakit.validation` remains the semantic reference for loading rules
- trustlayer reuses the datakit branch-power helper rather than copying the loading math locally

## Trustlayer runtime objects

`load_predictions()` returns a `PredictionSet` containing:

- `task_name`
- `channel_names`
- `scenario_ids`
- `bus_ids`
- `Vm`
- `Va`
- `channels`
- `bus_type_masks`

`load_ground_truth()` returns a `GroundTruthSet` containing:

- `scenario_ids`
- `bus_ids`
- `branch_ids`
- `n_outages`
- `outage_type`
- `system_load`
- `removed_mask`
- `Vm`
- `Va`
- `channels`
- `admittances`
- `f_bus`
- `t_bus`
- `rate_a`
- `base_kv`
- `sn_mva`
- `bus_type_masks`

## Trustlayer produced artifacts

### `.trustlayer` artifact

Produced by `compute`.

Contains:

- one calibrator for voltage targets, or
- one calibrator for line-loading targets, or
- a bundle containing separate calibrators for each target family in a mixed run

Two on-disk encodings exist, selected by `cp_type`. `load_interval_artifacts` detects the encoding from the file's magic bytes, so both are read through the same path and the file extension does not change.

| `cp_type` | Encoding | Stored state |
| --- | --- | --- |
| `element` | JSON | per-entity residual quantiles, optional per-stratum quantiles, stratification metadata, bin edges, fingerprint of the calibration predictions |
| `kernel` | compressed binary container (`npz`) with a `__meta__` JSON entry | everything above plus the calibration residual bank, per-entity valid masks, standardized calibration embeddings, embedding standardization statistics, per-stratum fallback quantiles, and the kernel settings (`distance`, `bandwidth`, `k_nn`, `n_eff_target`, `n_eff_min`, `embedding_pooling`), plus a `tuning` record (candidates, selection reason, folds) when `n_eff_target: auto` was used |

Kernel artifacts are therefore much larger than element artifacts, because they retain the calibration residuals and embeddings rather than reducing them to a quantile table.

### Calibrated voltage parquet

Produced by `calibrate` when bus-level targets are present.

Typical columns include:

- original prediction parquet columns
- `Vm_pi_lower`, `Vm_pi_upper` when `Vm` is calibrated
- `Va_pi_lower`, `Va_pi_upper` when `Va` is calibrated
- optional stratification columns such as `outage_type`, `outage_type_label`, `system_load_bin`, `system_load_bin_label`, `pred_bins_vm`, `pred_bins_vm_label`, `pred_bins_va`, and `pred_bins_va_label`
- kernel diagnostic columns `kcp_n_eff`, `kcp_bandwidth_sq`, and `kcp_fallback` when `cp_type: kernel`

### Calibrated line-loading parquet

Produced by `calibrate` when `line_loading` is requested.

Typical columns include:

- `scenario`
- `branch`
- `from_bus`
- `to_bus`
- `rate_a`
- `line_loading_pred`
- `line_loading_target`
- `line_loading_pi_lower`
- `line_loading_pi_upper`
- optional stratification columns such as `outage_type`, `outage_type_label`, `system_load_bin`, `system_load_bin_label`, `pred_bins_line_loading`, and `pred_bins_line_loading_label`
- kernel diagnostic columns `kcp_n_eff`, `kcp_bandwidth_sq`, and `kcp_fallback` when `cp_type: kernel`

### Analysis CSVs

Produced by `calibrate`, next to the calibrated parquets: `trust_analysis_{volt,line}.csv` (per entity), `trust_summary_{volt,line}.csv` (per group of entities) and `trust_reliability_{volt,line}.csv` (per variable and stratum).

- With stratification, rows carry `analysis_scope`, `stratification`, `stratum_value` and `stratum_label`.
- `system_load_bin` labels are percentile ranges with cut-point hints, e.g. `system_load_bin=p20-p40 [1450, 1620)`.
- `pred_bins` labels are per variable, e.g. `pred_bins_vm=p00-p50 (<1.0)` or `pred_bins_line_loading=p50-p100 (>=0.82)`.

Every column is explained in [Reading the results](../user-guide/results.md).

### Screening CSVs

Produced by `screen`.

Voltage side:

- `trust_screen_detail_volt.parquet/` (partitioned dataset)
- `trust_screen_entity_volt.csv`
- `trust_screen_summary_volt.csv`

Line side:

- `trust_screen_detail_line.parquet/` (partitioned dataset)
- `trust_screen_entity_line.csv`
- `trust_screen_summary_line.csv`

## Known limits

- Kernel artifacts embed the calibration residuals and embeddings, so their size grows with the calibration set and the embedding width, unlike element artifacts.
- OPF generator predictions are identified per scenario only: a generator index is not assumed to refer to the same physical generator across scenarios.
