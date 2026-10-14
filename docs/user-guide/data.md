# Preparing the data

Trustlayer turns a model's errors on scenarios with a known solution into bounds for new scenarios. Whether those bounds hold depends almost entirely on the data you calibrate and test on. This page explains what to prepare and why.

## What trustlayer reads

For each dataset, calibration and test:

| File | Produced by | Key in the config | Needed for |
| --- | --- | --- | --- |
| Predictions, one row per scenario and bus | graphkit `predict` | `predictions` | always |
| Bus ground truth (`raw/bus_data.parquet`) | datakit `generate` | `pf_node` | always |
| Branch ground truth (`raw/branch_data.parquet`) | datakit `generate` | `pf_edge` | always; supplies line parameters and ratings for `line_loading` |
| Embeddings, one row per scenario and bus | graphkit `predict --get_embeddings` | `embeddings` | `cp_type: kernel` only |
| Generator predictions | graphkit `predict` on an OPF model | `generator_predictions` | `task_name: OptimalPowerFlow`, optional |

Every input can be a single parquet file, as graphkit writes, or a partitioned directory such as datakit's `raw/bus_data.parquet/scenario_partition=k/`. Trustlayer reads only the columns it needs and, in `calibrate` and `screen`, one block of scenarios at a time.

Trustlayer uses the scenarios present in the **predictions** file and reads the matching rows from the ground truth, so the ground truth may contain more scenarios than were predicted. The exact columns are listed in [Data schemas](../reference/schemas.md).

New scenarios for `apply`, which have no known solution, need less: the predictions (for power flow, with the set-points `Vm_target` at PV buses and `Va_target` at the reference bus), and the bus and branch files only when the layer needs the outages, loads or branch parameters. See the [CLI reference](../reference/cli.md#apply).

The predictions do not have to come from a GridFM model. Any model works if its output is written in the same layout, or passed as arrays through the [Python API](python-api.md).

## Two separate datasets

The config always names two datasets:

```yaml
inputs:
  calibration:   # compute learns the error sizes here
    predictions: ...
    pf_node: ...
    pf_edge: ...
  test:          # calibrate and screen measure coverage here
    predictions: ...
    pf_node: ...
    pf_edge: ...
```

**Coverage only means something on data the layer was not fitted on.** On the calibration data itself, the intervals contain the true values by construction. Trustlayer therefore never splits a dataset internally and enforces the separation in two places:

- the config is rejected if both blocks point to the same predictions file;
- the artifact stores a fingerprint of the calibration predictions, and `calibrate` refuses test predictions identical to them, for example a renamed copy.

### Why not a random split of one dataset?

Datakit usually solves several perturbed topologies for each sampled load profile (`n_topology_variants` in its config). These scenarios differ by one or two outages and are otherwise near-identical. In a random split of one dataset, most test scenarios have a near-twin in the calibration set, and both the measured coverage and the gain from kernel localization look better than they will be on genuinely new scenarios.

Generate the two sets as two separate datakit runs instead: same configuration, different output directory, and different random seeds. Leaving `settings.seed: null` gives each run its own seed.

### What "drawn the same way" means

The coverage guarantee of `cp_type: element` assumes that calibration and test scenarios are **exchangeable**, i.e. drawn from the same distribution. In practice:

- use the **same network, model and datakit settings** for both sets, including the load model and the contingency distribution (`topology_perturbation`, `outage_count_probabilities`);
- calibrate **for the conditions you will apply the layer to**. Intervals calibrated on N-1 scenarios say nothing about N-3 scenarios, and intervals calibrated at moderate load say little about peak load;
- when **the model changes** (retraining, fine-tuning, a new checkpoint), its errors change: run `compute` again.

## How many scenarios

Bounds are fitted for every bus or line separately, and within each stratum when `strat` is set. Each of those cells needs enough calibration scenarios:

- **Minimum.** A finite bound needs at least $\lceil 2/\alpha \rceil - 1$ valid residuals, i.e. 19 at $\alpha = 0.1$ and 199 at $\alpha = 0.01$. With fewer, the bound is $\pm\infty$ rather than misleadingly narrow, and the `screen` step will flag every such case.
- **Stable.** Bounds from a few hundred residuals per cell vary noticeably between calibration sets; a few thousand give stable bounds. With `strat: n_outages`, remember that rare contingency levels (N-3 and above, typically) get few scenarios.
- **Kernel.** `cp_type: kernel` localizes within each stratum, so it needs more scenarios than `element`. A stratum with fewer calibration scenarios than `n_eff_target` (600 by default) gets uniform weights, which is exactly `element`; localization only has room to work when a stratum holds several times the target.

The test set only needs to be large enough to measure coverage precisely: with $n$ test scenarios, the standard error of a coverage estimate near $1-\alpha$ is about $\sqrt{\alpha(1-\alpha)/n}$, i.e. 0.003 at $\alpha = 0.1$ and $n = 10{,}000$.

## Power flow or optimal power flow data

- **PowerFlow** (`settings.mode: pf` in datakit). Topology changes are applied after the dispatch, so limits can be violated. This is the natural setting for contingency screening: the test set contains real overloads and voltage violations, and `screen` measures how many the bounds catch.
- **OptimalPowerFlow** (`settings.mode: opf`). Every scenario satisfies its limits by construction, so screening against the same limits finds no true violations. Screen against tighter, near-limit bounds instead, for example $V_m \in [0.97, 1.03]$ or loading $\le 0.95$. Pass the generator predictions with `generator_predictions` and set `task_name: OptimalPowerFlow`; see the `element_n_outages_opf.yaml` example.

## Embeddings for the kernel method

`cp_type: kernel` compares scenarios through the model's internal representation. Export it with `gridfm_graphkit predict --get_embeddings`, which writes `<grid>_bus_embeddings.parquet` next to the predictions, for both datasets. Trustlayer pools the per-bus vectors into one vector per scenario. [Kernel-weighted calibration](kernel.md) explains the options.

The embeddings must come from the **same model** that made the predictions, and they must cover every predicted scenario.
