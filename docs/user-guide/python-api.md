# Python API

The CLI covers the usual workflow from parquet files. The Python API exposes the same calibrators on plain NumPy arrays, so you can use them with any model, inside your own pipeline, or in a notebook. Import everything from the top-level package:

```python title="Public API"
from gridfm_trustlayer import (
    ChannelSet,  # aligned (scenario x entity) arrays
    NaiveIntervalCalibrator,  # cp_type: element
    KernelIntervalCalibrator,  # cp_type: kernel
    load_predictions,
    load_ground_truth,
    load_scenario_embeddings,
    compute_line_loading,
    save_interval_artifact,
    load_interval_artifact,
)
```

Every function and class is documented in the [API reference](../reference/api.md).

## Data: `ChannelSet`

A `ChannelSet` holds one or more variables ("channels") as arrays of shape `(n_scenarios, n_entities)`, where an entity is a bus or a line. Predictions and true values are two `ChannelSet`s with the same scenario and entity ids, in the same order.

| Field | Content |
| --- | --- |
| `scenario_ids` | `(n_scenarios,)` scenario identifiers |
| `entity_ids`, `entity_name` | `(n_entities,)` bus or branch identifiers, and `"bus"` or `"branch"` |
| `channels` | `{"Vm": array, ...}`, one `(n_scenarios, n_entities)` array per variable |
| `scenario_columns` | per-scenario values used for stratification, e.g. `{"n_outages": array}` |
| `valid_masks` | optional `{"Vm": bool array}`; `False` entries are ignored when fitting |
| `scenario_embeddings` | `(n_scenarios, d)` array, for `KernelIntervalCalibrator` only |

## Stratified conformal intervals

`NaiveIntervalCalibrator` is the `cp_type: element` method. This example uses synthetic data in which the prediction error grows with the contingency level:

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
# With real data, use separately generated datasets.
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

`fit` takes the predictions, the true values, `alpha` and the variables to calibrate. With `strat`, both `ChannelSet`s must carry that column in `scenario_columns`, and so must the predictions passed to `predict_intervals`: the stratum of a new scenario has to be known to pick its bounds. `predict_intervals` returns `{variable: (lower, upper)}`, each of the same shape as the predictions.

Two conventions worth knowing:

- **`alpha` is two-sided.** Each tail gets `alpha / 2`, so `alpha=0.1` gives 90% intervals whose upper bound alone holds with probability at least 95%. For a one-sided screen at level `1 - a`, pass `alpha = 2 * a` and use only the upper bound.
- **Too few calibration samples give infinite bounds.** A stratum needs at least `ceil(2 / alpha) - 1` valid residuals (19 at `alpha=0.1`) for a finite bound. With fewer, the bound is `+inf` / `-inf` rather than a falsely narrow interval.

## Kernel-weighted intervals

`KernelIntervalCalibrator` is the `cp_type: kernel` method. It needs one embedding vector per scenario, both in `fit` (`embeddings=`) and on the predictions passed to `predict_intervals` (`scenario_embeddings`). This example builds data where the model is four times less accurate in half of the scenarios, and that half is visible in the embedding; it compares both methods on easy and hard scenarios separately:

```python
import numpy as np
from gridfm_trustlayer import (
    ChannelSet,
    KernelIntervalCalibrator,
    NaiveIntervalCalibrator,
)

# Synthetic stand-in: 3,000 scenarios x 40 buses, one 2-dimensional embedding
# per scenario. The model is four times less accurate in scenarios whose first
# embedding coordinate is positive, so difficulty is visible in the embedding.
rng = np.random.default_rng(0)
n_scenarios, n_buses = 3000, 40
embeddings = rng.normal(size=(n_scenarios, 2))
hard = embeddings[:, 0] > 0
predicted = rng.uniform(0.95, 1.05, (n_scenarios, n_buses))
noise_scale = np.where(hard, 0.02, 0.005)[:, None]
true = predicted + rng.normal(0.0, 1.0, predicted.shape) * noise_scale


def channel_set(values, rows):
    return ChannelSet(
        scenario_ids=np.flatnonzero(rows),
        entity_ids=np.arange(n_buses),
        entity_name="bus",
        channels={"Vm": values[rows]},
        scenario_embeddings=embeddings[rows],
    )


calibration = np.arange(n_scenarios) < 2000
fit_args = dict(alpha=0.1, channel_names=("Vm",))
element = NaiveIntervalCalibrator.fit(
    channel_set(predicted, calibration), channel_set(true, calibration), **fit_args
)
kernel = KernelIntervalCalibrator.fit(
    channel_set(predicted, calibration),
    channel_set(true, calibration),
    embeddings=embeddings[calibration],
    embedding_pooling="mean",  # only recorded in the artifact here
    n_eff_target=300,  # bandwidth: n_eff is the default
    **fit_args,
)

test = channel_set(predicted, ~calibration)
test_true, test_hard = true[~calibration], hard[~calibration]
for name, calibrator in [("element", element), ("kernel", kernel)]:
    lower, upper = calibrator.predict_intervals(test)["Vm"]
    covered = (lower <= test_true) & (test_true <= upper)
    width = upper - lower
    print(
        f"{name:8s} coverage easy {covered[~test_hard].mean():.2f}, "
        f"hard {covered[test_hard].mean():.2f} | "
        f"width easy {width[~test_hard].mean():.3f}, hard {width[test_hard].mean():.3f}"
    )
```

```text title="Output"
element  coverage easy 1.00, hard 0.80 | width easy 0.051, hard 0.051
kernel   coverage easy 0.95, hard 0.90 | width easy 0.027, hard 0.066
```

Both reach about 90% overall, but `element` gives every scenario the same margin: easy scenarios are over-covered and hard ones under-covered. The kernel adapts the margin to each scenario and gets much closer to 90% in both groups. After `predict_intervals`, `kernel.diagnostics` holds the per-scenario `n_eff`, `kernel_bandwidth_sq` and `kernel_fallback` arrays ([what they mean](kernel.md#checking-that-it-works)).

The keyword arguments mirror the config keys: `bandwidth`, `n_eff_target`, `k_nn`, `n_eff_min`, `distance` and `strat`. In practice, embeddings come from `load_scenario_embeddings`, which reads graphkit's embeddings parquet and pools it per scenario.

## From GridFM parquet files

The loaders read the files the CLI reads. `load_predictions` returns a `PredictionSet` and `load_ground_truth` a `GroundTruthSet`; both hold `(n_scenarios, n_buses)` arrays such as `.Vm` and `.Va`, aligned on the same scenarios when you pass `scenario_ids`. The ground truth also provides the per-scenario `n_outages`, `outage_type` and `system_load` for stratification:

```python title="Calibrating Vm and Va from GridFM files"
from gridfm_trustlayer import (
    ChannelSet,
    NaiveIntervalCalibrator,
    load_ground_truth,
    load_predictions,
    save_interval_artifact,
)


def load(predictions_path, bus_path, branch_path):
    """Predicted and true Vm, Va as ChannelSets, with each scenario's outage count."""
    predictions = load_predictions(predictions_path, task_name="PowerFlow")
    truth = load_ground_truth(
        bus_path, branch_path, scenario_ids=predictions.scenario_ids
    )

    def channel_set(source):
        return ChannelSet(
            scenario_ids=predictions.scenario_ids,
            entity_ids=predictions.bus_ids,
            entity_name="bus",
            channels={"Vm": source.Vm, "Va": source.Va},
            scenario_columns={"n_outages": truth.n_outages},
        )

    return channel_set(predictions), channel_set(truth)


cal_predicted, cal_true = load(
    "predictions/calibration/case118_ieee_predictions.parquet",
    "data_calibration/case118_ieee/raw/bus_data.parquet",
    "data_calibration/case118_ieee/raw/branch_data.parquet",
)
calibrator = NaiveIntervalCalibrator.fit(
    cal_predicted, cal_true, alpha=0.1, channel_names=("Vm", "Va"), strat="n_outages"
)
save_interval_artifact(calibrator, "case118_vm_va.trustlayer")

test_predicted, test_true = load(
    "predictions/test/case118_ieee_predictions.parquet",
    "data_test/case118_ieee/raw/bus_data.parquet",
    "data_test/case118_ieee/raw/branch_data.parquet",
)
lower, upper = calibrator.predict_intervals(test_predicted)["Vm"]
```

`load_predictions` applies task-specific corrections: for `PowerFlow`, `Vm` is clamped on PV and reference buses and `Va` on the reference bus, as described in [Data schemas](../reference/schemas.md#powerflow-semantics). For line loading, `compute_line_loading(Vm, Va, ground_truth)` returns the `(n_scenarios, n_branches)` loading of given voltages, using datakit's branch-flow routine (requires the `graphkit` extra). Apply it to the predicted and to the true voltages and calibrate the two arrays like any other channel.

## Saving and loading

```python title="Artifacts"
from gridfm_trustlayer import load_interval_artifact, save_interval_artifact

save_interval_artifact(calibrator, "intervals.trustlayer")
calibrator = load_interval_artifact("intervals.trustlayer")
```

The file format is the one the CLI writes: JSON for `NaiveIntervalCalibrator`, a compressed NumPy archive for `KernelIntervalCalibrator`.

## Choosing the kernel bandwidth from Python

The cross-validated choice of `n_eff_target` behind `n_eff_target: auto` is available as `gridfm_trustlayer.conformal.tuning.tune_n_eff_target`. It takes the calibration `ChannelSet`s and a `TuningOptions` object, and returns the chosen target and distance with a table of all candidates. See [Kernel-weighted calibration](kernel.md#choosing-the-bandwidth-automatically) for the procedure.
