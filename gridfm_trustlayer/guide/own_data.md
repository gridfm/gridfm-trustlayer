# Using trustlayer with your own data

Trustlayer adds calibrated bounds to the predictions of any model: a GridFM model, another neural network, a DC power flow, a commercial tool. The command-line tool reads GridFM's file layout; with any other model, use the Python API as described here. You need:

- a **calibration set**: scenarios with your model's predictions and the true values (e.g. from an AC power flow), separate from the scenarios you evaluate or operate on;
- the **predictions for new scenarios** (and, to check the coverage on test scenarios, their true values);
- for the kernel method (`KernelIntervalCalibrator`), **one vector per scenario** that describes it, e.g. your model's embeddings averaged over the buses.

This guide ships with the package: `gridfm_trustlayer guide` prints it and `gridfm_trustlayer guide --path` gives its location, so that you, or an AI coding assistant, can read the version that matches your installation. It is also online at <https://gridfm.github.io/gridfm-trustlayer/user-guide/own-data/>.

## 1. Put your data in a long table

One row per scenario and element (bus or line):

| Column | Content | Needed |
| --- | --- | --- |
| scenario id | integer | always |
| element id | integer, numbered the same way in every dataset | always |
| prediction | one column per variable, e.g. `loading_pred` | always |
| true value | one column per variable, e.g. `loading_true` | calibration, and test scenarios to check coverage |
| validity | boolean, `False` where a value must not be calibrated | optional |
| stratum | integer per scenario, e.g. the number of outages | optional, for `strat` |

For the kernel method, add a second table with one row per scenario: the scenario id as its index (or a `scenario` column) and one column per embedding dimension.

Every scenario should list the same elements. An element that does not apply in a scenario (e.g. an out-of-service line) can be left out of the table or marked invalid: either way it is not used for fitting and gets no interval.

## 2. Mistakes to avoid

- **Calibrating and evaluating on the same scenarios.** Coverage measured on the calibration data says nothing about new scenarios. Use separately generated datasets.
- **Values the model does not predict.** Inputs that the model copies (e.g. set-points) are exact and would make the bounds too narrow: mark them invalid.
- **Embeddings in another order than the predictions.** Pass them as a table indexed by scenario id; `ChannelSet.from_frame` aligns them by id.
- **One embedding per bus.** The kernel needs one vector per scenario: average (or otherwise pool) the per-bus vectors of each scenario first, the same way for calibration and new scenarios.
- **Different elements or numbering** between the calibration set and new scenarios.
- **Strata that are not integers**, or that change within a scenario: bin continuous quantities (e.g. the total load) into integers first.

## 3. Example

A model's outputs on 4,000 contingency scenarios of 20 lines, as a long table. Its errors grow with the number of outages and are three times larger in half of the scenarios ("hard" ones), which its embedding can tell apart:

```python
import numpy as np
import pandas as pd

from gridfm_trustlayer import (
    ChannelSet,
    KernelIntervalCalibrator,
    NaiveIntervalCalibrator,
    check_inputs,
)

rng = np.random.default_rng(0)
n_scenarios, n_lines = 4000, 20
scenario = np.repeat(np.arange(n_scenarios), n_lines)
line = np.tile(np.arange(n_lines), n_scenarios)
n_outages = rng.integers(1, 4, n_scenarios)  # contingency level of each scenario
hard = rng.random(n_scenarios) < 0.5  # scenarios where the model is less accurate
predicted = rng.uniform(0.2, 1.1, scenario.size)
scale = 0.01 * n_outages[scenario] * np.where(hard[scenario], 3.0, 1.0)
table = pd.DataFrame(
    {
        "scenario": scenario,
        "line": line,
        "n_outages": n_outages[scenario],
        "loading_pred": predicted,
        "loading_true": predicted + rng.normal(0.0, scale),
        "in_service": rng.random(scenario.size) > 0.02,  # False: no interval
    }
)
# One vector per scenario, e.g. the model's embeddings averaged over the buses.
embeddings = pd.DataFrame(
    {
        "z1": hard + rng.normal(0.0, 0.2, n_scenarios),
        "z2": rng.normal(size=n_scenarios),
    },
    index=pd.Index(np.arange(n_scenarios), name="scenario"),
)
```

Build the predictions and the true values from the same rows, then check them before fitting. Here the first half of the scenarios is the calibration set:

```python
def channel_sets(rows):
    """Predictions and true values of these rows, as aligned ChannelSets."""

    common = {
        "entity": "line",
        "entity_name": "branch",
        "valid": "in_service",
        "strata": ["n_outages"],
        "embeddings": embeddings,
    }
    predictions = ChannelSet.from_frame(
        rows, {"line_loading": "loading_pred"}, **common
    )
    true_values = ChannelSet.from_frame(
        rows, {"line_loading": "loading_true"}, **common
    )
    return predictions, true_values


calibration = table[table["scenario"] < 2000]
calibration_predictions, calibration_true = channel_sets(calibration)
print(
    check_inputs(
        calibration_predictions,
        calibration_true,
        strat="n_outages",
        alpha=0.1,
        kernel=True,
    )
)
```

`check_inputs` prints `No problems found.`, or the errors to fix and the warnings to read (section 4). Fit both methods with 90% intervals (`alpha=0.1`, 5% per tail), stratified by the number of outages:

```python
settings = {"alpha": 0.1, "channel_names": ("line_loading",), "strat": "n_outages"}
scp = NaiveIntervalCalibrator.fit(calibration_predictions, calibration_true, **settings)
kcp = KernelIntervalCalibrator.fit(
    calibration_predictions,
    calibration_true,
    embeddings=calibration_predictions.scenario_embeddings,
    n_eff_target=200,  # about 200 calibration scenarios weigh in each bound
    **settings,
)
```

Apply them to the other scenarios. New scenarios need only the predictions, with the same validity, strata and embeddings; here the true values are known, so the coverage can be checked:

```python
test = table[table["scenario"] >= 2000]
test_predictions, test_true = channel_sets(test)
print(check_inputs(test_predictions, calibrator=kcp))

valid = test_predictions.valid_masks["line_loading"]
truth = test_true.channels["line_loading"]
hard_test = hard[test_predictions.scenario_ids][:, None]
for name, calibrator in (("SCP", scp), ("KCP", kcp)):
    lower, upper = calibrator.predict_intervals(test_predictions)["line_loading"]
    covered = (lower <= truth) & (truth <= upper)
    print(
        f"{name}: coverage {covered[valid].mean():.3f} "
        f"(easy {covered[valid & ~hard_test].mean():.3f}, "
        f"hard {covered[valid & hard_test].mean():.3f}), "
        f"mean width {(upper - lower)[valid].mean():.3f}"
    )
```

```text title="Output"
No problems found.
No problems found.
SCP: coverage 0.905 (easy 1.000, hard 0.805), mean width 0.157
KCP: coverage 0.924 (easy 0.944, hard 0.903), mean width 0.141
```

SCP covers 90% overall, as guaranteed, but with one margin per line and stratum: easy scenarios are over-covered and hard ones under-covered. KCP weighs the calibration scenarios by their distance to each test scenario in the embedding, so its margins follow the difficulty of each scenario: it gets close to 90% in both groups, with narrower intervals on average. Its coverage is checked on test data, not guaranteed: here it is slightly above the target.

## 4. What `check_inputs` reports

`check_inputs(predictions, true_values, strat=..., alpha=..., kernel=...)` before fitting, and `check_inputs(new_predictions, calibrator=...)` before predicting, return an `InputCheck` with:

- **errors**, to fix first: predictions and true values with different or differently ordered scenarios or elements; ids that are not integers; a missing variable or stratum column; strata that are not integers; missing, NaN or constant embeddings for the kernel; new data with other elements or another embedding width than the fitted calibrator;
- **warnings**: values that are NaN but marked valid; elements without any valid calibration value; elements with too few calibration values in a stratum for a finite bound there (they use their quantile over all strata, without a guarantee within that stratum), or too few overall (infinite bounds); new scenarios in strata absent from the calibration data.

`check.ok` is `True` when there are no errors, and `check.raise_for_errors()` raises them.

## 5. Choosing a method

- **`NaiveIntervalCalibrator` (SCP)** needs only predictions and true values. When calibration and new scenarios are exchangeable, its intervals cover with probability at least `1 - alpha` within each stratum, for any calibration size. An element needs at least `ceil(2 / alpha) - 1` valid calibration values in a stratum (19 at `alpha=0.1`) for a finite bound there.
- **`KernelIntervalCalibrator` (KCP)** also needs one vector per scenario, and adapts the margin to each scenario. Choose `n_eff_target` with `gridfm_trustlayer.conformal.tuning.tune_n_eff_target`, a cross-validation on the calibration set alone, or check the coverage on held-out data. A model without embeddings can use any vector that describes the scenario (e.g. its loads and outages); whether it separates easy from hard scenarios has to be checked the same way.
- `alpha` is two-sided: `alpha=0.1` gives 90% intervals with 5% per tail. For a one-sided screen at level `1 - a`, fit with `alpha = 2 * a` and use the upper bound.

## 6. Next steps

- Screening against limits: `classify_status` and `scenario_status` give the status of every value and scenario (within limits, uncertain, violation; secure, uncertain, insecure).
- Saving and loading a fitted calibrator: `save_interval_artifact` and `load_interval_artifact`.
- More: the Python API guide (<https://gridfm.github.io/gridfm-trustlayer/user-guide/python-api/>), the API reference (<https://gridfm.github.io/gridfm-trustlayer/reference/api/>) and the kernel method (<https://gridfm.github.io/gridfm-trustlayer/user-guide/kernel/>).
