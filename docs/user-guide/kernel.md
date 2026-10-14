# Kernel-weighted calibration

With `cp_type: kernel`, the bound for a test scenario is learned mostly from the calibration scenarios that **resemble it**, rather than from all calibration scenarios in its stratum. Resemblance is measured in the model's own embedding space. Where similar scenarios had small errors, the bound is narrow; where they had large errors, it is wide.

This page explains what to provide, which options matter, and how to check that it works. The statistics are in the [Methodology](../methodology.md#cp_type-kernel).

## Minimal config

```yaml
cp_type: kernel
strat: n_outages        # optional; the kernel then localizes within each stratum
alpha: 0.1
# Defaults, shown for clarity:
embedding_pooling: mean
distance: euclidean
bandwidth: n_eff
n_eff_target: 600

inputs:
  calibration:
    predictions: predictions/calibration/case118_ieee_predictions.parquet
    embeddings: predictions/calibration/case118_ieee_bus_embeddings.parquet
    pf_node: ...
    pf_edge: ...
  test:
    predictions: predictions/test/case118_ieee_predictions.parquet
    embeddings: predictions/test/case118_ieee_bus_embeddings.parquet
    pf_node: ...
    pf_edge: ...
```

Embeddings are written by `gridfm_graphkit predict --get_embeddings`, one row per scenario and bus. Both datasets need them, from the same model that made the predictions. Complete examples: `kernel_n_eff_pf.yaml` and `kernel_n_eff_tuned_pf.yaml` in [`examples/config/`](https://github.com/gridfm/gridfm-trustlayer/tree/main/examples/config).

## How it works, in four steps

For each test scenario, at `calibrate` time:

1. **Represent.** The per-bus embeddings are pooled into one vector per scenario and standardized with the mean and spread of the calibration set.
2. **Compare.** Its distance to every calibration scenario in the same stratum is computed.
3. **Weight.** Each calibration scenario gets a weight $u_j = \exp(-d_j^2/h^2)$: close to 1 for near scenarios, close to 0 for distant ones. The **bandwidth** $h^2$ sets what counts as near.
4. **Bound.** For every bus or line, the bound is the weighted quantile of the calibration residuals, with an extra point mass for the unseen test residual that keeps the estimate conservative.

`compute` only stores what these steps need: the standardized calibration embeddings, all calibration residuals, and fallback quantiles per stratum. That is why kernel artifacts are much larger than element artifacts.

## The bandwidth

The bandwidth is the main lever:

- **Too small:** a few neighbours carry all the weight, the quantile is noisy, and the bounds become conservative (coverage above target, wider than needed).
- **Too large:** all weights approach 1, and the result is the same as `cp_type: element`.

Two modes set it per test scenario.

### `bandwidth: n_eff` (default)

The bandwidth is solved so that the weights correspond to a chosen **effective sample size**,

$$
n_{\mathrm{eff}} = \frac{\left(\sum_j u_j\right)^2}{\sum_j u_j^2} = n^{\mathrm{target}}_{\mathrm{eff}},
$$

i.e. the bound behaves as if it were estimated from `n_eff_target` equally weighted calibration scenarios. The default target is 600. The same target means the same thing for any embedding width, stratum size or network, which is why it is the default.

- A stratum with fewer calibration scenarios than the target gets uniform weights, i.e. the `element` result.
- Lower targets localize harder: more adaptive bounds, but noisier.
- `n_eff_target: auto` chooses the target from the calibration data ([below](#choosing-the-bandwidth-automatically)).

### `bandwidth: knn`

The bandwidth is the squared distance to the `k_nn`-th nearest calibration scenario in the stratum (default `k_nn: 20`). It adapts to local density, but the same `k_nn` localizes very differently depending on the embedding width: distances concentrate as the dimension grows, so in a wide embedding `k_nn: 20` can already give nearly uniform weights. Prefer `n_eff` unless you have a reason to fix the neighbour count.

!!! warning "Do not tune on the test set"
    `n_eff_target` and `k_nn` are hyperparameters. Choosing them by looking at coverage or width on the test set makes the reported coverage optimistic. Choose them on the calibration set, which `n_eff_target: auto` does for you.

## Choosing the bandwidth automatically

With `n_eff_target: auto`, `compute` chooses the target by cross-validation **inside the calibration set**; the test set is never used. It requires `cp_type: kernel` and `bandwidth: n_eff`, and is off by default.

```yaml
cp_type: kernel
bandwidth: n_eff
n_eff_target: auto
tuning:                       # optional; these are the defaults
  folds: 5
  search: grid
  grid: [100, 200, 400, 600, 800, 1200]
  group_by: auto
  seed: 0
  coverage_tolerance: auto
```

What happens:

1. The calibration scenarios are split into `folds` folds. Scenarios that share a load profile (datakit's `load_scenario_idx`) stay in the same fold, so near-duplicates never end up on both sides.
2. For every candidate target and every fold, the kernel is fitted on the other folds and applied to the held-out one. This gives a cross-validated coverage, overall and in every stratum, and a mean interval width.
3. A candidate is **eligible** if its coverage reaches $1-\alpha$ minus a small tolerance, overall and in every stratum, for every calibrated variable. The default tolerance is two standard errors, $2\sqrt{\alpha(1-\alpha)/n}$, with $n$ the number of scenarios behind each check.
4. Among eligible candidates, the one with the **narrowest intervals** wins. If none is eligible, the kernel does not reach the target on this data, and the weights are made uniform, which reproduces `element`. The report says so.

Coverage comes first and width second, because a narrow interval is only useful if it holds. Voltage and line-loading targets are tuned separately.

`compute` prints one line per target family, for example:

```text
kernel tuning (line): n_eff_target = 600, distance = euclidean (narrowest eligible candidate; 6 candidates, grid search, 5 grouped folds). Report: trustlayer_outputs/kernel/n_eff_tuned_pf_kernel_tuning.csv
```

When no candidate is eligible, the reason reads `no candidate met the coverage target: uniform weights (equivalent to element)`.

and writes two reports next to the artifact: `<artifact>_kernel_tuning.csv`, one row per candidate (coverage, worst margin to the target, relative width, fallback share, whether it was eligible and selected), and `<artifact>_kernel_tuning_details.csv`, every individual coverage check. The choice is also recorded in the artifact.

### Searching over the distance as well

```yaml
tuning:
  distances: [euclidean, cosine]   # also choose the distance
  search: random
  budget: 12                       # number of (n_eff_target, distance) pairs to try
  n_eff_range: [100, 1500]         # targets drawn log-uniformly from this range
```

`search: grid` tries every combination of `grid` and `distances`; `search: random` tries `budget` pairs drawn at random, which bounds the cost when both vary. With a small budget spread over two distances, each distance gets only a few targets; prefer `search: grid` when only the target varies.

**Cost.** Roughly one kernel run per fold and candidate, each on $(K-1)/K$ of the calibration data: `folds × len(grid) × len(distances)` runs for a grid search, `folds × budget` for a random one. The weighted quantiles dominate, so the cost grows with the number of calibration scenarios and of buses or lines. On a laptop CPU, the default grid takes about ten seconds for 10,000 calibration scenarios on a network with 186 lines.

All options are listed in the [configuration reference](../reference/configuration.md#tuning).

## Distance

- `distance: euclidean` (default): squared Euclidean distance between the standardized scenario vectors.
- `distance: cosine`: the vectors are scaled to unit length first. Since standardization centers them, cosine compares only the *direction* in which two scenarios deviate from the average scenario, not *how far* they deviate.

Which works better depends on whether the size of the deviation carries information about the model's error, which varies by model and dataset. Rather than choosing by hand, let the tuning compare both with `distances: [euclidean, cosine]`. Cosine does not cure distance concentration in wide embeddings; reducing the embedding width does.

## Pooling

`embedding_pooling` reduces the per-bus embeddings of a scenario (n buses × d dimensions) to one vector:

| Value | Width | Notes |
| --- | --- | --- |
| `mean` | d | Default and recommended: the average over buses. |
| `max` | d | Largest value per dimension; emphasizes the most activated bus. |
| `meanmax` | 2d | Both of the above, concatenated. |
| `concat` | n·d | All buses side by side. Only valid when every scenario has the same buses; very wide, so distances concentrate and the kernel may stop localizing. |

The pooling is stored in the artifact and applied again at `calibrate`.

## Fallback: `n_eff_min`

If a test scenario's weights collapse onto too few calibration scenarios, its bound falls back to the stratum-wide quantile, i.e. the `element` result:

- when its effective sample size is below `n_eff_min` (default 5);
- when the weighted quantile cannot reach the target level. That happens when no calibration scenario is close, all weights are small, and the point mass of the unseen test residual dominates. The method declines to extrapolate.

Under `bandwidth: n_eff` the effective sample size is pinned to the target, so the first case only arises in strata smaller than the target. For the target level to be reachable at all, the weights must sum to at least $2/\alpha - 1$ (19 at $\alpha = 0.1$), so a guard below that never binds first. To make `n_eff_min` do real work, set it to at least $2/\alpha$.

## Checking that it works

`calibrate` adds three columns per scenario to the calibrated parquet:

| Column | What to look for |
| --- | --- |
| `kcp_n_eff` | Effective number of calibration scenarios behind the bound. Close to the stratum size means the kernel barely localizes (≈ `element`). Under `bandwidth: n_eff` it equals the target by construction. |
| `kcp_bandwidth_sq` | The bandwidth used. Large values mark regions of the embedding space with few calibration scenarios. |
| `kcp_fallback` | `True` when the scenario fell back to the stratum quantile for at least one bus or line. A high share means the bandwidth is too small for the geometry: raise `n_eff_target` (or `k_nn`). |

Then compare with an `element` run on the same data:

- **Coverage** in `coverage/summary.md` (or `coverage_by_stratum.csv`) should be close to $1-\alpha$ in every stratum. Kernel coverage is checked empirically, and it can land above target when difficulty is very uneven: raising coverage on the hard scenarios lifts the average.
- **Width** should be lower on average, or at least redistributed: narrower on easy scenarios, wider on hard ones.
- **Screening**: at similar recall, fewer false alarms means the kernel is paying off.
