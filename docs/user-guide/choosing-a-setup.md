# Choosing a setup

Three choices shape the intervals: the coverage level `alpha`, the method `cp_type`, and the grouping `strat`. This page explains each, with a recommended default.

!!! tip "Recommended starting point for N-k screening"
    `cp_type: element`, `strat: n_outages`, `alpha: 0.1`. It has a finite-sample coverage guarantee, needs no embeddings, and widens the bounds with contingency severity. Try `cp_type: kernel` afterwards to see whether it narrows the bounds on your data.

## Coverage level: `alpha`

`alpha` is the share of true values allowed to fall outside the interval. It is **two-sided**: each tail gets $\alpha/2$.

| `alpha` | Interval | Each one-sided bound holds with probability |
| --- | --- | --- |
| 0.2 | 80% | 90% |
| 0.1 | 90% | 95% |
| 0.05 | 95% | 97.5% |
| 0.02 | 98% | 99% |

A screen against an upper limit, such as line loading, only uses the upper bound. To have the upper bound alone hold with probability $1-a$, set `alpha` to $2a$: for a 99% upper bound, `alpha: 0.02`.

Smaller `alpha` means wider bounds, more flagged lines, fewer missed violations, and more calibration data needed: a finite bound needs at least $\lceil 2/\alpha \rceil - 1$ residuals per bus or line and stratum (19 at 0.1, 99 at 0.02).

## Method: `cp_type`

Both methods take the same residuals (true value minus prediction) on the calibration set. They differ in which residuals set the bound for a given test scenario.

**`element`: stratified split-conformal prediction.** For every bus or line, and every stratum, `compute` takes a quantile of that entity's residuals. All test scenarios in the same stratum get the same margin around their prediction.

- If calibration and test scenarios are exchangeable, coverage is **guaranteed** to be at least $1-\alpha$ in every stratum, for any number of calibration scenarios.
- It is fast, needs no embeddings, and the artifact is a small JSON table.
- Its weakness: within a stratum, easy and hard scenarios get the same margin. Easy ones are over-covered and hard ones under-covered, and only the average is $1-\alpha$.

**`kernel`: kernel-weighted conformal prediction.** For each test scenario, calibration residuals are weighted by how similar their scenario is to the test scenario, measured in the model's embedding space, and the quantile is taken over the weighted residuals.

- The margin **adapts to each scenario**: narrower where similar calibration scenarios had small errors, wider where they had large ones.
- Coverage is **verified empirically** on the test set rather than guaranteed in finite samples.
- It needs embeddings for both datasets, more calibration data, and more computation at `calibrate` time. The artifact keeps all calibration residuals and embeddings.

Use `kernel` when the model's errors vary between scenarios in ways the strata do not capture, and check on the test set that it reaches the coverage target. [Kernel-weighted calibration](kernel.md) covers its options. A practical way to decide is to run both on the same data and compare their `coverage_by_stratum.csv` and `screening_by_stratum.csv`.

## Grouping: `strat`

Without `strat`, each bus or line gets one set of bounds learned from all calibration scenarios. With `strat`, the scenarios are split into groups (strata) and bounds are fitted per group, so the guarantee of `element` holds **within each group**, not only on average.

| `strat` | Groups scenarios by | Use it when |
| --- | --- | --- |
| *(none)* | nothing | the model's errors are similar across scenarios, or data is scarce |
| `n_outages` | number of out-of-service branches and generators (0, 1, 2, ...) | **N-k screening**: errors grow with the number of outages, and each contingency level should get its own bounds |
| `outage_type` | kind of outage: `none`, `branch_only`, `gen_only`, `mixed` | errors differ between branch and generator outages more than between counts |
| `system_load_bin` | percentile bins of total active load, with `system_load_n_bins` bins | errors grow with system stress, e.g. near peak load |
| `pred_bins` | percentile bins of the predicted value itself, with `pred_n_bins` bins | errors depend on the operating point of each bus or line, e.g. heavily loaded lines; `element` only |

Bin edges for `system_load_bin` and `pred_bins` are learned from the calibration set during `compute` and stored in the artifact, so test scenarios are assigned with the same edges.

!!! warning "Small strata"
    Each stratum is calibrated on its own scenarios only. A bus or line with fewer than $\lceil 2/\alpha \rceil - 1$ calibration samples in a stratum cannot get a finite bound from that stratum alone (see [Sparse strata](#sparse-strata-sparse_strata)), and one with a few dozen gets wide, noisy bounds. Check the number of calibration scenarios per stratum (the `n_scenarios` column of `coverage/coverage_by_stratum.csv` reports it on the test side) and merge rare levels, for example by generating fewer N-3 contingencies or by not stratifying.

    What counts is the number of samples per bus or line, which can be much smaller than the number of scenarios. In a power flow, the voltage magnitude of a generator bus is only predicted in the scenarios where its generator is out of service (otherwise it is a set-point, see [Known values](results.md#known-values-are-not-counted)). In the N-1 stratum that is only the scenarios where that generator is the single outage, often too few for a finite bound. `compute` lists the strata affected and how many buses or lines each.

### Sparse strata: `sparse_strata`

A finite bound at level $\alpha/2$ needs at least $\lceil 2/\alpha \rceil - 1$ valid calibration residuals of that bus or line in its stratum:

| `alpha` | 0.2 | 0.1 | 0.05 | 0.01 |
| --- | --- | --- | --- | --- |
| minimum samples | 9 | 19 | 39 | 199 |

Valid residuals exclude [known values](results.md#known-values-are-not-counted) and masked entries, such as an out-of-service branch. A test stratum that the calibration data does not contain at all (for example N-6 when calibration stops at N-5) has none.

With fewer, the element gets its quantile over all calibration scenarios in that stratum (`sparse_strata: pooled`, the default). The bounds are finite, but they lose the guarantee within the stratum: the pooled quantile is calibrated on all strata together, and a rare stratum can have larger errors than the average. Nothing is silent: `compute` lists the strata concerned, the calibrated predictions mark the rows in a `<variable>_pooled` column, and the coverage report counts them in `n_pooled`. Other elements and strata are unchanged. With `cp_type: kernel`, the same rule applies to the stratum quantile the kernel falls back to.

`sparse_strata: infinite` gives those elements infinite bounds instead. That keeps the per-stratum guarantee, and `screen` flags every such case, but the bounds carry no information. Bounds stay infinite in either mode when an element has fewer than $\lceil 2/\alpha \rceil - 1$ valid residuals over all calibration scenarios together.

```yaml
strat: n_outages
sparse_strata: infinite   # default: pooled
```

`strat` also works with `cp_type: kernel`: the kernel then localizes *within* each stratum, so an N-2 test scenario is only compared with N-2 calibration scenarios. `pred_bins` is the exception; it assigns groups per bus or line, while the kernel compares whole scenarios.

## Variables

- `Vm` and `Va` are calibrated from the predictions directly.
- `line_loading` is derived: trustlayer computes the branch flows from the predicted voltages and the branch parameters, divides by the rating `rate_a`, and calibrates the result. It needs the `datakit` extra.
- You can calibrate several at once. Voltage and line results are written to separate files, because one is per bus and the other per branch.

Only `Vm` and `line_loading` have operating limits to screen against. If you calibrate `Va` too, restrict screening with `commands.screen.variables: [Vm, line_loading]`.

## Summary

| Goal | Setup |
| --- | --- |
| First run, N-k screening | `element`, `strat: n_outages`, `alpha: 0.1` |
| A formal guarantee is required | `element`, with the stratification that matches how you will use the bounds |
| Narrower bounds on easy scenarios | `kernel` with `strat: n_outages`, `bandwidth: n_eff`, and `n_eff_target: auto` to choose the bandwidth from the calibration data |
| Errors depend on the operating point | `element`, `strat: pred_bins` |
| Very few calibration scenarios | `element` without `strat`, or a larger `alpha` |
