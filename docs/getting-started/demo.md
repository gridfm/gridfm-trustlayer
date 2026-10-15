# Demo notebook

The [demo notebook](https://github.com/gridfm/gridfm-trustlayer/blob/main/examples/demo/demo.ipynb) runs the whole workflow on real model predictions, with the data included, in under a minute. It is saved with its outputs, so you can read it on GitHub without running it.

**The setup.** Contingency scenarios of the IEEE 14-bus grid with one to five outages of branches or generators, generated with gridfm-datakit, and predictions from the released GridFM checkpoint [`gridfm/genco-pf-datakit`](https://huggingface.co/gridfm/genco-pf-datakit) (`case14_ieee/small/seed0`), which was trained on N-1 contingencies. Three separately generated sets: 9,723 calibration scenarios, 4,848 test scenarios, and 294 new scenarios with their AC solution removed.

**Scope.** The 14-bus grid keeps the demo small enough to run on a laptop. It shows how the layer works and what it reports; the numbers are specific to this grid, model and data, not a measure of performance on larger systems.

**What it shows.**

1. **The problem.** The model's line-loading error grows from 0.00007 at N-1 to 0.05 at N-5, and screening on its point prediction misses real overloads.
2. **Fit and validate.** `compute` fits bounds per number of outages; `calibrate` shows that they cover close to 90% of the true values in every stratum, while widening with the number of outages.
3. **Why stratify.** With one set of bounds for all contingencies, coverage is 69% at N-5, far below the target, and 100% at N-1, where the bounds are 250 times wider than needed: the 90% would hold only on average. Strata should follow what the errors depend on, here the number of outages.
4. **Screening.** `screen` sorts the test scenarios into secure, uncertain and insecure, and checks them against the AC solution:

    | | Secure (wrong) | Uncertain | Insecure (wrong) |
    | --- | --- | --- | --- |
    | Intervals | 3,710 (7) | 473 | 665 (14) |
    | Point prediction | 4,135 (74) | – | 713 (32) |

    The point prediction calls 74 insecure scenarios secure; the intervals 7. Only the 10% of uncertain scenarios need an AC power flow. A figure shows every line overload the point prediction misses (110 of 806) with its interval: the bounds catch 85 of them. Most misses are at N-4 and N-5, where the model is weakest.
5. **Operating.** `apply` gives the same statuses for the 294 new scenarios, without their solution: 221 secure, 25 uncertain, 48 insecure.

**Going further.** Sections 1 to 5 are the whole workflow; sections 6 to 8 look closer at the results:

- **Kernel-weighted bounds**, built on the model's embeddings. Cross-validation chooses how local they are: each test scenario weighs the calibration scenarios of its contingency level (1,212 to 2,530 per level) by similarity, with an effective sample size of 400. A figure shows how they adapt to each scenario: wide where the model tends to be wrong, narrow elsewhere. Screening with them catches 103 of the 110 overloads the point prediction misses (`element`: 85), with fewer scenarios sent to an AC power flow (330 against 473).
- **Choosing `alpha`.** A sweep from 0.3 to 0.02 for both methods, plotting the scenarios that need an AC power flow against the overloads missed: the trade-off `alpha` sets.
- **Where the guarantee is weaker.** Coverage holds per line and contingency level, but lines predicted within 10% of their rating exceed the upper bound 21% of the time with `element`, against the nominal 5%. Strata by predicted loading (`strat: pred_bins`) barely help (18%), and without strata by outages they cover only 72% at N-5; the kernel brings it down to 5%.

**Run it.**

```bash
pip install "gridfm-trustlayer[datakit]" jupyterlab
git clone https://github.com/gridfm/gridfm-trustlayer.git
cd gridfm-trustlayer/examples/demo
jupyter lab demo.ipynb
```

How the data was generated (datakit settings, tool versions) is described in the [demo's README](https://github.com/gridfm/gridfm-trustlayer/blob/main/examples/demo/README.md). To do the same with your own grid and model, follow the [Quick start](quickstart.md).
