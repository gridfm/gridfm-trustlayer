# Methodology

This page states the methods precisely: what is computed during `compute` and `calibrate`, and under which assumptions the coverage statements hold. For a practical introduction, start with [Choosing a setup](user-guide/choosing-a-setup.md). The method and its application to N-k screening are described in the [paper](https://arxiv.org/abs/2602.07995).

## Goal

`gridfm-trustlayer` wraps existing GridFM prediction artifacts with calibrated uncertainty sets.

The core idea is:

1. start from model predictions and matching ground truth on a calibration dataset
2. measure the calibration residuals
3. convert those residuals into conformal interval adjustments (`compute`)
4. apply the resulting intervals to a separate test dataset, where coverage is measured (`calibrate`)
5. optionally screen those intervals against operating limits

The repository is intentionally post-hoc and model-agnostic: it does not train GridFM models and it does not depend on Lightning or GPU-specific runtime in the statistical core.

## Target types

Two families of targets are supported.

### Direct prediction targets

These are channels already predicted by the model, such as:

- `Vm`
- `Va`

Trustlayer reads these directly from graphkit prediction parquets and calibrates them against matching target values or external ground truth.

### Derived physical targets

These are not direct model heads. They are computed from predictions plus topology and physical metadata.

The derived target is:

- `line_loading`

Trustlayer computes line loading through the datakit-backed branch-power path and then calibrates the derived quantity in the same conformal framework.

## Methods

Two calibration methods are implemented:

- `cp_type: element`: stratified conformal prediction (SCP)
- `cp_type: kernel`: kernel-weighted conformal prediction (KCP)

They differ in *where the quantile comes from*, not in what is calibrated. Both consume the same residuals, both support the same target families, and both compose with `strat`.

| | `element` | `kernel` |
| --- | --- | --- |
| What `compute` stores | a fixed residual-quantile table per entity (and per stratum) | the full calibration residual bank plus one embedding per calibration scenario |
| What `calibrate` does | table lookup, then add the offset to the prediction | computes distances, bandwidth, and kernel weights, then evaluates a weighted quantile per test scenario |
| Interval offset varies by | entity, and stratum if `strat` is set | entity, stratum, **and test scenario** |
| Artifact encoding | JSON | compressed binary container |
| Extra inputs | none | `embeddings` in `inputs.calibration` and `inputs.test` |

## `cp_type: element`

Characteristics:

- separate calibration per target dimension
- optional stratification by `n_outages`, where the outage count includes inactive branches and inactive generators per scenario
- optional stratification by `outage_type`, which groups scenarios into `none`, `branch_only`, `gen_only`, and `mixed`
- optional stratification by `system_load_bin`, which groups scenarios by percentile bins of total active load derived from raw bus `Pd` and exposes human-readable labels like `p20-p40 [1450, 1620)`
- optional stratification by `pred_bins`, which groups individual predictions by percentile bins of the model's own outputs and exposes variable-specific columns such as `pred_bins_vm`, `pred_bins_va`, or `pred_bins_line_loading`
- mixed bus-level and branch-level runs supported through separate target families


## `cp_type: kernel`

Kernel-weighted conformal prediction replaces the fixed quantile with one estimated per test scenario, weighting calibration residuals by how similar their scenario is to the scenario being evaluated.

Characteristics:

- one scenario representation `z(x)` per scenario, pooled from a per-entity embeddings parquet
- a choice of Euclidean or cosine distance over those representations
- an adaptive bandwidth, selected either from the `k_nn`-th nearest-neighbour distance or by solving for a target effective sample size
- a fallback to the stratum-wide quantile when the effective sample size drops below `n_eff_min`
- composes with `n_outages`, `outage_type`, and `system_load_bin` stratification; incompatible with `pred_bins`

### Scenario representation

The model's per-entity embeddings are pooled to one vector per scenario. `embedding_pooling: mean` (the default) computes

$$
z_s = \frac{1}{|E|}\sum_{e \in E} h_{s,e},
$$

where $h_{s,e}$ is the model's embedding for entity $e$ in scenario $s$. `max`, `meanmax`, and `concat` are also available.

During `compute`, trustlayer fits a per-dimension mean and standard deviation on the calibration embeddings and stores them in the artifact:

$$
\tilde{z}_s = \frac{z_s - \mu}{\sigma}.
$$

Both calibration and test embeddings are standardized with the same stored statistics. This matters because raw latent dimensions have no reason to share a scale, and a single high-variance dimension would otherwise dominate the distance.

### Distances and weights

For a test scenario with representation $\tilde{z}_\ast$ in stratum $g$, trustlayer computes squared distances to every calibration scenario **in the same stratum**:

$$
d_j^2 = \lVert \tilde{z}_\ast - \tilde{z}_j \rVert_2^2, \qquad j \in \mathcal{D}^{(g)}_{\mathrm{cal}}.
$$

Two metrics are available. `distance: euclidean` (the default) uses the standardized representations directly. `distance: cosine` projects them onto the unit sphere first, so that

$$
d_j^2 = 2 - 2\cos(\tilde{z}_\ast, \tilde{z}_j),
$$

which is the same formula evaluated on normalized inputs. Since standardization has already centered the representations, cosine compares the *direction* in which two scenarios deviate from the mean scenario and discards the magnitude of that deviation. Whether that is desirable depends on whether the magnitude of the deviation carries information about the model's error, which varies by model and dataset.

Because both metrics are functions of the covariates only, the choice affects efficiency but never validity.

Weights follow a Gaussian kernel with a per-test-point bandwidth $h^2$:

$$
u_j = \exp(-d_j^2 / h^2).
$$

### Bandwidth selection

Two modes are implemented. Both produce a per-test-point $h^2$; everything downstream is identical.

**`bandwidth: knn`.** The bandwidth is the distance to the $k_{\mathrm{nn}}$-th nearest neighbour within the stratum:

$$
h^2 = d^2_{(k_{\mathrm{nn}})}.
$$

This adapts to local density — tight in well-sampled regions, wider in sparse ones.

Its limitation is that $k_{\mathrm{nn}}$ is not portable across representations. Squared Euclidean distances concentrate as dimension grows: the nearest and the farthest calibration scenarios end up at similar distances, so the weights implied by a fixed $k_{\mathrm{nn}}$ approach uniform in wide embeddings, while the same $k_{\mathrm{nn}}$ localizes sharply in narrow ones.

**`bandwidth: n_eff` (default).** The bandwidth is instead solved, per test point, so the resulting weights reach a target effective sample size:

$$
h^2 : \quad n_{\mathrm{eff}}(h^2) = n^{\mathrm{target}}_{\mathrm{eff}}.
$$

$n_{\mathrm{eff}}$ increases monotonically from 1 (all weight on the nearest calibration scenario) to the stratum size (uniform weights), so this is a well-posed one-dimensional root find, solved by log-space bisection across a whole block of test points at once. Because $n_{\mathrm{eff}}$ is invariant to a common rescaling of the weights, the solve shifts the exponent by each row's minimum distance, which avoids underflow at small bandwidths without changing the result.

This parameterization is invariant to embedding width and stratum size. The default target is $n^{\mathrm{target}}_{\mathrm{eff}} = 600$. A stratum with fewer calibration scenarios than the target gets uniform weights (equivalent to `cp_type: element`), and `n_eff_target: auto` chooses the target from the calibration data.

Both remain hyperparameters. Selecting either by inspecting calibrated results on the set you intend to report on breaks the coverage guarantee; a disjoint tuning partition is required.

### Cross-validated choice of $n^{\mathrm{target}}_{\mathrm{eff}}$

With `n_eff_target: auto`, the target is chosen by $K$-fold cross-validation inside the calibration dataset, so no data beyond it is needed and the test set stays untouched. For each candidate $c$ and fold $f$, the kernel is fitted on the other folds and applied to fold $f$. This gives the cross-validated coverage $\widehat{\mathrm{cov}}_g(c)$ in every stratum $g$ (and overall) and the mean interval width $\bar{w}(c)$. A candidate is eligible if

$$
\widehat{\mathrm{cov}}_g(c) \;\ge\; 1-\alpha - 2\sqrt{\alpha(1-\alpha)/n_g} \quad \text{for every } g,
$$

with $n_g$ the number of calibration scenarios behind the estimate (the default tolerance; a fixed tolerance can be set instead). Among eligible candidates the narrowest, relative to the largest candidate, is chosen. If none is eligible, the weights are made uniform within each stratum, which reproduces `cp_type: element`.

Optionally, the distance (Euclidean or cosine) is chosen in the same way, as a second dimension, over a full grid or a random sample of a fixed budget. Folds keep scenarios that share a load profile together (datakit's `load_scenario_idx`). Near-duplicate scenarios split across folds would otherwise make localization look better than it is on new data. Because $n_{\mathrm{eff}}$ is an absolute effective sample size, a target chosen on folds holding $(K-1)/K$ of the data carries over to the final fit on all of it. The result is an empirical, not a formal, coverage check. Options and outputs are described in [Kernel-weighted calibration](user-guide/kernel.md#choosing-the-bandwidth-automatically).

### Effective sample size and fallback

The effective sample size measures how many calibration scenarios actually contribute:

$$
n_{\mathrm{eff}} = \frac{\left(\sum_j u_j\right)^2}{\sum_j u_j^2}.
$$

It equals the stratum size under uniform weights and falls toward `1` when weight concentrates on a single neighbour. When $n_{\mathrm{eff}} <$ `n_eff_min`, trustlayer reverts that scenario to the stratum-wide quantile, which is exactly the `element` result, and flags it in `kcp_fallback`.

Under `bandwidth: n_eff` this guard rarely binds, because the target is hit by construction; it only matters when a stratum is too small to reach the target at all.

A second, subtler reachability constraint follows from the weighted quantile below. The unobserved test residual enters as a point mass of weight 1, since the test scenario sits at distance zero from itself. For the $1-\alpha/2$ level to be attainable the calibration weights must satisfy $\sum_j u_j \ge 2/\alpha - 1$, which is 19 at $\alpha = 0.1$. Because $u_j \le 1$ implies $\sum_j u_j \le n_{\mathrm{eff}}$, an $n_{\mathrm{eff}}$ of at least 19 is necessary but not sufficient. The default `n_eff_min` of 5 therefore never binds before unreachability does — a scenario in that regime falls back through the unreachable path instead. When no calibration scenario is close in the kernel's own metric, all $u_j \ll 1$, the point mass dominates, and the method declines to extrapolate. That is the guarantee working, not a defect.

### Weighted quantile

For each entity $e$, calibration residuals in the stratum are sorted and the weighted quantile is taken with a point mass at $+\infty$ representing the unobserved test residual, following Tibshirani et al. (2019):

$$
q^+_{\ast,e} = \inf\left\{ r : \sum_{j : r_{j,e} \le r} u_j \;\ge\; \left(1 - \tfrac{\alpha}{2}\right)\left(\textstyle\sum_j u_j + 1\right) \right\}.
$$

The `+1` in the denominator is that atom. Without it the estimator would always return a finite quantile and lose the finite-sample guarantee; with it, a scenario whose weighted mass cannot reach the target level falls back explicitly instead.

The lower bound uses the mirrored tail, accumulating weight from the largest residual downward, so its atom sits at $-\infty$ — the conservative direction for a lower bound:

$$
q^-_{\ast,e} = \sup\left\{ r : \sum_{j : r_{j,e} \ge r} u_j \;\ge\; \left(1 - \tfrac{\alpha}{2}\right)\left(\textstyle\sum_j u_j + 1\right) \right\}.
$$

Masked or invalid calibration entries receive zero weight for that entity, so they never enter either sum.

The interval is then formed exactly as in the `element` case, but with test-scenario-specific offsets:

$$
\ell_{\ast,e} = \hat{y}_{\ast,e} + q^-_{\ast,e}, \qquad
u_{\ast,e} = \hat{y}_{\ast,e} + q^+_{\ast,e}.
$$

### Interaction with stratification

Kernel weighting and stratification are nested, not alternatives. With `strat: n_outages`, a test scenario with two outages draws its neighbours, its bandwidth, and its quantile only from two-outage calibration scenarios. A zero-outage calibration scenario can never receive weight, regardless of embedding distance.

This is why `pred_bins` is rejected for `cp_type: kernel`: prediction bins are assigned per element, while kernel localization is defined per scenario, so the two groupings do not nest.

### Practical behavior

The bandwidth controls how sharply the kernel localizes and is the main tuning knob:

- a small bandwidth gives low `n_eff`, strongly conditional intervals, and a tendency to overshoot the nominal marginal coverage
- a large bandwidth gives near-uniform weights, `n_eff` approaching the stratum size, and results converging to `cp_type: element`

Under `bandwidth: knn` the lever is `k_nn`; under `bandwidth: n_eff` it is `n_eff_target`.

Squared Euclidean distances concentrate as embedding width grows, which weakens localization for very wide representations. The `kcp_n_eff` column written by `calibrate` is the direct diagnostic: if it sits near the stratum size, the kernel is not localizing and the run is effectively `element`.

Because KCP flattens conditional coverage, its *marginal* coverage can sit above `1 - alpha` on skewed difficulty distributions: the easy majority is already near full coverage, so raising coverage on the hard tail lifts the average. A larger bandwidth (higher `n_eff_target` or `k_nn`) moves it back toward `element`.

## Implementation details

This section describes the math the code applies. It is intentionally limited to the quantities needed to understand the artifact and the outputs.

### Notation

For one variable such as `Vm`, `Va`, or `line_loading`:

- let $s$ index scenarios
- let $e$ index entities, either buses or branches
- let $\hat{y}_{s,e}$ be the model prediction
- let $y_{s,e}$ be the matching ground-truth value

Trustlayer calibrates each variable independently and, within a variable, each entity position independently.

### Residuals used during `compute`

For every aligned scenario-entity pair in the calibration dataset, trustlayer computes the signed residual

$$
r_{s,e} = y_{s,e} - \hat{y}_{s,e}.
$$

The sign matters. A positive residual means the model underpredicted the target, and a negative residual means it overpredicted it.

For each entity $e$, with $n_e$ valid calibration residuals sorted as $r_{(1),e} \le \dots \le r_{(n_e),e}$, trustlayer stores the split-conformal residual quantiles

$$
q^-_e = r_{(\lfloor (n_e+1)\,\alpha/2 \rfloor),\,e}, \qquad
q^+_e = r_{(\lceil (n_e+1)(1-\alpha/2) \rceil),\,e},
$$

with $q^-_e = -\infty$ if the lower rank is below 1 and $q^+_e = +\infty$ if the upper rank exceeds $n_e$. The $+1$ is the finite-sample correction of split conformal prediction: if the calibration and test residuals of an entity are exchangeable, the interval $[\hat{y} + q^-_e,\ \hat{y} + q^+_e]$ contains the true value with probability at least $1-\alpha$ (and each one-sided bound with probability at least $1-\alpha/2$), for any number of calibration samples. The plain empirical quantile does not carry this guarantee.

A finite bound therefore needs $n_e \ge \lceil 2/\alpha \rceil - 1$ valid residuals, i.e. 19 at `alpha = 0.1`. With fewer, the bound is infinite rather than a falsely narrow interval. For large $n_e$ the correction is negligible: with `alpha = 0.1` and a few thousand residuals, the bounds are within one order statistic of the empirical 5th and 95th percentiles.

The class implementing this is still named `NaiveIntervalCalibrator`, for historical reasons; it is the stratified split-conformal method of the paper.

### Stratified residual quantiles

If `strat` is enabled, trustlayer first assigns each scenario to a stratum $g_s$ and then fits separate quantiles inside each stratum:

$$
q^-_{e,g} = Q^{\mathrm{CP}}_{\alpha/2}(\{r_{s,e} : g_s = g\}), \qquad
q^+_{e,g} = Q^{\mathrm{CP}}_{1-\alpha/2}(\{r_{s,e} : g_s = g\}),
$$

where $Q^{\mathrm{CP}}$ denotes the corrected order statistic above, computed over the $n_{e,g}$ residuals of the stratum. The guarantee then holds within each stratum (Mondrian conformal prediction), provided calibration and test scenarios are exchangeable within it.

The supported strata are:

- `n_outages`
- `outage_type`
- `system_load_bin`
- `pred_bins`

If no stratification is used, the artifact only stores the global per-entity quantiles $q^-_e$ and $q^+_e$.

### How `system_load_bin` is constructed

When `strat: system_load_bin` is requested, trustlayer derives one total active-load value per scenario:

$$
L_s = \sum_b Pd_{s,b},
$$

where the sum runs over buses in the ground-truth input.

If the requested number of bins is $B$, the code builds percentile cut points from the calibration scenarios only:

$$
E_j = Q_{j/B}(\{L_s\}_s), \qquad j = 1, \dots, B-1.
$$

Those interior cut points are stored in the `.trustlayer` artifact. Each scenario is then assigned with a right-sided search over the stored edges:

$$
g_s = \mathrm{searchsorted}(E, L_s).
$$

In practical terms:

- `compute` learns the load-bin edges from the calibration dataset
- `calibrate` reuses those same edges to assign new scenarios to bins
- the human-readable labels such as `p20-p40 [1450, 1620)` are derived from those stored percentile edges

### How `pred_bins` is constructed

When `strat: pred_bins` is requested, trustlayer pools all calibration predictions for each calibrated variable and learns percentile cut points directly from those values.

For a variable $v$, let $\hat{y}^{(v)}_{s,e}$ denote one predicted value at scenario $s$ and entity $e$. If the requested number of bins is $B$, trustlayer learns separate cut points for each variable from the calibration predictions:

$$
E^{(v)}_j = Q_{j/B}(\{\hat{y}^{(v)}_{s,e}\}_{s,e}), \qquad j = 1, \dots, B-1.
$$

Those variable-specific cut points are stored in the `.trustlayer` artifact. During both fitting and application, each prediction value is assigned to a bin using the stored edges for its variable.

In practical terms:

- `compute` learns one edge set per calibrated variable from all calibration predictions for that variable
- residual quantiles are fitted separately for each entity and each prediction bin
- `calibrate` reapplies those same variable-specific edge sets to each prediction row
- the calibrated parquet and analysis outputs expose variable-specific columns such as `pred_bins_vm`, `pred_bins_va`, and `pred_bins_line_loading`

### Interval construction during `calibrate`

For a new prediction $\hat{y}_{s,e}$, trustlayer adds the saved residual quantiles back to the prediction.

Without stratification:

$$
\ell_{s,e} = \hat{y}_{s,e} + q^-_e, \qquad
u_{s,e} = \hat{y}_{s,e} + q^+_e.
$$

With stratification:

$$
\ell_{s,e} = \hat{y}_{s,e} + q^-_{e,g_s}, \qquad
u_{s,e} = \hat{y}_{s,e} + q^+_{e,g_s}.
$$

The stored output columns are therefore prediction intervals of the form:

$$
[\ell_{s,e}, u_{s,e}].
$$

For invalid or masked entries, the implementation does not expand an interval and leaves the lower and upper endpoints equal to the original prediction.

### Analysis quantities written to CSV

The trust-analysis outputs are derived directly from those interval endpoints.

For each scenario-entity pair, the interval width is $u_{s,e} - \ell_{s,e}$.

Coverage is recorded as a binary indicator: it is `1` when $\ell_{s,e} \le y_{s,e} \le u_{s,e}$ and `0` otherwise.

The entity-level `empirical_coverage` reported in the analysis CSV is the average of that indicator over scenarios for each bus or branch, and the reported width summaries are the mean, min, and max interval widths over scenarios.

The analysis figures are built from those same entity-level rows. In particular, the width-density plots use the distribution of `width_mean` across entities within each stratum, so they visualize per-entity average interval widths rather than raw per-sample interval-width values.

The reliability CSV outputs (`trust_reliability_volt.csv` and `trust_reliability_line.csv`) aggregate those analysis rows by variable and stratum, reporting expected coverage (`1 - alpha`), weighted empirical coverage, coverage gap, and width aggregates.

## Workflow

### `compute`

`compute` fits interval calibrators from aligned prediction and ground-truth artifacts.

Inputs:

- prediction parquet
- bus ground-truth parquet
- branch ground-truth parquet when derived loading is requested
- scenario embeddings parquet when `cp_type: kernel`
- conformal settings such as `alpha`, `cp_type`, `strat`, and `variables`

Outputs:

- one `.trustlayer` artifact containing one or more fitted calibrators

For `cp_type: element` the artifact holds a quantile table. For `cp_type: kernel` it holds the calibration residual bank, the standardized calibration embeddings, the standardization statistics, the per-stratum fallback quantiles, and the kernel settings. `compute` also precomputes the per-entity residual sort order so the weighted quantile at application time is a permute-and-accumulate pass rather than a sort per test point.

### `calibrate`

`calibrate` applies the fitted calibrators to predictions.

Outputs:

- calibrated bus parquet for voltage variables
- calibrated branch parquet for `line_loading`
- trust-analysis CSVs
- trust-summary CSVs
- figures
- for `cp_type: kernel`, the per-scenario diagnostics `kcp_n_eff`, `kcp_bandwidth_sq`, and `kcp_fallback`

For `cp_type: kernel` this is also where the localization happens: the test embeddings are loaded and pooled, distances to the stored bank are computed within each stratum, and the weighted quantile is evaluated per test scenario.

This step is intentionally split by target family because bus-level and branch-level results live on different entity axes.

### `screen`

`screen` evaluates calibrated intervals against operating bounds defined in the YAML config.

Example bounds:

- `Vm` in `[0.95, 1.05]`
- `line_loading` <= `1.0`

Outputs:

- detail CSVs with row-level predicted and true violation flags
- entity-level summaries by bus or branch
- aggregate summaries by bus type or `rate_a`
- screening figures

## Screening interpretation

`screen` uses a possible-violation interpretation.

For an upper limit:

- the interval raises an alarm when `pi_upper` crosses the limit

For a lower limit:

- the interval raises an alarm when `pi_lower` crosses the limit

This is conservative by design. It answers:

- could this operating point violate the bound, given the calibrated uncertainty?

It does not require the true target to violate the bound.

That distinction matters especially for OPF data:

- PF data often contains realized violations, so precision/recall style screening is naturally informative
- OPF data is often feasible by construction, so the same screen behaves more like a near-limit risk alarm

## Task semantics

Trustlayer does not assume that all prediction parquets mean the same thing.

Task-aware behavior includes:

- PF-specific clamping/correction rules applied behind task adapters
- OPF support for separate bus and generator prediction artifacts
- preservation of task-specific direct channels instead of forcing a PF-only interpretation

This is why the repository keeps schema handling inside the I/O layer rather than in the conformal core.

## References

- V. Vovk, A. Gammerman, and G. Shafer. *Algorithmic Learning in a Random World*. Springer, 2005.
- R. J. Tibshirani, R. F. Barber, E. Candes, and A. Ramdas. Conformal prediction under covariate shift. *Advances in Neural Information Processing Systems*, 32, 2019. (Weighted conformal inference, including the point mass used by the kernel quantile.)
