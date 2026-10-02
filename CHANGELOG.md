# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Before 1.0, minor versions may
include breaking changes; they are listed under **Changed**.

## [0.1.0] - Unreleased

First public release.

### Added
- Two conformal methods for continuous grid-state predictions:
  - stratified split-conformal prediction (`cp_type: element`);
  - kernel-weighted conformal prediction (`cp_type: kernel`), localized in a scenario-embedding space, with `knn` or `n_eff` bandwidth selection and `euclidean` or `cosine` distance.
- Stratification by `n_outages`, `outage_type`, `system_load_bin` and `pred_bins`.
- Direct targets (`Vm`, `Va` and other continuous task outputs) and the derived `line_loading`.
- The config-first CLI, `gridfm_trustlayer compute | calibrate | screen`, with nine example configs in `examples/config/`.
- A top-level Python API: `from gridfm_trustlayer import NaiveIntervalCalibrator, KernelIntervalCalibrator, ChannelSet, ...`. A worked example is in the README, under "Python API".
- `CITATION.cff`.
- A config builder: `gridfm_trustlayer.interactive.config_builder()`, a Jupyter form (optional extra `interactive`; notebook in `examples/notebooks/config_builder.ipynb`). It shows only the options that apply and validates the result with the CLI's own checks. The underlying `build_config()` works without Jupyter.

### Changed (relative to the internal 0.0.1 snapshot)
- **Breaking: calibration and test data must be separate datasets.** The config now takes `inputs.calibration` (read by `compute`) and `inputs.test` (read by `calibrate` and `screen`), each with its own `predictions`, `pf_node`, `pf_edge` and, for `cp_type: kernel`, `embeddings`.
  - The in-file split is removed: `calibration_fraction`, `random_seed`, `--calibration-fraction` and `--random-seed` are no longer accepted.
  - So is the in-sample mode, in which `calibrate` evaluated on the scenarios `compute` was fitted on.
  - A random split of one file hides near-duplicate scenarios on both sides, which inflates coverage and kernel gains. The in-sample mode measured coverage on the fitting data.
  - The config is rejected if both blocks point at the same file, and `calibrate` refuses test predictions identical to the calibration predictions, using a fingerprint stored in the artifact.
  - **Migration:** move the old flat `inputs` keys under `inputs.calibration` and add an `inputs.test` block for an independently generated dataset. A flat `inputs` block now fails with a message pointing here.
  - `analysis_pf_node` / `analysis_pf_edge` and per-command `commands.<verb>.inputs` overrides are removed, since each command reads its own dataset block.
- **Calibration quantiles now use the split-conformal finite-sample correction.** Each tail at level `alpha / 2` takes the `ceil((n + 1)(1 - alpha / 2))`-th smallest residual for the upper bound and the `floor((n + 1) alpha / 2)`-th for the lower bound. Version 0.0.1 used the interpolated empirical quantile, which does not carry the finite-sample coverage guarantee. Bounds are therefore slightly wider (by at most one order statistic per tail). This applies to `cp_type: element` and to the stratum fallback quantiles of `cp_type: kernel`.
- **Strata with too few calibration samples now get infinite bounds.** This covers the case of no valid residual at all, where 0.0.1 returned a zero-width interval. A finite bound at level `alpha / 2` needs at least `ceil(2 / alpha) - 1` valid residuals, e.g. 19 at `alpha = 0.1`. Analysis figures plot the finite widths; the CSV outputs keep the infinite ones.
- The package version is defined once, in `gridfm_trustlayer/_version.py`, and read by the build.

### Fixed
- The `kernel_fallback` diagnostic of `KernelIntervalCalibrator` no longer flags every test scenario when some element has no valid calibration residual in its stratum, e.g. a branch that is out of service in every scenario. Such an element still uses the fallback bound, but it no longer counts as a kernel fallback. The bounds themselves were not affected.
- Box plots in the analysis figures no longer produce `NaN` percentiles (and a `RuntimeWarning`) when some widths are infinite.
