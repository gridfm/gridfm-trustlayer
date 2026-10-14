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
  - kernel-weighted conformal prediction (`cp_type: kernel`), localized in a scenario-embedding space, with `n_eff` (default, target 600) or `knn` bandwidth selection and `euclidean` or `cosine` distance.
- Stratification by `n_outages`, `outage_type`, `system_load_bin` and `pred_bins`.
- Direct targets (`Vm`, `Va` and other continuous task outputs) and the derived `line_loading`.
- The config-first CLI, `gridfm_trustlayer compute | calibrate | screen`, with ten example configs in `examples/config/`.
- A top-level Python API: `from gridfm_trustlayer import NaiveIntervalCalibrator, KernelIntervalCalibrator, ChannelSet, ...`. A worked example is in the README, under "Python API".
- `CITATION.cff`.
- Cross-validated choice of the kernel bandwidth: `n_eff_target: auto` (with `cp_type: kernel`, `bandwidth: n_eff`) makes `compute` pick the target, and optionally the distance (`distances: [euclidean, cosine]`), by K-fold cross-validation on the calibration set only, over a grid or a random sample with a fixed `budget`, grouped by load profile when datakit's `load_scenario_idx` is available. The rule is coverage first, then the narrowest intervals, with uniform weights as the fallback. Off by default; options under `tuning:`. The choice is recorded in the artifact and reported in `<artifact>_kernel_tuning*.csv`. Also available from Python as `gridfm_trustlayer.conformal.tuning.tune_n_eff_target`.
- A config builder: `gridfm_trustlayer.interactive.config_builder()`, a Jupyter form (optional extra `interactive`; notebook in `examples/notebooks/config_builder.ipynb`). It shows only the options that apply and validates the result with the CLI's own checks. The underlying `build_config()` works without Jupyter.
- A documentation site, <https://gridfm.github.io/gridfm-trustlayer/>, built with MkDocs from `docs/` and published to GitHub Pages on every push to `main` (`.github/workflows/docs.yml`): getting-started guides, a user guide, references for the CLI, the configuration keys, the data schemas and the Python API, and the methodology.

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
- Kernel defaults: `bandwidth: n_eff` with `n_eff_target: 600` (0.0.1: `bandwidth: knn`, `k_nn: 20`). Artifacts record their bandwidth, so existing artifacts are read unchanged.
- The package version is defined once, in `gridfm_trustlayer/_version.py`, and read by the build.
- **Large outputs are partitioned parquet datasets**, in the layout of gridfm-datakit: the calibrated predictions (`*_with_intervals.parquet`) and the screening detail are directories with one `scenario_partition=k/` folder per 200 scenarios, zstd-compressed. `pd.read_parquet(path)` reads them as one table. The screening detail is now parquet (`trust_screen_detail_*.parquet/`) instead of CSV, about eight times smaller. Inputs may be single files or partitioned directories.
- **`calibrate` and `screen` process the test set block by block** (about 500,000 rows at a time): each block is read with only the needed columns and scenarios, calibrated, screened and written, and the reports are built from per-entity sums, so memory no longer grows in proportion to the test set. Reports and rows are identical whatever the block size (tested). On a 20,000-scenario IEEE-118 test set, peak memory fell from 6.3 to 2.8 GB for `calibrate` and from 3.1 to 1.8 GB for `screen`, which also runs about seven times faster. The loaders read only the columns they need and no longer re-sort the data for every column, which also lowers the memory of `compute`.
- The calibrated voltage predictions include `Vm_target` / `Va_target` from the ground truth when the predictions file has none, so `screen` never needs to reload it.
- The bandwidth solve of `bandwidth: n_eff` stops each test scenario as soon as it converges, instead of iterating all scenarios of a batch until the last one converges. A scenario's bandwidth no longer depends on which other scenarios are processed with it; values change within the solver tolerance (0.1% of the target).

### Fixed
- `screen` now finds the calibrated parquets inside a directory. Before, it failed when `outputs.calibrated_predictions` was a directory (a mixed voltage and line-loading run), reading the whole directory as one parquet. It also failed with the default output paths whenever fewer variables were screened than calibrated, e.g. screening `Vm` after calibrating `Vm` and `Va`.
- In a run with one target family, `calibrate` now writes to `outputs.calibrated_predictions_volt` / `_line` when set, which is where `screen` reads from.
- The `kernel_fallback` diagnostic of `KernelIntervalCalibrator` no longer flags every test scenario when some element has no valid calibration residual in its stratum, e.g. a branch that is out of service in every scenario. Such an element still uses the fallback bound, but it no longer counts as a kernel fallback. The bounds themselves were not affected.
- Box plots in the analysis figures no longer produce `NaN` percentiles (and a `RuntimeWarning`) when some widths are infinite.
