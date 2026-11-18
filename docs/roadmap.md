# Roadmap

What we plan to add next. These are directions, not commitments or dates. To ask for one of them, or to help with it, open an issue on [GitHub](https://github.com/gridfm/gridfm-trustlayer/issues). When a feature ships, it moves to the [Changelog](changelog.md) and leaves this page.

## Optimal power flow

**Bounds on generator dispatch (`Pg`, `Qg`).** In optimal power flow, the decisions are the generators' active and reactive power, and a dispatch is only usable within each unit's limits. Trustlayer would calibrate `Pg` and `Qg` per generator and screen them against `Pmin`/`Pmax` and `Qmin`/`Qmax`, read from the data rather than set by hand. GridFM's OPF predictions already carry `Pg` and `Qg`, and the per-element limits added in 0.2.0 are the basis for the screening. It needs a stable identity for each generator across scenarios.

**Reactive power at buses, and branch flows.** Bounds on the reactive power of generator buses, and on the active and reactive power flowing through each branch, next to line loading.

## Embeddings for the kernel method

**Generator embeddings.** GridFM models with generator nodes write generator embeddings (`<grid>_gen_embeddings.parquet`) next to the bus embeddings. Trustlayer would read them and average them per scenario, as it does for buses, so that the kernel can weigh calibration scenarios by their dispatch. They are the natural choice for bounds on `Pg` and `Qg`. From Python, any vector per scenario already works (`ChannelSet.from_frame(..., embeddings=...)`).

**Bus and generator embeddings together.** Both, for every variable, since the dispatch also shapes voltages and flows. The cross-validated choice of `n_eff_target` (`n_eff_target: auto`) would also choose which embeddings to use, as it can already choose the distance.

## Screening near operating limits

The coverage guarantee holds per element and stratum, not for values predicted close to a limit, where screening decisions are made. In the [demo](getting-started/demo.md), the true line loading exceeds the upper bound of `cp_type: element` 21% of the time when the prediction is within 10% of the rating, against the nominal 5% (the kernel method brings it down to 5%).

**A diagnostic in `screen`.** Per variable, how often the true value is beyond the bound on the limit's side, among values predicted near the limit, next to the nominal rate: a table and a line in `summary.md`.

**Strata by closeness to the limit.** The number of outages combined with how close each prediction is to its limit, so that bounds near the limit are fitted on errors near the limit.

## Models outside GridFM

**The command-line tool on plain tables.** Today the commands read GridFM's files; a model of your own goes through the [Python API](user-guide/own-data.md). A plain input layout (one row per scenario and element, with the prediction, the true value and optional strata) would give any model the commands, reports and figures.

**Line loading predicted directly.** Many models predict flows or loadings rather than voltages. The commands would take a predicted `line_loading` as it is, instead of always deriving it from the predicted voltages.

**A check that new scenarios resemble the calibration set.** A warning when new scenarios are far from every calibration scenario in the embedding, where the bounds rest on little data.

**Helpers.** Averaging per-element embeddings into one vector per scenario from a table, and the coverage figures of `calibrate` from Python.

## Platform

**Python 3.13 and 3.14.** The tests and the commands already pass on both; what remains is lifting the `<3.13` cap of the package and testing them in CI. gridfm-datakit and gridfm-graphkit, installed with the extras of the same name, keep their own limits.

**CI on macOS and Windows**, next to Linux.
