# Demo: N-k contingency screening on the IEEE 14-bus grid

[`demo.ipynb`](demo.ipynb) runs the whole trustlayer workflow on real model
predictions in under a minute: it fits the layer, checks its coverage, shows
why the bounds are fitted per number of outages (against one set of bounds), screens contingencies into
secure, uncertain and insecure, and applies it to new scenarios without their
AC solution. It then compares kernel-weighted bounds, shows how `alpha` trades
AC power flows for missed overloads, and where the coverage guarantee is
weaker: for lines loaded near their rating. The notebook is saved with its outputs, so
it can be read without running it.

```bash
pip install "gridfm-trustlayer[datakit]" jupyterlab
cd examples/demo
jupyter lab demo.ipynb
```

The notebook writes its results to `outputs/` (not tracked by git).

## The data

Contingency scenarios of the IEEE 14-bus grid (`case14_ieee` from pglib), with
one to five outages of branches or generators, as in the paper's IEEE-118
study. Three separately generated sets, in `data/` (32 MB):

| Set | Scenarios | Used by | Contents |
| --- | --- | --- | --- |
| `calibration` | 9,723 | `compute` | predictions, AC solution, embeddings |
| `test` | 4,848 | `calibrate`, `screen` | predictions, AC solution, embeddings |
| `new` | 294 | `apply` | predictions with only the set-points, scenario description without solved values |

The predictions come from the released GENCO small checkpoint
[`gridfm/genco-pf-datakit`](https://huggingface.co/gridfm/genco-pf-datakit)
`case14_ieee/small/seed0`, used as released; it was trained on N-1
contingencies. Each set holds the predictions (`case14_ieee_predictions.parquet`),
datakit's bus, branch and generator tables (`raw/`), reduced to the columns
trustlayer reads, and the model's embeddings averaged per scenario
(`case14_ieee_embeddings.parquet`).

The scenarios were generated with gridfm-datakit 1.1.0 (commit `c045442`), with
the settings of the GENCO training data (`gridfm/pf_small_case14_ieee`) except
for the contingencies: one to five outages of branches and generators, each
count equally likely, one topology per load scenario, and a different seed per
set (3001, 3002, 3003). The predictions and embeddings come from
gridfm-graphkit 0.9.0.3 (commit `f70230c`) `predict --get_embeddings`, on CPU.

The 14-bus grid keeps the demo small enough to run on a laptop. It shows how
the layer works and what it reports; the numbers are specific to this grid,
model and data, not a measure of performance on larger systems.
