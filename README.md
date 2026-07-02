# gridfm-trustlayer

Conformal trustworthiness layer for GridFM foundation models.

This repository implements a post-hoc conformal prediction layer for continuous grid-state predictions and produces statistically valid upper bounds for downstream contingency screening.

> [!WARNING]
> This project is under active development.
> Do not rely on it yet for production use, operational decisions, published benchmarks, or stable downstream integrations.
> Interfaces, file formats, and implementation details may change without notice while the package is being built out and validated. ⚠️

## Scope

- Consume prediction parquet artifacts produced by GridFM model workflows.
- Use the established GridFM validation pipeline as the source of truth for line-loading computation.
- Keep the statistical core independent from Lightning, PyG, and GPU-bound dependencies.

## Installation

Create or activate a Python 3.10-3.12 environment, then install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"
```

## Reference

- Paper: https://arxiv.org/pdf/2602.07995
