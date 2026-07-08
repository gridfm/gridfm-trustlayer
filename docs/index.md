# gridfm-trustlayer

`gridfm-trustlayer` is a conformal trustworthiness layer for GridFM foundation models.

The package provides:

- parquet-based data ingestion for predictions and ground truth
- calibration support for direct prediction targets such as `Vm`, `Va`, and other task-specific continuous outputs
- calibration support for derived physical metrics such as line loading
- a config-first CLI for calibration and screening workflows

## Current workflow

The current public workflow is:

1. `compute` to fit interval calibrators and write a `.trustlayer` artifact
2. `calibrate` to apply interval columns and write calibrated parquet outputs plus analyses
3. `screen` to compare intervals against operating limits and write aggregate summaries

All verbs require `--config`, and CLI flags act as overrides of the YAML.

## Current supported targets

- direct voltage variables such as `Vm` and `Va`
- derived branch loading via `line_loading`
- mixed runs that combine voltage and line-loading targets in one config

## Documentation map

- [methodology.md](methodology.md): current trustlayer method, workflow, implementation details, and screening interpretation
- [cli.md](cli.md): CLI verbs, options, and workflow behavior
- [configuration.md](configuration.md): YAML config structure, merge rules, and mixed-run path rules
- [schemas.md](schemas.md): current prediction, ground-truth, and produced-artifact schema reference

## Current implementation notes

- `cp_type: element` is the canonical name for the current residual-quantile interval workflow
- `strat: n_outages`, `strat: outage_type`, and `strat: system_load_bin` are the current supported residual-stratification modes
- `system_load_bin` labels are exported as percentile ranges with cut-point hints instead of opaque integer-only names
- screening currently uses a possible-violation interpretation based on interval endpoints crossing the configured bounds
