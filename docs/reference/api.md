# API reference

Everything below is importable from the top-level package, `from gridfm_trustlayer import ...`. The [Python API guide](../user-guide/python-api.md) shows how the pieces fit together.

## Calibrators

::: gridfm_trustlayer.NaiveIntervalCalibrator
    options:
      members: [fit, predict_intervals]

::: gridfm_trustlayer.KernelIntervalCalibrator
    options:
      members: [fit, predict_intervals, diagnostics]

## Data containers

::: gridfm_trustlayer.ChannelSet
    options:
      members: false

::: gridfm_trustlayer.PredictionSet
    options:
      members: false

::: gridfm_trustlayer.GroundTruthSet
    options:
      members: false

## Loading GridFM files

::: gridfm_trustlayer.load_predictions

::: gridfm_trustlayer.load_ground_truth

::: gridfm_trustlayer.load_scenario_embeddings

::: gridfm_trustlayer.compute_line_loading

## Artifacts

::: gridfm_trustlayer.save_interval_artifact

::: gridfm_trustlayer.load_interval_artifact

::: gridfm_trustlayer.load_interval_artifacts

## Kernel bandwidth tuning

::: gridfm_trustlayer.conformal.tuning.tune_n_eff_target

::: gridfm_trustlayer.conformal.tuning.TuningOptions
    options:
      members: false

## Config builder

::: gridfm_trustlayer.interactive.config_builder

::: gridfm_trustlayer.interactive.build_config

::: gridfm_trustlayer.interactive.validate_config

::: gridfm_trustlayer.interactive.save_config
