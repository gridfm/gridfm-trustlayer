# Building the config

All three commands read one YAML config. Which keys it needs depends on your earlier choices: kernel options only matter for `cp_type: kernel`, bin counts only for binned stratifications, screening bounds only if you screen. The **config builder** is a Jupyter form that asks only the questions that apply, explains each option in one line, and checks the result with the same validation as the CLI before saving it.

## Open the form

Install the `interactive` extra, and Jupyter if you don't have it:

```bash
python -m pip install "gridfm-trustlayer[interactive] @ git+https://github.com/gridfm/gridfm-trustlayer"
python -m pip install jupyterlab
```

Then either open the notebook shipped with the repository, [`examples/notebooks/config_builder.ipynb`](https://github.com/gridfm/gridfm-trustlayer/blob/main/examples/notebooks/config_builder.ipynb), or run this in any notebook cell:

```python
from gridfm_trustlayer.interactive import config_builder

config_builder()
```

Keep `config_builder()` as the last line of the cell: the cell's output is the form.

??? question "My virtual environment does not appear in the notebook's kernel list"
    Jupyter (or the VS Code notebook editor) only lists environments that are registered as kernels. Register the one where trustlayer is installed, then reload the notebook and pick it:

    ```bash
    python -m pip install ipykernel
    python -m ipykernel install --user --name trustlayer --display-name "Python (trustlayer)"
    ```

## Fill it in

The form is organized top to bottom in the order of the decisions:

1. **Method.** Task (`PowerFlow` or `OptimalPowerFlow`), `cp_type` (`element` or `kernel`), `alpha`, and the variables to calibrate.
2. **Stratification.** None, or one of `n_outages`, `outage_type`, `system_load_bin`, `pred_bins`. The number of bins appears only for the binned ones, and `pred_bins` is not offered with `kernel`.
3. **Kernel options** (only for `kernel`). Pooling, distance, bandwidth mode and its value, and the optional cross-validated tuning of the bandwidth with its search options.
4. **Data.** The predictions, bus and branch ground-truth files of the calibration set and of the test set, plus the embeddings files for `kernel` and the generator predictions for `OptimalPowerFlow`.
5. **Screening.** Turn it off to only produce intervals, or set the limits for `Vm` and `line_loading`.
6. **Output directory and file name.**

A live preview on the right shows the YAML as you go. A status line under the output fields says whether the config is valid; while it is not, it names the problem, for example a missing file path, and **Save config** stays disabled. [Choosing a setup](../user-guide/choosing-a-setup.md) helps with the method and stratification choices.

## Without Jupyter

The form is a thin layer over plain functions, which you can call from a script:

```python
from gridfm_trustlayer.interactive import build_config, save_config, to_yaml

config = build_config(
    calibration={
        "predictions": "predictions/calibration/case118_ieee_predictions.parquet",
        "pf_node": "data_calibration/case118_ieee/raw/bus_data.parquet",
        "pf_edge": "data_calibration/case118_ieee/raw/branch_data.parquet",
    },
    test={
        "predictions": "predictions/test/case118_ieee_predictions.parquet",
        "pf_node": "data_test/case118_ieee/raw/bus_data.parquet",
        "pf_edge": "data_test/case118_ieee/raw/branch_data.parquet",
    },
    cp_type="element",
    strat="n_outages",
    alpha=0.1,
    variables=["Vm", "Va", "line_loading"],
    screen_bounds={
        "Vm": {"lower": 0.95, "upper": 1.05},
        "line_loading": {"upper": 1.0},
    },
)
print(to_yaml(config))  # preview
save_config(config, "trust_config.yaml")  # validates, then writes
```

`build_config` drops every option that does not apply to the choices, and fills in output paths under `trustlayer_outputs/` (change the directory with `output_dir=`). It also restricts screening to the variables that have bounds. `validate_config(config)` runs the checks without writing anything.

## Or start from an example

The repository ships ten complete configs in [`examples/config/`](https://github.com/gridfm/gridfm-trustlayer/tree/main/examples/config), one per combination of options, each commented line by line. Copy the closest one and replace the `path/to/...` placeholders. Every key is documented in the [configuration reference](../reference/configuration.md).
