# Installation

`gridfm-trustlayer` needs **Python 3.10, 3.11 or 3.12**. It runs on CPU only and has no PyTorch or GPU dependency.

We recommend a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## Install the package

```bash
python -m pip install gridfm-trustlayer
```

The development version, from the `main` branch:

```bash
python -m pip install "git+https://github.com/gridfm/gridfm-trustlayer"
```

Check that the command-line tool is available:

```bash
gridfm_trustlayer --help
```

## Optional extras

Some features need extra packages. Install them by adding the extra's name in brackets:

| Extra | Adds | You need it to |
| --- | --- | --- |
| `datakit` | `gridfm-datakit` | calibrate **`line_loading`**: trustlayer computes line loading with datakit's branch-flow routine. Not needed for `Vm` and `Va` alone. |
| `graphkit` | `gridfm-graphkit` and `gridfm-datakit` | the same, in an environment where you also run models with graphkit, which needs PyTorch (install it first, following graphkit's guide). |
| `interactive` | `ipywidgets` | use the [config builder](config-builder.md), a Jupyter form that writes the config file for you. |

For example, both at once:

```bash
python -m pip install "gridfm-trustlayer[datakit,interactive]"
```

!!! tip "Using the GridFM tools in one environment"
    If you also generate data with gridfm-datakit or run models with gridfm-graphkit, you can install trustlayer into the same environment. The `graphkit` extra sets the minimum versions trustlayer works with. See the [datakit](https://github.com/gridfm/gridfm-datakit) and [graphkit](https://github.com/gridfm/gridfm-graphkit) installation guides for their own requirements.

## Install for development

Clone the repository and install it in editable mode with the development and test tools:

```bash
git clone https://github.com/gridfm/gridfm-trustlayer.git
cd gridfm-trustlayer
python -m pip install -e ".[dev,test]"
pre-commit install
```

`pre-commit run --all-files` then runs the same checks as the CI: ruff, mypy and the test suite. To build this documentation locally, run `mkdocs serve` and open <http://127.0.0.1:8000>.

## Next step

Follow the [Quick start](quickstart.md) to run the full workflow on your own data.
