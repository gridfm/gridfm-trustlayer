# Contributing to gridfm-trustlayer

Contributions are welcome: bug reports, fixes, new conformal methods or stratifications, support for new targets or tasks, documentation and examples. For anything larger than a small fix, please open an issue first so the approach can be agreed before you start.

## What belongs here

`gridfm-trustlayer` is a library: the calibration methods, the loaders for GridFM data, the command-line interface, and their documentation. Code tied to one specific study or deployment (experiment scripts, paper figures, one-off pipelines) belongs in its own repository and can use trustlayer as a dependency.

The statistical core (`gridfm_trustlayer/conformal/`) works on plain NumPy arrays and must stay free of PyTorch, Lightning and GPU dependencies.

## Checklist

Before opening a pull request:

### 1. Set up

- [ ] Create a branch from `main`.
- [ ] Install the package with the development and test tools, and the pre-commit hooks:

  ```bash
  python -m pip install -e ".[dev,test]"
  pre-commit install
  ```

### 2. Code

- [ ] Clear, readable code; comments where the reason is not obvious from the code.
- [ ] Type hints on all function arguments and return values.
- [ ] Google-style docstrings on public functions and classes; they appear in the [API reference](https://gridfm.github.io/gridfm-trustlayer/reference/api/).
- [ ] No debug prints or temporary files left behind.
- [ ] New dependencies added to `pyproject.toml`, as an optional extra if only some users need them.

### 3. Statistical changes

If the change affects how bounds are computed:

- [ ] A test compares the result with a straightforward reference implementation (see `tests/conformal_reference.py`), or checks the coverage it is meant to deliver.
- [ ] The pull request says whether existing results change, and by how much.

### 4. Configuration

If you add or change a config key:

- [ ] Validation in `gridfm_trustlayer/cli/config.py`.
- [ ] The config builder (`gridfm_trustlayer/interactive.py`), so the Jupyter form offers it.
- [ ] The example configs in `examples/config/`, where relevant.
- [ ] [`docs/reference/configuration.md`](docs/reference/configuration.md).

### 5. Tests

- [ ] Unit tests for the new behaviour and its edge cases, on small synthetic data. Do not commit datasets or large files.
- [ ] The full suite passes:

  ```bash
  pytest
  ```

### 6. Checks and documentation

- [ ] The pre-commit hooks pass (ruff, mypy and the tests; the same checks as the CI):

  ```bash
  pre-commit run --all-files
  ```

- [ ] The documentation builds without warnings, and pages affected by the change are updated:

  ```bash
  mkdocs build --strict
  mkdocs serve        # preview at http://127.0.0.1:8000
  ```

- [ ] An entry in [`CHANGELOG.md`](CHANGELOG.md), under the unreleased version.

### 7. Pull request

- [ ] Every commit is signed off (Developer Certificate of Origin), which the DCO check requires:

  ```bash
  git commit -s -m "Describe the change"
  git rebase --signoff main   # signs off commits already on the branch
  ```

- [ ] Rebase onto the latest `main`, then open a pull request with a short description of the change and why it is needed.
- [ ] The CI checks pass. A maintainer will review the pull request.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE), the license of the project.
