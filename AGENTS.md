# Notes for AI coding assistants

## Using trustlayer with a user's own model and data

Follow the guide in `gridfm_trustlayer/guide/own_data.md` (printed by `gridfm_trustlayer guide`; online at <https://gridfm.github.io/gridfm-trustlayer/user-guide/own-data/>): put the data in a long table, build the predictions and true values with `ChannelSet.from_frame`, run `check_inputs` and fix its errors, fit (`TrustLayer` for buses and lines together, and to save the result), and check the coverage on scenarios that were not used for calibration (`coverage_report`). Do not silence a `KernelFallbackWarning`: it means the scenario vector does not work for the kernel method. `examples/notebooks/own_model.ipynb` is a complete example with a model outside GridFM.

## Working on this repository

- Setup, layout and the checklist for changes are in [CONTRIBUTING.md](CONTRIBUTING.md).
- Before calling a change done, run `pre-commit run --all-files` (ruff, mypy, pytest) twice, since the first run may reformat files, and `mkdocs build --strict` when the docs change. CI also runs the tests on Python 3.10, where pip installs numpy 2.2, whose type stubs are stricter.
- Bounds must not change silently: the kernel tests compare against the full-scan definitions in `tests/conformal_reference.py`. Say in the pull request whether results change, and add user-visible changes to `CHANGELOG.md`.
- The code blocks of the guide run as a test (`tests/test_own_data.py`); update its `Output` block when their output changes.
- Scripts that call the CLI or `tune_n_eff_target` with worker processes (`n_jobs` for blocks) must start from an `if __name__ == "__main__":` block, since the processes are spawned.
