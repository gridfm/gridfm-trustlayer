"""Build a trustlayer YAML config from choices instead of writing it by hand.

Two layers:

* ``build_config`` / ``validate_config`` / ``to_yaml``: plain functions holding all
  the logic (which keys apply to which choices). Usable from scripts and tests.
* ``config_builder()``: an ipywidgets form for Jupyter that calls them. It needs
  the optional extra: ``pip install "gridfm-trustlayer[interactive]"``.

Every config is checked with the CLI's own validation before it can be saved, so
the builder cannot produce a config that ``compute``, ``calibrate`` or ``screen``
would reject.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from gridfm_trustlayer.cli.config import (
    resolve_command_config,
    validate_command_config,
    validate_config_schema,
)
from gridfm_trustlayer.conformal.kernel import (
    DEFAULT_BANDWIDTH,
    DEFAULT_DISTANCE,
    DEFAULT_K_NN,
    DEFAULT_N_EFF_MIN,
    DEFAULT_N_EFF_TARGET,
    SUPPORTED_BANDWIDTHS,
    SUPPORTED_DISTANCES,
)
from gridfm_trustlayer.conformal.tuning import (
    DEFAULT_TUNING_BUDGET,
    DEFAULT_TUNING_FOLDS,
    DEFAULT_TUNING_GRID,
    TuningOptions,
)
from gridfm_trustlayer.io.embeddings import DEFAULT_POOLING, POOLING_REGISTRY

TASK_NAMES = ("PowerFlow", "OptimalPowerFlow")
CP_TYPES = ("element", "kernel")
STRATS = ("n_outages", "outage_type", "system_load_bin", "pred_bins")
VARIABLES = ("Vm", "Va", "line_loading")
# Va has no natural operating limit, so it is calibrated but never screened.
SCREENABLE = ("Vm", "line_loading")
_DATASET_KEYS = (
    "predictions",
    "generator_predictions",
    "embeddings",
    "pf_node",
    "pf_edge",
)
_REQUIRED_DATASET_KEYS = ("predictions", "pf_node", "pf_edge")


# ------------------------------------------------------------------ pure logic
def build_config(
    *,
    calibration: Mapping[str, str | None],
    test: Mapping[str, str | None],
    task_name: str = "PowerFlow",
    cp_type: str = "element",
    alpha: float = 0.1,
    variables: Sequence[str] = ("Vm", "Va"),
    strat: str | None = None,
    system_load_n_bins: int | None = None,
    pred_n_bins: int | None = None,
    sparse_strata: str = "pooled",
    embedding_pooling: str = DEFAULT_POOLING,
    distance: str = DEFAULT_DISTANCE,
    bandwidth: str = DEFAULT_BANDWIDTH,
    k_nn: int = DEFAULT_K_NN,
    n_eff_target: float | None = None,
    n_eff_min: float = DEFAULT_N_EFF_MIN,
    tune_n_eff_target: bool = False,
    tuning: Mapping[str, Any] | None = None,
    screen_bounds: Mapping[str, Mapping[str, float]] | None = None,
    apply: Mapping[str, str | None] | None = None,
    output_dir: str = "trustlayer_outputs",
) -> dict[str, Any]:
    """Assemble a config dict, keeping only the keys that apply to the choices.

    Options that do not apply are dropped rather than written, e.g. kernel knobs
    for ``cp_type="element"``, ``k_nn`` under ``bandwidth="n_eff"``, embeddings for
    an element run, or bounds for variables that are not calibrated.

    ``tune_n_eff_target=True`` (kernel with ``bandwidth="n_eff"`` only) writes
    ``n_eff_target: auto`` and a ``tuning:`` block, so ``compute`` chooses the target
    by cross-validation on the calibration set; ``tuning`` overrides its defaults
    (``folds``, ``grid``, ``distances``, ``search``, ``budget``, ``n_eff_range``,
    ``group_by``, ``seed``, ``coverage_tolerance``).

    ``screen_bounds=None`` (or bounds only for unselected variables) gives a config
    without a screening step: ``compute`` and ``calibrate`` produce the prediction
    intervals, and no ``screen_bounds``, ``screen_dir`` or ``screen`` command is set.

    ``sparse_strata="infinite"`` (with ``strat``) gives elements in a stratum too
    small for a finite bound infinite bounds instead of their quantile over all
    calibration scenarios (the default, ``"pooled"``).

    ``apply`` (optional) describes new scenarios without a known solution, for the
    ``apply`` command: their predictions, and the bus and branch files describing
    them when the layer is stratified by outages or load or calibrates
    ``line_loading``.
    """

    kernel = cp_type == "kernel"
    variables = [name for name in VARIABLES if name in variables]
    config: dict[str, Any] = {
        "version": 1,
        "task_name": task_name,
        "cp_type": cp_type,
        "alpha": float(alpha),
    }
    if strat:
        config["strat"] = strat
        if strat == "system_load_bin":
            config["system_load_n_bins"] = system_load_n_bins
        if strat == "pred_bins":
            config["pred_n_bins"] = pred_n_bins
        if sparse_strata != "pooled":
            config["sparse_strata"] = sparse_strata
    if kernel:
        config["embedding_pooling"] = embedding_pooling
        config["distance"] = distance
        config["bandwidth"] = bandwidth
        if bandwidth == "knn":
            config["k_nn"] = int(k_nn)
        elif tune_n_eff_target:
            options = TuningOptions.from_config(tuning)
            config["n_eff_target"] = "auto"
            block: dict[str, Any] = {"folds": options.folds, "search": options.search}
            if options.search == "grid":
                block["grid"] = list(options.grid)
            else:
                block["budget"] = options.budget
                block["n_eff_range"] = list(options.resolved_range())
            if options.distances:
                block["distances"] = list(options.distances)
            block.update(
                group_by=options.group_by,
                seed=options.seed,
                coverage_tolerance=options.coverage_tolerance,
            )
            config["tuning"] = block
        else:
            # Written out even when left at the default, so the saved config is
            # explicit and keeps its meaning if the library default changes.
            config["n_eff_target"] = float(
                DEFAULT_N_EFF_TARGET if n_eff_target is None else n_eff_target
            )
        config["n_eff_min"] = float(n_eff_min)
    config["variables"] = variables

    bounds = {
        name: dict(screen_bounds[name])
        for name in SCREENABLE
        if screen_bounds and name in screen_bounds and name in variables
    }
    if bounds:
        config["screen_bounds"] = bounds

    def dataset(values: Mapping[str, str | None]) -> dict[str, str]:
        block = {
            key: str(values[key]).strip() for key in _DATASET_KEYS if values.get(key)
        }
        if not kernel:
            block.pop("embeddings", None)
        return block

    config["inputs"] = {"calibration": dataset(calibration), "test": dataset(test)}
    new_scenarios = dataset(apply or {})
    if new_scenarios.get("predictions"):
        config["inputs"]["apply"] = new_scenarios

    out = output_dir.rstrip("/")
    names = "_".join(name.lower() for name in variables)
    outputs: dict[str, str] = {"artifact": f"{out}/{cp_type}_{names}.trustlayer"}
    families = {"line" if name == "line_loading" else "volt" for name in variables}
    if len(families) > 1:
        outputs["calibrated_predictions_volt"] = (
            f"{out}/calibrated/test_volt_with_intervals.parquet"
        )
        outputs["calibrated_predictions_line"] = (
            f"{out}/calibrated/test_line_with_intervals.parquet"
        )
    else:
        outputs["calibrated_predictions"] = (
            f"{out}/calibrated/test_{names}_with_intervals.parquet"
        )
    outputs["coverage_dir"] = f"{out}/coverage"
    if "apply" in config["inputs"]:
        if len(families) > 1:
            outputs["applied_predictions_volt"] = f"{out}/new_volt_bounds.parquet"
            outputs["applied_predictions_line"] = f"{out}/new_line_bounds.parquet"
        else:
            outputs["applied_predictions"] = f"{out}/new_{names}_bounds.parquet"
    if bounds:
        outputs["screen_dir"] = f"{out}/screen"
        if "apply" in config["inputs"]:
            # `apply` also gives the status of every new value and scenario.
            outputs["status_dir"] = f"{out}/status"
    config["outputs"] = outputs

    screened = [name for name in variables if name in bounds]
    if bounds and screened != variables:
        config["commands"] = {"screen": {"variables": screened}}
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if ``compute``, ``calibrate`` or ``screen`` would reject it."""

    config = dict(config)
    validate_config_schema(config)
    inputs = config.get("inputs") or {}
    for dataset in ("calibration", "test"):
        block = inputs.get(dataset) or {}
        missing = [key for key in _REQUIRED_DATASET_KEYS if not block.get(key)]
        if config.get("cp_type") == "kernel" and not block.get("embeddings"):
            missing.append("embeddings")
        if missing:
            raise ValueError(f"inputs.{dataset} is missing: {', '.join(missing)}.")
    if not config.get("variables"):
        raise ValueError("Select at least one variable to calibrate.")
    commands = ["compute", "calibrate", "screen"]
    if "apply" in inputs:
        commands.append("apply")
        if not (inputs.get("apply") or {}).get("predictions"):
            raise ValueError("inputs.apply is missing: predictions.")
    for command in commands:
        validate_command_config(command, resolve_command_config(config, command))


def to_yaml(config: Mapping[str, Any], *, path: str = "trust_config.yaml") -> str:
    """YAML text with a header listing the commands that use the file."""

    header = [
        "# trustlayer config, generated with gridfm_trustlayer.interactive.",
        "# Run:",
        f"#   gridfm_trustlayer compute   --config {path}",
        f"#   gridfm_trustlayer calibrate --config {path}",
    ]
    if "screen_bounds" in config:
        header.append(f"#   gridfm_trustlayer screen    --config {path}")
    if "apply" in (config.get("inputs") or {}):
        header.append(
            f"#   gridfm_trustlayer apply     --config {path}   # new scenarios"
        )
    body = yaml.safe_dump(dict(config), sort_keys=False, default_flow_style=False)
    return "\n".join(header) + "\n\n" + body


def save_config(config: Mapping[str, Any], path: str) -> Path:
    """Validate and write ``config`` to ``path``; returns the written path."""

    validate_config(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_yaml(config, path=str(path)), encoding="utf-8")
    return target


# ------------------------------------------------------------------ Jupyter form
def config_builder() -> Any:
    """Return an interactive form that builds and saves a trustlayer config.

    Call it as the last line of a Jupyter cell so the form is displayed.
    """

    try:
        import ipywidgets as w  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            'config_builder() needs ipywidgets: pip install "gridfm-trustlayer[interactive]"',
        ) from exc

    def wide() -> Any:
        # A fresh Layout per widget: fields are hidden through layout.display, and a
        # Layout shared between widgets would hide or show all of them together.
        return w.Layout(width="420px")

    style = {"description_width": "150px"}

    def note(text: str) -> Any:
        return w.HTML(
            f"<div style='color:#666;font-size:12px;margin:-2px 0 6px 155px'>{text}</div>"
        )

    def header(text: str) -> Any:
        return w.HTML(f"<h4 style='margin:14px 0 4px'>{text}</h4>")

    def show(widget: Any, visible: bool) -> None:
        widget.layout.display = None if visible else "none"

    # --- method
    task = w.Dropdown(
        options=TASK_NAMES,
        value="PowerFlow",
        description="Task",
        style=style,
        layout=wide(),
    )
    cp_type = w.ToggleButtons(
        options=CP_TYPES, value="element", description="Method", style=style
    )
    alpha = w.BoundedFloatText(
        value=0.1,
        min=0.001,
        max=0.999,
        step=0.01,
        description="alpha",
        style=style,
        layout=wide(),
    )
    variables = {
        name: w.Checkbox(value=name != "line_loading", description=name, indent=False)
        for name in VARIABLES
    }

    # --- stratification
    strat = w.Dropdown(
        options=[("none", None), *[(name, name) for name in STRATS]],
        value=None,
        description="Stratify by",
        style=style,
        layout=wide(),
    )
    load_bins = w.BoundedIntText(
        value=5, min=2, max=100, description="Load bins", style=style, layout=wide()
    )
    pred_bins = w.BoundedIntText(
        value=5,
        min=2,
        max=100,
        description="Prediction bins",
        style=style,
        layout=wide(),
    )

    sparse = w.Dropdown(
        options=[
            ("pooled quantile", "pooled"),
            ("infinite bounds", "infinite"),
        ],
        value="pooled",
        description="Sparse strata",
        style=style,
        layout=wide(),
    )

    # --- kernel
    pooling = w.Dropdown(
        options=sorted(POOLING_REGISTRY),
        value=DEFAULT_POOLING,
        description="Embedding pooling",
        style=style,
        layout=wide(),
    )
    distance = w.Dropdown(
        options=SUPPORTED_DISTANCES,
        value=DEFAULT_DISTANCE,
        description="Distance",
        style=style,
        layout=wide(),
    )
    bandwidth = w.ToggleButtons(
        # Default first.
        options=[
            DEFAULT_BANDWIDTH,
            *[name for name in SUPPORTED_BANDWIDTHS if name != DEFAULT_BANDWIDTH],
        ],
        value=DEFAULT_BANDWIDTH,
        description="Bandwidth",
        style=style,
    )
    k_nn = w.BoundedIntText(
        value=DEFAULT_K_NN,
        min=1,
        max=100000,
        description="k_nn",
        style=style,
        layout=wide(),
    )
    n_eff_target = w.Text(
        value="",
        placeholder="blank = 600 (default)",
        description="n_eff_target",
        style=style,
        layout=wide(),
    )
    n_eff_min = w.BoundedFloatText(
        value=DEFAULT_N_EFF_MIN,
        min=1.0,
        max=1e6,
        description="n_eff_min",
        style=style,
        layout=wide(),
    )
    tune = w.Checkbox(
        value=False,
        description="Tune n_eff_target by cross-validation on the calibration set",
        indent=False,
        layout=w.Layout(width="520px"),
    )
    tune_note = note(
        "compute splits the calibration scenarios into folds, fits the kernel on all but one "
        "fold and checks coverage and width on the held-out one, for each candidate. It picks "
        "the narrowest candidate that meets the coverage target overall and in every stratum; "
        "if none does, the weights become uniform (equivalent to element). The test set is "
        "never used. Off by default: costs about one kernel run per fold and candidate."
    )
    folds = w.BoundedIntText(
        value=DEFAULT_TUNING_FOLDS,
        min=2,
        max=20,
        description="Folds",
        style=style,
        layout=wide(),
    )
    grid = w.Text(
        value=", ".join(f"{value:g}" for value in DEFAULT_TUNING_GRID),
        description="Candidates",
        style=style,
        layout=wide(),
    )
    tune_distance = w.Checkbox(
        value=False,
        description="Also compare distances (euclidean and cosine)",
        indent=False,
        layout=w.Layout(margin="0 0 0 155px"),
    )
    search = w.ToggleButtons(
        options=["grid", "random"], value="grid", description="Search", style=style
    )
    budget = w.BoundedIntText(
        value=DEFAULT_TUNING_BUDGET,
        min=1,
        max=1000,
        description="Budget (tries)",
        style=style,
        layout=wide(),
    )
    n_eff_range = w.Text(
        value=f"{min(DEFAULT_TUNING_GRID):g}, {max(DEFAULT_TUNING_GRID):g}",
        description="n_eff range",
        style=style,
        layout=wide(),
    )
    grid_row = w.VBox(
        [
            grid,
            note(
                "n_eff_target values to compare (comma-separated). Values above a stratum's "
                "size give uniform weights in that stratum."
            ),
        ]
    )
    random_row = w.VBox(
        [
            budget,
            n_eff_range,
            note(
                "Evaluates this many distinct settings, drawing n_eff_target log-uniformly "
                "from the range (low, high) and the distance at random. Use it to bound the "
                "cost when both dimensions vary; the cost grows with the budget, not with "
                "the size of the space."
            ),
        ]
    )
    group_by = w.Dropdown(
        options=[
            ("auto: load_scenario_idx if present", "auto"),
            ("load_scenario_idx (required)", "load_scenario_idx"),
            ("none: split scenarios at random", "none"),
        ],
        value="auto",
        description="Group folds by",
        style=style,
        layout=wide(),
    )
    seed = w.IntText(value=0, description="Seed", style=style, layout=wide())
    tolerance = w.Text(
        value="auto",
        description="Coverage tolerance",
        style=style,
        layout=wide(),
    )
    tuning_box = w.VBox(
        [
            folds,
            note(
                "More folds use more calibration data per fit but take longer; 5 is a good default."
            ),
            tune_distance,
            note(
                "Adds the distance as a second dimension. Cosine keeps only the direction in "
                "which a scenario deviates from the average one; it doubles the cost of a "
                "grid search."
            ),
            search,
            note(
                "grid: every combination of the candidates (and distances). random: a fixed "
                "budget of tries drawn from a range."
            ),
            grid_row,
            random_row,
            group_by,
            note(
                "Scenarios sharing a load profile (datakit load_scenario_idx) stay in one fold, "
                "so near-duplicates cannot make the kernel look better than it is."
            ),
            seed,
            tolerance,
            note(
                "How far below 1 - alpha cross-validated coverage may fall and still count as "
                "meeting the target. auto: two standard errors for the number of scenarios "
                "checked; or a fixed number such as 0.005."
            ),
        ]
    )
    knn_row = w.VBox(
        [
            k_nn,
            note(
                "Neighbour rank that sets the bandwidth. Tune it on data disjoint from the test set."
            ),
        ]
    )
    neff_row = w.VBox(
        [
            n_eff_target,
            note(
                "Effective sample size per test scenario; blank uses the default, 600. Transfers "
                "across networks and embedding widths, unlike k_nn. Or tick the box above to "
                "choose it by cross-validation on the calibration set."
            ),
        ]
    )
    sparse_box = w.VBox(
        [
            sparse,
            note(
                "For an element in a stratum with too few calibration samples for a finite bound "
                "(fewer than ceil(2/alpha) - 1), or a stratum absent from the calibration data. "
                "The pooled quantile (over all calibration scenarios, the default) is finite but "
                "carries no guarantee within that stratum; infinite bounds keep the guarantee."
            ),
        ]
    )
    kernel_box = w.VBox(
        [
            header("Kernel options"),
            pooling,
            distance,
            bandwidth,
            knn_row,
            tune,
            tune_note,
            neff_row,
            tuning_box,
            n_eff_min,
        ]
    )

    # --- datasets
    def dataset_fields(name: str) -> dict[str, Any]:
        return {
            key: w.Text(
                placeholder=f"path/to/{name}/{default}",
                description=key,
                style={"description_width": "150px"},
                layout=wide(),
            )
            for key, default in [
                ("predictions", "predictions.parquet"),
                ("pf_node", "raw/bus_data.parquet"),
                ("pf_edge", "raw/branch_data.parquet"),
                ("embeddings", "bus_embeddings.parquet"),
                ("generator_predictions", "predictions_gen.parquet"),
            ]
        }

    cal_fields, test_fields = dataset_fields("calibration"), dataset_fields("test")
    new_fields = dataset_fields("new")
    use_apply = w.Checkbox(
        value=False,
        description="Also apply the layer to new scenarios (no known solution)",
        indent=False,
    )
    apply_box = w.VBox(
        [
            w.HTML(
                "<div style='color:#666;font-size:12px;margin:0 0 6px 155px'>"
                "For <code>gridfm_trustlayer apply</code>. pf_node and pf_edge describe "
                "the new scenarios (topology, loads); they are needed for outage or load "
                "stratification and for line_loading, and need no AC solution. With screening "
                "bounds, apply also gives each new scenario its status (secure, uncertain, "
                "insecure).</div>"
            ),
            *new_fields.values(),
        ],
        layout=w.Layout(),
    )

    # --- screening and outputs
    screen = w.Checkbox(
        value=True, description="Screen against operating limits", indent=False
    )
    screen_note = note(
        "Off: the config only produces prediction intervals (compute, calibrate). "
        "On: <code>screen</code> also flags predicted violations of the bounds below."
    )
    no_screenable = note(
        "Va has no operating limit: select Vm or line_loading to screen anything."
    )
    vm_lower = w.FloatText(
        value=0.95, description="Vm lower (pu)", style=style, layout=wide()
    )
    vm_upper = w.FloatText(
        value=1.05, description="Vm upper (pu)", style=style, layout=wide()
    )
    line_upper = w.FloatText(
        value=1.0, description="Line loading upper", style=style, layout=wide()
    )
    output_dir = w.Text(
        value="trustlayer_outputs",
        description="Output directory",
        style=style,
        layout=wide(),
    )
    filename = w.Text(
        value="trust_config.yaml", description="Config file", style=style, layout=wide()
    )

    status = w.HTML()
    preview = w.HTML()
    save = w.Button(description="Save config", button_style="primary", icon="save")
    saved = w.HTML()

    def current_config() -> dict[str, Any]:
        raw_target = n_eff_target.value.strip()
        raw_tolerance = tolerance.value.strip().lower()

        def numbers(text: str) -> list[float]:
            return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]

        tuning_values: dict[str, Any] = {
            "folds": folds.value,
            "grid": numbers(grid.value),
            "search": search.value,
            "budget": budget.value,
            "n_eff_range": numbers(n_eff_range.value),
            "distances": list(SUPPORTED_DISTANCES) if tune_distance.value else None,
            "group_by": group_by.value,
            "seed": seed.value,
            "coverage_tolerance": raw_tolerance
            if raw_tolerance in ("", "auto")
            else float(raw_tolerance),
        }
        if tuning_values["coverage_tolerance"] == "":
            tuning_values["coverage_tolerance"] = "auto"
        bounds: dict[str, dict[str, float]] = {
            "Vm": {"lower": vm_lower.value, "upper": vm_upper.value},
            "line_loading": {"upper": line_upper.value},
        }
        return build_config(
            calibration={key: field.value for key, field in cal_fields.items()},
            test={key: field.value for key, field in test_fields.items()},
            apply={key: field.value for key, field in new_fields.items()}
            if use_apply.value
            else None,
            task_name=task.value,
            cp_type=cp_type.value,
            alpha=alpha.value,
            variables=[name for name, box in variables.items() if box.value],
            strat=strat.value,
            system_load_n_bins=load_bins.value,
            pred_n_bins=pred_bins.value,
            sparse_strata=sparse.value,
            embedding_pooling=pooling.value,
            distance=distance.value,
            bandwidth=bandwidth.value,
            k_nn=k_nn.value,
            n_eff_target=float(raw_target) if raw_target else None,
            n_eff_min=n_eff_min.value,
            tune_n_eff_target=tune.value,
            tuning=tuning_values,
            screen_bounds=bounds if screen.value else None,
            output_dir=output_dir.value,
        )

    def refresh(_: Any = None) -> None:
        kernel = cp_type.value == "kernel"
        # pred_bins assigns bins per element, kernel localizes per scenario: incompatible.
        options = [
            ("none", None),
            *[(n, n) for n in STRATS if not (kernel and n == "pred_bins")],
        ]
        if list(strat.options) != options:
            value = strat.value if strat.value != "pred_bins" or not kernel else None
            strat.options = options
            strat.value = value
        show(load_bins, strat.value == "system_load_bin")
        show(pred_bins, strat.value == "pred_bins")
        show(sparse_box, strat.value is not None)
        show(kernel_box, kernel)
        show(knn_row, bandwidth.value == "knn")
        n_eff_bandwidth = bandwidth.value == "n_eff"
        show(tune, n_eff_bandwidth)
        show(tune_note, n_eff_bandwidth)
        show(neff_row, n_eff_bandwidth and not tune.value)
        show(tuning_box, n_eff_bandwidth and tune.value)
        show(grid_row, search.value == "grid")
        show(random_row, search.value == "random")
        for fields in (cal_fields, test_fields):
            show(fields["embeddings"], kernel)
            show(fields["generator_predictions"], task.value == "OptimalPowerFlow")
        show(apply_box, use_apply.value)
        show(new_fields["embeddings"], kernel)
        show(new_fields["generator_predictions"], task.value == "OptimalPowerFlow")
        show(vm_lower, screen.value and variables["Vm"].value)
        show(vm_upper, screen.value and variables["Vm"].value)
        show(line_upper, screen.value and variables["line_loading"].value)
        show(
            no_screenable,
            screen.value and not any(variables[name].value for name in SCREENABLE),
        )
        saved.value = ""
        try:
            config = current_config()
            validate_config(config)
        except (ValueError, TypeError) as exc:
            status.value = (
                f"<b style='color:#b00'>Not valid yet:</b> {html.escape(str(exc))}"
            )
            save.disabled = True
            config = None
        else:
            status.value = "<b style='color:#080'>Valid</b>: accepted by compute, calibrate and screen."
            save.disabled = False
        if config is None:
            try:
                config = current_config()
            except (ValueError, TypeError):
                config = {}
        text = to_yaml(config, path=filename.value) if config else ""
        preview.value = (
            "<pre style='background:#f6f6f6;padding:8px;font-size:12px;max-height:520px;overflow:auto'>"
            f"{html.escape(text)}</pre>"
        )

    def on_save(_: Any) -> None:
        try:
            config = current_config()
            path = save_config(config, filename.value)
        except (ValueError, OSError) as exc:
            saved.value = (
                f"<b style='color:#b00'>Not saved:</b> {html.escape(str(exc))}"
            )
            return
        saved.value = (
            f"<b style='color:#080'>Saved to {html.escape(str(path.resolve()))}</b><br>"
            f"Next: <code>gridfm_trustlayer compute --config {html.escape(filename.value)}</code>, "
            + (
                "then <code>calibrate</code> and <code>screen</code>"
                if "screen_bounds" in config
                else "then <code>calibrate</code> (no screening step)"
            )
            + (
                "; <code>apply</code> for the new scenarios."
                if "apply" in config["inputs"]
                else "."
            )
        )

    form = w.VBox(
        [
            header("Method"),
            task,
            cp_type,
            note(
                "<b>element</b>: stratified split conformal, with a finite-sample guarantee. "
                "<b>kernel</b>: adapts the margin to each scenario; coverage verified empirically."
            ),
            alpha,
            note(
                "Two-sided: each tail gets alpha / 2, so 0.1 gives 90% intervals. For a one-sided 1 - a screen, use alpha = 2a."
            ),
            w.HTML("<div style='margin-left:155px'>Variables:</div>"),
            w.HBox(list(variables.values()), layout=w.Layout(margin="0 0 0 155px")),
            header("Stratification"),
            strat,
            note(
                "One quantile per stratum and entity. n_outages is the usual choice for N-k screening. "
                "Small strata give wider (or infinite) bounds."
            ),
            load_bins,
            pred_bins,
            sparse_box,
            kernel_box,
            header("Data: two separately generated datasets"),
            note(
                "compute fits on the calibration set; calibrate and screen evaluate on the test set. "
                "They must be different data, e.g. datakit runs with different seeds."
            ),
            w.HBox(
                [
                    w.VBox([w.HTML("<b>Calibration</b>"), *cal_fields.values()]),
                    w.VBox([w.HTML("<b>Test</b>"), *test_fields.values()]),
                ]
            ),
            w.HBox([use_apply], layout=w.Layout(margin="8px 0 0 155px")),
            apply_box,
            header("Screening"),
            w.HBox([screen], layout=w.Layout(margin="0 0 0 155px")),
            screen_note,
            no_screenable,
            vm_lower,
            vm_upper,
            line_upper,
            header("Output"),
            output_dir,
            filename,
            status,
            w.HBox([save]),
            saved,
        ]
    )
    for widget in [
        task,
        cp_type,
        alpha,
        strat,
        load_bins,
        pred_bins,
        pooling,
        distance,
        bandwidth,
        k_nn,
        n_eff_target,
        n_eff_min,
        tune,
        folds,
        grid,
        tune_distance,
        search,
        budget,
        n_eff_range,
        group_by,
        seed,
        tolerance,
        screen,
        vm_lower,
        vm_upper,
        line_upper,
        output_dir,
        filename,
        *variables.values(),
        *cal_fields.values(),
        *test_fields.values(),
        use_apply,
        *new_fields.values(),
    ]:
        widget.observe(refresh, names="value")
    save.on_click(on_save)
    refresh()
    return w.HBox(
        [
            form,
            w.VBox(
                [header("Preview"), preview],
                layout=w.Layout(margin="0 0 0 24px", width="520px"),
            ),
        ]
    )
