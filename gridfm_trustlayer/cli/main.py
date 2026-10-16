"""Command-line interface: `compute` fits the layer on calibration data, `calibrate`
applies it to test data, `screen` compares the intervals with operating limits,
and `apply` gives bounds and statuses for new scenarios."""

from __future__ import annotations

import argparse
import os
import sys

import pyarrow as pa  # type: ignore[import-untyped]

from gridfm_trustlayer._version import __version__
from gridfm_trustlayer.cli.apply import (
    _run_apply,
)
from gridfm_trustlayer.cli.calibrate import (
    _run_calibrate,
)
from gridfm_trustlayer.cli.compute import (
    _run_compute,
)
from gridfm_trustlayer.cli.screen import (
    _run_screen,
)
from gridfm_trustlayer.conformal.calibrator import (
    SPARSE_STRATA_CHOICES,
)


def main(argv: list[str] | None = None) -> int:
    """Run the gridfm_trustlayer command-line interface."""

    # The commands read and free one block of scenarios at a time. Arrow's default
    # allocators keep freed blocks reserved; the system allocator returns them,
    # which keeps the peak memory of a long run close to that of one block. Each
    # decoding thread also keeps its own freed memory, and beyond a few threads
    # reading a block is no faster, so the thread count is capped.
    pa.set_memory_pool(pa.system_memory_pool())
    pa.set_cpu_count(min(4, pa.cpu_count()))
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "compute": _run_compute,
        "calibrate": _run_calibrate,
        "apply": _run_apply,
        "screen": _run_screen,
    }
    try:
        commands[args.command](args)
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        # Bad configs, paths and data, and a missing optional extra, are the
        # user's to fix: one line, not a traceback. GRIDFM_TRUSTLAYER_DEBUG=1
        # shows the traceback.
        if os.environ.get(DEBUG_VARIABLE):
            raise
        print(f"gridfm_trustlayer {args.command}: error: {exc}", file=sys.stderr)
        return 2
    return 0


DEBUG_VARIABLE = "GRIDFM_TRUSTLAYER_DEBUG"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="gridfm_trustlayer",
        description=(
            "Conformal trustworthiness layer for GridFM foundation models: "
            "compute, calibrate, then screen; apply gives bounds for new scenarios."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute_parser = subparsers.add_parser(
        "compute",
        help="Fit the layer on the calibration dataset (inputs.calibration) and save a .trustlayer artifact.",
    )
    compute_parser.add_argument(
        "--config",
        required=True,
        help="Required trustlayer YAML config file. CLI flags override config values.",
    )
    compute_parser.add_argument(
        "--predictions", help="Predictions parquet with *_pred columns."
    )
    compute_parser.add_argument(
        "--pf-node", help="Ground-truth bus parquet used to fit residual quantiles."
    )
    compute_parser.add_argument(
        "--pf-edge", help="Ground-truth branch parquet used to fit residual quantiles."
    )
    compute_parser.add_argument(
        "--output", help="Destination .trustlayer artifact path."
    )
    compute_parser.add_argument(
        "--alpha", type=float, help="Miscoverage level for conformal intervals."
    )
    compute_parser.add_argument(
        "--cp-type",
        help="Calibration type: 'element' (stratified) or 'kernel' (kernel-weighted).",
    )
    compute_parser.add_argument(
        "--strat",
        help="Optional residual stratification. Supported values: 'n_outages', 'outage_type', 'system_load_bin', 'pred_bins'.",
    )
    compute_parser.add_argument(
        "--system-load-n-bins",
        type=int,
        help="Requested number of percentile bins when --strat system_load_bin is used.",
    )
    compute_parser.add_argument(
        "--pred-n-bins",
        type=int,
        help="Requested number of percentile bins when --strat pred_bins is used.",
    )
    compute_parser.add_argument(
        "--variables",
        nargs="+",
        help="Variables to calibrate: any subset of Vm, Va, and line_loading.",
    )
    compute_parser.add_argument(
        "--embeddings",
        help="Scenario embeddings parquet (emb_* columns). Required for cp_type 'kernel'.",
    )
    compute_parser.add_argument(
        "--embedding-pooling",
        help="How per-entity embeddings are pooled into one vector per scenario: mean, max, meanmax, or concat.",
    )
    compute_parser.add_argument(
        "--distance",
        help="Distance over scenario representations: 'euclidean' (default) or 'cosine'.",
    )
    compute_parser.add_argument(
        "--sparse-strata",
        choices=SPARSE_STRATA_CHOICES,
        help="Bounds of an element in a stratum with too few calibration samples: 'pooled' (default, its quantile over all strata) or 'infinite'.",
    )
    compute_parser.add_argument(
        "--bandwidth",
        help="How the kernel bandwidth is chosen: 'n_eff' (default, solve for a target effective sample size) or 'knn' (distance to the k_nn-th neighbour).",
    )
    compute_parser.add_argument(
        "--k-nn",
        type=int,
        help="Nearest-neighbour rank whose distance sets the kernel bandwidth (--bandwidth knn).",
    )
    compute_parser.add_argument(
        "--n-eff-target",
        help="Target effective sample size solved for per test scenario (--bandwidth n_eff). Defaults to 600. 'auto' chooses it by cross-validation on the calibration data (options under 'tuning' in the config).",
    )
    compute_parser.add_argument(
        "--n-eff-min",
        type=float,
        help="Minimum effective sample size before a scenario falls back to the stratum-wide quantile (cp_type 'kernel').",
    )
    compute_parser.add_argument(
        "--task-name", help="Task name used to interpret prediction schema."
    )
    compute_parser.add_argument(
        "--generator-predictions",
        help="Optional generator predictions parquet for OPF artifacts.",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Apply the artifact to the test dataset (inputs.test): write the predictions with their bounds, coverage reports and figures.",
    )
    calibrate_parser.add_argument(
        "--config",
        required=True,
        help="Required trustlayer YAML config file. CLI flags override config values.",
    )
    calibrate_parser.add_argument(
        "--predictions",
        help="Predictions parquet to augment or transform into branch outputs.",
    )
    calibrate_parser.add_argument(
        "--artifact", help="Previously computed .trustlayer artifact."
    )
    calibrate_parser.add_argument(
        "--output", help="Destination parquet for calibrated predictions."
    )
    calibrate_parser.add_argument(
        "--cp-type",
        help="Calibration type: 'element' (stratified) or 'kernel' (kernel-weighted).",
    )
    calibrate_parser.add_argument(
        "--strat",
        help="Optional residual stratification override. Supported values: 'n_outages', 'outage_type', 'system_load_bin', 'pred_bins'. Must match the saved artifact when provided.",
    )
    calibrate_parser.add_argument(
        "--pf-node",
        help="Ground-truth bus parquet; required for line_loading analysis when targets are not stored in predictions.",
    )
    calibrate_parser.add_argument(
        "--pf-edge",
        help="Ground-truth branch parquet; required for line_loading analysis when targets are not stored in predictions.",
    )
    calibrate_parser.add_argument(
        "--embeddings",
        help="Scenario embeddings parquet for the application scenarios. Required when the artifact is a kernel artifact.",
    )
    calibrate_parser.add_argument(
        "--task-name", help="Task name used to interpret prediction schema."
    )
    calibrate_parser.add_argument(
        "--generator-predictions",
        help="Optional generator predictions parquet for OPF artifacts.",
    )
    calibrate_figures_group = calibrate_parser.add_mutually_exclusive_group()
    calibrate_figures_group.add_argument(
        "--figures",
        dest="save_figures",
        action="store_true",
        default=None,
        help="Write analysis figures during calibrate (default behavior).",
    )
    calibrate_figures_group.add_argument(
        "--no-figures",
        dest="save_figures",
        action="store_false",
        help="Skip analysis figure generation and write only parquet/CSV outputs.",
    )

    apply_parser = subparsers.add_parser(
        "apply",
        help="Write bounds for new scenarios (inputs.apply), which need no known AC solution, and, with screen_bounds, the status of every value and scenario.",
    )
    apply_parser.add_argument(
        "--config",
        required=True,
        help="Required trustlayer YAML config file. CLI flags override config values.",
    )
    apply_parser.add_argument(
        "--predictions", help="Predictions parquet of the new scenarios."
    )
    apply_parser.add_argument(
        "--artifact", help="Previously computed .trustlayer artifact."
    )
    apply_parser.add_argument(
        "--output",
        help="Output dataset path for one target family, or directory for both.",
    )
    apply_parser.add_argument(
        "--pf-node",
        help="Bus file describing the new scenarios (loads, bus types); needed for "
        "strat n_outages, outage_type or system_load_bin, and for line_loading.",
    )
    apply_parser.add_argument(
        "--pf-edge",
        help="Branch file describing the new scenarios (status, parameters, ratings); "
        "needed with --pf-node.",
    )
    apply_parser.add_argument(
        "--embeddings",
        help="Scenario embeddings parquet of the new scenarios (kernel artifacts).",
    )
    apply_parser.add_argument(
        "--task-name", help="Task name used to interpret prediction schema."
    )
    apply_parser.add_argument(
        "--generator-predictions",
        help="Optional generator predictions parquet for OPF artifacts.",
    )

    screen_parser = subparsers.add_parser(
        "screen",
        help="Compare the test bounds with the operating limits (screen_bounds) and count caught, missed and false alarms.",
    )
    screen_parser.add_argument(
        "--config",
        required=True,
        help="Required trustlayer YAML config file. CLI flags override config values.",
    )
    screen_parser.add_argument(
        "--variables",
        nargs="+",
        help="Variables to screen. Defaults to config values for the screen command.",
    )
    screen_parser.add_argument(
        "--lower-bound",
        type=float,
        help="Override lower screening bound for a single selected variable. Not used for line_loading.",
    )
    screen_parser.add_argument(
        "--upper-bound",
        type=float,
        help="Override upper screening bound for a single selected variable.",
    )
    screen_parser.add_argument(
        "--output", help="Destination directory for screening CSVs and figures."
    )
    screen_parser.add_argument(
        "--calibrated-predictions",
        help="Single calibrated parquet input for one target family.",
    )
    screen_parser.add_argument(
        "--calibrated-predictions-volt",
        help="Calibrated voltage parquet input override.",
    )
    screen_parser.add_argument(
        "--calibrated-predictions-line",
        help="Calibrated line-loading parquet input override.",
    )
    screen_parser.add_argument(
        "--pf-node",
        help="Ground-truth bus parquet used when target columns are not stored in the calibrated voltage parquet.",
    )
    screen_parser.add_argument(
        "--pf-edge",
        help="Ground-truth branch parquet used when target columns are not stored in the calibrated outputs.",
    )
    screen_parser.add_argument(
        "--task-name", help="Task name used for default output-path resolution."
    )
    screen_figures_group = screen_parser.add_mutually_exclusive_group()
    screen_figures_group.add_argument(
        "--figures",
        dest="save_figures",
        action="store_true",
        default=None,
        help="Write screening figures (default behavior).",
    )
    screen_figures_group.add_argument(
        "--no-figures",
        dest="save_figures",
        action="store_false",
        help="Skip screening figures and write only the tables and the detail.",
    )

    blocks_help = (
        "Processes for blocks of scenarios (-1 for all CPUs). Off by default: on "
        "one machine, one process is usually as fast and uses less memory. "
        "Overrides n_jobs in the config."
    )
    for name, subparser in subparsers.choices.items():
        subparser.add_argument(
            "--n-jobs",
            type=int,
            help=(
                "Processes for the cross-validated bandwidth choice (default 4, at "
                "most the CPU count; -1 for all CPUs; 1 for one). When given, blocks "
                "of scenarios are also read in parallel, which is off by default. "
                "Overrides n_jobs in the config."
                if name == "compute"
                else blocks_help
            ),
        )
    return parser
