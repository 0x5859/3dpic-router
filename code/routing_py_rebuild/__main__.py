"""CLI entry: ``python -m routing_py_rebuild ...``.

Three subcommands:

  optimize   Run an optimizer end-to-end (graph → solve → JSON → plots).
  plot       Reload a saved subgraph JSON and render a chosen style.
  report     Render convergence + timings PNGs from a ``run_report.json``.

Defaults match the original demo (k=12, dual_annealing, plain 'visualize').
"""
from __future__ import annotations

import argparse
import json as _json
import sys

from .api import run_optimization
from .plotting import plot_from_json
from .plotting.convergence import plot_convergence, plot_timings


def _parse_kwargs(s: str | None) -> dict:
    if not s:
        return {}
    return _json.loads(s)


def _add_optimize_parser(sub):
    p = sub.add_parser("optimize", help="Run optimizer end-to-end")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--optimizer", default="dual_annealing",
                   choices=[
                       "dual_annealing",
                       "dual_annealing_with_swap_polish",
                       "differential_evolution",
                   ])
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--seed", type=int, default=5859)
    p.add_argument("--optimizer-kwargs", default=None,
                   help="JSON dict forwarded to the scipy optimizer")
    p.add_argument("--output-dir", default="./assets/run/")
    # M3 (REFACTOR_GOALS.md §2-3) — multi-layer + edge coupler config. Default
    # L=2 / edge_coupler_layer=0 keeps the legacy bit-exact baseline; pass
    # explicit values for L>=3 (recommended ecl = L // 2).
    p.add_argument("--L", type=int, default=2, dest="L",
                   help="Number of physical SiN layers (default 2).")
    p.add_argument("--edge-coupler-layer", type=int, default=0,
                   help="Layer index of the fiber-to-chip edge coupler "
                        "(default 0; recommended L//2 for L>=3).")
    p.add_argument("--perimeter-layer", type=int, default=None,
                   help="Layer that the perimeter ring is pinned to "
                        "(default = edge-coupler-layer).")
    p.add_argument("--layer-pitch-um", type=float, default=1.2,
                   help="Physical SiN layer pitch in μm (default 1.2).")
    p.add_argument("--waveguides-per-link", type=int, default=2,
                   help="Parallel waveguides per logical edge (default 2 = "
                        "bidirectional). Affects "
                        "run_report.summary.crosslayer_crossings_total; "
                        "loss formula is not parameterized on it in M3 "
                        "(see REFACTOR_GOALS.md §7 Q-e).")
    p.add_argument("--loss-crossing", type=float, default=0.3)
    p.add_argument("--loss-taper", type=float, default=0.05)
    p.add_argument("--loss-interlayercrossing", type=float, default=0.006)
    # M4 (§2-2): crosstalk coefficients + analysis knobs. When both
    # coefficients are None / 0 the engine short-circuits; when at least
    # one is non-zero the rank-3 tensor is computed once after
    # optimization and embedded in subgraphsdata.json.
    p.add_argument("--loss-intralayer-crosstalk", type=float, default=None,
                   help="Intralayer crossing leakage to orthogonal "
                        "branch (fractional power per crossing). "
                        "Recommended start: 1e-4 (−40 dB).")
    p.add_argument("--loss-interlayer-crosstalk", type=float, default=None,
                   help="Adjacent-layer crossing leakage to orthogonal "
                        "branch (fractional power per crossing). "
                        "Recommended start: 1e-5 (−50 dB). Non-adjacent "
                        "layer pairs (|Δlayer|>=2) are not coupled per "
                        "§2-3 目标 D.")
    p.add_argument("--no-crosstalk-analysis", action="store_true",
                   help="Skip post-optimization crosstalk tensor "
                        "computation even when coefficients are non-zero. "
                        "Useful for perf benchmarks of the optimizer "
                        "loop alone (REFACTOR_GOALS.md §2-2).")
    p.add_argument("--crosstalk-max-hops", type=int, default=3,
                   help="Maximum recursion depth for branch propagation "
                        "(default 3 per §2-2; single-hop ≈ −40 dB, hop=4 "
                        "≈ −160 dB ≪ threshold).")
    p.add_argument("--crosstalk-threshold-db", type=float, default=-60.0,
                   help="Prune branches below this linear-power "
                        "threshold (in dB); default −60.")
    p.add_argument("--crosstalk-include-in-loss", action="store_true",
                   help="Include crosstalk in the loss function. "
                        "Reserved for future use; currently a no-op flag "
                        "preserved on run_report.config.")
    p.add_argument("--plot-style", default="visualize",
                   choices=["visualize"])
    p.add_argument("--plot-kwargs", default=None,
                   help="JSON dict forwarded to the plot style function")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-loss-analysis", action="store_true")
    p.add_argument("--no-json", action="store_true")
    p.add_argument("--collect-statistics", action="store_true",
                   help="Record per-iter trace + phase timings; emits run_report.{json,md}.")


def _add_plot_parser(sub):
    p = sub.add_parser("plot", help="Reload subgraph JSON and plot")
    p.add_argument("--json", required=True, help="Path to subgraphsdata.json")
    p.add_argument("--style", default="visualize",
                   choices=["visualize"])
    p.add_argument("--out-dir", default=None,
                   help="Output directory; defaults to the JSON file's directory.")
    p.add_argument("--plot-kwargs", default=None,
                   help="JSON dict forwarded to the plot style function")
    p.add_argument("--also-loss-analysis", action="store_true")


def _add_report_parser(sub):
    p = sub.add_parser("report", help="Render convergence + timings from run_report.json")
    p.add_argument("--report", required=True, help="Path to run_report.json")
    p.add_argument("--out-dir", default=None,
                   help="Output directory; defaults to the report file's directory.")


def _build_parser():
    parser = argparse.ArgumentParser(prog="routing_py_rebuild")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_optimize_parser(sub)
    _add_plot_parser(sub)
    _add_report_parser(sub)
    return parser


def _run_optimize(args):
    res = run_optimization(
        k=args.k,
        optimizer=args.optimizer,
        maxiter=args.maxiter,
        seed=args.seed,
        optimizer_kwargs=_parse_kwargs(args.optimizer_kwargs),
        output_dir=args.output_dir,
        L=args.L,
        edge_coupler_layer=args.edge_coupler_layer,
        perimeter_layer=args.perimeter_layer,
        layer_pitch_um=args.layer_pitch_um,
        waveguides_per_link=args.waveguides_per_link,
        loss_crossing=args.loss_crossing,
        loss_taper=args.loss_taper,
        loss_interlayercrossing=args.loss_interlayercrossing,
        loss_intralayer_crosstalk=args.loss_intralayer_crosstalk,
        loss_interlayer_crosstalk=args.loss_interlayer_crosstalk,
        compute_crosstalk=not args.no_crosstalk_analysis,
        crosstalk_max_hops=args.crosstalk_max_hops,
        crosstalk_threshold_db=args.crosstalk_threshold_db,
        crosstalk_include_in_loss=args.crosstalk_include_in_loss,
        plot=not args.no_plot,
        plot_style=args.plot_style,
        plot_kwargs=_parse_kwargs(args.plot_kwargs),
        save_json=not args.no_json,
        run_loss_analysis=not args.no_loss_analysis,
        collect_statistics=args.collect_statistics,
    )
    print(f"Final loss: {res['loss']}")
    print(f"JSON: {res['json_path']}")
    if res.get("report_path"):
        print(f"Report: {res['report_path']}")


def _run_plot(args):
    plot_from_json(
        args.json,
        style=args.style,
        out_dir=args.out_dir,
        also_loss_analysis=args.also_loss_analysis,
        **_parse_kwargs(args.plot_kwargs),
    )


def _run_report(args):
    import os
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.report))
    conv = plot_convergence(args.report, out_path=os.path.join(out_dir, "convergence.png"))
    tim = plot_timings(args.report, out_path=os.path.join(out_dir, "timings.png"))
    print(f"Convergence: {conv}")
    print(f"Timings: {tim}")


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.cmd == "optimize":
        _run_optimize(args)
    elif args.cmd == "plot":
        _run_plot(args)
    elif args.cmd == "report":
        _run_report(args)
    else:
        sys.exit("unknown command")


if __name__ == "__main__":
    main()
