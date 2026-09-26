"""High-level API: ``run_optimization`` (optimize → JSON → plot) and helpers.

Designed for both library use and the CLI in ``__main__.py``.

Post-optimization plotting goes through a :class:`plotting.PlotData`
snapshot, which the plot styles consume in place of a live graph. By
default (``save_json=True``) the snapshot is rebuilt from the JSON written
by :meth:`core.SiNInterconnectionGraph.save_subgraphs_to_json` — the
file is the contract and plotting never touches the graph. With
``save_json=False`` the orchestrator instead runs ``graph.analyze_loss()``
itself and snapshots the in-memory graph via
:meth:`plotting.PlotData.from_graph`; that escape hatch is intended for
tests and ad-hoc use, while still keeping the plotting code read-only.

When ``collect_statistics=True``, a :class:`statistics.RunRecorder` is
constructed and threaded through the optimizer + each major phase via
:class:`statistics.PhaseTimer`; ``run_report.json`` + ``run_report.md`` are
written next to ``subgraphsdata.json`` (REFACTOR_GOALS.md §1-1-a).

M3 (REFACTOR_GOALS.md §2-3 + §4 M3 附注): exposes ``L``,
``edge_coupler_layer``, ``perimeter_layer``, ``layer_pitch_um``,
``waveguides_per_link``. M4 (§2-2) wires the ``loss_intralayer_crosstalk`` /
``loss_interlayer_crosstalk`` coefficients and the new
``compute_crosstalk`` / ``crosstalk_max_hops`` / ``crosstalk_threshold_db`` /
``crosstalk_include_in_loss`` knobs end-to-end. The
aggregate stats writer applies the
``crosslayer_crossings_total = waveguides_per_link * geometric_events``
formula with explicit convention labelling — see
:func:`_extract_aggregate_stats`.

``progress="live"`` / ``"record"`` visualizes the optimization process
itself (see :mod:`.progress`): a live-updating figure while the optimizer
runs, or every improvement saved to ``optimization_history.json`` and
replayed as an animation afterwards. The default ``"off"`` leaves the run
untouched.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .core import (
    DEFAULT_LOSS_CROSSING,
    DEFAULT_LOSS_INTERLAYERCROSSING,
    DEFAULT_LOSS_TAPER,
    SiNInterconnectionGraph,
)
from .crosstalk import compute_crosstalk_tensor
from .optimizers import get_optimizer
from .plotting import (
    PlotData,
    plot_from_json,
    visualize_layers,
    visualize_loss_analysis,
    visualize_ring,
)
from .positions import distribute_nodes
from .progress import (
    ProgressOptions,
    ProgressTracker,
    chain_observers,
    normalize_progress_mode,
)
from .statistics import NullSink, PhaseTimer, RunRecorder, write_run_report


def _extract_aggregate_stats(graph: SiNInterconnectionGraph) -> dict:
    """Read per-layer aggregate stats from a graph whose ``analyze_loss()``
    has been called (so ``sub_G`` + ``total_crossings_of_sub_G`` are fresh).

    Implements §1-1-a item 3 "聚合统计" with the M3 rewrite (§4 M3 附注):

    - ``geometric_events`` = ``Σ_e interlayercrossings_above`` over all
      sub_G — each cross-layer geometric event triggers exactly one
      ``_above`` increment (on the lower edge), so this is bit-equal to
      the historical ``Σ interlayercrossings // 2`` (which double-counted
      because of the endpoint stamp).
    - **Legacy fallback**: when no ``_above`` field is present on any
      edge (e.g. an in-memory graph loaded from a v1.x snapshot that
      pre-dates the M3 split), reconstruct ``geometric_events`` from the
      legacy ``interlayercrossings`` field via ``// 2`` to honour
      §3-1 兼容性策略 line 769 ("legacy v1.x ... writer 走 fallback 公式
      `wpl * (sum(interlayercrossings) // 2)`"). Without this branch a
      legacy-loaded graph silently reports zero, which masquerades as a
      perfectly-routed network for any tool that aggregates stats post-load.
    - ``crosslayer_crossings_total`` =
      ``waveguides_per_link * geometric_events``. The post-M8 default
      (2026-05-15) is ``wpl=1`` (``convention="geometric"``, matches
      the paper's geometric crossing count); passing
      ``waveguides_per_link=2`` yields the historical physical
      convention (``convention="physical"``) aligned with the loss
      model's implicit 2× endpoint double-counting.
    - ``convention="experimental"`` is emitted for ``wpl ∈ {3, 4, ...}``
      because §4 M3 附注 "Warning 策略" calls those values out as
      semantically distinct (stats/loss mismatch documented in §7 Q-e).
      The constructor / loader notice is a runtime signal; the
      ``run_report.json`` convention field is the durable record for
      offline readers.

    The ``layers`` list keys are aligned to ``sub_G`` (length ``graph.L``);
    empty layers report ``edge_count=0`` and ``crossings=0`` rather than
    being elided so downstream consumers can index by layer number.
    """
    layers_stats: list[dict[str, int]] = []
    for i, sg in enumerate(graph.sub_G):
        edge_count = int(sg.number_of_edges())
        if i < len(graph.total_crossings_of_sub_G):
            total = int(graph.total_crossings_of_sub_G[i])
        else:
            total = 0
        layers_stats.append(
            {"layer": i, "edge_count": edge_count, "crossings": total}
        )

    # Try the M3 direct path (Σ _above) first. If no edge carries _above
    # (e.g. legacy v1.x graph), fall back to the historical formula.
    saw_above_field = False
    geometric_events = 0
    legacy_sum = 0
    for sg in graph.sub_G:
        for _u, _v, d in sg.edges(data=True):
            if "interlayercrossings_above" in d:
                saw_above_field = True
                geometric_events += int(d["interlayercrossings_above"])
            legacy_sum += int(d.get("interlayercrossings", 0))
    if not saw_above_field:
        geometric_events = legacy_sum // 2

    wpl = int(graph.waveguides_per_link)
    crosslayer_total = wpl * geometric_events
    if wpl == 1:
        convention = "geometric"
    elif wpl == 2:
        convention = "physical"
    else:
        convention = "experimental"

    return {
        "layers": layers_stats,
        "crosslayer_crossings_total": int(crosslayer_total),
        "crosslayer_crossings_convention": convention,
    }


def make_graph(
    *,
    k: int,
    positions: dict | None = None,
    output_dir: str | None = None,
    L: int = 2,
    edge_coupler_layer: int = 0,
    perimeter_layer: int | None = None,
    layer_pitch_um: float = 1.2,
    waveguides_per_link: int = 1,
    loss_crossing: float = DEFAULT_LOSS_CROSSING,
    loss_taper: float = DEFAULT_LOSS_TAPER,
    loss_interlayercrossing: float = DEFAULT_LOSS_INTERLAYERCROSSING,
    loss_intralayer_crosstalk: float | None = None,
    loss_interlayer_crosstalk: float | None = None,
    coherence_model: str = "incoherent_v1",
    polarization: str = "TE0_only",
    shape: str = "square",
) -> SiNInterconnectionGraph:
    """Construct a graph, defaulting positions to ``distribute_nodes(shape, k/4)``.

    See :class:`core.SiNInterconnectionGraph` for the multi-layer parameter
    contract; ``L`` / ``edge_coupler_layer`` / ``perimeter_layer`` /
    ``layer_pitch_um`` / ``waveguides_per_link`` plumb through unchanged.
    """
    if positions is None:
        if shape != "square":
            raise ValueError("Auto-positions only supported for shape='square'; "
                             "pass `positions` explicitly otherwise.")
        if k % 4 != 0:
            raise ValueError(f"Auto-positions need k divisible by 4; got k={k}.")
        positions = distribute_nodes(shape="square", nodes_per_side=k // 4, side_length=1)
    return SiNInterconnectionGraph(
        k=k,
        positions=positions,
        filepath=output_dir,
        L=L,
        edge_coupler_layer=edge_coupler_layer,
        perimeter_layer=perimeter_layer,
        layer_pitch_um=layer_pitch_um,
        waveguides_per_link=waveguides_per_link,
        loss_crossing=loss_crossing,
        loss_taper=loss_taper,
        loss_interlayercrossing=loss_interlayercrossing,
        loss_intralayer_crosstalk=loss_intralayer_crosstalk,
        loss_interlayer_crosstalk=loss_interlayer_crosstalk,
        coherence_model=coherence_model,
        polarization=polarization,
    )


def run_optimization(
    *,
    k: int = 12,
    optimizer: str = "dual_annealing",
    maxiter: int = 500,
    seed: int | None = 5859,
    optimizer_kwargs: dict | None = None,
    positions: dict | None = None,
    output_dir: str = "./assets/run/",
    L: int = 2,
    edge_coupler_layer: int = 0,
    perimeter_layer: int | None = None,
    layer_pitch_um: float = 1.2,
    waveguides_per_link: int = 1,
    loss_crossing: float = DEFAULT_LOSS_CROSSING,
    loss_taper: float = DEFAULT_LOSS_TAPER,
    loss_interlayercrossing: float = DEFAULT_LOSS_INTERLAYERCROSSING,
    loss_intralayer_crosstalk: float | None = None,
    loss_interlayer_crosstalk: float | None = None,
    coherence_model: str = "incoherent_v1",
    polarization: str = "TE0_only",
    compute_crosstalk: bool = True,
    crosstalk_max_hops: int = 3,
    crosstalk_threshold_db: float = -60.0,
    crosstalk_include_in_loss: bool = False,
    plot: bool = True,
    plot_style: str = "visualize",
    plot_kwargs: dict | None = None,
    save_json: bool = True,
    run_loss_analysis: bool = True,
    collect_statistics: bool = False,
    trace_stride: int = 1,
    fixed_layers: list[int] | None = None,
    eval_observer: Callable[..., None] | None = None,
    progress: str = "off",
    progress_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run an optimization end-to-end and return artifacts.

    Flow:
        1. build graph
        2. optimize
        3. ``save_subgraphs_to_json`` (writes positions, per-layer subgraphs,
           complete graph, and loss-analysis into a self-contained JSON)
        4. plotting loads that JSON via :class:`PlotData.from_json` — no
           direct graph access

    When ``save_json=False`` the snapshot is built directly from the live
    graph (:meth:`PlotData.from_graph`) so the rest of the pipeline still
    works in-memory only.

    When ``collect_statistics=True``, the optimizer writes a per-iter trace
    + phase timings to ``run_report.{json,md}`` in the same directory as
    ``subgraphsdata.json`` (REFACTOR_GOALS.md §1-1-a).

    M3 (§2-3 + §4 M3 附注): multi-layer parameters are accepted by both
    ``make_graph`` and ``run_optimization``. The default
    ``edge_coupler_layer=0`` preserves the legacy single-coupler-at-layer-0
    baseline; for L>=3 the recommended value is ``L // 2``. Aggregate
    ``crosslayer_crossings_total`` is reported with an explicit
    ``crosslayer_crossings_convention`` so downstream readers can tell
    physical-vs-geometric without inspecting ``waveguides_per_link``.

    M4 (§2-2): when ``compute_crosstalk`` is True AND at least one of
    ``loss_intralayer_crosstalk`` / ``loss_interlayer_crosstalk`` is a
    non-zero number, the rank-3 crosstalk tensor is computed once after
    optimization (default ``crosstalk_max_hops=3`` /
    ``crosstalk_threshold_db=-60.0``) and embedded in
    ``subgraphsdata.json`` under the top-level ``crosstalk`` key (per the
    schema in §3-1). ``crosstalk_include_in_loss=True`` is reserved for
    future use; today the engine is a pure post-optimization analysis and
    does not feed back into the optimizer loop.

    ``progress`` visualizes the optimization process (default ``"off"``
    = unchanged behavior, final result only):

      - ``"live"`` — a figure of the current best routing + loss curve,
        redrawn while the optimizer runs (GUI window; Jupyter output;
        or ``optimization_live.png`` rewritten on a headless machine).
      - ``"record"`` — every improvement is saved to
        ``optimization_history.json`` next to ``subgraphsdata.json``; after
        the run the whole process is rendered to
        ``optimization_progress.gif`` + ``optimization_progress.html``
        (interactive player), the final state to
        ``optimization_progress.pdf`` / ``.png`` (450 ppi still), and it
        is replayed when a display is available.

    ``progress_kwargs`` tunes it — see :class:`progress.ProgressOptions`
    (``interval``, ``show``, ``formats``, ``max_frames``, ``fps``, ``dpi``).
    Tracking is an ``eval_observer`` and never changes the optimization;
    ``"live"`` does add its drawing time to the optimizer wall time.

    Returns a dict with keys:
      - ``graph`` — the SiNInterconnectionGraph instance
      - ``best_layers`` — list of optimized layer assignments per edge
      - ``loss`` — final loss value reported by the optimizer
      - ``json_path`` — path to ``subgraphsdata.json`` (or None if not saved)
      - ``plot_data`` — the :class:`PlotData` snapshot used for plotting
        (always populated when ``plot`` or ``run_loss_analysis`` is True)
      - ``report_path`` — path to ``run_report.json`` (or None if
        ``collect_statistics`` is False)
      - ``crosstalk`` — the rank-3 tensor payload (or None if not computed)
      - ``progress`` — None when ``progress="off"``; otherwise
        ``{"mode", ...paths}`` with ``history`` / ``gif`` / ``html``
        (record) or ``live_image`` (live, headless) when written
    """
    if save_json and not output_dir:
        raise ValueError(
            "save_json=True requires a non-empty output_dir; "
            "got None. Pass output_dir or set save_json=False."
        )
    if collect_statistics and not output_dir:
        raise ValueError(
            "collect_statistics=True requires a non-empty output_dir "
            "(run_report.json is written next to subgraphsdata.json)."
        )
    progress_mode = normalize_progress_mode(progress)
    progress_options = ProgressOptions.from_kwargs(progress_kwargs)
    if progress_mode != "off" and fixed_layers is not None:
        raise ValueError(
            f"progress={progress_mode!r} needs an optimizer run to show; "
            "it cannot be combined with fixed_layers."
        )
    if progress_mode == "record" and not output_dir:
        raise ValueError(
            "progress='record' requires a non-empty output_dir "
            "(optimization_history.json is written next to subgraphsdata.json)."
        )

    # M8 parity-driver path (REFACTOR_GOALS.md §6 T6). When `fixed_layers`
    # is provided we skip the optimizer entirely and apply the assignment
    # directly. Mirrors the C++ `SINIC_FIXED_LAYERS_JSON` env var. The
    # optimizer-trajectory cross-impl delta is out of scope; this driver
    # is for bit-exact per-edge / per-loss comparison on a shared layer
    # assignment.
    fixed_layers_mode = fixed_layers is not None
    if fixed_layers_mode and collect_statistics:
        # Note: `run_report.schema.json` does NOT enforce `trace minItems
        # >= 1` (it's an open array). The raise is semantic: a "skipped
        # the optimizer 0 times" run_report is misleading and would
        # bypass the `summary.iterations` / `best_at_iter` / etc. fields
        # that downstream readers expect. Mirrored on the C++ side via
        # `main.cpp::main`'s SINIC_FIXED_LAYERS_JSON +
        # SINIC_WRITE_RUN_REPORT mutual-exclusion guard.
        raise ValueError(
            "fixed_layers and collect_statistics are mutually exclusive: "
            "skipping the optimizer leaves trace[] empty and the resulting "
            "run_report would be semantically misleading."
        )
    if collect_statistics:
        run_id = (
            f"{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%S')}"
            f"_k{k}_{optimizer}_seed{seed}"
        )
        # NOTE: keys here are snake_case, deliberately not the Title-Case
        # "General Parameters" naming used in subgraphsdata.json. The
        # subgraphsdata casing is a legacy of the original SiN3DIC code; the
        # new run_report.json gets a clean schema. §3-2 docstring "mirror of
        # General Parameters" is intentional shape (same fields), not a
        # literal key-rename mirror.
        config = {
            "k": k,
            "L": L,
            "edge_coupler_layer": edge_coupler_layer,
            "perimeter_layer": (
                edge_coupler_layer if perimeter_layer is None else perimeter_layer
            ),
            "layer_pitch_um": layer_pitch_um,
            "waveguides_per_link": waveguides_per_link,
            "coherence_model": coherence_model,
            "polarization": polarization,
            "optimizer": optimizer,
            "maxiter": maxiter,
            "seed": seed,
            "loss_crossing": loss_crossing,
            "loss_taper": loss_taper,
            "loss_interlayercrossing": loss_interlayercrossing,
            "loss_intralayer_crosstalk": loss_intralayer_crosstalk,
            "loss_interlayer_crosstalk": loss_interlayer_crosstalk,
            "crosstalk_max_hops": crosstalk_max_hops,
            "crosstalk_threshold_db": crosstalk_threshold_db,
            "crosstalk_include_in_loss": crosstalk_include_in_loss,
        }
        sink: RunRecorder | NullSink = RunRecorder(
            run_id=run_id, config=config, trace_stride=trace_stride
        )
    else:
        sink = NullSink()

    with PhaseTimer(sink, "graph_build"):
        graph = make_graph(
            k=k,
            positions=positions,
            output_dir=output_dir,
            L=L,
            edge_coupler_layer=edge_coupler_layer,
            perimeter_layer=perimeter_layer,
            layer_pitch_um=layer_pitch_um,
            waveguides_per_link=waveguides_per_link,
            loss_crossing=loss_crossing,
            loss_taper=loss_taper,
            loss_interlayercrossing=loss_interlayercrossing,
            loss_intralayer_crosstalk=loss_intralayer_crosstalk,
            loss_interlayer_crosstalk=loss_interlayer_crosstalk,
            coherence_model=coherence_model,
            polarization=polarization,
        )

    # REFACTOR_GOALS.md §3-2 ``initial_crossing_count_ms``. Phase A
    # (crossing-pair index build, §2-1) is lazy by default; surface it as a
    # named phase so the run report distinguishes graph allocation from
    # one-time geometry, which is the dominant cost at k=160.
    with PhaseTimer(sink, "initial_crossing_count"):
        graph.build_crossings_index()

    if fixed_layers_mode:
        # Skip the optimizer entirely — apply the fixed assignment and
        # synthesize a minimal result object exposing the same `.fun` /
        # `.best_layers` interface that the downstream JSON-writer reads.
        # NOTE: any new General-Parameters field added to the writer
        # below (via **kwargs to save_subgraphs_to_json) MUST be mirrored
        # on the C++ side in routing_cpp_rebuild/src/io.cpp's
        # `subgraphsdata_v1x_to_json` OR added to
        # `tests/parity_harness.py::KNOWN_GP_ASYMMETRIES`. The harness
        # comparator fails loudly on unknown drift, but a developer
        # surprised by the failure should look here first.
        fixed_list = list(fixed_layers)
        n_edges = graph.G.number_of_edges()
        if len(fixed_list) != n_edges:
            raise ValueError(
                f"fixed_layers has {len(fixed_list)} entries but graph "
                f"has {n_edges} edges"
            )
        loss_value = float(graph.loss_function(fixed_list))
        graph.apply_optimization_result(fixed_list)

        class _FixedResult:
            fun = loss_value
            best_layers = fixed_list

        result = _FixedResult()
        tracker = None
    else:
        tracker: ProgressTracker | None = None
        if progress_mode != "off":
            tracker = ProgressTracker(
                graph,
                mode=progress_mode,
                options=progress_options,
                out_dir=graph.filepath,
                run={
                    "optimizer": optimizer,
                    "maxiter": maxiter,
                    "seed": seed,
                    "loss_crossing": loss_crossing,
                    "loss_taper": loss_taper,
                    "loss_interlayercrossing": loss_interlayercrossing,
                },
            )
        optimizer_cls = get_optimizer(optimizer)
        opt = optimizer_cls(
            graph,
            seed=seed,
            sink=sink,
            eval_observer=chain_observers(eval_observer, tracker),
            **(optimizer_kwargs or {}),
        )

        try:
            with PhaseTimer(sink, "optimization_wall"):
                result = opt.optimize(maxiter=maxiter)
        except KeyboardInterrupt:
            # Keep what a long recorded run has found so far.
            partial = tracker.write_history() if tracker is not None else None
            if partial:
                print(
                    f"[progress] interrupted — {len(tracker.frames)} improvements "
                    f"saved to {partial}; replay with "
                    f"`python -m routing_py_rebuild replay --history {partial}`"
                )
            raise
        if tracker is not None:
            tracker.finish(result.best_layers, result.fun)

    json_path: str | None = None
    plot_data: PlotData | None = None

    # M4 (§2-2): if both coefficients are zero/None the engine short-circuits
    # to an all-None tensor — see ``crosstalk.compute_crosstalk_tensor``. We
    # check the same condition here so we don't even pay the PhaseTimer
    # overhead for a no-op call. ``compute_crosstalk=False`` is the explicit
    # opt-out for benchmarks that want to skip the analysis entirely.
    has_any_crosstalk = (
        (loss_intralayer_crosstalk is not None and loss_intralayer_crosstalk != 0.0)
        or (loss_interlayer_crosstalk is not None and loss_interlayer_crosstalk != 0.0)
    )
    crosstalk_payload: dict | None = None

    if save_json:
        json_path = os.path.join(graph.filepath, "subgraphsdata.json")
        # M2 split: ``loss_analysis_ms`` covers the analyze_loss compute;
        # ``json_write_ms`` covers only serialization. ``skip_analysis=True``
        # below avoids re-running analyze_loss inside save_subgraphs_to_json.
        with PhaseTimer(sink, "loss_analysis"):
            graph.analyze_loss()
        if compute_crosstalk and has_any_crosstalk:
            with PhaseTimer(sink, "crosstalk_analysis"):
                crosstalk_payload = compute_crosstalk_tensor(
                    graph,
                    max_hops=crosstalk_max_hops,
                    threshold_db=crosstalk_threshold_db,
                )
        with PhaseTimer(sink, "json_write"):
            graph.save_subgraphs_to_json(
                json_path,
                skip_analysis=True,
                crosstalk=crosstalk_payload,
                optimizer=optimizer,
                maxiter=maxiter,
                seed=seed,
                best_loss=result.fun,
            )
        if plot or run_loss_analysis:
            plot_data = PlotData.from_json(json_path)
    elif plot or run_loss_analysis:
        # PlotData.from_graph is read-only; the orchestrator (not plotting)
        # is responsible for getting the graph into an analyzed state first.
        with PhaseTimer(sink, "loss_analysis"):
            graph.analyze_loss()
        if compute_crosstalk and has_any_crosstalk:
            with PhaseTimer(sink, "crosstalk_analysis"):
                crosstalk_payload = compute_crosstalk_tensor(
                    graph,
                    max_hops=crosstalk_max_hops,
                    threshold_db=crosstalk_threshold_db,
                )
        plot_data = PlotData.from_graph(graph)
        if crosstalk_payload is not None:
            # ``from_graph`` doesn't see ``crosstalk`` (it isn't on the
            # graph object) — surface it explicitly so plotting paths
            # that operate on the in-memory PlotData (e.g. unit tests,
            # interactive notebooks) match the from-JSON path.
            plot_data.general_params["crosstalk"] = crosstalk_payload
    elif collect_statistics:
        # M3 / opus-review P0-1: when neither save_json nor any plot branch
        # fired but the caller still asked for statistics, analyze_loss
        # must run so ``sub_G`` is populated for the aggregate writer
        # below. Otherwise ``crosslayer_crossings_total`` /
        # ``crosslayer_crossings_convention`` silently drop out of
        # ``run_report.json`` despite the schema permitting their
        # omission. Tested by
        # ``tests/test_multilayer.py::test_collect_statistics_no_save_json_still_emits_aggregate``.
        with PhaseTimer(sink, "loss_analysis"):
            graph.analyze_loss()
        # M4 dual-review (opus + codex P1): mirror the crosstalk pass
        # so a ``collect_statistics=True`` + ``save_json=False`` +
        # ``plot=False`` + ``run_loss_analysis=False`` caller still
        # gets ``crosstalk_analysis_ms`` in ``run_report.timings`` and
        # a non-None ``res["crosstalk"]`` return value. Pre-fix the
        # tensor was silently dropped — same class of bug as the M3
        # P0-1 aggregate-stats fix immediately above.
        if compute_crosstalk and has_any_crosstalk:
            with PhaseTimer(sink, "crosstalk_analysis"):
                crosstalk_payload = compute_crosstalk_tensor(
                    graph,
                    max_hops=crosstalk_max_hops,
                    threshold_db=crosstalk_threshold_db,
                )

    if plot and plot_data is not None:
        with PhaseTimer(sink, "plotting"):
            visualize_layers(plot_data, style=plot_style, **(plot_kwargs or {}))

    if run_loss_analysis and plot_data is not None:
        # ``loss_analysis_plot_ms`` is chart rendering only; the analyze_loss
        # compute is timed separately under ``loss_analysis_ms`` (M2 split,
        # REFACTOR_GOALS.md §3-2).
        with PhaseTimer(sink, "loss_analysis_plot"):
            visualize_loss_analysis(plot_data)

    if tracker is not None and progress_mode == "record":
        with PhaseTimer(sink, "progress_history_write"):
            tracker.write_history()
        with PhaseTimer(sink, "progress_replay"):
            tracker.write_replay()
        for kind, path in tracker.outputs.items():
            print(f"[progress] {kind}: {path}")

    report_path: str | None = None
    if collect_statistics:
        # §1-1-a item 3 + §4 M3 附注: push per-layer aggregate stats into
        # the recorder before finalize. The aggregate dict carries
        # ``crosslayer_crossings_total`` (computed with the wpl multiplier)
        # AND ``crosslayer_crossings_convention`` ("physical" / "geometric")
        # so the run report explicitly labels which scale is in use.
        if graph.sub_G:
            sink.set_aggregate(**_extract_aggregate_stats(graph))
        report = sink.finalize()
        report_dir = graph.filepath if graph.filepath else output_dir
        report_path, _md_path = write_run_report(report, report_dir)

    progress_info: dict[str, str] | None = None
    if tracker is not None:
        # Last: a replay window / held live window blocks until closed, and
        # every file is already on disk by then.
        tracker.present()
        progress_info = {"mode": progress_mode, **tracker.outputs}

    return {
        "graph": graph,
        "best_layers": result.best_layers,
        "loss": result.fun,
        "json_path": json_path,
        "plot_data": plot_data,
        "report_path": report_path,
        "crosstalk": crosstalk_payload,
        "progress": progress_info,
    }


__all__ = [
    "make_graph",
    "run_optimization",
    "plot_from_json",
    "visualize_layers",
    "visualize_ring",
    "visualize_loss_analysis",
]
