"""Refactored SiN routing package.

Public entry points:
  - :func:`run_optimization` — graph → optimize → JSON → plot
  - :func:`plot_from_json`   — reload JSON → plot
  - :class:`SiNInterconnectionGraph` — core data model

Optimizers are pluggable via :func:`optimizers.register_optimizer`.
Plotting styles are pluggable via the registry in :mod:`plotting.layers`.
"""
from __future__ import annotations

from .api import (
    make_graph,
    run_optimization,
)
from .core import SiNInterconnectionGraph
from .crosstalk import compute_crosstalk_tensor
from .optimizers import (
    DifferentialEvolutionOptimizer,
    DualAnnealingOptimizer,
    OptimizationResult,
    Optimizer,
    get_optimizer,
    register_optimizer,
)
from .plotting import (
    DEFAULT_COLORMAP,
    NODE_FILL_COLOR,
    PlotData,
    crossings_color,
    get_colormap,
    list_styles,
    plot_convergence,
    plot_crosstalk_heatmap,
    plot_from_json,
    plot_layer_edges,
    plot_timings,
    register_style,
    visualize_layers,
    visualize_loss_analysis,
    visualize_ring,
)
from .positions import (
    POSITIONS_12_NODES,
    distribute_nodes,
    distribute_nodes_around_circle,
    distribute_nodes_around_partial_rectangle,
    distribute_nodes_around_polygon,
    distribute_nodes_around_rectangle,
    distribute_nodes_around_square,
    distribute_nodes_around_triangle,
)
from .statistics import (
    IterEvent,
    NullSink,
    PhaseTimer,
    RunRecorder,
    StatsSink,
    write_run_report,
)

__all__ = [
    "SiNInterconnectionGraph",
    "Optimizer",
    "OptimizationResult",
    "DualAnnealingOptimizer",
    "DifferentialEvolutionOptimizer",
    "get_optimizer",
    "register_optimizer",
    "make_graph",
    "run_optimization",
    "PlotData",
    "plot_from_json",
    "visualize_layers",
    "register_style",
    "list_styles",
    "visualize_ring",
    "visualize_loss_analysis",
    "plot_layer_edges",
    "plot_convergence",
    "plot_timings",
    "plot_crosstalk_heatmap",
    "compute_crosstalk_tensor",
    "distribute_nodes",
    "distribute_nodes_around_square",
    "distribute_nodes_around_rectangle",
    "distribute_nodes_around_circle",
    "distribute_nodes_around_triangle",
    "distribute_nodes_around_polygon",
    "distribute_nodes_around_partial_rectangle",
    "POSITIONS_12_NODES",
    "DEFAULT_COLORMAP",
    "NODE_FILL_COLOR",
    "get_colormap",
    "crossings_color",
    # statistics (REFACTOR_GOALS.md §1-1-a)
    "IterEvent",
    "StatsSink",
    "NullSink",
    "RunRecorder",
    "PhaseTimer",
    "write_run_report",
]
