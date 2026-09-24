"""Plotting routines for SiN routing graphs.

All plotting consumes a :class:`plot_data.PlotData` snapshot — this
package never imports :mod:`core` and never mutates a
:class:`SiNInterconnectionGraph` instance. The contract is:

    optimizer → save_subgraphs_to_json → PlotData.from_json → plot

:func:`plot_from_json` is the convenience entry that wires the JSON load
and the style dispatch together.
"""
from __future__ import annotations

from ._colormap import (
    CMAP_LOWER,
    CMAP_SCALE,
    CMAP_UPPER,
    DEFAULT_COLORMAP,
    NODE_FILL_COLOR,
    crossings_color,
    get_colormap,
)
from .convergence import plot_convergence, plot_timings
from .crosstalk_heatmap import plot_crosstalk_heatmap
from .from_json import plot_from_json
from .layer_edges import plot_layer_edges
from .layers import list_styles, register_style, visualize_layers
from .loss_analysis import visualize_loss_analysis
from .plot_data import PlotData
from .progress_figure import ProgressData
from .progress_replay import render_progress_replay, show_progress_replay
from .ring import visualize_ring

__all__ = [
    "PlotData",
    "visualize_layers",
    "register_style",
    "list_styles",
    "visualize_ring",
    "plot_layer_edges",
    "visualize_loss_analysis",
    "plot_from_json",
    # run_report renderers (REFACTOR_GOALS.md §1-1-a)
    "plot_convergence",
    "plot_timings",
    # crosstalk renderer (REFACTOR_GOALS.md §2-2 / M4)
    "plot_crosstalk_heatmap",
    # optimization-process replay (run_optimization(progress="record"))
    "ProgressData",
    "render_progress_replay",
    "show_progress_replay",
    # color palette
    "DEFAULT_COLORMAP",
    "NODE_FILL_COLOR",
    "CMAP_LOWER",
    "CMAP_UPPER",
    "CMAP_SCALE",
    "get_colormap",
    "crossings_color",
]
