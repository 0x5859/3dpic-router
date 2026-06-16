"""Convenience: load a saved subgraph JSON and dispatch to a plotting style.

This module does not import :mod:`core`; it loads JSON into a
:class:`plot_data.PlotData` and hands it to the plot styles directly.
"""
from __future__ import annotations

import os

from .layers import visualize_layers
from .loss_analysis import visualize_loss_analysis
from .plot_data import PlotData


def plot_from_json(
    json_path: str,
    *,
    style: str = "visualize",
    out_dir: str | None = None,
    also_loss_analysis: bool = False,
    **style_kwargs,
) -> PlotData:
    """Load JSON → :class:`PlotData` → render.

    Parameters
    ----------
    json_path : str
        Path to a ``subgraphsdata.json`` previously written by
        :meth:`core.SiNInterconnectionGraph.save_subgraphs_to_json`.
        The file is self-contained — positions, per-layer subgraphs, the
        complete-graph baseline, loss-analysis, and loss parameters are
        all read from here.
    style : str
        Name of a registered layer-plotting style (default ``"visualize"``).
    out_dir : str | None
        Output directory for figures. Defaults to the JSON file's directory.
    also_loss_analysis : bool
        If True, additionally render the loss-analysis bars.
    **style_kwargs
        Forwarded to the chosen style function (see e.g.
        :func:`plotting.layers._visualize`).

    Returns
    -------
    PlotData
        The loaded snapshot — useful when the caller wants to render
        additional figures off the same data.
    """
    plot_data = PlotData.from_json(json_path)
    effective_out_dir = out_dir or plot_data.filepath

    visualize_layers(plot_data, style=style, out_dir=effective_out_dir, **style_kwargs)
    if also_loss_analysis:
        loss_out = (
            os.path.join(effective_out_dir, "lossanalysis.pdf")
            if effective_out_dir
            else None
        )
        visualize_loss_analysis(plot_data, out_path=loss_out)
    return plot_data
