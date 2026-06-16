"""Ring + complete-mesh diagram (XPU-style outer arrows)."""
from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ._colormap import DEFAULT_COLORMAP, NODE_FILL_COLOR, crossings_color
from ._style import FONT_SIZE, LINE_WIDTH, apply_rcparams


def visualize_ring(
    plot_data,
    *,
    num_nodes: int | None = None,
    use_current_positions: bool = False,
    layout: str = "circle",
    node_label: str = "{n} XPU\nnode",
    radius: float = 1.0,
    out_path: str | None = None,
    cmap=None,
):
    """Draw outer ring + all-to-all internal mesh + outward bidirectional arrows.

    Parameters
    ----------
    plot_data : :class:`plot_data.PlotData`
        Self-contained snapshot. ``k`` and ``positions`` are read; edge
        color is driven by the complete-graph crossing counts stored on
        ``plot_data.complete_graph``.
    num_nodes : int | None
        Override the snapshot's ``k`` for layout purposes (uses an empty
        complete graph if different).
    use_current_positions : bool
        If True, use ``plot_data.positions`` as-is (no recompute).
    layout : {"circle", "square"}
    node_label : str
        Format string with ``{n}`` for the node index.
    radius : float
    out_path : str | None
        If given, an editable vector ``.pdf`` is saved (the ``.pdf``
        extension is appended); alpha is preserved.
    cmap : matplotlib.colors.Colormap | None
    """
    apply_rcparams()
    if cmap is None:
        cmap = DEFAULT_COLORMAP

    k = num_nodes or plot_data.k
    if k == plot_data.k:
        # use the complete-graph baseline so edge colors reflect the same
        # crossing count the layer plots use (per-layer subgraphs have
        # smaller, layer-local counts that would mis-color this view)
        G = plot_data.complete_graph
    else:
        G = nx.complete_graph(k)

    if use_current_positions:
        ring_pos = {n: tuple(p) for n, p in plot_data.positions.items()}
    elif layout.lower() == "circle":
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        ring_pos = {i: (radius * np.cos(a), radius * np.sin(a)) for i, a in enumerate(angles)}
    elif layout.lower() == "square":
        ring_pos = {}
        for i in range(k):
            t = i / k * 4
            if t < 1:
                x, y = -radius + 2 * radius * t, radius
            elif t < 2:
                x, y = radius, radius - 2 * radius * (t - 1)
            elif t < 3:
                x, y = radius - 2 * radius * (t - 2), -radius
            else:
                x, y = -radius, -radius + 2 * radius * (t - 3)
            ring_pos[i] = (x, y)
    else:
        raise ValueError("layout must be 'circle' or 'square'")

    crosses = nx.get_edge_attributes(G, "crossings")
    if crosses:
        vmax = max(crosses.values()) or 1
        norm = plt.Normalize(vmin=0, vmax=vmax, clip=True)
        edge_cols = [crossings_color(crosses[e], norm, cmap) for e in G.edges]
    else:
        edge_cols = ["tab:orange"] * G.number_of_edges()

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0)

    nx.draw_networkx_nodes(G, pos=ring_pos, ax=ax,
                           node_shape="o", node_color=NODE_FILL_COLOR, node_size=1300)
    nx.draw_networkx_edges(G, pos=ring_pos, ax=ax,
                           edge_color=edge_cols, width=LINE_WIDTH,
                           arrows=True, arrowstyle="<->",
                           min_source_margin=17, min_target_margin=17)

    arrow_color = "tab:orange"
    cx = sum(p[0] for p in ring_pos.values()) / k
    cy = sum(p[1] for p in ring_pos.values()) / k
    inner_off = 0.25 * radius
    outer_off = 0.55 * radius

    for (x, y) in ring_pos.values():
        dx, dy = x - cx, y - cy
        length = (dx * dx + dy * dy) ** 0.5 or 1e-9
        ux, uy = dx / length, dy / length
        start = (x + ux * inner_off, y + uy * inner_off)
        end = (x + ux * outer_off, y + uy * outer_off)
        ax.add_patch(mpatches.FancyArrowPatch(
            start, end, arrowstyle="<->", mutation_scale=12,
            linewidth=LINE_WIDTH, color=arrow_color, shrinkA=0, shrinkB=0,
        ))

    labels = {i: node_label.format(n=i) for i in range(k)}
    # label color follows the standard text color (black); the previous
    # white-on-navy combo was tied to the old dark-blue node fill.
    nx.draw_networkx_labels(G, pos=ring_pos, labels=labels,
                            font_size=FONT_SIZE, ax=ax)

    ax.axis("off")
    ax.set_aspect("equal")

    if out_path:
        base, _ = os.path.splitext(str(out_path))
        fig.savefig(base + ".pdf", bbox_inches="tight")
        print(f"[visualize_ring] saved to {base}.pdf")
        plt.close(fig)
    else:
        plt.show()
