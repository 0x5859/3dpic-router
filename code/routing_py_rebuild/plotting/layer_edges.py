"""Single-layer black-line subset plot."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import networkx as nx

from ._style import FONT_SIZE, LINE_WIDTH, apply_rcparams


def plot_layer_edges(
    plot_data,
    layer: int = 0,
    nodes_subset: list[int] | None = None,
    *,
    out_path: str | None = None,
    node_params: dict | None = None,
    panel_size: tuple = (4, 4, "cm"),
):
    """Draw layer ``layer`` in solid black (optionally restricted to ``nodes_subset``).

    Parameters
    ----------
    plot_data : :class:`plot_data.PlotData`
    layer : int
        Index into ``plot_data.layers``.
    nodes_subset : list[int] | None
        Restrict drawn edges to those whose endpoints lie in this set.
        Nodes outside the highlighted set are still drawn in faded grey
        for context.
    """
    apply_rcparams()
    if layer < 0 or layer >= len(plot_data.layers):
        raise IndexError(
            f"layer={layer} out of range; plot_data has {len(plot_data.layers)} layers"
        )

    layer_graph = plot_data.layers[layer]
    all_edges = list(layer_graph.edges())
    if nodes_subset is not None:
        nodes_subset = set(nodes_subset)
        all_edges = [(u, v) for u, v in all_edges if u in nodes_subset and v in nodes_subset]

    G_sel = nx.Graph()
    G_sel.add_edges_from(all_edges)
    highlight_nodes = set(G_sel.nodes())

    cm2in = lambda v: v / 2.54
    pt2in = lambda v: v / 72.0
    w, h, unit = panel_size
    if unit == "cm":
        fig_size = (cm2in(w), cm2in(h))
    elif unit == "pt":
        fig_size = (pt2in(w), pt2in(h))
    else:
        fig_size = (w, h)

    pos_all = dict(plot_data.positions)
    all_nodes = set(range(plot_data.k))
    missing = [n for n in all_nodes if n not in pos_all]
    if missing:
        sub = nx.Graph()
        sub.add_nodes_from(missing)
        fallback_pos = nx.spring_layout(sub, seed=42)
        pos_all.update(fallback_pos)

    style = {
        "node_shape": "o",
        "node_size": 400,
        "node_color": "#FFFFFF",
        "edgecolors": "black",
        "with_labels": True,
        "label_fontsize": FONT_SIZE,
    }
    if node_params:
        style.update(node_params)

    labels_flag = style.pop("with_labels", True)
    label_fs = style.pop("label_fontsize", FONT_SIZE)

    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    ax.set_facecolor("none")

    nx.draw_networkx_edges(
        G_sel, pos={n: pos_all[n] for n in G_sel.nodes()},
        ax=ax, edge_color="black", width=LINE_WIDTH,
    )
    nx.draw_networkx_nodes(
        G_sel, pos={n: pos_all[n] for n in highlight_nodes},
        ax=ax, **style,
    )

    other_nodes = all_nodes - highlight_nodes
    if other_nodes:
        faded = style.copy()
        faded["node_color"] = "#DDDDDD"
        ctx_graph = nx.Graph()
        ctx_graph.add_nodes_from(other_nodes)
        nx.draw_networkx_nodes(
            ctx_graph, pos={n: pos_all[n] for n in other_nodes},
            ax=ax, **faded,
        )

    if labels_flag:
        all_graph = nx.Graph()
        all_graph.add_nodes_from(all_nodes)
        nx.draw_networkx_labels(all_graph, pos=pos_all, ax=ax, font_size=label_fs)

    xs = [p[0] for p in pos_all.values()]
    ys = [p[1] for p in pos_all.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    side_len = max(xmax - xmin, ymax - ymin)
    pad = 0.05 * side_len
    ax.set_xlim(xmin - pad, xmin + side_len + pad)
    ax.set_ylim(ymin - pad, ymin + side_len + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if out_path is None:
        tag = f"subset_{len(highlight_nodes)}nodes" if nodes_subset else "subset"
        base_dir = plot_data.filepath or "."
        out_path = os.path.join(base_dir, f"layer{layer}_{tag}.pdf")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, facecolor="none")
    plt.close(fig)
    print(f"[plot_layer_edges] saved figure to {out_path}")
