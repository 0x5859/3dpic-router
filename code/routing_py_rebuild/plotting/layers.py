"""Layer visualization (single style: ``visualize``).

The shipped style renders a 2-row figure:

  * row 0 — one **square** edge-graph panel per subgraph
    (``L`` layer rows + 1 complete-graph), nodes on the racetrack
    perimeter, every node number labelled, same-side non-adjacent
    edges bowed (arc3) but anchored on the node;
  * row 1 — the matching crossing-count bar charts, equal-spaced
    integer x-ticks, one shared bar width.

Input contract
--------------
Every style takes a :class:`plot_data.PlotData` as its first argument —
plotting never imports :mod:`core` and never mutates a live graph. The
only output is ``layers_combined.pdf`` under the resolved ``out_dir``.

Registry note
-------------
The plot-style registry (:func:`register_style`, :func:`list_styles`) is
kept so downstream users can add styles without touching the package.
Only ``visualize`` ships built-in.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ._colormap import DEFAULT_COLORMAP, NODE_FILL_COLOR, crossings_color
from ._style import FONT_SIZE, LINE_WIDTH, apply_rcparams

# ---------------------------------------------------------------------------
# Tunable knobs
# ---------------------------------------------------------------------------
CURVE_FACTOR = 0.27     # boundary-chord arc bow, scaled by span / nodes on that line
ON_LINE_TOL = 1e-3      # chord within this fraction of the span of a node = runs through it
XTICK_NBINS = 12        # max equal-spaced integer x-ticks per bar chart
BAR_WIDTH_FRAC = 0.8    # bar fill vs. median neighbour spacing (<1 => gaps)
HIST_H_FRAC = 0.5       # bar-chart row height as a fraction of the graph panel
_FIG_PANEL_W_CM = 9.0   # per-graph panel width at k == _FIG_SCALE_REF_K
_FIG_SCALE_REF_K = 12
_FIG_SCALE_MAX = 3.0


# ---------------------------------------------------------------------------
# Style registry
# ---------------------------------------------------------------------------
_STYLES: dict[str, Callable[..., object]] = {}


def register_style(name: str):
    """Decorator: register ``fn(plot_data, **kwargs)`` as a layer style."""

    def _wrap(fn):
        _STYLES[name] = fn
        return fn

    return _wrap


def list_styles() -> list[str]:
    """Return the names of all registered layer-plotting styles."""
    return sorted(_STYLES.keys())


def visualize_layers(plot_data, *, style: str = "visualize", **kwargs):
    """Dispatch to a named layer-visualization style.

    Parameters
    ----------
    plot_data : :class:`plot_data.PlotData`
        Self-contained snapshot built from JSON (or, for tests, from a
        live graph via :meth:`PlotData.from_graph`).
    style : str
        Registered style name; defaults to ``"visualize"``.
    """
    if style not in _STYLES:
        raise KeyError(
            f"Unknown style '{style}'. Available: {sorted(_STYLES.keys())}"
        )
    return _STYLES[style](plot_data, **kwargs)


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------
def _edge_bends(positions, edges, nodes=None) -> list[float]:
    """``arc3`` ``rad`` for each of ``edges`` (same order): nonzero for
    chords that run along the boundary through other nodes, 0.0 for the rest.

    Works for any node layout (square, rectangle, triangle, polygon,
    circle, arbitrary coordinates). An edge is bowed when its straight
    chord passes through another node of ``nodes`` (within
    ``ON_LINE_TOL`` of the layout span) — its ends sit on one straight
    stretch of the boundary with nodes in between, where the chord would
    hide behind those nodes and the stretch's other chords. The bow is
    ``CURVE_FACTOR * (nodes strictly between + 1) / (nodes on that
    line)``, signed so that matplotlib's ``arc3`` (control point at the
    chord midpoint + ``rad * (dy, -dx)``) bends toward the layout's
    interior (the node centroid). On the square layouts this is exactly
    the former top / right / bottom / left side rule.
    """
    nodes = list(positions) if nodes is None else list(nodes)
    pts = np.array([positions[n] for n in nodes], dtype=float).reshape(-1, 2)
    centroid = pts.mean(axis=0)
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-12)
    tol = ON_LINE_TOL * span
    bends: list[float] = []
    for u, v in edges:
        p0 = np.asarray(positions[u], dtype=float)
        p2 = np.asarray(positions[v], dtype=float)
        d = p2 - p0
        length2 = float(d @ d)
        if length2 == 0.0:
            bends.append(0.0)
            continue
        rel = pts - p0
        on_line = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) <= tol * math.sqrt(length2)
        t = (rel @ d) / length2
        between = int(np.count_nonzero(on_line & (t > 1e-9) & (t < 1 - 1e-9)))
        if between == 0:
            bends.append(0.0)
            continue
        rad = CURVE_FACTOR * ((between + 1) / int(np.count_nonzero(on_line)))
        outward = float((centroid - (p0 + p2) / 2.0) @ np.array([d[1], -d[0]])) < 0.0
        bends.append(-rad if outward else rad)
    return bends


# ---------------------------------------------------------------------------
# Single sources of truth
# ---------------------------------------------------------------------------
def _node_size(k: int) -> float:
    """Node marker size (pt**2 — an AREA), the ONLY definition of it.

    Radius is 2/3 of the original 280-area baseline (area x (2/3)**2 =
    4/9), easing off further for large k. Used for both the node markers
    and the edge-shrink radius so arcs meet the node boundary exactly:
    networkx shrinks ``arrows=True`` (FancyArrowPatch) edges by the
    radius implied by ``node_size`` (default 300). If the markers are
    smaller and that is not forwarded, the arc ends are clipped to a
    phantom size-300 radius and float clear of the node. Keeping one
    definition and sharing it makes that desync structurally impossible.
    """
    return max(9.0, 280.0 * (24.0 / max(k, 24)) * (4.0 / 9.0))


def _label_fontsize(k: int) -> float:
    """All node numbers are always drawn; only the font shrinks with k."""
    return min(FONT_SIZE, max(3.0, FONT_SIZE * math.sqrt(24.0 / max(k, 1))))


def _layer_bar_width(layers) -> float:
    """One bar width for every row, from the layer rows' pooled distinct
    crossing values (median neighbour gap x BAR_WIDTH_FRAC). The
    complete-graph row reuses it so all rows share the same bar width."""
    vals = sorted({v for lg in layers
                   for v in nx.get_edge_attributes(lg, "crossings").values()})
    gaps = [b - a for a, b in zip(vals, vals[1:])]
    return BAR_WIDTH_FRAC * (float(np.median(gaps)) if gaps else 1.0)


# ---------------------------------------------------------------------------
# Panel drawers
# ---------------------------------------------------------------------------
def _draw_curved_edges(ax, g, positions, edge_cols, node_size):
    """Draw edges; chords that run along the boundary through other nodes
    bow inward (arc3, see :func:`_edge_bends`) while every endpoint stays
    anchored on its node.

    ``node_size`` is forwarded to every ``draw_networkx_edges`` call so
    the FancyArrowPatch shrink matches the real node radius
    (boundary-anchored). It is supplied by :func:`_draw_graph_panel`
    from the single :func:`_node_size`; do not call this directly.
    """
    edgelist = list(g.edges())
    bends = _edge_bends(positions, edgelist, nodes=g.nodes())
    for (u, v), ec, rad in zip(edgelist, edge_cols, bends, strict=True):
        if rad == 0.0:
            nx.draw_networkx_edges(
                g, pos=positions, edgelist=[(u, v)], width=LINE_WIDTH,
                edge_color=[ec], ax=ax, node_size=node_size,
            )
        else:
            nx.draw_networkx_edges(
                g, pos=positions, edgelist=[(u, v)], width=LINE_WIDTH,
                edge_color=[ec], connectionstyle=f"arc3,rad={rad}",
                arrows=True, arrowstyle="-", ax=ax, node_size=node_size,
            )


def _draw_graph_panel(ax, g, pos, *, k, norm, cmap):
    """One square racetrack panel: anchored curved edges + nodes + every
    node label. Owns node_size (the single :func:`_node_size`) and hands
    the same value to the edge drawer, so the two cannot desync."""
    ns = _node_size(k)
    edge_cols = [crossings_color(g.edges[e]["crossings"], norm, cmap)
                 for e in g.edges()]
    _draw_curved_edges(ax, g, pos, edge_cols, ns)
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, node_shape="o",
                           node_size=ns, node_color=NODE_FILL_COLOR)
    nx.draw_networkx_labels(g, pos=pos, ax=ax, font_size=_label_fontsize(k))
    ax.axis("off")
    ax.set_box_aspect(1)  # square plotting region


def _draw_hist_panel(ax, g, *, norm, cmap, bar_width, lo, hi, max_count,
                     log_y):
    """One crossing-count bar chart: shared bar width, equal-spaced
    integer x-ticks spanning ~[0, max]."""
    counts = Counter(nx.get_edge_attributes(g, "crossings").values())
    xs = sorted(counts)
    ax.bar(xs, [counts[x] for x in xs], width=bar_width,
           color=[crossings_color(x, norm, cmap) for x in xs])

    pad = max(1.0, (hi - lo) * 0.02)
    ax.set_xlim(-pad, hi + pad)  # min ~0; no end-bar clip
    ax.xaxis.set_major_locator(MaxNLocator(nbins=XTICK_NBINS, integer=True))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{int(round(v))}")
    )
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(0.8, max_count * 1.3)
    else:
        ax.set_ylim(0, max_count + 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("crossings per edge", fontsize=FONT_SIZE)
    ax.set_ylabel("edge count", fontsize=FONT_SIZE)
    for sp in ax.spines.values():
        sp.set_linewidth(LINE_WIDTH)


# ---------------------------------------------------------------------------
# The one shipped style: "visualize"
# ---------------------------------------------------------------------------
@register_style("visualize")
def _visualize(plot_data, *, out_dir=None):
    """Render the 2-row figure (row 0 = square edge graphs, row 1 = the
    crossing-count bar charts) to ``layers_combined.pdf``.

    Parameters
    ----------
    plot_data : :class:`plot_data.PlotData`
        Self-contained snapshot. Plotting reads ``positions``,
        ``layers``, ``complete_graph`` and ``k`` only.
    out_dir : str | None
        Output directory. Falls back to ``plot_data.filepath`` (the
        source JSON's directory).
    """
    if out_dir is None:
        out_dir = plot_data.filepath
    if out_dir is None:
        raise ValueError(
            "visualize: no output directory available. Either pass "
            "out_dir=... or build PlotData from a JSON / graph whose "
            "filepath is set."
        )
    os.makedirs(out_dir, exist_ok=True)
    apply_rcparams()

    k = plot_data.k
    layers = plot_data.layers
    graphs = list(layers) + [plot_data.complete_graph]

    # Crossing-value stats shared across every panel for one common scale.
    max_cross = 0
    max_count = 0
    all_vals: list[int] = []
    for g in graphs:
        vals = list(nx.get_edge_attributes(g, "crossings").values())
        if not vals:
            continue
        all_vals += vals
        max_cross = max(max_cross, max(vals))
        max_count = max(max_count, max(Counter(vals).values()))
    lo, hi = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
    norm = plt.Normalize(vmin=0, vmax=max_cross or 1, clip=True)
    cmap = DEFAULT_COLORMAP
    bar_width = _layer_bar_width(layers)

    # log y only when the edge-count dynamic range is wide.
    cnt = list(Counter(all_vals).values())
    log_y = bool(cnt) and (max(cnt) / max(1, min(cnt)) > 20)

    # Layout: row 0 = square graph panels, row 1 = their bar charts.
    scale = min(_FIG_SCALE_MAX, max(1.0, math.sqrt(k / _FIG_SCALE_REF_K)))
    ncol = len(graphs)
    panel_w = _FIG_PANEL_W_CM * scale
    hist_h = panel_w * HIST_H_FRAC
    fig, axes = plt.subplots(
        nrows=2, ncols=ncol,
        figsize=((ncol * panel_w) / 2.54, (panel_w + hist_h) / 2.54),
        gridspec_kw={"height_ratios": [panel_w, hist_h]},
        constrained_layout=True,
    )
    if ncol == 1:
        axes = axes.reshape(2, 1)

    for j, g in enumerate(graphs):
        _draw_graph_panel(axes[0, j], g, plot_data.positions,
                          k=k, norm=norm, cmap=cmap)
        _draw_hist_panel(axes[1, j], g, norm=norm, cmap=cmap,
                         bar_width=bar_width, lo=lo, hi=hi,
                         max_count=max_count, log_y=log_y)

    fig.savefig(os.path.join(out_dir, "layers_combined.pdf"))
    plt.close(fig)
