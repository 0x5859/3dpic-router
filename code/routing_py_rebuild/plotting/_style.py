"""Shared plotting style — Arial 7pt, 0.5pt lines.

Every public plot function in this package calls :func:`apply_rcparams`
before drawing so that fonts, line widths, and tick widths are consistent
across :mod:`layers`, :mod:`ring`, :mod:`layer_edges`, and
:mod:`loss_analysis`. Override globally by mutating these module constants
before the first draw, or per-figure by supplying matplotlib rcParams in
the calling code.

The font is requested as a stack: ``FONT_FAMILY`` first, then
``FONT_FALLBACKS`` (Helvetica, then the metric-compatible Arial clones
Liberation Sans / Arimo that most Linux systems ship), so a machine without
Arial still renders Arial-like text instead of matplotlib's DejaVu default.

:func:`style_axes` / :func:`style_colorbar` apply the plotting-box rules
used by the optimization-progress figure: four black 0.5 pt spines, inward
ticks, and on log axes minor ticks at 2..9 x 10^n with a light major +
minor grid.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

FONT_FAMILY = "Arial"
FONT_FALLBACKS = ("Helvetica", "Liberation Sans", "Arimo", "DejaVu Sans")
FONT_SIZE = 7        # pt — applies to titles, ticks, labels, annotations
LINE_WIDTH = 0.5     # pt — graph edges, bar edges, axis spines, ticks
TICK_LENGTH = 1.8    # pt — major ticks (drawn inward by style_axes)
MINOR_TICK_LENGTH = 1.2  # pt — log-axis minor ticks
GRID_COLOR = "gray"
SAVEFIG_DPI = 500


def font_stack() -> list[str]:
    """``FONT_FAMILY`` followed by the fallbacks, without duplicates."""
    return [FONT_FAMILY, *(f for f in FONT_FALLBACKS if f != FONT_FAMILY)]


def apply_rcparams() -> None:
    """Apply Arial / 7pt / 0.5pt rcParams to the current matplotlib session."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": font_stack(),
        "font.size": FONT_SIZE,
        "axes.linewidth": LINE_WIDTH,
        "axes.edgecolor": "k",
        "xtick.color": "k",
        "ytick.color": "k",
        "xtick.major.width": LINE_WIDTH,
        "ytick.major.width": LINE_WIDTH,
        "xtick.minor.width": LINE_WIDTH,
        "ytick.minor.width": LINE_WIDTH,
        "lines.linewidth": LINE_WIDTH,
        "patch.linewidth": LINE_WIDTH,
        "savefig.dpi": SAVEFIG_DPI,
    })


def style_axes(ax, *, grid: bool | None = None) -> None:
    """Four black ``LINE_WIDTH`` spines, inward ticks, log minor ticks.

    ``grid=None`` turns the grid on for log axes only: major ``--`` 0.25 pt
    and, on the log axis, minor ``:`` 0.2 pt, both ``GRID_COLOR`` at 50 %.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_WIDTH)
        spine.set_color("k")
    ax.tick_params(which="major", direction="in", length=TICK_LENGTH,
                   width=LINE_WIDTH, colors="k", labelsize=FONT_SIZE)
    log_axes = [(name, axis) for name, axis, scale in (
        ("x", ax.xaxis, ax.get_xscale()), ("y", ax.yaxis, ax.get_yscale()),
    ) if scale == "log"]
    if grid is None:
        grid = bool(log_axes)
    ax.grid(False, which="both")
    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.25,
                color=GRID_COLOR, alpha=0.5)
    for name, axis in log_axes:
        axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        axis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis=name, which="minor", direction="in",
                       length=MINOR_TICK_LENGTH, width=LINE_WIDTH, colors="k")
        if grid:
            ax.grid(True, axis=name, which="minor", linestyle=":", linewidth=0.2,
                    color=GRID_COLOR, alpha=0.5)
    ax.set_axisbelow(True)


def style_colorbar(cbar, *, label: str | None = None) -> None:
    """Match a colorbar to :func:`style_axes` (0.5 pt outline, inward ticks)."""
    cbar.outline.set_linewidth(LINE_WIDTH)
    cbar.ax.tick_params(direction="in", length=TICK_LENGTH, width=LINE_WIDTH,
                        colors="k", labelsize=FONT_SIZE)
    if label is not None:
        cbar.set_label(label, fontsize=FONT_SIZE)
