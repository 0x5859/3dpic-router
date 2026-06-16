"""Shared plotting style — Arial 7pt, 0.5pt lines.

Every public plot function in this package calls :func:`apply_rcparams`
before drawing so that fonts, line widths, and tick widths are consistent
across :mod:`layers`, :mod:`ring`, :mod:`layer_edges`, and
:mod:`loss_analysis`. Override globally by mutating these module constants
before the first draw, or per-figure by supplying matplotlib rcParams in
the calling code.
"""
from __future__ import annotations

import matplotlib.pyplot as plt


FONT_FAMILY = "Arial"
FONT_SIZE = 7        # pt — applies to titles, ticks, labels, annotations
LINE_WIDTH = 0.5     # pt — graph edges, bar edges, axis spines, ticks
SAVEFIG_DPI = 500


def apply_rcparams() -> None:
    """Apply Arial / 7pt / 0.5pt rcParams to the current matplotlib session."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
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
