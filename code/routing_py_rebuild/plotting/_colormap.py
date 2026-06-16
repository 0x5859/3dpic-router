"""Encapsulated colormap + node fill for routing_py_rebuild plotting.

Mirrors the older ``code/plot_olpapper/_colormap.py`` API but trims down
to exactly what this package uses:

  - :data:`DEFAULT_COLORMAP` — matplotlib's standard ``coolwarm`` sampled
    evenly on [0, 1] at 256 stops and wrapped as a
    :class:`~matplotlib.colors.ListedColormap`. No truncation.
  - :data:`NODE_FILL_COLOR` — the canonical light-cyan node fill
    (``#AFE0E8``) used by layer-plots and the ring view.
  - :data:`CMAP_LOWER` / :data:`CMAP_UPPER` / :data:`CMAP_SCALE` — the
    [0.0, 0.95] window applied on top of ``norm(value)``. Reserves the
    top 5% so the highest-crossing edge does not land on near-black at
    the extreme end of coolwarm.

The canonical formula plotting code applies for value → color is::

    color = cmap(CMAP_LOWER + CMAP_SCALE * norm(value))

with ``norm`` a matplotlib :class:`~matplotlib.colors.Normalize`
configured by the caller (typically ``vmin=0, vmax=max_crossings,
clip=True``). The :func:`crossings_color` helper bundles the formula so
callers do not have to spell it out.
"""
from __future__ import annotations

from functools import lru_cache

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, ListedColormap


# Default node fill for layer-plot and ring nodes.
NODE_FILL_COLOR: str = "#AFE0E8"

# [cmap_lower, cmap_upper] window applied on top of the caller's norm.
# Reserves the top 5% to avoid the near-black extreme of the colormap.
CMAP_LOWER: float = 0.0
CMAP_UPPER: float = 0.95
CMAP_SCALE: float = CMAP_UPPER - CMAP_LOWER


@lru_cache(maxsize=None)
def get_colormap(name: str = "coolwarm", n: int = 256) -> Colormap:
    """Return a :class:`ListedColormap` resampling ``name`` evenly on [0, 1].

    ``coolwarm`` is the default and mirrors the older _colormap.py
    contract — full range, no truncation, 256 stops. Other matplotlib
    colormap names are accepted and resampled the same way.
    """
    base = matplotlib.colormaps[name]  # KeyError if unknown
    samples = base(np.linspace(0.0, 1.0, n))[:, :3]
    colors = [tuple(float(c) for c in row) for row in samples]
    return ListedColormap(colors, name=f"sin_{name.lower()}")


DEFAULT_COLORMAP: Colormap = get_colormap("coolwarm", n=256)


def crossings_color(value, norm, cmap: Colormap | None = None):
    """Map a value to an RGBA tuple via the canonical formula.

    Parameters
    ----------
    value
        Numeric input (typically a per-edge crossing count).
    norm
        A configured matplotlib :class:`~matplotlib.colors.Normalize`
        (``vmin=0, vmax=max_crossings, clip=True`` is the usual choice).
    cmap
        Defaults to :data:`DEFAULT_COLORMAP`; pass a custom colormap to
        deviate from coolwarm for a single plot.
    """
    cmap = cmap or DEFAULT_COLORMAP
    return cmap(CMAP_LOWER + CMAP_SCALE * norm(value))


__all__ = [
    "NODE_FILL_COLOR",
    "CMAP_LOWER",
    "CMAP_UPPER",
    "CMAP_SCALE",
    "DEFAULT_COLORMAP",
    "get_colormap",
    "crossings_color",
]
