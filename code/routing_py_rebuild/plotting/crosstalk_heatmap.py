"""Crosstalk heatmap renderers (REFACTOR_GOALS.md §2-2 / M4).

Two aggregation views over the rank-3 tensor ``crosstalk[s][d][t]`` (dB):

- ``worst_over_d``: for each (s, t), take the maximum over d — the
  worst-case victim leakage regardless of destination. Useful for "is
  there ANY way to drive this s→? that lights up t?" audits.
- ``for_specific_d``: pin d to a chosen node — heatmap of the actual
  ``crosstalk[s][:, t]`` slice. Useful when comparing routes.

The plot consumes the tensor from ``plot_data.general_params["crosstalk"]``,
which :meth:`plotting.PlotData.from_json` already passes through (M3
opus-review-3 P2-A). It does **not** call :mod:`.core` — the JSON file
remains the contract.

``None`` entries in the JSON (diagonals and below-threshold arrivals) map
to ``np.nan`` in the dB array; matplotlib renders them as the colormap's
``bad`` value (transparent / masked), so the heatmap visibly separates
"no data" from extreme dB values.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ._style import apply_rcparams


_VIEWS = ("worst_over_d", "for_specific_d")


def plot_crosstalk_heatmap(
    plot_data,
    *,
    view: str = "worst_over_d",
    d_node: int | None = None,
    out_path: str | None = None,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    fig_size: tuple[float, float] = (4.5, 4.0),
):
    """Render a crosstalk heatmap from a :class:`PlotData` snapshot.

    Parameters
    ----------
    plot_data : :class:`plotting.PlotData`
        Must carry a non-None ``general_params["crosstalk"]`` blob —
        ``PlotData.from_json`` populates it from the top-level
        ``crosstalk`` field of ``subgraphsdata.json`` when the tensor was
        emitted by the writer (M4).
    view : {"worst_over_d", "for_specific_d"}
        Aggregation strategy. See module docstring.
    d_node : int | None
        Required for ``view="for_specific_d"``. Ignored for
        ``"worst_over_d"``.
    out_path : str | None
        Output file path. Defaults to ``{plot_data.filepath}/
        crosstalk_heatmap_{view}.pdf``.
    cmap : str
        Matplotlib colormap name. ``"magma"`` chosen so darker = lower
        leakage (cleaner channel); reviewers can override.
    vmin, vmax : float | None
        Optional explicit dB scale bounds. Defaults to the data extent
        (``np.nanmin`` / ``np.nanmax`` of the aggregated slice).
    title : str | None
        Overrides the default panel title.
    fig_size : tuple
        Inches.

    Returns
    -------
    str
        The path to the written file.

    Raises
    ------
    ValueError
        If ``plot_data`` carries no crosstalk tensor, ``view`` is unknown,
        or ``view="for_specific_d"`` without ``d_node``.
    """
    if view not in _VIEWS:
        raise ValueError(f"Unknown view {view!r}; choose from {_VIEWS}.")

    crosstalk = plot_data.general_params.get("crosstalk")
    if crosstalk is None:
        raise ValueError(
            "PlotData carries no crosstalk tensor — re-run with "
            "compute_crosstalk=True and non-zero loss_*_crosstalk "
            "coefficients (REFACTOR_GOALS.md §2-2)."
        )

    shape = crosstalk.get("shape")
    if shape is None or len(shape) != 3:
        raise ValueError(
            f"crosstalk.shape must be a 3-element list; got {shape!r}."
        )
    k_dim = int(shape[0])
    if int(shape[1]) != k_dim or int(shape[2]) != k_dim:
        raise ValueError(
            f"crosstalk tensor must be k×k×k (got shape={shape})."
        )

    # ``None`` → NaN so np.nanmax / imshow handle them as missing data.
    raw = crosstalk["values"]
    arr = np.array(
        [
            [[(v if v is not None else np.nan) for v in row] for row in slab]
            for slab in raw
        ],
        dtype=np.float64,
    )

    if view == "worst_over_d":
        with warnings.catch_warnings():
            # When every entry along the d-axis is NaN (e.g. s on the
            # perimeter so the row has no proper-crossing path), nanmax
            # emits a RuntimeWarning. We expect this for the off-diagonal
            # bookkeeping and silence it — the resulting NaN is the right
            # signal for the colormap.
            warnings.simplefilter("ignore", RuntimeWarning)
            slab = np.nanmax(arr, axis=1)
        panel_title = title or "Crosstalk: worst case over d (dB)"
    else:
        if d_node is None:
            raise ValueError(
                "view='for_specific_d' requires d_node (the pinned "
                "destination node index)."
            )
        if not (0 <= int(d_node) < k_dim):
            raise ValueError(
                f"d_node={d_node} out of range [0, {k_dim})."
            )
        slab = arr[:, int(d_node), :]
        panel_title = title or f"Crosstalk for d={int(d_node)} (dB)"

    apply_rcparams()

    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray", alpha=0.6)
    im = ax.imshow(
        slab,
        cmap=cmap_obj,
        aspect="equal",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("victim t")
    ax.set_ylabel("source s")
    ax.set_title(panel_title)
    ax.set_xticks(range(k_dim))
    ax.set_yticks(range(k_dim))

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("dB")

    if out_path is None:
        base_dir = plot_data.filepath or os.getcwd()
        os.makedirs(base_dir, exist_ok=True)
        if view == "worst_over_d":
            out_path = os.path.join(base_dir, "crosstalk_heatmap_worst.pdf")
        else:
            out_path = os.path.join(
                base_dir, f"crosstalk_heatmap_d{int(d_node)}.pdf"
            )

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


__all__ = ["plot_crosstalk_heatmap"]
