"""Optimization-progress data model + the figure that renders one frame.

A *frame* is one routing the optimizer found on its way to the final
result: every time a loss evaluation beats the best loss so far, the
tracker (:mod:`routing_py_rebuild.progress`) snapshots the effective
per-edge layer assignment and the per-edge same-layer crossing counts.
:class:`ProgressData` bundles those frames with the static geometry
(positions, edge list) and a bounded min/max envelope of every evaluated
loss, and round-trips through ``optimization_history.json``.

:class:`ProgressFigure` draws a frame at publication scale — 7 pt
Arial-stack text, 0.5 pt lines, a centimetre layout, no figure title
(see :mod:`._style`): one routing panel per layer, shaped like the
layout's bounding box (edges colored by crossings like
``layers_combined.pdf``; edges that changed layer since the previous
frame drawn heavier with a dark outline, when only a few did) above the
best-loss-so-far curve on a log evaluation axis. Artists
are created once and updated in place, so the live view and the replay
renderer redraw quickly.

Like the rest of :mod:`plotting`, nothing here imports :mod:`core` or
touches a graph instance — the data is the contract.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

from ._colormap import CMAP_LOWER, CMAP_SCALE, CMAP_UPPER, DEFAULT_COLORMAP, NODE_FILL_COLOR
from ._style import FONT_SIZE, LINE_WIDTH, apply_rcparams, style_axes, style_colorbar
from .layers import _edge_bends, _label_fontsize, _node_size

HISTORY_FILENAME = "optimization_history.json"
HISTORY_SCHEMA_VERSION = "1.0"

_HISTORY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schema" / "optimization_history.schema.json"
)

# Line weights follow ``layers_combined.pdf`` (0.5 pt); only the moved-edge
# highlight is heavier, drawn over a thin dark outline.
EDGE_WIDTH = LINE_WIDTH          # pt — edges that kept their layer
MOVED_EDGE_WIDTH = 1.25          # pt — edges that changed layer since the previous frame
MOVED_OUTLINE_WIDTH = MOVED_EDGE_WIDTH + 0.7
MOVED_OUTLINE = "#1a1a1a"
# Outline moved edges only when few moved: a reshuffle of dozens of edges
# (annealing jumps) would bury the panel in outlines; the labels keep the count.
HIGHLIGHT_MAX_MOVED = 12
HIGHLIGHT_MAX_FRACTION = 0.08
BEST_COLOR = "k"                 # best-so-far line: one series, plain black
CONTEXT_COLOR = "#9a9a9a"        # replay: the part of the curve still ahead
MARKER_COLOR = "#B2182B"         # the frame's point — the one accent color
ENVELOPE_COLOR = "#C1DCF3"       # light blue
ENVELOPE_ALPHA = 0.45
MUTED = "#555555"                # secondary text (counts, run label)
_ARC_POINTS = 24

# Raster resolutions. Frames are screen / animation output; the PNG still
# uses the 450 ppi publication default.
LIVE_WINDOW_DPI = 120
FRAME_DPI = 150
STILL_DPI = 450


@lru_cache(maxsize=1)
def _load_history_schema() -> dict:
    with open(_HISTORY_SCHEMA_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ProgressFrame:
    """One routing snapshot (a new best, or the final applied result)."""

    eval_index: int         # 0-based loss-evaluation index that produced it
    wall_ms: int            # milliseconds since the optimizer started
    loss: float
    layers: np.ndarray      # (E,) effective layer per edge (perimeter pinned)
    crossings: np.ndarray   # (E,) same-layer crossings per edge
    final: bool = False     # True for the result written to subgraphsdata.json


@dataclass
class EvalEnvelope:
    """Min/max of the evaluated losses per bucket of consecutive evaluations.

    Bucket ``j`` covers the 1-based evaluation numbers ``start[j] ..
    start[j + 1] - 1`` (the last one runs to ``count``); NaN marks a
    bucket without a finite loss.
    """

    count: int
    start: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    def steps(self, until: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x, lo, hi)`` for a ``step="post"`` fill over evaluations
        ``1 .. until`` (the last bucket is closed at its end)."""
        keep = self.start <= until
        x = self.start[keep].astype(float)
        if not len(x):
            return x, x, x
        end = min(float(until), float(self.count)) + 1.0
        return (np.append(x, end), np.append(self.lo[keep], self.lo[keep][-1]),
                np.append(self.hi[keep], self.hi[keep][-1]))


@dataclass
class ProgressData:
    """Everything needed to draw the optimization process of one run."""

    k: int
    L: int
    edge_coupler_layer: int
    perimeter_layer: int
    positions: dict[int, tuple[float, float]]
    edges: list[tuple[int, int]]
    frames: list[ProgressFrame]
    envelope: EvalEnvelope | None = None
    run: dict[str, Any] = field(default_factory=dict)
    wall_ms: int = 0
    # Frames that were strict new bests; one less than ``len(frames)`` when
    # the final result differs from the last new best and was appended.
    improvements: int | None = None
    source_path: str | None = None

    # -- derived views ---------------------------------------------------
    @property
    def evaluations(self) -> int:
        if self.envelope is not None:
            return int(self.envelope.count)
        return self.frames[-1].eval_index + 1 if self.frames else 0

    def best_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """``(evaluation number, loss)`` of every frame (1-based numbers)."""
        xs = np.array([f.eval_index + 1 for f in self.frames], dtype=float)
        ys = np.array([f.loss for f in self.frames], dtype=float)
        return xs, ys

    def max_crossings(self) -> int:
        return max((int(f.crossings.max(initial=0)) for f in self.frames), default=0)

    def run_label(self) -> str:
        """Short run identity, e.g. ``dual_annealing, k = 12, L = 2, seed 5859``."""
        parts = []
        if self.run.get("optimizer"):
            parts.append(str(self.run["optimizer"]))
        parts.append(f"k = {self.k}")
        parts.append(f"L = {self.L}")
        if self.run.get("seed") is not None:
            parts.append(f"seed {self.run['seed']}")
        return ", ".join(parts)

    # -- JSON ------------------------------------------------------------
    def to_json_dict(self) -> dict[str, Any]:
        frames = [
            {
                "eval_index": int(f.eval_index),
                "wall_ms": int(f.wall_ms),
                "loss": float(f.loss),
                "layers": [int(v) for v in f.layers],
                "crossings": [int(v) for v in f.crossings],
                "final": bool(f.final),
            }
            for f in self.frames
        ]
        out: dict[str, Any] = {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "k": int(self.k),
            "L": int(self.L),
            "edge_coupler_layer": int(self.edge_coupler_layer),
            "perimeter_layer": int(self.perimeter_layer),
            "positions": {
                str(int(n)): [float(x), float(y)] for n, (x, y) in self.positions.items()
            },
            "edges": [[int(u), int(v)] for u, v in self.edges],
            "run": dict(self.run),
            "frames": frames,
            "summary": {
                "evaluations": int(self.evaluations),
                "improvements": int(
                    len(self.frames) if self.improvements is None else self.improvements
                ),
                "initial_loss": float(self.frames[0].loss) if self.frames else None,
                "final_loss": float(self.frames[-1].loss) if self.frames else None,
                "wall_ms": int(self.wall_ms),
            },
        }
        if self.envelope is not None:
            env = self.envelope
            out["evaluations"] = {
                "count": int(env.count),
                "start": [int(v) for v in env.start],
                "min": [float(v) if math.isfinite(v) else None for v in env.lo],
                "max": [float(v) if math.isfinite(v) else None for v in env.hi],
            }
        return out

    def write_json(self, path: str | Path) -> str:
        """Validate against ``optimization_history.schema.json``, then write."""
        payload = self.to_json_dict()
        _validate_history(payload, source=str(path))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, allow_nan=False, separators=(",", ":"))
        return str(path)

    @classmethod
    def from_json(cls, path: str | Path) -> ProgressData:
        """Load (and validate) an ``optimization_history.json``."""
        with open(path) as f:
            payload = json.load(f)
        _validate_history(payload, source=str(path))
        frames = [
            ProgressFrame(
                eval_index=int(fr["eval_index"]),
                wall_ms=int(fr["wall_ms"]),
                loss=float(fr["loss"]),
                layers=np.asarray(fr["layers"], dtype=np.int16),
                crossings=np.asarray(fr["crossings"], dtype=np.int32),
                final=bool(fr["final"]),
            )
            for fr in payload["frames"]
        ]
        envelope = None
        ev = payload.get("evaluations")
        if ev is not None:
            envelope = EvalEnvelope(
                count=int(ev["count"]),
                start=np.asarray(ev["start"], dtype=np.int64),
                lo=np.array([np.nan if v is None else v for v in ev["min"]], dtype=float),
                hi=np.array([np.nan if v is None else v for v in ev["max"]], dtype=float),
            )
        summary = payload["summary"]
        return cls(
            k=int(payload["k"]),
            L=int(payload["L"]),
            edge_coupler_layer=int(payload["edge_coupler_layer"]),
            perimeter_layer=int(payload["perimeter_layer"]),
            positions={int(n): (float(p[0]), float(p[1])) for n, p in payload["positions"].items()},
            edges=[(int(u), int(v)) for u, v in payload["edges"]],
            frames=frames,
            envelope=envelope,
            run=dict(payload["run"]),
            wall_ms=int(summary["wall_ms"]),
            improvements=int(summary["improvements"]),
            source_path=str(path),
        )


def _validate_history(payload: dict[str, Any], *, source: str) -> None:
    """Schema validation plus the cross-field checks JSON Schema can't express."""
    jsonschema.validate(instance=payload, schema=_load_history_schema())
    n_edges = len(payload["edges"])
    nodes = {int(n) for n in payload["positions"]}
    L = int(payload["L"])
    for fld in ("edge_coupler_layer", "perimeter_layer"):
        if not 0 <= int(payload[fld]) < L:
            raise ValueError(f"{source}: {fld} out of range [0, L) with L={L}.")
    for u, v in payload["edges"]:
        if u not in nodes or v not in nodes:
            raise ValueError(f"{source}: edge [{u}, {v}] references a node without a position.")
    for i, fr in enumerate(payload["frames"]):
        for key in ("layers", "crossings"):
            if len(fr[key]) != n_edges:
                raise ValueError(
                    f"{source}: frames[{i}].{key} has {len(fr[key])} entries, "
                    f"expected {n_edges} (one per edge)."
                )
        if any(not 0 <= v < L for v in fr["layers"]):
            raise ValueError(f"{source}: frames[{i}].layers has a layer outside [0, {L}).")
    ev = payload.get("evaluations")
    if ev is not None:
        n = len(ev["start"])
        if len(ev["min"]) != n or len(ev["max"]) != n:
            raise ValueError(f"{source}: evaluations.start / min / max lengths differ.")
        if n and (ev["start"][0] != 1 or ev["start"][-1] > ev["count"]
                  or any(b <= a for a, b in zip(ev["start"], ev["start"][1:], strict=False))):
            raise ValueError(
                f"{source}: evaluations.start must increase from 1 and stay <= count."
            )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def edge_polylines(positions, edges) -> list[np.ndarray]:
    """One polyline per edge, matching the ``visualize`` style's geometry:
    straight chords, except chords that run along the boundary through
    other nodes, which bow inward as the ``arc3`` quadratic Bézier
    :func:`layers._draw_curved_edges` draws (see :func:`layers._edge_bends`)."""
    edges = list(edges)
    t = np.linspace(0.0, 1.0, _ARC_POINTS)[:, None]
    polylines = []
    for (u, v), rad in zip(edges, _edge_bends(positions, edges), strict=True):
        p0 = np.asarray(positions[u], dtype=float)
        p2 = np.asarray(positions[v], dtype=float)
        if rad == 0.0:
            polylines.append(np.stack([p0, p2]))
            continue
        d = p2 - p0
        ctrl = (p0 + p2) / 2.0 + rad * np.array([d[1], -d[0]])
        polylines.append((1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p2)
    return polylines


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
@dataclass
class CurveView:
    """What the loss panel shows for one frame (x = 1-based evaluation no.)."""

    best_x: np.ndarray                  # best-so-far steps up to "now"
    best_y: np.ndarray
    x_now: float                        # evaluations completed at "now"
    marker: tuple[float, float] | None  # (evaluation no., loss) of the frame shown
    envelope: EvalEnvelope | None = None
    x_max: float | None = None          # fixed x range (replay); None = follow x_now
    y_range: tuple[float, float] | None = None


def _y_limits(ys: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.min(ys)), float(np.max(ys))
    span = hi - lo
    pad = span * 0.08 if span > 0 else max(abs(hi) * 0.02, 1e-6)
    return lo - pad, hi + 2.5 * pad


def _fmt_count(v, _pos=None) -> str:
    return f"{int(round(v)):,}"


def _signed_pct(value: float) -> str:
    """``+1.0%`` / ``−34.1%`` with a typographic minus sign."""
    return f"{value:+.1f}%".replace("-", "\u2212")


def _rolling(values: np.ndarray, op, radius: int = 2) -> np.ndarray:
    """Centered rolling ``op`` (np.fmin / np.fmax) over ``2 * radius + 1``
    buckets — turns the per-bucket extremes into a smooth envelope."""
    out = values.copy()
    for s in range(1, min(radius, len(values) - 1) + 1):
        out[s:] = op(out[s:], values[:-s])
        out[:-s] = op(out[:-s], values[s:])
    return out


# Fixed layout in centimetres. The width is a double-column 17.8 cm at
# k <= 12 and grows with sqrt(k / 12) (like ``layers_combined.pdf``) so
# dense graphs keep readable panels; nothing moves between frames.
_CM = 1 / 2.54
_WIDTH_CM = 17.8
_SIDE_CM = 0.3          # outer left / right margin
_GAP_CM = 0.35          # between layer panels
_CBAR_BLOCK_CM = 1.45   # colorbar with its tick labels and label
_CBAR_W_CM = 0.22
_LABEL_ROW_CM = 0.75    # two-line label above each routing panel
_LINE_CM = 0.3          # 7 pt line pitch
_LOSS_LEFT_CM = 1.3     # loss-panel y tick labels + y label
_LOSS_BOTTOM_CM = 0.85  # loss-panel x tick labels + x label
_LOSS_H_CM = 3.4
_STATUS_ROW_CM = 0.45   # status line above the loss panel
_PANEL_GAP_CM = 0.3     # routing panels to status line
_TOP_CM = 0.1
_MIN_PANEL_CM = 4.5
_ASPECT_RANGE = (0.4, 1.5)  # routing-panel height / width, from the layout's bounding box


class ProgressFigure:
    """Figure + artists for :class:`ProgressData` frames, updated in place.

    Parameters
    ----------
    data : ProgressData
        Geometry (positions / edges / k / L) is read at construction;
        frames are passed to :meth:`draw`.
    dpi : float
        Raster resolution; the physical size comes from the cm layout.
    pyplot : bool
        Create the figure through :mod:`matplotlib.pyplot` so it gets a
        GUI window (live view / interactive replay). Otherwise a bare
        Agg-backed :class:`~matplotlib.figure.Figure` is used, which never
        touches pyplot's global figure registry.
    context : bool
        Draw the complete best-so-far curve faintly behind the progressing
        one (replay: you see where the run is heading).
    annotate : bool
        Show the status line and the run label (animation frames). Stills
        for a paper leave them to the caption.
    footer_cm : float
        Space kept free below the loss panel (interactive replay widgets).
    """

    def __init__(
        self,
        data: ProgressData,
        *,
        dpi: float = FRAME_DPI,
        pyplot: bool = False,
        context: bool = False,
        annotate: bool = True,
        footer_cm: float = 0.0,
    ):
        apply_rcparams()
        self.data = data
        k, L = data.k, data.L
        self._paths = edge_polylines(data.positions, data.edges)

        # Routing panels take the layout's shape (a 5 x 3 rectangle gets
        # wide panels) instead of letterboxing it in a square.
        xs = np.array([p[0] for p in data.positions.values()])
        ys = np.array([p[1] for p in data.positions.values()])
        span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-9)
        margin = 0.09 * span
        aspect = (ys.max() - ys.min() + 2 * margin) / (xs.max() - xs.min() + 2 * margin)
        aspect = min(max(aspect, _ASPECT_RANGE[0]), _ASPECT_RANGE[1])

        scale = min(2.2, max(1.0, math.sqrt(k / 12.0)))
        fixed = 2 * _SIDE_CM + _CBAR_BLOCK_CM + (L - 1) * _GAP_CM
        panel = max(_MIN_PANEL_CM * scale, (_WIDTH_CM * scale - fixed) / L)
        panel_h = panel * aspect
        width = fixed + L * panel
        loss_y = footer_cm + _LOSS_BOTTOM_CM
        panels_y = loss_y + _LOSS_H_CM + _STATUS_ROW_CM + _PANEL_GAP_CM
        height = panels_y + panel_h + _LABEL_ROW_CM + _TOP_CM
        right = _SIDE_CM + L * panel + (L - 1) * _GAP_CM  # right edge of the panels
        self.footer_cm = footer_cm

        if pyplot:
            self.fig = plt.figure(figsize=(width * _CM, height * _CM), dpi=dpi)
        else:
            self.fig = Figure(figsize=(width * _CM, height * _CM), dpi=dpi)
            FigureCanvasAgg(self.fig)
        fig = self.fig
        fig.patch.set_facecolor("white")

        def rect(x, y, w, h):  # cm → figure fraction
            return (x / width, y / height, w / width, h / height)

        self.layer_axes = [
            fig.add_axes(rect(_SIDE_CM + j * (panel + _GAP_CM), panels_y, panel, panel_h))
            for j in range(L)
        ]
        self.cax = fig.add_axes(rect(right + 0.3, panels_y + 0.15 * panel_h,
                                     _CBAR_W_CM, 0.7 * panel_h))
        self.loss_ax = fig.add_axes(rect(_LOSS_LEFT_CM, loss_y, right - _LOSS_LEFT_CM, _LOSS_H_CM))

        # No figure title: progress goes in a status line above the loss
        # panel, the run identity in a muted note under its right end.
        self.status_text = fig.text(
            _LOSS_LEFT_CM / width, (loss_y + _LOSS_H_CM + 0.12) / height, "",
            ha="left", va="bottom", fontsize=FONT_SIZE, visible=annotate,
        )
        self.run_text = fig.text(
            right / width, (footer_cm + 0.12) / height, data.run_label(),
            ha="right", va="bottom", fontsize=FONT_SIZE, color=MUTED, visible=annotate,
        )

        # Layer panels: nodes + labels are static; per frame only the two
        # edge collections change (outlines of moved edges under, edges over).
        ns = _node_size(k)
        fs = _label_fontsize(k)
        label_top = (panels_y + panel_h + _LABEL_ROW_CM) / height
        self._edge_lcs: list[LineCollection] = []
        self._outline_lcs: list[LineCollection] = []
        self._panel_counts = []
        for j, ax in enumerate(self.layer_axes):
            ax.set_xlim(xs.min() - margin, xs.max() + margin)
            ax.set_ylim(ys.min() - margin, ys.max() + margin)
            ax.set_aspect("equal")
            ax.axis("off")
            outline = LineCollection([], colors=MOVED_OUTLINE, linewidths=MOVED_OUTLINE_WIDTH,
                                     capstyle="round", zorder=1)
            edges = LineCollection([], linewidths=EDGE_WIDTH, capstyle="round", zorder=2)
            ax.add_collection(outline)
            ax.add_collection(edges)
            self._outline_lcs.append(outline)
            self._edge_lcs.append(edges)
            ax.scatter(xs, ys, s=ns, c=NODE_FILL_COLOR, edgecolors="none", zorder=3)
            for n, (x, y) in data.positions.items():
                ax.text(x, y, str(n), ha="center", va="center", fontsize=fs, zorder=4)
            x0 = (_SIDE_CM + j * (panel + _GAP_CM)) / width
            name = f"Layer {j}" + (" (coupler)" if j == data.edge_coupler_layer else "")
            fig.text(x0, label_top, name, ha="left", va="top", fontsize=FONT_SIZE)
            self._panel_counts.append(fig.text(
                x0, label_top - _LINE_CM / height, "", ha="left", va="top",
                fontsize=FONT_SIZE, color=MUTED,
            ))

        # Colorbar over the same [CMAP_LOWER, CMAP_UPPER] window that
        # ``crossings_color`` applies, so the bar matches the edges.
        self._cmap = DEFAULT_COLORMAP
        self._norm = Normalize(vmin=0, vmax=1, clip=True)
        bar_cmap = ListedColormap(DEFAULT_COLORMAP(np.linspace(CMAP_LOWER, CMAP_UPPER, 256)))
        self._mappable = ScalarMappable(norm=Normalize(0, 1), cmap=bar_cmap)
        self._cbar = fig.colorbar(self._mappable, cax=self.cax)
        style_colorbar(self._cbar, label="Crossings per edge")
        self._cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        self._vmax: float | None = None

        # Loss panel: log evaluation axis — improvements cluster early.
        lax = self.loss_ax
        lax.set_xscale("log")
        self._env_artist = None
        if context and data.frames:
            bx, by = data.best_curve()
            end = max(float(data.evaluations), float(bx[-1]))
            lax.plot(np.append(bx, end), np.append(by, by[-1]), drawstyle="steps-post",
                     color=CONTEXT_COLOR, linewidth=LINE_WIDTH, zorder=2)
        self._best_line, = lax.plot([], [], drawstyle="steps-post", color=BEST_COLOR,
                                    linewidth=LINE_WIDTH, zorder=3, label="Best so far")
        self._marker, = lax.plot([], [], "o", color=MARKER_COLOR, markersize=4,
                                 markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        lax.set_xlabel("Evaluations", fontsize=FONT_SIZE)
        lax.set_ylabel("Mean edge loss (dB)", fontsize=FONT_SIZE)
        lax.xaxis.set_major_locator(LogLocator(base=10))
        lax.xaxis.set_major_formatter(FuncFormatter(_fmt_count))
        lax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axes(lax)
        lax.legend(
            handles=[
                self._best_line,
                Patch(facecolor=ENVELOPE_COLOR, alpha=ENVELOPE_ALPHA, linewidth=0,
                      label="Evaluated (range)"),
            ],
            loc="upper right", fontsize=FONT_SIZE, frameon=False, handlelength=1.6,
            borderaxespad=0.4,
        )

    # ------------------------------------------------------------------
    @property
    def color_limit(self) -> float:
        """Current top of the crossings color scale (0 until first set)."""
        return self._vmax or 0.0

    def set_color_limit(self, vmax: float) -> None:
        """Fix the crossings color scale (0 .. ``vmax``) for every panel."""
        vmax = max(float(vmax), 1.0)
        if vmax == self._vmax:
            return
        self._vmax = vmax
        self._norm.vmax = vmax
        self._mappable.set_clim(0, vmax)

    def _edge_colors(self, values: np.ndarray) -> np.ndarray:
        return self._cmap(CMAP_LOWER + CMAP_SCALE * self._norm(values))

    def draw(
        self,
        frame: ProgressFrame,
        prev: ProgressFrame | None,
        curve: CurveView,
        status: str,
    ) -> None:
        """Update every artist to show ``frame``; edges whose layer differs
        from ``prev`` are highlighted."""
        if self._vmax is None:
            self.set_color_limit(frame.crossings.max(initial=0))
        layers = np.asarray(frame.layers)
        cross = np.asarray(frame.crossings)
        moved = (layers != prev.layers) if prev is not None else np.zeros(len(layers), bool)
        highlight = moved.sum() <= max(HIGHLIGHT_MAX_MOVED, HIGHLIGHT_MAX_FRACTION * len(layers))
        for j in range(self.data.L):
            on = np.flatnonzero(layers == j)
            on = np.concatenate([on[~moved[on]], on[moved[on]]])  # moved edges on top
            is_moved = moved[on]
            outlined = is_moved if highlight else np.zeros(len(on), bool)
            lc = self._edge_lcs[j]
            lc.set_segments([self._paths[i] for i in on])
            lc.set_color(self._edge_colors(cross[on]))
            lc.set_linewidth(np.where(outlined, MOVED_EDGE_WIDTH, EDGE_WIDTH))
            self._outline_lcs[j].set_segments([self._paths[i] for i in on[outlined]])
            counts = f"{len(on)} edges, {int(cross[on].sum()) // 2} crossings"
            if is_moved.any():
                counts += f", {int(is_moved.sum())} moved in"
            self._panel_counts[j].set_text(counts)

        self.status_text.set_text(status)
        self._draw_curve(curve)

    def _draw_curve(self, curve: CurveView) -> None:
        lax = self.loss_ax
        x_now = max(float(curve.x_now), 1.0)
        if len(curve.best_x):
            self._best_line.set_data(np.append(curve.best_x, x_now),
                                     np.append(curve.best_y, curve.best_y[-1]))
        else:
            self._best_line.set_data([], [])
        if curve.marker is not None:
            self._marker.set_data([curve.marker[0]], [curve.marker[1]])
        else:
            self._marker.set_data([], [])

        if self._env_artist is not None:
            self._env_artist.remove()
            self._env_artist = None
        if curve.envelope is not None:
            ex, lo, hi = curve.envelope.steps(x_now)
            if len(ex):
                self._env_artist = lax.fill_between(
                    ex, _rolling(lo, np.fmin), _rolling(hi, np.fmax), step="post",
                    color=ENVELOPE_COLOR, alpha=ENVELOPE_ALPHA, linewidth=0, zorder=1,
                )

        x_max = curve.x_max if curve.x_max is not None else max(10.0, x_now * 1.3)
        lax.set_xlim(0.8, max(x_max, 2.0) * 1.05)
        if curve.y_range is not None:
            lax.set_ylim(*curve.y_range)
        elif len(curve.best_y):
            lax.set_ylim(*_y_limits(curve.best_y))

    def close(self) -> None:
        """Close the pyplot window, if this figure has one."""
        if plt.fignum_exists(getattr(self.fig, "number", -1)):
            plt.close(self.fig)


def frame_status(frame: ProgressFrame, *, index: int, total: int,
                 evaluations: int, initial_loss: float) -> str:
    """Status line for a replay frame (sentence case, 7 pt)."""
    rel = (frame.loss - initial_loss) / abs(initial_loss) * 100 if initial_loss else 0.0
    head = "Final result" if frame.final else f"Improvement {index + 1} of {total}"
    change = "initial" if index == 0 and not frame.final else _signed_pct(rel)
    return (
        f"{head}, evaluation {frame.eval_index + 1:,} of {evaluations:,}, "
        f"loss {frame.loss:.5g} dB ({change}), {frame.wall_ms / 1000:.2f} s"
    )


__all__ = [
    "HISTORY_FILENAME",
    "HISTORY_SCHEMA_VERSION",
    "CurveView",
    "EvalEnvelope",
    "ProgressData",
    "ProgressFigure",
    "ProgressFrame",
    "edge_polylines",
    "frame_status",
]
