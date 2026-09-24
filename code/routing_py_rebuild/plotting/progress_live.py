"""Live optimization-progress view (``run_optimization(progress="live")``).

Where the figure goes depends on the active matplotlib backend
(:func:`live_display_mode`):

* ``"window"`` — an interactive backend (macosx, QtAgg, TkAgg, ipympl,
  ...): a GUI window redrawn in place; the event loop is spun on every
  redraw so the window stays responsive.
* ``"notebook"`` — Jupyter (inline, ipympl / widget or nbagg backend):
  one output updated in place through an IPython display handle, which
  reaches the browser even while the cell is still running.
* ``"file"`` — a non-interactive backend (Agg on a headless machine):
  ``optimization_live.png`` is atomically rewritten on every redraw;
  image viewers / IDEs that watch the file show it refreshing.

The view only receives plain data (:class:`LiveState`); the tracker in
:mod:`routing_py_rebuild.progress` owns the graph.
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .progress_figure import (
    FRAME_DPI,
    LIVE_WINDOW_DPI,
    CurveView,
    EvalEnvelope,
    ProgressData,
    ProgressFigure,
    ProgressFrame,
    _signed_pct,
)

LIVE_IMAGE_FILENAME = "optimization_live.png"
WINDOW_TITLE = "3D-PIC router — optimization progress"

_NON_GUI_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
# Jupyter backends. Widget canvases (ipympl, nbagg) only repaint when the
# kernel is idle, so a busy optimizer would freeze them — use display().
_NOTEBOOK_BACKEND_TAGS = ("inline", "ipympl", "widget", "nbagg")


def live_display_mode(backend: str | None = None) -> str:
    """``"window"``, ``"notebook"`` or ``"file"`` for a matplotlib backend."""
    name = (backend or matplotlib.get_backend()).lower()
    if any(tag in name for tag in _NOTEBOOK_BACKEND_TAGS):
        return "notebook"
    if name in _NON_GUI_BACKENDS:
        return "file"
    return "window"


@dataclass
class LiveState:
    """What the optimizer has produced so far."""

    frame: ProgressFrame        # current best routing
    best_x: np.ndarray          # 1-based evaluation number of every improvement so far
    best_y: np.ndarray          # loss of every improvement so far
    envelope: EvalEnvelope | None
    evaluations: int
    elapsed_ms: int
    improvements: int
    done: bool = False


def live_status(state: LiveState) -> str:
    """Status line for the live view (sentence case, 7 pt)."""
    start = float(state.best_y[0]) if len(state.best_y) else state.frame.loss
    rel = (state.frame.loss - start) / abs(start) * 100 if start else 0.0
    head = "Finished" if state.done else "Running"
    return (
        f"{head}, {state.evaluations:,} evaluations, best loss {state.frame.loss:.5g} dB "
        f"({_signed_pct(rel)}), {state.improvements} improvements, "
        f"{state.elapsed_ms / 1000:.1f} s"
    )


class LiveProgressView:
    """Redraws a :class:`ProgressFigure` as :class:`LiveState` arrives.

    Parameters
    ----------
    geometry : ProgressData
        Static part (k / L / positions / edges / run); frames unused.
    out_dir : str | None
        Directory for ``optimization_live.png`` in ``"file"`` mode; a
        temporary directory is used when None.
    dpi : float | None
        Resolution; None = ``LIVE_WINDOW_DPI`` for a window, ``FRAME_DPI``
        for the notebook / file image.
    mode : str | None
        Force a display mode (tests); default :func:`live_display_mode`.
    """

    def __init__(
        self,
        geometry: ProgressData,
        *,
        out_dir: str | None = None,
        dpi: float | None = None,
        mode: str | None = None,
    ):
        self.geometry = geometry
        self.mode = mode or live_display_mode()
        self.out_dir = out_dir
        self._dpi = dpi
        self.image_path: str | None = None
        self._pf: ProgressFigure | None = None
        self._handle = None
        self._closed = False
        self._shown: ProgressFrame | None = None
        self._shown_prev: ProgressFrame | None = None

    @property
    def dpi(self) -> float:
        if self._dpi is not None:
            return self._dpi
        return LIVE_WINDOW_DPI if self.mode == "window" else FRAME_DPI

    # ------------------------------------------------------------------
    def open(self) -> None:
        """Create the figure (and show the window) before the first frame,
        so the setup cost stays out of the optimizer's loop. Idempotent."""
        if self._pf is not None:
            return
        if self.mode == "window":
            try:
                self._pf = ProgressFigure(self.geometry, dpi=self.dpi, pyplot=True)
                manager = self._pf.fig.canvas.manager
                if manager is not None:
                    manager.set_window_title(f"{WINDOW_TITLE} ({self.geometry.run_label()})")
                self._pf.status_text.set_text("Starting…")
                plt.show(block=False)
                self._pf.fig.canvas.flush_events()
                return
            except Exception as exc:  # e.g. a GUI backend without a display
                print(f"[progress] no GUI window ({exc!r}); writing images instead.")
                if self._pf is not None:
                    self._pf.close()
                self.mode = "file"
        self._pf = ProgressFigure(self.geometry, dpi=self.dpi)
        if self.mode == "file":
            out_dir = self.out_dir or tempfile.mkdtemp(prefix="sinic_progress_")
            os.makedirs(out_dir, exist_ok=True)
            self.image_path = os.path.join(out_dir, LIVE_IMAGE_FILENAME)
            print(f"[progress] live view: {self.image_path} (rewritten while optimizing)")

    def _present(self) -> None:
        fig = self._pf.fig
        if self.mode == "window":
            if not plt.fignum_exists(fig.number):
                print("[progress] live window closed; the optimization continues.")
                self._closed = True
                return
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        elif self.mode == "notebook":
            # PNG bytes, not the Figure: IPython's figure formatter is only
            # registered once pyplot has loaded the inline backend.
            from IPython.display import Image, display

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=self.dpi)
            image = Image(data=buf.getvalue(), format="png")
            if self._handle is None:
                self._handle = display(image, display_id=True)
            else:
                self._handle.update(image)
        else:
            tmp = self.image_path + ".tmp.png"
            fig.savefig(tmp, dpi=self.dpi)
            os.replace(tmp, self.image_path)  # viewers never see a partial file

    def update(self, state: LiveState) -> None:
        if self._closed:
            return
        self.open()
        if state.frame is not self._shown:
            self._shown_prev, self._shown = self._shown, state.frame
        # The crossings color scale only grows, so earlier colors stay valid.
        self._pf.set_color_limit(
            max(self._pf.color_limit, float(state.frame.crossings.max(initial=0)))
        )
        marker = None
        if len(state.best_x):
            marker = (float(state.best_x[-1]), float(state.best_y[-1]))
        curve = CurveView(
            best_x=state.best_x,
            best_y=state.best_y,
            x_now=float(state.evaluations),
            marker=marker,
            envelope=state.envelope,
        )
        self._pf.draw(state.frame, self._shown_prev, curve, live_status(state))
        self._present()

    def finish(self, state: LiveState) -> None:
        """Draw the final state (always, regardless of the redraw interval)."""
        self.update(state)
        if self.mode == "file" and self.image_path:
            print(f"[progress] final live view: {self.image_path}")

    def hold(self) -> None:
        """Block until the user closes the window (``"window"`` mode only)."""
        if self.mode != "window" or self._pf is None or self._closed:
            return
        if plt.fignum_exists(self._pf.fig.number):
            print("[progress] optimization finished — close the progress window to exit.")
            plt.show(block=True)

    def close(self) -> None:
        if self._pf is not None:
            self._pf.close()


__all__ = [
    "LIVE_IMAGE_FILENAME",
    "LiveProgressView",
    "LiveState",
    "live_display_mode",
    "live_status",
]
