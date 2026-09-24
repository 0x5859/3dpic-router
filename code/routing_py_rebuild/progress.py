"""Optimization-progress tracking for ``run_optimization(progress=...)``.

Three modes (``PROGRESS_MODES``):

* ``"off"`` (default) — no tracking; the run is bit-identical to before.
* ``"live"`` — keep a figure updated while the optimizer runs: a GUI
  window when matplotlib has an interactive backend, an in-place output
  in Jupyter, otherwise ``optimization_live.png`` rewritten on disk.
* ``"record"`` — save every improvement to ``optimization_history.json``
  and, after the run, render the whole process as an animated GIF plus a
  self-contained HTML player (and open it when a display is available).

:class:`ProgressTracker` is an ``eval_observer`` (see
:class:`optimizers.base.Optimizer`): every optimizer calls it once per
loss evaluation, after the loss is computed, so tracking never changes
the optimization. Each time a loss beats the best so far, the tracker
snapshots the routing that loss scored — :meth:`SiNInterconnectionGraph.
effective_layers` plus the per-edge same-layer crossing counts from the
cached crossing index — as a :class:`plotting.ProgressFrame`.
"""
from __future__ import annotations

import html
import math
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from .plotting.progress_figure import HISTORY_FILENAME, EvalEnvelope, ProgressData, ProgressFrame
from .plotting.progress_live import LiveProgressView, LiveState, live_display_mode
from .plotting.progress_replay import REPLAY_FORMATS, render_progress_replay, show_progress_replay

PROGRESS_MODES = ("off", "live", "record")

# Live redraws run inside the optimizer's loss evaluations; after each one
# the next is deferred long enough that drawing takes at most this share of
# the wall time (large k draws slowly — a fixed interval would starve the run).
LIVE_MAX_OVERHEAD = 0.2


@dataclass
class ProgressOptions:
    """Tuning knobs, passed as ``run_optimization(progress_kwargs={...})``."""

    interval: float = 0.5
    """live: minimum seconds between redraws. The optimizer waits while a
    frame is drawn, so the gap also stretches automatically to keep drawing
    under ``LIVE_MAX_OVERHEAD`` of the run time."""
    show: bool = True
    """live: after the run, keep the window open until it is closed;
    record: open the replay at the end (a window, or the inline player in
    Jupyter). No effect without a display."""
    formats: tuple[str, ...] = REPLAY_FORMATS
    """record: replay files to write — any of ``"gif"``, ``"html"``
    (``()`` = history JSON only)."""
    max_frames: int = 200
    """record: most frames per animation (the JSON keeps every one)."""
    fps: float | None = None
    """record: animation speed; None picks one from the frame count."""
    dpi: float = 100
    """Figure resolution of the live view and the replay."""

    def __post_init__(self) -> None:
        if isinstance(self.formats, str):
            self.formats = tuple(f for f in self.formats.split(",") if f)
        self.formats = tuple(str(f).strip().lower() for f in self.formats)
        unknown = sorted(set(self.formats) - set(REPLAY_FORMATS))
        if unknown:
            raise ValueError(f"unknown replay format(s) {unknown}; choose from {REPLAY_FORMATS}.")
        if self.interval < 0:
            raise ValueError(f"interval must be >= 0; got {self.interval}.")
        if self.max_frames < 2:
            raise ValueError(f"max_frames must be >= 2; got {self.max_frames}.")
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be > 0; got {self.fps}.")

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any] | None) -> ProgressOptions:
        kwargs = dict(kwargs or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise TypeError(
                f"unknown progress_kwargs {unknown}; expected a subset of {sorted(known)}."
            )
        return cls(**kwargs)


def normalize_progress_mode(progress: str | None) -> str:
    """Validate a ``progress`` argument (``None`` counts as ``"off"``)."""
    mode = "off" if progress is None else str(progress).strip().lower()
    if mode not in PROGRESS_MODES:
        raise ValueError(f"progress must be one of {PROGRESS_MODES}; got {progress!r}.")
    return mode


class LossEnvelope:
    """Streaming min/max of every evaluated loss, in log-spaced buckets.

    Evaluation ``n`` (1-based) falls in a bucket of width 1 while
    ``n < 2 * per_octave``; beyond that each doubling of ``n`` is split
    into ``per_octave`` equal buckets. The bucket count therefore grows
    only with ``log2(evaluations)`` (~900 at 10⁹ evaluations with the
    default 32), and the buckets are evenly spaced on the log-scaled
    evaluation axis the progress figure uses. Non-finite losses count as
    evaluations but are not folded into the min/max.
    """

    def __init__(self, per_octave: int = 32):
        if per_octave < 1 or per_octave & (per_octave - 1):
            raise ValueError(f"per_octave must be a power of two; got {per_octave}.")
        self._q = int(per_octave)
        self._p = self._q.bit_length() - 1
        self.count = 0
        self._start: list[int] = []
        self._lo: list[float] = []
        self._hi: list[float] = []

    def add(self, loss: float) -> None:
        n = self.count + 1
        self.count = n
        if n < 2 * self._q:
            b = n - 1
        else:
            d = n.bit_length() - 1 - self._p  # this octave's bucket width is 2**d
            b = self._q * d + (n >> d) - 1
        if b == len(self._lo):
            self._start.append(n)
            self._lo.append(math.inf)
            self._hi.append(-math.inf)
        if math.isfinite(loss):
            if loss < self._lo[b]:
                self._lo[b] = loss
            if loss > self._hi[b]:
                self._hi[b] = loss

    def snapshot(self) -> EvalEnvelope:
        lo = np.array(self._lo, dtype=float)
        hi = np.array(self._hi, dtype=float)
        lo[~np.isfinite(lo)] = np.nan
        hi[~np.isfinite(hi)] = np.nan
        return EvalEnvelope(
            count=self.count, start=np.array(self._start, dtype=np.int64), lo=lo, hi=hi
        )


class ProgressTracker:
    """``eval_observer`` that snapshots every new-best routing.

    Parameters
    ----------
    graph : SiNInterconnectionGraph
        Read-only: only :meth:`effective_layers` /
        :meth:`intralayer_crossing_counts` are called (per improvement).
    mode : {"live", "record"}
        ``"record"`` keeps every frame for the history; ``"live"`` keeps
        only the latest and drives a :class:`LiveProgressView`.
    options : ProgressOptions | None
    run : dict | None
        Free-form metadata stored in the history (optimizer, seed, ...).
    out_dir : str | None
        Where the history / replay / live image go (the run directory).
    display : str | None
        Force the live display mode (``"window"`` / ``"notebook"`` /
        ``"file"``); default: from the matplotlib backend.
    """

    def __init__(
        self,
        graph,
        *,
        mode: str = "record",
        options: ProgressOptions | None = None,
        run: dict[str, Any] | None = None,
        out_dir: str | None = None,
        display: str | None = None,
    ):
        if mode not in ("live", "record"):
            raise ValueError(f"ProgressTracker mode must be 'live' or 'record'; got {mode!r}.")
        graph.build_crossings_index()
        self._graph = graph
        self.mode = mode
        self.options = options or ProgressOptions()
        self.run = dict(run or {})
        self.out_dir = out_dir
        self.edges: list[tuple[int, int]] = [(int(u), int(v)) for u, v in graph.G.edges()]
        self.frames: list[ProgressFrame] = []
        self.envelope = LossEnvelope()
        self.best_loss = math.inf
        self.improvements = 0
        self.outputs: dict[str, str] = {}
        self._best_x: list[int] = []
        self._best_y: list[float] = []
        self._last: ProgressFrame | None = None
        self._interval = float(self.options.interval)
        self._next_tick = 0.0
        self._t0 = time.perf_counter()
        self._end_ms: int | None = None  # optimizer wall time, set by finish()
        self.live: LiveProgressView | None = None
        if mode == "live":
            self.live = LiveProgressView(
                self.data(), out_dir=out_dir, dpi=self.options.dpi, mode=display
            )
            self._guard_live(self.live.open)

    # -- eval_observer ------------------------------------------------------
    def __call__(self, x, loss) -> None:
        index = self.envelope.count
        loss = float(loss)
        self.envelope.add(loss)
        if loss < self.best_loss and math.isfinite(loss):  # NaN compares False
            self.best_loss = loss
            self.improvements += 1
            self._add_frame(self._snapshot(x, loss, index))
        if self.live is not None:
            now = time.perf_counter()
            if now >= self._next_tick:
                self._push_live(done=False)
                done = time.perf_counter()
                spent = done - now
                self._next_tick = done + max(
                    self._interval, spent * (1.0 - LIVE_MAX_OVERHEAD) / LIVE_MAX_OVERHEAD
                )

    # -----------------------------------------------------------------------
    @property
    def evaluations(self) -> int:
        return self.envelope.count

    def _elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def _snapshot(self, layers, loss: float, index: int, *, final: bool = False) -> ProgressFrame:
        eff = self._graph.effective_layers(layers)
        return ProgressFrame(
            eval_index=int(index),
            wall_ms=self._elapsed_ms(),
            loss=float(loss),
            layers=eff.astype(np.int16),
            crossings=self._graph.intralayer_crossing_counts(eff).astype(np.int32),
            final=final,
        )

    def _add_frame(self, frame: ProgressFrame) -> None:
        self._best_x.append(frame.eval_index + 1)  # 1-based evaluation number
        self._best_y.append(frame.loss)
        if self.mode == "record":
            self.frames.append(frame)
        self._last = frame

    def _push_live(self, *, done: bool) -> None:
        if self._last is None:
            return
        state = LiveState(
            frame=self._last,
            best_x=np.asarray(self._best_x, dtype=float),
            best_y=np.asarray(self._best_y, dtype=float),
            envelope=self.envelope.snapshot(),
            evaluations=self.evaluations,
            elapsed_ms=self._elapsed_ms(),
            improvements=self.improvements,
            done=done,
        )
        self._guard_live(self.live.finish if done else self.live.update, state)

    def _guard_live(self, fn, *args) -> None:
        """Run a live-view call; a display problem must never abort the run."""
        try:
            fn(*args)
        except Exception as exc:
            warnings.warn(f"live progress view disabled after an error: {exc!r}", stacklevel=3)
            self.live = None

    # -- after the optimizer returns -----------------------------------------
    def finish(self, best_layers, best_loss: float) -> ProgressData:
        """Mark the optimizer's returned result as the final frame.

        When the result equals the last new best (the usual case) that
        frame is flagged ``final``; otherwise the result is appended as an
        extra final frame, so the history always ends on the routing that
        :meth:`apply_optimization_result` published.
        """
        self._end_ms = self._elapsed_ms()
        final = self._snapshot(best_layers, best_loss, max(self.evaluations - 1, 0), final=True)
        last = self._last
        if last is not None and np.array_equal(last.layers, final.layers):
            last.final = True
        else:
            if self.mode == "record":
                self.frames.append(final)
            self._best_x.append(final.eval_index + 1)
            self._best_y.append(final.loss)
            self._last = final
        if self.live is not None:
            self._push_live(done=True)
        if self.live is not None and self.live.image_path:
            self.outputs["live_image"] = self.live.image_path
        return self.data()

    def data(self) -> ProgressData:
        g = self._graph
        return ProgressData(
            k=int(g.k),
            L=int(g.L),
            edge_coupler_layer=int(g.edge_coupler_layer),
            perimeter_layer=int(g.perimeter_layer),
            positions={int(n): (float(p[0]), float(p[1])) for n, p in g.positions.items()},
            edges=list(self.edges),
            frames=list(self.frames),
            envelope=self.envelope.snapshot(),
            run=dict(self.run),
            wall_ms=self._elapsed_ms() if self._end_ms is None else self._end_ms,
            improvements=self.improvements,
        )

    def write_history(self) -> str | None:
        """record: write ``optimization_history.json`` (also mid-run, e.g.
        after an interrupt). Returns its path, or None without frames."""
        if self.mode != "record" or not self.frames or not self.out_dir:
            return None
        path = self.data().write_json(os.path.join(self.out_dir, HISTORY_FILENAME))
        self.outputs["history"] = path
        return path

    def write_replay(self) -> dict[str, str]:
        """record: render the replay files chosen by ``options.formats``."""
        if self.mode != "record" or not self.frames or not self.out_dir:
            return {}
        opts = self.options
        paths = render_progress_replay(
            self.data(), out_dir=self.out_dir, formats=opts.formats,
            fps=opts.fps, max_frames=opts.max_frames, dpi=opts.dpi,
        )
        self.outputs.update(paths)
        return paths

    def present(self) -> None:
        """End-of-run display (``options.show``).

        live: keep the window open until closed. record: open the
        interactive replay window, or show the HTML player inline in
        Jupyter. Nothing to show on a headless backend.
        """
        if not self.options.show:
            return
        if self.live is not None:
            self.live.hold()
            return
        if self.mode != "record" or not self.frames:
            return
        mode = live_display_mode()
        if mode == "window":
            opts = self.options
            print("[progress] replaying the optimization — close the window to exit.")
            show_progress_replay(self.data(), fps=opts.fps, max_frames=opts.max_frames,
                                 dpi=opts.dpi)
        elif mode == "notebook" and "html" in self.outputs:
            from IPython.display import HTML, display

            with open(self.outputs["html"], encoding="utf-8") as f:
                page = f.read()
            # An iframe keeps the player's CSS / element ids out of the notebook.
            display(HTML(
                f'<iframe srcdoc="{html.escape(page, quote=True)}" '
                'style="width: 100%; height: 760px; border: 0;"></iframe>'
            ))


def chain_observers(*observers: Callable | None) -> Callable | None:
    """Combine ``eval_observer`` callables (``None`` entries are skipped)."""
    active = [o for o in observers if o is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _chained(x, loss):
        for obs in active:
            obs(x, loss)

    return _chained


__all__ = [
    "LIVE_MAX_OVERHEAD",
    "PROGRESS_MODES",
    "LossEnvelope",
    "ProgressOptions",
    "ProgressTracker",
    "chain_observers",
    "normalize_progress_mode",
]
