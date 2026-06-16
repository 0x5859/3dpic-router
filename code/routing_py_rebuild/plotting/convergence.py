"""Convergence + timings plotters for ``run_report.json``.

Per REFACTOR_GOALS.md §1-1-a 输出格式. These functions consume a path to
``run_report.json`` directly (not a :class:`PlotData` instance) and so are
**not** registered via :func:`register_style` — that registry is for
``visualize_layers`` style dispatch. ``plot_convergence`` and
``plot_timings`` live alongside :mod:`loss_analysis` / :mod:`ring` as
plain re-exports of the :mod:`plotting` package.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ._style import apply_rcparams


def _load_report(run_report_path: str | os.PathLike) -> dict[str, Any]:
    with open(run_report_path) as f:
        return json.load(f)


def plot_convergence(
    run_report_path: str | os.PathLike,
    *,
    out_path: str | os.PathLike | None = None,
) -> str:
    """Render per-eval loss curve with new-best markers.

    The X axis is ``wall_ms``; the Y axis is ``loss``. Points where
    ``is_new_best`` is true are highlighted as filled markers. Saves a PNG
    to ``out_path``; if ``out_path`` is None, defaults to
    ``<report_dir>/convergence.png``.

    Returns the path written.
    """
    apply_rcparams()
    report = _load_report(run_report_path)
    trace = report["trace"]
    if not trace:
        raise ValueError(f"run_report at {run_report_path} has empty trace.")

    wall_ms = [ev["wall_ms"] for ev in trace]
    loss = [ev["loss"] for ev in trace]
    best_idx = [i for i, ev in enumerate(trace) if ev["is_new_best"]]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot(wall_ms, loss, "-", color="#5470c6", linewidth=0.5, label="loss eval")
    if best_idx:
        ax.plot(
            [wall_ms[i] for i in best_idx],
            [loss[i] for i in best_idx],
            "o",
            markersize=2.5,
            markerfacecolor="#d33",
            markeredgecolor="#d33",
            label="new best",
        )
    ax.set_xlabel("wall time (ms, since optimize() entry)")
    ax.set_ylabel("loss")
    ax.set_title(f"convergence — {report.get('run_id', '<no run_id>')}")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    out = Path(out_path) if out_path else Path(run_report_path).parent / "convergence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_timings(
    run_report_path: str | os.PathLike,
    *,
    out_path: str | os.PathLike | None = None,
) -> str:
    """Render a horizontal bar chart of phase wall_ms.

    Bars are sorted by descending duration. Saves a PNG to ``out_path``;
    if ``out_path`` is None, defaults to ``<report_dir>/timings.png``.

    Returns the path written.
    """
    apply_rcparams()
    report = _load_report(run_report_path)
    timings = report["timings"]

    if not timings:
        # render an empty placeholder rather than crash — a run with no
        # phase timers is technically valid per the schema
        fig, ax = plt.subplots(figsize=(4.5, 2.0))
        ax.text(0.5, 0.5, "no timings recorded", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        ax.set_axis_off()
    else:
        items = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
        names = [k for k, _ in items]
        values = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(4.5, max(1.6, 0.25 * len(items) + 0.8)))
        ax.barh(names, values, height=0.5, color="#5470c6", edgecolor="#222", linewidth=0.5)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v}", va="center", fontsize=6)
        ax.set_xlabel("wall time (ms)")
        ax.invert_yaxis()
        ax.set_title(f"phase timings — {report.get('run_id', '<no run_id>')}")

    fig.tight_layout()
    out = Path(out_path) if out_path else Path(run_report_path).parent / "timings.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return str(out)
