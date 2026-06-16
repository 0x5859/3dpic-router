"""RunRecorder — in-memory accumulator for trace + timings + summary.

The recorder is the canonical sink for end-to-end runs. ``finalize()`` returns
a dict that matches ``code/schema/run_report.schema.json`` v1.0.

Per REFACTOR_GOALS.md §1-1-a: ``is_new_best`` is recomputed here from
``best_so_far``; whatever value the wrapper passes on IterEvent is ignored.
"""
from __future__ import annotations

import math
from typing import Any

from .sink import IterEvent


class RunRecorder:
    """Accumulate per-iter trace + per-phase wall_ms.

    Parameters
    ----------
    run_id : str
        Unique identifier; emitted to ``report["run_id"]``.
    config : dict[str, Any]
        Mirror of the optimizer's General Parameters; emitted opaquely to
        ``report["config"]``.
    """

    def __init__(
        self, *, run_id: str, config: dict[str, Any], trace_stride: int = 1
    ):
        self._run_id = str(run_id)
        self._config = dict(config)
        # Equal-budget runs (max_nfe = 10x N_DA, 2026-06-06 reform) can do
        # tens of millions of evals; storing one trace row per eval would
        # bloat run_report.json to multi-GB and OOM the aggregator.
        # ``trace_stride > 1`` stores only every Nth finite eval — the
        # anytime curve is log-strided downstream and best_loss_so_far is
        # monotone, so a strided sample preserves the curve shape. The
        # EXACT finite-eval count and best-eval index are tracked
        # independently (``_n_iter`` / ``_best_at_iter``), so
        # ``summary.iterations`` and ``best_at_iter`` stay exact regardless
        # of stride. Default 1 = store every eval, byte-identical to the
        # pre-stride behavior (parity fixtures + unit tests never set it).
        self._trace_stride = max(1, int(trace_stride))
        self._n_iter = 0
        self._trace: list[dict[str, Any]] = []
        self._timings: dict[str, int] = {}
        self._best_so_far: float = math.inf
        self._best_at_iter: int = -1
        self._initial_loss: float | None = None
        self._aggregate: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # StatsSink interface
    # ------------------------------------------------------------------
    def on_iter(self, ev: IterEvent) -> None:
        loss = float(ev.loss)
        # Non-finite losses (NaN, +inf, -inf) corrupt downstream consumers:
        #   - json.dump emits literal `NaN` / `Infinity`, which strict JSON
        #     parsers (notably nlohmann::json — the C++ side at M5 parity)
        #     reject.
        #   - matplotlib silently drops the points, hiding the run shape.
        # Record the eval (so the user can still spot the failure via
        # trace[].loss being NaN-suppressed) but exclude it from best/initial
        # tracking and the trace list itself.
        if not math.isfinite(loss):
            return

        is_new_best = loss < self._best_so_far
        if is_new_best:
            self._best_so_far = loss
            self._best_at_iter = int(ev.iter)
        if self._initial_loss is None:
            self._initial_loss = loss
        self._n_iter += 1
        # Store the first finite eval and every stride-th one thereafter.
        # iterations/best_at_iter are tracked exactly above (independent
        # of what is stored), so the stride only thins the trace[] array.
        if (
            self._trace_stride == 1
            or (self._n_iter - 1) % self._trace_stride == 0
        ):
            self._trace.append(
                {
                    "iter": int(ev.iter),
                    "loss": loss,
                    "wall_ms": int(ev.wall_ms),
                    "is_new_best": bool(is_new_best),
                }
            )

    def on_phase(self, name: str, wall_ms: int) -> None:
        # Accumulates across re-entries: two ``with PhaseTimer(sink, "x"):``
        # blocks add together. Intentional — lets callers time the same phase
        # in a loop or across both branches of a conditional.
        key = name if name.endswith("_ms") else f"{name}_ms"
        self._timings[key] = self._timings.get(key, 0) + int(wall_ms)

    # ------------------------------------------------------------------
    # Recorder-specific extension (NOT on StatsSink protocol)
    # ------------------------------------------------------------------
    def set_aggregate(
        self,
        *,
        layers: list[dict[str, int]],
        crosslayer_crossings_total: int,
        crosslayer_crossings_convention: str | None = None,
    ) -> None:
        """Stash per-layer aggregate stats (§1-1-a item 3) for finalize().

        Caller (typically ``api.run_optimization``) extracts this from a
        freshly analyzed graph after ``analyze_loss()`` has populated
        ``sub_G`` + ``total_crossings_of_sub_G``.

        M3 (§4 M3 附注): ``crosslayer_crossings_convention`` ("physical" or
        "geometric") records whether ``crosslayer_crossings_total`` has
        been multiplied by ``waveguides_per_link``. None means "don't emit
        the field" for legacy M1/M2 callers; M3 ``run_optimization`` always
        passes one.
        """
        # M3 review pass: enum widened to include "experimental" so
        # ``waveguides_per_link ∈ {3, 4, ...}`` users get a durable
        # label in the run report. Stay in sync with the schema enum
        # at ``code/schema/run_report.schema.json``.
        valid_conventions = {"physical", "geometric", "experimental", None}
        if crosslayer_crossings_convention not in valid_conventions:
            raise ValueError(
                f"crosslayer_crossings_convention must be one of "
                f"{valid_conventions - {None}}; got {crosslayer_crossings_convention!r}."
            )
        agg: dict[str, Any] = {
            "layers": [
                {
                    "layer": int(d["layer"]),
                    "edge_count": int(d["edge_count"]),
                    "crossings": int(d["crossings"]),
                }
                for d in layers
            ],
            "crosslayer_crossings_total": int(crosslayer_crossings_total),
        }
        if crosslayer_crossings_convention is not None:
            agg["crosslayer_crossings_convention"] = str(
                crosslayer_crossings_convention
            )
        self._aggregate = agg

    def finalize(self) -> dict[str, Any]:
        if self._n_iter > 0:
            initial = float(self._initial_loss) if self._initial_loss is not None else 0.0
            final = float(self._best_so_far) if math.isfinite(self._best_so_far) else 0.0
            # iterations is the EXACT finite-eval count, not len(trace) —
            # they are equal when trace_stride == 1 (parity-preserving) and
            # diverge only when the trace is strided for a large-budget run.
            iterations = self._n_iter
            best_at_iter = int(self._best_at_iter)
            relative_drop = (initial - final) / abs(initial) if initial != 0 else 0.0
        else:
            initial = 0.0
            final = 0.0
            iterations = 0
            best_at_iter = -1
            relative_drop = 0.0

        summary: dict[str, Any] = {
            "initial_loss": float(initial),
            "final_loss": float(final),
            "relative_drop": float(relative_drop),
            "iterations": int(iterations),
            "best_at_iter": int(best_at_iter),
        }
        if self._aggregate:
            summary["layers"] = list(self._aggregate["layers"])
            summary["crosslayer_crossings_total"] = self._aggregate["crosslayer_crossings_total"]
            if "crosslayer_crossings_convention" in self._aggregate:
                summary["crosslayer_crossings_convention"] = self._aggregate[
                    "crosslayer_crossings_convention"
                ]

        return {
            "schema_version": "1.0",
            "run_id": self._run_id,
            "config": dict(self._config),
            "trace": list(self._trace),
            "timings": dict(self._timings),
            "summary": summary,
        }
