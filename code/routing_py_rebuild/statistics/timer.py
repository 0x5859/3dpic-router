"""PhaseTimer context manager.

Usage:
    with PhaseTimer(sink, "graph_build"):
        graph = make_graph(...)
"""
from __future__ import annotations

import time

from .sink import StatsSink


class PhaseTimer:
    """Context-manager timer; emits one ``on_phase(name, elapsed_ms)`` on exit."""

    def __init__(self, sink: StatsSink, name: str):
        self.sink = sink
        self.name = name
        self._t0: int | None = None

    def __enter__(self) -> "PhaseTimer":
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        assert self._t0 is not None, "PhaseTimer.__exit__ called without __enter__"
        elapsed_ms = (time.perf_counter_ns() - self._t0) // 1_000_000
        self.sink.on_phase(self.name, int(elapsed_ms))
        return False  # never suppress exceptions
