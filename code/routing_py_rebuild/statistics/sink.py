"""Sink protocol + IterEvent dataclass + NullSink default.

Per REFACTOR_GOALS.md §1-1-a. sink.py is the leaf of the dependency chain:
nothing else in `statistics/` imports from it through indirection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class IterEvent:
    """One loss_function evaluation. See §3-2 trace[] schema."""

    iter: int
    loss: float
    wall_ms: int
    is_new_best: bool = False


@runtime_checkable
class StatsSink(Protocol):
    """Sink interface.

    Implementations should stay picklable so that a future multiprocessing
    swap (§1-1-a 已知风险) does not require an interface break.
    """

    def on_iter(self, ev: IterEvent) -> None: ...

    def on_phase(self, name: str, wall_ms: int) -> None: ...

    def finalize(self) -> dict: ...


class NullSink:
    """No-op sink. Default for Optimizer; zero state, picklable."""

    def on_iter(self, ev: IterEvent) -> None:
        return None

    def on_phase(self, name: str, wall_ms: int) -> None:
        return None

    def finalize(self) -> dict:
        return {}
