"""Statistics module — optimizer trace + phase timings + reports.

See REFACTOR_GOALS.md §1-1-a for the contract.
"""
from .recorder import RunRecorder
from .reporters import write_run_report
from .sink import IterEvent, NullSink, StatsSink
from .timer import PhaseTimer

__all__ = [
    "IterEvent",
    "NullSink",
    "PhaseTimer",
    "RunRecorder",
    "StatsSink",
    "write_run_report",
]
