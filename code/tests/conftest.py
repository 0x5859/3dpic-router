"""Shared pytest fixtures for the routing test suite.

The post-2026-05-15 wpl=1 default emits an informational notice exactly
once per process (REFACTOR_GOALS.md §4 M3 附注 "Warning 策略"). For
tests to reliably observe the notice via ``pytest.warns``, the
process-wide flag must be reset between tests; do it autouse so no
individual test has to remember.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_wpl_one_notice_flags() -> None:
    """Reset both wpl=1 one-time flags before every test.

    The constructor path lives in ``routing_py_rebuild.core`` and the
    loader path in ``routing_py_rebuild.plotting.plot_data``; each owns
    its own flag (the §1-3-c boundary forbids plotting from importing
    core). Resetting both keeps every test starting from a clean state.
    """
    from routing_py_rebuild import core
    from routing_py_rebuild.plotting import plot_data

    core._reset_wpl_notice_for_tests()
    plot_data._reset_wpl_notice_for_tests()
    yield
