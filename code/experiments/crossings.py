"""Geometric intralayer crossing counter (experiment harness, spec §7;
REFACTOR_GOALS.md §2-1/§2-3).

Reports the geometric-event count (spec D4): each same-layer edge-pair
intersection counted once. Reuses the cached integer crossing-pair index
(``SiNInterconnectionGraph._crossing_pairs`` / ``_perimeter_mask``,
populated by the public idempotent ``build_crossings_index()``) so the
per-eval cost is O(X) in the number of cached crossing pairs — a per-eval
shapely O(E²) recount is prohibited (spec §7 hard performance constraint).

Reconciliation (spec §17 open question, resolved by reading core.py during
M-D):

* ``loss_function`` derives ``pinned`` as ``np.round(layers)`` → ``int64``
  → ``pinned[_perimeter_mask] = perimeter_layer`` (core.py:855-862); this
  counter replicates that derivation byte-for-byte (using
  ``graph.perimeter_layer``, the M3 generalization, not a hardcoded 0).
* Python ``run_report.summary.layers[i].crossings``
  ``== int(graph.total_crossings_of_sub_G[i])``
  ``== count_crossings_with_detail(sub_G[i])`` ``num_crosses``
  (core.py:336 increments once per ``i<j`` crossing pair) — already
  geometric, no ``// 2`` on the intralayer side. C++ emits the same as
  per-layer ``crossings = total_intra / 2`` (spec D4, main.cpp:509).
* ``create_subgraphs`` buckets edges by ``d["layer"] == layer`` for
  ``layer in range(L)`` (core.py:795-798): an edge whose post-pin layer
  is outside ``[0, L)`` is in no per-layer subgraph and counts toward no
  ``summary.layers[].crossings``. This counter applies the identical
  ``[0, L)`` gate so it reconciles exactly with the run_report value on
  both backends — the same raw-value/no-clamp Python↔C++ parity contract
  that REFACTOR_GOALS.md §1-2 design-tradeoff #4 / §1-4 P2-O5 lock for
  ``loss_function`` / ``apply_optimization_result``. In production the
  harness clamps ``x`` to ``[0, L-1]`` before calling this counter
  (spec §5), so the gate is a no-op on real optimizer output; it locks
  the contract and is covered by an explicit out-of-range oracle case.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from routing_py_rebuild.core import SiNInterconnectionGraph


def count_intralayer_geometric(
    graph: SiNInterconnectionGraph, layers: Sequence[float]
) -> int:
    """Geometric intralayer crossing count for ``layers`` on ``graph``.

    O(X) in the number of cached crossing pairs. The perimeter ring
    (``|u-v|`` in ``{1, k-1}``) is pinned to ``graph.perimeter_layer``
    exactly as ``core.py::loss_function`` does, so the result is
    consistent with the run_report / subgraph convention on both backends.

    Parameters
    ----------
    layers:
        One value per edge in ``graph._edge_list`` order. Rounded the way
        ``loss_function`` rounds (``np.round`` → ``int64``); a wrong-length
        vector raises ``ValueError`` with the same message shape as
        ``loss_function`` (core.py:856-859).

    Returns
    -------
    int
        Count of geometrically-crossing same-layer edge pairs whose shared
        post-pin layer is in ``[0, graph.L)`` (each pair once).
    """
    graph.build_crossings_index()  # idempotent; ensures the cached index
    n_edges = len(graph._edge_list)

    layers_int = np.round(np.asarray(layers, dtype=float)).astype(np.int64)
    if layers_int.shape != (n_edges,):
        raise ValueError(
            f"layers has shape {layers_int.shape}, expected ({n_edges},)"
        )

    pinned = layers_int.copy()
    pinned[graph._perimeter_mask] = graph.perimeter_layer

    pairs = graph._crossing_pairs
    if pairs.shape[0] == 0:
        return 0

    la = pinned[pairs[:, 0]]
    lb = pinned[pairs[:, 1]]
    same = la == lb
    # [0, L) reconciliation gate (see module docstring): create_subgraphs
    # only buckets layers in range(L); out-of-range same-layer pairs drop
    # from every per-layer subgraph and so are absent from
    # run_report.summary.layers[].crossings. `same` already implies
    # la == lb, so gating on `la` alone is sufficient and correct.
    in_range = (la >= 0) & (la < graph.L)
    return int(np.count_nonzero(same & in_range))
