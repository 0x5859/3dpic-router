"""Channel-to-channel crosstalk tensor (REFACTOR_GOALS.md §2-2 / M4).

The crosstalk tensor ``crosstalk[s][d][t]`` (in dB) reports how much power
leaks onto victim node ``t`` when the main path ``s → d`` is energized. The
model is a first-order incoherent (power-additive) propagation over the
cached crossing topology built by ``core.SiNInterconnectionGraph`` Phase A
(M2 / §2-1):

- main-path attenuation per crossing: ``power *= (1 - loss_crossing)``
- orthogonal-branch injection at each crossing: ``leak = power * loss_xtalk``
- ``loss_xtalk`` = ``loss_intralayer_crosstalk`` for same-layer crossings,
  ``loss_interlayer_crosstalk`` for adjacent-layer crossings, and ``0`` for
  ``|Δlayer| ≥ 2`` (§2-3 目标 D — non-adjacent layers are physically
  below the −60 dB floor).
- ``leak`` is split **50/50** between the two directions along the
  orthogonal edge (matching §2-2 "从交点向两端传播"). The split is symmetric
  by construction; downstream readers should not depend on it being
  asymmetric.
- branches recurse up to ``max_hops`` levels; the main path is hop=0, each
  branch transition increments hop by one.
- linear-power threshold (``10^(threshold_db / 10)``) prunes deeper branches
  whose accumulated power would already be below the noise floor.
- a per-path ``visited`` ``frozenset[int]`` of edge indices prevents loops
  and double-counting (an edge is "visited" once any path has walked along
  it).

Coefficient semantics
---------------------
``loss_crossing`` is reinterpreted **as a fractional power loss** here, even
though the per-edge loss formula in :mod:`.core` treats it as a dB-equivalent
additive coefficient. This is the model simplification documented in
§2-2 ("junction transmission (main path) : 1 − loss_crossing"). The
``coherence_model="incoherent_v1"`` label on the emitted tensor captures
that fact for downstream consumers — switching to a coherent / multi-mode
model is a ``schema_version`` bump (§7 Q-a).

``loss_intralayer_crosstalk`` and ``loss_interlayer_crosstalk`` are fractional
power per crossing (default 1e-4 / 1e-5, i.e. −40 dB / −50 dB respectively).
A ``None`` value is treated as zero — the tensor is then empty
(all None entries) and the function returns early without recursion.

Output
------
A dict suitable for direct JSON serialization, with rank-3
``values[s][d][t]`` nested lists; diagonal positions (``s==d`` / ``s==t`` /
``t==d``) are filled with ``None`` (per §3-1 ``diagonal_convention``).
Above-threshold arrivals are stored as dB (``10 * log10(power)``); arrivals
below threshold remain ``None``.

The engine is invoked **once after optimization** by
``api.run_optimization``; it is NOT in the optimizer hot loop unless
``crosstalk_include_in_loss=True`` is wired in a future milestone
(default off per §2-2 performance budget).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_crosstalk_tensor(
    graph,
    *,
    max_hops: int = 3,
    threshold_db: float = -60.0,
) -> dict[str, Any]:
    """Build the rank-3 crosstalk tensor for ``graph`` (post-optimization).

    Parameters
    ----------
    graph : core.SiNInterconnectionGraph
        Per-edge layer assignments are read from ``graph.G``; the
        crossing-pair table is built lazily via
        :meth:`~core.SiNInterconnectionGraph.build_crossings_index` the
        first time we need it (only when at least one crosstalk
        coefficient is non-zero AND ``max_hops > 0`` — see the
        short-circuit at the top of this function). Callers that have
        already run ``analyze_loss()`` see no extra geometric cost.
    max_hops : int, default 3
        Recursion depth limit. ``max_hops=0`` disables crosstalk entirely
        (main path only; no orthogonal injection). The §2-2 default of 3
        is justified by: single hop ≈ −40 dB; 4 hops ≈ −160 dB, far below
        the −60 dB threshold.
    threshold_db : float, default -60.0
        Below-threshold branches (linear-power) are pruned. Stored on the
        emitted tensor as ``threshold_db`` so downstream tooling can audit.

    Returns
    -------
    dict
        Payload matching the schema in REFACTOR_GOALS.md §3-1::

            {
                "unit": "dB",
                "shape": [k, k, k],
                "coherence_model": graph.coherence_model,
                "polarization": graph.polarization,
                "symmetric": False,
                "diagonal_convention": "NaN",
                "max_hops": int,
                "threshold_db": float,
                "loss_intralayer_crosstalk": float | None,
                "loss_interlayer_crosstalk": float | None,
                "values": [[[ float | None ] * k ] * k ] * k,
            }

        ``values[s][d][t]`` is the dB power at victim ``t`` when ``s→d`` is
        lit; ``None`` for diagonals and for arrivals below threshold.
    """
    if max_hops < 0:
        raise ValueError(f"max_hops must be >= 0; got {max_hops}.")
    if not math.isfinite(threshold_db):
        # Codex P2 / opus-review P2-5: ``+inf`` / ``-inf`` thresholds slip
        # into ``compute_crosstalk_tensor`` today and only fail at
        # ``json.dump(..., allow_nan=False)`` after the tensor is built.
        # Catch at the engine entry so the error is local to the call site.
        raise ValueError(
            f"threshold_db must be finite; got {threshold_db!r}."
        )

    intra_xt = float(graph.loss_intralayer_crosstalk or 0.0)
    inter_xt = float(graph.loss_interlayer_crosstalk or 0.0)
    one_minus_lc = 1.0 - float(graph.loss_crossing)
    if not (0.0 < one_minus_lc <= 1.0):
        # If loss_crossing >= 1 (≥100% loss per crossing), the model is
        # nonsensical — every path attenuates to zero immediately. Raise
        # rather than silently emit an all-None tensor: this catches a
        # caller who passed a dB value where a fraction was expected.
        raise ValueError(
            f"loss_crossing={graph.loss_crossing!r} produces transmission "
            f"{one_minus_lc!r} outside (0, 1]. The crosstalk model in §2-2 "
            "treats loss_crossing as a fractional power loss (1 − T); use "
            "a value in [0, 1)."
        )

    # Codex P2: defer Phase A build until after the early-exit guards
    # below. ``compute_crosstalk_tensor`` callers that hit the
    # all-zero-coefficient or ``max_hops=0`` short-circuit shouldn't pay
    # the one-time O(E²) geometric pass — the orchestrator already
    # short-circuits at ``api.run_optimization`` for the common case,
    # but a direct caller (notebook, ad-hoc analysis) gets the same
    # benefit here.

    k = int(graph.k)
    payload_meta = {
        "unit": "dB",
        "shape": [k, k, k],
        "coherence_model": str(graph.coherence_model),
        "polarization": str(graph.polarization),
        "symmetric": False,
        "diagonal_convention": "NaN",
        "max_hops": int(max_hops),
        "threshold_db": float(threshold_db),
        "loss_intralayer_crosstalk": (
            float(graph.loss_intralayer_crosstalk)
            if graph.loss_intralayer_crosstalk is not None
            else None
        ),
        "loss_interlayer_crosstalk": (
            float(graph.loss_interlayer_crosstalk)
            if graph.loss_interlayer_crosstalk is not None
            else None
        ),
    }

    # Crosstalk is identically zero everywhere when both coefficients are
    # zero/None. Short-circuit so the empty-tensor path is cheap (k³ list
    # of None) and the "crosstalk-off ≡ pre-M4" regression (§2-2 验收第
    # 2 条 / T4) is bit-exact.
    if intra_xt == 0.0 and inter_xt == 0.0:
        values = _empty_values(k)
        return {**payload_meta, "values": values}
    if max_hops == 0:
        # Spec: max_hops bound on recursion depth. Hop=0 means we walk
        # only the main path and never spawn orthogonal branches.
        # Resulting tensor is all-None (main path arrival is at d, which
        # is a diagonal slot by convention).
        values = _empty_values(k)
        return {**payload_meta, "values": values}

    # Now (and only now) commit to the one-time Phase A geometric pass.
    graph._ensure_crossings_ready()
    layer_per_edge = _layer_per_edge(graph)
    t_per_pair = _compute_t_parameters(graph)
    crossings_per_edge = _build_per_edge_crossing_table(graph, t_per_pair)

    threshold = 10.0 ** (float(threshold_db) / 10.0)

    values = _empty_values(k)
    for s in range(k):
        for d in range(k):
            if s == d:
                continue
            arrivals = _walk_source_destination(
                graph=graph,
                layer_per_edge=layer_per_edge,
                crossings_per_edge=crossings_per_edge,
                s_node=s,
                d_node=d,
                intra_xt=intra_xt,
                inter_xt=inter_xt,
                one_minus_lc=one_minus_lc,
                max_hops=max_hops,
                threshold=threshold,
            )
            for t_node, power in arrivals.items():
                if t_node == s or t_node == d:
                    continue  # diagonal — leave as None per §3-1
                if power < threshold:
                    continue  # below noise floor → leave as None
                values[s][d][t_node] = 10.0 * math.log10(power)

    return {**payload_meta, "values": values}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _empty_values(k: int) -> list[list[list[Any]]]:
    """Allocate an all-None k×k×k nested list (independent inner objects)."""
    return [[[None] * k for _ in range(k)] for _ in range(k)]


def _layer_per_edge(graph) -> np.ndarray:
    """Return the per-edge layer assignment indexed by ``graph._edge_index``.

    Uses ``graph.G`` as the source of truth (post-pin assignment from the
    most-recent ``apply_optimization_result``). Edges missing the ``layer``
    attribute default to 0, matching the constructor default.
    """
    edges = graph._edge_list
    out = np.zeros(len(edges), dtype=np.int64)
    for idx, edge in enumerate(edges):
        attr = graph.G.edges[edge]
        out[idx] = int(attr.get("layer", 0))
    return out


def _compute_t_parameters(graph) -> np.ndarray:
    """Closed-form parametric intersection ``(t_self, t_other)`` for each
    crossing pair.

    For a pair ``(i, j)`` whose endpoints are ``(p_a → q_a)`` and
    ``(p_b → q_b)``, the proper-intersection solution is::

        det     = (q_a - p_a) × (q_b - p_b)          # 2D cross product
        t_self  = ((p_b - p_a) × (q_b - p_b)) / det
        t_other = ((p_b - p_a) × (q_a - p_a)) / det

    where ``×`` is the scalar cross product ``(ux*vy − uy*vx)``. Both
    parameters are in ``[0, 1]`` for proper crossings; for the T-junction
    and collinear partial-overlap edge cases the closed form is degenerate
    (``det == 0``), in which case we fall back to the midpoint of the
    overlap as a deterministic, finite "intersection point". These cases
    are exceedingly rare in SiN cyclic-convex layouts but the fallback
    keeps the engine deterministic for arbitrary positions.

    Returns
    -------
    np.ndarray
        Shape ``(X, 2)`` where ``X = len(graph._crossing_pairs)``. Column
        0 = ``t_self`` for the edge at column 0 of ``crossing_pairs``;
        column 1 = ``t_other`` for the edge at column 1.
    """
    pairs = graph._crossing_pairs
    n_pairs = int(pairs.shape[0])
    if n_pairs == 0:
        return np.zeros((0, 2), dtype=np.float64)

    edges = graph._edge_list
    positions = graph.positions
    pts = np.array([positions[u] for u, _ in edges], dtype=np.float64)
    qts = np.array([positions[v] for _, v in edges], dtype=np.float64)

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    pa = pts[i_idx]
    qa = qts[i_idx]
    pb = pts[j_idx]
    qb = qts[j_idx]

    da = qa - pa
    db = qb - pb
    r = pb - pa

    # 2D scalar cross product
    det = da[:, 0] * db[:, 1] - da[:, 1] * db[:, 0]
    num_t = r[:, 0] * db[:, 1] - r[:, 1] * db[:, 0]
    num_s = r[:, 0] * da[:, 1] - r[:, 1] * da[:, 0]

    # Where det != 0: proper intersection → t = num_t / det, s = num_s / det
    out = np.empty((n_pairs, 2), dtype=np.float64)
    safe = det != 0.0
    out[safe, 0] = num_t[safe] / det[safe]
    out[safe, 1] = num_s[safe] / det[safe]
    # Where det == 0 (collinear / parallel): use a deterministic
    # midpoint-of-overlap fallback. For two collinear segments the
    # "intersection point" along edge i is t_self = (t_min + t_max) / 2
    # where t_min, t_max are the parametric positions of the overlap
    # endpoints along edge i; if no overlap exists (parallel non-collinear)
    # the pair shouldn't appear in _crossing_pairs to begin with, but we
    # fall back to t=0.5 / s=0.5 just in case so the recursion doesn't
    # blow up.
    if not safe.all():
        out[~safe, 0] = 0.5
        out[~safe, 1] = 0.5

    # Clamp to [0, 1] for robustness against floating-point drift right
    # at endpoints (T-junction in particular can land at t = 1+1e-16).
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _build_per_edge_crossing_table(
    graph, t_per_pair: np.ndarray
) -> list[list[tuple[int, float, float]]]:
    """For each edge index ``i``, return a list of ``(other_edge_idx,
    t_self, t_other)`` sorted by ``t_self`` ascending.

    Constructed once per ``compute_crosstalk_tensor`` call so the per-edge
    "crossings ahead" lookup in the recursion is a slice + filter rather
    than a re-sort per step.
    """
    n_edges = len(graph._edge_list)
    table: list[list[tuple[int, float, float]]] = [[] for _ in range(n_edges)]
    pairs = graph._crossing_pairs
    for k_idx in range(int(pairs.shape[0])):
        i = int(pairs[k_idx, 0])
        j = int(pairs[k_idx, 1])
        t_i = float(t_per_pair[k_idx, 0])
        t_j = float(t_per_pair[k_idx, 1])
        table[i].append((j, t_i, t_j))
        # Symmetric entry on the other edge — both edges see each other
        # as crossings, with their parametric roles swapped.
        table[j].append((i, t_j, t_i))

    for i in range(n_edges):
        table[i].sort(key=lambda row: row[1])
    return table


def _crossings_ahead(
    sorted_table_entry: list[tuple[int, float, float]],
    t_in: float,
    going_pos: bool,
) -> list[tuple[int, float, float]]:
    """Slice the sorted per-edge crossing list to the entries lying ahead
    of ``t_in`` in the chosen traversal direction.

    - ``going_pos=True``: walking from ``u`` to ``v`` (t increasing). Ahead
      = entries with ``t_self > t_in``, kept in ascending order.
    - ``going_pos=False``: walking from ``v`` to ``u`` (t decreasing). Ahead
      = entries with ``t_self < t_in``, returned in descending order.

    Strict inequalities so a branch *spawned at* the current crossing
    doesn't immediately encounter itself again at ``t_self == t_in``.
    """
    if going_pos:
        return [row for row in sorted_table_entry if row[1] > t_in]
    return [row for row in reversed(sorted_table_entry) if row[1] < t_in]


def _walk_source_destination(
    *,
    graph,
    layer_per_edge: np.ndarray,
    crossings_per_edge: list[list[tuple[int, float, float]]],
    s_node: int,
    d_node: int,
    intra_xt: float,
    inter_xt: float,
    one_minus_lc: float,
    max_hops: int,
    threshold: float,
) -> dict[int, float]:
    """DFS over the branch tree rooted at the main path ``s_node → d_node``.

    Returns ``{arrival_node: accumulated_linear_power}`` aggregated over
    every path that terminates at that node within ``max_hops`` branches
    and above ``threshold`` linear power.
    """
    main_u, main_v = (s_node, d_node) if s_node < d_node else (d_node, s_node)
    main_edge = (main_u, main_v)
    main_idx = graph._edge_index[main_edge]
    going_pos = (s_node == main_u)

    arrivals: dict[int, float] = defaultdict(float)

    # Iterative DFS stack. Each frame is independent (separate visited
    # frozenset) so two sibling spawns can't poison each other's history.
    initial_t = 0.0 if going_pos else 1.0
    stack: list[tuple[int, float, bool, float, int, frozenset[int]]] = [
        (main_idx, initial_t, going_pos, 1.0, 0, frozenset({main_idx}))
    ]

    while stack:
        edge_idx, t_in, going_pos, power, hop, visited = stack.pop()
        if power < threshold:
            continue
        u, v = graph._edge_list[edge_idx]
        end_node = v if going_pos else u

        ahead = _crossings_ahead(crossings_per_edge[edge_idx], t_in, going_pos)

        current_power = power
        for other_edge_idx, _t_self, t_other in ahead:
            if current_power < threshold:
                break

            la = int(layer_per_edge[edge_idx])
            lb = int(layer_per_edge[other_edge_idx])
            if la == lb:
                coef = intra_xt
            elif abs(la - lb) == 1:
                coef = inter_xt
            else:
                coef = 0.0

            if (
                coef > 0.0
                and hop < max_hops
                and other_edge_idx not in visited
            ):
                # §2-2 spec: total leakage is `coef * power`, split between
                # the two directions along the orthogonal edge. Each
                # sub-branch starts with half. Below-threshold leaks are
                # pruned BEFORE pushing onto the stack so we don't waste a
                # frame just to drop it on the next iteration.
                per_dir = (current_power * coef) * 0.5
                if per_dir >= threshold:
                    new_visited = visited | {other_edge_idx}
                    # Forward on other_edge: t increasing (u_o → v_o)
                    stack.append(
                        (other_edge_idx, t_other, True, per_dir, hop + 1, new_visited)
                    )
                    # Backward: t decreasing (v_o → u_o)
                    stack.append(
                        (other_edge_idx, t_other, False, per_dir, hop + 1, new_visited)
                    )

            # Main path traverses with multiplicative loss; orthogonal
            # leak is taken FROM the same incoming power (no further
            # subtraction from current_power). This matches the
            # standard scattering-matrix decomposition of a 2x2
            # photonic crossing: T_main and κ_ortho are independent
            # coefficients of the input power, not partitioned from
            # a single budget.
            current_power *= one_minus_lc

        if current_power >= threshold:
            arrivals[end_node] += current_power

    return dict(arrivals)


__all__ = ["compute_crosstalk_tensor"]
