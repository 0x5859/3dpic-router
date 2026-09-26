"""M2 acceptance tests for cached crossing topology (REFACTOR_GOALS.md §2-1).

Two layers of parity coverage:

1. **Phase A oracle** — the cached crossing-pair index must agree, pair-by
   -pair, with the pre-M2 O(E^2) brute-force ``edge_crosses`` implementation,
   across both the cyclic-convex fast path (SiN square layouts) and the
   geometric fallback (non-convex / non-cyclic positions). Covers k ∈
   {8, 12, 20, 40, 80, 160} per the §2-1 acceptance list, plus explicit
   degenerate-case fixtures (shared endpoint, collinear partial overlap,
   T-junction, endpoint-on-edge).

2. **Phase B oracle** — given the cached index, the new ``loss_function``
   must produce the same per-edge intra/inter crossing counts and the
   same aggregated loss as the pre-M2 path (``_apply_layer_assignment``
   → ``create_subgraphs`` → unique-layer mean). Uses hypothesis for
   property-based coverage on small k; deterministic fixed-seed inputs at
   larger k where the brute-force oracle would be prohibitively slow.

Also smoke-tests the §3-2 ``initial_crossing_count_ms`` /
``loss_analysis_ms`` phase emission added by M2.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from routing_py_rebuild.api import make_graph, run_optimization
from routing_py_rebuild.core import SiNInterconnectionGraph
from routing_py_rebuild.positions import (
    distribute_nodes,
    distribute_nodes_around_polygon,
    distribute_nodes_around_triangle,
)


# ---------------------------------------------------------------------------
# Reference helpers — pre-M2 brute force, used as ground truth.
# ---------------------------------------------------------------------------


def _brute_force_pair_set(graph: SiNInterconnectionGraph) -> set[tuple[int, int]]:
    """Compute the full crossing-pair index using only the original
    ``edge_crosses`` method — no Phase A code path involved.

    This is the canonical ground truth that the cached index must reproduce.
    """
    edges = list(graph.G.edges())
    pairs: set[tuple[int, int]] = set()
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if graph.edge_crosses(edges[i], edges[j]):
                pairs.add((i, j))
    return pairs


def _pre_m2_per_edge_counts(
    graph: SiNInterconnectionGraph, layers: np.ndarray
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """Run the pre-M2 algorithm (``_apply_layer_assignment`` +
    ``create_subgraphs``) and harvest per-edge crossings /
    interlayer-crossings from sub_G.
    """
    nx.set_edge_attributes(graph.G, 0, "layer")
    nx.set_edge_attributes(graph.G, 0, "crossings")
    nx.set_edge_attributes(graph.G, 0, "loss")
    nx.set_edge_attributes(graph.G, 0, "interlayercrossings")

    layers_int = np.round(np.asarray(layers)).astype(int)
    graph._apply_layer_assignment(layers_int)
    graph.create_subgraphs()

    intra: dict[tuple[int, int], int] = {}
    inter: dict[tuple[int, int], int] = {}
    for sg in graph.sub_G:
        for u, v, d in sg.edges(data=True):
            intra[(u, v)] = int(d.get("crossings", 0))
            inter[(u, v)] = int(d.get("interlayercrossings", 0))
    return intra, inter


def _phase_b_per_edge_counts(
    graph: SiNInterconnectionGraph, layers: np.ndarray
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """Reproduce Phase B's per-edge counting in-place so we can compare
    against the pre-M2 oracle directly. Mirrors the body of
    ``loss_function`` up to the intra/inter computation.
    """
    graph._ensure_crossings_ready()
    n_edges = len(graph._edge_list)
    layers_int = np.round(np.asarray(layers, dtype=float)).astype(np.int64)
    pinned = layers_int.copy()
    pinned[graph._perimeter_mask] = 0

    pairs = graph._crossing_pairs
    intra_arr = np.zeros(n_edges, dtype=np.int64)
    inter_arr = np.zeros(n_edges, dtype=np.int64)
    if pairs.shape[0]:
        la = pinned[pairs[:, 0]]
        lb = pinned[pairs[:, 1]]
        same = la == lb
        adj = np.abs(la - lb) == 1
        if same.any():
            flat = np.concatenate((pairs[same, 0], pairs[same, 1]))
            intra_arr = np.bincount(flat, minlength=n_edges).astype(np.int64)
        if adj.any():
            flat = np.concatenate((pairs[adj, 0], pairs[adj, 1]))
            inter_arr = np.bincount(flat, minlength=n_edges).astype(np.int64)

    intra = {graph._edge_list[i]: int(intra_arr[i]) for i in range(n_edges)}
    inter = {graph._edge_list[i]: int(inter_arr[i]) for i in range(n_edges)}
    return intra, inter


# ---------------------------------------------------------------------------
# Phase A — pair-set parity vs the brute-force oracle.
# ---------------------------------------------------------------------------

# k=80 / k=160 reach E²/2 ≈ 5e6 / 8e7 brute-force pair comparisons; bounded
# by ``edge_crosses``'s shapely overhead (~5 µs / call) they would run for
# tens of seconds and minutes respectively, which is too slow for routine
# CI. Phase A's algorithmic equivalence is already pinned down on the
# smaller k via this test, so we keep ground-truth coverage at k ≤ 40 and
# add a smoke test for k=80, 160 that asserts pair count == \binom{k}{4}
# (the exact crossing count for convex K_n, which the SiN square layout
# produces).
@pytest.mark.parametrize("k", [8, 12, 20, 40])
def test_phase_a_matches_brute_force_sin_layout(k: int) -> None:
    """Cached pair index agrees with the brute-force ``edge_crosses`` set on
    SiN square layouts (the cyclic-convex fast path)."""
    graph = make_graph(k=k, output_dir=None)
    graph.build_crossings_index()
    cached = set(map(tuple, graph._crossing_pairs.tolist()))
    expected = _brute_force_pair_set(graph)
    assert cached == expected, (
        f"k={k}: cached pair set diverges; missing={sorted(expected - cached)[:5]}, "
        f"extra={sorted(cached - expected)[:5]}"
    )


@pytest.mark.parametrize("k", [80, 160])
def test_phase_a_pair_count_smoke(k: int) -> None:
    """At k≥80 the brute-force oracle is too slow for CI; assert the cached
    index has the expected \\binom{k}{4} crossing count AND the matching
    per-edge degree sum for the SiN convex layout. The cardinality alone
    could be matched by a buggy implementation that emits the right *count*
    of arbitrary pairs, so we also verify each pair contributes to exactly
    two endpoints (sum of per-edge degrees = 2 × pair count = 4 × \\binom{k}{4}).
    """
    graph = make_graph(k=k, output_dir=None)
    graph.build_crossings_index()
    pairs = graph._crossing_pairs
    expected_pairs = k * (k - 1) * (k - 2) * (k - 3) // 24
    assert pairs.shape[0] == expected_pairs
    n_edges = len(graph._edge_list)
    # Pair indices must be in canonical (i < j) ordering and within bounds.
    assert pairs.dtype == np.int64
    assert pairs[:, 0].max() < n_edges
    assert pairs[:, 1].max() < n_edges
    assert bool(np.all(pairs[:, 0] < pairs[:, 1]))
    # No duplicates — degree sum alone is tautological for in-bound (N,2),
    # but pair-set cardinality after dedup catches an implementation that
    # emits the right count of *non-unique* pairs.
    unique_pairs = np.unique(pairs, axis=0)
    assert unique_pairs.shape == pairs.shape
    # Per-edge degree distribution sanity. For K_n in convex position,
    # the per-edge crossing count has a closed form analytic value; here we
    # just verify the row-sum equals 2 × pair count (each pair contributes
    # once to each of its endpoints), which is non-trivial when paired with
    # uniqueness above.
    deg = np.bincount(pairs.flatten(), minlength=n_edges)
    assert int(deg.sum()) == 2 * expected_pairs


# ---------------------------------------------------------------------------
# Phase A — robustness of ``_is_cyclic_convex_positions``.
# ---------------------------------------------------------------------------


def _regular_polygon(k: int, perm: list[int]) -> dict[int, tuple[float, float]]:
    """Place k regular-polygon vertices and return positions indexed by the
    given permutation. ``perm = list(range(k))`` gives the natural cyclic
    polygon; a permutation like ``[0, 2, 4, 1, 3]`` for k=5 traces a
    pentagram (winding 2).
    """
    pts = [
        (float(np.cos(2 * np.pi * i / k)), float(np.sin(2 * np.pi * i / k)))
        for i in range(k)
    ]
    return {i: pts[perm[i]] for i in range(k)}


def test_phase_a_rejects_pentagram_fast_path() -> None:
    """A pentagram-ordered layout has all-same-sign turns but winding 2;
    the fast path's alternating-endpoints test would produce wrong crossing
    pairs. ``_is_cyclic_convex_positions`` must reject it so the geometric
    fallback handles it. Verify the fallback path matches brute force.
    """
    positions = _regular_polygon(5, perm=[0, 2, 4, 1, 3])  # pentagram order
    graph = SiNInterconnectionGraph(k=5, positions=positions)
    assert not graph._is_cyclic_convex_positions(), (
        "pentagram (winding 2) must NOT pass the cyclic-convex detector"
    )
    graph.build_crossings_index()
    cached = set(map(tuple, graph._crossing_pairs.tolist()))
    expected = _brute_force_pair_set(graph)
    assert cached == expected


def test_phase_a_rejects_coincident_positions() -> None:
    """If two distinct node indices map to the same (x, y), the alternating
    test's open-interval semantics break. Detector must reject and force
    the geometric fallback.
    """
    positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 1.0), 3: (1.0, 0.0)}
    # nodes 1 and 3 share coords
    graph = SiNInterconnectionGraph(k=4, positions=positions)
    assert not graph._is_cyclic_convex_positions(), (
        "coincident vertices must NOT pass the cyclic-convex detector"
    )
    graph.build_crossings_index()
    cached = set(map(tuple, graph._crossing_pairs.tolist()))
    expected = _brute_force_pair_set(graph)
    assert cached == expected


def test_phase_a_accepts_simple_regular_polygon() -> None:
    """Sanity check the detector's TRUE branch: a regular hexagon in cyclic
    index order does pass and the fast-path output matches brute force.
    """
    positions = _regular_polygon(6, perm=list(range(6)))
    graph = SiNInterconnectionGraph(k=6, positions=positions)
    assert graph._is_cyclic_convex_positions()
    graph.build_crossings_index()
    cached = set(map(tuple, graph._crossing_pairs.tolist()))
    expected = _brute_force_pair_set(graph)
    assert cached == expected


# ---------------------------------------------------------------------------
# Phase A — geometric fallback path (non-cyclic positions).
# ---------------------------------------------------------------------------


def _non_convex_positions(n: int, seed: int) -> dict[int, tuple[float, float]]:
    """Random points inside a unit square — no convex-polygon structure,
    forces ``_ensure_crossings_ready`` down the geometric fallback path.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    return {i: (float(pts[i, 0]), float(pts[i, 1])) for i in range(n)}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_phase_a_fallback_matches_brute_force(seed: int) -> None:
    """Random non-cyclic-convex positions exercise the geometric fallback
    branch. Compare against brute-force ``edge_crosses``.
    """
    k = 12
    positions = _non_convex_positions(k, seed)
    graph = SiNInterconnectionGraph(k=k, positions=positions)
    assert not graph._is_cyclic_convex_positions(), (
        "fixture should NOT be cyclic-convex"
    )
    graph.build_crossings_index()
    cached = set(map(tuple, graph._crossing_pairs.tolist()))
    expected = _brute_force_pair_set(graph)
    assert cached == expected


# ---------------------------------------------------------------------------
# Phase A — collinear side runs (triangle / regular polygon).
#
# Triangle and regular-polygon layouts place several nodes on each straight
# side. Interpolation leaves those side nodes ~1e-17 (relative) off their
# line with either sign, so an exact-zero turn test rejected the layouts and
# the geometric fallback then resolved every near-collinear configuration by
# round-off: triangle k=12 gave 463 pairs, and 471 once the coordinates were
# rounded to 12 decimals. ``_is_cyclic_convex_positions`` treats a turn with
# |sin| <= ``_COLLINEAR_TOL`` as a straight step, so these layouts take the
# cyclic fast path. With every node on a convex outline, ``edge_crosses``
# semantics (T-junctions and partial collinear overlaps cross, containment
# does not) reduce to the alternating-endpoints rule, so they cross exactly
# like the circle with the same node order — whose brute-force set has no
# collinear nodes for round-off to get wrong. REFACTOR_GOALS.md §2-1.
# ---------------------------------------------------------------------------


def _triangle_positions(k: int) -> dict[int, tuple[float, float]]:
    """Equilateral-triangle boundary layout with ``k`` total nodes split as
    evenly as possible across the 3 sides (corners excluded)."""
    base, rem = divmod(k, 3)
    counts = [base + (1 if s < rem else 0) for s in range(3)]
    return distribute_nodes_around_triangle(counts)


@functools.cache
def _convex_reference_pairs(k: int) -> frozenset[tuple[int, int]]:
    """Brute-force ``edge_crosses`` pairs of the circle with ``k`` nodes:
    the crossing set of every layout with its nodes on a convex outline in
    index order, free of collinear nodes."""
    graph = SiNInterconnectionGraph(k=k, positions=distribute_nodes(shape="circle", k=k))
    return frozenset(_brute_force_pair_set(graph))


def _cached_pairs(positions: dict[int, tuple[float, float]]) -> set[tuple[int, int]]:
    graph = SiNInterconnectionGraph(k=len(positions), positions=positions, L=3)
    graph.build_crossings_index()
    return set(map(tuple, graph._crossing_pairs.tolist()))


def test_square_collinear_sides_cross_like_the_circle() -> None:
    """Grounds the reference: the square's side nodes are exactly collinear
    (exact float coordinates), and its brute-force ``edge_crosses`` set —
    T-junctions and partial overlaps included — equals the circle's."""
    for k in (12, 20):
        square = SiNInterconnectionGraph(
            k=k, positions=distribute_nodes(shape="square", nodes_per_side=k // 4)
        )
        assert _brute_force_pair_set(square) == _convex_reference_pairs(k)


@pytest.mark.parametrize("k", [9, 12, 20])
def test_phase_a_collinear_triangle_takes_fast_path(k: int) -> None:
    """Triangle boundary layout (collinear per-side runs, ~1e-17 off their
    lines): the convexity detector accepts it, its pairs are the convex
    reference set (C(k, 4) of them), and rounding the coordinates — which
    used to move the fallback's count from 463 to 471 at k=12 — changes
    nothing.
    """
    positions = _triangle_positions(k)
    graph = SiNInterconnectionGraph(k=k, positions=positions, L=3)
    assert graph._is_cyclic_convex_positions(), (
        "collinear-side triangle must take the cyclic fast path"
    )
    cached = _cached_pairs(positions)
    assert cached == _convex_reference_pairs(k)
    assert len(cached) == math.comb(k, 4)
    rounded = {n: (round(x, 12), round(y, 12)) for n, (x, y) in positions.items()}
    assert _cached_pairs(rounded) == cached


@pytest.mark.parametrize(
    "n_sides,nodes_per_side",
    [(3, 4), (4, 3), (5, 4), (3, 7)],
)
def test_phase_a_collinear_polygon_takes_fast_path(
    n_sides: int, nodes_per_side: int
) -> None:
    """Regular-polygon boundary layout (collinear per-side runs, k =
    n_sides * nodes_per_side incl. k=12 and k=20): accepted by the
    convexity detector, crossing like the convex reference.
    """
    k = n_sides * nodes_per_side
    positions = distribute_nodes_around_polygon(n_sides, nodes_per_side)
    graph = SiNInterconnectionGraph(k=k, positions=positions, L=3)
    assert graph._is_cyclic_convex_positions(), (
        "collinear-side polygon must take the cyclic fast path"
    )
    assert _cached_pairs(positions) == _convex_reference_pairs(k)


def test_phase_a_collinear_tolerance_is_round_off_only() -> None:
    """The straight-step tolerance covers round-off, not geometry: a real
    dent (a side node pushed 1e-6 of the side inward) and a straight step
    that reverses direction both still fail the convexity check."""
    dented = distribute_nodes(shape="square", nodes_per_side=3)  # side 4
    x, y = dented[1]  # top side, middle node
    dented[1] = (x, y - 4e-6)
    assert not SiNInterconnectionGraph(k=12, positions=dented)._is_cyclic_convex_positions()
    row = {0: (1.0, 0.0), 1: (2.0, 0.0), 2: (3.0, 0.0), 3: (4.0, 0.0)}
    assert not SiNInterconnectionGraph(k=4, positions=row)._is_cyclic_convex_positions()


# ---------------------------------------------------------------------------
# Phase A — explicit degenerate-case coverage.
# ---------------------------------------------------------------------------


def test_phase_a_shared_endpoint_not_crossing() -> None:
    """Edges meeting at a vertex are NOT counted as crossings (matches the
    pre-M2 ``edge_crosses`` shared-endpoint exclusion).
    """
    # K_4 on a 2x2 grid: nodes are corners. Edges that share a corner
    # cannot cross.
    positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
    graph = SiNInterconnectionGraph(k=4, positions=positions)
    graph.build_crossings_index()
    cached = set(map(tuple, graph._crossing_pairs.tolist()))
    expected = _brute_force_pair_set(graph)
    # The only proper crossing in this K_4 is the diagonals (0,3) and (1,2).
    # Edge index for (0, 3) and (1, 2) in nx.complete_graph(4) order:
    edges = list(graph.G.edges())
    diag1 = edges.index((0, 3))
    diag2 = edges.index((1, 2))
    expected_pair = (min(diag1, diag2), max(diag1, diag2))
    assert cached == {expected_pair}
    assert cached == expected


def test_phase_a_collinear_partial_overlap_counts_as_cross() -> None:
    """Bit-exact reproduction of ``edge_crosses`` partial-collinear quirk.

    Four collinear points 0, 1, 2, 3 along the x-axis at positions
    (1, 0), (2, 0), (3, 0), (4, 0). Edges (0, 2) and (1, 3) partially
    overlap; ``edge_crosses`` returns True for this configuration (one
    endpoint of each segment lies strictly inside the other's interval),
    so Phase A must include it too.
    """
    positions = {0: (1.0, 0.0), 1: (2.0, 0.0), 2: (3.0, 0.0), 3: (4.0, 0.0)}
    graph = SiNInterconnectionGraph(k=4, positions=positions)
    # All four points are collinear; the polygon-convexity test returns
    # False so we hit the geometric fallback (cyclic alternating would
    # still give the right answer here, but this exercises the slower
    # path's degenerate-handling code).
    assert not graph._is_cyclic_convex_positions()
    graph.build_crossings_index()
    cached = set(map(tuple, graph._crossing_pairs.tolist()))
    expected = _brute_force_pair_set(graph)
    assert cached == expected
    # Sanity: (0, 2) vs (1, 3) is in the set; (0, 3) vs (1, 2) is not
    # (full containment).
    edges = list(graph.G.edges())
    e02 = edges.index((0, 2))
    e13 = edges.index((1, 3))
    e03 = edges.index((0, 3))
    e12 = edges.index((1, 2))
    assert (min(e02, e13), max(e02, e13)) in cached
    assert (min(e03, e12), max(e03, e12)) not in cached


def test_phase_a_t_junction_endpoint_on_interior() -> None:
    """T-junction: endpoint of one edge lies strictly on the interior of
    another (non-collinear case). ``edge_crosses`` returns True.
    """
    # Edge (0, 2): from (0,0) to (2,0). Edge (1, 3): from (1,0) to (1, 1).
    # Endpoint (1, 0) of edge (1, 3) lies on segment (0, 2) interior.
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
        3: (1.0, 1.0),
    }
    graph = SiNInterconnectionGraph(k=4, positions=positions)
    graph.build_crossings_index()
    cached = set(map(tuple, graph._crossing_pairs.tolist()))
    expected = _brute_force_pair_set(graph)
    assert cached == expected


# ---------------------------------------------------------------------------
# Phase B — per-edge crossing counts match the pre-M2 path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [8, 12, 20, 40])
@pytest.mark.parametrize("pattern", ["all0", "all1", "alternating", "random_seed_2026"])
def test_phase_b_per_edge_counts_match_pre_m2(k: int, pattern: str) -> None:
    """Per-edge intra / inter crossing counts must agree bit-exact with the
    pre-M2 ``create_subgraphs`` + ``count_interlayercrossings`` path.

    For L=2 the new vectorized classification (Phase B) is bit-identical
    to the original — the §2-1 acceptance criterion ``逐边 crossing 计数
    bit-exact 一致``.
    """
    graph_a = make_graph(k=k, output_dir=None)
    graph_a.build_crossings_index()
    graph_b = make_graph(k=k, output_dir=None)
    graph_b.build_crossings_index()
    n_edges = len(graph_a._edge_list)

    if pattern == "all0":
        layers = np.zeros(n_edges, dtype=float)
    elif pattern == "all1":
        layers = np.ones(n_edges, dtype=float)
    elif pattern == "alternating":
        layers = np.array([i % 2 for i in range(n_edges)], dtype=float)
    else:  # random_seed_2026
        rng = np.random.default_rng(seed=2026 + k)
        layers = rng.integers(0, 2, size=n_edges).astype(float)

    pre_intra, pre_inter = _pre_m2_per_edge_counts(graph_a, layers)
    new_intra, new_inter = _phase_b_per_edge_counts(graph_b, layers)

    # Pre-M2 counts only stamp non-zero layers in their respective subgraph;
    # for parity we compare per-edge totals — edges not in pre_intra map to 0.
    for edge in graph_b._edge_list:
        assert pre_intra.get(edge, 0) == new_intra[edge], (
            f"k={k} pattern={pattern} edge={edge}: pre={pre_intra.get(edge, 0)} "
            f"new={new_intra[edge]}"
        )
        assert pre_inter.get(edge, 0) == new_inter[edge], (
            f"k={k} pattern={pattern} edge={edge}: pre={pre_inter.get(edge, 0)} "
            f"new={new_inter[edge]}"
        )


@pytest.mark.parametrize("k", [8, 12, 20, 40])
@pytest.mark.parametrize("pattern", ["all0", "all1", "alternating", "random_seed_2026"])
def test_phase_b_loss_value_matches_pre_m2(k: int, pattern: str) -> None:
    """The aggregated mean-loss return value matches the pre-M2 path to
    within float-summation precision (1e-12 absolute).
    """
    graph_a = make_graph(k=k, output_dir=None)
    graph_a.build_crossings_index()
    graph_b = make_graph(k=k, output_dir=None)
    graph_b.build_crossings_index()
    n_edges = len(graph_a._edge_list)

    if pattern == "all0":
        layers = np.zeros(n_edges, dtype=float)
    elif pattern == "all1":
        layers = np.ones(n_edges, dtype=float)
    elif pattern == "alternating":
        layers = np.array([i % 2 for i in range(n_edges)], dtype=float)
    else:
        rng = np.random.default_rng(seed=2026 + k)
        layers = rng.integers(0, 2, size=n_edges).astype(float)

    # Pre-M2: reproduce the unique-layer mean.
    nx.set_edge_attributes(graph_a.G, 0, "layer")
    nx.set_edge_attributes(graph_a.G, 0, "crossings")
    nx.set_edge_attributes(graph_a.G, 0, "loss")
    nx.set_edge_attributes(graph_a.G, 0, "interlayercrossings")
    layers_int = np.round(np.asarray(layers)).astype(int)
    graph_a._apply_layer_assignment(layers_int)
    graph_a.create_subgraphs()
    losses: list[float] = []
    for _layer in np.unique(layers_int):
        if _layer >= len(graph_a.sub_G):
            continue
        sg = graph_a.sub_G[int(_layer)]
        for edge in sg.edges():
            losses.append(sg.edges[edge]["loss"])
    pre_mean = float(np.mean(losses)) if losses else 0.0

    new_mean = graph_b.loss_function(layers)
    assert abs(new_mean - pre_mean) < 1e-12, (
        f"k={k} pattern={pattern} pre={pre_mean!r} new={new_mean!r}"
    )


# ---------------------------------------------------------------------------
# Property-based: arbitrary random layer arrays at k=12.
# ---------------------------------------------------------------------------


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    layers=st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=66,  # K_12 has 66 edges
        max_size=66,
    )
)
def test_phase_b_property_based_k12(layers: list[int]) -> None:
    """Property-based: any binary layer assignment at k=12 produces
    bit-exact per-edge counts matching the pre-M2 implementation.
    """
    arr = np.asarray(layers, dtype=float)
    graph_a = make_graph(k=12, output_dir=None)
    graph_a.build_crossings_index()
    graph_b = make_graph(k=12, output_dir=None)
    graph_b.build_crossings_index()

    pre_intra, pre_inter = _pre_m2_per_edge_counts(graph_a, arr)
    new_intra, new_inter = _phase_b_per_edge_counts(graph_b, arr)

    for edge in graph_b._edge_list:
        assert pre_intra.get(edge, 0) == new_intra[edge]
        assert pre_inter.get(edge, 0) == new_inter[edge]


# ---------------------------------------------------------------------------
# §3-2 / M2 orchestration: ``initial_crossing_count_ms`` + ``loss_analysis_ms``
# show up as distinct phase keys in run_report.json.
# ---------------------------------------------------------------------------


def test_run_report_contains_m2_phase_timings(tmp_path: Path) -> None:
    """``run_optimization(collect_statistics=True)`` must emit both the
    Phase A timer (``initial_crossing_count_ms``) and the analyze-loss
    compute timer (``loss_analysis_ms``), separate from ``json_write_ms``.
    """
    result = run_optimization(
        k=12,
        optimizer="dual_annealing",
        maxiter=5,
        seed=42,
        output_dir=str(tmp_path),
        plot=False,
        run_loss_analysis=False,
        save_json=True,
        collect_statistics=True,
    )
    report_path = Path(result["report_path"])
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    timings = report["timings"]
    assert "initial_crossing_count_ms" in timings, (
        f"M2 Phase A timing missing from run_report; got keys: {sorted(timings)}"
    )
    assert "loss_analysis_ms" in timings, (
        f"M2 loss_analysis split missing from run_report; got keys: {sorted(timings)}"
    )
    assert "json_write_ms" in timings
    assert timings["initial_crossing_count_ms"] >= 0
    assert timings["loss_analysis_ms"] >= 0


# ---------------------------------------------------------------------------
# Sanity: build_crossings_index is idempotent.
# ---------------------------------------------------------------------------


def test_build_crossings_index_is_idempotent() -> None:
    graph = make_graph(k=12, output_dir=None)
    graph.build_crossings_index()
    pairs_1 = graph._crossing_pairs.copy()
    graph.build_crossings_index()
    pairs_2 = graph._crossing_pairs
    np.testing.assert_array_equal(pairs_1, pairs_2)
