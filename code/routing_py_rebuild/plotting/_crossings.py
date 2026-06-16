"""Standalone edge-crossing detection for plotting.

Mirrors :meth:`core.SiNInterconnectionGraph.edge_crosses` but takes
``(edges, positions)`` directly so plotting code can compute crossings
without holding a graph instance — used by the ``minimize.b16`` overlay
row in :mod:`plotting.layers`, which needs crossings against an
alternative position set.
"""
from __future__ import annotations

import shapely
from shapely.geometry import LineString, Point


def edge_crosses(e1: tuple, e2: tuple, positions: dict) -> bool:
    """True if ``e1`` and ``e2`` cross properly (shared endpoints don't count)."""
    line1 = LineString([positions[e1[0]], positions[e1[1]]])
    line2 = LineString([positions[e2[0]], positions[e2[1]]])

    bool_result = shapely.intersects(line1, line2)
    e1_pts = [positions[e1[0]], positions[e1[1]]]
    e2_pts = [positions[e2[0]], positions[e2[1]]]

    for pt in e1_pts:
        if pt in e2_pts:
            bool_result = False
    if shapely.intersects(line1, Point(*e2_pts[0])) and shapely.intersects(
        line1, Point(*e2_pts[1])
    ):
        bool_result = False
    if shapely.intersects(line2, Point(*e1_pts[0])) and shapely.intersects(
        line2, Point(*e1_pts[1])
    ):
        bool_result = False
    return bool_result


def count_crossings_with_detail(edges, positions: dict) -> tuple[int, dict]:
    """Count crossings among ``edges`` and return per-edge counts."""
    edges = list(edges)
    num_crosses = 0
    edge_cross_count = {edge: 0 for edge in edges}
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if edge_crosses(edges[i], edges[j], positions):
                num_crosses += 1
                edge_cross_count[edges[i]] += 1
                edge_cross_count[edges[j]] += 1
    return num_crosses, edge_cross_count
