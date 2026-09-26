"""Geometric helpers that produce node-position dicts for SiN graphs.

Besides the shape generators, :func:`load_positions_json` reads arbitrary
node coordinates from a JSON file (the C++ ``SINIC_POSITIONS_JSON``
format) and :func:`perimeter_is_simple` checks that their index order
walks the boundary, which the perimeter-ring pin assumes.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path


def _require_int(name: str, value: object, minimum: int) -> int:
    """Validate ``value`` is a real ``int`` (not ``bool``) >= ``minimum``.

    Raises ``ValueError`` (never ``TypeError``) so every generator reports
    the same documented failure mode. ``bool`` is rejected explicitly
    because it is an ``int`` subclass and silently accepting ``True`` as
    1 would contradict the "integer >= N" contract.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}; got {value!r}."
        )
    return int(value)


def distribute_nodes_around_square(nodes_per_side: int, side_length: float = 4) -> dict:
    """Place ``nodes_per_side`` nodes on each side of a square (corners excluded)."""
    positions = {}
    idx = 0
    spacing = side_length / (nodes_per_side + 1)

    for i in range(1, nodes_per_side + 1):
        positions[idx] = (spacing * i, side_length)
        idx += 1
    for i in range(1, nodes_per_side + 1):
        positions[idx] = (side_length, side_length - spacing * i)
        idx += 1
    for i in range(1, nodes_per_side + 1):
        positions[idx] = (side_length - spacing * i, 0)
        idx += 1
    for i in range(1, nodes_per_side + 1):
        positions[idx] = (0, spacing * i)
        idx += 1

    return positions


def distribute_nodes_around_rectangle(
    nodes_on_length: int,
    nodes_on_width: int,
    length: float = 5,
    width: float = 3,
) -> dict:
    """Place nodes around a rectangle — long sides get ``nodes_on_length`` each."""
    positions = {}
    idx = 0
    spacing_length = length / (nodes_on_length + 1)
    spacing_width = width / (nodes_on_width + 1)

    for i in range(1, nodes_on_length + 1):
        positions[idx] = (spacing_length * i, width)
        idx += 1
    for i in range(1, nodes_on_width + 1):
        positions[idx] = (length, width - spacing_width * i)
        idx += 1
    for i in range(1, nodes_on_length + 1):
        positions[idx] = (length - spacing_length * i, 0)
        idx += 1
    for i in range(1, nodes_on_width + 1):
        positions[idx] = (0, spacing_width * i)
        idx += 1

    return positions


def distribute_nodes_around_circle(
    k: int, radius: float = 1.0, start_angle: float = 0.0
) -> dict:
    """Place ``k`` nodes equally spaced on a circle (counter-clockwise).

    Node ``i`` sits at angle ``start_angle + 2*pi*i/k``. Index order is the
    boundary order, so the perimeter-pin rule stays valid.
    """
    k = _require_int("k", k, 3)
    positions = {}
    for i in range(k):
        theta = start_angle + 2.0 * math.pi * i / k
        positions[i] = (radius * math.cos(theta), radius * math.sin(theta))
    return positions


def distribute_nodes_around_triangle(
    nodes_per_side, side_length: float = 1.0
) -> dict:
    """Place nodes along the 3 edges of an equilateral triangle.

    ``nodes_per_side`` is an int (same count per edge) or a length-3
    sequence ``[n0, n1, n2]`` (corners excluded). Index order walks
    edge 0 (V0->V1), edge 1 (V1->V2), edge 2 (V2->V0).
    """
    if isinstance(nodes_per_side, int):
        counts = [nodes_per_side, nodes_per_side, nodes_per_side]
    else:
        counts = list(nodes_per_side)
        if len(counts) != 3:
            raise ValueError(
                f"triangle nodes_per_side must be an int or a length-3 "
                f"sequence; got {nodes_per_side!r}."
            )
    counts = [_require_int("triangle per-side count", c, 1) for c in counts]
    length = float(side_length)
    v0 = (0.0, 0.0)
    v1 = (length, 0.0)
    v2 = (length / 2.0, length * math.sqrt(3.0) / 2.0)
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    positions = {}
    idx = 0
    for (va, vb), n in zip(edges, counts, strict=True):
        for j in range(1, n + 1):
            t = j / (n + 1)
            positions[idx] = (
                va[0] + (vb[0] - va[0]) * t,
                va[1] + (vb[1] - va[1]) * t,
            )
            idx += 1
    return positions


def distribute_nodes_around_polygon(
    n_sides: int, nodes_per_side: int, circumradius: float = 1.0
) -> dict:
    """Place ``nodes_per_side`` nodes on each edge of a regular convex
    polygon with ``n_sides`` (corners excluded).

    Vertices sit on a circle of ``circumradius`` starting at the top
    (angle pi/2), counter-clockwise. Index order walks the perimeter.
    ``k = n_sides * nodes_per_side``.
    """
    n_sides = _require_int("n_sides", n_sides, 3)
    nodes_per_side = _require_int("nodes_per_side", nodes_per_side, 1)
    r = float(circumradius)
    verts = []
    for v in range(n_sides):
        ang = math.pi / 2.0 + 2.0 * math.pi * v / n_sides
        verts.append((r * math.cos(ang), r * math.sin(ang)))
    positions = {}
    idx = 0
    for v in range(n_sides):
        va = verts[v]
        vb = verts[(v + 1) % n_sides]
        for j in range(1, nodes_per_side + 1):
            t = j / (nodes_per_side + 1)
            positions[idx] = (
                va[0] + (vb[0] - va[0]) * t,
                va[1] + (vb[1] - va[1]) * t,
            )
            idx += 1
    return positions


_PARTIAL_RECT_SIDES = ("top", "right", "bottom", "left")


def distribute_nodes_around_partial_rectangle(
    side_counts: dict, length: float = 5.0, width: float = 3.0
) -> dict:
    """Place nodes on 2 or 3 sides of a rectangle perimeter (corners
    excluded), arbitrary count per chosen side.

    ``side_counts`` keys are a 2- or 3-element subset of
    ``{"top", "right", "bottom", "left"}``, values are ints >= 1. Nodes
    are emitted in the fixed perimeter order top -> right -> bottom ->
    left (absent sides skipped), so index order follows the boundary
    path and the perimeter-pin rule stays valid.

    Side parametrisations (t = j/(n+1), j = 1..n):
      top:    y = width,  x = length * t
      right:  x = length,  y = width - width * t
      bottom: y = 0,       x = length - length * t
      left:   x = 0,       y = width * t
    """
    if not isinstance(side_counts, dict):
        raise ValueError(
            f"side_counts must be a dict; got {type(side_counts).__name__}."
        )
    keys = list(side_counts)
    if len(keys) not in (2, 3):
        raise ValueError(
            f"partial_rectangle needs 2 or 3 sides; got {len(keys)}: {keys}."
        )
    unknown = [s for s in keys if s not in _PARTIAL_RECT_SIDES]
    if unknown:
        raise ValueError(
            f"partial_rectangle unknown side(s) {unknown}; "
            f"valid: {list(_PARTIAL_RECT_SIDES)}."
        )
    for side, count in side_counts.items():
        _require_int(f"partial_rectangle side '{side}' count", count, 1)
    length_f = float(length)
    width_f = float(width)
    positions = {}
    idx = 0
    for side in _PARTIAL_RECT_SIDES:  # fixed perimeter traversal order
        if side not in side_counts:
            continue
        n = side_counts[side]
        for j in range(1, n + 1):
            t = j / (n + 1)
            if side == "top":
                point = (length_f * t, width_f)
            elif side == "right":
                point = (length_f, width_f - width_f * t)
            elif side == "bottom":
                point = (length_f - length_f * t, 0.0)
            else:  # left
                point = (0.0, width_f * t)
            positions[idx] = point
            idx += 1
    return positions


_KNOWN_SHAPES = (
    "square",
    "rectangle",
    "circle",
    "triangle",
    "polygon",
    "partial_rectangle",
)


def distribute_nodes(
    shape: str = "square",
    nodes_per_side: int = 3,
    side_length: float = 4,
    nodes_on_length: int = 3,
    nodes_on_width: int = 3,
    length: float = 5,
    width: float = 3,
    *,
    k: int | None = None,
    radius: float = 1.0,
    start_angle: float = 0.0,
    triangle_nodes_per_side: int | list[int] | None = None,
    triangle_side_length: float = 1.0,
    polygon_n_sides: int | None = None,
    polygon_nodes_per_side: int | None = None,
    polygon_circumradius: float = 1.0,
    side_counts: dict | None = None,
) -> dict:
    """Dispatch to the shape-specific generator.

    The original ``square`` / ``rectangle`` positional contract is kept
    intact (existing callers unaffected). New shapes take keyword-only
    parameters.
    """
    if shape == "square":
        return distribute_nodes_around_square(nodes_per_side, side_length)
    if shape == "rectangle":
        return distribute_nodes_around_rectangle(
            nodes_on_length, nodes_on_width, length, width
        )
    if shape == "circle":
        if k is None:
            raise ValueError(
                "shape='circle' requires k= (number of nodes)."
            )
        return distribute_nodes_around_circle(k, radius, start_angle)
    if shape == "triangle":
        if triangle_nodes_per_side is None:
            raise ValueError(
                "shape='triangle' requires triangle_nodes_per_side= "
                "(an int or a length-3 list [n0, n1, n2])."
            )
        return distribute_nodes_around_triangle(
            triangle_nodes_per_side, triangle_side_length
        )
    if shape == "polygon":
        if polygon_n_sides is None or polygon_nodes_per_side is None:
            raise ValueError(
                "shape='polygon' requires polygon_n_sides= and "
                "polygon_nodes_per_side= (both ints)."
            )
        return distribute_nodes_around_polygon(
            polygon_n_sides, polygon_nodes_per_side, polygon_circumradius
        )
    if shape == "partial_rectangle":
        if side_counts is None:
            raise ValueError(
                "shape='partial_rectangle' requires side_counts= "
                "(a dict of 2-3 sides)."
            )
        return distribute_nodes_around_partial_rectangle(
            side_counts, length, width
        )
    raise ValueError(
        "The 'shape' parameter must be one of: " + ", ".join(_KNOWN_SHAPES)
    )


# Pre-computed example positions kept for backwards-compatible imports.
POSITIONS_12_NODES = {
    0: (1, 4), 1: (2, 4), 2: (3, 4), 3: (4, 3),
    4: (4, 2), 5: (4, 1), 6: (3, 0), 7: (2, 0),
    8: (1, 0), 9: (0, 1), 10: (0, 2), 11: (0, 3),
}


_CANONICAL_KEY = re.compile(r"0|[1-9][0-9]*")


def _is_coordinate(value: object) -> bool:
    """A finite JSON number (not ``true`` / ``false``, not an integer too
    large for a float)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def load_positions_json(path: str | Path) -> dict[int, tuple[float, float]]:
    """Read node coordinates ``{"<node>": [x, y], ...}`` from a JSON file.

    Same contract as the C++ loader behind ``SINIC_POSITIONS_JSON``: keys
    are canonical non-negative integers (``"0"``, ``"12"``; no sign,
    space or leading zero) forming the contiguous set ``0 .. k-1``, and
    each value is a two-number array. The index order must walk the
    boundary (see :func:`perimeter_is_simple`). A saved
    ``subgraphsdata.json`` or ``optimization_history.json`` is accepted
    too: its ``positions`` block is used, so a run's layout can be reused.

    Returns ``{node: (x, y)}`` ordered by node; ``k`` is its length.
    Raises ``ValueError`` naming the file on any violation.
    """
    path = Path(path)
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("positions"), dict):
        payload = payload["positions"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f'{path}: expected a non-empty object {{"<node>": [x, y], ...}}.'
        )
    positions: dict[int, tuple[float, float]] = {}
    for key, value in payload.items():
        if not _CANONICAL_KEY.fullmatch(key):
            raise ValueError(
                f"{path}: node key {key!r} must be a canonical non-negative "
                f'integer ("0" or [1-9][0-9]*; no sign, space or leading zero).'
            )
        if not (isinstance(value, list) and len(value) == 2
                and all(_is_coordinate(c) for c in value)):
            raise ValueError(
                f"{path}: node {key} must map to a two-number array [x, y]; got {value!r}."
            )
        positions[int(key)] = (float(value[0]), float(value[1]))
    k = len(positions)
    missing = sorted(set(range(k)) - set(positions))
    if missing:
        raise ValueError(
            f"{path}: node keys must be the contiguous set 0..{k - 1} (missing {missing[0]})."
        )
    return {n: positions[n] for n in range(k)}


def perimeter_is_simple(positions: dict) -> bool:
    """True when walking nodes ``0, 1, ..., k-1`` and back to ``0`` traces
    a closed boundary that never crosses itself.

    The optimizer pins the perimeter ring (edges ``i -> i+1`` and
    ``k-1 -> 0``) to one layer, so node indices have to follow the
    boundary; a crossing ring means they do not (e.g. the nodes of a
    rectangle numbered row by row instead of around it).
    """
    from shapely.geometry import LinearRing

    k = len(positions)
    if k < 3:
        return False
    return bool(LinearRing([positions[i] for i in range(k)]).is_simple)
