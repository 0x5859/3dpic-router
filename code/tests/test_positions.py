"""Unit tests for the M-A boundary position generators (spec §6).

Purely geometric: counts, coordinates, boundary ordering, validation.
No core.py / crossing interaction (that is M-D, spec §7/§14).
"""
from __future__ import annotations

import json
import math

import pytest
from routing_py_rebuild.positions import (
    distribute_nodes,
    distribute_nodes_around_circle,
    distribute_nodes_around_partial_rectangle,
    distribute_nodes_around_polygon,
    distribute_nodes_around_triangle,
    load_positions_json,
    perimeter_is_simple,
)


def test_circle_count_and_keys():
    pos = distribute_nodes_around_circle(8, radius=2.0)
    assert len(pos) == 8
    assert sorted(pos) == list(range(8))  # contiguous 0..k-1


def test_circle_coordinates_k4_unit():
    pos = distribute_nodes_around_circle(4, radius=1.0, start_angle=0.0)
    assert pos[0] == pytest.approx((1.0, 0.0), abs=1e-12)
    assert pos[1] == pytest.approx((0.0, 1.0), abs=1e-12)
    assert pos[2] == pytest.approx((-1.0, 0.0), abs=1e-12)
    assert pos[3] == pytest.approx((0.0, -1.0), abs=1e-12)


def test_circle_all_points_on_radius():
    pos = distribute_nodes_around_circle(13, radius=3.5)
    for x, y in pos.values():
        assert math.hypot(x, y) == pytest.approx(3.5, abs=1e-12)


def test_circle_rejects_k_below_3():
    with pytest.raises(ValueError, match=r"k must be an integer >= 3"):
        distribute_nodes_around_circle(2)


def test_triangle_int_count_three_sides():
    pos = distribute_nodes_around_triangle(1, side_length=1.0)
    assert len(pos) == 3
    assert sorted(pos) == [0, 1, 2]
    # edge 0 = V0(0,0) -> V1(1,0): single interior node at t=1/2
    assert pos[0] == pytest.approx((0.5, 0.0), abs=1e-12)
    # edge 1 = V1(1,0) -> V2(0.5, sqrt3/2): midpoint
    assert pos[1] == pytest.approx((0.75, math.sqrt(3) / 4.0), abs=1e-12)
    # edge 2 = V2 -> V0(0,0): midpoint
    assert pos[2] == pytest.approx((0.25, math.sqrt(3) / 4.0), abs=1e-12)


def test_triangle_per_side_list_counts():
    pos = distribute_nodes_around_triangle([2, 1, 1])
    assert len(pos) == 4  # 2 + 1 + 1


def test_triangle_rejects_bad_length_list():
    with pytest.raises(ValueError, match="length-3 sequence"):
        distribute_nodes_around_triangle([1, 2])


def test_triangle_rejects_nonpositive_count():
    with pytest.raises(ValueError, match=">= 1"):
        distribute_nodes_around_triangle(0)


def test_polygon_count():
    pos = distribute_nodes_around_polygon(5, 3, circumradius=2.0)
    assert len(pos) == 15  # 5 sides * 3 per side
    assert sorted(pos) == list(range(15))


def test_polygon_square_vertices_via_midpoints():
    # n_sides=4, 1 node/side, R=1: vertices at 90/180/270/360 deg.
    pos = distribute_nodes_around_polygon(4, 1, circumradius=1.0)
    assert len(pos) == 4
    # edge 0 midpoint between (0,1) and (-1,0)
    assert pos[0] == pytest.approx((-0.5, 0.5), abs=1e-12)


def test_polygon_all_within_circumradius():
    pos = distribute_nodes_around_polygon(6, 2, circumradius=4.0)
    for x, y in pos.values():
        assert math.hypot(x, y) <= 4.0 + 1e-9


def test_polygon_rejects_few_sides():
    with pytest.raises(ValueError, match=r"n_sides must be an integer >= 3"):
        distribute_nodes_around_polygon(2, 1)


def test_polygon_rejects_nonpositive_nodes():
    with pytest.raises(ValueError, match=r"nodes_per_side must be an integer >= 1"):
        distribute_nodes_around_polygon(5, 0)


def test_partial_rect_two_sides_order_and_coords():
    pos = distribute_nodes_around_partial_rectangle(
        {"top": 2, "left": 1}, length=4.0, width=2.0
    )
    assert len(pos) == 3
    assert sorted(pos) == [0, 1, 2]
    # Fixed emit order top -> right -> bottom -> left, skipping absent.
    # top: y=W, x = L*t for t in {1/3, 2/3}
    assert pos[0] == pytest.approx((4.0 / 3.0, 2.0), abs=1e-12)
    assert pos[1] == pytest.approx((8.0 / 3.0, 2.0), abs=1e-12)
    # left: x=0, y = W*t for t=1/2
    assert pos[2] == pytest.approx((0.0, 1.0), abs=1e-12)


def test_partial_rect_three_sides_count():
    pos = distribute_nodes_around_partial_rectangle(
        {"top": 3, "right": 2, "bottom": 1}
    )
    assert len(pos) == 6


def test_partial_rect_rejects_one_side():
    with pytest.raises(ValueError, match="2 or 3 sides"):
        distribute_nodes_around_partial_rectangle({"top": 4})


def test_partial_rect_rejects_unknown_side():
    with pytest.raises(ValueError, match="unknown side"):
        distribute_nodes_around_partial_rectangle({"top": 1, "diag": 2})


def test_partial_rect_rejects_nonpositive_count():
    with pytest.raises(ValueError, match=">= 1"):
        distribute_nodes_around_partial_rectangle({"top": 1, "left": 0})


def test_dispatch_existing_square_unchanged():
    # Backwards-compat: existing api.make_graph call shape.
    direct = distribute_nodes(shape="square", nodes_per_side=3, side_length=1)
    assert len(direct) == 12


def test_dispatch_circle():
    pos = distribute_nodes(shape="circle", k=10, radius=2.0)
    assert len(pos) == 10


def test_dispatch_triangle():
    pos = distribute_nodes(shape="triangle", triangle_nodes_per_side=4)
    assert len(pos) == 12


def test_dispatch_polygon():
    pos = distribute_nodes(
        shape="polygon", polygon_n_sides=6, polygon_nodes_per_side=2
    )
    assert len(pos) == 12


def test_dispatch_partial_rectangle():
    pos = distribute_nodes(
        shape="partial_rectangle", side_counts={"top": 3, "bottom": 3}
    )
    assert len(pos) == 6


def test_dispatch_circle_requires_k():
    with pytest.raises(ValueError, match="requires k="):
        distribute_nodes(shape="circle")


def test_dispatch_unknown_shape_raises():
    with pytest.raises(ValueError, match="must be one of"):
        distribute_nodes(shape="hexagram")


def test_dispatch_rectangle():
    pos = distribute_nodes(
        shape="rectangle", nodes_on_length=3, nodes_on_width=2
    )
    assert len(pos) == 10  # 2 * (3 + 2)


def test_dispatch_triangle_requires_param():
    with pytest.raises(ValueError, match="requires triangle_nodes_per_side="):
        distribute_nodes(shape="triangle")


def test_dispatch_polygon_requires_params():
    with pytest.raises(ValueError, match="requires polygon_n_sides="):
        distribute_nodes(shape="polygon", polygon_n_sides=5)


def test_dispatch_partial_rectangle_requires_side_counts():
    with pytest.raises(ValueError, match="requires side_counts="):
        distribute_nodes(shape="partial_rectangle")


def test_generators_are_public_api():
    import routing_py_rebuild as rpr

    for name in (
        "distribute_nodes_around_circle",
        "distribute_nodes_around_triangle",
        "distribute_nodes_around_polygon",
        "distribute_nodes_around_partial_rectangle",
    ):
        assert hasattr(rpr, name), f"{name} not re-exported"
        assert name in rpr.__all__, f"{name} missing from __all__"


def test_triangle_list_path_coordinates():
    pos = distribute_nodes_around_triangle([1, 2, 1], side_length=1.0)
    assert len(pos) == 4
    assert sorted(pos) == [0, 1, 2, 3]
    s3 = math.sqrt(3)
    # edge0 V0(0,0)->V1(1,0): 1 node, t=1/2
    assert pos[0] == pytest.approx((0.5, 0.0), abs=1e-12)
    # edge1 V1(1,0)->V2(0.5, s3/2): 2 nodes, t=1/3, 2/3
    assert pos[1] == pytest.approx(
        (1.0 - 0.5 / 3.0, (s3 / 2.0) / 3.0), abs=1e-12
    )
    assert pos[2] == pytest.approx(
        (1.0 - 1.0 / 3.0, (s3 / 2.0) * 2.0 / 3.0), abs=1e-12
    )
    # edge2 V2->V0(0,0): 1 node, t=1/2
    assert pos[3] == pytest.approx((0.25, s3 / 4.0), abs=1e-12)


def test_triangle_side_length_scales_linearly():
    base = distribute_nodes_around_triangle(2, side_length=1.0)
    scaled = distribute_nodes_around_triangle(2, side_length=3.0)
    for i in base:
        assert scaled[i] == pytest.approx(
            (base[i][0] * 3.0, base[i][1] * 3.0), abs=1e-12
        )


def test_polygon_circumradius_scales_linearly():
    base = distribute_nodes_around_polygon(5, 2, circumradius=1.0)
    scaled = distribute_nodes_around_polygon(5, 2, circumradius=4.0)
    for i in base:
        assert scaled[i] == pytest.approx(
            (base[i][0] * 4.0, base[i][1] * 4.0), abs=1e-12
        )


def test_partial_rect_rejects_four_sides():
    with pytest.raises(ValueError, match="2 or 3 sides"):
        distribute_nodes_around_partial_rectangle(
            {"top": 1, "right": 1, "bottom": 1, "left": 1}
        )


def test_count_validation_rejects_bool_and_non_int():
    # bool is an int subclass — must be rejected, not silently treated as 1
    with pytest.raises(ValueError, match="must be an integer"):
        distribute_nodes_around_triangle(True)
    with pytest.raises(ValueError, match="must be an integer"):
        distribute_nodes_around_polygon(4, True)
    with pytest.raises(ValueError, match="must be an integer"):
        distribute_nodes_around_partial_rectangle({"top": True, "left": 1})
    with pytest.raises(ValueError, match="must be an integer"):
        distribute_nodes_around_circle(True)
    # non-int -> ValueError (not TypeError)
    with pytest.raises(ValueError, match="must be an integer"):
        distribute_nodes_around_polygon(4, "x")


# ---------------------------------------------------------------------------
# Arbitrary coordinates from JSON (the C++ SINIC_POSITIONS_JSON format)
# ---------------------------------------------------------------------------
def _write(tmp_path, payload) -> str:
    path = tmp_path / "positions.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_load_positions_json_round_trips_any_layout(tmp_path):
    pos = distribute_nodes("triangle", triangle_nodes_per_side=[3, 2, 4])
    # written in shuffled key order: the loader returns node order 0..k-1
    payload = {str(n): list(pos[n]) for n in sorted(pos, reverse=True)}
    loaded = load_positions_json(_write(tmp_path, payload))
    assert list(loaded) == list(range(len(pos)))
    assert loaded == {n: (float(x), float(y)) for n, (x, y) in pos.items()}


def test_load_positions_json_takes_a_saved_runs_positions_block(tmp_path):
    saved = {"General Parameters": {"k": 4}, "Layer_0": {"edges": []},
             "positions": {"0": [0, 0], "1": [2, 0], "2": [2, 1], "3": [0, 1]}}
    assert load_positions_json(_write(tmp_path, saved))[2] == (2.0, 1.0)


@pytest.mark.parametrize("payload, match", [
    ([[0, 0], [1, 0]], "non-empty object"),
    ({}, "non-empty object"),
    ({"0": [0, 0], "01": [1, 0]}, "canonical"),
    ({"0": [0, 0], "-1": [1, 0]}, "canonical"),
    ({"0": [0, 0], "2": [1, 0]}, "contiguous set 0..1"),
    ({"0": [0, 0], "1": [1]}, "two-number array"),
    ({"0": [0, 0], "1": [1, "2"]}, "two-number array"),
    ({"0": [0, 0], "1": [1, True]}, "two-number array"),
])
def test_load_positions_json_rejects_malformed_files(tmp_path, payload, match):
    with pytest.raises(ValueError, match=match):
        load_positions_json(_write(tmp_path, payload))


def test_load_positions_json_rejects_non_finite_and_bad_json(tmp_path):
    path = tmp_path / "positions.json"
    path.write_text('{"0": [0, 0], "1": [NaN, 1]}')
    with pytest.raises(ValueError, match="two-number array"):
        load_positions_json(path)
    path.write_text('{"0": [0, 0], "1": [1' + "0" * 400 + ', 1]}')  # overflows a float
    with pytest.raises(ValueError, match="two-number array"):
        load_positions_json(path)
    path.write_text('{"0": [0, 0],')
    with pytest.raises(ValueError, match="malformed JSON"):
        load_positions_json(path)


@pytest.mark.parametrize("pos", [
    distribute_nodes("square", nodes_per_side=3),
    distribute_nodes("rectangle", nodes_on_length=4, nodes_on_width=2),
    distribute_nodes("circle", k=9),
    distribute_nodes("triangle", triangle_nodes_per_side=4),
    distribute_nodes("polygon", polygon_n_sides=6, polygon_nodes_per_side=2),
    distribute_nodes("partial_rectangle", side_counts={"top": 3, "right": 2, "bottom": 3}),
], ids=["square", "rectangle", "circle", "triangle", "hexagon", "partial_rectangle"])
def test_generators_number_nodes_along_the_boundary(pos):
    assert perimeter_is_simple(pos)


def test_perimeter_is_simple_flags_a_crossing_node_order():
    # A 2 x 3 grid of ports numbered row by row, not around the outline.
    rows = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (0, 1), 4: (1, 1), 5: (2, 1)}
    assert not perimeter_is_simple(rows)
    around = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (2, 1), 4: (1, 1), 5: (0, 1)}
    assert perimeter_is_simple(around)
    assert not perimeter_is_simple({0: (0, 0), 1: (1, 0)})
