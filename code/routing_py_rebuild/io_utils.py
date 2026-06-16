"""Binary file readers and point-set transforms used by overlay plotting."""
from __future__ import annotations

import struct


def read_and_interpret_b16_file(filename: str, points_per_set: int) -> list[list[int]]:
    """Read a *.b16 file of little-endian 16-bit unsigned ints into chunks.

    Each chunk is ``2 * points_per_set`` integers — i.e. ``points_per_set``
    (x, y) pairs.
    """
    point_sets: list[list[int]] = []
    current_point_set: list[int] = []
    bytes_per_set = 2 * points_per_set * 2

    with open(filename, "rb") as f:
        byte_count = 0
        while True:
            bytes_read = f.read(2)
            if not bytes_read:
                if current_point_set:
                    point_sets.append(current_point_set)
                break
            coordinate = struct.unpack("<H", bytes_read)[0]
            current_point_set.append(coordinate)
            byte_count += 2
            if byte_count == bytes_per_set:
                point_sets.append(current_point_set)
                current_point_set = []
                byte_count = 0
    return point_sets


def transform_point_sets(point_sets: list[list[int]], k: int) -> list[dict]:
    """Convert raw int chunks into a list of ``{node_index: (x, y)}`` dicts."""
    return [
        {i: (point_set[2 * i], point_set[2 * i + 1]) for i in range(k)}
        for point_set in point_sets
    ]


def print_point_sets(point_sets: list[list[int]], points_per_set: int) -> None:
    for index, point_set in enumerate(point_sets):
        print(f"Point Set {index}:")
        for i in range(points_per_set):
            x, y = point_set[2 * i], point_set[2 * i + 1]
            print(f"  Point {i}: ({x}, {y})")
