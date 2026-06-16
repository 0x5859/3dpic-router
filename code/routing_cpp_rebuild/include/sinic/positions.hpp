#pragma once

// Node-placement generators. The legacy main.cpp had one function:
// `distribute_nodes_around_square`; M6 (REFACTOR_GOALS.md §2-1 implementation
// summary) adds `distribute_nodes_around_rectangle` to mirror Python
// `routing_py_rebuild/positions.py`. Both are free functions so the
// orchestration in main.cpp does not depend on the graph class.

#include <string>

#include "sinic/types.hpp"

namespace sinic {

// Place nodes_per_side nodes on each side of an axis-aligned square
// (corners excluded). Returns a positions map indexed 0..(4*nodes_per_side-1)
// with the same ordering as legacy main.cpp. The traversal order
// (top → right → bottom → left) produces a cyclic-convex node layout,
// which the M6 Phase A fast path detects to skip geometric crossing
// tests entirely (see `crossings_cache.hpp`).
PositionMap distribute_nodes_around_square(int nodes_per_side,
                                           double side_length = 4.0);

// Place nodes around an axis-aligned rectangle (corners excluded). Same
// cyclic node ordering as the square version (long-side top → short-side
// right → long-side bottom → short-side left) so the M6 Phase A fast
// path still applies. Mirrors
// `routing_py_rebuild/positions.py::distribute_nodes_around_rectangle`.
PositionMap distribute_nodes_around_rectangle(int nodes_on_length,
                                              int nodes_on_width,
                                              double length = 5.0,
                                              double width  = 3.0);

// C1 (REFACTOR_GOALS.md §1-2): load a node-position map from a JSON
// object `{"<node>": [x, y], ...}` (the `subgraphsdata.json` positions
// shape, §3-1). Keys must be the dense set 0..k-1; values 2-number
// arrays. Used by the experiment harness to feed byte-identical
// positions to both backends (spec D6). Throws std::runtime_error on
// any malformed input (missing file / not an object / bad key / bad
// value / non-contiguous keys / empty).
PositionMap load_positions_from_json(const std::string& path);

} // namespace sinic
