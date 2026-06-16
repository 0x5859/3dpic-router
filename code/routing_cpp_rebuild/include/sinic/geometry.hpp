#pragma once

// 2D segment intersection primitives. Extracted verbatim from the
// pre-M5 main.cpp file-local statics (`cross`, `isPointOnSegment`,
// `segmentsIntersect`, `doSegmentsIntersectStrict`). Behavior is
// preserved bit-for-bit so that crossing counts stay identical to
// the legacy executable; M6 landed the cached-topology path
// (REFACTOR_GOALS.md §2-1, `crossings_cache.hpp`) but this primitive
// is still used as the geometric oracle for the fallback path and
// the test-oracle comparisons.

#include "sinic/types.hpp"

namespace sinic::geo {

// Signed 2× triangle area / 2D scalar cross product.
double cross_product(const Point& a, const Point& b, const Point& c) noexcept;

// True iff `point` lies on the closed segment [segStart, segEnd].
bool is_point_on_segment(const Point& seg_start,
                         const Point& seg_end,
                         const Point& point) noexcept;

// Generic segment intersection (returns true if the closed segments
// share any point, including endpoint-touching cases).
bool segments_intersect(const Point& a, const Point& b,
                        const Point& c, const Point& d) noexcept;

// "Strict" intersection: shared endpoints and edges lying entirely on
// each other are treated as NOT intersecting. This matches the Python
// `core.SiNInterconnectionGraph.edge_crosses` semantics that the M2
// cached-topology oracle relies on.
bool segments_intersect_strict(const Point& a, const Point& b,
                               const Point& c, const Point& d) noexcept;

} // namespace sinic::geo
