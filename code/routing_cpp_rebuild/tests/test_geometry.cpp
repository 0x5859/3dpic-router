#include <catch2/catch.hpp>

#include "sinic/geometry.hpp"

using sinic::Point;
using namespace sinic::geo;

TEST_CASE("cross_product signs", "[geometry]") {
    REQUIRE(cross_product({0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}) > 0);
    REQUIRE(cross_product({0.0, 0.0}, {1.0, 0.0}, {1.0, -1.0}) < 0);
    REQUIRE(cross_product({0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}) == Approx(0.0));
}

TEST_CASE("is_point_on_segment", "[geometry]") {
    REQUIRE(is_point_on_segment({0.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}));
    REQUIRE_FALSE(is_point_on_segment({0.0, 0.0}, {2.0, 0.0}, {1.0, 0.5}));
    REQUIRE(is_point_on_segment({0.0, 0.0}, {2.0, 2.0}, {1.0, 1.0}));
    REQUIRE_FALSE(is_point_on_segment({0.0, 0.0}, {2.0, 2.0}, {3.0, 3.0}));
}

TEST_CASE("segments_intersect_strict basics", "[geometry]") {
    // Two crossing segments forming an X.
    REQUIRE(segments_intersect_strict({0,0}, {2,2}, {0,2}, {2,0}));
    // Two parallel non-intersecting segments.
    REQUIRE_FALSE(segments_intersect_strict({0,0}, {2,0}, {0,1}, {2,1}));
    // T-junction: one segment's endpoint lies on the INTERIOR of the
    // other segment (no shared endpoint between the two endpoint
    // *pairs*). Pre-M5 `doSegmentsIntersectStrict` returns TRUE for
    // this case — `hasSharedEndpoint` only excludes touching endpoint
    // pairs, and the cross-product fall-through path reaches
    // `isPointOnSegment(a, b, c)`. Preserved for C++/Python parity
    // (Python `edge_crosses` is shapely-based and gives the same
    // answer for this configuration).
    REQUIRE(segments_intersect_strict({0,0}, {2,0}, {1,0}, {1,1}));
    // Shared endpoint.
    REQUIRE_FALSE(segments_intersect_strict({0,0}, {1,1}, {1,1}, {2,2}));
    // One segment lies entirely on the other (overlap).
    REQUIRE_FALSE(segments_intersect_strict({0,0}, {4,0}, {1,0}, {3,0}));
}
