#include <catch2/catch.hpp>

#include "sinic/crossings.hpp"
#include "sinic/crossings_cache.hpp"
#include "sinic/positions.hpp"

#include <algorithm>
#include <cmath>
#include <set>

using sinic::CrossingTopology;
using sinic::EdgeList;
using sinic::EdgeData;
using sinic::PositionMap;
using sinic::count_crossings_with_detail;
using sinic::distribute_nodes_around_square;
using sinic::distribute_nodes_around_rectangle;

namespace {

// Brute-force oracle: build the K_k edge list, run the legacy
// O(E²) geometric scanner, and return the set of (i, j) crossing pairs
// expressed as indices into the same row-major edge ordering used by
// CrossingTopology. This is the ground truth the Phase A fast path
// must agree with.
std::set<std::pair<int, int>> oracle_pairs(int k, const PositionMap& pos) {
    EdgeList edges;
    for (int u = 0; u < k; ++u) {
        for (int v = u + 1; v < k; ++v) {
            edges.emplace_back(u, v, EdgeData{});
        }
    }
    std::set<std::pair<int, int>> out;
    const int n = static_cast<int>(edges.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (sinic::edge_crosses(edges[i], edges[j], pos)) {
                out.emplace(i, j);
            }
        }
    }
    return out;
}

PositionMap pentagram_positions() {
    // 5 vertices of a regular pentagon, but indexed in star-traversal
    // order (winding number 2). Cyclic-convex detector must reject.
    PositionMap pos;
    const double TAU = 2.0 * 3.14159265358979323846;
    for (int i = 0; i < 5; ++i) {
        const int idx = (i * 2) % 5;  // pentagram order
        const double a = -TAU / 4.0 + idx * (TAU / 5.0);
        pos[i] = {std::cos(a), std::sin(a)};
    }
    return pos;
}

// Equilateral triangle with `n` nodes per side (corners excluded), built
// like Python's distribute_nodes_around_triangle: the interpolated side
// nodes sit ~1e-17 off their side line, with either sign.
PositionMap triangle_positions(int n) {
    const double h = std::sqrt(3.0) / 2.0;
    const sinic::Point v[3] = {{0.0, 0.0}, {1.0, 0.0}, {0.5, h}};
    PositionMap pos;
    int idx = 0;
    for (int s = 0; s < 3; ++s) {
        const sinic::Point& a = v[s];
        const sinic::Point& b = v[(s + 1) % 3];
        for (int j = 1; j <= n; ++j) {
            const double t = static_cast<double>(j) / (n + 1);
            pos[idx++] = {a.first + (b.first - a.first) * t,
                          a.second + (b.second - a.second) * t};
        }
    }
    return pos;
}

} // namespace

TEST_CASE("CrossingTopology builds row-major edge_list and perimeter mask",
          "[cache]") {
    const int nps = 3;
    const int k = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    CrossingTopology topo(k, pos);

    REQUIRE(topo.num_edges() == k * (k - 1) / 2);
    // Row-major u<v ordering.
    const auto& edges = topo.edge_list();
    REQUIRE(static_cast<int>(edges.size()) == topo.num_edges());
    REQUIRE(edges.front() == std::make_pair(0, 1));
    for (const auto& e : edges) REQUIRE(e.first < e.second);

    // Perimeter ring should be every (u, v) with |u-v| in {1, k-1}.
    int perim_count = 0;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const int u = edges[i].first;
        const int v = edges[i].second;
        const int diff = std::abs(u - v);
        const bool expect = (diff == 1) || (diff == (k - 1));
        REQUIRE(topo.perimeter_mask()[i] == expect);
        if (expect) ++perim_count;
    }
    REQUIRE(perim_count == k); // k perimeter edges in a k-cycle.
}

TEST_CASE("cyclic-convex detector accepts square layouts and rejects "
          "pentagram-style winding-2 layouts",
          "[cache]") {
    REQUIRE(CrossingTopology::is_cyclic_convex(
                12, distribute_nodes_around_square(3, 10.0)));
    REQUIRE(CrossingTopology::is_cyclic_convex(
                10, distribute_nodes_around_rectangle(3, 2, 5.0, 3.0)));
    REQUIRE(!CrossingTopology::is_cyclic_convex(5, pentagram_positions()));
}

TEST_CASE("cyclic-convex detector takes round-off-collinear side nodes as "
          "straight steps",
          "[cache]") {
    // Triangle, 4 nodes per side: accepted, and it crosses exactly like
    // the square with the same node count and order (exact coordinates,
    // so the brute-force oracle there is round-off free): one pair per
    // 4 nodes, C(12, 4) = 495.
    const auto tri = triangle_positions(4);
    REQUIRE(CrossingTopology::is_cyclic_convex(12, tri));
    CrossingTopology topo(12, tri);
    REQUIRE(topo.used_cyclic_fast_path());
    const auto pairs = topo.crossing_pairs();
    std::set<std::pair<int, int>> got(pairs.begin(), pairs.end());
    REQUIRE(got == oracle_pairs(12, distribute_nodes_around_square(3, 10.0)));
    REQUIRE(got.size() == 495);

    // A straight step that reverses direction is a spike, never convex.
    const PositionMap row{{0, {1.0, 0.0}}, {1, {2.0, 0.0}},
                          {2, {3.0, 0.0}}, {3, {4.0, 0.0}}};
    REQUIRE(!CrossingTopology::is_cyclic_convex(4, row));

    // A real dent, far above round-off, still fails the check.
    auto dented = distribute_nodes_around_square(3, 10.0);
    dented[1].second -= 1e-6;  // top-middle node pushed inward
    REQUIRE(!CrossingTopology::is_cyclic_convex(12, dented));
}

TEST_CASE("CrossingTopology fast path matches brute-force oracle on the "
          "SiN square (k=12)",
          "[cache]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    CrossingTopology topo(k, pos);
    REQUIRE(topo.used_cyclic_fast_path());

    const auto pairs = topo.crossing_pairs();
    const auto oracle = oracle_pairs(k, pos);
    REQUIRE(pairs.size() == oracle.size());
    std::set<std::pair<int, int>> got(pairs.begin(), pairs.end());
    REQUIRE(got == oracle);
}

TEST_CASE("CrossingTopology fast path matches brute-force oracle on the "
          "SiN square (k=20)",
          "[cache]") {
    const int nps = 5;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    CrossingTopology topo(k, pos);
    REQUIRE(topo.used_cyclic_fast_path());

    const auto pairs = topo.crossing_pairs();
    const auto oracle = oracle_pairs(k, pos);
    REQUIRE(pairs.size() == oracle.size());
    std::set<std::pair<int, int>> got(pairs.begin(), pairs.end());
    REQUIRE(got == oracle);
}

TEST_CASE("CrossingTopology fast path matches oracle on a rectangle layout",
          "[cache]") {
    const int nl = 3, nw = 2;
    const int k  = 2 * (nl + nw);
    const auto pos = distribute_nodes_around_rectangle(nl, nw, 5.0, 3.0);
    CrossingTopology topo(k, pos);
    REQUIRE(topo.used_cyclic_fast_path());

    const auto pairs = topo.crossing_pairs();
    const auto oracle = oracle_pairs(k, pos);
    REQUIRE(pairs.size() == oracle.size());
    std::set<std::pair<int, int>> got(pairs.begin(), pairs.end());
    REQUIRE(got == oracle);
}

TEST_CASE("Geometric fallback agrees with oracle on jittered non-convex "
          "positions",
          "[cache]") {
    // Take the square layout, perturb a node off the perimeter — that
    // breaks the cyclic-convex detector (winding-1 polygon but some
    // edges from the perturbed node are no longer chords of the convex
    // hull), forcing the geometric fallback path. Oracle agreement
    // is the property test.
    auto pos = distribute_nodes_around_square(3, 10.0);
    pos[6] = {3.5, 3.5}; // bottom-center pulled inward
    const int k = static_cast<int>(pos.size());
    CrossingTopology topo(k, pos);
    REQUIRE(!topo.used_cyclic_fast_path());

    const auto pairs = topo.crossing_pairs();
    const auto oracle = oracle_pairs(k, pos);
    REQUIRE(pairs.size() == oracle.size());
    std::set<std::pair<int, int>> got(pairs.begin(), pairs.end());
    REQUIRE(got == oracle);
}

TEST_CASE("Geometric fallback at k=20 with multiple perturbed nodes",
          "[cache]") {
    // Larger jittered case to exercise the geometric-fallback path's
    // memory + correctness at non-trivial scale. With 3 perturbed
    // nodes the layout is clearly non-cyclic-convex; the geometric
    // path must still bit-match the brute-force oracle.
    auto pos = distribute_nodes_around_square(5, 10.0);
    pos[2]  = {4.0, 4.0};   // inward
    pos[8]  = {5.5, 4.5};   // inward
    pos[14] = {2.0, 5.0};   // inward
    const int k = static_cast<int>(pos.size());
    CrossingTopology topo(k, pos);
    REQUIRE(!topo.used_cyclic_fast_path());

    const auto pairs = topo.crossing_pairs();
    const auto oracle = oracle_pairs(k, pos);
    REQUIRE(pairs.size() == oracle.size());
    std::set<std::pair<int, int>> got(pairs.begin(), pairs.end());
    REQUIRE(got == oracle);
}

TEST_CASE("CrossingTopology K_4 has exactly one crossing pair on the unit "
          "square (matches legacy count_crossings_with_detail)",
          "[cache]") {
    PositionMap pos;
    pos[0] = {0.0, 0.0};
    pos[1] = {1.0, 0.0};
    pos[2] = {1.0, 1.0};
    pos[3] = {0.0, 1.0};
    CrossingTopology topo(4, pos);
    REQUIRE(topo.crossing_pairs().size() == 1);
    // The crossing pair is edge (0,2) × (1,3); both are non-perimeter
    // (|u-v| == 2 for k=4 → diff != 1 and diff != k-1=3 → not in mask).
    const auto& edges = topo.edge_list();
    const auto& pr = topo.crossing_pairs()[0];
    const auto& e1 = edges[pr.first];
    const auto& e2 = edges[pr.second];
    REQUIRE(e1 == std::make_pair(0, 2));
    REQUIRE(e2 == std::make_pair(1, 3));
}
