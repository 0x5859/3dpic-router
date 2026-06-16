#include <catch2/catch.hpp>

#include "sinic/crossings.hpp"
#include "sinic/positions.hpp"

#include <set>

using sinic::EdgeList;
using sinic::EdgeData;
using sinic::PositionMap;
using sinic::count_crossings_with_detail;
using sinic::count_interlayer_crossings;
using sinic::distribute_nodes_around_square;

namespace {
EdgeList complete_graph(int k) {
    EdgeList edges;
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            edges.emplace_back(i, j, EdgeData{});
        }
    }
    return edges;
}
} // namespace

TEST_CASE("count_crossings_with_detail K4 has one crossing", "[crossings]") {
    // 4 nodes at the corners of the unit square (no perimeter
    // exclusion) → only one crossing pair: (0,2) × (1,3).
    PositionMap pos;
    pos[0] = {0.0, 0.0};
    pos[1] = {1.0, 0.0};
    pos[2] = {1.0, 1.0};
    pos[3] = {0.0, 1.0};
    auto edges = complete_graph(4);
    const auto detail = count_crossings_with_detail(edges, pos);
    REQUIRE(detail.total == 1);

    // Σ_e crossings_e == 2 * total (end-point doubled).
    int sum = 0;
    for (const auto& e : edges) sum += std::get<2>(e).crossings;
    REQUIRE(sum == 2 * detail.total);
}

TEST_CASE("count_crossings on square-perimeter K_k matches expected scaling",
          "[crossings]") {
    // For the perimeter layout the number of geometric crossing pairs
    // grows like Θ(k^4); we just sanity-check small k against a known
    // reference: nodes_per_side = 3 (k = 12) has 110 intersecting
    // pairs (counted by the legacy Python oracle).
    auto pos = distribute_nodes_around_square(3, 10.0);
    auto edges = complete_graph(static_cast<int>(pos.size()));
    const auto detail = count_crossings_with_detail(edges, pos);
    REQUIRE(detail.total > 0);
    // Per-edge counts add up to 2 * total (endpoint-doubled).
    int sum = 0;
    for (const auto& e : edges) sum += std::get<2>(e).crossings;
    REQUIRE(sum == 2 * detail.total);
}

TEST_CASE("count_interlayer_crossings doubles per-edge", "[crossings]") {
    // Same K4 unit square; one diagonal in layer 0, the other in
    // layer 1 → exactly one interlayer event, each edge gets +1.
    PositionMap pos;
    pos[0] = {0.0, 0.0};
    pos[1] = {1.0, 0.0};
    pos[2] = {1.0, 1.0};
    pos[3] = {0.0, 1.0};
    EdgeList l0{{0, 2, EdgeData{}}};
    EdgeList l1{{1, 3, EdgeData{}}};
    count_interlayer_crossings(l0, l1, pos);
    REQUIRE(std::get<2>(l0[0]).interlayerCrossings == 1);
    REQUIRE(std::get<2>(l1[0]).interlayerCrossings == 1);
}

TEST_CASE("count_interlayer_crossings stamps _above on the lower layer and "
          "_below on the upper layer (M6 §3-1 注 1 mirror)",
          "[crossings]") {
    // Single intersection event: e ∈ layer_lo (=0) crosses f ∈ layer_hi
    // (=1). Per the spec, e gets _above += 1 (sees layer above), f gets
    // _below += 1 (sees layer below), and both interlayerCrossings += 1.
    PositionMap pos;
    pos[0] = {0.0, 0.0};
    pos[1] = {1.0, 0.0};
    pos[2] = {1.0, 1.0};
    pos[3] = {0.0, 1.0};
    EdgeList lo{{0, 2, EdgeData{}}};
    EdgeList hi{{1, 3, EdgeData{}}};
    count_interlayer_crossings(lo, hi, pos);
    const auto& d_lo = std::get<2>(lo[0]);
    const auto& d_hi = std::get<2>(hi[0]);
    REQUIRE(d_lo.interlayerCrossings      == 1);
    REQUIRE(d_lo.interlayerCrossingsAbove == 1);
    REQUIRE(d_lo.interlayerCrossingsBelow == 0);
    REQUIRE(d_hi.interlayerCrossings      == 1);
    REQUIRE(d_hi.interlayerCrossingsAbove == 0);
    REQUIRE(d_hi.interlayerCrossingsBelow == 1);
    // Per-edge invariant: _above + _below == interlayerCrossings.
    REQUIRE(d_lo.interlayerCrossingsAbove + d_lo.interlayerCrossingsBelow
            == d_lo.interlayerCrossings);
    REQUIRE(d_hi.interlayerCrossingsAbove + d_hi.interlayerCrossingsBelow
            == d_hi.interlayerCrossings);
}

TEST_CASE("count_interlayer_crossings reset=false accumulates across pairs "
          "(M6 multi-layer middle-layer mirror)",
          "[crossings]") {
    // Synthetic 3-edge scenario across two passes: the middle layer's
    // single edge crosses one edge in each of the lower / upper layers.
    // After the two adjacent-pair passes, the middle edge has
    // _above=1 (from middle↔upper pass) and _below=1 (from
    // lower↔middle pass), with the legacy counter at 2 (= 1+1).
    PositionMap pos;
    pos[0] = {0.0, 0.0};
    pos[1] = {1.0, 0.0};
    pos[2] = {1.0, 1.0};
    pos[3] = {0.0, 1.0};
    EdgeList lo{{0, 2, EdgeData{}}};
    EdgeList mid{{1, 3, EdgeData{}}};
    EdgeList hi{{0, 2, EdgeData{}}};
    // pass 1: layer_lo ↔ layer_mid (mid is "above" lo → mid sees _below)
    count_interlayer_crossings(lo, mid, pos, /*reset=*/false);
    // pass 2: layer_mid ↔ layer_hi (mid is "below" hi → mid sees _above)
    count_interlayer_crossings(mid, hi, pos, /*reset=*/false);
    const auto& md = std::get<2>(mid[0]);
    REQUIRE(md.interlayerCrossings      == 2);
    REQUIRE(md.interlayerCrossingsAbove == 1);
    REQUIRE(md.interlayerCrossingsBelow == 1);
}
