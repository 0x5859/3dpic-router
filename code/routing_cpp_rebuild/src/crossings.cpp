#include "sinic/crossings.hpp"

#include "sinic/geometry.hpp"

namespace sinic {

bool edge_crosses(const Edge& e1, const Edge& e2, const PositionMap& positions) {
    const int n1 = std::get<0>(e1);
    const int n2 = std::get<1>(e1);
    const int n3 = std::get<0>(e2);
    const int n4 = std::get<1>(e2);
    return geo::segments_intersect_strict(
        positions.at(n1), positions.at(n2),
        positions.at(n3), positions.at(n4));
}

CrossingDetail count_crossings_with_detail(EdgeList& edges,
                                           const PositionMap& positions) {
    CrossingDetail detail;

    // Reset counts and seed the per-edge map with zero entries so that
    // edges with no crossings still appear (legacy main.cpp behavior;
    // some downstream loops iterate over the map's keys).
    for (auto& e : edges) {
        std::get<2>(e).crossings = 0;
    }
    for (const auto& e : edges) {
        detail.per_edge[{std::get<0>(e), std::get<1>(e)}] = 0;
    }

    const int n = static_cast<int>(edges.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (edge_crosses(edges[i], edges[j], positions)) {
                ++detail.total;
                const auto k1 = std::make_pair(std::get<0>(edges[i]),
                                               std::get<1>(edges[i]));
                const auto k2 = std::make_pair(std::get<0>(edges[j]),
                                               std::get<1>(edges[j]));
                ++detail.per_edge[k1];
                ++detail.per_edge[k2];
            }
        }
    }

    for (auto& e : edges) {
        const auto key = std::make_pair(std::get<0>(e), std::get<1>(e));
        std::get<2>(e).crossings = detail.per_edge[key];
    }
    return detail;
}

void count_interlayer_crossings(EdgeList& layer_lo,
                                EdgeList& layer_hi,
                                const PositionMap& positions,
                                bool reset) {
    if (reset) {
        for (auto& e : layer_lo) {
            auto& ed = std::get<2>(e);
            ed.interlayerCrossings      = 0;
            ed.interlayerCrossingsAbove = 0;
            ed.interlayerCrossingsBelow = 0;
        }
        for (auto& e : layer_hi) {
            auto& ed = std::get<2>(e);
            ed.interlayerCrossings      = 0;
            ed.interlayerCrossingsAbove = 0;
            ed.interlayerCrossingsBelow = 0;
        }
    }

    for (auto& e_lo : layer_lo) {
        for (auto& e_hi : layer_hi) {
            if (edge_crosses(e_lo, e_hi, positions)) {
                auto& d_lo = std::get<2>(e_lo);
                auto& d_hi = std::get<2>(e_hi);
                ++d_lo.interlayerCrossings;
                ++d_hi.interlayerCrossings;
                ++d_lo.interlayerCrossingsAbove;
                ++d_hi.interlayerCrossingsBelow;
            }
        }
    }
}

} // namespace sinic
