#pragma once

// Common scalar / aggregate types shared across the modular split.
// Pre-M5 these all lived in main.cpp. M5 extracts them into a leaf header
// with no internal dependencies so every other sinic module can `#include`
// it without circular pulls.

#include <cstddef>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sinic {

using Point = std::pair<double, double>;

// Per-edge attributes. Field names mirror the legacy main.cpp::EdgeData
// so JSON I/O stays bit-exact with pre-M5 writers (`save_subgraphs_to_json`
// / `save_subgraphs_to_json_newformat`). REFACTOR_GOALS.md §3-1 documents
// the on-disk shape.
//
// M6 (REFACTOR_GOALS.md §2-3 mirror): split direction-resolved
// interlayer counters land on EdgeData. Invariants per `core.py::
// count_interlayercrossings` doctring:
//   - `interlayerCrossingsAbove + interlayerCrossingsBelow ==
//      interlayerCrossings` per edge.
//   - Whole-graph sums:
//      Σ_e interlayerCrossingsAbove = Σ_e interlayerCrossingsBelow = G
//      (geometric event count); Σ_e interlayerCrossings = 2*G.
struct EdgeData {
    int    layer                    = 0;
    int    crossings                = 0;
    double loss                     = 0.0;
    int    interlayerCrossings      = 0;
    int    interlayerCrossingsAbove = 0;
    int    interlayerCrossingsBelow = 0;
};

// Hash combiner for std::pair<int,int>. Kept identical to the legacy
// pairHash in main.cpp (golden-ratio combiner) so existing unordered_map
// hash buckets reproduce.
struct pairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        auto h1 = std::hash<int>()(p.first);
        auto h2 = std::hash<int>()(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

using Edge       = std::tuple<int, int, EdgeData>;
using EdgeList   = std::vector<Edge>;
using PositionMap = std::unordered_map<int, Point>;

} // namespace sinic
