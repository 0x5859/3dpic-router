#pragma once

// High-level crossing detection on edge lists. Pre-M5 these were
// member functions of `SiNInterconnectionGraph`; M5 extracts them
// as free functions taking an explicit position map so the same
// primitives can be reused from tests and (M6) from the cached
// topology builder without instantiating a whole graph.
//
// Algorithmic semantics unchanged from the legacy main.cpp
// (`count_crossings_with_detail` / `count_interlayercrossings`):
//   - shared endpoint pairs are NOT counted
//   - per-edge counters are end-point-doubled (an event between e and f
//     increments both e.crossings and f.crossings) — this is the legacy
//     "geometric event" Σ_e = 2G convention the Python schema documents
//     in §3-1 注 1.
//
// REFACTOR_GOALS.md §1-2-a flagged this module as the M5 deliverable
// boundary; the cached topology (M6, `crossings_cache.hpp`) and the
// crosstalk engine (M7, `crosstalk.hpp`) landed as separate siblings
// without modifying this header.

#include "sinic/types.hpp"

#include <unordered_map>
#include <utility>

namespace sinic {

struct CrossingDetail {
    int total = 0;
    std::unordered_map<std::pair<int, int>, int, pairHash> per_edge;
};

// Count geometric crossings within a single edge list. Writes the
// per-edge count back into each edge's `EdgeData.crossings` field and
// returns (total events, per-edge map).
CrossingDetail count_crossings_with_detail(EdgeList& edges,
                                           const PositionMap& positions);

// Count interlayer crossings between two adjacent layer edge lists.
// `layer_lo` is at the *lower* layer index (Layer_i), `layer_hi` at
// the upper (Layer_{i+1}). For every crossing event between
// `e ∈ layer_lo` and `f ∈ layer_hi`:
//
//   - `e.interlayerCrossingsAbove += 1` (lower edge sees the layer above)
//   - `f.interlayerCrossingsBelow += 1` (upper edge sees the layer below)
//   - `e.interlayerCrossings += 1` and `f.interlayerCrossings += 1`
//     (legacy single-counter; full-graph sum = 2*G, M3 fallback path)
//
// `reset` controls whether the three counters are zeroed before
// counting. Default `false` mirrors Python
// `routing_py_rebuild/core.py::count_interlayercrossings` so the M6
// multi-layer aggregator (`graph.cpp::create_subgraphs`) sees
// matching defaults across both languages — middle layers accumulate
// events from both neighbours via repeated adjacent-pair calls.
// Callers that need a fresh count (e.g. unit tests with freshly-
// constructed EdgeData) can omit the argument since the default-
// constructed counters are already zero.
//
// REFACTOR_GOALS.md §3-1 注 1 / §4 M3 附注.
void count_interlayer_crossings(EdgeList& layer_lo,
                                EdgeList& layer_hi,
                                const PositionMap& positions,
                                bool reset = false);

// Predicate variant — useful for tests / debugging without mutating.
bool edge_crosses(const Edge& e1, const Edge& e2, const PositionMap& positions);

} // namespace sinic
