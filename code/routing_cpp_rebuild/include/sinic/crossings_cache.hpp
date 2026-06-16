#pragma once

// Phase A cached crossing topology — the M6 deliverable (REFACTOR_GOALS.md
// §2-1 C++ mirror; Python M2 ship at `core.py::_ensure_crossings_ready`).
//
// For a complete graph K_k with fixed node positions, the set of "which
// edge pairs geometrically cross" is **constant** throughout an optimizer
// run — only the per-edge layer labels change. Phase A computes the
// crossing-pair index exactly once; Phase B (`graph.cpp::loss_function`)
// then performs O(X) integer classification per loss eval (X = number of
// crossing events) instead of the legacy O(E²) geometric scan.
//
// Two-tier algorithm (mirrors Python `_build_crossing_pairs_cyclic` /
// `_build_crossing_pairs_geometric`):
//
//   1. **Cyclic-convex fast path** — when the k node positions trace a
//      simple convex polygon (every SiN square / rectangle layout from
//      `positions.cpp`), two chords (u_i, v_i), (u_j, v_j) of K_k cross
//      iff exactly one of {u_j, v_j} lies in the open integer interval
//      (u_i, v_i) AND they don't share a vertex. Purely integer XOR;
//      no float arithmetic; hits the §2-1 "≤ 5 s build at k=160" budget.
//
//   2. **Geometric fallback** — arbitrary positions: vectorized 2D
//      orientation test with closed-form handling of T-junctions and
//      fully-collinear partial overlaps. Bit-exact with the legacy
//      `segments_intersect_strict` shapely-equivalent semantics
//      (`crossings.cpp::edge_crosses`).
//
// Both paths produce the same crossing-pair set for cyclic-convex
// layouts; the oracle test `test_crossings_cache.cpp::Cyclic-convex
// matches geometric oracle` validates parity on k ∈ {8, 12, 20, 40}.
//
// The cache also memoizes the perimeter-edge mask (|u-v| ∈ {1, k-1})
// since Phase B needs it on every call to pin perimeter edges to the
// configured perimeter layer (REFACTOR_GOALS.md §2-3-B 补充 mirror).
//
// **Detection rejects pentagram-style winding-2 layouts**: signed turn
// angles sum to ±4π rather than ±2π. The fast path on such a layout
// would produce *wrong* crossing pairs; the detector forces a fallback
// to the geometric path. See `test_crossings_cache.cpp::pentagram
// rejected by cyclic detector`.

#include "sinic/types.hpp"

#include <utility>
#include <vector>

namespace sinic {

class CrossingTopology {
public:
    // One-shot constructor: builds the crossing-pair index for K_k with
    // the given position map. Either uses the cyclic-convex fast path
    // (when `is_cyclic_convex(positions)` returns true) or the geometric
    // fallback. The choice is recorded in `used_cyclic_fast_path()`
    // for diagnostics.
    //
    // `positions` is expected to map every node index in [0, k) to a
    // 2D point. Missing entries cause the detector to return false and
    // the build falls back to the geometric path (which will throw
    // `std::out_of_range` on `positions.at`).
    CrossingTopology(int k, const PositionMap& positions);

    int num_edges()          const noexcept { return n_edges_; }
    int num_crossing_pairs() const noexcept {
        return static_cast<int>(crossing_pairs_.size());
    }
    bool used_cyclic_fast_path() const noexcept { return used_cyclic_; }

    // Canonical edge ordering: `(u, v)` with `u < v` in row-major
    // construction order — matches `SiNInterconnectionGraph::allEdges_`
    // so callers can index both with the same `i ∈ [0, num_edges())`.
    const std::vector<std::pair<int, int>>& edge_list() const noexcept {
        return edge_list_;
    }

    // Each entry is `(i, j)` with `i < j`: both are indices into
    // `edge_list()`. The pair list is the M6 hot-loop input — Phase B
    // classifies each pair as intra / inter based on the post-pin
    // layer assignment of its two endpoint edges.
    const std::vector<std::pair<int, int>>& crossing_pairs() const noexcept {
        return crossing_pairs_;
    }

    // Perimeter mask: `perimeter_mask()[i]` is true iff edge `i` has
    // node-index difference `|u-v| ∈ {1, k-1}` (the K_k perimeter ring).
    // Phase B pins these edges to the configured perimeter layer
    // regardless of the optimizer's proposal.
    const std::vector<bool>& perimeter_mask() const noexcept {
        return perimeter_mask_;
    }

    // ---- detection helpers (exposed for testing) --------------------
    //
    // Returns true if `positions[0..k-1]` traces a *simple* convex
    // polygon (winding number ±1). A consistent turn-sign alone is
    // not sufficient (winding-2 pentagrams pass that), so the function
    // additionally requires |Σ signed turn angles| ≈ 2π within 1e-6
    // and all positions pairwise distinct. Mirrors Python
    // `core.py::_is_cyclic_convex_positions`.
    static bool is_cyclic_convex(int k, const PositionMap& positions);

private:
    int n_edges_     = 0;
    bool used_cyclic_ = false;
    std::vector<std::pair<int, int>> edge_list_;
    std::vector<std::pair<int, int>> crossing_pairs_;
    std::vector<bool> perimeter_mask_;

    void build_edge_list_(int k);
    void build_perimeter_mask_(int k);
    void build_cyclic_(int k);
    void build_geometric_(const PositionMap& positions);
};

} // namespace sinic
