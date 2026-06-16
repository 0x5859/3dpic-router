#include "sinic/loss.hpp"

#include <cstdlib>

namespace sinic {

void compute_per_edge_loss(EdgeList& edges,
                           double loss_crossing,
                           double loss_taper,
                           int    edge_coupler_layer) noexcept {
    for (auto& e : edges) {
        auto& ed = std::get<2>(e);
        // M6 (§2-3 目标 B): taper hops = |layer - edge_coupler_layer|;
        // for the legacy default ecl=0 this reduces to the pre-M3
        // `2 * layer * loss_taper` because `layer >= 0`.
        // The `* 2` on the crossings term accounts for the two
        // waveguides per link (bidirectional); see REFACTOR_GOALS.md
        // §2-2 现状段 hard-coded-vs-implicit 2× analysis.
        // int64 |layer - ecl| — same overflow/UB hardening as
        // graph.cpp::loss_function's taper site (REFACTOR_GOALS.md §1-2
        // 设计取舍 #8). This site is currently reachability-safe (both
        // callers feed only [0, L)-bucket-filtered edges via
        // create_subgraphs, or the planar layer=0 reference), so a
        // pathological raw `int` cannot arrive here today — but it is
        // widened anyway so "no signed `int` layer-difference anywhere"
        // is a codebase-wide invariant (defense-in-depth) rather than a
        // property that silently depends on the call graph. Identical
        // bits for every reachable value.
        const long long taper_hops = std::llabs(
            static_cast<long long>(ed.layer)
          - static_cast<long long>(edge_coupler_layer));
        ed.loss = loss_crossing * ed.crossings * 2
                + 2.0 * static_cast<double>(taper_hops) * loss_taper;
    }
}

void apply_interlayer_loss(EdgeList& edges,
                           double loss_interlayer_crossing) noexcept {
    for (auto& e : edges) {
        auto& ed = std::get<2>(e);
        ed.loss = ed.loss + ed.interlayerCrossings * loss_interlayer_crossing;
    }
}

} // namespace sinic
