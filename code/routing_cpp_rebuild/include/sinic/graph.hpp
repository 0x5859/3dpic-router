#pragma once

// SiNInterconnectionGraph — the domain model. Pre-M5 lived as a 700-line
// class inside main.cpp; M5 extracted it to its own translation unit and
// froze the optimizer-facing contract per REFACTOR_GOALS.md §1-3-c
// (`num_edges`, `layer_bounds`, `loss_function`,
// `apply_optimization_result`, `seed_from_routing`). Internal members are
// public-readable accessors so the optimizer / I/O modules don't reach
// into private state directly.
//
// M6 deliverables (REFACTOR_GOALS.md §2-1 + §2-3 mirror):
//   - Constructor accepts L / edge_coupler_layer / perimeter_layer /
//     layer_pitch_um / waveguides_per_link in addition to the legacy
//     loss coefficients. Validation rules mirror Python
//     `core.py::SiNInterconnectionGraph.__init__` + the module-level
//     `_validate_waveguides_per_link` helper (32-cap, bool reject,
//     wpl=1 / wpl ∉ {1, 2} stderr warnings).
//   - `layer_bounds()` returns `{0.0, L-1}` for the optimizer.
//   - `loss_function` uses the Phase A `CrossingTopology` cache and
//     runs O(X) integer classification per call (no geometry).
//   - `create_subgraphs` emits **L entries** (including empty layers)
//     and runs `count_interlayer_crossings` for every adjacent pair
//     with `reset=false` so middle layers accumulate from both sides
//     (mirrors `core.py::create_subgraphs` M3 behavior).
//   - `apply_optimization_result` pins perimeter edges to
//     `perimeter_layer` (configurable, default = ecl) and resets all
//     three interlayer counters (legacy, `_above`, `_below`).
//   - Taper formula via `compute_per_edge_loss(..., edge_coupler_layer)`.
//
// M7 deliverables (REFACTOR_GOALS.md §2-2 mirror):
//   - Constructor accepts `loss_intralayer_crosstalk` /
//     `loss_interlayer_crosstalk` (std::optional<double> to mirror
//     Python's `None` semantics) plus `coherence_model` /
//     `polarization` strings. Members are exposed via accessors so
//     `compute_crosstalk_tensor` (in `crosstalk.hpp`) can read them
//     and so the I/O writer can emit them under
//     `General Parameters.Loss of Intralayer Crosstalk` etc.
//   - The graph itself does **not** compute the crosstalk tensor —
//     `crosstalk.hpp` is a separate pure analytical module (matching
//     Python's `routing_py_rebuild/crosstalk.py` boundary).

#include "sinic/types.hpp"

#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sinic {

class CrossingTopology;  // forward decl — implementation in crossings_cache.hpp

class SiNInterconnectionGraph {
public:
    // Multi-layer constructor (M6). All multi-layer parameters are
    // optional with defaults that match Python `core.py::__init__`.
    // Validation rules:
    //   * L >= 1 (raise `std::invalid_argument`)
    //   * 0 <= edge_coupler_layer < L
    //   * 0 <= perimeter_layer    < L (default = edge_coupler_layer if -1)
    //   * layer_pitch_um > 0
    //   * waveguides_per_link in [1, 32]; not bool; integer
    //
    // `waveguides_per_link == 1` is the post-2026-05-15 default
    // (REFACTOR_GOALS.md §7 Q-e "M8 后默认翻转") and emits a one-time
    // informational notice to `std::cerr` (paper-aligned geometric
    // convention; stats/loss mismatch context surfaced once per
    // process). `wpl ∉ {1, 2}` still emits a per-call experimental
    // warning. `wpl <= 0` / `wpl > 32` raise.
    SiNInterconnectionGraph(
        int k,
        const PositionMap& positions,
        double loss_crossing            = 0.1,
        double loss_taper               = 0.05,
        double loss_interlayer_crossing = 0.001,
        int    num_layers               = 2,
        int    edge_coupler_layer       = 0,
        int    perimeter_layer          = -1,    // -1 sentinel → = edge_coupler_layer
        double layer_pitch_um           = 1.2,
        int    waveguides_per_link      = 1,
        // M7 (REFACTOR_GOALS.md §2-2): crosstalk coefficients +
        // physical-model labels. `std::nullopt` mirrors Python's `None`
        // and triggers the short-circuit in `compute_crosstalk_tensor`.
        std::optional<double> loss_intralayer_crosstalk = std::nullopt,
        std::optional<double> loss_interlayer_crosstalk = std::nullopt,
        std::string coherence_model = "incoherent_v1",
        std::string polarization    = "TE0_only");

    // Non-copyable / non-movable — the optimizer holds a non-owning
    // reference and the Phase A `CrossingTopology` cache contains
    // back-references via indices into our `allEdges_` ordering.
    SiNInterconnectionGraph(const SiNInterconnectionGraph&)            = delete;
    SiNInterconnectionGraph& operator=(const SiNInterconnectionGraph&) = delete;
    SiNInterconnectionGraph(SiNInterconnectionGraph&&)                 = delete;
    SiNInterconnectionGraph& operator=(SiNInterconnectionGraph&&)      = delete;
    ~SiNInterconnectionGraph();

    // ---- §1-3-c "core surface to optimizer" --------------------------
    int    num_edges() const noexcept { return static_cast<int>(allEdges_.size()); }
    int    k()         const noexcept { return k_; }
    int    L()         const noexcept { return L_; }
    int    edge_coupler_layer() const noexcept { return edge_coupler_layer_; }
    int    perimeter_layer()    const noexcept { return perimeter_layer_; }
    double layer_pitch_um()     const noexcept { return layer_pitch_um_; }
    int    waveguides_per_link() const noexcept { return waveguides_per_link_; }

    // M6: bounds widen to [0, L-1] (Python M3 §2-3 目标 A mirror). L=1
    // is rejected at the optimizer surface — see `optimizer.cpp`.
    std::pair<double, double> layer_bounds() const noexcept {
        return {0.0, static_cast<double>(L_ - 1)};
    }

    // Apply a layer assignment (one int per edge) to `allEdges_`.
    // Edges on the perimeter (|u-v| == 1 or k-1) are forced to
    // `perimeter_layer` (M6 generalization of the legacy hard-coded 0).
    // Non-perimeter values are taken AS-IS — NO clamp to [0, L-1] —
    // matching `loss_function` and Python `_apply_layer_assignment`
    // (raw `int(layer)`). The optimizer surface already guarantees
    // [0, L-1] via `decode_layer_index`; direct/fixed-layer callers
    // feeding out-of-range values get Python-equivalent behavior (such
    // edges drop from every per-layer subgraph in `create_subgraphs`,
    // staying only in the planar `complete_graph`). Do NOT reintroduce
    // a clamp here: it silently broke Python↔C++ parity on the
    // fixed-layers path (REFACTOR_GOALS.md §1-2-b; §1-2 design-tradeoff
    // #4; §1-4 M8 dual-review Opus P2-O5 RESOLVED).
    //
    // Resets per-edge `crossings` / `loss` / interlayer counters to 0
    // so that callers reading `graph.all_edges()` directly see a clean
    // post-optimize snapshot (M6 mirror of Python
    // `apply_optimization_result` `nx.set_edge_attributes(..., 0, ...)`
    // block). The `.crossings` / `.loss` are still recomputed inside
    // `create_subgraphs` / `loss_function` on demand — this reset only
    // affects readers between `apply_optimization_result` and the next
    // create/loss call.
    void apply_optimization_result(const std::vector<int>& layers);

    // Evaluate the optimizer objective on a candidate layer assignment.
    // Phase B (M6): O(X) integer classification using the cached
    // `CrossingTopology` topology. Does NOT mutate `allEdges_`.
    //
    // Returns `alpha * mean(per_edge_loss[selected]) + beta * var(...)`
    // where `selected` is the subset of edges whose post-pin layer
    // appears in `unique(layers)`. The selection rule mirrors Python
    // `core.py::loss_function` — for L=2 / ecl=0 / perimeter=0 it is
    // bit-equivalent to the pre-M2 `for layer in unique: include sub_G[layer]`
    // aggregation (see REFACTOR_GOALS.md §2-3 验收第 1 条).
    double loss_function(const std::vector<int>& layers);

    // Idempotent entry point for Phase A precomputation. Surface this
    // separately from the constructor so the orchestrator can time the
    // precompute (emitted as `initial_crossing_count_ms` in
    // `run_report.json` per §3-2). Subsequent calls are no-ops.
    void build_crossings_index();

    // ---- legacy initial-seed routine (Python routing_method_1) ------
    // M6: default `layer` argument is `(edge_coupler_layer + 1) % L` —
    // the "non-coupler" layer mirroring Python `_seed_from_routing`.
    void seed_from_routing(const std::vector<int>& takeaway, int layer = -1);

    // ---- read-only accessors (I/O + statistics modules) -------------
    const EdgeList&    all_edges()         const noexcept { return allEdges_; }
    EdgeList&          all_edges_mut()           noexcept { return allEdges_; }
    const EdgeList&    planar_reference()  const noexcept { return _G_Planar_; }
    const std::vector<EdgeList>& sub_graphs() const noexcept { return sub_G_; }
    const PositionMap& positions()         const noexcept { return positions_; }
    double             loss_crossing()     const noexcept { return loss_crossing_; }
    double             loss_taper()        const noexcept { return loss_taper_; }
    double             loss_interlayer_crossing() const noexcept { return loss_interlayer_crossing_; }
    // M7: crosstalk coefficients / physical-model labels.
    const std::optional<double>& loss_intralayer_crosstalk() const noexcept {
        return loss_intralayer_crosstalk_;
    }
    const std::optional<double>& loss_interlayer_crosstalk() const noexcept {
        return loss_interlayer_crosstalk_;
    }
    const std::string& coherence_model() const noexcept { return coherence_model_; }
    const std::string& polarization()    const noexcept { return polarization_; }
    const CrossingTopology* topology() const noexcept { return topology_.get(); }

    // ---- analysis helpers --------------------------------------------
    void create_subgraphs();                  // rebuild sub_G_ from allEdges_
    void analyze_loss(std::ostream& out);     // mirror legacy "analyze_loss"

    // M5 LOSS WEIGHTS — the optimizer objective combines mean and
    // variance of per-edge losses. Pre-M5 these were hard-coded as
    // `alpha=100000.0, beta=0.0` literals inside `loss_function`;
    // M5 keeps the same defaults but exposes them as a configurable
    // surface for future tuning.
    void set_loss_weights(double alpha, double beta) noexcept {
        alpha_ = alpha;
        beta_  = beta;
    }
    double alpha() const noexcept { return alpha_; }
    double beta()  const noexcept { return beta_; }

private:
    int           k_;
    int           L_;
    int           edge_coupler_layer_;
    int           perimeter_layer_;
    double        layer_pitch_um_;
    int           waveguides_per_link_;
    PositionMap   positions_;
    double        loss_crossing_;
    double        loss_taper_;
    double        loss_interlayer_crossing_;

    // M7: optional crosstalk coefficients + physical-model labels.
    std::optional<double> loss_intralayer_crosstalk_;
    std::optional<double> loss_interlayer_crosstalk_;
    std::string           coherence_model_;
    std::string           polarization_;

    EdgeList      allEdges_;
    EdgeList      _G_Planar_;
    std::vector<EdgeList> sub_G_;

    double        alpha_ = 100000.0;
    double        beta_  = 0.0;

    // Phase A cache — built lazily on first `loss_function` /
    // `build_crossings_index` call. Owned via unique_ptr to keep the
    // header free of `crossings_cache.hpp` (forward decl above).
    std::unique_ptr<CrossingTopology> topology_;

    void set_edge_layer_(int u, int v, int layer);
    int  k_base_no_carry_add_(std::initializer_list<int> args) const;
    int  k_base_no_carry_sub_(std::initializer_list<int> args) const;
    void ensure_topology_();
};

} // namespace sinic
