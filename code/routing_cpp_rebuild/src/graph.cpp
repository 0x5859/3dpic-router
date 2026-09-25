#include "sinic/graph.hpp"

#include "sinic/crossings.hpp"
#include "sinic/crossings_cache.hpp"
#include "sinic/loss.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace sinic {

namespace {

// Mirrors Python `core.py::_validate_waveguides_per_link`. The
// `_WPL_PHYSICAL_MAX = 32` cap and the wpl=1 / wpl-experimental
// messages match the Python wording so reviewers diffing the two
// implementations can grep across both languages.
//
// Post-2026-05-15 (REFACTOR_GOALS.md §7 Q-e "M8 后默认翻转"): the
// default is wpl=1, and the wpl=1 path emits an *informational*
// one-line notice at most once per process — never per call — so a
// default run does not spam stderr while still surfacing the
// stats/loss-misalignment context for the user once.
constexpr int kWplPhysicalMax = 32;

// Process-wide one-time flag for the wpl=1 informational notice. Lives
// at namespace scope (anonymous, internal linkage). Static-init zeroes
// it; ``reset_wpl_one_notice_for_tests`` resets it from test code.
bool g_wpl_one_notice_emitted = false;

void emit_wpl_one_notice_once() {
    if (g_wpl_one_notice_emitted) {
        return;
    }
    g_wpl_one_notice_emitted = true;
    std::cerr <<
        "[sinic info] waveguides_per_link=1 (default): stats use the "
        "geometric crossing convention (matches paper reporting). The "
        "per-edge loss formula keeps an implicit 2x from endpoint "
        "double-counting; pass waveguides_per_link=2 for the physical "
        "convention aligned with the loss model. See REFACTOR_GOALS.md "
        "§7 Q-e.\n";
}

int validate_waveguides_per_link(int wpl) {
    if (wpl <= 0) {
        std::ostringstream os;
        os << "waveguides_per_link must be a positive integer; got " << wpl << ".";
        throw std::invalid_argument(os.str());
    }
    if (wpl > kWplPhysicalMax) {
        std::ostringstream os;
        os << "waveguides_per_link=" << wpl << " exceeds the physical sanity cap of "
           << kWplPhysicalMax << "; typical photonic interconnects use 1–8. "
              "If you really need a larger value, raise kWplPhysicalMax intentionally.";
        throw std::invalid_argument(os.str());
    }
    if (wpl == 1) {
        emit_wpl_one_notice_once();
    } else if (wpl != 2) {
        std::cerr <<
            "[sinic] waveguides_per_link=" << wpl
            << " is experimental; stats/loss semantics are not aligned "
               "until §7 Q-e is resolved.\n";
    }
    return wpl;
}

} // namespace

SiNInterconnectionGraph::SiNInterconnectionGraph(
    int k,
    const PositionMap& positions,
    double loss_crossing,
    double loss_taper,
    double loss_interlayer_crossing,
    int    num_layers,
    int    edge_coupler_layer,
    int    perimeter_layer,
    double layer_pitch_um,
    int    waveguides_per_link,
    std::optional<double> loss_intralayer_crosstalk,
    std::optional<double> loss_interlayer_crosstalk,
    std::string coherence_model,
    std::string polarization)
    : k_(k),
      L_(num_layers),
      edge_coupler_layer_(edge_coupler_layer),
      perimeter_layer_((perimeter_layer < 0) ? edge_coupler_layer : perimeter_layer),
      layer_pitch_um_(layer_pitch_um),
      waveguides_per_link_(validate_waveguides_per_link(waveguides_per_link)),
      positions_(positions),
      loss_crossing_(loss_crossing),
      loss_taper_(loss_taper),
      loss_interlayer_crossing_(loss_interlayer_crossing),
      loss_intralayer_crosstalk_(loss_intralayer_crosstalk),
      loss_interlayer_crosstalk_(loss_interlayer_crosstalk),
      coherence_model_(std::move(coherence_model)),
      polarization_(std::move(polarization)) {

    if (L_ < 1) {
        std::ostringstream os;
        os << "L must be >= 1; got " << L_ << ".";
        throw std::invalid_argument(os.str());
    }
    if (edge_coupler_layer_ < 0 || edge_coupler_layer_ >= L_) {
        std::ostringstream os;
        os << "edge_coupler_layer must be in [0, L); got "
           << edge_coupler_layer_ << " with L=" << L_ << ".";
        throw std::invalid_argument(os.str());
    }
    if (perimeter_layer_ < 0 || perimeter_layer_ >= L_) {
        std::ostringstream os;
        os << "perimeter_layer must be in [0, L); got "
           << perimeter_layer_ << " with L=" << L_ << ".";
        throw std::invalid_argument(os.str());
    }
    if (!(layer_pitch_um_ > 0.0)) {
        std::ostringstream os;
        os << "layer_pitch_um must be > 0; got " << layer_pitch_um_ << ".";
        throw std::invalid_argument(os.str());
    }

    // Complete graph K_k in u<v row-major order.
    allEdges_.reserve(static_cast<std::size_t>(k_) * (k_ - 1) / 2);
    for (int i = 0; i < k_; ++i) {
        for (int j = i + 1; j < k_; ++j) {
            allEdges_.emplace_back(i, j, EdgeData{});
        }
    }
    _G_Planar_ = allEdges_;
    // M6 (P0 fix): planar reference stamping is **lazy** — deferred to
    // `ensure_topology_()` so the O(E²) geometric pass is covered by
    // the `initial_crossing_count` PhaseTimer (which main.cpp wraps
    // around `build_crossings_index()`). This mirrors Python
    // `core.py::_ensure_crossings_ready` which stamps `_G_Planar.crossings`
    // inside the same lazy entry point as the topology cache. Pre-fix
    // the constructor paid the O(E²) cost regardless of whether the
    // caller ever invoked the optimizer.
}

SiNInterconnectionGraph::~SiNInterconnectionGraph() = default;

void SiNInterconnectionGraph::ensure_topology_() {
    if (topology_) return;
    topology_ = std::make_unique<CrossingTopology>(k_, positions_);

    // Stamp the planar reference (every edge on its default layer=0
    // baseline) using the topology cache to count per-edge crossings
    // without re-running geometry. Mirrors Python
    // `core.py::_ensure_crossings_ready` planar stamping. `_G_Planar_`
    // edges retain their default `layer=0`; per-edge loss uses the
    // taper formula `2 * |0 - ecl| * loss_taper` so the "all-on-layer-0
    // baseline" makes sense even when ecl != 0 (matches Python where
    // `_G_Planar` keeps `nx.set_edge_attributes(graph, 0, "layer")`).
    const auto& pairs = topology_->crossing_pairs();
    std::vector<int> counts(static_cast<std::size_t>(_G_Planar_.size()), 0);
    for (const auto& pr : pairs) {
        ++counts[pr.first];
        ++counts[pr.second];
    }
    for (std::size_t i = 0; i < _G_Planar_.size(); ++i) {
        std::get<2>(_G_Planar_[i]).crossings = counts[i];
    }
    compute_per_edge_loss(_G_Planar_, loss_crossing_, loss_taper_,
                          edge_coupler_layer_);
}

void SiNInterconnectionGraph::build_crossings_index() {
    ensure_topology_();
}

void SiNInterconnectionGraph::set_edge_layer_(int u, int v, int layer) {
    if (u > v) std::swap(u, v);
    for (auto& e : allEdges_) {
        if (std::get<0>(e) == u && std::get<1>(e) == v) {
            std::get<2>(e).layer = layer;
            return;
        }
    }
}

int SiNInterconnectionGraph::k_base_no_carry_add_(
    std::initializer_list<int> args) const {
    int sum = 0;
    for (auto x : args) sum += x;
    return sum % k_;
}

int SiNInterconnectionGraph::k_base_no_carry_sub_(
    std::initializer_list<int> args) const {
    auto it = args.begin();
    int res = *it++;
    for (; it != args.end(); ++it) {
        res = (res - *it) % k_;
    }
    if (res < 0) res += k_;
    return res;
}

void SiNInterconnectionGraph::seed_from_routing(
    const std::vector<int>& takeaway, int layer) {
    // M6: default `layer = (ecl + 1) % L` (mirrors Python optimizer
    // seed_layer; for L=2/ecl=0 yields layer 1, preserving legacy
    // behavior). Caller may override to any [0, L) index.
    if (layer < 0) layer = (edge_coupler_layer_ + 1) % L_;
    if (layer < 0 || layer >= L_) {
        std::ostringstream os;
        os << "seed_from_routing: layer must be in [0, L); got " << layer
           << " with L=" << L_ << ".";
        throw std::invalid_argument(os.str());
    }
    for (auto m : takeaway) {
        for (int i = 0; i < k_; ++i) {
            const int p1 = k_base_no_carry_add_({i, m, 1});
            const int p2 = k_base_no_carry_sub_({i, m, 1});
            set_edge_layer_(i, p1, layer);
            set_edge_layer_(i, p2, layer);
        }
    }
}

void SiNInterconnectionGraph::apply_optimization_result(
    const std::vector<int>& layers) {
    // Length guard — checked BEFORE any mutation so a wrong-length
    // call cannot half-apply a layer assignment (fail fast, no side
    // effects). This matches the sibling §1-3-c optimizer-surface
    // function `loss_function` (same `std::invalid_argument` type and
    // message idiom). Pre-fix this path silently zero-filled a
    // too-short vector and ignored a too-long tail — a direct-API
    // divergence from `loss_function` (Codex P2 post-79f5e79). Not a
    // production regression: both fixed-layers entry points reject
    // wrong lengths upstream (`main.cpp` SINIC_FIXED_LAYERS_JSON +
    // Python `api.py::run_optimization`), and the optimizer always
    // emits exactly `num_edges()` (`decode_layer_index`). Deliberate
    // divergence from Python `_apply_layer_assignment`'s accidental
    // `zip()` truncation, which is itself unreachable behind
    // `api.py`'s upstream length check — see REFACTOR_GOALS.md §1-2
    // 设计取舍 #8. The no-clamp/raw-value contract (取舍 #4 / P2-O5)
    // is orthogonal and unchanged: in-range and out-of-range *values*
    // of a correctly-sized vector still flow through raw.
    if (layers.size() != allEdges_.size()) {
        std::ostringstream os;
        os << "apply_optimization_result: layers has size "
           << layers.size() << ", expected " << allEdges_.size() << ".";
        throw std::invalid_argument(os.str());
    }
    // Reset all per-edge accumulators (M6 mirror of Python's
    // `nx.set_edge_attributes(self.G, 0, "<field>")` block) so callers
    // reading `allEdges_` directly between `apply_optimization_result`
    // and the next create/loss call see a clean post-optimize snapshot.
    for (auto& e : allEdges_) {
        auto& ed = std::get<2>(e);
        ed.layer                    = 0;
        ed.crossings                = 0;
        ed.loss                     = 0.0;
        ed.interlayerCrossings      = 0;
        ed.interlayerCrossingsAbove = 0;
        ed.interlayerCrossingsBelow = 0;
    }
    for (std::size_t i = 0; i < allEdges_.size(); ++i) {
        auto& e = allEdges_[i];
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        const int diff = std::abs(u - v);
        if (diff == 1 || diff == (k_ - 1)) {
            std::get<2>(e).layer = perimeter_layer_;
        } else {
            // Non-perimeter values are taken AS-IS (no clamp to
            // [0, L-1]) — same raw-value contract as `loss_function`
            // (see the no-clamp note there) and Python
            // `core.py::_apply_layer_assignment` (`set_edge_layer(edge,
            // int(layer))`, no clamp). The original clamp was a
            // Python↔C++ parity footgun: on the fixed-layers path
            // (`SINIC_FIXED_LAYERS_JSON` / parity `fixed_layers`) it
            // folded out-of-range inputs into [0, L-1] so they entered
            // a per-layer bucket, while Python kept them raw and
            // `create_subgraphs` dropped them from every bucket —
            // silent `subgraphsdata.json` drift (REFACTOR_GOALS.md
            // §1-2-b; §1-2 design-tradeoff #4 rejected the identical
            // clamp in `loss_function`, and the M8 dual-review item
            // Opus P2-O5 — deferred there — records this sibling clamp
            // as RESOLVED). Out-of-range edges flow through unchanged:
            // `create_subgraphs` (here and in Python) buckets only
            // layers in [0, L) so they drop from all per-layer
            // subgraphs identically while remaining in the planar
            // `complete_graph`. The optimizer hot path never feeds
            // out-of-range values (`decode_layer_index` already pins
            // to [0, L-1]); direct callers get Python-equivalent
            // behavior on raw input. The top-of-function length guard
            // makes `i < layers.size()` invariant here → unconditional
            // raw passthrough (the old `: 0` zero-fill fallback was the
            // silently-degrading footgun removed in §1-2 取舍 #8).
            std::get<2>(e).layer = layers[i];
        }
    }
}

double SiNInterconnectionGraph::loss_function(const std::vector<int>& layers) {
    // Phase B (REFACTOR_GOALS.md §2-1 mirror): integer-only
    // classification against the cached crossing-pair topology.
    // Bit-equivalent (modulo float summation order) with the pre-M2
    // create_subgraphs-based implementation for L=2 / ecl=0 /
    // perimeter=0; see §2-3 验收第 1 条.
    ensure_topology_();
    const int n_edges = topology_->num_edges();
    if (static_cast<int>(layers.size()) != n_edges) {
        std::ostringstream os;
        os << "loss_function: layers has size " << layers.size()
           << ", expected " << n_edges << ".";
        throw std::invalid_argument(os.str());
    }

    // Materialize the post-pin layer vector. Perimeter edges (|u-v| ∈
    // {1, k-1}) are forced to `perimeter_layer_` regardless of the
    // input. Non-perimeter values are taken AS-IS (no clamp to
    // [0, L-1]) — Python `core.py::loss_function` only rounds, and
    // the M8 PythonParityFixture compares both sides at the boundary.
    // Out-of-range values still flow through the taper formula
    // (`taper_hops = |layer - ecl|`) and the `unique(layers)` selection
    // rule, matching Python behavior. Callers should treat raw input
    // in [0, L-1]; the optimizer's decode already ensures this.
    std::vector<int> pinned(static_cast<std::size_t>(n_edges));
    const auto& pmask = topology_->perimeter_mask();
    for (int i = 0; i < n_edges; ++i) {
        pinned[i] = pmask[i] ? perimeter_layer_ : layers[i];
    }

    // Classification + bincount accumulation. For each crossing pair
    // (i, j) we read `pinned[i]` / `pinned[j]`:
    //   - same layer (intra) → both edges' intra-counter += 1
    //   - layer diff == 1 (adjacent inter) → both edges' inter-counter += 1
    //   - layer diff >= 2 (M6 §2-3 目标 D) → no contribution (non-adjacent
    //     evanescent coupling is below the -60 dB floor)
    std::vector<int> intra(static_cast<std::size_t>(n_edges), 0);
    std::vector<int> inter(static_cast<std::size_t>(n_edges), 0);
    for (const auto& pr : topology_->crossing_pairs()) {
        const int i = pr.first;
        const int j = pr.second;
        const int la = pinned[i];
        const int lb = pinned[j];
        if (la == lb) {
            ++intra[i];
            ++intra[j];
        } else if (std::llabs(static_cast<long long>(la)
                            - static_cast<long long>(lb)) == 1) {
            // int64 subtraction: post-P2-O5 raw layer values are
            // unclamped, so a pathological accepted `int` (INT_MIN /
            // INT_MAX from a direct/fixed-layers caller) would overflow
            // a signed `int` `la - lb` (UB; wrap folds extreme-opposite
            // to |Δ|==1, spuriously "adjacent"). Bit-identical for all
            // in-range / large-but-safe values. Codex P2 post-79f5e79;
            // REFACTOR_GOALS.md §1-2 设计取舍 #8.
            ++inter[i];
            ++inter[j];
        }
    }

    // Per-edge loss formula (mirrors Python `loss_function`):
    //   per_edge = 2 * loss_crossing * intra
    //            + 2 * |pinned - edge_coupler_layer| * loss_taper
    //            + loss_interlayer_crossing * inter
    std::vector<double> per_edge_loss(static_cast<std::size_t>(n_edges));
    for (int i = 0; i < n_edges; ++i) {
        // int64 |pinned - ecl|: same overflow/UB rationale as the
        // adjacency site above (std::abs(INT_MIN - ecl) is UB).
        // Bit-identical for in-range / large-but-safe values.
        const long long taper_hops = std::llabs(
            static_cast<long long>(pinned[i])
          - static_cast<long long>(edge_coupler_layer_));
        per_edge_loss[i] =
            2.0 * loss_crossing_ * intra[i]
          + 2.0 * static_cast<double>(taper_hops) * loss_taper_
          + loss_interlayer_crossing_ * inter[i];
    }

    // Aggregation: mean over edges whose post-pin layer is in
    // `unique(layers)`. Mirrors Python `np.unique(layers_int)` +
    // `np.isin(pinned, unique_in)`. For L=2 / ecl=0 / perimeter=0
    // this is bit-equivalent to the pre-M2 "iterate over occupied
    // sub_G entries" aggregation.
    std::set<int> unique_in(layers.begin(), layers.end());
    std::vector<double> selected;
    selected.reserve(static_cast<std::size_t>(n_edges));
    for (int i = 0; i < n_edges; ++i) {
        if (unique_in.find(pinned[i]) != unique_in.end()) {
            selected.push_back(per_edge_loss[i]);
        }
    }

    if (selected.empty()) return 0.0;

    double sum = 0.0;
    for (auto v : selected) sum += v;
    const double mean = sum / static_cast<double>(selected.size());

    double var = 0.0;
    for (auto v : selected) {
        const double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(selected.size());

    return alpha_ * mean + beta_ * var;
}

void SiNInterconnectionGraph::create_subgraphs() {
    // M6 mirror of Python `core.py::create_subgraphs` (§2-3 目标 A):
    //   1. sub_G has exactly L entries (one per layer index, including
    //      empty layers)
    //   2. each entry gets intra-layer crossings + per-edge loss stamped
    //   3. every adjacent pair (i, i+1) gets interlayer crossings with
    //      the `count_interlayer_crossings` accounting (the lower edge
    //      sees one "above", the upper edge one "below"; middle layers
    //      accumulate from both sides)
    //   4. apply_interlayer_loss for each layer at the end
    //
    // Crossings come from the cached `CrossingTopology` — the relation
    // `loss_function` scores — so the stamped counts always match what
    // the optimizer saw, on every layout, without an O(E²) geometry pass
    // per layer. Wherever `edge_crosses` is exact (everything but float
    // round-off on collinear side runs) they equal the old per-layer
    // `count_crossings_with_detail` / `count_interlayer_crossings`.
    //
    // M6 stage-2 review fix (codex P1-1): `ensure_topology_()` also
    // stamps `_G_Planar_` before any downstream consumer (e.g. the JSON
    // writer emitting `complete_graph` +
    // `loss_analysis.avg_loss_completegraph`) reads it.
    ensure_topology_();

    // `allEdges_` shares the topology's row-major edge order (see the
    // `_G_Planar_` stamping in `ensure_topology_()`).
    const std::size_t n = allEdges_.size();
    std::vector<int> intra(n, 0);
    std::vector<int> above(n, 0);
    std::vector<int> below(n, 0);
    for (const auto& pr : topology_->crossing_pairs()) {
        const auto i = static_cast<std::size_t>(pr.first);
        const auto j = static_cast<std::size_t>(pr.second);
        const int la = std::get<2>(allEdges_[i]).layer;
        const int lb = std::get<2>(allEdges_[j]).layer;
        if (la == lb) {
            if (la >= 0 && la < L_) {
                ++intra[i];
                ++intra[j];
            }
        } else if (std::llabs(static_cast<long long>(la)
                              - static_cast<long long>(lb)) == 1
                   && std::min(la, lb) >= 0 && std::max(la, lb) < L_) {
            ++above[la < lb ? i : j];
            ++below[la < lb ? j : i];
        }
    }

    sub_G_.clear();
    sub_G_.resize(static_cast<std::size_t>(L_));

    for (int layer = 0; layer < L_; ++layer) {
        EdgeList& bucket = sub_G_[static_cast<std::size_t>(layer)];
        for (std::size_t i = 0; i < n; ++i) {
            if (std::get<2>(allEdges_[i]).layer != layer) continue;
            bucket.push_back(allEdges_[i]);
            auto& ed = std::get<2>(bucket.back());
            ed.crossings                = intra[i];
            ed.interlayerCrossings      = above[i] + below[i];
            ed.interlayerCrossingsAbove = above[i];
            ed.interlayerCrossingsBelow = below[i];
        }
        if (!bucket.empty()) {
            compute_per_edge_loss(bucket, loss_crossing_, loss_taper_,
                                  edge_coupler_layer_);
        }
    }

    // Layer-by-layer apply_interlayer_loss after all neighbour
    // accumulation is done.
    for (auto& bucket : sub_G_) {
        if (!bucket.empty()) {
            apply_interlayer_loss(bucket, loss_interlayer_crossing_);
        }
    }
}

namespace {

struct LayerStats {
    std::size_t count   = 0;
    double      avg     = 0.0;
    double      var     = 0.0;
    double      stdev   = 0.0;
    double      range   = 0.0;
};

LayerStats compute_stats(const std::vector<double>& vals) {
    LayerStats s;
    s.count = vals.size();
    if (vals.empty()) return s;
    double sum = 0.0;
    for (auto v : vals) sum += v;
    s.avg = sum / vals.size();
    double var = 0.0;
    double minv = vals.front();
    double maxv = vals.front();
    for (auto v : vals) {
        const double d = v - s.avg;
        var += d * d;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }
    s.var   = var / vals.size();
    s.stdev = std::sqrt(s.var);
    s.range = maxv - minv;
    return s;
}

} // namespace

void SiNInterconnectionGraph::analyze_loss(std::ostream& out) {
    // Make sure the planar reference is stamped (mirror of Python's
    // `_ensure_crossings_ready` call at the top of `analyze_loss`).
    ensure_topology_();
    sub_G_.clear();
    create_subgraphs();
    out << "============ Loss Analysis ============\n";
    std::vector<double> all_vals;
    for (std::size_t i = 0; i < sub_G_.size(); ++i) {
        std::vector<double> vals;
        for (const auto& e : sub_G_[i]) {
            vals.push_back(std::get<2>(e).loss);
        }
        if (vals.empty()) {
            out << "Layer " << i << " => (empty)\n";
            continue;
        }
        const auto s = compute_stats(vals);
        out << "Layer " << i
            << " => count=" << s.count
            << ", avg="   << s.avg
            << ", var="   << s.var
            << ", std="   << s.stdev
            << ", range=" << s.range << "\n";
        for (auto v : vals) all_vals.push_back(v);
    }
    if (!all_vals.empty()) {
        const auto s = compute_stats(all_vals);
        out << "[Flattened layers] => count=" << s.count
            << ", avg="   << s.avg
            << ", var="   << s.var
            << ", std="   << s.stdev
            << ", range=" << s.range << "\n";
    }

    std::vector<double> planar_vals;
    for (const auto& e : _G_Planar_) {
        planar_vals.push_back(std::get<2>(e).loss);
    }
    if (!planar_vals.empty()) {
        const auto s = compute_stats(planar_vals);
        out << "[Planar Graph] => avg=" << s.avg
            << ", var="   << s.var
            << ", std="   << s.stdev
            << ", range=" << s.range << "\n";
    }
}

} // namespace sinic
