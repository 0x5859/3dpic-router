#include "sinic/crosstalk.hpp"

#include "sinic/crossings_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sinic {

namespace {

// `(t_self, t_other)` parametric intersection for one crossing pair.
struct TParams {
    double t_self;
    double t_other;
};

// Per-edge sorted entry: (other_edge_idx, t_self_on_this_edge,
// t_on_other_edge). Sorted ascending by `t_self` so direction-aware
// slicing in `_crossings_ahead` is a single `lower_bound` / reverse
// iteration.
struct AheadEntry {
    int    other;
    double t_self;
    double t_other;
};

// Allocate an all-null k×k×k nested JSON array (matches Python's
// `_empty_values`).
nlohmann::json empty_values(int k) {
    nlohmann::json outer = nlohmann::json::array();
    for (int i = 0; i < k; ++i) {
        nlohmann::json mid = nlohmann::json::array();
        for (int j = 0; j < k; ++j) {
            nlohmann::json inner = nlohmann::json::array();
            for (int t = 0; t < k; ++t) {
                inner.push_back(nullptr);
            }
            mid.push_back(std::move(inner));
        }
        outer.push_back(std::move(mid));
    }
    return outer;
}

// Closed-form parametric intersection for one (edge_i, edge_j) pair.
// Mirrors Python `_compute_t_parameters`:
//   det     = (q_a - p_a) × (q_b - p_b)
//   t_self  = ((p_b - p_a) × (q_b - p_b)) / det
//   t_other = ((p_b - p_a) × (q_a - p_a)) / det
// where `×` is the scalar cross product. `det == 0` (parallel /
// collinear) falls back to (0.5, 0.5) — deterministic, finite, and
// only hit on degenerate inputs (SiN cyclic-convex layouts always
// produce proper intersections). Output clamped to [0, 1] for
// robustness against floating-point drift right at endpoints.
TParams compute_t_pair(const Point& pa, const Point& qa,
                       const Point& pb, const Point& qb) noexcept {
    const double dax = qa.first  - pa.first;
    const double day = qa.second - pa.second;
    const double dbx = qb.first  - pb.first;
    const double dby = qb.second - pb.second;
    const double rx  = pb.first  - pa.first;
    const double ry  = pb.second - pa.second;

    const double det   = dax * dby - day * dbx;
    const double num_t = rx  * dby - ry  * dbx;
    const double num_s = rx  * day - ry  * dax;

    double t_self  = 0.5;
    double t_other = 0.5;
    if (det != 0.0) {
        t_self  = num_t / det;
        t_other = num_s / det;
    }
    if (t_self  < 0.0) t_self  = 0.0;
    if (t_self  > 1.0) t_self  = 1.0;
    if (t_other < 0.0) t_other = 0.0;
    if (t_other > 1.0) t_other = 1.0;
    return {t_self, t_other};
}

// Stack frame for the iterative DFS. Independent `visited` set per
// frame so two sibling spawns can't poison each other's history.
struct WalkFrame {
    int             edge_idx;
    double          t_in;
    bool            going_pos;
    double          power;
    int             hop;
    std::set<int>   visited;
};

// DFS over the branch tree rooted at the main path `s_node → d_node`.
// Returns a `{arrival_node: accumulated_linear_power}` map; the caller
// converts above-threshold arrivals to dB and folds in the diagonal
// mask.
std::map<int, double> walk_source_destination(
    int s_node, int d_node,
    const std::vector<std::pair<int, int>>& edge_list,
    const std::vector<int>& layer_per_edge,
    const std::vector<std::vector<AheadEntry>>& crossings_per_edge,
    int main_idx,
    double intra_xt, double inter_xt,
    double one_minus_lc,
    int max_hops, double threshold) {

    const int main_u = edge_list[main_idx].first;
    const int main_v = edge_list[main_idx].second;
    const bool initial_going_pos = (s_node == main_u);
    const double initial_t = initial_going_pos ? 0.0 : 1.0;

    std::map<int, double> arrivals;

    std::vector<WalkFrame> stack;
    stack.reserve(64);
    {
        WalkFrame root{main_idx, initial_t, initial_going_pos, 1.0, 0, {}};
        root.visited.insert(main_idx);
        stack.push_back(std::move(root));
    }

    while (!stack.empty()) {
        WalkFrame frame = std::move(stack.back());
        stack.pop_back();

        if (frame.power < threshold) continue;

        const int edge_idx = frame.edge_idx;
        const bool going_pos = frame.going_pos;
        const double t_in = frame.t_in;
        double current_power = frame.power;
        const int hop = frame.hop;

        const int u = edge_list[edge_idx].first;
        const int v = edge_list[edge_idx].second;
        const int end_node = going_pos ? v : u;

        const auto& sorted_table = crossings_per_edge[edge_idx];

        // Slice the sorted-ascending list to entries lying ahead of
        // `t_in` in the chosen traversal direction. Strict inequality
        // so a branch spawned at the current crossing doesn't see
        // itself again at `t_self == t_in` (mirror of Python
        // `_crossings_ahead`).
        if (going_pos) {
            for (std::size_t k_idx = 0; k_idx < sorted_table.size(); ++k_idx) {
                const auto& row = sorted_table[k_idx];
                if (!(row.t_self > t_in)) continue;
                if (current_power < threshold) break;

                const int other_edge_idx = row.other;
                const double t_other     = row.t_other;
                const int la = layer_per_edge[edge_idx];
                const int lb = layer_per_edge[other_edge_idx];
                double coef = 0.0;
                if (la == lb) coef = intra_xt;
                // int64 subtraction: post-P2-O5 raw layer values are
                // unclamped, so a pathological accepted `int` (INT_MIN
                // / INT_MAX) would overflow signed `int` `la - lb` (UB;
                // wrap folds extreme-opposite to |Δ|==1, spuriously
                // "adjacent"). Bit-identical for in-range / large-safe
                // values. Codex P2 post-79f5e79; REFACTOR_GOALS.md §1-2
                // 设计取舍 #8.
                else if (std::llabs(static_cast<long long>(la)
                                  - static_cast<long long>(lb)) == 1)
                    coef = inter_xt;
                // else: |Δlayer| ≥ 2 → 0.0 (§2-3 目标 D)

                if (coef > 0.0 && hop < max_hops
                    && frame.visited.find(other_edge_idx) == frame.visited.end()) {
                    const double per_dir = (current_power * coef) * 0.5;
                    if (per_dir >= threshold) {
                        std::set<int> new_visited = frame.visited;
                        new_visited.insert(other_edge_idx);
                        // Forward on other_edge: t increasing
                        stack.push_back(WalkFrame{
                            other_edge_idx, t_other, true,
                            per_dir, hop + 1, new_visited});
                        // Backward: t decreasing
                        stack.push_back(WalkFrame{
                            other_edge_idx, t_other, false,
                            per_dir, hop + 1, std::move(new_visited)});
                    }
                }

                current_power *= one_minus_lc;
            }
        } else {
            // going_neg: iterate sorted_table in reverse, keeping only
            // entries with `t_self < t_in`.
            for (std::size_t k_idx = sorted_table.size(); k_idx > 0; --k_idx) {
                const auto& row = sorted_table[k_idx - 1];
                if (!(row.t_self < t_in)) continue;
                if (current_power < threshold) break;

                const int other_edge_idx = row.other;
                const double t_other     = row.t_other;
                const int la = layer_per_edge[edge_idx];
                const int lb = layer_per_edge[other_edge_idx];
                double coef = 0.0;
                if (la == lb) coef = intra_xt;
                // int64 subtraction — see going_pos branch above
                // (overflow/UB rationale; §1-2 设计取舍 #8).
                else if (std::llabs(static_cast<long long>(la)
                                  - static_cast<long long>(lb)) == 1)
                    coef = inter_xt;

                if (coef > 0.0 && hop < max_hops
                    && frame.visited.find(other_edge_idx) == frame.visited.end()) {
                    const double per_dir = (current_power * coef) * 0.5;
                    if (per_dir >= threshold) {
                        std::set<int> new_visited = frame.visited;
                        new_visited.insert(other_edge_idx);
                        stack.push_back(WalkFrame{
                            other_edge_idx, t_other, true,
                            per_dir, hop + 1, new_visited});
                        stack.push_back(WalkFrame{
                            other_edge_idx, t_other, false,
                            per_dir, hop + 1, std::move(new_visited)});
                    }
                }

                current_power *= one_minus_lc;
            }
        }

        if (current_power >= threshold) {
            arrivals[end_node] += current_power;
        }
    }

    return arrivals;
}

} // namespace

nlohmann::json compute_crosstalk_tensor(
    SiNInterconnectionGraph& graph,
    int max_hops,
    double threshold_db) {

    // ---- entry guards (mirror Python's eager-raise discipline) ----
    if (max_hops < 0) {
        std::ostringstream os;
        os << "max_hops must be >= 0; got " << max_hops << ".";
        throw std::invalid_argument(os.str());
    }
    if (!std::isfinite(threshold_db)) {
        std::ostringstream os;
        os << "threshold_db must be finite; got " << threshold_db << ".";
        throw std::invalid_argument(os.str());
    }

    const double intra_xt = graph.loss_intralayer_crosstalk()
        ? *graph.loss_intralayer_crosstalk() : 0.0;
    const double inter_xt = graph.loss_interlayer_crosstalk()
        ? *graph.loss_interlayer_crosstalk() : 0.0;
    const double one_minus_lc = 1.0 - graph.loss_crossing();
    if (!(one_minus_lc > 0.0 && one_minus_lc <= 1.0)) {
        std::ostringstream os;
        os << "loss_crossing=" << graph.loss_crossing()
           << " produces transmission " << one_minus_lc
           << " outside (0, 1]. The crosstalk model in REFACTOR_GOALS.md "
              "§2-2 treats loss_crossing as a fractional power loss "
              "(1 − T); use a value in [0, 1).";
        throw std::invalid_argument(os.str());
    }

    const int k = graph.k();

    auto make_meta_payload = [&]() {
        nlohmann::json meta = nlohmann::json::object();
        meta["unit"]                = "dB";
        meta["shape"]               = nlohmann::json::array({k, k, k});
        meta["coherence_model"]     = graph.coherence_model();
        meta["polarization"]        = graph.polarization();
        meta["symmetric"]           = false;
        meta["diagonal_convention"] = "NaN";
        meta["max_hops"]            = max_hops;
        meta["threshold_db"]        = threshold_db;
        if (graph.loss_intralayer_crosstalk()) {
            meta["loss_intralayer_crosstalk"] =
                *graph.loss_intralayer_crosstalk();
        } else {
            meta["loss_intralayer_crosstalk"] = nullptr;
        }
        if (graph.loss_interlayer_crosstalk()) {
            meta["loss_interlayer_crosstalk"] =
                *graph.loss_interlayer_crosstalk();
        } else {
            meta["loss_interlayer_crosstalk"] = nullptr;
        }
        return meta;
    };

    // ---- short-circuits (zero coefs / max_hops=0) -----------------
    // Mirror Python codex-P2 fix: defer Phase A geometric pass until
    // after these early-exit guards so direct callers of the engine
    // (non-orchestrator paths) don't pay the O(E²) one-shot cost.
    if (intra_xt == 0.0 && inter_xt == 0.0) {
        auto payload = make_meta_payload();
        payload["values"] = empty_values(k);
        return payload;
    }
    if (max_hops == 0) {
        auto payload = make_meta_payload();
        payload["values"] = empty_values(k);
        return payload;
    }

    // ---- commit to Phase A geometric pass + per-pair t-parameters --
    graph.build_crossings_index();
    const CrossingTopology* topology = graph.topology();
    if (topology == nullptr) {
        throw std::runtime_error(
            "compute_crosstalk_tensor: graph.build_crossings_index() did "
            "not produce a topology cache.");
    }

    const auto& edge_list      = topology->edge_list();
    const auto& crossing_pairs = topology->crossing_pairs();
    const int n_edges          = static_cast<int>(edge_list.size());

    // Per-edge layer vector (post-pin, from
    // apply_optimization_result). Edge i corresponds to allEdges_[i]
    // because both are built in u<v row-major order.
    const auto& all_edges = graph.all_edges();
    std::vector<int> layer_per_edge(static_cast<std::size_t>(n_edges), 0);
    for (int i = 0; i < n_edges; ++i) {
        layer_per_edge[i] = std::get<2>(all_edges[i]).layer;
    }

    // Edge → row-major (u, v) index map, mirroring Python's
    // `graph._edge_index`. Used to look up the canonical edge index
    // for an (s, d) main path.
    std::map<std::pair<int, int>, int> edge_index;
    for (int i = 0; i < n_edges; ++i) {
        edge_index.emplace(edge_list[i], i);
    }

    // Per-edge sorted (other_edge_idx, t_self, t_other) table.
    // Sorted ascending by t_self so direction-aware slicing in the
    // DFS is a single forward-or-reverse pass with strict inequality.
    std::vector<std::vector<AheadEntry>> crossings_per_edge(
        static_cast<std::size_t>(n_edges));
    {
        const PositionMap& positions = graph.positions();
        for (const auto& pr : crossing_pairs) {
            const int ei = pr.first;
            const int ej = pr.second;
            const auto& ai = edge_list[ei];
            const auto& aj = edge_list[ej];
            const Point& pai = positions.at(ai.first);
            const Point& qai = positions.at(ai.second);
            const Point& paj = positions.at(aj.first);
            const Point& qaj = positions.at(aj.second);
            const TParams tp = compute_t_pair(pai, qai, paj, qaj);
            crossings_per_edge[ei].push_back({ej, tp.t_self, tp.t_other});
            crossings_per_edge[ej].push_back({ei, tp.t_other, tp.t_self});
        }
        for (auto& row : crossings_per_edge) {
            // M7 dual-review R2 P1 (codex): use stable_sort so equal-
            // `t_self` ties keep insertion order. Python's `list.sort`
            // is stable; in SiN cyclic-convex layouts ties are
            // geometrically unlikely (would need 3+ edges through the
            // same point), but boundary-clamped fallbacks plus the
            // M8 PythonParityFixture make any drift here a parity
            // hazard. stable_sort costs at most a constant factor.
            std::stable_sort(row.begin(), row.end(),
                             [](const AheadEntry& a, const AheadEntry& b) {
                                 return a.t_self < b.t_self;
                             });
        }
    }

    const double threshold = std::pow(10.0, threshold_db / 10.0);

    nlohmann::json values = empty_values(k);

    for (int s = 0; s < k; ++s) {
        for (int d = 0; d < k; ++d) {
            if (s == d) continue;

            const int main_u = std::min(s, d);
            const int main_v = std::max(s, d);
            const auto it = edge_index.find(std::make_pair(main_u, main_v));
            if (it == edge_index.end()) continue;
            const int main_idx = it->second;

            const auto arrivals = walk_source_destination(
                s, d, edge_list, layer_per_edge, crossings_per_edge,
                main_idx, intra_xt, inter_xt, one_minus_lc,
                max_hops, threshold);

            for (const auto& kv : arrivals) {
                const int t_node = kv.first;
                const double power = kv.second;
                if (t_node == s || t_node == d) continue; // diagonal
                if (power < threshold) continue;
                values[s][d][t_node] = 10.0 * std::log10(power);
            }
        }
    }

    auto payload = make_meta_payload();
    payload["values"] = std::move(values);
    return payload;
}

} // namespace sinic
