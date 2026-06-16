#include "sinic/crossings_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <utility>

namespace sinic {

namespace {

// Sign helper: −1 / 0 / +1 with strict zero (we rely on exact
// zero for collinear / T-junction branches below).
inline int sign(double v) noexcept {
    if (v > 0.0) return 1;
    if (v < 0.0) return -1;
    return 0;
}

// 2D scalar cross product: positive when ABC turns left, negative
// right, zero when collinear.
inline double cross2(double ax, double ay, double bx, double by) noexcept {
    return ax * by - ay * bx;
}

} // namespace

bool CrossingTopology::is_cyclic_convex(int k, const PositionMap& positions) {
    // Degenerate sizes are trivially "convex" (no chord pair exists).
    if (k < 3) return true;

    // Pull positions into a tight contiguous buffer; missing key →
    // can't claim cyclic-convex, fall back to geometric path.
    std::vector<Point> pts;
    pts.reserve(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        auto it = positions.find(i);
        if (it == positions.end()) return false;
        pts.push_back(it->second);
    }

    // Reject duplicate positions — orient(P, P, R) is identically zero
    // and the alternating-endpoint test relies on each vertex labeling
    // a geometrically distinct point.
    //
    // M6 stage-review P1 fix: bit-cast both halves into **unsigned** 64-bit
    // ints (avoids undefined behavior on signed left-shift for negative
    // coordinates) and store the (kx, ky) PAIR in a std::set keyed on the
    // pair (avoids hash collisions in a 64-bit combiner — a collision
    // would silently fall back to the geometric path, costing perf on a
    // valid cyclic-convex layout). Bit-cast equality treats `-0.0` and
    // `+0.0` as distinct (Python `set` treats them equal); SiN layouts
    // from `positions.cpp` never produce both, so this is documented as
    // a known edge-case divergence.
    {
        std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
        for (const auto& p : pts) {
            const double x = p.first;
            const double y = p.second;
            std::uint64_t kx = 0;
            std::uint64_t ky = 0;
            std::memcpy(&kx, &x, sizeof(kx));
            std::memcpy(&ky, &y, sizeof(ky));
            if (!seen.insert(std::make_pair(kx, ky)).second) return false;
        }
    }

    // Compute exterior turn angles around the polygon. For a simple
    // convex polygon, all non-zero `cross` values share a sign AND
    // the signed sum of turn angles equals ±2π. Pentagram-style
    // winding-2 layouts have all-same-sign cross but |Σ| ≈ 4π → we
    // detect via the angle sum.
    bool any_nz = false;
    bool seen_pos = false;
    bool seen_neg = false;
    double total_turn = 0.0;

    for (int i = 0; i < k; ++i) {
        const auto& p0 = pts[i];
        const auto& p1 = pts[(i + 1) % k];
        const auto& p2 = pts[(i + 2) % k];

        const double e1x = p1.first  - p0.first;
        const double e1y = p1.second - p0.second;
        const double e2x = p2.first  - p1.first;
        const double e2y = p2.second - p1.second;

        const double c = cross2(e1x, e1y, e2x, e2y);
        const double d = e1x * e2x + e1y * e2y;

        if (c != 0.0) {
            any_nz = true;
            if (c > 0.0) seen_pos = true;
            else         seen_neg = true;
        }
        total_turn += std::atan2(c, d);
    }
    if (!any_nz) return false;
    if (seen_pos && seen_neg) return false;

    constexpr double TAU      = 2.0 * 3.14159265358979323846;
    constexpr double TURN_TOL = 1e-6;
    return std::abs(std::abs(total_turn) - TAU) < TURN_TOL;
}

void CrossingTopology::build_edge_list_(int k) {
    edge_list_.clear();
    edge_list_.reserve(static_cast<std::size_t>(k) * (k - 1) / 2);
    for (int u = 0; u < k; ++u) {
        for (int v = u + 1; v < k; ++v) {
            edge_list_.emplace_back(u, v);
        }
    }
    n_edges_ = static_cast<int>(edge_list_.size());
}

void CrossingTopology::build_perimeter_mask_(int k) {
    perimeter_mask_.assign(static_cast<std::size_t>(n_edges_), false);
    for (int i = 0; i < n_edges_; ++i) {
        const int u = edge_list_[i].first;
        const int v = edge_list_[i].second;
        const int diff = std::abs(u - v);
        if (diff == 1 || diff == (k - 1)) {
            perimeter_mask_[i] = true;
        }
    }
}

void CrossingTopology::build_cyclic_(int k) {
    // Fast path: alternating-endpoints test on a convex polygon.
    //
    // For K_k with node indices laid out cyclically around a convex
    // polygon (the SiN square/rectangle layouts), two chords (u_i, v_i)
    // and (u_j, v_j) cross iff exactly one of {u_j, v_j} lies strictly
    // inside the open integer interval (u_i, v_i) AND the chords do not
    // share a vertex. Purely combinatorial — no float arithmetic.
    //
    // The check below scans all (i, j > i) pairs in O(E²) integer work.
    // At k=160 (E=12 720) this is ~80 M cheap integer ops; well within
    // the §2-1 "≤ 5 s build at k=160" budget on a modern CPU.
    crossing_pairs_.clear();
    crossing_pairs_.reserve(static_cast<std::size_t>(n_edges_));

    for (int i = 0; i < n_edges_ - 1; ++i) {
        const int ui = edge_list_[i].first;
        const int vi = edge_list_[i].second;
        for (int j = i + 1; j < n_edges_; ++j) {
            const int uj = edge_list_[j].first;
            const int vj = edge_list_[j].second;
            const bool uj_in = (uj > ui) && (uj < vi);
            const bool vj_in = (vj > ui) && (vj < vi);
            const bool shared =
                (uj == ui) || (uj == vi) || (vj == ui) || (vj == vi);
            if ((uj_in ^ vj_in) && !shared) {
                crossing_pairs_.emplace_back(i, j);
            }
        }
    }
    (void)k; // k is implicit in edge_list_ ordering; kept for symmetry.
}

void CrossingTopology::build_geometric_(const PositionMap& positions) {
    // Fallback: vectorized orientation test with closed-form handling of
    // T-junctions and fully-collinear partial overlaps. Bit-exact with
    // `crossings.cpp::edge_crosses` (which delegates to
    // `geometry.cpp::segments_intersect_strict`) — and that in turn
    // matches Python `core.py::edge_crosses` shapely semantics.
    //
    // Per-i nested loop with O(E) work per i (E² total). For arbitrary
    // layouts at k=160 this is still tractable (~150 M float ops).
    crossing_pairs_.clear();
    crossing_pairs_.reserve(static_cast<std::size_t>(n_edges_));

    // Stage endpoint positions in contiguous arrays for tight loops.
    std::vector<double> p1x(static_cast<std::size_t>(n_edges_));
    std::vector<double> p1y(static_cast<std::size_t>(n_edges_));
    std::vector<double> p2x(static_cast<std::size_t>(n_edges_));
    std::vector<double> p2y(static_cast<std::size_t>(n_edges_));
    for (int i = 0; i < n_edges_; ++i) {
        const int u = edge_list_[i].first;
        const int v = edge_list_[i].second;
        const auto& pu = positions.at(u);
        const auto& pv = positions.at(v);
        p1x[i] = pu.first;  p1y[i] = pu.second;
        p2x[i] = pv.first;  p2y[i] = pv.second;
    }

    for (int i = 0; i < n_edges_ - 1; ++i) {
        const double a1x = p1x[i];
        const double a1y = p1y[i];
        const double a2x = p2x[i];
        const double a2y = p2y[i];
        const int iu = edge_list_[i].first;
        const int iv = edge_list_[i].second;
        const double ax = a2x - a1x;
        const double ay = a2y - a1y;
        const double denom_a = ax * ax + ay * ay;
        const double safe_a  = denom_a > 0.0 ? denom_a : 1.0;
        const double amin_x  = (a1x <= a2x) ? a1x : a2x;
        const double amax_x  = (a1x >= a2x) ? a1x : a2x;
        const double amin_y  = (a1y <= a2y) ? a1y : a2y;
        const double amax_y  = (a1y >= a2y) ? a1y : a2y;

        for (int j = i + 1; j < n_edges_; ++j) {
            const int ju = edge_list_[j].first;
            const int jv = edge_list_[j].second;
            if (ju == iu || ju == iv || jv == iu || jv == iv) continue;

            const double b1x = p1x[j];
            const double b1y = p1y[j];
            const double b2x = p2x[j];
            const double b2y = p2y[j];
            const double bx = b2x - b1x;
            const double by = b2y - b1y;

            // o1, o2 = orientation of A vs endpoints of B.
            const double o1 = ax * (b1y - a1y) - ay * (b1x - a1x);
            const double o2 = ax * (b2y - a1y) - ay * (b2x - a1x);
            // o3, o4 = orientation of B vs endpoints of A.
            const double o3 = bx * (a1y - b1y) - by * (a1x - b1x);
            const double o4 = bx * (a2y - b1y) - by * (a2x - b1x);
            const int s1 = sign(o1);
            const int s2 = sign(o2);
            const int s3 = sign(o3);
            const int s4 = sign(o4);

            const bool proper = (s1 * s2 < 0) && (s3 * s4 < 0);

            // T-junction branches: exactly one of {o1, o2} or {o3, o4}
            // is zero AND that zero-point lies inside the bounding box
            // of the other segment. Matches Python
            // `core.py::_build_crossing_pairs_geometric` T-junction
            // semantics block.
            const bool z1 = (s1 == 0);
            const bool z2 = (s2 == 0);
            const bool z3 = (s3 == 0);
            const bool z4 = (s4 == 0);

            const bool b1_on_a = (b1x >= amin_x) && (b1x <= amax_x)
                              && (b1y >= amin_y) && (b1y <= amax_y);
            const bool b2_on_a = (b2x >= amin_x) && (b2x <= amax_x)
                              && (b2y >= amin_y) && (b2y <= amax_y);
            const double bmin_x = (b1x <= b2x) ? b1x : b2x;
            const double bmax_x = (b1x >= b2x) ? b1x : b2x;
            const double bmin_y = (b1y <= b2y) ? b1y : b2y;
            const double bmax_y = (b1y >= b2y) ? b1y : b2y;
            const bool a1_on_b = (a1x >= bmin_x) && (a1x <= bmax_x)
                              && (a1y >= bmin_y) && (a1y <= bmax_y);
            const bool a2_on_b = (a2x >= bmin_x) && (a2x <= bmax_x)
                              && (a2y >= bmin_y) && (a2y <= bmax_y);

            const bool t_b1_only = z1 && !z2 && !z3 && !z4 && b1_on_a;
            const bool t_b2_only = !z1 && z2 && !z3 && !z4 && b2_on_a;
            const bool t_a1_only = !z1 && !z2 && z3 && !z4 && a1_on_b;
            const bool t_a2_only = !z1 && !z2 && !z3 && z4 && a2_on_b;

            // Fully collinear (all four orient = 0): partial overlap
            // returns true; full containment (one segment inside the
            // other) returns false — matches shapely-equivalent
            // semantics in Python.
            bool collinear_cross = false;
            if (z1 && z2 && z3 && z4) {
                const double denom_b = bx * bx + by * by;
                if (denom_a > 0.0 && denom_b > 0.0) {
                    const double t_b1 = ((b1x - a1x) * ax + (b1y - a1y) * ay) / safe_a;
                    const double t_b2 = ((b2x - a1x) * ax + (b2y - a1y) * ay) / safe_a;
                    const double safe_b = denom_b;
                    const double s_a1 = ((a1x - b1x) * bx + (a1y - b1y) * by) / safe_b;
                    const double s_a2 = ((a2x - b1x) * bx + (a2y - b1y) * by) / safe_b;
                    const bool both_b_in_a = (t_b1 >= 0.0) && (t_b1 <= 1.0)
                                          && (t_b2 >= 0.0) && (t_b2 <= 1.0);
                    const bool both_a_in_b = (s_a1 >= 0.0) && (s_a1 <= 1.0)
                                          && (s_a2 >= 0.0) && (s_a2 <= 1.0);
                    const double tb_min = (t_b1 <= t_b2) ? t_b1 : t_b2;
                    const double tb_max = (t_b1 >= t_b2) ? t_b1 : t_b2;
                    const bool overlap  = (tb_min < 1.0) && (tb_max > 0.0);
                    collinear_cross = overlap && !both_b_in_a && !both_a_in_b;
                }
            }

            if (proper || t_b1_only || t_b2_only || t_a1_only || t_a2_only
                || collinear_cross) {
                crossing_pairs_.emplace_back(i, j);
            }
        }
    }
}

CrossingTopology::CrossingTopology(int k, const PositionMap& positions) {
    build_edge_list_(k);
    build_perimeter_mask_(k);

    if (is_cyclic_convex(k, positions)) {
        used_cyclic_ = true;
        build_cyclic_(k);
    } else {
        used_cyclic_ = false;
        build_geometric_(positions);
    }
}

} // namespace sinic
