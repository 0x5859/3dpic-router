#include <catch2/catch.hpp>

#include "sinic/graph.hpp"
#include "sinic/positions.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

using sinic::SiNInterconnectionGraph;
using sinic::distribute_nodes_around_square;

TEST_CASE("graph constructor builds K_k with no missing edges", "[graph]") {
    const int nps = 3;
    const int k = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);

    REQUIRE(g.num_edges() == k * (k - 1) / 2);
    REQUIRE(g.k() == k);

    // All edges store u < v; iteration order is row-major.
    const auto& edges = g.all_edges();
    REQUIRE(edges.size() == static_cast<size_t>(g.num_edges()));
    for (const auto& e : edges) {
        REQUIRE(std::get<0>(e) < std::get<1>(e));
    }
}

TEST_CASE("loss_function — all-layer-0 vs all-layer-1 differs by taper", "[graph]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    // Set alpha=1, beta=0 so we can read mean-loss directly.
    SiNInterconnectionGraph g(k, pos, /*cl=*/0.3, /*tl=*/1.0, /*itl=*/0.006);
    g.set_loss_weights(1.0, 0.0);

    std::vector<int> all0(g.num_edges(), 0);
    std::vector<int> all1(g.num_edges(), 1);

    const double L0 = g.loss_function(all0);
    const double L1 = g.loss_function(all1);

    // With all edges in layer 1, every non-perimeter edge pays an extra
    // 2 * 1 * loss_taper. The total mean loss therefore must be HIGHER
    // for the layer-1 case (no interlayer crossings, but taper > 0).
    REQUIRE(L1 > L0);
}

TEST_CASE("perimeter pin is enforced regardless of layer assignment", "[graph]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);

    std::vector<int> all1(g.num_edges(), 1);
    g.apply_optimization_result(all1);

    for (const auto& e : g.all_edges()) {
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        const int diff = std::abs(u - v);
        if (diff == 1 || diff == (k - 1)) {
            REQUIRE(std::get<2>(e).layer == 0);
        } else {
            REQUIRE(std::get<2>(e).layer == 1);
        }
    }
}

TEST_CASE("create_subgraphs keeps L buckets (including empty), with "
          "bucket[i] == layer-i edges (M6 §2-3 mirror)",
          "[graph]") {
    // M6 change vs M5: sub_G_ now has exactly L entries — empty layers
    // present as empty buckets. The previous M5 test asserted
    // `sub_graphs().size() == 1` when only layer 0 was populated; that
    // contract is replaced by "sub_G_[i] is the layer-i bucket" so the
    // aggregate writer reads `bucket[i] ↔ layer i` directly. Empty
    // layers still emit a null entry in run_report.json.summary.layers
    // (writer skips them).
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);

    std::vector<int> all0(g.num_edges(), 0);
    g.apply_optimization_result(all0);
    g.create_subgraphs();
    REQUIRE(g.sub_graphs().size() == static_cast<size_t>(g.L())); // L entries
    REQUIRE(!g.sub_graphs()[0].empty());
    REQUIRE(g.sub_graphs()[1].empty());
    REQUIRE(std::get<2>(g.sub_graphs()[0].front()).layer == 0);

    std::vector<int> all1(g.num_edges(), 1);
    g.apply_optimization_result(all1);
    g.create_subgraphs();
    REQUIRE(g.sub_graphs().size() == static_cast<size_t>(g.L()));
    // sub_G_[0] holds the perimeter (pinned to layer 0) edges; sub_G_[1]
    // holds the interior (layer 1) edges. M6: bucket index == layer id.
    REQUIRE(!g.sub_graphs()[0].empty());
    REQUIRE(!g.sub_graphs()[1].empty());
    REQUIRE(std::get<2>(g.sub_graphs()[0].front()).layer == 0);
    REQUIRE(std::get<2>(g.sub_graphs()[1].front()).layer == 1);
}

TEST_CASE("create_subgraphs splits into layer 0 and 1", "[graph]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);

    std::vector<int> mixed(g.num_edges(), 0);
    for (size_t i = 0; i < mixed.size(); i += 2) mixed[i] = 1;
    g.apply_optimization_result(mixed);
    g.create_subgraphs();

    // M6: sub_G_ always has L entries.
    REQUIRE(g.sub_graphs().size() == static_cast<size_t>(g.L()));
    int total = 0;
    for (const auto& sg : g.sub_graphs()) total += static_cast<int>(sg.size());
    REQUIRE(total == g.num_edges());
}

// Robustness hardening (Codex P2 finding 1, post-79f5e79). The two
// §1-3-c optimizer-surface functions must agree on input validation:
// `loss_function` already throws std::invalid_argument on a
// wrong-length `layers` (graph.cpp), but `apply_optimization_result`
// historically silently zero-filled a too-short vector (and ignored a
// too-long tail). Production never reaches this — both fixed-layers
// entry points reject wrong lengths upstream (main.cpp +
// api.py::run_optimization) and the optimizer always emits exactly
// num_edges() — so this only closes a direct-API divergence. The fix
// makes apply_optimization_result reject with the SAME exception type
// and message idiom as loss_function, and must not half-mutate state
// on the error path.
TEST_CASE("apply_optimization_result rejects wrong-length layers "
          "consistently with loss_function", "[graph]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);

    const std::size_t n = static_cast<std::size_t>(g.num_edges());
    std::vector<int> too_short(n - 1, 0);
    std::vector<int> too_long(n + 1, 0);

    // Establish a DISTINCT valid baseline (interior layer 5, perimeter
    // pinned to 0) so "unchanged" is unambiguous and cannot coincide
    // with the post-reset zero state.
    auto require_baseline_all5 = [&]() {
        for (const auto& e : g.all_edges()) {
            const int u = std::get<0>(e);
            const int v = std::get<1>(e);
            const int diff = std::abs(u - v);
            if (diff == 1 || diff == (k - 1)) {
                REQUIRE(std::get<2>(e).layer == 0);   // perimeter pin
            } else {
                REQUIRE(std::get<2>(e).layer == 5);   // interior baseline
            }
        }
    };
    g.apply_optimization_result(std::vector<int>(n, 5));
    require_baseline_all5();

    // The guard runs BEFORE the reset/assignment loop, so a throwing
    // wrong-length call must mutate NOTHING. Assert the baseline is
    // byte-for-byte intact immediately after each throw, with NO
    // intervening valid apply to mask a partial write (non-vacuous).
    REQUIRE_THROWS_AS(g.apply_optimization_result(too_short),
                      std::invalid_argument);
    require_baseline_all5();
    REQUIRE_THROWS_AS(g.apply_optimization_result(too_long),
                      std::invalid_argument);
    require_baseline_all5();

    // Same wrong lengths are rejected the same way by the sibling
    // optimizer-surface function (consistency anchor).
    REQUIRE_THROWS_AS(g.loss_function(too_short), std::invalid_argument);
    REQUIRE_THROWS_AS(g.loss_function(too_long), std::invalid_argument);

    // Graph remains usable after the throw sequence: a subsequent
    // correct call still produces the canonical perimeter pin +
    // interior layer.
    g.apply_optimization_result(std::vector<int>(n, 1));
    for (const auto& e : g.all_edges()) {
        const int u = std::get<0>(e);
        const int v = std::get<1>(e);
        const int diff = std::abs(u - v);
        if (diff == 1 || diff == (k - 1)) {
            REQUIRE(std::get<2>(e).layer == 0);   // perimeter pin
        } else {
            REQUIRE(std::get<2>(e).layer == 1);
        }
    }
}

// Robustness hardening (Codex P2 finding 2, post-79f5e79) — taper-hop
// site. After the P2-O5 raw-value/no-clamp contract, a pathological
// accepted `int` layer value from a direct/fixed-layers caller flows
// into `taper_hops = std::abs(pinned - edge_coupler_layer)`.
// `std::abs(INT_MIN)` and `INT_MIN - ecl` are signed-overflow UB.
// Widening the difference to int64 makes the taper monotone in
// |layer - ecl| for ALL accepted ints; in-range and large-but-safe
// values stay bit-identical. alpha=1/beta=0 so the return is the
// selected-edge mean directly.
TEST_CASE("loss_function taper-hop math does not overflow on extreme "
          "int layer inputs", "[graph]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    // ecl=0, perimeter=0, lic=0 so only the taper term carries the
    // signal (the adjacency site is exercised by the crosstalk-file
    // sibling test).
    SiNInterconnectionGraph g(k, pos, /*cl=*/0.3, /*tl=*/1.0, /*itl=*/0.0);
    g.set_loss_weights(1.0, 0.0);

    const std::size_t n = static_cast<std::size_t>(g.num_edges());
    auto f = [&](int V) {
        return g.loss_function(std::vector<int>(n, V));
    };

    constexpr int kIntMax = std::numeric_limits<int>::max();
    constexpr int kIntMin = std::numeric_limits<int>::min();

    // Safe, in-range monotonicity (taper grows with |layer - ecl|);
    // identical pre- and post-fix — regression anchor.
    REQUIRE(f(0) < f(5));
    REQUIRE(f(5) < f(1000000));
    // INT_MAX taper (= |INT_MAX - 0|) never overflowed the subtraction
    // even pre-fix: this is the "large but safe" anchor — must stay
    // bit-identical and keep growing.
    REQUIRE(f(1000000) < f(kIntMax));
    // INT_MIN is the UB case: pre-fix std::abs(INT_MIN - 0) yields a
    // NEGATIVE taper, collapsing the loss below the in-range values
    // (RED). Post-fix int64 |INT_MIN - 0| = 2147483648 → loss is the
    // largest of all and strictly positive (GREEN).
    REQUIRE(f(kIntMin) > 0.0);
    REQUIRE(f(kIntMin) > f(1000000));
    // |INT_MIN| and |INT_MAX| differ by exactly 1 hop → losses are
    // numerically adjacent, not sign-flipped.
    REQUIRE(f(kIntMin) == Approx(f(kIntMax)).epsilon(1e-6));
}
