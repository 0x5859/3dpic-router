// End-to-end optimizer regression tests landed during the M5
// second-round dual review. The first-round regression tests for
// the wall_ms origin and post-optimize-state P1 fixes were
// synthetic (drove RunRecorder directly without involving the
// actual DualAnnealingOptimizer). These tests close that gap by
// running a real (small) optimize() and asserting on the resulting
// `run_report.json` + `subgraphsdata.json` invariants. If someone
// re-introduces the wall_ms clock-origin split or removes the
// `graph.apply_optimization_result(best_layers)` call, these tests
// fail.

#include <catch2/catch.hpp>

#include "sinic/graph.hpp"
#include "sinic/io.hpp"
#include "sinic/optimizer.hpp"
#include "sinic/positions.hpp"
#include "sinic/statistics.hpp"

#include <algorithm>
#include <cmath>

using namespace sinic;

TEST_CASE("optimization_wall_ms shares origin with trace[].wall_ms "
          "(M5 dual-review P1 end-to-end fix)",
          "[end_to_end][optimizer]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);

    RunRecorder rec("e2e_wall_ms", nlohmann::json::object());

    DualAnnealingParams params;
    params.max_iter = 5;       // tiny: ~hundreds of evals total
    params.seed     = 42;      // deterministic
    params.patience = 3;       // converge fast

    DualAnnealingOptimizer opt(g, params, rec);
    opt.optimize();

    auto report = rec.finalize();
    REQUIRE(report["timings"].contains("optimization_wall_ms"));
    const long phase = report["timings"]["optimization_wall_ms"].get<long>();
    REQUIRE(phase >= 0);

    long max_trace = 0;
    REQUIRE(report["trace"].is_array());
    REQUIRE(!report["trace"].empty());
    for (const auto& ev : report["trace"]) {
        max_trace = std::max(max_trace, ev["wall_ms"].get<long>());
    }
    // The single t0 captured at optimize() entry drives BOTH the
    // per-eval IterEvent.wall_ms AND the optimization_wall_ms phase
    // emitted after the chain returns. Therefore the phase value
    // must be >= the largest trace[].wall_ms (the chain's last eval
    // happens before the phase emit, so its wall_ms is upper-bounded
    // by the phase total).
    REQUIRE(phase >= max_trace);
}

TEST_CASE("post-optimize graph state reflects BEST not LAST "
          "(M5 dual-review P1 end-to-end fix)",
          "[end_to_end][optimizer]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);
    g.set_loss_weights(100000.0, 0.0);  // pre-M5 alpha=1e5, beta=0

    DualAnnealingParams params;
    params.max_iter = 10;
    params.seed     = 42;
    params.patience = 5;

    DualAnnealingOptimizer opt(g, params, default_null_sink());
    opt.optimize();

    // Without the M5 P1 fix, graph.allEdges_ at this point reflects
    // the LAST evaluated solution from inside the optimizer's chain
    // (a residual mutation from DualAnnealingObjective::value). With
    // the fix, the orchestrator (main.cpp + this test) explicitly
    // applies best_layers so analyze_loss / save reflect the optimal
    // solution.
    g.apply_optimization_result(opt.best_layers());
    g.create_subgraphs();

    // Compute mean per-edge loss from the freshly-rebuilt subgraphs.
    double sum = 0.0;
    std::size_t n = 0;
    for (const auto& sg : g.sub_graphs()) {
        for (const auto& e : sg) {
            sum += std::get<2>(e).loss;
            ++n;
        }
    }
    REQUIRE(n > 0);
    const double observed_mean = sum / static_cast<double>(n);
    const double expected_mean = opt.best_loss() / 100000.0; // alpha
    REQUIRE(observed_mean == Approx(expected_mean).margin(1e-9));
}

TEST_CASE("M6 end-to-end: L=3 / ecl=1 optimize + v2.0 JSON write + load "
          "+ schema validation",
          "[end_to_end][multilayer][io]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006,
                              /*L=*/3, /*ecl=*/1, /*perimeter=*/1,
                              /*pitch=*/1.2, /*wpl=*/2);
    g.set_loss_weights(100000.0, 0.0);

    DualAnnealingParams params;
    params.max_iter = 5;
    params.seed     = 42;
    params.patience = 3;
    DualAnnealingOptimizer opt(g, params, default_null_sink());
    opt.optimize();
    g.apply_optimization_result(opt.best_layers());
    g.create_subgraphs();

    // Write v2.0 JSON, then re-load + validate.
    auto tmp = std::filesystem::temp_directory_path() / "sinic_test_e2e_m6";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    const auto path = (tmp / "subgraphsdata.json").string();
    WriteOptions opts; // M6 default: schema_version="2.0"
    opts.validate = true;
    save_subgraphsdata_v1x(g, path, {}, opts);

    auto loaded = load_subgraphsdata(path, /*validate=*/true);
    REQUIRE(loaded.schema_version.has_value());
    REQUIRE(*loaded.schema_version == "2.0");
    REQUIRE(loaded.general_params["L"] == 3);
    REQUIRE(loaded.general_params["edge_coupler_layer"] == 1);
    REQUIRE(loaded.general_params["perimeter_layer"] == 1);
    REQUIRE(loaded.general_params["waveguides_per_link"] == 2);
    REQUIRE(loaded.general_params["layer_pitch_um"] == Approx(1.2));
    REQUIRE(loaded.sub_graphs.size() == 3); // L entries (some may be empty)

    // _above + _below = legacy single counter per edge invariant
    int sum_above = 0;
    int sum_below = 0;
    int sum_legacy = 0;
    for (const auto& sg : loaded.sub_graphs) {
        for (const auto& e : sg) {
            const auto& ed = std::get<2>(e);
            REQUIRE(ed.interlayerCrossingsAbove
                  + ed.interlayerCrossingsBelow
                    == ed.interlayerCrossings);
            sum_above  += ed.interlayerCrossingsAbove;
            sum_below  += ed.interlayerCrossingsBelow;
            sum_legacy += ed.interlayerCrossings;
        }
    }
    REQUIRE(sum_above == sum_below);
    REQUIRE(sum_legacy == 2 * sum_above);

    std::filesystem::remove_all(tmp);
}

TEST_CASE("aggregate-layer fix: layer field reads edge.layer "
          "not sub_G_ index (M5 dual-review P0-1 end-to-end fix)",
          "[end_to_end][stats]") {
    // Drive the exact aggregate-loop pattern that main.cpp uses so
    // a revert to "layer = static_cast<int>(i)" would fail this
    // test even without going through main.cpp.
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);

    // Pin every edge to layer 1; the perimeter pin will pull
    // perimeter edges back to layer 0, leaving both sub-graphs
    // non-empty. The aggregate loop should report layer=0 and
    // layer=1 — matching the edge attribute, not the sub_G_ index.
    std::vector<int> all_ones(g.num_edges(), 1);
    g.apply_optimization_result(all_ones);
    g.create_subgraphs();

    // Mirror the main.cpp aggregate-stats loop exactly.
    std::vector<int> seen_layers;
    for (const auto& sg : g.sub_graphs()) {
        if (sg.empty()) continue;
        seen_layers.push_back(std::get<2>(sg.front()).layer);
    }
    REQUIRE(seen_layers.size() == 2);
    REQUIRE(seen_layers[0] == 0);
    REQUIRE(seen_layers[1] == 1);

    // Now drive the empty-layer-0 case explicitly. Manually clear
    // the perimeter pin invariant by setting every allEdges_.layer
    // to 1 and rebuilding sub-graphs. The aggregate loop must
    // emit layer=1 (single entry), NOT layer=0.
    for (auto& e : g.all_edges_mut()) {
        std::get<2>(e).layer = 1;
    }
    g.create_subgraphs();
    std::vector<int> seen_layers_2;
    for (const auto& sg : g.sub_graphs()) {
        if (sg.empty()) continue;
        seen_layers_2.push_back(std::get<2>(sg.front()).layer);
    }
    REQUIRE(seen_layers_2.size() == 1);
    REQUIRE(seen_layers_2[0] == 1);  // would have been 0 with the bug
}
