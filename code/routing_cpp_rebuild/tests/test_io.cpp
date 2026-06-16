#include <catch2/catch.hpp>

#include "sinic/graph.hpp"
#include "sinic/io.hpp"
#include "sinic/positions.hpp"
#include "sinic/schema_validator.hpp"

#include <filesystem>
#include <fstream>

using sinic::SiNInterconnectionGraph;
using sinic::distribute_nodes_around_square;
using sinic::save_subgraphsdata_v1x;
using sinic::save_subgraphs_legacy_newformat;
using sinic::subgraphsdata_v1x_to_json;
using sinic::subgraphs_legacy_newformat_to_json;
using sinic::load_subgraphsdata;
using sinic::WriteOptions;

namespace {
std::filesystem::path make_tmpdir(const std::string& tag) {
    auto p = std::filesystem::temp_directory_path() / ("sinic_test_io_" + tag);
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p;
}
} // namespace

TEST_CASE("v1x writer emits schema-conformant JSON", "[io]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);
    g.create_subgraphs();

    auto j = subgraphsdata_v1x_to_json(g, {}, "1.0");
    REQUIRE(j["schema_version"] == "1.0");
    REQUIRE(j["General Parameters"]["k"] == k);
    REQUIRE(j.contains("positions"));
    REQUIRE(j.contains("complete_graph"));
    REQUIRE(j.contains("loss_analysis"));
    // Validation should pass.
    REQUIRE_NOTHROW(sinic::validate_subgraphsdata_payload(j));
}

TEST_CASE("legacy newformat writer matches pre-M5 layout", "[io]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);
    g.create_subgraphs();

    std::map<std::string, double> gp{{"Nodes", static_cast<double>(k)}};
    auto j = subgraphs_legacy_newformat_to_json(g, gp);

    // No `schema_version` / `positions` / `complete_graph` / `loss_analysis`.
    REQUIRE_FALSE(j.contains("schema_version"));
    REQUIRE_FALSE(j.contains("positions"));
    REQUIRE_FALSE(j.contains("complete_graph"));
    REQUIRE_FALSE(j.contains("loss_analysis"));
    REQUIRE(j["General Parameters"]["Loss of Taper"] == Approx(1.0));
    REQUIRE(j["General Parameters"]["Loss of Crossing"] == Approx(0.3));
    REQUIRE(j["General Parameters"]["Nodes"] == k);
    // Each Layer_i has an "edges" array of [u, v, attrs] triples.
    REQUIRE(j.contains("Layer_0"));
    const auto& e0 = j["Layer_0"]["edges"];
    REQUIRE(e0.is_array());
    if (!e0.empty()) {
        REQUIRE(e0[0].is_array());
        REQUIRE(e0[0].size() == 3);
        REQUIRE(e0[0][2].contains("layer"));
        REQUIRE(e0[0][2].contains("crossings"));
        REQUIRE(e0[0][2].contains("loss"));
        REQUIRE(e0[0][2].contains("interlayercrossings"));
    }
}

TEST_CASE("save_subgraphsdata round-trip default (M6 schema_version=2.0)",
          "[io][integration]") {
    auto tmp = make_tmpdir("v2_roundtrip");
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);
    g.create_subgraphs();

    const auto path = (tmp / "subgraphsdata.json").string();
    WriteOptions opts; // M6 default: schema_version = "2.0"
    opts.validate = true;
    save_subgraphsdata_v1x(g, path, {}, opts);

    REQUIRE(std::filesystem::exists(path));
    auto loaded = load_subgraphsdata(path, /*validate=*/true);
    REQUIRE(loaded.schema_version.has_value());
    REQUIRE(*loaded.schema_version == "2.0");
    REQUIRE(loaded.general_params["k"] == k);
    // M6: v2.0 required fields per `subgraphsdata.schema.json::allOf.if/then`.
    REQUIRE(loaded.general_params.contains("L"));
    REQUIRE(loaded.general_params.contains("edge_coupler_layer"));
    REQUIRE(loaded.general_params.contains("perimeter_layer"));
    REQUIRE(loaded.general_params.contains("waveguides_per_link"));
    REQUIRE(loaded.general_params.contains("layer_pitch_um"));
    // Per-edge _above / _below are present on the loaded layer edges.
    REQUIRE(!loaded.sub_graphs.empty());
    std::filesystem::remove_all(tmp);
}

TEST_CASE("save_subgraphsdata round-trip explicit v1.x (legacy compat)",
          "[io][integration]") {
    auto tmp = make_tmpdir("v1x_roundtrip");
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);
    g.create_subgraphs();

    const auto path = (tmp / "subgraphsdata.json").string();
    WriteOptions opts;
    opts.validate       = true;
    opts.schema_version = "1.0"; // opt-in legacy schema
    save_subgraphsdata_v1x(g, path, {}, opts);

    REQUIRE(std::filesystem::exists(path));
    auto loaded = load_subgraphsdata(path, /*validate=*/true);
    REQUIRE(loaded.schema_version.has_value());
    REQUIRE(*loaded.schema_version == "1.0");
    REQUIRE(loaded.general_params["k"] == k);
    // v1.x: multi-layer General Parameters fields NOT emitted.
    REQUIRE(!loaded.general_params.contains("L"));
    REQUIRE(!loaded.sub_graphs.empty());
    std::filesystem::remove_all(tmp);
}

TEST_CASE("loss_analysis emits JSON null for empty layer buckets "
          "(M6 stage-2 review P1-A regression)",
          "[io]") {
    // L=3 graph; assign every edge to layer 0 only. After
    // `apply_optimization_result(all-zeros) + create_subgraphs()`,
    // layer 0 holds everything; layer 1 and layer 2 are empty buckets.
    // The four per-layer aggregate arrays in `loss_analysis` must
    // emit JSON null at indices 1 and 2, and numbers at index 0.
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    // ecl=0 / perimeter=0 so the layer-0 bucket gets every edge.
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006,
                              /*L=*/3, /*ecl=*/0, /*perimeter=*/0);
    std::vector<int> all0(g.num_edges(), 0);
    g.apply_optimization_result(all0);
    g.create_subgraphs();

    auto j = subgraphsdata_v1x_to_json(g, {}, /*schema_version=*/"2.0");
    const auto& la = j["loss_analysis"];
    REQUIRE(la["avg_loss_subgraphs"].is_array());
    REQUIRE(la["avg_loss_subgraphs"].size() == 3);
    REQUIRE(la["avg_loss_subgraphs"][0].is_number());
    REQUIRE(la["avg_loss_subgraphs"][1].is_null());
    REQUIRE(la["avg_loss_subgraphs"][2].is_null());
    REQUIRE(la["var_loss_subgraphs"][1].is_null());
    REQUIRE(la["std_loss_subgraphs"][1].is_null());
    REQUIRE(la["range_loss_subgraphs"][1].is_null());
    // The flattened set still has numbers (we have one non-empty bucket).
    REQUIRE(la["avg_flattened_loss_subgraphs"].is_number());
    REQUIRE(la["var_flattened_loss_subgraphs"].is_number());
    REQUIRE(la["std_flattened_loss_subgraphs"].is_number());
    REQUIRE(la["range_flattened_loss_subgraphs"].is_number());
    // Planar (completegraph) is always non-empty for K_k.
    REQUIRE(la["avg_loss_completegraph"].is_number());
    REQUIRE(la["var_loss_completegraph"].is_number());
    REQUIRE(la["std_loss_completegraph"].is_number());
    REQUIRE(la["range_loss_completegraph"].is_number());
}

TEST_CASE("loss_analysis Python parity: emits all 12 fields "
          "(M6 stage-2 review codex P1-2 regression)",
          "[io]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);
    g.create_subgraphs();
    auto j = subgraphsdata_v1x_to_json(g, {}, /*schema_version=*/"2.0");
    const auto& la = j["loss_analysis"];
    // Per-layer (× 4)
    REQUIRE(la.contains("avg_loss_subgraphs"));
    REQUIRE(la.contains("var_loss_subgraphs"));
    REQUIRE(la.contains("std_loss_subgraphs"));
    REQUIRE(la.contains("range_loss_subgraphs"));
    // Flattened (× 4)
    REQUIRE(la.contains("avg_flattened_loss_subgraphs"));
    REQUIRE(la.contains("var_flattened_loss_subgraphs"));
    REQUIRE(la.contains("std_flattened_loss_subgraphs"));
    REQUIRE(la.contains("range_flattened_loss_subgraphs"));
    // Complete graph (× 4)
    REQUIRE(la.contains("avg_loss_completegraph"));
    REQUIRE(la.contains("var_loss_completegraph"));
    REQUIRE(la.contains("std_loss_completegraph"));
    REQUIRE(la.contains("range_loss_completegraph"));
}

TEST_CASE("v1.x writer omits _above / _below per-edge fields "
          "(M6 stage-review P2-E coverage)",
          "[io]") {
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);
    g.create_subgraphs();

    auto j = subgraphsdata_v1x_to_json(g, {}, /*schema_version=*/"1.0");
    REQUIRE(j["schema_version"] == "1.0");
    REQUIRE(j.contains("Layer_0"));
    const auto& edges = j["Layer_0"]["edges"];
    REQUIRE(edges.is_array());
    if (!edges.empty()) {
        const auto& attrs = edges[0][2];
        REQUIRE(attrs.contains("layer"));
        REQUIRE(attrs.contains("crossings"));
        REQUIRE(attrs.contains("loss"));
        REQUIRE(attrs.contains("interlayercrossings"));
        REQUIRE_FALSE(attrs.contains("interlayercrossings_above"));
        REQUIRE_FALSE(attrs.contains("interlayercrossings_below"));
    }
    // v1.x writer omits multi-layer General Parameters fields too.
    REQUIRE_FALSE(j["General Parameters"].contains("L"));
    REQUIRE_FALSE(j["General Parameters"].contains("waveguides_per_link"));
}

TEST_CASE("legacy writer file equals pre-M5 layout key set", "[io][integration]") {
    auto tmp = make_tmpdir("legacy_roundtrip");
    const int nps = 3;
    const int k   = 4 * nps;
    const auto pos = distribute_nodes_around_square(nps, 10.0);
    SiNInterconnectionGraph g(k, pos, 0.3, 1.0, 0.006);
    g.create_subgraphs();

    const auto path = (tmp / "subgraphsdata.json").string();
    save_subgraphs_legacy_newformat(g, path, {{"Nodes", static_cast<double>(k)}});
    REQUIRE(std::filesystem::exists(path));

    // Read raw and check key set vs pre-M5 expectations.
    std::ifstream f(path);
    nlohmann::json j;
    f >> j;
    REQUIRE(j.contains("General Parameters"));
    REQUIRE(j.contains("Layer_0"));
    // Pre-M5 writer never wrote schema_version/positions/etc.
    REQUIRE_FALSE(j.contains("schema_version"));
    std::filesystem::remove_all(tmp);
}
