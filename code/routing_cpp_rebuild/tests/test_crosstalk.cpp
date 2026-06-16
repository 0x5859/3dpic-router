// M7 acceptance tests for the C++ crosstalk tensor (REFACTOR_GOALS.md
// §2-2). Mirrors `code/tests/test_crosstalk.py` (Python M4) so the
// M8 PythonParityFixture can compare both ends against the same
// fixtures (4-node unit-square + 12-node K_12).
//
// Test taxonomy (matches Python coverage where possible):
//   - 4-node intralayer analytical sanity (≈ −43.0103 dB)
//   - 4-node interlayer analytical sanity (≈ −53.0103 dB) + |Δ|=2 decouple
//   - Perimeter paths produce no leakage
//   - max_hops=0 / coefs=0 / coefs=nullopt short-circuit (engine doesn't
//     even touch Phase A)
//   - Threshold above/below signal
//   - loss_crossing>=1 raise; max_hops<0 raise; threshold_db non-finite raise
//   - Determinism (two consecutive calls return identical tensors)
//   - L=3 mixed adjacency (intra + inter on same fixture)
//   - Writer end-to-end: shape-validates, schema-validates, round-trips
//   - Cross-field shape validator rejection branches (wrong shape,
//     missing shape, bool shape, inner slab mismatch, leaf row mismatch)
//   - Phase A skipped when short-circuited (graph.topology() stays null)
#include <catch2/catch.hpp>

#include "sinic/crosstalk.hpp"
#include "sinic/graph.hpp"
#include "sinic/io.hpp"
#include "sinic/schema_validator.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using sinic::SiNInterconnectionGraph;
using sinic::compute_crosstalk_tensor;
using sinic::PositionMap;
using sinic::save_subgraphsdata_v1x;
using sinic::subgraphsdata_v1x_to_json;
using sinic::validate_crosstalk_payload_shape;
using sinic::validate_subgraphsdata_payload;
using sinic::WriteOptions;

namespace {

PositionMap unit_square_positions() {
    // Matches Python `_unit_square_positions`: corners 0=top-left,
    // 1=top-right, 2=bottom-right, 3=bottom-left. Diagonals (0,2) and
    // (1,3) cross at (0.5, 0.5) so the parametric intersection is
    // exactly t=0.5 on both edges — analytical closed form is exact.
    return {
        {0, {0.0, 1.0}},
        {1, {1.0, 1.0}},
        {2, {1.0, 0.0}},
        {3, {0.0, 0.0}},
    };
}

SiNInterconnectionGraph make_4node_graph(
    int L = 2,
    int ecl = 0,
    int perimeter = 0,
    std::optional<double> intra = std::optional<double>(1e-4),
    std::optional<double> inter = std::optional<double>(1e-5),
    double loss_crossing = 0.3,
    double loss_taper = 0.0,
    double loss_interlayer_crossing = 0.0) {
    return SiNInterconnectionGraph(
        /*k=*/4, unit_square_positions(),
        loss_crossing, loss_taper, loss_interlayer_crossing,
        /*L=*/L, /*edge_coupler_layer=*/ecl, /*perimeter_layer=*/perimeter,
        /*layer_pitch_um=*/1.2, /*waveguides_per_link=*/2,
        intra, inter);
}

// Helper: apply uniform layer assignment to the 4-node graph (sets
// all non-perimeter edges to `layer` via the apply_optimization_result
// pin — perimeter edges land on `perimeter_layer`).
void apply_uniform_layers(SiNInterconnectionGraph& g, int layer) {
    g.build_crossings_index();
    const int n_edges = g.num_edges();
    std::vector<int> layers(static_cast<std::size_t>(n_edges), layer);
    g.apply_optimization_result(layers);
}

// Helper: build the (u, v) → index map mirroring Python
// `graph._edge_index`. Used in tests to look up specific diagonal
// indices for hand-painted layer assignments.
std::map<std::pair<int, int>, int> build_edge_index(
    const SiNInterconnectionGraph& g) {
    std::map<std::pair<int, int>, int> idx;
    const auto& all = g.all_edges();
    for (std::size_t i = 0; i < all.size(); ++i) {
        idx[{std::get<0>(all[i]), std::get<1>(all[i])}] = static_cast<int>(i);
    }
    return idx;
}

std::filesystem::path make_tmpdir(const std::string& tag) {
    auto p = std::filesystem::temp_directory_path()
           / ("sinic_test_crosstalk_" + tag);
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p;
}

} // namespace

// ---------------------------------------------------------------------------
// T3 — 4-node analytical sanity (intralayer)
// ---------------------------------------------------------------------------

TEST_CASE("4-node intralayer analytical sanity (-43.01 dB)", "[crosstalk]") {
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);

    const auto ct = compute_crosstalk_tensor(g);
    const double expected_db = 10.0 * std::log10(0.5 * 1e-4);

    const std::vector<std::pair<int, std::vector<int>>> sd_victims = {
        {0, {1, 3}}, {2, {1, 3}}, {1, {0, 2}}, {3, {0, 2}}};
    const std::vector<std::pair<int, int>> sd_pairs = {
        {0, 2}, {2, 0}, {1, 3}, {3, 1}};

    for (const auto& sd : sd_pairs) {
        const int s = sd.first;
        const int d = sd.second;
        std::vector<int> victims;
        for (const auto& sv : sd_victims) {
            if (sv.first == s) victims = sv.second;
        }
        for (int t : victims) {
            const auto& v = ct["values"][s][d][t];
            REQUIRE_FALSE(v.is_null());
            REQUIRE(std::abs(v.get<double>() - expected_db) < 1e-6);
        }
    }
}

TEST_CASE("4-node intralayer perimeter paths leak nothing", "[crosstalk]") {
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    const auto ct = compute_crosstalk_tensor(g);

    const std::vector<std::pair<int, int>> perimeter_sd = {
        {0, 1}, {1, 2}, {2, 3}, {0, 3},
        {1, 0}, {2, 1}, {3, 2}, {3, 0}};
    for (const auto& sd : perimeter_sd) {
        for (int t = 0; t < 4; ++t) {
            REQUIRE(ct["values"][sd.first][sd.second][t].is_null());
        }
    }
}

// ---------------------------------------------------------------------------
// T3 (extension) — 4-node analytical sanity (interlayer + |Δ|=2 decouple)
// ---------------------------------------------------------------------------

TEST_CASE("4-node interlayer analytical sanity and |Δlayer|=2 decouple",
          "[crosstalk]") {
    // |Δlayer| = 2 → coupling coefficient is 0 (§2-3 目标 D).
    auto g = make_4node_graph(/*L=*/3, /*ecl=*/1, /*perimeter=*/1);
    g.build_crossings_index();
    const auto edge_index = build_edge_index(g);
    const int diag_02 = edge_index.at({0, 2});
    const int diag_13 = edge_index.at({1, 3});

    const int n_edges = g.num_edges();
    std::vector<int> layers(static_cast<std::size_t>(n_edges), 1);
    layers[diag_02] = 0;
    layers[diag_13] = 2;
    g.apply_optimization_result(layers);

    auto ct_nonadj = compute_crosstalk_tensor(g);
    REQUIRE(ct_nonadj["values"][0][2][1].is_null());
    REQUIRE(ct_nonadj["values"][0][2][3].is_null());

    // |Δlayer| = 1 → coupling coefficient = loss_interlayer_crosstalk.
    layers[diag_02] = 1;
    layers[diag_13] = 0;
    g.apply_optimization_result(layers);
    auto ct_adj = compute_crosstalk_tensor(g);
    const double expected_db = 10.0 * std::log10(0.5 * 1e-5);
    for (int s : {0, 2}) {
        const int d = (s == 0) ? 2 : 0;
        for (int t : {1, 3}) {
            const auto& v = ct_adj["values"][s][d][t];
            REQUIRE_FALSE(v.is_null());
            REQUIRE(std::abs(v.get<double>() - expected_db) < 1e-6);
        }
    }
}

// ---------------------------------------------------------------------------
// T4 — crosstalk off ≡ pre-M4 (engine + loss model independence)
// ---------------------------------------------------------------------------

TEST_CASE("crosstalk all-zero coefficients produce all-null tensor",
          "[crosstalk]") {
    auto g = make_4node_graph(
        2, 0, 0,
        std::optional<double>(0.0),
        std::optional<double>(0.0));
    apply_uniform_layers(g, 0);
    auto ct = compute_crosstalk_tensor(g);
    for (int s = 0; s < 4; ++s)
        for (int d = 0; d < 4; ++d)
            for (int t = 0; t < 4; ++t)
                REQUIRE(ct["values"][s][d][t].is_null());
}

TEST_CASE("crosstalk nullopt coefficients produce all-null tensor",
          "[crosstalk]") {
    auto g = make_4node_graph(
        2, 0, 0, std::nullopt, std::nullopt);
    apply_uniform_layers(g, 0);
    auto ct = compute_crosstalk_tensor(g);
    REQUIRE(ct["loss_intralayer_crosstalk"].is_null());
    REQUIRE(ct["loss_interlayer_crosstalk"].is_null());
    for (int s = 0; s < 4; ++s)
        for (int d = 0; d < 4; ++d)
            for (int t = 0; t < 4; ++t)
                REQUIRE(ct["values"][s][d][t].is_null());
}

TEST_CASE("loss_function is independent of crosstalk coefficients",
          "[crosstalk]") {
    // Mirror Python `test_loss_function_independent_of_crosstalk_coefficients`:
    // crosstalk coefficients do not feed back into `loss_function`.
    const auto pos = unit_square_positions();
    SiNInterconnectionGraph g_off(
        4, pos, 0.3, 0.0, 0.0, 2, 0, 0, 1.2, 2,
        std::nullopt, std::nullopt);
    SiNInterconnectionGraph g_on(
        4, pos, 0.3, 0.0, 0.0, 2, 0, 0, 1.2, 2,
        std::optional<double>(1e-4),
        std::optional<double>(1e-5));
    g_off.build_crossings_index();
    g_on.build_crossings_index();
    const int n_edges = g_off.num_edges();
    // Same arbitrary layer pattern for both.
    std::vector<int> layers(static_cast<std::size_t>(n_edges));
    for (int i = 0; i < n_edges; ++i) layers[i] = i % 2;
    REQUIRE(g_off.loss_function(layers) == g_on.loss_function(layers));
}

// ---------------------------------------------------------------------------
// Hop / threshold pruning
// ---------------------------------------------------------------------------

TEST_CASE("max_hops=0 produces empty tensor", "[crosstalk]") {
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    auto ct = compute_crosstalk_tensor(g, /*max_hops=*/0);
    REQUIRE(ct["max_hops"] == 0);
    for (int s = 0; s < 4; ++s)
        for (int d = 0; d < 4; ++d)
            for (int t = 0; t < 4; ++t)
                REQUIRE(ct["values"][s][d][t].is_null());
}

TEST_CASE("threshold_db above signal level prunes everything",
          "[crosstalk]") {
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    auto ct = compute_crosstalk_tensor(g, 3, /*threshold_db=*/-30.0);
    for (int s = 0; s < 4; ++s)
        for (int d = 0; d < 4; ++d)
            for (int t = 0; t < 4; ++t)
                REQUIRE(ct["values"][s][d][t].is_null());
}

TEST_CASE("threshold_db below signal level keeps signal", "[crosstalk]") {
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    auto ct = compute_crosstalk_tensor(g, 3, -60.0);
    REQUIRE_FALSE(ct["values"][0][2][1].is_null());
    REQUIRE_FALSE(ct["values"][0][2][3].is_null());
}

// ---------------------------------------------------------------------------
// Error-raising guards (must mirror Python's eager-raise)
// ---------------------------------------------------------------------------

TEST_CASE("loss_crossing >= 1 raises (model assumption)", "[crosstalk]") {
    auto g = make_4node_graph(
        2, 0, 0,
        std::optional<double>(1e-4),
        std::optional<double>(1e-5),
        /*loss_crossing=*/1.0);
    apply_uniform_layers(g, 0);
    REQUIRE_THROWS_AS(compute_crosstalk_tensor(g), std::invalid_argument);
    REQUIRE_THROWS_WITH(compute_crosstalk_tensor(g),
                        Catch::Matchers::Contains("produces transmission"));
}

TEST_CASE("negative max_hops raises", "[crosstalk]") {
    auto g = make_4node_graph();
    REQUIRE_THROWS_AS(compute_crosstalk_tensor(g, -1),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(compute_crosstalk_tensor(g, -1),
                        Catch::Matchers::Contains("must be >= 0"));
}

TEST_CASE("non-finite threshold_db raises", "[crosstalk]") {
    auto g = make_4node_graph();
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -inf;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(compute_crosstalk_tensor(g, 3, inf),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(compute_crosstalk_tensor(g, 3, ninf),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(compute_crosstalk_tensor(g, 3, nan),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(compute_crosstalk_tensor(g, 3, inf),
                        Catch::Matchers::Contains("threshold_db must be finite"));
}

// ---------------------------------------------------------------------------
// Determinism + L=3 mixed adjacency
// ---------------------------------------------------------------------------

TEST_CASE("compute_crosstalk_tensor is deterministic", "[crosstalk]") {
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    auto ct1 = compute_crosstalk_tensor(g);
    auto ct2 = compute_crosstalk_tensor(g);
    REQUIRE(ct1["values"] == ct2["values"]);
}

TEST_CASE("L=3 mixed adjacency: intra vs inter coefs differ", "[crosstalk]") {
    auto g = make_4node_graph(/*L=*/3, /*ecl=*/1, /*perimeter=*/1);
    g.build_crossings_index();
    const auto edge_index = build_edge_index(g);
    const int diag_02 = edge_index.at({0, 2});
    const int diag_13 = edge_index.at({1, 3});
    const int n_edges = g.num_edges();
    std::vector<int> layers(static_cast<std::size_t>(n_edges), 1);

    // Interlayer Δ=1
    layers[diag_02] = 1;
    layers[diag_13] = 0;
    g.apply_optimization_result(layers);
    auto ct_inter = compute_crosstalk_tensor(g);
    const double expected_inter = 10.0 * std::log10(0.5 * 1e-5);
    REQUIRE_FALSE(ct_inter["values"][0][2][1].is_null());
    REQUIRE(std::abs(ct_inter["values"][0][2][1].get<double>() - expected_inter)
            < 1e-6);

    // Intralayer (both on layer 1)
    layers[diag_02] = 1;
    layers[diag_13] = 1;
    g.apply_optimization_result(layers);
    auto ct_intra = compute_crosstalk_tensor(g);
    const double expected_intra = 10.0 * std::log10(0.5 * 1e-4);
    REQUIRE_FALSE(ct_intra["values"][0][2][1].is_null());
    REQUIRE(std::abs(ct_intra["values"][0][2][1].get<double>() - expected_intra)
            < 1e-6);

    // Two regimes are distinct by ~10 dB.
    REQUIRE(std::abs(ct_inter["values"][0][2][1].get<double>()
                     - ct_intra["values"][0][2][1].get<double>()) > 5.0);
}

// ---------------------------------------------------------------------------
// Phase A is not built when the engine short-circuits
// ---------------------------------------------------------------------------

TEST_CASE("Phase A skipped when coefficients are zero", "[crosstalk]") {
    auto g = make_4node_graph(
        2, 0, 0,
        std::optional<double>(0.0),
        std::optional<double>(0.0));
    // No build_crossings_index() — leave Phase A lazy.
    REQUIRE(g.topology() == nullptr);
    auto ct = compute_crosstalk_tensor(g);
    REQUIRE(g.topology() == nullptr);
    REQUIRE(ct["values"][0][1][2].is_null());
}

TEST_CASE("Phase A skipped when max_hops=0", "[crosstalk]") {
    auto g = make_4node_graph();  // nonzero coefficients
    REQUIRE(g.topology() == nullptr);
    auto ct = compute_crosstalk_tensor(g, /*max_hops=*/0);
    REQUIRE(g.topology() == nullptr);
    REQUIRE(ct["values"][0][1][2].is_null());
}

// ---------------------------------------------------------------------------
// Payload metadata shape contract
// ---------------------------------------------------------------------------

TEST_CASE("compute_crosstalk_tensor payload shape + metadata", "[crosstalk]") {
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    auto ct = compute_crosstalk_tensor(g);
    REQUIRE(ct["unit"] == "dB");
    REQUIRE(ct["shape"] == nlohmann::json::array({4, 4, 4}));
    REQUIRE(ct["coherence_model"] == "incoherent_v1");
    REQUIRE(ct["polarization"] == "TE0_only");
    REQUIRE(ct["symmetric"] == false);
    REQUIRE(ct["diagonal_convention"] == "NaN");
    REQUIRE(ct["max_hops"] == 3);
    REQUIRE(ct["threshold_db"] == -60.0);
    REQUIRE(ct["loss_intralayer_crosstalk"] == 1e-4);
    REQUIRE(ct["loss_interlayer_crosstalk"] == 1e-5);
    REQUIRE(ct["values"].size() == 4);
    for (const auto& slab : ct["values"]) {
        REQUIRE(slab.size() == 4);
        for (const auto& row : slab) {
            REQUIRE(row.size() == 4);
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end: writer carries crosstalk + schema-validates
// ---------------------------------------------------------------------------

TEST_CASE("save_subgraphsdata_v1x carries crosstalk and schema-validates",
          "[crosstalk][io]") {
    auto tmp = make_tmpdir("e2e");
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    g.create_subgraphs();
    auto ct = compute_crosstalk_tensor(g);

    const auto path = (tmp / "subgraphsdata.json").string();
    WriteOptions opts;
    opts.validate = true;
    opts.schema_version = "2.0";
    save_subgraphsdata_v1x(g, path, {}, opts, &ct);

    std::ifstream ifs(path);
    nlohmann::json on_disk;
    ifs >> on_disk;
    REQUIRE(on_disk.contains("crosstalk"));
    REQUIRE(on_disk["crosstalk"]["unit"] == "dB");
    REQUIRE(on_disk["crosstalk"]["shape"]
            == nlohmann::json::array({4, 4, 4}));
    REQUIRE(on_disk["crosstalk"]["coherence_model"] == "incoherent_v1");
    REQUIRE(on_disk["crosstalk"]["polarization"] == "TE0_only");
    // Schema validate the on-disk file (defense in depth).
    REQUIRE_NOTHROW(validate_subgraphsdata_payload(on_disk));

    // M7: General Parameters now mirror the graph's crosstalk knobs.
    REQUIRE(on_disk["General Parameters"]["Loss of Intralayer Crosstalk"]
            == 1e-4);
    REQUIRE(on_disk["General Parameters"]["Loss of Interlayer Crosstalk"]
            == 1e-5);
    REQUIRE(on_disk["General Parameters"]["coherence_model"]
            == "incoherent_v1");
    REQUIRE(on_disk["General Parameters"]["polarization"] == "TE0_only");
}

TEST_CASE("save_subgraphsdata_v1x omits crosstalk when payload is null",
          "[crosstalk][io]") {
    auto tmp = make_tmpdir("null");
    auto g = make_4node_graph(
        2, 0, 0, std::nullopt, std::nullopt);
    apply_uniform_layers(g, 0);
    g.create_subgraphs();

    const auto path = (tmp / "subgraphsdata.json").string();
    WriteOptions opts;
    opts.validate = true;
    opts.schema_version = "2.0";
    save_subgraphsdata_v1x(g, path, {}, opts, nullptr);

    std::ifstream ifs(path);
    nlohmann::json on_disk;
    ifs >> on_disk;
    REQUIRE_FALSE(on_disk.contains("crosstalk"));
    // General Parameters explicit-null for both coefficients.
    REQUIRE(on_disk["General Parameters"]["Loss of Intralayer Crosstalk"]
            .is_null());
    REQUIRE(on_disk["General Parameters"]["Loss of Interlayer Crosstalk"]
            .is_null());
}

// ---------------------------------------------------------------------------
// Cross-field shape validator branches
// ---------------------------------------------------------------------------

TEST_CASE("validate_crosstalk_payload_shape: shape array length mismatch",
          "[crosstalk][validator]") {
    nlohmann::json bad = nlohmann::json::object();
    bad["unit"] = "dB";
    bad["shape"] = nlohmann::json::array({4, 4});   // 2 entries, not 3
    bad["values"] = nlohmann::json::array();
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad, 4),
                        Catch::Matchers::Contains("3-element array"));
}

TEST_CASE("validate_crosstalk_payload_shape: shape != graph k",
          "[crosstalk][validator]") {
    nlohmann::json bad = nlohmann::json::object();
    bad["unit"] = "dB";
    bad["shape"] = nlohmann::json::array({3, 3, 3});  // wrong k
    bad["values"] = nlohmann::json::array();
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad, 4),
                        Catch::Matchers::Contains("does not match graph k"));
}

TEST_CASE("validate_crosstalk_payload_shape: missing shape",
          "[crosstalk][validator]") {
    nlohmann::json bad = nlohmann::json::object();
    bad["unit"] = "dB";
    // shape missing entirely
    bad["values"] = nlohmann::json::array();
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad, 4),
                        Catch::Matchers::Contains("3-element array"));
}

TEST_CASE("validate_crosstalk_payload_shape: bool shape entries",
          "[crosstalk][validator]") {
    nlohmann::json bad = nlohmann::json::object();
    bad["unit"] = "dB";
    // Mirror Python `test_writer_rejects_bool_shape_entries`. JSON true
    // is `boolean` type in nlohmann::json, so this is naturally rejected
    // by `is_number_integer()`. Explicit guard documented.
    bad["shape"] = nlohmann::json::array({true, true, true});
    bad["values"] = nlohmann::json::array();
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad, 4),
                      std::invalid_argument);
}

TEST_CASE("validate_crosstalk_payload_shape: outer list length mismatch",
          "[crosstalk][validator]") {
    nlohmann::json bad = nlohmann::json::object();
    bad["unit"] = "dB";
    bad["shape"] = nlohmann::json::array({4, 4, 4});
    // values has 3 outer slabs instead of 4
    nlohmann::json values = nlohmann::json::array();
    for (int i = 0; i < 3; ++i) {
        nlohmann::json slab = nlohmann::json::array();
        for (int j = 0; j < 4; ++j) {
            nlohmann::json row = nlohmann::json::array();
            for (int t = 0; t < 4; ++t) row.push_back(nullptr);
            slab.push_back(std::move(row));
        }
        values.push_back(std::move(slab));
    }
    bad["values"] = std::move(values);
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad, 4),
                        Catch::Matchers::Contains("does not match shape[0]"));
}

TEST_CASE("validate_crosstalk_payload_shape: inner slab not a list / wrong length",
          "[crosstalk][validator]") {
    auto build_base = [] {
        nlohmann::json b = nlohmann::json::object();
        b["unit"] = "dB";
        b["shape"] = nlohmann::json::array({4, 4, 4});
        nlohmann::json values = nlohmann::json::array();
        for (int i = 0; i < 4; ++i) {
            nlohmann::json slab = nlohmann::json::array();
            for (int j = 0; j < 4; ++j) {
                nlohmann::json row = nlohmann::json::array();
                for (int t = 0; t < 4; ++t) row.push_back(nullptr);
                slab.push_back(std::move(row));
            }
            values.push_back(std::move(slab));
        }
        b["values"] = std::move(values);
        return b;
    };

    // (a-1) slab is not a list — replace values[0] with a string
    nlohmann::json bad1 = build_base();
    bad1["values"][0] = std::string("not a list");
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad1, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad1, 4),
                        Catch::Matchers::Contains("values[0]"));

    // (a-2) slab list with wrong length — replace values[0] with 3-row list
    nlohmann::json bad2 = build_base();
    nlohmann::json short_slab = nlohmann::json::array();
    for (int j = 0; j < 3; ++j) {
        nlohmann::json row = nlohmann::json::array();
        for (int t = 0; t < 4; ++t) row.push_back(nullptr);
        short_slab.push_back(std::move(row));
    }
    bad2["values"][0] = std::move(short_slab);
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad2, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad2, 4),
                        Catch::Matchers::Contains("values[0]"));
}

TEST_CASE("validate_crosstalk_payload_shape: leaf row not a list / wrong length",
          "[crosstalk][validator]") {
    auto build_base = [] {
        nlohmann::json b = nlohmann::json::object();
        b["unit"] = "dB";
        b["shape"] = nlohmann::json::array({4, 4, 4});
        nlohmann::json values = nlohmann::json::array();
        for (int i = 0; i < 4; ++i) {
            nlohmann::json slab = nlohmann::json::array();
            for (int j = 0; j < 4; ++j) {
                nlohmann::json row = nlohmann::json::array();
                for (int t = 0; t < 4; ++t) row.push_back(nullptr);
                slab.push_back(std::move(row));
            }
            values.push_back(std::move(slab));
        }
        b["values"] = std::move(values);
        return b;
    };

    // (b-1) leaf row is not a list — replace values[0][0] with a string
    nlohmann::json bad1 = build_base();
    bad1["values"][0][0] = std::string("not a list");
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad1, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad1, 4),
                        Catch::Matchers::Contains("values[0][0]"));

    // (b-2) leaf row with wrong length — replace values[0][0] with 3-entry
    nlohmann::json bad2 = build_base();
    nlohmann::json short_row = nlohmann::json::array();
    for (int t = 0; t < 3; ++t) short_row.push_back(nullptr);
    bad2["values"][0][0] = std::move(short_row);
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad2, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad2, 4),
                        Catch::Matchers::Contains("values[0][0]"));
}

TEST_CASE("save_subgraphsdata_v1x omits crosstalk in v1.x mode even with payload",
          "[crosstalk][io]") {
    // M7 dual-review R1 P1 fix: v1.x is a v2.0-only field per §3-1.
    // The writer must omit `crosstalk` even if the caller passes a
    // valid payload. (Pre-fix the gate was only `crosstalk != null`,
    // letting a v1.x JSON inherit a v2.0 field.)
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    g.create_subgraphs();
    auto ct = compute_crosstalk_tensor(g);

    auto j = subgraphsdata_v1x_to_json(g, {}, "1.0", &ct);
    REQUIRE(j["schema_version"] == "1.0");
    REQUIRE_FALSE(j.contains("crosstalk"));
    // v1.x General Parameters also should not carry any of the four
    // v2.0-only crosstalk + physical-model fields.
    REQUIRE_FALSE(j["General Parameters"].contains(
        "Loss of Intralayer Crosstalk"));
    REQUIRE_FALSE(j["General Parameters"].contains(
        "Loss of Interlayer Crosstalk"));
    REQUIRE_FALSE(j["General Parameters"].contains("coherence_model"));
    REQUIRE_FALSE(j["General Parameters"].contains("polarization"));
}

TEST_CASE("validate_crosstalk_payload_shape: zero shape entries rejected",
          "[crosstalk][validator]") {
    // M7 dual-review R1 P1 fix (opus): Python validator rejects
    // `x <= 0`; C++ must match. Pre-fix accepted `x == 0` (only
    // caught later by `dim != expected_k` if expected_k != 0).
    nlohmann::json bad = nlohmann::json::object();
    bad["unit"] = "dB";
    bad["shape"] = nlohmann::json::array({0, 0, 0});
    bad["values"] = nlohmann::json::array();
    REQUIRE_THROWS_AS(validate_crosstalk_payload_shape(bad, 4),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(validate_crosstalk_payload_shape(bad, 4),
                        Catch::Matchers::Contains("positive integers"));
}

TEST_CASE("custom coherence_model / polarization round-trip through writer",
          "[crosstalk][io]") {
    // M7 dual-review R1 P2 (opus): exercise non-default physical-
    // model labels so that future readers diffing JSON for model-tag
    // drift have a regression handle. Constructor-supplied strings
    // must propagate through both `compute_crosstalk_tensor` and the
    // v2.0 writer.
    auto tmp = make_tmpdir("custom_model");
    SiNInterconnectionGraph g(
        4, unit_square_positions(),
        0.3, 0.0, 0.0,
        2, 0, 0, 1.2, 2,
        std::optional<double>(1e-4),
        std::optional<double>(1e-5),
        "coherent_v2_experimental",
        "TM0_only");
    apply_uniform_layers(g, 0);
    g.create_subgraphs();
    auto ct = compute_crosstalk_tensor(g);
    REQUIRE(ct["coherence_model"] == "coherent_v2_experimental");
    REQUIRE(ct["polarization"] == "TM0_only");

    const auto path = (tmp / "subgraphsdata.json").string();
    WriteOptions opts;
    opts.validate = true;
    opts.schema_version = "2.0";
    save_subgraphsdata_v1x(g, path, {}, opts, &ct);

    std::ifstream ifs(path);
    nlohmann::json on_disk;
    ifs >> on_disk;
    REQUIRE(on_disk["General Parameters"]["coherence_model"]
            == "coherent_v2_experimental");
    REQUIRE(on_disk["General Parameters"]["polarization"] == "TM0_only");
    REQUIRE(on_disk["crosstalk"]["coherence_model"]
            == "coherent_v2_experimental");
    REQUIRE(on_disk["crosstalk"]["polarization"] == "TM0_only");
}

TEST_CASE("writer rejects mismatched-shape crosstalk payload at write time",
          "[crosstalk][io]") {
    auto g = make_4node_graph();
    apply_uniform_layers(g, 0);
    g.create_subgraphs();

    nlohmann::json bad = nlohmann::json::object();
    bad["unit"]  = "dB";
    bad["shape"] = nlohmann::json::array({4, 4, 4});
    // Build 3×3×3 values — mismatch against declared shape.
    nlohmann::json vals = nlohmann::json::array();
    for (int i = 0; i < 3; ++i) {
        nlohmann::json slab = nlohmann::json::array();
        for (int j = 0; j < 3; ++j) {
            nlohmann::json row = nlohmann::json::array();
            for (int t = 0; t < 3; ++t) row.push_back(nullptr);
            slab.push_back(std::move(row));
        }
        vals.push_back(std::move(slab));
    }
    bad["values"] = std::move(vals);

    REQUIRE_THROWS_AS(
        subgraphsdata_v1x_to_json(g, {}, "2.0", &bad),
        std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Robustness hardening (Codex P2 finding 2, post-79f5e79) — the
// |Δlayer| adjacency classification site shared by loss_function and
// the crosstalk DFS. After the P2-O5 raw-value/no-clamp contract a
// direct/fixed-layers caller can feed pathological accepted `int`
// layer values; `std::abs(la - lb)` then overflows the signed
// subtraction (UB; with two's-complement wrap INT_MAX - INT_MIN folds
// to -1, so |Δ| spuriously reads as 1 == "adjacent"). Widening the
// difference to int64 keeps every in-range / large-but-safe value
// bit-identical while making extreme-opposite layers correctly
// decouple (|Δ| ≥ 2 → no contribution, §2-3 目标 D).
// ---------------------------------------------------------------------------

TEST_CASE("loss_function |Δlayer|==1 classification does not overflow "
          "on extreme int layer inputs", "[crosstalk][graph]") {
    // cl=0, tl=0, lic>0 → per-edge loss is purely the interlayer term,
    // so the return is exactly lic for genuinely adjacent crossing
    // edges and 0 when they decouple. The only crossing pair in the
    // 4-node square is the diagonals (0,2)/(1,3).
    auto g = make_4node_graph(/*L=*/2, /*ecl=*/0, /*perimeter=*/0,
                              /*intra=*/std::nullopt, /*inter=*/std::nullopt,
                              /*loss_crossing=*/0.0, /*loss_taper=*/0.0,
                              /*loss_interlayer_crossing=*/0.006);
    g.set_loss_weights(1.0, 0.0);
    g.build_crossings_index();
    const auto edge_index = build_edge_index(g);
    const int diag_02 = edge_index.at({0, 2});
    const int diag_13 = edge_index.at({1, 3});
    const std::size_t n = static_cast<std::size_t>(g.num_edges());

    constexpr int kIntMax = std::numeric_limits<int>::max();
    constexpr int kIntMin = std::numeric_limits<int>::min();

    // Extreme-opposite layers: |Δ| = 4294967295, must decouple → 0.
    // Pre-fix the int subtraction wraps to -1 → |Δ|==1 → counted as an
    // interlayer crossing → loss == lic (RED). Post-fix → 0 (GREEN).
    {
        std::vector<int> layers(n, kIntMax);
        layers[static_cast<std::size_t>(diag_13)] = kIntMin;
        REQUIRE(g.loss_function(layers) == 0.0);
    }
    // Control — genuine adjacency (|Δ|==1) still counts. Both
    // diagonals selected via unique_in; the 4 perimeter edges pin to
    // layer 0 and are also selected (∈ {0,1}) but carry zero inter.
    // mean = (lic + lic + 0*4) / 6 = lic / 3.
    {
        std::vector<int> layers(n, 0);
        layers[static_cast<std::size_t>(diag_13)] = 1;
        REQUIRE(g.loss_function(layers) == Approx(0.006 / 3.0));
    }
    // Control — large-but-safe non-adjacency (|Δ| = 1e6) decouples
    // identically pre- and post-fix (no overflow at this magnitude).
    {
        std::vector<int> layers(n, 0);
        layers[static_cast<std::size_t>(diag_13)] = 1000000;
        REQUIRE(g.loss_function(layers) == 0.0);
    }
}

TEST_CASE("crosstalk |Δlayer|==1 classification does not overflow on "
          "extreme int layer inputs", "[crosstalk]") {
    auto g = make_4node_graph();  // intra=1e-4, inter=1e-5
    g.build_crossings_index();
    const auto edge_index = build_edge_index(g);
    const int diag_02 = edge_index.at({0, 2});
    const int diag_13 = edge_index.at({1, 3});
    const std::size_t n = static_cast<std::size_t>(g.num_edges());

    constexpr int kIntMax = std::numeric_limits<int>::max();
    constexpr int kIntMin = std::numeric_limits<int>::min();

    // Extreme-opposite layers → |Δ| huge → decouple (mirrors the
    // existing |Δlayer|=2 decouple case). Pre-fix the wrapped int
    // subtraction reads |Δ|==1 → spurious interlayer coupling → the
    // (0,2)↔(1,3) victims are NON-null (RED). Post-fix → null (GREEN).
    {
        std::vector<int> layers(n, 1);
        layers[static_cast<std::size_t>(diag_02)] = kIntMax;
        layers[static_cast<std::size_t>(diag_13)] = kIntMin;
        g.apply_optimization_result(layers);
        auto ct = compute_crosstalk_tensor(g);
        REQUIRE(ct["values"][0][2][1].is_null());
        REQUIRE(ct["values"][0][2][3].is_null());
        REQUIRE(ct["values"][2][0][1].is_null());
        REQUIRE(ct["values"][2][0][3].is_null());
    }
    // Control — genuine adjacency (|Δ|==1) still couples at the
    // interlayer coefficient (10·log10(0.5·1e-5)).
    {
        std::vector<int> layers(n, 1);
        layers[static_cast<std::size_t>(diag_02)] = 1;
        layers[static_cast<std::size_t>(diag_13)] = 0;
        g.apply_optimization_result(layers);
        auto ct = compute_crosstalk_tensor(g);
        const double expected_db = 10.0 * std::log10(0.5 * 1e-5);
        for (int s : {0, 2}) {
            const int d = (s == 0) ? 2 : 0;
            for (int t : {1, 3}) {
                const auto& v = ct["values"][s][d][t];
                REQUIRE_FALSE(v.is_null());
                REQUIRE(std::abs(v.get<double>() - expected_db) < 1e-6);
            }
        }
    }
    // Control — large-but-safe non-adjacency (|Δ| = 1e6) decouples
    // identically pre- and post-fix.
    {
        std::vector<int> layers(n, 1);
        layers[static_cast<std::size_t>(diag_02)] = 1;
        layers[static_cast<std::size_t>(diag_13)] = 1000001;
        g.apply_optimization_result(layers);
        auto ct = compute_crosstalk_tensor(g);
        REQUIRE(ct["values"][0][2][1].is_null());
        REQUIRE(ct["values"][0][2][3].is_null());
    }
}
