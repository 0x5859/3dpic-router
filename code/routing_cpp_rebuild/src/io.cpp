#include "sinic/io.hpp"

#include "sinic/schema_validator.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sinic {

namespace {

// Sorting predicate so on-disk positions / complete_graph orderings
// are deterministic across runs. The legacy writer relied on edge
// vector insertion order (u < v from constructor); we preserve that.

nlohmann::json positions_to_json(const PositionMap& positions) {
    nlohmann::json j = nlohmann::json::object();
    std::vector<int> keys;
    keys.reserve(positions.size());
    for (const auto& kv : positions) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (int k : keys) {
        const auto& p = positions.at(k);
        j[std::to_string(k)] = nlohmann::json::array({p.first, p.second});
    }
    return j;
}

// Legacy v1.x edge attrs — no _above / _below fields.
nlohmann::json edge_array_legacy(const Edge& e) {
    // `[u, v, {layer, crossings, loss, interlayercrossings}]`
    const int u = std::get<0>(e);
    const int v = std::get<1>(e);
    const auto& ed = std::get<2>(e);
    nlohmann::json attrs = nlohmann::json::object();
    attrs["layer"]               = ed.layer;
    attrs["crossings"]           = ed.crossings;
    attrs["loss"]                = ed.loss;
    attrs["interlayercrossings"] = ed.interlayerCrossings;
    return nlohmann::json::array({u, v, attrs});
}

// M6 v2.0 edge attrs — adds `interlayercrossings_above` /
// `interlayercrossings_below` per REFACTOR_GOALS.md §3-1.
nlohmann::json edge_array_v2(const Edge& e) {
    const int u = std::get<0>(e);
    const int v = std::get<1>(e);
    const auto& ed = std::get<2>(e);
    nlohmann::json attrs = nlohmann::json::object();
    attrs["layer"]                      = ed.layer;
    attrs["crossings"]                  = ed.crossings;
    attrs["loss"]                       = ed.loss;
    attrs["interlayercrossings"]        = ed.interlayerCrossings;
    attrs["interlayercrossings_above"]  = ed.interlayerCrossingsAbove;
    attrs["interlayercrossings_below"]  = ed.interlayerCrossingsBelow;
    return nlohmann::json::array({u, v, attrs});
}

nlohmann::json layer_to_json(const EdgeList& layer, bool v2_fields) {
    nlohmann::json layer_json = nlohmann::json::object();
    nlohmann::json edges_arr  = nlohmann::json::array();
    for (const auto& e : layer) {
        edges_arr.push_back(v2_fields ? edge_array_v2(e)
                                       : edge_array_legacy(e));
    }
    layer_json["edges"] = edges_arr;
    return layer_json;
}

struct LayerStats {
    double avg, var, stdev, range;
    std::size_t count;
};

LayerStats compute_stats(const std::vector<double>& xs) {
    LayerStats s{0, 0, 0, 0, 0};
    s.count = xs.size();
    if (xs.empty()) return s;
    double sum = 0.0;
    for (auto v : xs) sum += v;
    s.avg = sum / xs.size();
    double var = 0.0;
    double mn = xs.front();
    double mx = xs.front();
    for (auto v : xs) {
        const double d = v - s.avg;
        var += d * d;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    s.var   = var / xs.size();
    s.stdev = std::sqrt(s.var);
    s.range = mx - mn;
    return s;
}

nlohmann::json loss_analysis(const SiNInterconnectionGraph& graph) {
    // Mirror Python `core.analyze_loss` output. Pre-M5 C++ printed
    // these to stdout but did not persist; M5 captures them in JSON.
    //
    // M6 stage-review P1 fix (Python parity): empty layer buckets emit
    // JSON `null` in the per-layer aggregate arrays (`avg_loss_subgraphs`,
    // `var_loss_subgraphs`, `std_loss_subgraphs`, `range_loss_subgraphs`).
    // Schema permits `[number, null]` on each leaf; readers (Python
    // `PlotData.from_json`) tell "layer unoccupied" from "layer at
    // zero loss" via the null marker.
    nlohmann::json out = nlohmann::json::object();
    nlohmann::json avg_loss = nlohmann::json::array();
    nlohmann::json var_loss = nlohmann::json::array();
    nlohmann::json std_loss = nlohmann::json::array();
    nlohmann::json range_loss = nlohmann::json::array();
    std::vector<double> all_vals;
    for (const auto& sg : graph.sub_graphs()) {
        std::vector<double> vals;
        for (const auto& e : sg) vals.push_back(std::get<2>(e).loss);
        if (vals.empty()) {
            avg_loss.push_back(nullptr);
            var_loss.push_back(nullptr);
            std_loss.push_back(nullptr);
            range_loss.push_back(nullptr);
            continue;
        }
        const auto s = compute_stats(vals);
        avg_loss.push_back(s.avg);
        var_loss.push_back(s.var);
        std_loss.push_back(s.stdev);
        range_loss.push_back(s.range);
        for (auto v : vals) all_vals.push_back(v);
    }
    out["avg_loss_subgraphs"]   = avg_loss;
    out["var_loss_subgraphs"]   = var_loss;
    out["std_loss_subgraphs"]   = std_loss;
    out["range_loss_subgraphs"] = range_loss;
    // M6 stage-2 review fix (codex P1-2): emit the full
    // `loss_analysis` field set — Python `core.py::analyze_loss`
    // produces 12 keys (`avg/var/std/range` × 3 scopes:
    // per-layer / flattened / completegraph). Pre-fix C++ only
    // emitted 6 keys, breaking `plotting/loss_analysis.py` which
    // indexes the full set unconditionally.
    if (!all_vals.empty()) {
        const auto s = compute_stats(all_vals);
        out["avg_flattened_loss_subgraphs"]   = s.avg;
        out["var_flattened_loss_subgraphs"]   = s.var;
        out["std_flattened_loss_subgraphs"]   = s.stdev;
        out["range_flattened_loss_subgraphs"] = s.range;
    } else {
        out["avg_flattened_loss_subgraphs"]   = nullptr;
        out["var_flattened_loss_subgraphs"]   = nullptr;
        out["std_flattened_loss_subgraphs"]   = nullptr;
        out["range_flattened_loss_subgraphs"] = nullptr;
    }
    std::vector<double> planar_vals;
    for (const auto& e : graph.planar_reference()) {
        planar_vals.push_back(std::get<2>(e).loss);
    }
    if (!planar_vals.empty()) {
        const auto s = compute_stats(planar_vals);
        out["avg_loss_completegraph"]   = s.avg;
        out["var_loss_completegraph"]   = s.var;
        out["std_loss_completegraph"]   = s.stdev;
        out["range_loss_completegraph"] = s.range;
    } else {
        out["avg_loss_completegraph"]   = nullptr;
        out["var_loss_completegraph"]   = nullptr;
        out["std_loss_completegraph"]   = nullptr;
        out["range_loss_completegraph"] = nullptr;
    }
    return out;
}

nlohmann::json complete_graph_to_json(const EdgeList& planar, bool v2_fields) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& e : planar) {
        arr.push_back(v2_fields ? edge_array_v2(e) : edge_array_legacy(e));
    }
    return arr;
}

} // namespace

void validate_crosstalk_payload_shape(const nlohmann::json& crosstalk,
                                      int expected_k) {
    // Mirror of Python `core.py::_validate_crosstalk_payload_shape`.
    // We run this before the structural schema check so callers see
    // the precise mismatch reason (Python `ValueError` ↔ C++
    // `std::invalid_argument`).
    if (!crosstalk.is_object()) {
        throw std::invalid_argument(
            "crosstalk payload must be an object");
    }
    if (!crosstalk.contains("shape") || !crosstalk["shape"].is_array()) {
        throw std::invalid_argument(
            "crosstalk.shape must be a 3-element array of integers");
    }
    const auto& shape = crosstalk["shape"];
    if (shape.size() != 3) {
        throw std::invalid_argument(
            "crosstalk.shape must be a 3-element array of integers");
    }
    int dims[3] = {0, 0, 0};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto& v = shape[i];
        // `bool` is a separate `value_t` in nlohmann::json (unlike
        // Python where `bool` is a subclass of `int`); explicit
        // rejection here keeps the error message uniform with Python.
        if (v.is_boolean() || !v.is_number_integer()) {
            throw std::invalid_argument(
                "crosstalk.shape must be a 3-element array of integers");
        }
        const int dim = v.get<int>();
        // M7 dual-review R1 P1 (opus): Python validator at
        // `core.py::_validate_crosstalk_payload_shape` rejects `x <= 0`,
        // not just `x < 0`. Tighten C++ to match — protects against a
        // degenerate `shape=[0,0,0]` payload sneaking through when
        // expected_k=0 (which would otherwise pass the `dim !=
        // expected_k` check below).
        if (dim <= 0) {
            throw std::invalid_argument(
                "crosstalk.shape entries must be positive integers");
        }
        dims[i] = dim;
    }
    if (dims[0] != expected_k || dims[1] != expected_k
        || dims[2] != expected_k) {
        std::ostringstream os;
        os << "crosstalk.shape [" << dims[0] << ", " << dims[1] << ", "
           << dims[2] << "] does not match graph k=" << expected_k;
        throw std::invalid_argument(os.str());
    }
    if (!crosstalk.contains("values") || !crosstalk["values"].is_array()) {
        throw std::invalid_argument(
            "crosstalk.values must be a rank-3 array");
    }
    const auto& values = crosstalk["values"];
    if (static_cast<int>(values.size()) != dims[0]) {
        std::ostringstream os;
        os << "crosstalk.values length " << values.size()
           << " does not match shape[0]=" << dims[0];
        throw std::invalid_argument(os.str());
    }
    for (int i = 0; i < dims[0]; ++i) {
        const auto& slab = values[i];
        if (!slab.is_array()
            || static_cast<int>(slab.size()) != dims[1]) {
            std::ostringstream os;
            os << "crosstalk.values[" << i << "] length "
               << (slab.is_array() ? std::to_string(slab.size()) : "<not-list>")
               << " does not match shape[1]=" << dims[1];
            throw std::invalid_argument(os.str());
        }
        for (int j = 0; j < dims[1]; ++j) {
            const auto& row = slab[j];
            if (!row.is_array()
                || static_cast<int>(row.size()) != dims[2]) {
                std::ostringstream os;
                os << "crosstalk.values[" << i << "][" << j << "] length "
                   << (row.is_array() ? std::to_string(row.size()) : "<not-list>")
                   << " does not match shape[2]=" << dims[2];
                throw std::invalid_argument(os.str());
            }
        }
    }
}

nlohmann::json subgraphsdata_v1x_to_json(
    const SiNInterconnectionGraph& graph,
    const std::map<std::string, double>& general_params,
    const std::string& schema_version,
    const nlohmann::json* crosstalk) {
    // M8 parity contract: any new General-Parameters key emitted below
    // (or via the `general_params` argument from main.cpp) MUST either
    // (a) be mirrored on the Python side in
    // `routing_py_rebuild/core.py::save_subgraphs_to_json` OR (b) be
    // added to `code/tests/parity_harness.py::KNOWN_GP_ASYMMETRIES`.
    // The parity test fails loudly on unknown drift; this comment
    // exists so the first thing a developer touching the GP block sees
    // is the parity boundary.
    //
    // M6 (REFACTOR_GOALS.md §3-1): the writer emits the multi-layer
    // General Parameters set + per-edge _above / _below when the
    // requested schema_version is "2.0". For "1.x" / empty it stays at
    // the v1.x field set (legacy bit-exact superset).
    const bool v2_mode = (schema_version == "2.0");

    nlohmann::json j = nlohmann::json::object();
    if (!schema_version.empty()) {
        j["schema_version"] = schema_version;
    }

    nlohmann::json gp = nlohmann::json::object();
    gp["k"]                     = graph.k();
    gp["Loss of Taper"]         = graph.loss_taper();
    gp["Loss of Crossing"]      = graph.loss_crossing();
    gp["Loss of Interlayer Crossing"] = graph.loss_interlayer_crossing();
    if (v2_mode) {
        // v2.0 mandatory fields per `subgraphsdata.schema.json::allOf.if/then`.
        gp["L"]                   = graph.L();
        gp["edge_coupler_layer"]  = graph.edge_coupler_layer();
        gp["perimeter_layer"]     = graph.perimeter_layer();
        gp["waveguides_per_link"] = graph.waveguides_per_link();
        gp["layer_pitch_um"]      = graph.layer_pitch_um();
        // M7: crosstalk coefficients + physical-model labels are now
        // sourced from the graph instance (vs. M6's hard-coded
        // `nullptr` / "incoherent_v1" / "TE0_only" placeholders).
        if (graph.loss_intralayer_crosstalk()) {
            gp["Loss of Intralayer Crosstalk"] =
                *graph.loss_intralayer_crosstalk();
        } else {
            gp["Loss of Intralayer Crosstalk"] = nullptr;
        }
        if (graph.loss_interlayer_crosstalk()) {
            gp["Loss of Interlayer Crosstalk"] =
                *graph.loss_interlayer_crosstalk();
        } else {
            gp["Loss of Interlayer Crosstalk"] = nullptr;
        }
        gp["coherence_model"]     = graph.coherence_model();
        gp["polarization"]        = graph.polarization();
    }
    for (const auto& kv : general_params) {
        gp[kv.first] = kv.second;
    }
    j["General Parameters"] = gp;

    j["positions"]      = positions_to_json(graph.positions());
    j["complete_graph"] = complete_graph_to_json(graph.planar_reference(), v2_mode);
    j["loss_analysis"]  = loss_analysis(graph);

    for (std::size_t i = 0; i < graph.sub_graphs().size(); ++i) {
        // M6: emit every layer slot (including empty), so readers can
        // tell "layer i unoccupied" from "layer i absent" (mirrors
        // Python `core.py::save_subgraphs_to_json` writing Layer_0..
        // Layer_{L-1} unconditionally).
        j["Layer_" + std::to_string(i)] =
            layer_to_json(graph.sub_graphs()[i], v2_mode);
    }

    // M7: top-level `crosstalk` field. Only emitted in v2.0 mode
    // (per REFACTOR_GOALS.md §3-1 — `crosstalk` is a v2.0-only field).
    // Shape-validate before merging so a mismatched payload raises
    // locally rather than failing downstream at jsonschema-validate;
    // mirrors Python's `_validate_crosstalk_payload_shape` defense-
    // in-depth.
    //
    // M7 dual-review R1 P1: v1.x mode previously emitted `crosstalk`
    // when a payload was supplied, contradicting the §3-1 contract.
    // Both Opus and Codex flagged this; the gate is now explicit.
    if (v2_mode && crosstalk != nullptr && !crosstalk->is_null()) {
        validate_crosstalk_payload_shape(*crosstalk, graph.k());
        j["crosstalk"] = *crosstalk;
    }

    return j;
}

void save_subgraphsdata_v1x(
    const SiNInterconnectionGraph& graph,
    const std::string& filename,
    const std::map<std::string, double>& general_params,
    WriteOptions opts,
    const nlohmann::json* crosstalk) {
    const auto j = subgraphsdata_v1x_to_json(graph, general_params,
                                              opts.schema_version,
                                              crosstalk);
    if (opts.validate) {
        validate_subgraphsdata_payload(j);
    }
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("save_subgraphsdata_v1x: cannot open " + filename);
    }
    ofs << std::setw(opts.json_indent) << j << std::endl;
    std::cout << "Subgraphs saved to " << filename << std::endl;
}

nlohmann::json subgraphs_legacy_newformat_to_json(
    const SiNInterconnectionGraph& graph,
    const std::map<std::string, double>& general_params) {
    // Bit-exact replica of pre-M5 main.cpp `save_subgraphs_to_json_newformat`
    // (no schema_version, no positions, no complete_graph, no loss_analysis).
    nlohmann::json j = nlohmann::json::object();
    nlohmann::json gp = nlohmann::json::object();
    gp["Loss of Taper"]    = graph.loss_taper();
    gp["Loss of Crossing"] = graph.loss_crossing();
    for (const auto& kv : general_params) {
        gp[kv.first] = kv.second;
    }
    j["General Parameters"] = gp;
    for (std::size_t i = 0; i < graph.sub_graphs().size(); ++i) {
        // Legacy writer omits _above / _below to stay bit-exact with the
        // pre-M5 main.cpp output.
        j["Layer_" + std::to_string(i)] =
            layer_to_json(graph.sub_graphs()[i], /*v2_fields=*/false);
    }
    return j;
}

void save_subgraphs_legacy_newformat(
    const SiNInterconnectionGraph& graph,
    const std::string& filename,
    const std::map<std::string, double>& general_params) {
    const auto j = subgraphs_legacy_newformat_to_json(graph, general_params);
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("save_subgraphs_legacy_newformat: cannot open "
                                 + filename);
    }
    ofs << std::setw(4) << j << std::endl;
    std::cout << "Subgraphs saved to " << filename << std::endl;
}

LoadedSubgraphs load_subgraphsdata(const std::string& filename, bool validate) {
    std::ifstream ifs(filename);
    if (!ifs) {
        throw std::runtime_error("load_subgraphsdata: cannot open " + filename);
    }
    nlohmann::json j;
    ifs >> j;
    if (validate) {
        validate_subgraphsdata_payload(j);
    }
    LoadedSubgraphs out;
    if (j.contains("schema_version") && j["schema_version"].is_string()) {
        out.schema_version = j["schema_version"].get<std::string>();
    }
    if (j.contains("General Parameters")) {
        out.general_params = j["General Parameters"];
    } else {
        out.general_params = nlohmann::json::object();
    }
    // Reconstruct layer edges. Handles both legacy "newformat" arrays
    // (`[[u, v, {...}], ...]`) and the obsolete pre-newformat dict
    // shape (`[{u, v, layer, ...}, ...]`). Only the array form is
    // emitted by current writers; dict form remains for read-only
    // backward compatibility with older fixtures.
    int layer_idx = 0;
    while (true) {
        const std::string key = "Layer_" + std::to_string(layer_idx);
        if (!j.contains(key)) break;
        const auto& layer_node = j[key];
        EdgeList edges;
        if (layer_node.contains("edges") && layer_node["edges"].is_array()) {
            for (const auto& e : layer_node["edges"]) {
                int u = 0, v = 0;
                EdgeData ed;
                if (e.is_array() && e.size() >= 3) {
                    u = e[0].get<int>();
                    v = e[1].get<int>();
                    const auto& a = e[2];
                    if (a.contains("layer"))               ed.layer = a["layer"].get<int>();
                    if (a.contains("crossings"))           ed.crossings = a["crossings"].get<int>();
                    if (a.contains("loss"))                ed.loss = a["loss"].get<double>();
                    if (a.contains("interlayercrossings")) ed.interlayerCrossings = a["interlayercrossings"].get<int>();
                    // M6: v2.0 fields — gracefully fall back to 0 if
                    // absent so legacy v1.x JSON still loads.
                    if (a.contains("interlayercrossings_above"))
                        ed.interlayerCrossingsAbove = a["interlayercrossings_above"].get<int>();
                    if (a.contains("interlayercrossings_below"))
                        ed.interlayerCrossingsBelow = a["interlayercrossings_below"].get<int>();
                } else if (e.is_object()) {
                    u = e.value("u", 0);
                    v = e.value("v", 0);
                    ed.layer                    = e.value("layer", 0);
                    ed.crossings                = e.value("crossings", 0);
                    ed.loss                     = e.value("loss", 0.0);
                    ed.interlayerCrossings      = e.value("interlayercrossings", 0);
                    ed.interlayerCrossingsAbove = e.value("interlayercrossings_above", 0);
                    ed.interlayerCrossingsBelow = e.value("interlayercrossings_below", 0);
                }
                edges.emplace_back(u, v, ed);
            }
        }
        out.sub_graphs.push_back(std::move(edges));
        ++layer_idx;
    }
    return out;
}

} // namespace sinic
