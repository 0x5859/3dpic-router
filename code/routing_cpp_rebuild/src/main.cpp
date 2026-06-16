// 3D-SiNIC routing optimizer — M5 slim entry point.
// Argument parsing + orchestration only; all domain logic lives in
// `sinic_core` (include/sinic/*.hpp). REFACTOR_GOALS.md §1-2 targets
// <200 lines here. M5 opt-in env vars (off by default → pre-M5 parity):
//   SINIC_WRITE_RUN_REPORT  emit run_report.{json,md}
//   SINIC_VALIDATE_SCHEMA   default 1; validate JSON before writing
//   SINIC_LEGACY_WRITER     fall back to pre-M5 writer (skips v1.x fields)
//   SINIC_SEED              deterministic seed (default 0 = random_device).
// M7 opt-in env vars (default behavior bit-exact with M6):
//   LOSS_INTRA_CROSSTALK    intralayer crosstalk fractional power
//                           (e.g. 1e-4 = -40 dB). Unset/empty → None.
//   LOSS_INTER_CROSSTALK    interlayer crosstalk fractional power
//                           (e.g. 1e-5 = -50 dB). Unset/empty → None.
//   CROSSTALK_MAX_HOPS      DFS recursion depth (default 3).
//   CROSSTALK_THRESHOLD_DB  prune threshold (default -60.0).
//   COMPUTE_CROSSTALK       explicit on/off (default: on iff at least
//                           one coefficient is **non-zero**; explicit
//                           zeros are treated the same as unset per
//                           §2-2 实施摘要 "None 或 0" rule).
// M8 parity-driver env vars (REFACTOR_GOALS.md §6 T6):
//   SINIC_FIXED_LAYERS_JSON path to a JSON file holding a fixed
//                           per-edge layer assignment to use INSTEAD of
//                           running the optimizer. File schema:
//                              {"layers": [<int>, ..., <int>]}
//                           Length must equal graph.num_edges(). When
//                           set, `dual_annealing::minimize` is skipped
//                           entirely (no `trace[]` is emitted; no
//                           `optimization_wall_ms` phase fires) and
//                           the loaded vector is applied via
//                           `graph.apply_optimization_result`. Used by
//                           `code/tests/test_parity.py` to feed
//                           identical assignments to both Python and
//                           C++ for bit-exact per-edge / per-loss
//                           comparison. The optimizer-trajectory
//                           cross-impl delta is out of scope for this
//                           driver (see §1-2-b: a separate column
//                           reports `cross_impl_delta` from independent
//                           optimizer runs). MUTUALLY EXCLUSIVE with
//                           `SINIC_WRITE_RUN_REPORT` (an empty trace[]
//                           would silently produce a misleading
//                           run_report) and `SINIC_KEEP_LAST_EVAL_STATE`
//                           (a fixed layer assignment IS the desired
//                           post-optimize state). Combinations raise
//                           early, mirroring Python's
//                           `api.py::run_optimization::fixed_layers_mode`
//                           which raises on `collect_statistics=True`.
//   SINIC_LOAD_AND_EXIT_JSON path to a `subgraphsdata.json`. If set,
//                           the binary calls `load_subgraphsdata(path)`
//                           and exits 0 on success / 1 on error. Used
//                           by `test_parity.py::test_cpp_loads_python_json`
//                           to verify the §1-2-b第3条 "C++ can load
//                           Python-produced JSON" direction (the M5
//                           self-roundtrip covered only C++ → C++; M8
//                           extends to Python → C++).
// M-C experiment-harness env var (REFACTOR_GOALS.md §1-2; spec D6/§8):
//   SINIC_POSITIONS_JSON    path to a JSON file
//                              {"<node>":[x,y], ...}
//                           with dense canonical keys 0..k-1. When set,
//                           positions are loaded from this file INSTEAD
//                           of the built-in square generator; k =
//                           #positions and the nps sweep runs exactly
//                           once. Default/unset ("") → the pre-C1 square
//                           path is bit-exact. A malformed or missing
//                           file produces a clean stderr message and
//                           exit 1 (mirrors SINIC_LOAD_AND_EXIT_JSON's
//                           opt-in style). Added by M-C for the
//                           experiment harness.

#include "sinic/crosstalk.hpp"
#include "sinic/graph.hpp"
#include "sinic/io.hpp"
#include "sinic/optimizer.hpp"
#include "sinic/positions.hpp"
#include "sinic/statistics.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::stoi(v) : def;
}

std::string env_str(const char* name, std::string def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : def;
}

std::vector<double> env_list(const char* name, std::vector<double> def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    std::vector<double> out;
    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stod(tok));
    }
    return out.empty() ? def : out;
}

bool env_flag(const char* name, bool def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "no");
}

// M7: optional double env var; returns std::nullopt when unset/empty.
// Matches Python's `compute_crosstalk` short-circuit which treats
// `None` exactly as "no crosstalk coefficient declared".
std::optional<double> env_optional_double(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return std::nullopt;
    return std::stod(v);
}

std::string make_run_id(int k, int dualsa_iter, double cl, double tl, double itl) {
    std::ostringstream os;
    os << "k" << k << "_dualSA_iter" << dualsa_iter
       << "_cl" << std::fixed << std::setprecision(3) << cl
       << "_tl" << std::fixed << std::setprecision(3) << tl
       << "_itl" << std::fixed << std::setprecision(3) << itl;
    return os.str();
}

nlohmann::json make_run_config(int k, int iter, double cl, double tl,
                                double itl, int seed,
                                int L, int ecl, int perimeter_lyr,
                                double pitch, int wpl,
                                std::optional<double> intra_xt,
                                std::optional<double> inter_xt,
                                int crosstalk_max_hops,
                                double crosstalk_threshold_db,
                                bool crosstalk_include_in_loss) {
    nlohmann::json j = {{"k", k}, {"maxiter", iter},
                        {"loss_crossing", cl}, {"loss_taper", tl},
                        {"loss_interlayer_crossing", itl},
                        {"seed", seed}, {"optimizer", "dual_annealing"},
                        {"L", L},
                        {"edge_coupler_layer", ecl},
                        {"perimeter_layer", perimeter_lyr},
                        {"layer_pitch_um", pitch},
                        {"waveguides_per_link", wpl},
                        {"crosstalk_max_hops", crosstalk_max_hops},
                        {"crosstalk_threshold_db", crosstalk_threshold_db},
                        {"crosstalk_include_in_loss",
                         crosstalk_include_in_loss}};
    j["loss_intralayer_crosstalk"] =
        intra_xt ? nlohmann::json(*intra_xt) : nlohmann::json(nullptr);
    j["loss_interlayer_crosstalk"] =
        inter_xt ? nlohmann::json(*inter_xt) : nlohmann::json(nullptr);
    return j;
}

} // namespace

int main() {
    using namespace sinic;

    // --- M8 SINIC_LOAD_AND_EXIT_JSON early-exit path ---------------
    // Runs before any other config so a load probe doesn't have to
    // satisfy the parameter-scan env-var contract. Used by
    // `test_parity.py::test_cpp_loads_python_json` to verify the
    // §1-2-b第3条 "C++ can load Python-produced subgraphsdata.json"
    // direction (the M5 self-roundtrip covered C++ → C++ only).
    {
        const char* load_path = std::getenv("SINIC_LOAD_AND_EXIT_JSON");
        if (load_path && *load_path) {
            try {
                auto loaded = load_subgraphsdata(load_path, /*validate=*/true);
                std::cout << "SINIC_LOAD_AND_EXIT_JSON: loaded "
                          << loaded.sub_graphs.size() << " layer(s); "
                          << "schema_version="
                          << (loaded.schema_version
                                  ? *loaded.schema_version
                                  : std::string("<absent>"))
                          << "\n";
                return 0;
            } catch (const std::exception& e) {
                std::cerr << "SINIC_LOAD_AND_EXIT_JSON: load failed: "
                          << e.what() << "\n";
                return 1;
            }
        }
    }

    // --- env config -------------------------------------------------
    const int nodes_min        = env_int("NODES_MIN", 10);
    const int nodes_max        = env_int("NODES_MAX", nodes_min);
    const int dualsa_iter      = env_int("DUALSA_ITER", 500);
    const auto cl_values       = env_list("CL_VALUES",  {0.1});
    const auto tl_values       = env_list("TL_VALUES",  {0.05});
    const auto itl_values      = env_list("ITL_VALUES", {0.0});
    const std::string base_path = env_str("OUTPUT_DIR", "./output/");
    // C1 (REFACTOR_GOALS.md §1-2): SINIC_POSITIONS_JSON — when set, load
    // node positions from that JSON file instead of the built-in square
    // generator; k = #positions and the nps sweep collapses to a single
    // iteration. Default "" → pre-C1 square path is bit-exact. Mirrors
    // the SINIC_LOAD_AND_EXIT_JSON opt-in style.
    const std::string positions_json_path =
        env_str("SINIC_POSITIONS_JSON", "");
    const bool positions_json_mode = !positions_json_path.empty();
    PositionMap loaded_positions;
    if (positions_json_mode) {
        try {
            loaded_positions = load_positions_from_json(positions_json_path);
        } catch (const std::exception& e) {
            std::cerr << "SINIC_POSITIONS_JSON: " << e.what() << "\n";
            return 1;
        }
        std::cout << "SINIC_POSITIONS_JSON: loaded "
                  << loaded_positions.size() << " positions from "
                  << positions_json_path << "\n";
    }
    const bool write_run_report_flag = env_flag("SINIC_WRITE_RUN_REPORT", false);
    const bool validate_schema_flag  = env_flag("SINIC_VALIDATE_SCHEMA",  true);
    const bool use_legacy_writer     = env_flag("SINIC_LEGACY_WRITER",    false);
    const int  optimizer_seed        = env_int ("SINIC_SEED", 0);
    // --- M6 multi-layer env config ---------------------------------
    // L / ecl / perimeter_layer default to legacy (L=2 / ecl=0 /
    // perimeter=ecl) so a fresh checkout reproduces pre-M6 behavior.
    // Use the sentinel -1 for perimeter_layer to mean "= ecl".
    const int num_layers        = env_int ("NUM_LAYERS", 2);
    const int edge_coupler_lyr  = env_int ("EDGE_COUPLER_LAYER", 0);
    const int perimeter_lyr_env = env_int ("PERIMETER_LAYER", -1);
    const int wpl               = env_int ("WAVEGUIDES_PER_LINK", 1);
    const auto layer_pitch_um   = [] {
        const char* v = std::getenv("LAYER_PITCH_UM");
        return (v && *v) ? std::stod(v) : 1.2;
    }();
    // --- M7 crosstalk env config -----------------------------------
    // Default behavior is bit-exact with M6: both coefficients are
    // `nullopt` ⇒ `compute_crosstalk_tensor` short-circuits ⇒
    // `crosstalk` field is omitted from JSON ⇒ no
    // `crosstalk_analysis_ms` PhaseTimer.
    const auto loss_intra_xt = env_optional_double("LOSS_INTRA_CROSSTALK");
    const auto loss_inter_xt = env_optional_double("LOSS_INTER_CROSSTALK");
    const int  crosstalk_max_hops     = env_int("CROSSTALK_MAX_HOPS", 3);
    const auto crosstalk_threshold_db = [] {
        const char* v = std::getenv("CROSSTALK_THRESHOLD_DB");
        return (v && *v) ? std::stod(v) : -60.0;
    }();
    // crosstalk_include_in_loss is accepted as config only — M7 does
    // not feed crosstalk into the loss model (mirrors Python's M4
    // behavior). The flag round-trips into run_report.json.config.
    const bool crosstalk_include_in_loss =
        env_flag("CROSSTALK_INCLUDE_IN_LOSS", false);
    // Engine runs iff `(COMPUTE_CROSSTALK is true) AND (at least one
    // coefficient is non-zero)`. Explicit zero coefs (e.g.
    // `LOSS_INTRA_CROSSTALK=0`) are treated the same as unset —
    // engine is short-circuited, no PhaseTimer fires, no
    // `crosstalk_analysis_ms` phase appears in run_report.
    //
    // This is bit-equivalent to Python's
    // `api.py::run_optimization::has_any_crosstalk and compute_crosstalk`
    // (defined at api.py:347-359) where `has_any_crosstalk` is the
    // "at-least-one-nonzero" predicate.
    //
    // M7 dual-review R1 P1 (codex): pre-fix `any_coef` was
    // `has_value()` so `LOSS_INTRA_CROSSTALK=0` lit the PhaseTimer
    // and emitted an empty tensor; that diverged from Python.
    const bool any_nonzero_coef =
        (loss_intra_xt.has_value() && *loss_intra_xt != 0.0)
     || (loss_inter_xt.has_value() && *loss_inter_xt != 0.0);
    const bool compute_crosstalk =
        any_nonzero_coef && env_flag("COMPUTE_CROSSTALK", true);

    std::cout << "Config: nodes_per_side=" << nodes_min
              << ".." << nodes_max
              << " (total " << nodes_min * 4 << ".." << nodes_max * 4 << " nodes)"
              << ", dualsa_iter="    << dualsa_iter
              << ", L="              << num_layers
              << ", ecl="            << edge_coupler_lyr
              << ", perimeter="      << perimeter_lyr_env
              << ", wpl="            << wpl
              << ", pitch="          << layer_pitch_um
              << ", cl="             << cl_values.size() << " vals"
              << ", tl="             << tl_values.size() << " vals"
              << ", itl="            << itl_values.size() << " vals"
              << ", out="            << base_path
              << ", legacy_writer="  << (use_legacy_writer ? "yes" : "no")
              << ", validate="       << (validate_schema_flag ? "yes" : "no")
              << ", run_report="     << (write_run_report_flag ? "yes" : "no")
              << ", seed="           << optimizer_seed
              << "\n";
    if (positions_json_mode) {
        std::cout << "Config: positions-json mode — nps sweep ignored; k="
                  << loaded_positions.size() << " from "
                  << positions_json_path << "\n";
    }

    // --- parameter scan --------------------------------------------
    // In positions-json mode the nps sweep runs exactly once (positions
    // and k come from the loaded file, not the square generator).
    const int scan_nps_max = positions_json_mode ? nodes_min : nodes_max;
    for (int nps = nodes_min; nps <= scan_nps_max; ++nps) {
        const int k = positions_json_mode
                          ? static_cast<int>(loaded_positions.size())
                          : nps * 4;
        const PositionMap positions =
            positions_json_mode ? loaded_positions
                                : distribute_nodes_around_square(nps, 10.0);
        for (const double cl : cl_values) {
            for (const double tl : tl_values) {
                for (const double itl : itl_values) {
                    std::ostringstream folder;
                    folder << "cl_" << std::fixed << std::setprecision(2) << cl
                           << "_tl_" << std::fixed << std::setprecision(2) << tl
                           << "_itl_" << std::fixed << std::setprecision(3) << itl
                           << "_nodes_" << k;
                    const std::string full_path = base_path + folder.str();
                    std::filesystem::create_directories(full_path);

                    SiNInterconnectionGraph graph(
                        k, positions, cl, tl, itl,
                        num_layers, edge_coupler_lyr,
                        perimeter_lyr_env, layer_pitch_um, wpl,
                        loss_intra_xt, loss_inter_xt);

                    // M6: build Phase A index up front so the optimizer
                    // can `loss_function` without paying the one-time
                    // geometric pass on its first call. Timing emitted
                    // as `initial_crossing_count` phase if a recorder
                    // is wired in (mirrors `api.py::run_optimization`).

                    // StatsSink wiring (M5 §1-1-b).
                    std::unique_ptr<StatsSink> sink_owner;
                    StatsSink* sink_ptr = &default_null_sink();
                    if (write_run_report_flag) {
                        sink_owner = std::make_unique<RunRecorder>(
                            make_run_id(k, dualsa_iter, cl, tl, itl),
                            make_run_config(k, dualsa_iter, cl, tl, itl,
                                             optimizer_seed, num_layers,
                                             edge_coupler_lyr,
                                             graph.perimeter_layer(),
                                             layer_pitch_um, wpl,
                                             loss_intra_xt, loss_inter_xt,
                                             crosstalk_max_hops,
                                             crosstalk_threshold_db,
                                             crosstalk_include_in_loss));
                        sink_ptr = sink_owner.get();
                    }
                    {
                        PhaseTimer t(*sink_ptr, "initial_crossing_count");
                        graph.build_crossings_index();
                    }

                    DualAnnealingParams params;
                    params.max_iter = dualsa_iter;
                    params.seed     = optimizer_seed;

                    // M8 parity-driver path. When the env var is set
                    // we skip optimization entirely and load the
                    // assignment from disk; otherwise we run the
                    // optimizer as normal.
                    //
                    // Two mutual-exclusion guards mirror Python
                    // `api.py::run_optimization::fixed_layers_mode`:
                    //   * `SINIC_WRITE_RUN_REPORT` — a skipped
                    //     optimizer leaves `trace[]` empty; while
                    //     `run_report.schema.json` does NOT require a
                    //     non-empty trace, a "ran the optimizer 0
                    //     times" run_report is semantically misleading
                    //     and the Python side raises symmetrically.
                    //   * `SINIC_KEEP_LAST_EVAL_STATE` — the M5
                    //     dual-review P1 escape hatch only makes sense
                    //     when an optimizer left graph.allEdges_ at a
                    //     LAST-eval state; fixed layers ARE the
                    //     desired post-state.
                    // Combinations raise early so the user sees the
                    // conflict instead of silent misbehavior.
                    const std::string fixed_layers_path =
                        env_str("SINIC_FIXED_LAYERS_JSON", "");
                    std::vector<int> best_layers;
                    if (!fixed_layers_path.empty()) {
                        if (write_run_report_flag) {
                            throw std::runtime_error(
                                "SINIC_FIXED_LAYERS_JSON and "
                                "SINIC_WRITE_RUN_REPORT are mutually "
                                "exclusive: skipping the optimizer "
                                "leaves trace[] empty and the run "
                                "report would be semantically "
                                "misleading. Mirror of Python "
                                "`api.py::run_optimization` "
                                "fixed_layers + collect_statistics "
                                "raise.");
                        }
                        if (env_flag("SINIC_KEEP_LAST_EVAL_STATE", false)) {
                            throw std::runtime_error(
                                "SINIC_FIXED_LAYERS_JSON and "
                                "SINIC_KEEP_LAST_EVAL_STATE are mutually "
                                "exclusive: a fixed layer assignment IS "
                                "the desired post-optimize state, "
                                "rendering the escape hatch meaningless.");
                        }
                        std::ifstream fl_ifs(fixed_layers_path);
                        if (!fl_ifs) {
                            throw std::runtime_error(
                                "SINIC_FIXED_LAYERS_JSON: cannot open " +
                                fixed_layers_path);
                        }
                        nlohmann::json fl_j;
                        try {
                            fl_ifs >> fl_j;
                        } catch (const nlohmann::json::exception& e) {
                            throw std::runtime_error(
                                std::string("SINIC_FIXED_LAYERS_JSON: "
                                            "malformed JSON in ") +
                                fixed_layers_path + ": " + e.what());
                        }
                        if (!fl_j.contains("layers") ||
                            !fl_j["layers"].is_array()) {
                            throw std::runtime_error(
                                "SINIC_FIXED_LAYERS_JSON: missing 'layers' "
                                "array in " + fixed_layers_path);
                        }
                        const auto& arr = fl_j["layers"];
                        if (static_cast<int>(arr.size()) !=
                                graph.num_edges()) {
                            std::ostringstream msg;
                            msg << "SINIC_FIXED_LAYERS_JSON: 'layers' has "
                                << arr.size() << " entries but graph has "
                                << graph.num_edges() << " edges";
                            throw std::runtime_error(msg.str());
                        }
                        best_layers.reserve(arr.size());
                        for (std::size_t i = 0; i < arr.size(); ++i) {
                            try {
                                best_layers.push_back(arr[i].get<int>());
                            } catch (const nlohmann::json::exception& e) {
                                std::ostringstream msg;
                                msg << "SINIC_FIXED_LAYERS_JSON: 'layers'["
                                    << i << "] is not an integer: "
                                    << e.what();
                                throw std::runtime_error(msg.str());
                            }
                        }
                        graph.apply_optimization_result(best_layers);
                    } else {
                        // optimization_wall_ms is emitted from inside
                        // `optimize()` to keep trace[].wall_ms and the
                        // phase timing on a single steady_clock origin
                        // (M5 dual-review P1 fix).
                        DualAnnealingOptimizer opt(graph, params, *sink_ptr);
                        opt.optimize();
                        best_layers = opt.best_layers();
                        // M5 dual-review P1 fix: pre-M5 left
                        // graph.allEdges_ at the LAST-eval state (not the
                        // best). Apply the optimizer's best layers so the
                        // saved JSON / analyze_loss output reflects the
                        // actual optimum. The legacy baseline behavior is
                        // recoverable by skipping this line (set
                        // SINIC_KEEP_LAST_EVAL_STATE=1).
                        if (!env_flag("SINIC_KEEP_LAST_EVAL_STATE", false)) {
                            graph.apply_optimization_result(best_layers);
                        }
                    }
                    {
                        PhaseTimer t(*sink_ptr, "loss_analysis");
                        graph.analyze_loss(std::cout);
                    }

                    // M7: compute crosstalk tensor once post-optimization
                    // (mirror of Python `api.py::run_optimization`). Only
                    // engages when at least one coefficient is set AND the
                    // user hasn't opted out via `COMPUTE_CROSSTALK=0`. The
                    // engine short-circuit handles zero/None coefs already
                    // (cheap), but we skip the PhaseTimer too so the JSON
                    // `crosstalk_analysis_ms` field is absent in the M6-
                    // baseline path (matches Python's
                    // `test_run_report_no_crosstalk_phase_when_disabled`).
                    nlohmann::json crosstalk_payload;
                    const nlohmann::json* crosstalk_for_writer = nullptr;
                    if (compute_crosstalk) {
                        PhaseTimer t(*sink_ptr, "crosstalk_analysis");
                        crosstalk_payload = compute_crosstalk_tensor(
                            graph, crosstalk_max_hops,
                            crosstalk_threshold_db);
                        crosstalk_for_writer = &crosstalk_payload;
                    }

                    std::map<std::string, double> gen_params;
                    gen_params["Nodes"] = static_cast<double>(k);
                    const std::string json_path = full_path + "/subgraphsdata.json";
                    // M8: opt-in env var lets the parity harness pin the
                    // writer to v1.x even on the default
                    // `save_subgraphsdata_v1x` path. Default "" keeps
                    // WriteOptions::schema_version = "2.0" (the M6
                    // default).
                    const std::string schema_version_override =
                        env_str("SINIC_SCHEMA_VERSION", "");
                    {
                        PhaseTimer t(*sink_ptr, "json_write");
                        if (use_legacy_writer) {
                            // Legacy writer is bit-exact pre-M5 — no
                            // crosstalk field even when computed.
                            save_subgraphs_legacy_newformat(graph, json_path, gen_params);
                        } else {
                            WriteOptions wopts;
                            wopts.validate = validate_schema_flag;
                            if (!schema_version_override.empty()) {
                                wopts.schema_version = schema_version_override;
                            }
                            save_subgraphsdata_v1x(graph, json_path, gen_params,
                                                    wopts, crosstalk_for_writer);
                        }
                    }
                    if (auto* rec = dynamic_cast<RunRecorder*>(sink_ptr)) {
                        // Per-layer aggregate stats (§1-1-a item 3). M6
                        // stage-review P1-3 fix (Python parity): emit
                        // **all L** layer entries (including empty
                        // layers with `edge_count=0` / `crossings=0`)
                        // so downstream consumers can index by layer
                        // number. Mirrors
                        // `routing_py_rebuild/api.py::_extract_aggregate_stats`.
                        //
                        // crosslayer total formula (§4 M3 附注):
                        //   crosslayer_crossings_total =
                        //     waveguides_per_link * Σ_e interlayercrossings_above
                        // Each cross-layer geometric event is stamped exactly
                        // once on the LOWER edge's `_above`, so summing
                        // `_above` across all buckets == geometric event
                        // count.
                        //
                        // Convention naming follows §4 M3 附注:
                        //   - wpl=2 → "physical" (loss-model-aligned)
                        //   - wpl=1 → "geometric" (legacy stats口径)
                        //   - wpl ∉ {1,2} → "experimental"
                        std::vector<RunRecorder::LayerAggregate> aggregates;
                        long geometric_events = 0;
                        for (std::size_t li = 0; li < graph.sub_graphs().size(); ++li) {
                            const auto& sg = graph.sub_graphs()[li];
                            const int layer_id = static_cast<int>(li);
                            int total_intra = 0;
                            int total_above = 0;
                            for (const auto& e : sg) {
                                total_intra += std::get<2>(e).crossings;
                                total_above += std::get<2>(e).interlayerCrossingsAbove;
                            }
                            aggregates.push_back({layer_id,
                                                   static_cast<int>(sg.size()),
                                                   total_intra / 2});
                            geometric_events += total_above;
                        }
                        const long crosslayer_total =
                            static_cast<long>(wpl) * geometric_events;
                        const char* convention =
                            (wpl == 2) ? "physical"
                            : (wpl == 1) ? "geometric"
                            : "experimental";
                        rec->set_aggregate(std::move(aggregates),
                                            crosslayer_total,
                                            std::string(convention));
                        auto report = rec->finalize();
                        write_run_report(report, full_path);
                    }
                    std::cout << "Completed: " << folder.str() << "\n";
                }
            }
        }
    }
    std::cout << "All parameter combinations scanned!\n";
    return 0;
}
