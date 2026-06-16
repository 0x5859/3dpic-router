#pragma once

// JSON I/O. Two output paths are exposed:
//
//   * `save_subgraphsdata_v1x` — schema-conformant v1.x writer that
//     emits all four required keys (General Parameters, positions,
//     complete_graph, loss_analysis) plus the Layer_i sub-objects.
//     This is the M5 default; the M8 parity fixture compares it to
//     the Python writer.
//
//   * `save_subgraphs_legacy_newformat` — bit-exact replica of
//     pre-M5 main.cpp's `save_subgraphs_to_json_newformat`. Retained
//     for regression tests and for the build_and_run.sh script which
//     compares against existing fixtures in `output/`. Does NOT
//     schema-validate.
//
// Both writers also have a `_to_json` non-IO variant returning the
// JSON object — used by validators and unit tests that don't want
// to touch the filesystem.
//
// See REFACTOR_GOALS.md §3-1 兼容性策略 for the v1.x vs v2.0 rules.

#include "sinic/graph.hpp"

#include <map>
#include <string>

#include <nlohmann/json.hpp>

namespace sinic {

struct WriteOptions {
    bool        validate     = true;    // validate output against schema
    int         json_indent  = 4;       // pretty-print indent (matches pre-M5)
    // M6 (REFACTOR_GOALS.md §3-1): default writer emits schema_version
    // "2.0" with multi-layer + waveguides_per_link General Parameters
    // fields and per-edge _above / _below counters. Set to "1.0" to
    // write a v1.x-compatible output. v1.x omits top-level
    // `crosstalk` even when a payload is supplied (M7). Set empty
    // string to omit the schema_version field (treated as legacy
    // v1.x by readers per §3-1 兼容性策略).
    std::string schema_version = "2.0";
};

// Build schema-conformant JSON for the current graph state. The
// function name is historical (M5 v1.x was the first writer); since
// M6 the default is "2.0" (mirroring `WriteOptions::schema_version`)
// so a direct caller behaves the same as going through
// `save_subgraphsdata_v1x`. Pass `"1.0"` explicitly to opt into the
// legacy v1.x field set.
//
// M7: when `schema_version == "2.0"` AND `crosstalk` is non-null the
// writer emits a top-level `crosstalk` field. v1.x outputs omit
// crosstalk regardless (§3-1 — `crosstalk` is a v2.0-only field).
// The payload is shape-validated against `shape == [k, k, k]` before
// being merged in — mismatches raise `std::invalid_argument` (mirror
// of Python `_validate_crosstalk_payload_shape`).
nlohmann::json subgraphsdata_v1x_to_json(
    const SiNInterconnectionGraph& graph,
    const std::map<std::string, double>& general_params,
    const std::string& schema_version = "2.0",
    const nlohmann::json* crosstalk = nullptr);

// Write the schema-conformant v1.x file. Validates against
// `subgraphsdata.schema.json` when `opts.validate == true`. Throws
// `SchemaValidationError` on schema violation.
void save_subgraphsdata_v1x(
    const SiNInterconnectionGraph& graph,
    const std::string& filename,
    const std::map<std::string, double>& general_params,
    WriteOptions opts = {},
    const nlohmann::json* crosstalk = nullptr);

// M7: cross-field shape validator for the `crosstalk` payload. JSON
// Schema Draft 2020-12 cannot express `len(values) == shape[i]`, so we
// run this guard in addition to the structural schema check. Rejects:
//   - missing/non-array `shape`
//   - `shape` not a 3-element array of **positive** integers (bool is
//     rejected explicitly: `nlohmann::json` reports `true` as a
//     `boolean`, not `integer`, so the type check below catches it)
//   - inner array length ≠ shape[axis]
//   - non-array inner slab / leaf row
// Mirrors Python `core.py::_validate_crosstalk_payload_shape` which
// requires `x > 0` for each shape entry. Throws
// `std::invalid_argument` (NOT `SchemaValidationError`) so the caller
// sees the same error class as Python's `ValueError`.
void validate_crosstalk_payload_shape(const nlohmann::json& crosstalk,
                                      int expected_k);

// Legacy bit-exact writer — preserved so existing reference fixtures
// (`output/cl_*/subgraphsdata.json`) remain reproducible.
void save_subgraphs_legacy_newformat(
    const SiNInterconnectionGraph& graph,
    const std::string& filename,
    const std::map<std::string, double>& general_params);

nlohmann::json subgraphs_legacy_newformat_to_json(
    const SiNInterconnectionGraph& graph,
    const std::map<std::string, double>& general_params);

// Reader for v1.x files (and pre-M5 legacy files via soft-fallback).
// Returns the General Parameters dict + reconstructed per-layer
// EdgeLists. Does NOT touch the original graph; callers can use the
// data to rebuild a SiNInterconnectionGraph or to drive a parity
// check.
struct LoadedSubgraphs {
    nlohmann::json                   general_params;
    std::vector<EdgeList>            sub_graphs;
    std::optional<std::string>       schema_version;
};

LoadedSubgraphs load_subgraphsdata(const std::string& filename,
                                   bool validate = true);

} // namespace sinic
