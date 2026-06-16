#pragma once

// Minimal JSON Schema Draft 2020-12 validator scoped to the features
// that `code/schema/subgraphsdata.schema.json` and `run_report.schema.json`
// actually use. Lives in C++ to satisfy REFACTOR_GOALS.md §3-3 item 2
// "Python: jsonschema; C++: validator runtime" without taking a
// dependency on an external library (the prose says `nlohmann::json-schema-validator`,
// but vendoring it would also require shadowing nlohmann::json itself —
// hand-rolling keeps M5 self-contained).
//
// Single source of truth for supported / unsupported keywords.
// Adding a keyword to either list requires updating BOTH the dispatch
// in `validate_impl` and the docstring below.
//
// **Supported (enforced)**:
//   type (single string or array, including "null"), enum, const,
//   required, properties, additionalProperties (true/false/object),
//   patternProperties, pattern, minimum, exclusiveMinimum, minLength,
//   minItems, maxItems, items, allOf, if/then.
//
// **Known unsupported (silently under-enforce — DO NOT add to a
// schema before extending this validator)**:
//   $ref, oneOf, anyOf, not, format, dependentRequired,
//   if/then/`else`-branch, contains, propertyNames, additionalItems.
//
// The current `subgraphsdata.schema.json` and `run_report.schema.json`
// use exactly the supported set above (M5 dual-review pass 2026-05-14
// verified — Opus 4.7 + Codex gpt-5.5 xhigh, both rounds). When
// adding any keyword from the unsupported list to a schema, first
// extend this validator and add a regression test in
// `tests/test_schema_validator.cpp`, then update both lists above.

#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace sinic {

class SchemaValidationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Generic entry point.
void validate_against_schema(const nlohmann::json& instance,
                             const nlohmann::json& schema);

// Convenience helpers — these load the schema file from the on-disk
// `code/schema/` directory (resolved via `find_schema_dir`). Throwing
// SchemaValidationError lets callers report a single typed exception.
void validate_subgraphsdata_payload(const nlohmann::json& instance);
void validate_run_report_payload  (const nlohmann::json& instance);

// Search-path resolution. Order:
//   1. $SINIC_SCHEMA_DIR (env var) if set
//   2. ../schema relative to the running executable
//   3. ../../schema, ../../../schema (walk up to find code/schema/)
// Throws SchemaValidationError if no directory containing the two
// schema files is found.
std::string find_schema_dir();

// Test seam: override the search-path. Pass an empty string to reset.
void set_schema_dir_override(std::string dir);

} // namespace sinic
