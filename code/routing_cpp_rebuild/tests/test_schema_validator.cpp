#include <catch2/catch.hpp>

#include "sinic/schema_validator.hpp"

#include <nlohmann/json.hpp>

using sinic::SchemaValidationError;
using sinic::validate_against_schema;
using sinic::validate_subgraphsdata_payload;
using sinic::validate_run_report_payload;

TEST_CASE("trivial type checks", "[schema]") {
    nlohmann::json schema = {{"type", "integer"}};
    REQUIRE_NOTHROW(validate_against_schema(42, schema));
    REQUIRE_THROWS_AS(validate_against_schema("hi", schema),
                      SchemaValidationError);
}

TEST_CASE("required keys enforced", "[schema]") {
    nlohmann::json schema = {
        {"type", "object"},
        {"required", {"a", "b"}},
    };
    REQUIRE_NOTHROW(validate_against_schema(
        nlohmann::json{{"a", 1}, {"b", 2}}, schema));
    REQUIRE_THROWS_AS(validate_against_schema(
        nlohmann::json{{"a", 1}}, schema), SchemaValidationError);
}

TEST_CASE("additionalProperties false rejects unknown keys", "[schema]") {
    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"a", {{"type", "integer"}}}}},
        {"additionalProperties", false},
    };
    REQUIRE_NOTHROW(validate_against_schema(nlohmann::json{{"a", 1}}, schema));
    REQUIRE_THROWS_AS(validate_against_schema(
        nlohmann::json{{"a", 1}, {"b", "extra"}}, schema),
                      SchemaValidationError);
}

TEST_CASE("patternProperties + pattern", "[schema]") {
    nlohmann::json schema = {
        {"type", "object"},
        {"patternProperties", {
            {"^[0-9]+$", {{"type", "integer"}}}
        }},
        {"additionalProperties", false},
    };
    REQUIRE_NOTHROW(validate_against_schema(
        nlohmann::json{{"0", 1}, {"42", 2}}, schema));
    REQUIRE_THROWS_AS(validate_against_schema(
        nlohmann::json{{"hello", 1}}, schema), SchemaValidationError);
}

TEST_CASE("enum + const", "[schema]") {
    nlohmann::json schema = {
        {"enum", {"a", "b", "c"}},
    };
    REQUIRE_NOTHROW(validate_against_schema("a", schema));
    REQUIRE_THROWS_AS(validate_against_schema("z", schema),
                      SchemaValidationError);

    nlohmann::json const_schema = {{"const", "x"}};
    REQUIRE_NOTHROW(validate_against_schema("x", const_schema));
    REQUIRE_THROWS_AS(validate_against_schema("y", const_schema),
                      SchemaValidationError);
}

TEST_CASE("array minItems/maxItems and items", "[schema]") {
    nlohmann::json schema = {
        {"type", "array"},
        {"minItems", 2},
        {"maxItems", 3},
        {"items", {{"type", "integer"}}},
    };
    REQUIRE_NOTHROW(validate_against_schema(
        nlohmann::json::array({1, 2}), schema));
    REQUIRE_THROWS_AS(validate_against_schema(
        nlohmann::json::array({1}), schema), SchemaValidationError);
    REQUIRE_THROWS_AS(validate_against_schema(
        nlohmann::json::array({1, 2, 3, 4}), schema), SchemaValidationError);
    REQUIRE_THROWS_AS(validate_against_schema(
        nlohmann::json::array({1, "two"}), schema), SchemaValidationError);
}

TEST_CASE("allOf with if/then conditional", "[schema]") {
    nlohmann::json schema = {
        {"type", "object"},
        {"allOf", {{
            {"if", {{"properties", {{"flag", {{"const", true}}}}}, {"required", {"flag"}}}},
            {"then", {{"required", {"data"}}}}
        }}},
    };
    REQUIRE_NOTHROW(validate_against_schema(
        nlohmann::json{{"flag", false}}, schema));  // if fails ⇒ then skipped
    REQUIRE_NOTHROW(validate_against_schema(
        nlohmann::json{{"flag", true}, {"data", 1}}, schema));
    REQUIRE_THROWS_AS(validate_against_schema(
        nlohmann::json{{"flag", true}}, schema), SchemaValidationError);
}

TEST_CASE("minLength enforced (M5 dual-review P0 fix)", "[schema]") {
    nlohmann::json schema = {
        {"type", "string"},
        {"minLength", 3},
    };
    REQUIRE_NOTHROW(validate_against_schema("abc", schema));
    REQUIRE_NOTHROW(validate_against_schema("hello", schema));
    REQUIRE_THROWS_AS(validate_against_schema("hi", schema),
                      SchemaValidationError);
    REQUIRE_THROWS_AS(validate_against_schema("", schema),
                      SchemaValidationError);
}

TEST_CASE("run_report.json rejects empty run_id (minLength=1 enforced)",
          "[schema][integration]") {
    // The schema declares run_id.minLength=1; before the M5 dual-review
    // fix the validator silently skipped minLength so empty run_ids
    // slipped through. This regression locks the fix in.
    nlohmann::json payload = {
        {"schema_version", "1.0"},
        {"run_id", ""},  // BAD
        {"config", nlohmann::json::object()},
        {"trace", nlohmann::json::array()},
        {"timings", nlohmann::json::object()},
        {"summary", {
            {"initial_loss", 0.0},
            {"final_loss", 0.0},
            {"relative_drop", 0.0},
            {"iterations", 0},
            {"best_at_iter", -1},
        }},
    };
    REQUIRE_THROWS_AS(validate_run_report_payload(payload),
                      SchemaValidationError);
}

TEST_CASE("run_report.schema.json round-trip", "[schema][integration]") {
    // The minimum payload the schema requires.
    nlohmann::json ok = {
        {"schema_version", "1.0"},
        {"run_id", "rid"},
        {"config", nlohmann::json::object()},
        {"trace", nlohmann::json::array()},
        {"timings", nlohmann::json::object()},
        {"summary", {
            {"initial_loss", 0.0},
            {"final_loss", 0.0},
            {"relative_drop", 0.0},
            {"iterations", 0},
            {"best_at_iter", -1},
        }},
    };
    REQUIRE_NOTHROW(validate_run_report_payload(ok));

    // Missing required field fails.
    auto bad = ok;
    bad.erase("summary");
    REQUIRE_THROWS_AS(validate_run_report_payload(bad), SchemaValidationError);

    // schema_version pattern: only `1.x` allowed.
    auto bad_ver = ok;
    bad_ver["schema_version"] = "2.0";
    REQUIRE_THROWS_AS(validate_run_report_payload(bad_ver), SchemaValidationError);
}

TEST_CASE("subgraphsdata.schema.json minimum payload validates", "[schema][integration]") {
    nlohmann::json payload = {
        {"General Parameters", {{"k", 12}}},
        {"positions", {{"0", {0.0, 0.0}}, {"1", {1.0, 0.0}}}},
        {"complete_graph", nlohmann::json::array()},
        {"loss_analysis", nlohmann::json::object()},
        {"Layer_0", {{"edges", nlohmann::json::array()}}},
    };
    REQUIRE_NOTHROW(validate_subgraphsdata_payload(payload));
}
