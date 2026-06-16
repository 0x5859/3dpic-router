#include "sinic/schema_validator.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>

namespace sinic {

namespace {

std::mutex      g_schema_mutex;
bool            g_schema_loaded = false;
std::string     g_schema_dir_override;
nlohmann::json  g_subgraphs_schema;
nlohmann::json  g_runreport_schema;

void throw_v(const std::string& path, const std::string& msg) {
    std::ostringstream os;
    os << "schema validation failed at "
       << (path.empty() ? std::string("<root>") : path)
       << ": " << msg;
    throw SchemaValidationError(os.str());
}

std::string append_path(const std::string& path, const std::string& key) {
    if (path.empty()) return key;
    return path + "." + key;
}

std::string append_idx(const std::string& path, std::size_t idx) {
    std::ostringstream os;
    os << path << "[" << idx << "]";
    return os.str();
}

// Returns the JSON-Schema "type" string (or "integer" for integral
// nlohmann numbers).
std::string json_type_name(const nlohmann::json& v) {
    switch (v.type()) {
        case nlohmann::json::value_t::null:           return "null";
        case nlohmann::json::value_t::boolean:        return "boolean";
        case nlohmann::json::value_t::number_integer: return "integer";
        case nlohmann::json::value_t::number_unsigned:return "integer";
        case nlohmann::json::value_t::number_float:   return "number";
        case nlohmann::json::value_t::string:         return "string";
        case nlohmann::json::value_t::array:          return "array";
        case nlohmann::json::value_t::object:         return "object";
        default:                                       return "unknown";
    }
}

bool type_matches(const std::string& actual, const std::string& expected) {
    if (actual == expected) return true;
    // "integer" satisfies "number" per JSON Schema.
    if (expected == "number" && actual == "integer") return true;
    return false;
}

void validate_impl(const nlohmann::json& instance,
                   const nlohmann::json& schema,
                   const std::string& path);

void check_type(const nlohmann::json& instance,
                const nlohmann::json& type_spec,
                const std::string& path) {
    const std::string actual = json_type_name(instance);
    if (type_spec.is_string()) {
        const std::string expected = type_spec.get<std::string>();
        if (!type_matches(actual, expected)) {
            throw_v(path, "type mismatch: expected '" + expected
                          + "', got '" + actual + "'");
        }
    } else if (type_spec.is_array()) {
        for (const auto& opt : type_spec) {
            if (opt.is_string()
                && type_matches(actual, opt.get<std::string>())) {
                return;
            }
        }
        std::ostringstream os;
        os << "type mismatch: expected one of " << type_spec.dump()
           << ", got '" << actual << "'";
        throw_v(path, os.str());
    }
}

void check_enum(const nlohmann::json& instance,
                const nlohmann::json& enum_spec,
                const std::string& path) {
    for (const auto& candidate : enum_spec) {
        if (candidate == instance) return;
    }
    throw_v(path, "value not in enum " + enum_spec.dump());
}

void check_const(const nlohmann::json& instance,
                 const nlohmann::json& const_val,
                 const std::string& path) {
    if (instance != const_val) {
        throw_v(path, "value does not match const " + const_val.dump());
    }
}

void check_pattern(const nlohmann::json& instance,
                   const nlohmann::json& pattern_spec,
                   const std::string& path) {
    if (!instance.is_string()) return; // pattern only applies to strings
    if (!pattern_spec.is_string()) return;
    try {
        std::regex rx(pattern_spec.get<std::string>(),
                      std::regex::ECMAScript);
        if (!std::regex_search(instance.get<std::string>(), rx)) {
            throw_v(path, "string does not match pattern "
                          + pattern_spec.dump());
        }
    } catch (const std::regex_error& e) {
        throw_v(path, std::string("invalid regex in schema pattern: ")
                      + e.what());
    }
}

void check_min_length(const nlohmann::json& instance,
                      const nlohmann::json& min_len_spec,
                      const std::string& path) {
    if (!instance.is_string()) return;
    if (!min_len_spec.is_number_integer()) return;
    const auto mn = min_len_spec.get<std::size_t>();
    if (instance.get<std::string>().size() < mn) {
        std::ostringstream os;
        os << "string length " << instance.get<std::string>().size()
           << " below minLength " << mn;
        throw_v(path, os.str());
    }
}

void check_numeric_bounds(const nlohmann::json& instance,
                          const nlohmann::json& schema,
                          const std::string& path) {
    if (!instance.is_number()) return;
    const double v = instance.get<double>();
    if (schema.contains("minimum") && schema["minimum"].is_number()) {
        const double mn = schema["minimum"].get<double>();
        if (v < mn) {
            std::ostringstream os;
            os << "value " << v << " below minimum " << mn;
            throw_v(path, os.str());
        }
    }
    if (schema.contains("exclusiveMinimum")
        && schema["exclusiveMinimum"].is_number()) {
        const double mn = schema["exclusiveMinimum"].get<double>();
        if (v <= mn) {
            std::ostringstream os;
            os << "value " << v << " <= exclusiveMinimum " << mn;
            throw_v(path, os.str());
        }
    }
}

void check_array_bounds(const nlohmann::json& instance,
                        const nlohmann::json& schema,
                        const std::string& path) {
    if (!instance.is_array()) return;
    if (schema.contains("minItems") && schema["minItems"].is_number_integer()) {
        const auto mn = schema["minItems"].get<std::size_t>();
        if (instance.size() < mn) {
            std::ostringstream os;
            os << "array length " << instance.size()
               << " < minItems " << mn;
            throw_v(path, os.str());
        }
    }
    if (schema.contains("maxItems") && schema["maxItems"].is_number_integer()) {
        const auto mx = schema["maxItems"].get<std::size_t>();
        if (instance.size() > mx) {
            std::ostringstream os;
            os << "array length " << instance.size()
               << " > maxItems " << mx;
            throw_v(path, os.str());
        }
    }
}

void check_items(const nlohmann::json& instance,
                 const nlohmann::json& items_spec,
                 const std::string& path) {
    if (!instance.is_array()) return;
    for (std::size_t i = 0; i < instance.size(); ++i) {
        validate_impl(instance[i], items_spec, append_idx(path, i));
    }
}

void check_required(const nlohmann::json& instance,
                    const nlohmann::json& required_spec,
                    const std::string& path) {
    if (!instance.is_object()) return;
    for (const auto& key_json : required_spec) {
        const auto key = key_json.get<std::string>();
        if (!instance.contains(key)) {
            throw_v(path, "missing required property '" + key + "'");
        }
    }
}

void check_properties(const nlohmann::json& instance,
                      const nlohmann::json& schema,
                      const std::string& path) {
    if (!instance.is_object()) return;
    static const nlohmann::json empty_obj = nlohmann::json::object();
    const auto& props    = schema.contains("properties")
                              ? schema["properties"] : empty_obj;
    const auto& patprops = schema.contains("patternProperties")
                              ? schema["patternProperties"] : empty_obj;
    const bool has_addl  = schema.contains("additionalProperties");
    // Pull additionalProperties via pointer to avoid the ambiguous
    // ternary between a json reference and a value_t enum.
    const nlohmann::json* addl_ptr = has_addl
                                       ? &schema["additionalProperties"]
                                       : nullptr;
    const bool addl_object = has_addl && addl_ptr->is_object();
    const bool addl_false  = has_addl && addl_ptr->is_boolean()
                                       && !addl_ptr->get<bool>();

    for (auto it = instance.begin(); it != instance.end(); ++it) {
        const std::string key = it.key();
        bool matched = false;
        if (props.is_object() && props.contains(key)) {
            validate_impl(it.value(), props[key], append_path(path, key));
            matched = true;
        }
        if (patprops.is_object()) {
            for (auto pit = patprops.begin(); pit != patprops.end(); ++pit) {
                try {
                    std::regex rx(pit.key(), std::regex::ECMAScript);
                    if (std::regex_search(key, rx)) {
                        validate_impl(it.value(), pit.value(),
                                      append_path(path, key));
                        matched = true;
                    }
                } catch (const std::regex_error&) {
                    // Schema bug; surface at validate time.
                    throw_v(path, "invalid patternProperties regex '"
                                  + pit.key() + "'");
                }
            }
        }
        if (!matched) {
            if (addl_false) {
                throw_v(path, "unexpected additional property '" + key + "'");
            }
            if (addl_object) {
                validate_impl(it.value(), *addl_ptr, append_path(path, key));
            }
        }
    }
}

void check_all_of(const nlohmann::json& instance,
                  const nlohmann::json& all_of_spec,
                  const std::string& path) {
    for (std::size_t i = 0; i < all_of_spec.size(); ++i) {
        const auto& sub = all_of_spec[i];
        if (sub.is_object() && sub.contains("if")) {
            // if/then conditional. We do not support `else` because the
            // current schemas do not use it.
            try {
                validate_impl(instance, sub["if"], path);
            } catch (const SchemaValidationError&) {
                // `if` failed → skip `then`.
                continue;
            }
            if (sub.contains("then")) {
                validate_impl(instance, sub["then"], path);
            }
        } else {
            validate_impl(instance, sub, path);
        }
    }
}

void validate_impl(const nlohmann::json& instance,
                   const nlohmann::json& schema,
                   const std::string& path) {
    if (!schema.is_object()) {
        // `true` schema accepts anything; non-object/non-true schema
        // is a malformed schema for our purposes.
        if (schema.is_boolean()) {
            if (!schema.get<bool>()) {
                throw_v(path, "schema is `false` — no instance is valid");
            }
            return;
        }
        return;
    }

    if (schema.contains("type")) {
        check_type(instance, schema["type"], path);
    }
    if (schema.contains("enum")) {
        check_enum(instance, schema["enum"], path);
    }
    if (schema.contains("const")) {
        check_const(instance, schema["const"], path);
    }
    if (schema.contains("pattern")) {
        check_pattern(instance, schema["pattern"], path);
    }
    if (schema.contains("minLength")) {
        check_min_length(instance, schema["minLength"], path);
    }
    check_numeric_bounds(instance, schema, path);
    check_array_bounds(instance, schema, path);
    if (schema.contains("items")) {
        check_items(instance, schema["items"], path);
    }
    if (schema.contains("required")) {
        check_required(instance, schema["required"], path);
    }
    // properties / patternProperties / additionalProperties are
    // handled in one pass to avoid double-walking the object.
    if (instance.is_object()
        && (schema.contains("properties")
         || schema.contains("patternProperties")
         || schema.contains("additionalProperties"))) {
        check_properties(instance, schema, path);
    }
    if (schema.contains("allOf")) {
        check_all_of(instance, schema["allOf"], path);
    }
}

std::filesystem::path resolve_schema_dir() {
    if (!g_schema_dir_override.empty()) {
        return std::filesystem::path(g_schema_dir_override);
    }
    if (const char* env = std::getenv("SINIC_SCHEMA_DIR")) {
        if (env && *env) return std::filesystem::path(env);
    }
    // Walk up from CWD looking for a `schema/` directory containing the
    // two .schema.json files. Bounded depth.
    std::filesystem::path here = std::filesystem::current_path();
    for (int depth = 0; depth < 8; ++depth) {
        std::filesystem::path candidate = here / "schema";
        if (std::filesystem::exists(candidate / "subgraphsdata.schema.json")
         && std::filesystem::exists(candidate / "run_report.schema.json")) {
            return candidate;
        }
        candidate = here / "code" / "schema";
        if (std::filesystem::exists(candidate / "subgraphsdata.schema.json")
         && std::filesystem::exists(candidate / "run_report.schema.json")) {
            return candidate;
        }
        if (!here.has_parent_path() || here == here.parent_path()) break;
        here = here.parent_path();
    }
    throw SchemaValidationError(
        "schema dir not found; set $SINIC_SCHEMA_DIR or invoke from a path "
        "with `schema/{subgraphsdata,run_report}.schema.json`");
}

void ensure_schemas_loaded() {
    std::lock_guard<std::mutex> lock(g_schema_mutex);
    if (g_schema_loaded) return;
    const auto dir = resolve_schema_dir();
    std::ifstream s1(dir / "subgraphsdata.schema.json");
    std::ifstream s2(dir / "run_report.schema.json");
    if (!s1 || !s2) {
        throw SchemaValidationError("schema files missing in " + dir.string());
    }
    s1 >> g_subgraphs_schema;
    s2 >> g_runreport_schema;
    g_schema_loaded = true;
}

} // namespace

void validate_against_schema(const nlohmann::json& instance,
                             const nlohmann::json& schema) {
    validate_impl(instance, schema, "");
}

void validate_subgraphsdata_payload(const nlohmann::json& instance) {
    ensure_schemas_loaded();
    validate_against_schema(instance, g_subgraphs_schema);
}

void validate_run_report_payload(const nlohmann::json& instance) {
    ensure_schemas_loaded();
    validate_against_schema(instance, g_runreport_schema);
}

std::string find_schema_dir() {
    return resolve_schema_dir().string();
}

void set_schema_dir_override(std::string dir) {
    std::lock_guard<std::mutex> lock(g_schema_mutex);
    g_schema_dir_override = std::move(dir);
    g_schema_loaded = false;  // force reload on next validate_*
}

} // namespace sinic
