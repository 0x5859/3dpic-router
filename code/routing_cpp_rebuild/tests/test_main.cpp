// Catch2 v2 single-header. The vendored library has v2 only; v3 would
// need an entirely different `catch_amalgamated` setup.
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "sinic/schema_validator.hpp"

int main(int argc, char* argv[]) {
    // The CMake build defines SINIC_SCHEMA_DIR_DEFAULT to the in-tree
    // `code/schema/` path. Without this, tests would have to be run
    // from the repo root or with `SINIC_SCHEMA_DIR` set.
#ifdef SINIC_SCHEMA_DIR_DEFAULT
    sinic::set_schema_dir_override(SINIC_SCHEMA_DIR_DEFAULT);
#endif
    return Catch::Session().run(argc, argv);
}
