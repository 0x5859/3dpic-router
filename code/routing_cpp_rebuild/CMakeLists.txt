cmake_minimum_required(VERSION 3.9)
project(SiNIC LANGUAGES CXX)

# C++17 required (statistics.hpp uses string_view, optional, filesystem).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All sinic_core headers live under include/; the external libraries
# under external/.
include_directories(${PROJECT_SOURCE_DIR}/include)

# External dual-annealing dependency (unchanged from pre-M5 wiring).
add_subdirectory(${PROJECT_SOURCE_DIR}/external/dual-annealing)

# -----------------------------------------------------------------------------
# sinic_core — the module split deliverable for M5 (REFACTOR_GOALS.md §1-2).
# Every cpp file is one module, every header in include/sinic/ has one .cpp.
# main.cpp links against this library; the test binary too.
# -----------------------------------------------------------------------------
add_library(sinic_core STATIC
    src/geometry.cpp
    src/crossings.cpp
    src/crossings_cache.cpp
    src/loss.cpp
    src/positions.cpp
    src/graph.cpp
    src/statistics.cpp
    src/schema_validator.cpp
    src/io.cpp
    src/crosstalk.cpp
    src/optimizers/dual_annealing.cpp
)
target_include_directories(sinic_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(sinic_core PUBLIC dual_annealing)
# Disable noisy conversion warnings that the legacy code triggers (the
# external dual-annealing headers do mass int→size_t conversions); we
# pin this to GNU to match the pre-M5 CMakeLists policy.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(sinic_core PRIVATE -Wno-sign-conversion -Wno-old-style-cast)
endif()

# -----------------------------------------------------------------------------
# Main executable. Slim entry point (<200 lines) per REFACTOR_GOALS.md §1-2.
# -----------------------------------------------------------------------------
add_executable(Autowiring_CPP src/main.cpp)
target_link_libraries(Autowiring_CPP PRIVATE sinic_core)

# -----------------------------------------------------------------------------
# Tests — Catch2 v2 single-header (vendored under external/).
# Toggle with `-DSINIC_BUILD_TESTS=OFF` for a release-only build.
# -----------------------------------------------------------------------------
option(SINIC_BUILD_TESTS "Build sinic_core unit tests" ON)
if(SINIC_BUILD_TESTS)
    enable_testing()
    add_executable(sinic_tests
        tests/test_main.cpp
        tests/test_geometry.cpp
        tests/test_crossings.cpp
        tests/test_crossings_cache.cpp
        tests/test_loss.cpp
        tests/test_graph.cpp
        tests/test_multilayer.cpp
        tests/test_statistics.cpp
        tests/test_schema_validator.cpp
        tests/test_io.cpp
        tests/test_positions_json.cpp
        tests/test_end_to_end.cpp
        tests/test_crosstalk.cpp
    )
    target_include_directories(sinic_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/external/dual-annealing/third_party/Catch2/single_include
    )
    target_link_libraries(sinic_tests PRIVATE sinic_core)
    add_test(NAME sinic_unit_tests COMMAND sinic_tests)

    # Tests bake the schema dir at configure time so they don't depend
    # on CWD. The validator can be overridden at runtime via
    # `set_schema_dir_override` (used in test_main.cpp).
    target_compile_definitions(sinic_tests PRIVATE
        SINIC_SCHEMA_DIR_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../schema")
endif()
