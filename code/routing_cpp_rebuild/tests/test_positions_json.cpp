#include <catch2/catch.hpp>

#include "sinic/positions.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using sinic::PositionMap;
using sinic::load_positions_from_json;

namespace {
std::string write_tmp(const std::string& name, const std::string& body) {
    auto p = std::filesystem::temp_directory_path() / name;
    std::ofstream(p) << body;
    return p.string();
}
}  // namespace

TEST_CASE("load_positions_from_json parses a valid object", "[positions_json]") {
    const auto path = write_tmp(
        "mc_pos_ok.json",
        R"({"0":[0.0,1.0],"1":[2.5,3.0],"2":[-1.0,4.0]})");
    PositionMap pos = load_positions_from_json(path);
    REQUIRE(pos.size() == 3);
    REQUIRE(pos.at(0) == std::make_pair(0.0, 1.0));
    REQUIRE(pos.at(1) == std::make_pair(2.5, 3.0));
    REQUIRE(pos.at(2) == std::make_pair(-1.0, 4.0));
}

TEST_CASE("load_positions_from_json rejects malformed input",
          "[positions_json]") {
    SECTION("missing file") {
        REQUIRE_THROWS_AS(load_positions_from_json("/no/such/mc_pos.json"),
                          std::runtime_error);
    }
    SECTION("top-level not an object") {
        const auto p = write_tmp("mc_pos_arr.json", R"([[0,1],[2,3]])");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("non-integer key") {
        const auto p = write_tmp("mc_pos_badkey.json",
                                 R"({"0":[0,1],"x":[2,3]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("value not a 2-number array") {
        const auto p = write_tmp("mc_pos_badval.json",
                                 R"({"0":[0,1],"1":[2]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("keys not contiguous 0..k-1") {
        const auto p = write_tmp("mc_pos_gap.json",
                                 R"({"0":[0,1],"2":[2,3]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("empty object") {
        const auto p = write_tmp("mc_pos_empty.json", R"({})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("leading-zero key collapses silently -> rejected") {
        const auto p = write_tmp("mc_pos_lz.json",
                                 R"({"0":[0,0],"01":[1,1]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("duplicate-after-parse key (\"0\" vs \"00\") -> rejected") {
        const auto p = write_tmp("mc_pos_dup.json",
                                 R"({"0":[0,0],"00":[1,1]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("leading-space key -> rejected") {
        const auto p = write_tmp("mc_pos_ws.json", R"({" 0":[0,0]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("plus-signed key -> rejected") {
        const auto p = write_tmp("mc_pos_plus.json",
                                 R"({"+1":[0,0],"0":[1,1]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("negative key -> rejected") {
        const auto p = write_tmp("mc_pos_neg.json",
                                 R"({"-1":[0,0],"0":[1,1]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("truncated/invalid JSON syntax -> rejected") {
        const auto p = write_tmp("mc_pos_trunc.json", R"({"0":[0,1])");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
    SECTION("canonical-but-int-overflow key -> rejected (out of int range)") {
        const auto p = write_tmp(
            "mc_pos_ovf.json",
            R"({"0":[0,0],"99999999999999999999":[1,1]})");
        REQUIRE_THROWS_AS(load_positions_from_json(p), std::runtime_error);
    }
}

// M-C stage-end review (accepted P2): a *literal duplicate* canonical
// JSON key is NOT rejected — nlohmann::json keep-last collapses it,
// identical to Python json.loads keep-last, so the two backends stay
// consistent (the real producer json.dump(dict) cannot emit dup keys).
// This documents/locks the intentional behavior.
TEST_CASE("load_positions_from_json: duplicate canonical key is keep-last "
          "(documented, == Python json.loads)", "[positions_json]") {
    const auto path = write_tmp(
        "mc_pos_dupcanon.json",
        R"({"0":[0.0,0.0],"0":[9.0,9.0],"1":[1.0,1.0]})");
    PositionMap pos;
    REQUIRE_NOTHROW(pos = load_positions_from_json(path));
    REQUIRE(pos.size() == 2);
    REQUIRE(pos.at(0) == std::make_pair(9.0, 9.0));  // last wins
    REQUIRE(pos.at(1) == std::make_pair(1.0, 1.0));
}
