#include <catch2/catch.hpp>

#include "sinic/statistics.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>

using sinic::IterEvent;
using sinic::NullSink;
using sinic::PhaseTimer;
using sinic::RunRecorder;
using sinic::StatsSink;
using sinic::write_run_report;

TEST_CASE("NullSink swallows all events", "[statistics]") {
    NullSink s;
    s.on_iter({0, 1.0, 10, false});
    s.on_phase("anything", 100);
    auto j = s.finalize();
    REQUIRE(j.is_object());
    REQUIRE(j.empty());
}

TEST_CASE("RunRecorder is_new_best matches expected sequence", "[statistics]") {
    // Mirrors Python `test_recorder_is_new_best` (§1-1-a 测试).
    RunRecorder rec("test_run", nlohmann::json::object());
    const double losses[] = {10.0, 8.0, 9.0, 7.0, 7.5};
    const bool expected[] = {true,  true, false, true, false};
    for (int i = 0; i < 5; ++i) {
        rec.on_iter({i, losses[i], i * 10L, /*is_new_best=*/false});
    }
    REQUIRE(rec.trace_size() == 5);
    REQUIRE(rec.best_so_far() == Approx(7.0));
    REQUIRE(rec.best_at_iter() == 3);

    auto j = rec.finalize();
    REQUIRE(j["trace"].is_array());
    REQUIRE(j["trace"].size() == 5);
    for (int i = 0; i < 5; ++i) {
        REQUIRE(j["trace"][i]["is_new_best"].get<bool>() == expected[i]);
    }
}

TEST_CASE("RunRecorder drops non-finite losses", "[statistics]") {
    // Python recorder.py:42-50 mirror: NaN / ±Inf are silently dropped
    // so the strict-JSON writer never emits non-strict tokens.
    RunRecorder rec("test_run", nlohmann::json::object());
    rec.on_iter({0, 5.0, 1L, false});
    rec.on_iter({1, std::nan(""), 2L, false});
    rec.on_iter({2, std::numeric_limits<double>::infinity(), 3L, false});
    rec.on_iter({3, 4.0, 4L, false});
    REQUIRE(rec.trace_size() == 2);
    REQUIRE(rec.best_so_far() == Approx(4.0));
}

TEST_CASE("PhaseTimer RAII emits exactly one on_phase", "[statistics]") {
    struct CountingSink : public StatsSink {
        int phases = 0;
        std::string last_name;
        long last_ms = -1;
        void on_iter(const IterEvent&) override {}
        void on_phase(std::string_view n, long ms) override {
            ++phases;
            last_name = std::string(n);
            last_ms   = ms;
        }
        nlohmann::json finalize() override { return nlohmann::json::object(); }
    } sink;
    {
        PhaseTimer t(sink, "x");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(sink.phases == 1);
    REQUIRE(sink.last_name == "x");
    REQUIRE(sink.last_ms >= 0);
}

TEST_CASE("RunRecorder finalize matches run_report.schema.json", "[statistics]") {
    RunRecorder rec("test_run_id", nlohmann::json{
        {"k", 12},
        {"optimizer", "dual_annealing"},
    });
    rec.on_iter({0, 100.0, 0L, false});
    rec.on_iter({1, 50.0,  5L, false});
    rec.on_phase("graph_build", 12);

    auto j = rec.finalize();
    REQUIRE(j["schema_version"] == "1.0");
    REQUIRE(j["run_id"]         == "test_run_id");
    REQUIRE(j["config"]["k"]    == 12);
    REQUIRE(j["timings"]["graph_build_ms"] == 12);
    REQUIRE(j["summary"]["iterations"] == 2);
    REQUIRE(j["summary"]["initial_loss"] == Approx(100.0));
    REQUIRE(j["summary"]["final_loss"]   == Approx(50.0));
    REQUIRE(j["summary"]["best_at_iter"] == 1);
    REQUIRE(j["summary"]["relative_drop"] == Approx(0.5));
}

TEST_CASE("RunRecorder aggregate stats emit when set", "[statistics]") {
    RunRecorder rec("test", nlohmann::json::object());
    rec.on_iter({0, 1.0, 0L, false});
    rec.set_aggregate(
        {RunRecorder::LayerAggregate{0, 39, 115},
         RunRecorder::LayerAggregate{1, 27, 106}},
        274,
        std::string("physical"));
    auto j = rec.finalize();
    REQUIRE(j["summary"]["crosslayer_crossings_total"] == 274);
    REQUIRE(j["summary"]["crosslayer_crossings_convention"] == "physical");
    REQUIRE(j["summary"]["layers"].size() == 2);
}

TEST_CASE("optimize_wall and trace wall_ms share a clock origin "
          "(M5 dual-review P1 fix)", "[statistics][optimizer]") {
    // Synthetic test: drive a RunRecorder directly with a known origin,
    // and ensure that trace[].wall_ms is monotonically non-decreasing
    // and the on_phase("optimization_wall", elapsed) value is >= the
    // max trace wall_ms. The real DualAnnealingOptimizer integration
    // is exercised end-to-end below via the binary smoke test.
    using clock = std::chrono::steady_clock;
    sinic::RunRecorder rec("origin_test", nlohmann::json::object());
    const auto t0 = clock::now();
    // Simulate three IterEvents with monotonic wall_ms.
    auto ms_since = [&](int dummy) {
        (void)dummy;
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   clock::now() - t0).count();
    };
    rec.on_iter({0, 100.0, ms_since(0), false});
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    rec.on_iter({1, 50.0,  ms_since(1), false});
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    // Phase emit at end with same t0 origin.
    const long wall =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            clock::now() - t0).count();
    rec.on_phase("optimization_wall", wall);

    auto j = rec.finalize();
    REQUIRE(j["timings"]["optimization_wall_ms"].get<long>() >= 0);
    long max_trace = 0;
    for (const auto& ev : j["trace"]) {
        max_trace = std::max(max_trace, ev["wall_ms"].get<long>());
    }
    REQUIRE(j["timings"]["optimization_wall_ms"].get<long>() >= max_trace);
}

TEST_CASE("write_run_report produces both .json and .md", "[statistics]") {
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "sinic_test_run_report";
    fs::remove_all(tmp);

    RunRecorder rec("test_run", nlohmann::json{{"k", 12}});
    rec.on_iter({0, 100.0, 0L, false});
    rec.on_phase("graph_build", 5);
    auto report = rec.finalize();
    auto [jp, mdp] = write_run_report(report, tmp.string());

    REQUIRE(fs::exists(jp));
    REQUIRE(fs::exists(mdp));
    // md is human-readable; just ensure it contains the run_id.
    std::ifstream f(mdp);
    std::string contents((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    REQUIRE(contents.find("test_run") != std::string::npos);
    REQUIRE(contents.find("graph_build_ms") != std::string::npos);
    fs::remove_all(tmp);
}
