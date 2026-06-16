#include "sinic/statistics.hpp"
#include "sinic/schema_validator.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace sinic {

StatsSink& default_null_sink() {
    // Function-local static: thread-safe initialization since C++11
    // (Meyers singleton). Destruction order vs other globals is
    // unspecified, but `NullSink::~NullSink` is trivial and only the
    // process-exit destructor matters here. Callers MUST NOT take the
    // address of the returned reference and store it past
    // `default_null_sink` itself going out of scope — but since this
    // is a function-static there's no such moment during program
    // execution, only at exit when all atexit destructors have already
    // fired.
    static NullSink instance;
    return instance;
}

// ---------------------------------------------------------------------
// RunRecorder
// ---------------------------------------------------------------------

RunRecorder::RunRecorder(std::string run_id, nlohmann::json config)
    : run_id_(std::move(run_id)),
      config_(std::move(config)),
      best_so_far_(std::numeric_limits<double>::infinity()),
      best_at_iter_(-1),
      initial_loss_(std::nullopt),
      aggregate_(nlohmann::json::object()) {}

void RunRecorder::on_iter(const IterEvent& ev) {
    const double loss = static_cast<double>(ev.loss);
    // Mirror Python recorder.py:42-50: drop non-finite losses so that
    // (a) the strict-JSON writer never emits NaN/Inf tokens (the
    // C++ json writer's default is non-strict but Python's
    // json.dump(allow_nan=False) is strict — both ends must agree
    // for M8 parity), and (b) the trace doesn't accumulate "best"
    // updates from bogus values.
    if (!std::isfinite(loss)) {
        return;
    }
    const bool is_new_best = loss < best_so_far_;
    if (is_new_best) {
        best_so_far_  = loss;
        best_at_iter_ = ev.iter;
    }
    if (!initial_loss_.has_value()) {
        initial_loss_ = loss;
    }
    trace_.push_back({
        {"iter",        ev.iter},
        {"loss",        loss},
        {"wall_ms",     ev.wall_ms},
        {"is_new_best", is_new_best},
    });
}

void RunRecorder::on_phase(std::string_view name, long wall_ms) {
    std::string key{name};
    // Mirror Python recorder.py:70-75 — auto-suffix `_ms`. Accumulates
    // across re-entries (two PhaseTimer scopes with the same name add).
    if (key.size() < 3 || key.substr(key.size() - 3) != "_ms") {
        key.append("_ms");
    }
    timings_[key] += wall_ms;
}

void RunRecorder::set_aggregate(std::vector<LayerAggregate> layers,
                                long crosslayer_crossings_total,
                                std::optional<std::string> convention) {
    if (convention.has_value()) {
        const auto& c = *convention;
        if (c != "physical" && c != "geometric" && c != "experimental") {
            throw std::invalid_argument(
                "crosslayer_crossings_convention must be one of "
                "{'physical','geometric','experimental'}");
        }
    }
    nlohmann::json agg = nlohmann::json::object();
    nlohmann::json lvec = nlohmann::json::array();
    for (const auto& l : layers) {
        lvec.push_back({
            {"layer",      l.layer},
            {"edge_count", l.edge_count},
            {"crossings",  l.crossings},
        });
    }
    agg["layers"] = lvec;
    agg["crosslayer_crossings_total"] = crosslayer_crossings_total;
    if (convention.has_value()) {
        agg["crosslayer_crossings_convention"] = *convention;
    }
    aggregate_ = agg;
}

nlohmann::json RunRecorder::finalize() {
    nlohmann::json report = nlohmann::json::object();
    report["schema_version"] = "1.0";
    report["run_id"]         = run_id_;
    report["config"]         = config_;
    report["trace"]          = trace_;

    // Build timings object so that key order is deterministic.
    nlohmann::json timings = nlohmann::json::object();
    for (const auto& kv : timings_) {
        timings[kv.first] = kv.second;
    }
    report["timings"] = timings;

    nlohmann::json summary = nlohmann::json::object();
    double initial = 0.0;
    double final_loss = 0.0;
    int    iterations = static_cast<int>(trace_.size());
    int    best_at_iter = -1;
    double relative_drop = 0.0;
    if (iterations > 0) {
        initial = initial_loss_.value_or(0.0);
        final_loss = std::isfinite(best_so_far_) ? best_so_far_ : 0.0;
        best_at_iter = best_at_iter_;
        if (initial != 0.0) {
            relative_drop = (initial - final_loss) / std::abs(initial);
        }
    }
    summary["initial_loss"]  = initial;
    summary["final_loss"]    = final_loss;
    summary["relative_drop"] = relative_drop;
    summary["iterations"]    = iterations;
    summary["best_at_iter"]  = best_at_iter;

    if (!aggregate_.empty() && aggregate_.is_object()) {
        if (aggregate_.contains("layers")) {
            summary["layers"] = aggregate_["layers"];
        }
        if (aggregate_.contains("crosslayer_crossings_total")) {
            summary["crosslayer_crossings_total"] =
                aggregate_["crosslayer_crossings_total"];
        }
        if (aggregate_.contains("crosslayer_crossings_convention")) {
            summary["crosslayer_crossings_convention"] =
                aggregate_["crosslayer_crossings_convention"];
        }
    }
    report["summary"] = summary;
    return report;
}

// ---------------------------------------------------------------------
// PhaseTimer
// ---------------------------------------------------------------------

PhaseTimer::PhaseTimer(StatsSink& sink, std::string_view name)
    : sink_(sink),
      name_(std::string(name)),
      t0_(std::chrono::steady_clock::now()) {}

PhaseTimer::~PhaseTimer() {
    const auto now = std::chrono::steady_clock::now();
    const auto dt  = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - t0_).count();
    sink_.on_phase(name_, static_cast<long>(dt));
}

// ---------------------------------------------------------------------
// write_run_report
// ---------------------------------------------------------------------

namespace {

std::string render_md(const nlohmann::json& report) {
    // Mirror routing_py_rebuild/statistics/reporters.py::_render_md.
    // Fields are accessed defensively but the dispatcher already
    // schema-validates, so missing keys are unreachable.
    const auto& summary = report.at("summary");
    const auto& timings = report.at("timings");
    const auto& config  = report.at("config");

    auto fmt_g = [](double v) {
        std::ostringstream oss;
        oss << std::setprecision(6) << v;
        return oss.str();
    };

    std::ostringstream os;
    os << "# Run report — " << report.at("run_id").get<std::string>() << "\n\n";
    os << "`schema_version`: " << report.at("schema_version").get<std::string>() << "\n\n";
    os << "## Summary\n\n";
    os << "- iterations: **" << summary.at("iterations").get<int>() << "**\n";
    os << "- initial loss: **" << fmt_g(summary.at("initial_loss").get<double>()) << "**\n";
    os << "- final loss (best): **" << fmt_g(summary.at("final_loss").get<double>()) << "**\n";
    os << "- relative drop: **" << std::fixed << std::setprecision(2)
       << (summary.at("relative_drop").get<double>() * 100.0) << "%**\n";
    os << "- best at iter: **" << summary.at("best_at_iter").get<int>() << "**\n\n";

    os << "## Timings (ms)\n\n";
    if (timings.is_object() && !timings.empty()) {
        // Sorted by key already (std::map in RunRecorder).
        for (auto it = timings.begin(); it != timings.end(); ++it) {
            os << "- `" << it.key() << "`: " << it.value() << "\n";
        }
    } else {
        os << "_(no phases recorded)_\n";
    }

    os << "\n## Config\n\n```json\n";
    os << config.dump(2) << "\n";
    os << "```\n\n";

    os << "_See `run_report.json` for the machine-readable trace._\n";
    return os.str();
}

} // namespace

std::pair<std::string, std::string>
write_run_report(const nlohmann::json& report, const std::string& out_dir) {
    validate_run_report_payload(report);

    std::filesystem::path out{out_dir};
    std::filesystem::create_directories(out);

    const std::filesystem::path json_path = out / "run_report.json";
    const std::filesystem::path md_path   = out / "run_report.md";

    {
        std::ofstream f(json_path);
        if (!f) {
            throw std::runtime_error("write_run_report: cannot open "
                                     + json_path.string());
        }
        f << std::setw(2) << report << std::endl;
    }
    {
        std::ofstream f(md_path);
        if (!f) {
            throw std::runtime_error("write_run_report: cannot open "
                                     + md_path.string());
        }
        f << render_md(report);
    }
    return {json_path.string(), md_path.string()};
}

} // namespace sinic
