#pragma once

// C++ mirror of routing_py_rebuild/statistics/. See REFACTOR_GOALS.md
// §1-1-b for the spec; behavior matches §1-1-a (the Python module) so
// the M8 parity fixture (`PythonParityFixture`) can compare both ends
// against the same deterministic input sequence.
//
// Design constraints (REFACTOR_GOALS.md §1-1-b 已知风险):
//   - `iter` semantics = "loss_function eval count", NOT optimizer
//     outer iteration. The external dual-annealing library, like
//     scipy's dual_annealing, has no per-step callback. This is a
//     fixed run_report.schema.json v1.0 invariant; changing it
//     requires a schema_version bump (§3-3 强制机制).
//   - Non-finite losses (NaN / ±Inf) are dropped at `RunRecorder::on_iter`
//     entry — see Python recorder.py:42-50. `summary.iterations` reflects
//     the *kept* count, which the M5/M8 PythonParityFixture must
//     duplicate exactly to compare summaries.
//   - PhaseTimer is RAII over std::chrono::steady_clock. On macOS / Linux
//     resolution is ns; Windows is ~16 ms default — `wall_ms` precision
//     is documented as "at least 1 ms". The `run_report.md` rendering
//     in `reporters.{h,cpp}` will surface this in the report header.

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace sinic {

struct IterEvent {
    int    iter;         // loss_function call count
    double loss;
    long   wall_ms;      // ms since Optimizer::optimize() entry
    bool   is_new_best;  // overwritten inside RunRecorder; ignored on entry
};

class StatsSink {
public:
    virtual ~StatsSink() = default;
    virtual void on_iter(const IterEvent& ev) = 0;
    virtual void on_phase(std::string_view name, long wall_ms) = 0;
    virtual nlohmann::json finalize() = 0;
};

// Returns a process-wide no-op sink. Useful as the default argument
// for optimizer constructors that take a StatsSink&. Singleton so
// callers do not need to track lifetime.
StatsSink& default_null_sink();

class NullSink final : public StatsSink {
public:
    void on_iter(const IterEvent&) override {}
    void on_phase(std::string_view, long) override {}
    nlohmann::json finalize() override { return nlohmann::json::object(); }
};

// In-memory accumulator. `finalize()` returns a dict matching
// `code/schema/run_report.schema.json` v1.0.
class RunRecorder final : public StatsSink {
public:
    struct LayerAggregate {
        int layer       = 0;
        int edge_count  = 0;
        int crossings   = 0;
    };

    // `config` is a free-form JSON object mirroring General Parameters.
    RunRecorder(std::string run_id, nlohmann::json config);

    // StatsSink interface
    void on_iter(const IterEvent& ev) override;
    void on_phase(std::string_view name, long wall_ms) override;
    nlohmann::json finalize() override;

    // Optional aggregate stats (REFACTOR_GOALS.md §1-1-a item 3).
    // `convention` must be one of "physical" / "geometric" / "experimental"
    // or std::nullopt (then the field is omitted from the report).
    void set_aggregate(std::vector<LayerAggregate> layers,
                       long crosslayer_crossings_total,
                       std::optional<std::string> convention = std::nullopt);

    // Inspection helpers (test-only; not part of the StatsSink contract).
    std::size_t trace_size() const noexcept { return trace_.size(); }
    double best_so_far() const noexcept { return best_so_far_; }
    int    best_at_iter() const noexcept { return best_at_iter_; }

private:
    std::string                run_id_;
    nlohmann::json             config_;
    std::vector<nlohmann::json> trace_;
    std::map<std::string, long> timings_;
    double                     best_so_far_;
    int                        best_at_iter_;
    std::optional<double>      initial_loss_;
    nlohmann::json             aggregate_;
};

// RAII helper. Scope open/close → one on_phase emission.
class PhaseTimer {
public:
    PhaseTimer(StatsSink& sink, std::string_view name);
    ~PhaseTimer();
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
    PhaseTimer(PhaseTimer&&) = delete;
    PhaseTimer& operator=(PhaseTimer&&) = delete;

private:
    StatsSink&                            sink_;
    std::string                           name_;
    std::chrono::steady_clock::time_point t0_;
};

// ---------------------------------------------------------------------
// Report writer (mirror of routing_py_rebuild/statistics/reporters.py)
//   - Schema-validates the report dict before touching disk
//   - Writes run_report.json (allow-nan=false equivalent: nlohmann's
//     default emits non-strict tokens for NaN/Inf, but RunRecorder
//     filters those upstream so the writer only ever sees finite numbers)
//   - Writes run_report.md with the same fields as the Python renderer
// ---------------------------------------------------------------------

// Returns (json_path, md_path).
std::pair<std::string, std::string>
write_run_report(const nlohmann::json& report, const std::string& out_dir);

} // namespace sinic
