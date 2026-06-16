#pragma once

// Optimizer abstraction. The pre-M5 main.cpp called the external
// `dual_annealing::minimize` via a nested `MyObjective` struct;
// M5 wraps that behind an `Optimizer` ABC (mirroring Python
// `routing_py_rebuild/optimizers/base.py`) so future strategies
// (differential evolution, swap-polish) plug into the same interface.
//
// The StatsSink hook is the M5 deliverable — `on_iter` fires once per
// loss-function evaluation inside `MyObjective::value`, populating
// the trace[] array of run_report.json (REFACTOR_GOALS.md §1-1-b).

#include "sinic/graph.hpp"
#include "sinic/statistics.hpp"

#include <memory>
#include <string>
#include <vector>

namespace sinic {

struct DualAnnealingParams {
    int   max_iter = 500;
    int   seed     = 0;     // 0 ⇒ std::random_device
    float q_V      = 1.3f;
    float q_A      = -5.0f;
    float t_0      = 5230.0f;
    int   patience = 20;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual std::string name() const = 0;
    virtual void   optimize() = 0;
    virtual double best_loss() const = 0;
    virtual const std::vector<int>& best_layers() const = 0;
};

// M6 (stage-2 review P1-2 mirror): expose the float-to-layer decode
// rule so tests can verify the banker's-rounding contract directly
// at half-points. The same function is used internally by
// `DualAnnealingObjective::value` and `DualAnnealingOptimizer::optimize`
// when reading the best.x array back. Uses `std::nearbyint` which
// respects FE_TONEAREST (banker's rounding), matching Python
// `np.round` semantics at exact half-integer inputs.
int decode_layer_index(float x, int L) noexcept;

class DualAnnealingOptimizer final : public Optimizer {
public:
    // The optimizer holds non-owning references to `graph` and `sink`.
    // Callers must keep both alive for the duration of `optimize()`.
    DualAnnealingOptimizer(SiNInterconnectionGraph& graph,
                           DualAnnealingParams params = {},
                           StatsSink& sink = default_null_sink());

    std::string name() const override { return "dual_annealing"; }
    void   optimize() override;
    double best_loss() const override { return best_loss_; }
    const std::vector<int>& best_layers() const override { return best_layers_; }

private:
    SiNInterconnectionGraph& graph_;
    DualAnnealingParams      params_;
    StatsSink&               sink_;
    double                   best_loss_;
    std::vector<int>         best_layers_;
};

} // namespace sinic
