#include "sinic/optimizer.hpp"

#include "sinic/graph.hpp"
#include "sinic/statistics.hpp"

// External library headers. These come from `external/dual-annealing/`
// via the legacy build wiring; we keep the include style identical to
// pre-M5 main.cpp so the library's per-translation-unit thread-local
// workspace caches behave the same.
#include "chain.hpp"
#include <pcg_random.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

namespace sinic {

namespace {

// Wraps a graph + sink into the functor shape the external
// `dual_annealing::minimize` expects (free function `.value(span)`
// returning float). Hoisted out of main.cpp::MyObjective.
//
// M6 (REFACTOR_GOALS.md §2-3 mirror): the external library has no
// explicit bounds API — its Tsallis visit distribution generates raw
// floats and clamps them via our `wrap` member function. Keeping
// `wrap` at [0, 1] preserves the library's fixed-step exploration
// behavior; multi-layer is wired in via the **decode** step in
// `value()` (`int(round(x * (L-1)))` instead of the L=2-specific
// `x >= 0.5`). For L=2 this reduces to the legacy threshold and
// preserves the §2-3 验收第 1 条 bit-exact baseline.
struct DualAnnealingObjective {
    SiNInterconnectionGraph* graph;
    StatsSink*                sink;
    std::chrono::steady_clock::time_point t0;
    int eval_counter = 0;

    // Stash the most recent layer assignment so the optimizer can
    // recover it after `dual_annealing::minimize` returns (the library
    // overwrites the input span with `best.x` but does NOT re-evaluate
    // the objective at the best point — see chain.hpp:443).
    std::vector<int> last_layers;

    float wrap(float x) const noexcept {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    }

    // M6: decode raw [0, 1] float → integer layer in [0, L-1].
    //
    // Uses `std::nearbyint`, which respects the current IEEE-754
    // rounding mode (default `FE_TONEAREST` = banker's rounding,
    // round-half-to-even) — matching Python `np.round` semantics.
    // This is the M6 stage-review P1-C fix; pre-fix the formula was
    // `floor(x*(L-1) + 0.5)` (round-half-away-from-zero) which
    // diverged from Python at exact half-points. For inputs that are
    // not exactly half-integers (the vast majority of cases) both
    // formulas agree, so the M5 baseline at L=2 / random seeds is
    // unaffected.
    static int decode_layer(float x, int L) noexcept {
        if (L <= 1) return 0;
        const double scaled = static_cast<double>(x) * static_cast<double>(L - 1);
        const int v = static_cast<int>(std::nearbyint(scaled));
        if (v < 0)     return 0;
        if (v >= L)    return L - 1;
        return v;
    }

    // Not const: the external `dual_annealing::minimize` calls the
    // objective via a non-const reference (chain.hpp:118-129), so
    // there is no benefit to const-qualifying `value` and it lets
    // us mutate `eval_counter` / `last_layers` without going through
    // `mutable` (Python recorder.py also tracks eval-count in a
    // non-const-equivalent field on its own object).
    float value(gsl::span<float const> x) {
        const int L = graph->L();
        std::vector<int> layers;
        layers.reserve(x.size());
        for (auto v : x) layers.push_back(decode_layer(v, L));

        // Snapshot for the optimizer driver. Last write wins.
        last_layers = layers;

        // Mutate graph.allEdges_ exactly like pre-M5 MyObjective::value
        // — perimeter pin (diff = 1 or k-1) → perimeter_layer, else
        // from layers[i]. This redundancy with graph.loss_function() is
        // intentional: pre-M5 binaries observed allEdges_ from the
        // LAST eval, so baseline parity requires the same mutation
        // pattern. M6: perimeter target generalized from hard-coded 0
        // to graph->perimeter_layer().
        const int k     = graph->k();
        const int p_lyr = graph->perimeter_layer();
        auto& all = graph->all_edges_mut();
        for (std::size_t i = 0; i < all.size(); ++i) {
            auto& e = all[i];
            const int u = std::get<0>(e);
            const int v = std::get<1>(e);
            const int diff = std::abs(u - v);
            if (diff == 1 || diff == (k - 1)) {
                std::get<2>(e).layer = p_lyr;
            } else {
                std::get<2>(e).layer = layers[i];
            }
        }

        const double cost = graph->loss_function(layers);
        const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t0).count();
        IterEvent ev{
            /*iter=*/eval_counter++,
            /*loss=*/cost,
            /*wall_ms=*/static_cast<long>(wall_ms),
            /*is_new_best=*/false,
        };
        sink->on_iter(ev);
        return static_cast<float>(cost);
    }

    double value_and_gradient(gsl::span<float const> x,
                              gsl::span<float>       grad) {
        for (auto& g : grad) g = 0.0f;
        return static_cast<double>(value(x));
    }
};

// `DualAnnealingObjective` must satisfy the library's `value`
// detector trait. We cannot tag the struct itself (the trait is in
// the external namespace), so we instead rely on the SFINAE detector
// in chain.hpp picking up `value` automatically.

} // namespace

int decode_layer_index(float x, int L) noexcept {
    return DualAnnealingObjective::decode_layer(x, L);
}

DualAnnealingOptimizer::DualAnnealingOptimizer(SiNInterconnectionGraph& graph,
                                               DualAnnealingParams      params,
                                               StatsSink&               sink)
    : graph_(graph),
      params_(params),
      sink_(sink),
      best_loss_(std::numeric_limits<double>::infinity()) {}

void DualAnnealingOptimizer::optimize() {
    // M6 (REFACTOR_GOALS.md §2-3 目标 A + 第三轮评审 P1-C mirror):
    // L < 2 is rejected at the optimizer surface. scipy
    // `dual_annealing(bounds=[(0,0)])` raises "Bounds are not consistent
    // min < max"; we want the same explicit failure on the C++ side so
    // callers can use a direct `loss_function([0]*E)` call if they only
    // need single-layer evaluation.
    if (graph_.L() < 2) {
        throw std::invalid_argument(
            "DualAnnealingOptimizer requires L >= 2; for L=1 call "
            "graph.loss_function(layers) directly.");
    }

    // M5 dual-review P1 fix (wall_ms origin): capture t0 at the very
    // start of optimize() so that (a) every IterEvent.wall_ms shares
    // the same origin, and (b) the `optimization_wall_ms` phase emitted
    // at the end of optimize() uses the same anchor. Previously the
    // objective's t0 was set after RNG seeding + initial-solution
    // generation, drifting from the PhaseTimer wrapper in main.cpp.
    const auto t0 = std::chrono::steady_clock::now();

    const int n_edges = graph_.num_edges();

    // RNG seeding: 0 ⇒ entropy device (mirrors pre-M5 random_device
    // path so the legacy CLI keeps non-deterministic behavior). Any
    // non-zero `seed` value yields a deterministic stream — required
    // for M5 baseline parity tests and the M8 Python parity fixture.
    pcg32 generator;
    if (params_.seed == 0) {
        std::random_device rd;
        generator.seed(rd());
    } else {
        generator.seed(static_cast<std::uint64_t>(params_.seed));
    }

    std::vector<float> x(n_edges);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for (auto& v : x) v = u01(generator);

    DualAnnealingObjective objective{
        /*graph=*/&graph_,
        /*sink=*/&sink_,
        /*t0=*/t0,
        /*eval_counter=*/0,
        /*last_layers=*/{},
    };

    const dual_annealing::param_t params{
        /*q_V=*/params_.q_V,
        /*q_A=*/params_.q_A,
        /*t_0=*/params_.t_0,
        /*num_iter=*/static_cast<size_t>(params_.max_iter),
        /*patience=*/static_cast<size_t>(params_.patience),
    };

    tcm::lbfgs::lbfgs_param_t local_search_params;
    local_search_params.x_tol    = 1e-5;
    local_search_params.max_iter = 0; // disabled — matches pre-M5 main.cpp

    auto result = dual_annealing::minimize(
        objective, gsl::span<float>{x.data(), x.size()},
        params, local_search_params, generator);

    best_loss_ = result.func;
    // `x` now holds best.x (memcpy at the end of minimize). Use it
    // directly rather than relying on objective.last_layers, which
    // points at the LAST evaluated solution (not the best). M6: the
    // L-aware decode lives in `DualAnnealingObjective::decode_layer`
    // — for L=2 it reduces to the legacy `>= 0.5` threshold.
    const int L = graph_.L();
    best_layers_.assign(n_edges, 0);
    for (int i = 0; i < n_edges; ++i) {
        best_layers_[i] = DualAnnealingObjective::decode_layer(x[i], L);
    }

    // Note: graph.all_edges_mut() is left in the state from the LAST
    // objective.value() call (i.e. `last_layers`, NOT `best_layers_`).
    // Pre-M5 main.cpp had the same property: the saved JSON reflects
    // whatever the chain last evaluated, not the best solution. The
    // M5 orchestrator (`main.cpp`) opts in to a fix by calling
    // graph.apply_optimization_result(opt.best_layers()) before
    // saving — that change is documented as the M5 "post-optimize
    // graph reflects best, not last" parity-breaking decision (see
    // REFACTOR_GOALS.md §1-2-b 验收口径).

    std::cout << "Optimal average loss by EXTERNAL Dual Annealing = "
              << best_loss_ << "\n";

    // Emit the optimization_wall_ms phase from the same t0 anchor we
    // gave the objective. M5 dual-review P1: this was previously
    // wrapped in a PhaseTimer in main.cpp, but that PhaseTimer's t0
    // was set BEFORE optimizer construction — drifting from the
    // objective's t0. Folding the phase emission into optimize()
    // guarantees a single time origin shared between trace[].wall_ms
    // and timings.optimization_wall_ms (per REFACTOR_GOALS.md §3-2).
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0).count();
    sink_.on_phase("optimization_wall", static_cast<long>(elapsed_ms));
}

} // namespace sinic
