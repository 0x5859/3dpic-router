#pragma once

// Channel-to-channel crosstalk tensor — C++ mirror of
// routing_py_rebuild/crosstalk.py (REFACTOR_GOALS.md §2-2 / M7).
//
// The crosstalk tensor `crosstalk[s][d][t]` (in dB) reports how much
// power leaks onto victim node `t` when the main path `s → d` is
// energized. The model is a first-order incoherent (power-additive)
// propagation over the cached crossing topology built by Phase A
// (M2 / §2-1, mirrored in C++ by `CrossingTopology` from M6).
//
// Coefficient semantics (matches Python `crosstalk.py` exactly):
//   - `loss_crossing` is reinterpreted **as a fractional power loss**
//     here, even though the per-edge loss formula in `loss.cpp`
//     treats it as a dB-equivalent additive coefficient. This is the
//     §2-2 model simplification ("junction transmission (main path):
//     1 − loss_crossing"). The `coherence_model="incoherent_v1"`
//     label on the emitted tensor records this fact.
//   - `loss_intralayer_crosstalk` / `loss_interlayer_crosstalk` are
//     fractional power per crossing (default 1e-4 / 1e-5, i.e.
//     −40 dB / −50 dB). A `std::nullopt` value is treated as zero —
//     the tensor is then empty (all-null entries) and the function
//     returns early without recursion.
//   - `leak` is split 50/50 between the two directions along the
//     orthogonal edge (matching §2-2 "从交点向两端传播"). The split
//     is symmetric by construction; downstream readers should not
//     depend on it being asymmetric.
//   - branches recurse up to `max_hops` levels; the main path is
//     hop=0, each branch transition increments hop by one.
//   - linear-power threshold (`10^(threshold_db / 10)`) prunes
//     deeper branches whose accumulated power would already be below
//     the noise floor.
//   - a per-path visited set of edge indices prevents loops and
//     double-counting (an edge is "visited" once any path has walked
//     along it).
//
// Output: a `nlohmann::json` object suitable for direct JSON
// serialization (matches the §3-1 `crosstalk` field schema). Diagonal
// positions (s==d / s==t / t==d) and below-threshold arrivals are
// JSON `null`. Above-threshold arrivals are stored as dB
// (`10 * log10(power)`).
//
// The engine is invoked **once after optimization** by
// `main.cpp` / external orchestrators; it is NOT in the optimizer
// hot loop. `compute_crosstalk_include_in_loss=true` is reserved
// for a future milestone (out of scope for M7, just like in M4).

#include "sinic/graph.hpp"

#include <nlohmann/json.hpp>

namespace sinic {

// Build the rank-3 crosstalk tensor for `graph` (post-optimization).
//
// Parameters:
//   graph        — the graph; reads per-edge `.layer` (post-pin
//                  assignment from the most-recent
//                  `apply_optimization_result`) plus the crosstalk
//                  coefficients / physical-model labels stored on
//                  the graph at construction.
//   max_hops     — recursion depth limit (default 3 per §2-2;
//                  `max_hops=0` disables crosstalk entirely, leaving
//                  the tensor all-null).
//   threshold_db — below-threshold branches (linear power) are
//                  pruned. Must be finite.
//
// Returns:
//   nlohmann::json with the structure documented in §3-1:
//     {
//       "unit": "dB",
//       "shape": [k, k, k],
//       "coherence_model": graph.coherence_model(),
//       "polarization":    graph.polarization(),
//       "symmetric": false,
//       "diagonal_convention": "NaN",
//       "max_hops": int,
//       "threshold_db": double,
//       "loss_intralayer_crosstalk": [number|null],
//       "loss_interlayer_crosstalk": [number|null],
//       "values": [[[number|null]*k]*k]*k,
//     }
//
// Raises (as `std::invalid_argument`):
//   - `max_hops < 0`
//   - `threshold_db` non-finite (matches Python's
//     `math.isfinite` guard against `±inf` / `NaN` leaking into the
//     downstream JSON writer)
//   - `loss_crossing >= 1` (transmission falls outside (0, 1] —
//     the model is non-physical; catches a caller who passed a dB
//     value where a fraction was expected)
//
// Side effects:
//   - Lazily builds `graph.topology()` via `build_crossings_index()`
//     **only when** at least one non-zero coefficient is provided
//     AND `max_hops > 0`. Callers that short-circuit (both
//     coefficients zero/`nullopt`, or `max_hops=0`) pay zero
//     geometric cost (mirrors Python's codex-P2 fix).
nlohmann::json compute_crosstalk_tensor(
    SiNInterconnectionGraph& graph,
    int max_hops = 3,
    double threshold_db = -60.0);

} // namespace sinic
