#pragma once

// Per-edge loss formulas. Pre-M5 these were two member functions on
// SiNInterconnectionGraph (`cal_loss_of_edge`, `update_interlayercrossings_loss`).
// M5 extracts them as free functions so the loss model can evolve
// (REFACTOR_GOALS.md §2-2 / §2-3) without touching the graph class
// boundary that the optimizer and I/O modules depend on.
//
// M6 (REFACTOR_GOALS.md §2-3 目标 B mirror): taper term generalized to
//   per_edge_loss = 2 * loss_crossing * crossings
//                 + 2 * |layer - edge_coupler_layer| * loss_taper
//                 + loss_interlayer * interlayerCrossings
// For the legacy default `edge_coupler_layer=0` this reduces to the
// pre-M3 `2 * layer * loss_taper` since `layer >= 0`, preserving the
// §2-3 验收第 1 条 bit-exact L=2/ecl=0 baseline.

#include "sinic/types.hpp"

namespace sinic {

// Writes EdgeData.loss in place. `crossings` and `layer` are read from
// each edge's EdgeData (filled in by count_crossings_with_detail and
// the layer assignment step respectively). `edge_coupler_layer` shifts
// the taper formula to `2 * |layer - ecl| * loss_taper` (M6); pass 0
// (the default) for legacy bit-exact behavior.
void compute_per_edge_loss(EdgeList& edges,
                           double loss_crossing,
                           double loss_taper,
                           int    edge_coupler_layer = 0) noexcept;

// Adds interlayer-crossing loss to each edge's existing
// EdgeData.loss (read from EdgeData.interlayerCrossings). Must be
// invoked AFTER compute_per_edge_loss because it is purely additive.
void apply_interlayer_loss(EdgeList& edges,
                           double loss_interlayer_crossing) noexcept;

} // namespace sinic
