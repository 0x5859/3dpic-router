"""dual_annealing + layer_swap polish (REFACTOR_GOALS.md §2-3 目标 C).

A two-phase optimizer for L >= 3 search spaces where vanilla scipy
``dual_annealing`` (operating on a continuous box and rounding to ints at
the end) leaves discrete moves on the table:

1. **Phase 1** — run ``scipy.optimize.dual_annealing`` exactly like
   :class:`DualAnnealingOptimizer`. The result vector is rounded to
   integer layers.
2. **Phase 2** — random-pair hill-climbing on the integer layer vector.
   Each polish iteration picks two distinct edge indices, swaps their
   layer labels, evaluates the loss, and reverts the swap if the new
   loss is not strictly lower (greedy accept). Stops after
   ``polish_iters`` evaluations (default 2000).

Registered separately as ``"dual_annealing_with_swap_polish"`` so that the
default ``"dual_annealing"`` baseline numbers do not change — per
REFACTOR_GOALS.md §7 Q-b suggestion.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.optimize import dual_annealing

from ..statistics import IterEvent
from .base import OptimizationResult, Optimizer, register_optimizer


@register_optimizer("dual_annealing_with_swap_polish")
class DualAnnealingWithSwapPolishOptimizer(Optimizer):
    """dual_annealing run, then random-pair layer-swap hill climbing.

    Optimizer kwargs:
      - ``polish_iters`` (int, default 2000): number of polish **loss
        evaluations** (not raw attempts; degenerate swaps that are
        skipped before the evaluator runs do not count).
      - ``polish_seed`` (int | None, default ``self.seed``): RNG seed used
        for the polish step (kept separate from scipy's ``seed`` so the
        same DA outcome can be polished with different random sweeps).
      - all other kwargs are forwarded to ``scipy.optimize.dual_annealing``.

    The ``StatsSink`` receives one ``IterEvent`` per loss evaluation across
    BOTH phases — scipy's evals and the polish swaps share the same
    monotonic ``iter`` counter and ``wall_ms`` clock origin (this method's
    entry). ``is_new_best`` is computed by the sink, so the polish phase's
    accepted swaps light up as new-best markers in ``run_report.json``.

    Perimeter edges are excluded from the swap sampling pool: pre-M3
    sampling drew from all edges, but :meth:`SiNInterconnectionGraph.loss_function`
    re-pins perimeter indices to ``perimeter_layer`` before evaluation,
    so a perimeter-perimeter swap is a definitional no-op and a
    perimeter/non-perimeter swap effectively erases the non-perimeter
    edge's contribution. Restricting the pool turns ``polish_iters`` into
    a useful budget of *actual* random-swap moves rather than wasted
    samples (opus-review P1-4 / gpt-review P1-6).
    """

    name = "dual_annealing_with_swap_polish"

    def optimize(self, maxiter: int = 1000) -> OptimizationResult:
        t0 = time.perf_counter_ns()

        n_edges = len(self.graph.G.edges)
        L = int(getattr(self.graph, "L", 2))
        if L < 2:
            raise ValueError(
                f"dual_annealing_with_swap_polish has no search space when "
                f"L={L}; L >= 2 is required."
            )
        bounds = [(0, L - 1)] * n_edges
        # opus-review-3 P1-B: same seed-layer parameterization as
        # ``dual_annealing``.
        ecl = int(getattr(self.graph, "edge_coupler_layer", 0))
        seed_layer = (ecl + 1) % L
        x0 = self._seed_from_routing(takeaway=[3, 5], layer=seed_layer)

        sink = self.sink
        loss_fn = self.graph.loss_function
        counter = itertools.count()

        def wrapped(x):
            loss = loss_fn(x)
            sink.on_iter(
                IterEvent(
                    iter=next(counter),
                    loss=float(loss),
                    wall_ms=(time.perf_counter_ns() - t0) // 1_000_000,
                )
            )
            if self.eval_observer is not None:
                self.eval_observer(x, float(loss))
            return loss

        polish_iters = int(self.kwargs.pop("polish_iters", 2000))
        polish_seed = self.kwargs.pop("polish_seed", self.seed)

        result = dual_annealing(
            wrapped,
            bounds,
            maxiter=maxiter,
            x0=x0,
            seed=self.seed,
            **self.kwargs,
        )

        # Phase 2: random-pair layer-swap hill climbing on the integer
        # layer vector. Polish operates on integer state, but evaluates via
        # ``wrapped`` (same loss_function) so every accept/reject is a real
        # ``IterEvent`` in the sink and the polish counter is contiguous
        # with scipy's evaluation count.
        current_layers = np.round(np.asarray(result.x)).astype(np.int64)
        current_loss = float(result.fun)
        rng = np.random.default_rng(polish_seed)

        # Build the non-perimeter index pool once. ``_perimeter_mask`` is
        # lazily built by Phase A (already populated above via the seed
        # call, which threads through ``_seed_from_routing``-triggered
        # graph state); request explicitly to be safe.
        self.graph.build_crossings_index()
        non_perim_idx = np.where(~self.graph._perimeter_mask)[0]

        if non_perim_idx.size < 2:
            # Degenerate (very small graph) — nothing to swap.
            evals_done = 0
        else:
            # opus-review-3 P1-A: bound total attempt count so a
            # post-DA homogeneous state (e.g. all non-perim edges at the
            # same layer — possible at small maxiter or large
            # loss_taper) does not spin forever in the ``continue``
            # branch. ``max_attempts = polish_iters * 8`` gives the
            # rejection-sampler comfortable room (P[same layer]≈1/L
            # at uniform pinned distribution, so 8× covers L up to
            # ~6 with margin) while preventing unbounded loops.
            max_attempts = max(polish_iters * 8, 32)
            evals_done = 0
            attempts = 0
            while evals_done < polish_iters and attempts < max_attempts:
                attempts += 1
                pair = rng.choice(non_perim_idx, size=2, replace=False)
                i, j = int(pair[0]), int(pair[1])
                if current_layers[i] == current_layers[j]:
                    # Same-layer swap is a definitional no-op; reject
                    # without consuming an evaluation so ``polish_iters``
                    # is the count of *actual* moves the user paid for.
                    continue
                current_layers[i], current_layers[j] = (
                    current_layers[j],
                    current_layers[i],
                )
                new_loss = wrapped(current_layers.astype(float))
                evals_done += 1
                if new_loss < current_loss:
                    current_loss = float(new_loss)
                else:
                    # Revert: greedy hill climb only accepts strictly lower.
                    current_layers[i], current_layers[j] = (
                        current_layers[j],
                        current_layers[i],
                    )
            if evals_done < polish_iters:
                import warnings as _w
                _w.warn(
                    f"dual_annealing_with_swap_polish exhausted "
                    f"max_attempts={max_attempts} after only "
                    f"{evals_done}/{polish_iters} effective evaluations — "
                    "post-DA state may be too homogeneous (e.g. all non-perim "
                    "edges on one layer) for random-pair swaps to find "
                    "differing layers. Consider lowering polish_iters or "
                    "increasing scipy maxiter to land a more diverse "
                    "post-DA state.",
                    stacklevel=2,
                )

        best_layers = [int(v) for v in current_layers.tolist()]
        self.graph.apply_optimization_result(best_layers)
        print(f"Optimal average loss (post-polish): {current_loss}")
        return OptimizationResult(
            best_layers=best_layers, fun=current_loss, raw=result
        )
