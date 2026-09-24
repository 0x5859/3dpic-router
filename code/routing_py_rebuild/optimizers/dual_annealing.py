"""Dual-annealing optimizer (scipy.optimize.dual_annealing)."""
from __future__ import annotations

import itertools
import time

from scipy.optimize import dual_annealing

from ..statistics import IterEvent
from .base import (
    BudgetCapExceeded,
    BudgetGuard,
    OptimizationResult,
    Optimizer,
    observer_copy,
    register_optimizer,
)


@register_optimizer("dual_annealing")
class DualAnnealingOptimizer(Optimizer):
    """Wraps `scipy.optimize.dual_annealing`.

    Mirrors the original ``routing_method_dualsa`` behavior — initial guess
    seeded by ``routing_method_1(takeaway=[3, 5])``. Per REFACTOR_GOALS.md
    §1-1-a, the loss function is wrapped to emit one ``IterEvent`` per
    evaluation; ``wall_ms`` clock origin is this method's entry.

    M3 (§2-3): scipy bounds are derived from ``graph.L`` — ``(0, L-1)`` per
    edge — instead of the pre-M3 hardcoded ``(0, 1)``. For L=2 this is the
    legacy ``(0, 1)`` baseline.
    """

    name = "dual_annealing"

    def optimize(self, maxiter: int = 1000) -> OptimizationResult:
        # t0 first so trace[].wall_ms shares the same clock origin as the
        # outer optimization_wall_ms PhaseTimer (REFACTOR_GOALS.md §3-2).
        t0 = time.perf_counter_ns()

        n_edges = len(self.graph.G.edges)
        L = int(getattr(self.graph, "L", 2))
        # opus-review-3 P1-C: scipy rejects ``min == max`` bounds, so a
        # graph constructed with L=1 (a valid degenerate config — single
        # layer, no routing decision) would crash scipy with a cryptic
        # message. Catch it early with a clear error pointing the user
        # at ``loss_function`` directly.
        if L < 2:
            raise ValueError(
                f"dual_annealing has no search space when L={L}; "
                "L >= 2 is required. For an L=1 baseline, evaluate "
                "graph.loss_function(np.zeros(n_edges)) directly instead."
            )
        # scipy interprets bounds as a closed interval — (0, L-1) inclusive
        # is the correct multi-layer search space; ``apply_optimization_result``
        # rounds x back to the nearest integer in {0, ..., L-1}.
        bounds = [(0, L - 1)] * n_edges
        # opus-review-3 P1-B: seed target layer = (ecl + 1) % L so the
        # routing pattern lands on a NON-coupler layer regardless of
        # where the coupler sits. For L=2/ecl=0 (legacy) this still
        # resolves to layer 1 — bit-exact seed compat. For L=4/ecl=2 it
        # resolves to 3 instead of the previously-hardcoded 1, giving
        # scipy a starting point that's actually two taper hops away
        # from coupler and exercises the L>=3 search space.
        ecl = int(getattr(self.graph, "edge_coupler_layer", 0))
        seed_layer = (ecl + 1) % L
        x0 = self._seed_from_routing(takeaway=[3, 5], layer=seed_layer)

        sink = self.sink
        loss_fn = self.graph.loss_function
        counter = itertools.count()

        # Stage H equal-budget caps: pop from kwargs so scipy doesn't
        # receive unknown args. Defaults None ⇒ guard inactive ⇒
        # byte-exact pre-Stage-H behavior.
        max_nfe = self.kwargs.pop("max_nfe", None)
        wall_time_s = self.kwargs.pop("wall_time_s", None)
        guard = BudgetGuard(
            max_nfe=max_nfe, wall_time_s=wall_time_s, _t0_ns=t0
        )

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
                self.eval_observer(observer_copy(x), float(loss))
            # Track best-so-far + cap check. No-op (zero overhead) when
            # both max_nfe / wall_time_s are None.
            if guard.active:
                guard.check_and_record(x, float(loss))
            return loss

        # Mirror max_nfe into scipy's native maxfun as a redundant
        # safety net (scipy may also cap internally before our guard
        # fires on the last eval).
        scipy_kwargs = dict(self.kwargs)
        if max_nfe is not None and "maxfun" not in scipy_kwargs:
            scipy_kwargs["maxfun"] = int(max_nfe)

        try:
            result = dual_annealing(
                wrapped,
                bounds,
                maxiter=maxiter,
                x0=x0,
                seed=self.seed,
                **scipy_kwargs,
            )
            best_layers = [int(round(v)) for v in result.x]
            best_fun = float(result.fun)
            raw = result
        except BudgetCapExceeded as exc:
            # Cap fired: return guard's best-so-far state.
            best_layers = guard.best_layers
            best_fun = guard.best_loss
            raw = {"cap_reason": str(exc), "nfe": guard.nfe}

        self.graph.apply_optimization_result(best_layers)
        print(f"Optimal average loss: {best_fun}")
        return OptimizationResult(
            best_layers=best_layers, fun=best_fun, raw=raw
        )
