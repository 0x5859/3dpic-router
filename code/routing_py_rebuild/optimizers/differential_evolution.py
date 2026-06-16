"""Differential-evolution optimizer (scipy.optimize.differential_evolution)."""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.optimize import differential_evolution

from ..statistics import IterEvent
from .base import (
    BudgetCapExceeded,
    BudgetGuard,
    OptimizationResult,
    Optimizer,
    register_optimizer,
)


@register_optimizer("differential_evolution")
class DifferentialEvolutionOptimizer(Optimizer):
    """Wraps `scipy.optimize.differential_evolution`.

    Mirrors the original ``routing_method_differential_evolution`` — population
    seeded with four ``routing_method_1`` patterns and ``self.k - 1`` random
    members; ``strategy='rand2exp'``, ``popsize=k``, ``mutation=(0.3, 0.8)``,
    ``recombination=0.6`` are kept as defaults but can be overridden via
    ``optimizer_kwargs`` when constructing the optimizer.
    """

    name = "differential_evolution"

    DEFAULTS = dict(
        strategy="rand2exp",
        mutation=(0.3, 0.8),
        recombination=0.6,
    )

    def optimize(self, maxiter: int = 100) -> OptimizationResult:
        # t0 first so trace[].wall_ms shares the same clock origin as the
        # outer optimization_wall_ms PhaseTimer (REFACTOR_GOALS.md §3-2).
        t0 = time.perf_counter_ns()

        # Guard against multi-worker DE per §1-1-a Q-d 暂行决定:
        # workers > 1 would fork the sink across processes, which the
        # current closure-based wrapper and in-memory recorder do not
        # support. Fail loudly so the race is visible, not silent.
        if self.kwargs.get("workers", 1) not in (1, None):
            raise NotImplementedError(
                "differential_evolution with workers > 1 is not supported in M1 — "
                "the StatsSink + closure wrapper assumes single-process eval. "
                "See REFACTOR_GOALS.md §1-1-a 已知风险 (multi-process loss_function) "
                "and Q-d for the deferred design."
            )

        n_edges = len(self.graph.G.edges)
        L = int(getattr(self.graph, "L", 2))
        # opus-review-3 P1-C: scipy bounds require min < max; L=1 must
        # be rejected here rather than crashing inside scipy.
        if L < 2:
            raise ValueError(
                f"differential_evolution has no search space when L={L}; "
                "L >= 2 is required."
            )
        # M3 (§2-3): bounds widen to (0, L-1) per edge. For L=2 this is the
        # legacy (0, 1) baseline; for L>=3 the search space scales linearly.
        bounds = [(0, L - 1)] * n_edges

        # population size defaults to k (matches the original)
        popsize = self.kwargs.pop("popsize", self.graph.k)
        # let the caller override init (e.g. "sobol") to opt back into the
        # original buggy behavior; otherwise we seed with routing patterns.
        user_init = self.kwargs.pop("init", None)

        if user_init is None:
            rng = np.random.default_rng(self.seed)
            # opus-review-3 P1-B: parameterize seed target layer to
            # (ecl + 1) % L so the four DE seed patterns sit on a
            # non-coupler layer regardless of where the coupler is. For
            # L=2/ecl=0 (legacy) this preserves layer=1, bit-exact seed.
            ecl = int(getattr(self.graph, "edge_coupler_layer", 0))
            seed_layer = (ecl + 1) % L
            seed_patterns = [
                self._seed_from_routing(takeaway=[4, 5], layer=seed_layer),
                self._seed_from_routing(takeaway=[3, 4], layer=seed_layer),
                self._seed_from_routing(takeaway=[5], layer=seed_layer),
                self._seed_from_routing(takeaway=[3, 5], layer=seed_layer),
            ]
            # M3 / opus-review P1-1: scale random members across the full
            # [0, L-1] continuous range so the initial population actually
            # samples high layers when L>=3. Pre-fix ``rng.random((...))``
            # only produced floats in ``[0, 1)``; after scipy's
            # round-to-int it collapsed onto layers {0, 1}, leaving L>=3
            # search-space exploration entirely to mutation, which slows
            # convergence at the L-edge.
            init_population = rng.uniform(
                low=0.0, high=float(L - 1) if L > 1 else 0.0,
                size=(popsize, n_edges),
            )
            for slot, pattern in enumerate(seed_patterns):
                if slot < popsize:
                    init_population[slot] = pattern
            init_arg = init_population
        else:
            init_arg = user_init

        merged = {**self.DEFAULTS, **self.kwargs}

        # Stage H equal-budget caps (pop from merged so they don't reach scipy)
        max_nfe = merged.pop("max_nfe", None)
        wall_time_s = merged.pop("wall_time_s", None)
        guard = BudgetGuard(
            max_nfe=max_nfe, wall_time_s=wall_time_s, _t0_ns=t0
        )
        # Map max_nfe to native maxiter (DE's nfe ≈ popsize * (maxiter+1));
        # use as a redundant safety net so scipy stops itself even if our
        # guard misses by one eval at the tail.
        scipy_maxiter = maxiter
        if max_nfe is not None:
            # ceil to give scipy room to fire the last generation before
            # the guard catches the tail eval. popsize+1 buffer for the
            # initial generation (init_population).
            scipy_maxiter = max(1, max_nfe // popsize + 1)

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
            if guard.active:
                guard.check_and_record(x, float(loss))
            return loss

        try:
            result = differential_evolution(
                wrapped,
                bounds,
                maxiter=scipy_maxiter,
                popsize=popsize,
                init=init_arg,
                seed=self.seed,
                **merged,
            )
            best_layers = [int(round(v)) for v in result.x]
            best_fun = float(result.fun)
            raw = result
        except BudgetCapExceeded as exc:
            best_layers = guard.best_layers
            best_fun = guard.best_loss
            raw = {"cap_reason": str(exc), "nfe": guard.nfe}

        self.graph.apply_optimization_result(best_layers)
        print(f"Optimal average loss: {best_fun}")
        return OptimizationResult(
            best_layers=best_layers, fun=best_fun, raw=raw
        )
