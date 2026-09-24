"""Particle-Swarm-Optimization optimizer via pyswarms (REFACTOR_GOALS.md
M-J Stage 2).

E4-expanded baseline #2: real-valued global-best PSO on the continuous
[0, L-1]^E search space, same surface convention as DA / DE / swap-polish
/ ga_pymoo (continuous → banker's rounding decode in
``apply_optimization_result``). The swarm itself never sees discrete
variables — pyswarms' velocity + position updates operate on floats —
so cross-optimizer comparison stays about the *algorithm*, not the
encoding.

§1-1-a contract: pyswarms' ``optimize(f, iters)`` calls ``f`` with a
``(n_particles, n_var)`` batch matrix once per iteration, expecting back
a ``(n_particles,)`` cost vector. We iterate over the batch inside ``f``,
calling ``graph.loss_function`` once per particle and emitting one
``IterEvent`` + one ``eval_observer`` call per scalar evaluation — the
same one-eval-one-IterEvent semantics as the scipy / pymoo wrappers.
``t0`` is captured at ``optimize()`` entry so ``trace[].wall_ms`` shares
clock origin with the outer ``optimization_wall_ms`` PhaseTimer (§3-2).

Determinism: pyswarms 1.3 has no seed kwarg — it reads from the numpy
global RNG (``numpy.random``) directly. To get reproducible swarms
without leaking global state, we snapshot ``np.random.get_state()``,
seed the global RNG via ``np.random.seed(self.seed)``, run pyswarms,
then restore the prior state in a ``finally`` block. This gives
per-call determinism with no cross-call interference.

Optimizer kwargs (all optional, override-able via the experiments
harness ``[optimizer_kwargs.pso_pyswarms]`` TOML table):
  - ``iters`` (int, default 200): pyswarms ``optimize(iters=...)``.
    Independent of the scipy-style ``maxiter`` kwarg — a PSO iteration
    is not a scipy iter.
  - ``n_particles`` (int, default 50): swarm size. Total loss
    evaluations = ``iters * n_particles``.
  - ``c1`` (float, default 0.5): cognitive coefficient — pull toward
    each particle's personal best.
  - ``c2`` (float, default 0.3): social coefficient — pull toward the
    swarm's global best.
  - ``w`` (float, default 0.9): inertia weight — fraction of the
    previous velocity carried into the next step.
  - ``seed_with_routing_patterns`` (bool, default True): inject up to
    four ``routing_method_1`` patterns into the initial swarm so PSO
    starts near a feasible-looking layout instead of pure uniform
    random — mirrors DE / GA seeding.
"""
from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import pyswarms as ps

from ..statistics import IterEvent
from .base import (
    BudgetCapExceeded,
    BudgetGuard,
    OptimizationResult,
    Optimizer,
    observer_copy,
    register_optimizer,
)


@register_optimizer("pso_pyswarms")
class PSOPyswarmsOptimizer(Optimizer):
    """pyswarms global-best PSO on continuous [0, L-1]^E.

    Mirrors the scipy-wrapper structure (continuous bounds, optional
    routing-pattern seeds, per-eval sink emission). Final result is
    rounded by ``apply_optimization_result`` via banker's rounding,
    matching the cross-optimizer encoding convention. Defaults
    (``iters=200, n_particles=50``) ⇒ 10 000 loss evaluations per seed
    — within an order of magnitude of scipy DE's 6 000-eval budget and
    matched to ga_pymoo for fair side-by-side comparison.
    """

    name = "pso_pyswarms"

    _DEFAULT_ITERS = 200
    _DEFAULT_N_PARTICLES = 50
    _DEFAULT_C1 = 0.5
    _DEFAULT_C2 = 0.3
    _DEFAULT_W = 0.9

    def optimize(self, maxiter: int = 1000) -> OptimizationResult:
        # t0 first so trace[].wall_ms shares the same clock origin as the
        # outer optimization_wall_ms PhaseTimer (§3-2).
        t0 = time.perf_counter_ns()

        n_edges = len(self.graph.G.edges)
        L = int(getattr(self.graph, "L", 2))
        if L < 2:
            raise ValueError(
                f"pso_pyswarms has no search space when L={L}; L >= 2 is "
                "required. For an L=1 baseline, evaluate "
                "graph.loss_function(np.zeros(n_edges)) directly."
            )

        # ---- knob extraction (pop() so leftovers stay leftovers) ----
        kw: dict[str, Any] = dict(self.kwargs)
        # ``maxiter`` is the scipy-iter alias and is deliberately ignored —
        # a PSO iteration is not a scipy iter. The experiments harness
        # override path is ``[optimizer_kwargs.pso_pyswarms].iters``.
        iters = int(kw.pop("iters", self._DEFAULT_ITERS))
        n_particles = int(kw.pop("n_particles", self._DEFAULT_N_PARTICLES))
        c1 = float(kw.pop("c1", self._DEFAULT_C1))
        c2 = float(kw.pop("c2", self._DEFAULT_C2))
        w = float(kw.pop("w", self._DEFAULT_W))
        seed_with_routing = bool(kw.pop("seed_with_routing_patterns", True))
        # Stage H equal-budget caps
        max_nfe = kw.pop("max_nfe", None)
        wall_time_s = kw.pop("wall_time_s", None)
        if kw:
            raise TypeError(
                f"pso_pyswarms: unknown optimizer kwargs {sorted(kw.keys())!r}. "
                "Recognised: iters, n_particles, c1, c2, w, "
                "seed_with_routing_patterns, max_nfe, wall_time_s."
            )
        # Mirror max_nfe into native iters as redundant safety net
        if max_nfe is not None:
            iters = max(1, max_nfe // n_particles + 1)

        # ---- bounds + initial swarm position ----
        xu = float(L - 1) if L > 1 else 0.0
        low = np.zeros(n_edges)
        high = np.full(n_edges, xu)
        bounds = (low, high)

        # pyswarms reads numpy global RNG; snapshot + restore so this
        # optimizer is deterministic per-call AND non-perturbing to the
        # caller's RNG state.
        _saved_state = np.random.get_state()
        try:
            if self.seed is not None:
                np.random.seed(int(self.seed))
            rng = np.random.default_rng(self.seed)
            init_pos = rng.uniform(low=0.0, high=xu, size=(n_particles, n_edges))
            if seed_with_routing and n_particles >= 1:
                # Mirror DE / GA pattern seeds: four routing-method-1
                # variants on a NON-coupler layer (opus-review-3 P1-B
                # parity).
                ecl = int(getattr(self.graph, "edge_coupler_layer", 0))
                seed_layer = (ecl + 1) % L
                seed_patterns = [
                    self._seed_from_routing(takeaway=[4, 5], layer=seed_layer),
                    self._seed_from_routing(takeaway=[3, 4], layer=seed_layer),
                    self._seed_from_routing(takeaway=[5], layer=seed_layer),
                    self._seed_from_routing(takeaway=[3, 5], layer=seed_layer),
                ]
                for slot, pattern in enumerate(seed_patterns):
                    if slot < n_particles:
                        init_pos[slot] = pattern

            # ---- per-eval sink hookup ----
            sink = self.sink
            loss_fn = self.graph.loss_function
            counter = itertools.count()
            observer = self.eval_observer
            guard = BudgetGuard(
                max_nfe=max_nfe, wall_time_s=wall_time_s, _t0_ns=t0
            )

            def batch_loss(X: np.ndarray) -> np.ndarray:
                """pyswarms calls this with (n_particles, n_var); return
                (n_particles,) costs. We loop inside so every scalar
                ``loss_function`` evaluation lands one ``IterEvent`` +
                one ``eval_observer`` call (§1-1-a contract)."""
                costs = np.empty(X.shape[0])
                for i in range(X.shape[0]):
                    x = X[i]
                    loss = float(loss_fn(x))
                    sink.on_iter(
                        IterEvent(
                            iter=next(counter),
                            loss=loss,
                            wall_ms=(time.perf_counter_ns() - t0) // 1_000_000,
                        )
                    )
                    if observer is not None:
                        observer(observer_copy(x), loss)
                    if guard.active:
                        guard.check_and_record(x, loss)
                    costs[i] = loss
                return costs

            optimizer = ps.single.GlobalBestPSO(
                n_particles=n_particles,
                dimensions=n_edges,
                options={"c1": c1, "c2": c2, "w": w},
                bounds=bounds,
                init_pos=init_pos,
            )

            try:
                # pyswarms 1.3's ``optimize(verbose=False)`` is fully silent
                # on both stdout and stderr (M-J Stage 2 R1 P1).
                best_cost, best_pos = optimizer.optimize(
                    batch_loss, iters=iters, verbose=False
                )
                best_x = np.asarray(best_pos).reshape(-1)
                best_loss = float(best_cost)
                best_layers = [int(round(v)) for v in best_x]
                raw = None
            except BudgetCapExceeded as exc:
                best_layers = guard.best_layers
                best_loss = guard.best_loss
                raw = {"cap_reason": str(exc), "nfe": guard.nfe}
        finally:
            np.random.set_state(_saved_state)

        self.graph.apply_optimization_result(best_layers)
        print(f"Optimal average loss: {best_loss}")
        return OptimizationResult(
            best_layers=best_layers, fun=best_loss,
            raw=raw if raw is not None else optimizer,
        )
