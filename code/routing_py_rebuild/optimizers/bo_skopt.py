"""Bayesian-Optimization optimizer via scikit-optimize (REFACTOR_GOALS.md
M-J Stage 2).

E4-expanded baseline #4 (final): Gaussian-Process Bayesian Optimization
via ``skopt.gp_minimize`` as the surrogate-model / ML-adjacent
representative for the paper reviewer response. Like the other E4
optimizers, search runs on continuous [0, L-1]^E with banker's-rounding
decode (``apply_optimization_result``).

**Honest-baseline framing (REFACTOR_GOALS.md §1-5 #2 / R1 design取舍)**:
Bayesian Optimization's standard "comfort zone" is ~20 dimensions; at
E=66 (K_12 edges) the GP surrogate's quality degrades sharply and
acquisition-function maximization becomes the dominant runtime cost.
E4 includes BO at a deliberately limited budget (``n_calls=120``) to
honestly document this limit — the result is a *real* baseline that
shows where surrogate-model methods land relative to direct
evolutionary / swarm / ES strategies on this routing instance, NOT a
tuned-to-win submission. Per the paper-reviewer-response narrative,
BO underperforming GA / PSO / CMA-ES at E=66 is the intended scientific
conclusion, not a bug.

§1-1-a contract: ``skopt.gp_minimize`` takes a scalar objective
(``list[float] -> float``), so per-evaluation sink emission is exactly
one ``IterEvent`` + one ``eval_observer`` call per ``loss_function``
invocation. ``t0`` is captured at ``optimize()`` entry so
``trace[].wall_ms`` shares clock origin with the outer
``optimization_wall_ms`` PhaseTimer (§3-2). The function-call count
matches ``n_calls`` exactly (skopt does ``n_initial_points`` random /
LHS evaluations, then ``n_calls - n_initial_points`` GP-driven calls).

Determinism: skopt accepts a ``random_state`` int. We forward
``self.seed`` directly; ``None`` ⇒ skopt seeds from system entropy.

Optimizer kwargs (all optional, override-able via the experiments
harness ``[optimizer_kwargs.bo_skopt]`` TOML table):
  - ``n_calls`` (int, default 120): total budget of loss evaluations.
    Per the honest-baseline framing, this is intentionally small;
    scaling up does not buy proportional GP quality at E=66.
  - ``n_initial_points`` (int, default 20): random / LHS evaluations
    before the GP kicks in. Skopt requires this; the documented
    recommendation is ``max(10, 0.2 * n_calls)``, our 20 lands at the
    floor.
  - ``acq_func`` (str, default ``"EI"``): acquisition function. Other
    valid skopt values include ``"PI"`` (probability-of-improvement),
    ``"LCB"`` (lower-confidence-bound), ``"gp_hedge"`` (skopt's
    default, randomly chooses each call).
  - ``initial_point_generator`` (str, default ``"lhs"``):
    initial-point sampling strategy. ``"random"`` is uniform; ``"lhs"``
    (Latin hypercube) is better space-filling at low budget;
    ``"sobol"`` / ``"halton"`` are quasi-random alternatives.
"""
from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from ..statistics import IterEvent
from .base import (
    BudgetCapExceeded,
    BudgetGuard,
    OptimizationResult,
    Optimizer,
    register_optimizer,
)


@register_optimizer("bo_skopt")
class BOSkoptOptimizer(Optimizer):
    """scikit-optimize Bayesian Optimization on continuous [0, L-1]^E.

    Honest-baseline baseline for the paper reviewer response: BO at
    E=66 with ``n_calls=120`` is documented to underperform direct
    evolutionary / swarm / ES baselines, which IS the scientific
    conclusion the paper makes about surrogate-model methods on this
    routing-instance scale.
    """

    name = "bo_skopt"

    _DEFAULT_N_CALLS = 120
    _DEFAULT_N_INITIAL = 20
    _DEFAULT_ACQ_FUNC = "EI"
    _DEFAULT_INIT_GEN = "lhs"

    def optimize(self, maxiter: int = 1000) -> OptimizationResult:
        # t0 first so trace[].wall_ms shares clock origin with the outer
        # optimization_wall_ms PhaseTimer (§3-2).
        t0 = time.perf_counter_ns()

        n_edges = len(self.graph.G.edges)
        L = int(getattr(self.graph, "L", 2))
        if L < 2:
            raise ValueError(
                f"bo_skopt has no search space when L={L}; L >= 2 is required. "
                "For an L=1 baseline, evaluate "
                "graph.loss_function(np.zeros(n_edges)) directly."
            )

        # ---- knob extraction ----
        kw: dict[str, Any] = dict(self.kwargs)
        # ``maxiter`` is the scipy-iter alias and is deliberately ignored —
        # BO's evaluation budget is ``n_calls``, not iterations of an
        # outer loop. The experiments harness override path is
        # ``[optimizer_kwargs.bo_skopt].n_calls``.
        n_calls = int(kw.pop("n_calls", self._DEFAULT_N_CALLS))
        n_initial_points = int(
            kw.pop("n_initial_points", self._DEFAULT_N_INITIAL)
        )
        acq_func = str(kw.pop("acq_func", self._DEFAULT_ACQ_FUNC))
        init_gen = str(
            kw.pop("initial_point_generator", self._DEFAULT_INIT_GEN)
        )
        # Stage H equal-budget caps
        max_nfe = kw.pop("max_nfe", None)
        wall_time_s = kw.pop("wall_time_s", None)
        if kw:
            raise TypeError(
                f"bo_skopt: unknown optimizer kwargs {sorted(kw.keys())!r}. "
                "Recognised: n_calls, n_initial_points, acq_func, "
                "initial_point_generator, max_nfe, wall_time_s."
            )
        # max_nfe overrides n_calls natively (BO's n_calls IS its NFE).
        if max_nfe is not None:
            n_calls = int(max_nfe)

        if n_initial_points >= n_calls:
            # M-J Stage 2 R1 P1 (Opus): equality is the worst case —
            # ``n_initial_points == n_calls`` consumes the ENTIRE
            # budget on random / LHS sampling with ZERO GP-driven
            # calls (degenerate to LHS sweep, not BO). Block both
            # strict-greater AND equality.
            raise ValueError(
                f"bo_skopt: n_initial_points={n_initial_points} >= "
                f"n_calls={n_calls} would consume the entire budget on "
                "random sampling with no GP-driven calls (degenerates "
                "BO to LHS sweep); raise n_calls or lower n_initial_points."
            )

        # ---- search space ----
        xu = float(L - 1) if L > 1 else 0.0
        space = [Real(0.0, xu, prior="uniform") for _ in range(n_edges)]

        # ---- per-eval sink hookup ----
        sink = self.sink
        loss_fn = self.graph.loss_function
        counter = itertools.count()
        observer = self.eval_observer
        guard = BudgetGuard(
            max_nfe=max_nfe, wall_time_s=wall_time_s, _t0_ns=t0
        )

        def scalar_loss(x_list):
            """skopt calls this with a Python list (one candidate per
            call). Exactly one ``loss_function`` invocation ⇒ one
            ``IterEvent`` + one ``eval_observer`` call (§1-1-a)."""
            x = np.asarray(x_list, dtype=float)
            loss = float(loss_fn(x))
            sink.on_iter(
                IterEvent(
                    iter=next(counter),
                    loss=loss,
                    wall_ms=(time.perf_counter_ns() - t0) // 1_000_000,
                )
            )
            if observer is not None:
                observer(x, loss)
            if guard.active:
                guard.check_and_record(x, loss)
            return loss

        try:
            # skopt is occasionally noisy on stderr (GP convergence
            # warnings, acquisition-fn fallbacks). E4's honest-baseline
            # framing accepts this — those warnings ARE part of the
            # scientific signal that BO struggles at E=66. We do NOT
            # silence them.
            result = gp_minimize(
                func=scalar_loss,
                dimensions=space,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                initial_point_generator=init_gen,
                acq_func=acq_func,
                random_state=self.seed,
                verbose=False,
                # Explicit n_jobs=1: skopt default also 1; pin per §7 Q-d.
                n_jobs=1,
            )

            if result.x is None or result.fun is None:
                raise RuntimeError(
                    "bo_skopt: gp_minimize() returned x / fun as None — "
                    "this is a skopt-internal contract break. Inspect "
                    "n_calls, n_initial_points, and dimensions; do NOT "
                    "silently fall back (would pollute the sink trace "
                    "per §1-1-a)."
                )

            best_x = np.asarray(result.x).reshape(-1)
            best_loss = float(result.fun)
            best_layers = [int(round(v)) for v in best_x]
            raw = result
        except BudgetCapExceeded as exc:
            best_layers = guard.best_layers
            best_loss = guard.best_loss
            raw = {"cap_reason": str(exc), "nfe": guard.nfe}

        self.graph.apply_optimization_result(best_layers)
        print(f"Optimal average loss: {best_loss}")
        return OptimizationResult(
            best_layers=best_layers, fun=best_loss, raw=raw
        )
