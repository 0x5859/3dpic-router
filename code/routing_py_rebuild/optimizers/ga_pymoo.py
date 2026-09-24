"""Genetic-algorithm optimizer via pymoo (REFACTOR_GOALS.md M-J).

E4-expanded baseline #1: real-valued GA on the continuous [0, L-1]^E
search space. Layer integers are recovered post-optimization by
``apply_optimization_result`` (banker's rounding), matching the same
continuous-decode convention as ``dual_annealing`` and
``differential_evolution`` (§1-2-a / §1-3-c). The GA itself never sees
discrete variables — pymoo's SBX crossover + polynomial mutation operate
on floats — so the optimizer surface is *identical* across the three
existing scipy optimizers and ``ga_pymoo``; only the search strategy
differs. This keeps cross-optimizer comparison about the *algorithm*,
not the encoding.

§1-1-a contract: every loss-function evaluation emits one ``IterEvent``
through the optimizer's ``StatsSink`` and (when set) calls
``eval_observer(x, loss)`` exactly once, at the same site/order as the
scipy wrappers. ``t0`` is captured at ``optimize()`` entry so
``trace[].wall_ms`` shares a clock origin with the outer
``optimization_wall_ms`` ``PhaseTimer`` (§3-2). pymoo's vectorized
``Problem._evaluate`` would batch one ``IterEvent`` per population row
per generation; we use ``ElementwiseProblem`` instead so every individual
evaluation lands a single sink event — matching scipy's per-evaluation
semantics exactly.

Optimizer kwargs (all optional, all override-able via the experiments
harness ``[optimizer_kwargs.ga_pymoo]`` TOML table):
  - ``n_gen`` (int, default 200): pymoo termination ``("n_gen", n_gen)``.
    Independent of the legacy ``maxiter`` kwarg — a GA generation is
    not a scipy iter, so ``maxiter`` would mis-tune the budget by
    1-2 orders of magnitude (e.g. ``maxiter=500`` ⇒ 500 generations
    ⇒ 25 000 evals at default ``pop_size=50``; vs the documented
    10 000 default budget). ``maxiter`` is silently ignored — the
    experiments harness ``[optimizer_kwargs.ga_pymoo]`` table is the
    only way to override the GA budget. (M-J Stage 1 R1 P1: previous
    behaviour ``n_gen = maxiter or 200`` made the default unreachable
    in practice because every call-site passed ``maxiter``.)
  - ``pop_size`` (int, default 50): GA population size. Total loss
    evaluations ≈ ``pop_size * n_gen + pop_size`` (initial pop + each
    generation produces ``pop_size`` offspring).
  - ``crossover_prob`` (float, default 0.9), ``crossover_eta`` (float,
    default 15): SBX (simulated binary crossover) operator knobs.
  - ``mutation_prob`` (float | None, default None ⇒ ``1/n_var`` per
    Deb's polynomial-mutation literature), ``mutation_eta`` (float,
    default 20): polynomial mutation operator knobs. **CRITICAL**:
    pymoo's ``PM(prob=...)`` argument is **per-individual** (the
    probability that the PM kernel is applied to a given offspring at
    all), NOT per-gene. The per-gene mutation rate is pymoo's
    ``prob_var`` (default ``min(0.5, 1/n_var)`` — which IS the Deb
    standard). We therefore route our ``mutation_prob`` kwarg to
    ``prob_var`` and hard-pin pymoo's per-individual ``prob=1.0`` so
    every offspring receives the PM kernel and each gene mutates
    independently at the documented Deb rate. (M-J Stage 1 R1 review
    P0: previously routed to ``prob`` which collapsed mutation activity
    to ≈ 1/n_var² genes/individual, ~60× weaker than the Deb baseline.)
  - ``seed_with_routing_patterns`` (bool, default True): mirror DE's
    seeding behaviour — inject up to four ``routing_method_1`` patterns
    into the initial population so the GA starts near a feasible-looking
    layout instead of pure uniform random.
"""
from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from ..statistics import IterEvent
from .base import (
    BudgetCapExceeded,
    BudgetGuard,
    OptimizationResult,
    Optimizer,
    observer_copy,
    register_optimizer,
)


class _LossProblem(ElementwiseProblem):
    """One-objective elementwise problem wrapping ``graph.loss_function``.

    pymoo calls ``_evaluate`` once per individual per generation; that's
    exactly one ``loss_function`` invocation per call, so the ``sink`` /
    ``eval_observer`` hooks land one-for-one with scipy's wrapper
    semantics. The ``counter`` is shared (a module-level
    ``itertools.count``) so ``IterEvent.iter`` is monotonic across the
    initial population AND every generation, mirroring scipy's evaluation
    counter.
    """

    def __init__(
        self,
        *,
        n_var: int,
        xu: float,
        loss_fn,
        sink,
        t0: int,
        counter,
        eval_observer,
        guard=None,
    ) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=xu)
        self._loss_fn = loss_fn
        self._sink = sink
        self._t0 = t0
        self._counter = counter
        self._eval_observer = eval_observer
        self._guard = guard

    def _evaluate(self, x, out, *args, **kwargs):  # type: ignore[override]
        # ``x`` arrives as a 1-D ndarray of length ``n_var`` (one individual).
        # Treat it as read-only — see §1-1-a observer contract.
        loss = float(self._loss_fn(x))
        self._sink.on_iter(
            IterEvent(
                iter=next(self._counter),
                loss=loss,
                wall_ms=(time.perf_counter_ns() - self._t0) // 1_000_000,
            )
        )
        if self._eval_observer is not None:
            self._eval_observer(observer_copy(x), loss)
        # Stage H budget guard (raises BudgetCapExceeded on cap-hit; the
        # outer optimize() catches it and returns guard's best-so-far).
        if self._guard is not None and self._guard.active:
            self._guard.check_and_record(x, loss)
        out["F"] = loss


@register_optimizer("ga_pymoo")
class GAPymooOptimizer(Optimizer):
    """pymoo single-objective GA on continuous [0, L-1]^E.

    Mirrors the scipy-wrapper structure: build the same continuous
    bounds + (optionally) routing-pattern seeds + per-eval sink emission;
    the final result vector is rounded to ints by
    ``apply_optimization_result`` (banker's rounding via
    ``int(round(...))``), so the resulting layer integers come out of the
    same decode pipeline as ``dual_annealing`` / DE.

    Defaults are conservative: ``pop_size=50``, ``n_gen=200`` ⇒
    ~10 000 loss evaluations per seed — within an order of magnitude of
    scipy DE's ``500 generations * popsize`` budget. The experiments
    harness override knobs live under ``[optimizer_kwargs.ga_pymoo]`` in
    the E4 TOML.
    """

    name = "ga_pymoo"

    _DEFAULT_N_GEN = 200
    _DEFAULT_POP_SIZE = 50
    _DEFAULT_CROSSOVER_PROB = 0.9
    _DEFAULT_CROSSOVER_ETA = 15.0
    _DEFAULT_MUTATION_ETA = 20.0

    def optimize(self, maxiter: int = 1000) -> OptimizationResult:
        # t0 first so trace[].wall_ms shares the same clock origin as the
        # outer optimization_wall_ms PhaseTimer (§3-2).
        t0 = time.perf_counter_ns()

        n_edges = len(self.graph.G.edges)
        L = int(getattr(self.graph, "L", 2))
        if L < 2:
            raise ValueError(
                f"ga_pymoo has no search space when L={L}; L >= 2 is "
                "required. For an L=1 baseline, evaluate "
                "graph.loss_function(np.zeros(n_edges)) directly."
            )

        # ---- knob extraction (use pop() so leftovers stay leftovers) ----
        kw: dict[str, Any] = dict(self.kwargs)
        # maxiter is deliberately ignored — a GA generation is not a
        # scipy iter (see docstring). Default n_gen comes from
        # ``_DEFAULT_N_GEN``; the experiments harness override path is
        # ``[optimizer_kwargs.ga_pymoo].n_gen``.
        n_gen = int(kw.pop("n_gen", self._DEFAULT_N_GEN))
        pop_size = int(kw.pop("pop_size", self._DEFAULT_POP_SIZE))
        crossover_prob = float(
            kw.pop("crossover_prob", self._DEFAULT_CROSSOVER_PROB)
        )
        crossover_eta = float(
            kw.pop("crossover_eta", self._DEFAULT_CROSSOVER_ETA)
        )
        # Deb's PM literature: per-GENE mutation probability = 1/n_var.
        # pymoo's PM exposes two knobs that look similar but have very
        # different semantics:
        #   - PM(prob=p)     -> per-INDIVIDUAL gate (Mutation.do() does
        #                       `mut = rng.random(n_pop) <= prob`; only
        #                       the selected individuals receive ANY
        #                       mutation kernel application).
        #   - PM(prob_var=q) -> per-GENE rate (used inside the PM kernel;
        #                       default ``min(0.5, 1/n_var)`` ≈ Deb).
        # We route our user-facing ``mutation_prob`` to ``prob_var`` and
        # hard-pin ``prob=1.0`` so every offspring receives the PM kernel
        # and each gene mutates independently at the Deb rate. (M-J
        # Stage 1 R1 P0: previously routed to ``prob``, which collapsed
        # activity to ~1/n_var² genes/individual — paper-grade GA needs
        # actual mutation pressure.)
        mutation_prob_raw = kw.pop("mutation_prob", None)
        if mutation_prob_raw is None:
            mutation_prob = 1.0 / float(n_edges) if n_edges > 0 else 0.0
        else:
            mutation_prob = float(mutation_prob_raw)
        mutation_eta = float(
            kw.pop("mutation_eta", self._DEFAULT_MUTATION_ETA)
        )
        seed_with_routing = bool(
            kw.pop("seed_with_routing_patterns", True)
        )
        # Stage H equal-budget caps. Pop before unknown-kwarg check below.
        max_nfe = kw.pop("max_nfe", None)
        wall_time_s = kw.pop("wall_time_s", None)
        if kw:
            # Surface unknown kwargs loudly so a typo in TOML doesn't
            # silently no-op (mirrors DE's behaviour of forwarding
            # unknown args to scipy, but pymoo's GA constructor would
            # raise — we make the error message helpful).
            raise TypeError(
                f"ga_pymoo: unknown optimizer kwargs {sorted(kw.keys())!r}. "
                "Recognised: n_gen, pop_size, crossover_prob, crossover_eta, "
                "mutation_prob, mutation_eta, seed_with_routing_patterns."
            )

        # ---- initial population: optional routing-pattern seeds ----
        xu = float(L - 1) if L > 1 else 0.0
        rng = np.random.default_rng(self.seed)
        init_pop = rng.uniform(low=0.0, high=xu, size=(pop_size, n_edges))
        if seed_with_routing and pop_size >= 1:
            # Mirror DE's pattern set: four routing seeds on a NON-coupler
            # layer (parity with §1-1-a, opus-review-3 P1-B). For L=2/ecl=0
            # this lands on layer 1 — same legacy seed.
            ecl = int(getattr(self.graph, "edge_coupler_layer", 0))
            seed_layer = (ecl + 1) % L
            seed_patterns = [
                self._seed_from_routing(takeaway=[4, 5], layer=seed_layer),
                self._seed_from_routing(takeaway=[3, 4], layer=seed_layer),
                self._seed_from_routing(takeaway=[5], layer=seed_layer),
                self._seed_from_routing(takeaway=[3, 5], layer=seed_layer),
            ]
            for slot, pattern in enumerate(seed_patterns):
                if slot < pop_size:
                    init_pop[slot] = pattern

        # ---- Stage H budget guard ----
        guard = BudgetGuard(
            max_nfe=max_nfe, wall_time_s=wall_time_s, _t0_ns=t0
        )
        # Mirror max_nfe into native n_gen as redundant safety net:
        # nfe ≈ pop_size (initial) + n_gen * pop_size, so
        # n_gen ≈ (max_nfe / pop_size) - 1, with a small buffer.
        if max_nfe is not None:
            n_gen = max(1, max_nfe // pop_size + 1)

        # ---- per-eval sink hookup via ElementwiseProblem ----
        sink = self.sink
        loss_fn = self.graph.loss_function
        counter = itertools.count()
        problem = _LossProblem(
            n_var=n_edges,
            xu=xu,
            loss_fn=loss_fn,
            sink=sink,
            t0=t0,
            counter=counter,
            eval_observer=self.eval_observer,
            guard=guard,
        )

        # ---- pymoo GA assembly ----
        # PM: prob=1.0 (every offspring gets the kernel) + prob_var = our
        # per-gene Deb rate. See the mutation_prob comment block above
        # for the per-individual vs per-gene semantics rationale.
        algorithm = GA(
            pop_size=pop_size,
            sampling=init_pop,  # array form ⇒ pymoo uses these as gen 0
            crossover=SBX(prob=crossover_prob, eta=crossover_eta),
            mutation=PM(prob=1.0, prob_var=mutation_prob, eta=mutation_eta),
            eliminate_duplicates=True,
        )
        termination = get_termination("n_gen", n_gen)

        try:
            result = minimize(
                problem,
                algorithm,
                termination=termination,
                # ``seed=None`` ⇒ pymoo seeds from system entropy
                # (non-deterministic). Mirrors scipy DA's ``seed=None`` path
                # behaviour; the experiments harness always supplies an
                # integer ``cell["seed"]`` so non-determinism is unreachable
                # via that surface.
                seed=self.seed,
                verbose=False,
                save_history=False,
            )
            # pymoo's Result.X / .F should always be populated for a
            # single-objective GA with ``pop_size >= 1``. If we ever see
            # ``None`` it's a pymoo-internal contract break — raise loudly
            # rather than recover with side effects. (M-J Stage 1 R1 P1)
            if result.X is None or result.F is None:
                raise RuntimeError(
                    "ga_pymoo: pymoo minimize() returned Result.X / Result.F "
                    "as None — this is a pymoo-internal contract break for "
                    "single-objective GA. Inspect pop_size, n_gen, and "
                    "sampling; do NOT silently fall back (would pollute the "
                    "sink trace per §1-1-a)."
                )
            best_x = np.asarray(result.X).reshape(-1)
            best_loss = float(np.asarray(result.F).reshape(-1)[0])
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
