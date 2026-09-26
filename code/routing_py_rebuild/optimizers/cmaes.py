"""CMA-ES optimizer via the `cma` library (REFACTOR_GOALS.md M-J Stage 2).

E4-expanded baseline #3: Hansen's Covariance Matrix Adaptation Evolution
Strategy — the modern-standard derivative-free continuous optimizer.
Search runs on the same continuous [0, L-1]^E surface as DA / DE /
swap-polish / ga_pymoo / pso_pyswarms (final result rounded by
``apply_optimization_result``). CMA-ES is included in E4 as the
ES-family representative even though continuous ES on a discrete
problem is known to suffer variance collapse near integer-rounded
decision boundaries; this is a real comparison signal, not a bug —
documented in REFACTOR_GOALS.md §1-5 risk list.

§1-1-a contract: the cma library exposes an explicit ``ask()`` /
``tell()`` loop, so we evaluate each candidate scalar-by-scalar in a
plain Python loop — one ``IterEvent`` + one ``eval_observer`` call per
``loss_function`` evaluation, matching scipy / pymoo / pyswarms
semantics. ``t0`` is captured at ``optimize()`` entry so
``trace[].wall_ms`` shares clock origin with the outer
``optimization_wall_ms`` PhaseTimer (§3-2).

Determinism: cma 4.4 accepts a ``seed`` in its options dict, but it
ALSO calls ``np.random.seed(opts['seed'])`` internally on every
``CMAEvolutionStrategy`` construction (cma 4.4 ``evolution_strategy.py``
line ~1107) — so without protection, every cmaes ``optimize()`` call
clobbers the caller's numpy global RNG state. The wrapper snapshots
``np.random.get_state()`` at entry and restores it in a ``finally``
block, matching the ``pso_pyswarms`` isolation pattern (M-J Stage 2 R1
P1, Codex). ``self.seed`` is forwarded to cma's ``seed`` option; we
add 1 to non-negative seeds to avoid cma's "``seed=0`` ⇒ time-based
reseeding" path (verified empirically — two ``seed=0`` constructors
yield different ``opts['seed']`` values).

Optimizer kwargs (all optional, override-able via the experiments
harness ``[optimizer_kwargs.cmaes]`` TOML table):
  - ``maxiter`` (int, default 1000): cma generations (each generation
    evaluates ``popsize`` candidates). When ``max_nfe`` is set (the
    experiments equal-budget runs), ``maxiter`` is auto-expanded to
    ``max_nfe // popsize + 1`` so the NFE budget — not the generation
    count — is the binding stop (parity with DE / GA / PSO / BO).
  - ``popsize`` (int, default 14): population size. cma's documented
    default for n=66 is ``4 + floor(3*ln(66)) ≈ 16``; we pin to 14 so
    the total eval budget (``maxiter * popsize ≈ 14 000``) is matched
    to ga_pymoo / pso_pyswarms (~10 000) within the same ballpark.
  - ``sigma0`` (float, default 0.3): initial step size as a fraction of
    the bound range. cma's documentation recommends ``sigma0`` ≈
    20–30% of the search range; 0.3 puts the initial Gaussian sample
    spread at ``0.3 * (L - 1)``, roughly one layer-width for L=2.
  - ``seed_with_routing_patterns`` (bool, default True): if True the
    initial mean ``x0`` is seeded from a ``routing_method_1(takeaway=[3, 5])``
    pattern (the same seed DA uses); otherwise ``x0`` is the midpoint
    ``(L-1)/2`` (cma's documentation default for centred starts).
"""
from __future__ import annotations

import itertools
import time
from typing import Any

import cma
import numpy as np

from ..statistics import IterEvent
from .base import (
    BudgetCapExceeded,
    BudgetGuard,
    OptimizationResult,
    Optimizer,
    observer_copy,
    register_optimizer,
)


# cma 4.4 has many stopping criteria; the eval-budget framing E4 needs
# requires that ``maxiter`` is the SOLE termination signal at the
# experiments-harness budgets. We push each tolerance to either an
# absurdly-tight floor (so consecutive-evaluation equality is required
# to fire, which routing loss landscapes don't produce) or an absurdly-
# wide ceiling (so iteration / sigma stagnation can't fire within the
# maxiter horizon).
#
# Actual cma 4.4.4 defaults (per ``cma.CMAOptions()``):
#   ``tolfun``         = 1e-11     -> our override: 1e-12 (tighter)
#   ``tolfunhist``     = 1e-12     -> our override: 1e-12 (same)
#   ``tolfunrel``      = 0.0       -> our override: 0.0   (same)
#   ``tolflatfitness`` = 1         -> our override: 1e9
#   ``tolx``           = 1e-11     -> our override: 1e-12 (tighter)
#   ``tolstagnation``  = int(100 + 100*N**1.5/popsize)  e.g. ≈ 3929
#                                    for N=66, popsize=14 (won't fire at
#                                    shipped maxiter=1000 but could at
#                                    maxiter > ~4000)         -> 1e9
# Other tolerances (``tolxstagnation`` / ``tolfacupx=1e3`` /
# ``tolconditioncov=1e14`` / ``tolupsigma=1e20``) are wide-defaulted by
# cma and not in scope for the experiments-harness budget. M-J Stage 2
# R1 P1 (both reviewers): the previous comment incorrectly cited cma
# defaults as ``tolfun=1e-12 / tolflatfitness=10*maxiter``; the real
# defaults are 1e-11 / 1 (verified by ``cma.CMAOptions()``).
_INTRINSIC_STOP_DISABLED = {
    # Stage I (2026-05-27): tolfun / tolfunhist / tolx pushed to 0
    # instead of 1e-12. Routing-loss landscape is INTEGER-quantized
    # (loss = cl*crossings + tl*tapers, both ints), so a converged
    # CMA-ES population rounds to the same integer layer assignment
    # and produces an objective range of EXACTLY 0 across the
    # population. With tolfun=1e-12 the predicate `range < tolfun`
    # becomes `0 < 1e-12 = True` => CMA-ES stops prematurely (Stage H
    # observed n_fe ~ 4k vs the 5x-N_DA budget of 336k). Setting these
    # to 0 changes the predicate to `0 < 0 = False` => CMA-ES keeps
    # going until either maxiter / max_nfe / max_wall fires, matching
    # the budget-fair Stage H comparison framing.
    #
    # If a user genuinely wants the legacy 1e-12 behavior (e.g. for
    # non-quantized objectives) they can pass tolfun=1e-12 via
    # optimizer_kwargs.cmaes — the wrapper now plumbs this through.
    "tolfun": 0.0,
    "tolfunhist": 0.0,
    "tolfunrel": 0.0,
    "tolflatfitness": 10 ** 9,
    "tolx": 0.0,
    "tolstagnation": 10 ** 9,
    # 2026-06-06 budget-fairness reform: the equal-budget runs set
    # max_nfe = 10x N_DA and require the NFE/wall budget to be the SOLE
    # artificial stop. tolfun/tolx/tolstagnation alone were insufficient
    # — on the INTEGER-quantized routing loss CMA-ES also self-terminated
    # FAR below budget at k=12 (~4.1k nfe) and k=20 (~49.5k nfe) via the
    # step-size / covariance / x-stagnation criteria below, which fire
    # once sigma shrinks past the integer-resolution of the loss. Push
    # them all out of range so the ONLY things that can stop CMA-ES are
    # (a) the max_nfe / wall_time_s budget, or (b) cma's un-disablable
    # ``noeffectcoord`` / ``noeffectaxis`` checks — which represent
    # GENUINE numerical convergence (a local step no longer changes the
    # mean), not budget starvation, and are correct to keep. cma accepts
    # ``tolxstagnation=False`` to turn that criterion off entirely.
    "tolconditioncov": 1e99,
    "tolfacupx": 1e99,
    "tolupsigma": 1e99,
    "tolxstagnation": False,
}


@register_optimizer("cmaes")
class CMAESOptimizer(Optimizer):
    """Hansen CMA-ES on continuous [0, L-1]^E via the ``cma`` library.

    Defaults (``maxiter=1000, popsize=14, sigma0=0.3``) ⇒ ~14 000 loss
    evaluations per seed — matched to ga_pymoo / pso_pyswarms (~10 000)
    within an order of magnitude. The bound box is enforced via
    ``inopts['bounds']``; cma's bound-handling reflects out-of-box
    candidates back inside [0, L-1], so the optimizer never proposes a
    point outside the layer-assignment domain. ``apply_optimization_result``
    rounds the final mean to int layers via banker's rounding.
    """

    name = "cmaes"

    _DEFAULT_MAXITER = 1000
    _DEFAULT_POPSIZE = 14
    _DEFAULT_SIGMA0 = 0.3

    def optimize(self, maxiter: int = 1000) -> OptimizationResult:
        # t0 first so trace[].wall_ms shares clock origin with the outer
        # optimization_wall_ms PhaseTimer (§3-2).
        t0 = time.perf_counter_ns()

        n_edges = len(self.graph.G.edges)
        L = int(getattr(self.graph, "L", 2))
        if L < 2:
            raise ValueError(
                f"cmaes has no search space when L={L}; L >= 2 is required. "
                "For an L=1 baseline, evaluate "
                "graph.loss_function(np.zeros(n_edges)) directly."
            )

        # ---- knob extraction ----
        kw: dict[str, Any] = dict(self.kwargs)
        # The scipy-style ``maxiter`` API kwarg shares its name with
        # cma's ``maxiter`` option, so we DO honor it as the default
        # generations — unlike ga_pymoo / pso_pyswarms which use
        # algorithm-native iter concepts. Override via TOML ``maxiter``
        # in ``[optimizer_kwargs.cmaes]``.
        kw_maxiter = int(kw.pop("maxiter", maxiter if maxiter else self._DEFAULT_MAXITER))
        popsize = int(kw.pop("popsize", self._DEFAULT_POPSIZE))
        sigma0 = float(kw.pop("sigma0", self._DEFAULT_SIGMA0))
        seed_with_routing = bool(kw.pop("seed_with_routing_patterns", True))
        # Stage H equal-budget caps
        max_nfe = kw.pop("max_nfe", None)
        wall_time_s = kw.pop("wall_time_s", None)
        # Stage I tolerance overrides (opt-in; defaults via
        # _INTRINSIC_STOP_DISABLED). Useful when the routing loss
        # quantization causes premature termination.
        user_tols = {
            k: kw.pop(k) for k in (
                "tolfun", "tolfunhist", "tolfunrel",
                "tolflatfitness", "tolx", "tolstagnation",
            ) if k in kw
        }
        if kw:
            raise TypeError(
                f"cmaes: unknown optimizer kwargs {sorted(kw.keys())!r}. "
                "Recognised: maxiter, popsize, sigma0, "
                "seed_with_routing_patterns, max_nfe, wall_time_s, "
                "tolfun, tolfunhist, tolfunrel, tolflatfitness, tolx, "
                "tolstagnation."
            )

        # ---- initial mean x0 + sigma0 (scaled to range) ----
        xu = float(L - 1) if L > 1 else 0.0
        if seed_with_routing:
            ecl = int(getattr(self.graph, "edge_coupler_layer", 0))
            seed_layer = (ecl + 1) % L
            x0 = self._seed_from_routing(
                takeaway=[3, 5], layer=seed_layer
            ).astype(float)
        else:
            x0 = np.full(n_edges, xu / 2.0)
        # sigma0 is fraction-of-range — scale to absolute units for cma.
        sigma_abs = max(sigma0 * xu, 1e-6) if xu > 0 else 1e-6

        # Budget-fairness (2026-06-06): when max_nfe is set, EXPAND the
        # generation cap so the NFE budget (not the generation count) is
        # the binding stop — mirrors DE/GA/PSO/BO which all derive their
        # native iteration count from max_nfe (differential_evolution.py
        # scipy_maxiter, ga_pymoo.py n_gen, pso_pyswarms.py iters,
        # bo_skopt.py n_calls). cma evaluates ``popsize`` candidates per
        # generation, so ``maxiter = max_nfe // popsize + 1`` sits just
        # above the NFE budget; the BudgetGuard + ``maxfevals`` enforce
        # the exact cap. WITHOUT this, the shipped ``maxiter`` fired first
        # (5000 gen * popsize 18 = 90 000 evals << the 24.8M k=32 budget)
        # — CMA-ES was the ONLY optimizer whose equal-budget caps never
        # bound (REFACTOR_GOALS.md §1-5 takeaway #8 predates the Stage H
        # budget machinery and was never reconciled with it).
        if max_nfe is not None:
            kw_maxiter = max(kw_maxiter, int(max_nfe) // popsize + 1)

        # ---- cma options ----
        opts: dict[str, Any] = {
            "maxiter": kw_maxiter,
            "popsize": popsize,
            "bounds": [[0.0] * n_edges, [xu] * n_edges],
            "verbose": -9,  # silence the cma library's stdout chatter
            "verb_disp": 0,
            "verb_log": 0,
        }
        # Stage H: mirror max_nfe into native maxfevals safety-net cap
        if max_nfe is not None:
            opts["maxfevals"] = int(max_nfe)
        opts.update(_INTRINSIC_STOP_DISABLED)
        # User-provided tol overrides win over the wrapper's defaults
        # (so a caller can opt back into legacy 1e-12 tolfun for
        # non-quantized objectives).
        opts.update(user_tols)
        if self.seed is not None:
            # cma's docs warn ``seed`` must be ``>= 0``; the
            # experiments harness uses small non-negative ints, but
            # guard against accidental negatives with a clear error.
            seed_int = int(self.seed)
            if seed_int < 0:
                raise ValueError(
                    f"cmaes: seed must be non-negative for cma (got {seed_int})"
                )
            # cma seeds with seed=0 are accepted but documented as a
            # request for cma's internal "from clock" path; we add 1 to
            # avoid that ambiguity (preserves determinism — every input
            # seed maps to a distinct cma seed).
            opts["seed"] = seed_int + 1

        # ---- per-eval sink hookup ----
        sink = self.sink
        loss_fn = self.graph.loss_function
        counter = itertools.count()
        observer = self.eval_observer

        guard = BudgetGuard(
            max_nfe=max_nfe, wall_time_s=wall_time_s, _t0_ns=t0
        )

        def emit_one(x: np.ndarray, loss: float) -> None:
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

        # ---- numpy global RNG isolation (M-J Stage 2 R1 P1, Codex) ----
        # cma 4.4.4 ``CMAEvolutionStrategy.__init__`` calls
        # ``np.random.seed(opts['seed'])`` (see cma's
        # ``evolution_strategy.py``), clobbering the caller's numpy
        # global RNG state. We snapshot pre-call, run cma, then restore
        # in ``finally`` — mirrors ``pso_pyswarms``'s isolation pattern.
        _saved_state = np.random.get_state()
        es = None
        raw = None
        try:
            # ---- ask/tell loop ----
            es = cma.CMAEvolutionStrategy(x0, sigma_abs, opts)
            try:
                while not es.stop():
                    solutions = es.ask()
                    # Defensive copy: cma's ``ask()`` returns ndarrays
                    # that share storage with strategy state (see §1-1-a
                    # D1 defense-in-depth comment).
                    eval_copies = [np.asarray(x).copy() for x in solutions]
                    fitnesses: list[float] = []
                    for xa in eval_copies:
                        loss = float(loss_fn(xa))
                        emit_one(xa, loss)  # may raise BudgetCapExceeded
                        fitnesses.append(loss)
                    es.tell(solutions, fitnesses)

                best = es.best
                if best is None or best.x is None:
                    raise RuntimeError(
                        "cmaes: es.best is None — cma did not record a best "
                        "candidate. Inspect maxiter, popsize, and bounds; "
                        "do NOT silently fall back (§1-1-a)."
                    )
                best_x = np.asarray(best.x).reshape(-1)
                best_loss = float(best.f)
                raw = es
            except BudgetCapExceeded as exc:
                # Cap fired mid ask/tell: fall back to guard's snapshot.
                best_layers_cap = guard.best_layers
                best_loss = guard.best_loss
                # Also stash for later
                raw = {"cap_reason": str(exc), "nfe": guard.nfe}
                best_x = None  # sentinel — use best_layers_cap below
        finally:
            np.random.set_state(_saved_state)

        if best_x is None:
            best_layers = best_layers_cap
        else:
            best_layers = [int(round(v)) for v in best_x]
        self.graph.apply_optimization_result(best_layers)
        print(f"Optimal average loss: {best_loss}")
        return OptimizationResult(
            best_layers=best_layers, fun=best_loss, raw=raw
        )
