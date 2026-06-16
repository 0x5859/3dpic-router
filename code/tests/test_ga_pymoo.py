"""Unit tests for the pymoo-backed Genetic Algorithm optimizer
(REFACTOR_GOALS.md M-J Stage 1).

Scope mirrors the scipy-optimizer smoke tests in ``test_statistics.py``:
  - registry: ``ga_pymoo`` looks up the right class
  - per-evaluation sink + ``eval_observer`` contract (§1-1-a)
  - deterministic for fixed seed
  - L<2 rejection (matches DA / DE behaviour)
  - unknown kwargs surface as a clear ``TypeError``
  - x-mutating observer cannot perturb the optimizer (D1 defense-in-depth)

These tests run on a tiny K_8 graph so a single full GA run completes
in well under a second.
"""
from __future__ import annotations

import numpy as np
import pytest


def _tiny_graph(k: int = 8):
    """K_k square graph with the M-A+ standard test pattern. Mirrors
    ``test_statistics._tiny_graph`` so the GA smoke tests sit on the
    same shape as the scipy-optimizer tests."""
    from routing_py_rebuild.core import SiNInterconnectionGraph
    from routing_py_rebuild.positions import distribute_nodes_around_square

    pos = distribute_nodes_around_square(k // 4, side_length=1)
    return SiNInterconnectionGraph(k=k, positions=pos)


def test_ga_pymoo_registers_in_registry() -> None:
    """The decorator-driven registration in ``optimizers/ga_pymoo.py``
    must be picked up by the package ``__init__`` import side-effect."""
    from routing_py_rebuild.optimizers import GAPymooOptimizer, get_optimizer

    cls = get_optimizer("ga_pymoo")
    assert cls is GAPymooOptimizer
    assert cls.name == "ga_pymoo"


def test_ga_pymoo_runs_and_returns_finite_loss() -> None:
    """End-to-end: GA constructs, runs, and produces a finite loss + a
    per-edge integer layer assignment in {0, ..., L-1}."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("ga_pymoo")(
        g, seed=5859, n_gen=3, pop_size=8,
    )
    res = opt.optimize(maxiter=1000)  # maxiter unused when n_gen explicit
    assert isinstance(res.best_layers, list)
    assert len(res.best_layers) == len(g.G.edges)
    assert all(isinstance(v, int) for v in res.best_layers)
    L = int(g.L)
    assert all(0 <= v <= L - 1 for v in res.best_layers)
    assert np.isfinite(res.fun)


def test_ga_pymoo_invokes_observer_once_per_eval() -> None:
    """§1-1-a contract: every loss-function evaluation lands one
    ``eval_observer`` call. Matches the assertion pattern in
    ``test_statistics.test_observer_call_count_equals_eval_count_all_optimizers``
    (extends that style to ``ga_pymoo``)."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    g = _tiny_graph()
    g.build_crossings_index()
    rec = RunRecorder(run_id="t_ga", config={})
    seen: list[float] = []
    opt = get_optimizer("ga_pymoo")(
        g, seed=5859, sink=rec,
        eval_observer=lambda x, loss, _s=seen: _s.append(float(loss)),
        n_gen=3, pop_size=8,
    )
    opt.optimize(maxiter=1000)
    rep = rec.finalize()
    trace_losses = [t["loss"] for t in rep["trace"]]
    finite_seen = [v for v in seen if np.isfinite(v)]
    # observer fires on EVERY eval (incl. non-finite); RunRecorder drops
    # non-finite from trace[]. So `seen` (all) >= trace, and the FINITE
    # observer losses must equal trace[].loss EXACTLY.
    assert len(seen) >= len(trace_losses) > 0
    assert finite_seen == trace_losses


def test_ga_pymoo_deterministic_with_seed() -> None:
    """Same seed + same kwargs ⇒ identical best_loss + identical
    best_layers across two independent runs (no global RNG leak)."""
    from routing_py_rebuild.optimizers import get_optimizer

    def _run() -> tuple[float, list[int]]:
        g = _tiny_graph()
        g.build_crossings_index()
        opt = get_optimizer("ga_pymoo")(
            g, seed=5859, n_gen=3, pop_size=8,
        )
        r = opt.optimize(maxiter=1000)
        return float(r.fun), list(r.best_layers)

    a_loss, a_layers = _run()
    b_loss, b_layers = _run()
    assert a_loss == b_loss
    assert a_layers == b_layers


def test_ga_pymoo_rejects_l1() -> None:
    """L=1 has no search space (a single layer ⇒ no routing decision).
    Match the scipy optimizers' early-raise behaviour (§2-3 三轮评审
    P1-C mirror) so the failure message points at the optimizer rather
    than crashing inside pymoo."""
    from routing_py_rebuild.optimizers import get_optimizer

    # Build a graph then force L=1 by setting num_layers post-hoc.
    g = _tiny_graph()
    g.build_crossings_index()
    g.L = 1
    opt = get_optimizer("ga_pymoo")(g, seed=5859, n_gen=2, pop_size=4)
    with pytest.raises(ValueError, match="L=1"):
        opt.optimize(maxiter=1)


def test_ga_pymoo_unknown_kwargs_raise_typeerror() -> None:
    """A typo in the experiments harness TOML (e.g. ``[optimizer_kwargs.ga_pymoo]``
    with a misspelled key like ``n_generations`` instead of ``n_gen``)
    must surface loudly, not silently no-op. The unknown key reaches
    ``self.kwargs``; ``optimize()`` raises with a helpful message."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("ga_pymoo")(
        g, seed=5859, n_gen=2, pop_size=4, n_generations=2,  # typo!
    )
    with pytest.raises(TypeError, match="unknown optimizer kwargs"):
        opt.optimize(maxiter=1)


def test_ga_pymoo_x_mutating_observer_cannot_perturb_optimizer() -> None:
    """§1-1-a defense-in-depth: an x-mutating observer must not change
    the optimizer trajectory. Same contract as the scipy optimizers —
    pymoo's ``ElementwiseProblem._evaluate`` computes loss BEFORE calling
    the observer, and the GA holds its own per-individual vector."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    def _trace(obs):
        g = _tiny_graph()
        g.build_crossings_index()
        rec = RunRecorder(run_id="t_ga_mut", config={})
        opt = get_optimizer("ga_pymoo")(
            g, seed=5859, sink=rec, eval_observer=obs,
            n_gen=3, pop_size=8,
        )
        r = opt.optimize(maxiter=1000)
        rep = rec.finalize()
        return [t["loss"] for t in rep["trace"]], list(r.best_layers)

    def _mutating(x, _loss):
        arr = np.asarray(x)
        if arr.size:
            arr[0] = 1.0 - float(arr[0])  # corrupt attempt

    losses_a, layers_a = _trace(None)
    losses_b, layers_b = _trace(_mutating)
    assert losses_a == losses_b, "x-mutating observer perturbed the optimizer"
    assert layers_a == layers_b


def test_ga_pymoo_seed_with_routing_patterns_toggle() -> None:
    """``seed_with_routing_patterns=False`` skips the routing-seed
    injection — the GA starts from pure random initial population. The
    run must still complete and produce a finite loss (smoke only;
    different layout from the seeded variant is expected and intended)."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("ga_pymoo")(
        g, seed=5859, n_gen=3, pop_size=8,
        seed_with_routing_patterns=False,
    )
    res = opt.optimize(maxiter=1000)
    assert np.isfinite(res.fun)
    assert len(res.best_layers) == len(g.G.edges)


def test_run_optimization_routes_kwargs_to_ga() -> None:
    """End-to-end: ``api.run_optimization(optimizer='ga_pymoo',
    optimizer_kwargs={'n_gen': 2, 'pop_size': 4})`` must forward the
    kwargs all the way to the GA constructor and write a normal
    run_report.json. This is the contract the experiments harness
    PythonBackend depends on (REFACTOR_GOALS.md M-J).

    M-J Stage 1 R1 P1 (both reviewers): the assertion is the
    trace-length empirical formula ``len(trace) == n_gen * pop_size``.
    Without this, a regression that silently drops ``optimizer_kwargs``
    would still pass (default ``n_gen=200 * pop_size=50 = 10 000`` evals
    still yields a finite loss). The empirical formula was verified by
    direct measurement at multiple ``(n_gen, pop_size)`` settings —
    pymoo's GA evaluates the initial population AS generation 1, so
    total evals = ``n_gen * pop_size`` exactly."""
    import json
    import tempfile
    from pathlib import Path

    from routing_py_rebuild.api import run_optimization

    n_gen, pop_size = 2, 4
    with tempfile.TemporaryDirectory() as td:
        res = run_optimization(
            k=8,
            optimizer="ga_pymoo",
            maxiter=1000,  # ignored by ga_pymoo (see optimize() docstring)
            seed=5859,
            optimizer_kwargs={"n_gen": n_gen, "pop_size": pop_size},
            output_dir=str(Path(td)) + "/",
            plot=False,
            run_loss_analysis=False,
            collect_statistics=True,
        )
        assert np.isfinite(res["loss"])
        assert "report_path" in res and res["report_path"]
        # Trace length locks kwargs routing: if optimizer_kwargs were
        # silently dropped, len(trace) would be ~10 000 (defaults), not 8.
        with open(res["report_path"]) as fh:
            rep = json.load(fh)
        assert len(rep["trace"]) == n_gen * pop_size, (
            f"trace length {len(rep['trace'])} != n_gen * pop_size = "
            f"{n_gen * pop_size}; optimizer_kwargs may have been dropped"
        )
