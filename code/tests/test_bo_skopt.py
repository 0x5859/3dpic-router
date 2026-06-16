"""Unit tests for the scikit-optimize-backed Bayesian-Optimization
optimizer (REFACTOR_GOALS.md M-J Stage 2).

Mirror of ``test_ga_pymoo.py`` structure: registry + smoke + §1-1-a
sink/observer contract + determinism + L<2 reject + unknown-kwargs
TypeError + x-mutating-observer safety + `api.run_optimization`
end-to-end with trace-length assertion. Plus a BO-specific
`n_initial_points > n_calls` ValueError guard.

BO is the honest-baseline E4 entry — n_calls is intentionally small
(60-120) for the paper-reviewer-response framing; unit tests use even
smaller budgets (n_calls=12) for speed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _tiny_graph(k: int = 8):
    from routing_py_rebuild.core import SiNInterconnectionGraph
    from routing_py_rebuild.positions import distribute_nodes_around_square

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos = distribute_nodes_around_square(k // 4, side_length=1)
        return SiNInterconnectionGraph(k=k, positions=pos)


def test_bo_skopt_registers_in_registry() -> None:
    from routing_py_rebuild.optimizers import BOSkoptOptimizer, get_optimizer

    cls = get_optimizer("bo_skopt")
    assert cls is BOSkoptOptimizer
    assert cls.name == "bo_skopt"


def test_bo_skopt_runs_and_returns_finite_loss() -> None:
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("bo_skopt")(
        g, seed=5859, n_calls=12, n_initial_points=5,
    )
    res = opt.optimize(maxiter=9999)
    assert isinstance(res.best_layers, list)
    assert len(res.best_layers) == len(g.G.edges)
    L = int(g.L)
    assert all(0 <= v <= L - 1 for v in res.best_layers)
    assert np.isfinite(res.fun)


def test_bo_skopt_invokes_observer_once_per_eval() -> None:
    """§1-1-a contract: trace length == n_calls exactly."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    g = _tiny_graph()
    g.build_crossings_index()
    rec = RunRecorder(run_id="t_bo", config={})
    seen: list[float] = []
    opt = get_optimizer("bo_skopt")(
        g, seed=5859, sink=rec,
        eval_observer=lambda x, loss, _s=seen: _s.append(float(loss)),
        n_calls=12, n_initial_points=5,
    )
    opt.optimize(maxiter=9999)
    rep = rec.finalize()
    trace_losses = [t["loss"] for t in rep["trace"]]
    finite_seen = [v for v in seen if np.isfinite(v)]
    assert len(trace_losses) == 12, len(trace_losses)
    assert len(seen) >= len(trace_losses)
    assert finite_seen == trace_losses


def test_bo_skopt_deterministic_with_seed() -> None:
    from routing_py_rebuild.optimizers import get_optimizer

    def _run() -> tuple[float, list[int]]:
        g = _tiny_graph()
        g.build_crossings_index()
        opt = get_optimizer("bo_skopt")(
            g, seed=5859, n_calls=12, n_initial_points=5,
        )
        r = opt.optimize(maxiter=9999)
        return float(r.fun), list(r.best_layers)

    a_loss, a_layers = _run()
    b_loss, b_layers = _run()
    assert a_loss == b_loss
    assert a_layers == b_layers


def test_bo_skopt_rejects_l1() -> None:
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    g.L = 1
    opt = get_optimizer("bo_skopt")(g, seed=5859, n_calls=4, n_initial_points=2)
    with pytest.raises(ValueError, match="L=1"):
        opt.optimize(maxiter=1)


def test_bo_skopt_rejects_initial_points_above_or_equal_to_n_calls() -> None:
    """``n_initial_points >= n_calls`` consumes the entire budget on
    random sampling with NO GP-driven calls (degenerate to LHS sweep)
    — fail-fast with a clear error instead of silently degrading BO
    to LHS. M-J Stage 2 R1 P1 (Opus): both strict-greater AND equality
    cases must trip the guard."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    for n_calls, n_init in [(5, 10), (5, 5)]:  # > and ==
        opt = get_optimizer("bo_skopt")(
            g, seed=5859, n_calls=n_calls, n_initial_points=n_init,
        )
        with pytest.raises(ValueError, match="n_initial_points"):
            opt.optimize(maxiter=1)


def test_bo_skopt_unknown_kwargs_raise_typeerror() -> None:
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("bo_skopt")(
        g, seed=5859, n_calls=6, n_initial_points=3, n_call=6,  # typo!
    )
    with pytest.raises(TypeError, match="unknown optimizer kwargs"):
        opt.optimize(maxiter=1)


def test_bo_skopt_x_mutating_observer_cannot_perturb_optimizer() -> None:
    """§1-1-a defense-in-depth: observer mutating ``x`` must not
    perturb skopt — ``scalar_loss`` casts the incoming list via
    ``np.asarray(..., dtype=float)`` which always returns a fresh
    ndarray (a list-to-ndarray conversion copies)."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    def _trace(obs):
        g = _tiny_graph()
        g.build_crossings_index()
        rec = RunRecorder(run_id="t_bo_mut", config={})
        opt = get_optimizer("bo_skopt")(
            g, seed=5859, sink=rec, eval_observer=obs,
            n_calls=12, n_initial_points=5,
        )
        r = opt.optimize(maxiter=9999)
        rep = rec.finalize()
        return [t["loss"] for t in rep["trace"]], list(r.best_layers)

    def _mutating(x, _loss):
        arr = np.asarray(x)
        if arr.size:
            arr[0] = 1.0 - float(arr[0])

    losses_a, layers_a = _trace(None)
    losses_b, layers_b = _trace(_mutating)
    assert losses_a == losses_b, "x-mutating observer perturbed BO"
    assert layers_a == layers_b


def test_run_optimization_routes_kwargs_to_bo() -> None:
    """End-to-end: trace-length lock for kwargs routing."""
    import json
    import tempfile
    from pathlib import Path

    from routing_py_rebuild.api import run_optimization

    n_calls, n_initial = 10, 4
    with tempfile.TemporaryDirectory() as td:
        res = run_optimization(
            k=8,
            optimizer="bo_skopt",
            maxiter=1000,  # ignored
            seed=5859,
            optimizer_kwargs={
                "n_calls": n_calls,
                "n_initial_points": n_initial,
            },
            output_dir=str(Path(td)) + "/",
            plot=False,
            run_loss_analysis=False,
            collect_statistics=True,
        )
        assert np.isfinite(res["loss"])
        with open(res["report_path"]) as fh:
            rep = json.load(fh)
        assert len(rep["trace"]) == n_calls, (
            f"trace length {len(rep['trace'])} != n_calls = {n_calls}; "
            "optimizer_kwargs may have been dropped"
        )
