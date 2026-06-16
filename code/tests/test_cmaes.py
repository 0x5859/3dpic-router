"""Unit tests for the ``cma``-backed CMA-ES optimizer (REFACTOR_GOALS.md
M-J Stage 2).

Mirror of ``test_ga_pymoo.py`` structure: registry + smoke + §1-1-a
sink/observer contract + determinism + L<2 reject + unknown-kwargs
TypeError + x-mutating-observer safety + `api.run_optimization`
end-to-end with trace-length assertion. Plus a CMA-ES-specific
non-negative-seed check (cma's `seed` option rejects negatives).
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


def test_cmaes_registers_in_registry() -> None:
    from routing_py_rebuild.optimizers import CMAESOptimizer, get_optimizer

    cls = get_optimizer("cmaes")
    assert cls is CMAESOptimizer
    assert cls.name == "cmaes"


def test_cmaes_runs_and_returns_finite_loss() -> None:
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("cmaes")(g, seed=5859, maxiter=3, popsize=6)
    res = opt.optimize(maxiter=3)
    assert isinstance(res.best_layers, list)
    assert len(res.best_layers) == len(g.G.edges)
    assert all(isinstance(v, int) for v in res.best_layers)
    L = int(g.L)
    assert all(0 <= v <= L - 1 for v in res.best_layers)
    assert np.isfinite(res.fun)


def test_cmaes_invokes_observer_once_per_eval() -> None:
    """§1-1-a contract: maxiter=3, popsize=6 → exactly 18 evals."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    g = _tiny_graph()
    g.build_crossings_index()
    rec = RunRecorder(run_id="t_cma", config={})
    seen: list[float] = []
    opt = get_optimizer("cmaes")(
        g, seed=5859, sink=rec,
        eval_observer=lambda x, loss, _s=seen: _s.append(float(loss)),
        maxiter=3, popsize=6,
    )
    opt.optimize(maxiter=3)
    rep = rec.finalize()
    trace_losses = [t["loss"] for t in rep["trace"]]
    finite_seen = [v for v in seen if np.isfinite(v)]
    assert len(trace_losses) == 18, len(trace_losses)
    assert len(seen) >= len(trace_losses)
    assert finite_seen == trace_losses


def test_cmaes_deterministic_with_seed() -> None:
    from routing_py_rebuild.optimizers import get_optimizer

    def _run() -> tuple[float, list[int]]:
        g = _tiny_graph()
        g.build_crossings_index()
        opt = get_optimizer("cmaes")(g, seed=5859, maxiter=3, popsize=6)
        r = opt.optimize(maxiter=3)
        return float(r.fun), list(r.best_layers)

    a_loss, a_layers = _run()
    b_loss, b_layers = _run()
    assert a_loss == b_loss
    assert a_layers == b_layers


def test_cmaes_does_not_leak_numpy_global_rng() -> None:
    """cma 4.4 calls ``np.random.seed(opts['seed'])`` internally on
    every ``CMAEvolutionStrategy`` construction; without isolation,
    every CMA-ES ``optimize()`` would clobber the caller's global RNG.
    The wrapper snapshots + restores ``np.random.get_state()`` in
    ``try`` / ``finally`` — verify the caller sees their pre-call
    state intact after the optimizer runs.

    Mirror of ``test_pso_pyswarms_does_not_leak_numpy_global_rng``;
    M-J Stage 2 R1 P1 (Codex) caught this asymmetric leak before fix.
    """
    from routing_py_rebuild.optimizers import get_optimizer

    np.random.seed(12345)
    before = np.random.random()
    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("cmaes")(
        g, seed=99, maxiter=2, popsize=4,
    )
    opt.optimize(maxiter=2)
    after = np.random.random()

    # Control: same starting state, no cmaes in between.
    np.random.seed(12345)
    ctrl_before = np.random.random()
    ctrl_after = np.random.random()
    assert before == ctrl_before
    assert after == ctrl_after, (
        "cmaes perturbed the caller's numpy global RNG — "
        "snapshot/restore must isolate the optimizer."
    )


def test_cmaes_rejects_l1() -> None:
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    g.L = 1
    opt = get_optimizer("cmaes")(g, seed=5859, maxiter=2, popsize=4)
    with pytest.raises(ValueError, match="L=1"):
        opt.optimize(maxiter=1)


def test_cmaes_rejects_negative_seed() -> None:
    """cma's ``seed`` option requires ``>= 0`` — we surface a clear
    error before constructing the strategy."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("cmaes")(g, seed=-1, maxiter=2, popsize=4)
    with pytest.raises(ValueError, match="non-negative"):
        opt.optimize(maxiter=1)


def test_cmaes_unknown_kwargs_raise_typeerror() -> None:
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("cmaes")(
        g, seed=5859, maxiter=2, popsize=4, n_gen=2,  # typo!
    )
    with pytest.raises(TypeError, match="unknown optimizer kwargs"):
        opt.optimize(maxiter=1)


def test_cmaes_x_mutating_observer_cannot_perturb_optimizer() -> None:
    """§1-1-a defense-in-depth: observer mutating ``x`` must not
    perturb CMA-ES — ``ask()`` returns fresh ndarrays each call."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    def _trace(obs):
        g = _tiny_graph()
        g.build_crossings_index()
        rec = RunRecorder(run_id="t_cma_mut", config={})
        opt = get_optimizer("cmaes")(
            g, seed=5859, sink=rec, eval_observer=obs,
            maxiter=3, popsize=6,
        )
        r = opt.optimize(maxiter=3)
        rep = rec.finalize()
        return [t["loss"] for t in rep["trace"]], list(r.best_layers)

    def _mutating(x, _loss):
        arr = np.asarray(x)
        if arr.size:
            arr[0] = 1.0 - float(arr[0])

    losses_a, layers_a = _trace(None)
    losses_b, layers_b = _trace(_mutating)
    assert losses_a == losses_b, "x-mutating observer perturbed CMA-ES"
    assert layers_a == layers_b


def test_cmaes_seed_with_routing_patterns_toggle() -> None:
    """``seed_with_routing_patterns=False`` ⇒ ``x0`` is the midpoint
    ``(L-1)/2`` instead of a routing pattern."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("cmaes")(
        g, seed=5859, maxiter=3, popsize=6,
        seed_with_routing_patterns=False,
    )
    res = opt.optimize(maxiter=3)
    assert np.isfinite(res.fun)
    assert len(res.best_layers) == len(g.G.edges)


def test_run_optimization_routes_kwargs_to_cmaes() -> None:
    """End-to-end: trace-length lock for kwargs routing."""
    import json
    import tempfile
    from pathlib import Path

    from routing_py_rebuild.api import run_optimization

    maxiter, popsize = 3, 6
    with tempfile.TemporaryDirectory() as td:
        res = run_optimization(
            k=8,
            optimizer="cmaes",
            maxiter=maxiter,
            seed=5859,
            optimizer_kwargs={"popsize": popsize},
            output_dir=str(Path(td)) + "/",
            plot=False,
            run_loss_analysis=False,
            collect_statistics=True,
        )
        assert np.isfinite(res["loss"])
        with open(res["report_path"]) as fh:
            rep = json.load(fh)
        assert len(rep["trace"]) == maxiter * popsize, (
            f"trace length {len(rep['trace'])} != maxiter * popsize = "
            f"{maxiter * popsize}; optimizer_kwargs may have been dropped"
        )
