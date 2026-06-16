"""Unit tests for the pyswarms-backed PSO optimizer (REFACTOR_GOALS.md
M-J Stage 2).

Mirror of ``test_ga_pymoo.py`` structure: registry + smoke + §1-1-a
sink/observer contract + determinism + L<2 reject + unknown-kwargs
TypeError + x-mutating-observer safety + `api.run_optimization`
end-to-end with trace-length assertion. Plus a PSO-specific RNG
non-perturbation check (pyswarms reads numpy global RNG; the optimizer
must snapshot/restore so callers' global state survives unchanged).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _tiny_graph(k: int = 8):
    """Mirror of ``test_ga_pymoo._tiny_graph`` so PSO sits on the same
    K_8 fixture as the other M-J optimizer tests."""
    from routing_py_rebuild.core import SiNInterconnectionGraph
    from routing_py_rebuild.positions import distribute_nodes_around_square

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos = distribute_nodes_around_square(k // 4, side_length=1)
        return SiNInterconnectionGraph(k=k, positions=pos)


def test_pso_pyswarms_registers_in_registry() -> None:
    from routing_py_rebuild.optimizers import PSOPyswarmsOptimizer, get_optimizer

    cls = get_optimizer("pso_pyswarms")
    assert cls is PSOPyswarmsOptimizer
    assert cls.name == "pso_pyswarms"


def test_pso_pyswarms_runs_and_returns_finite_loss() -> None:
    """End-to-end: PSO runs, returns finite loss + per-edge layers in
    {0, ..., L-1}."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("pso_pyswarms")(
        g, seed=5859, iters=3, n_particles=6,
    )
    res = opt.optimize(maxiter=9999)  # ignored
    assert isinstance(res.best_layers, list)
    assert len(res.best_layers) == len(g.G.edges)
    assert all(isinstance(v, int) for v in res.best_layers)
    L = int(g.L)
    assert all(0 <= v <= L - 1 for v in res.best_layers)
    assert np.isfinite(res.fun)


def test_pso_pyswarms_invokes_observer_once_per_eval() -> None:
    """§1-1-a contract: one eval = one IterEvent + one eval_observer
    call. With ``iters=3, n_particles=6``, expected ``trace`` length is
    exactly ``18``."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    g = _tiny_graph()
    g.build_crossings_index()
    rec = RunRecorder(run_id="t_pso", config={})
    seen: list[float] = []
    opt = get_optimizer("pso_pyswarms")(
        g, seed=5859, sink=rec,
        eval_observer=lambda x, loss, _s=seen: _s.append(float(loss)),
        iters=3, n_particles=6,
    )
    opt.optimize(maxiter=9999)
    rep = rec.finalize()
    trace_losses = [t["loss"] for t in rep["trace"]]
    finite_seen = [v for v in seen if np.isfinite(v)]
    assert len(trace_losses) == 18, len(trace_losses)
    assert len(seen) >= len(trace_losses)
    assert finite_seen == trace_losses


def test_pso_pyswarms_deterministic_with_seed() -> None:
    """Same seed + same kwargs ⇒ same best_loss + best_layers."""
    from routing_py_rebuild.optimizers import get_optimizer

    def _run() -> tuple[float, list[int]]:
        g = _tiny_graph()
        g.build_crossings_index()
        opt = get_optimizer("pso_pyswarms")(
            g, seed=5859, iters=3, n_particles=6,
        )
        r = opt.optimize(maxiter=9999)
        return float(r.fun), list(r.best_layers)

    a_loss, a_layers = _run()
    b_loss, b_layers = _run()
    assert a_loss == b_loss
    assert a_layers == b_layers


def test_pso_pyswarms_does_not_leak_numpy_global_rng() -> None:
    """pyswarms reads from ``numpy.random`` directly; PSO must snapshot
    + restore the global state so calling it does not perturb the
    caller's RNG.

    Procedure: sample one float from the global RNG BEFORE PSO runs,
    run PSO, sample again, then run a "control" sequence that just
    does two samples back-to-back. The two-sample sequence post-PSO
    must equal the control sequence."""
    from routing_py_rebuild.optimizers import get_optimizer

    np.random.seed(12345)
    before = np.random.random()
    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("pso_pyswarms")(
        g, seed=99, iters=2, n_particles=4,
    )
    opt.optimize(maxiter=9999)
    after = np.random.random()

    # Control: same starting state, no PSO in between.
    np.random.seed(12345)
    ctrl_before = np.random.random()
    ctrl_after = np.random.random()
    assert before == ctrl_before
    assert after == ctrl_after, (
        "pso_pyswarms perturbed the caller's numpy global RNG — "
        "snapshot/restore must isolate the optimizer."
    )


def test_pso_pyswarms_rejects_l1() -> None:
    """L=1 has no search space — match the DA/DE/GA early-raise."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    g.L = 1
    opt = get_optimizer("pso_pyswarms")(g, seed=5859, iters=2, n_particles=4)
    with pytest.raises(ValueError, match="L=1"):
        opt.optimize(maxiter=1)


def test_pso_pyswarms_unknown_kwargs_raise_typeerror() -> None:
    """Typo in TOML must surface loudly, not silently no-op."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("pso_pyswarms")(
        g, seed=5859, iters=2, n_particles=4, n_particle=5,  # typo!
    )
    with pytest.raises(TypeError, match="unknown optimizer kwargs"):
        opt.optimize(maxiter=1)


def test_pso_pyswarms_x_mutating_observer_cannot_perturb_optimizer() -> None:
    """§1-1-a defense-in-depth: observer mutating its ``x`` argument
    must not perturb the swarm. pyswarms holds its own (n_particles,
    n_var) ndarray; our ``batch_loss`` indexes a row out of it, so
    mutating the row would mutate the swarm — verify pyswarms' indexing
    returns a view we treat as read-only (no mutation by observer)."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    def _trace(obs):
        g = _tiny_graph()
        g.build_crossings_index()
        rec = RunRecorder(run_id="t_pso_mut", config={})
        opt = get_optimizer("pso_pyswarms")(
            g, seed=5859, sink=rec, eval_observer=obs,
            iters=3, n_particles=6,
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
    assert losses_a == losses_b, "x-mutating observer perturbed the swarm"
    assert layers_a == layers_b


def test_pso_pyswarms_seed_with_routing_patterns_toggle() -> None:
    """``seed_with_routing_patterns=False`` skips routing seeds — pure
    random init swarm. Run must still complete + finite loss."""
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    opt = get_optimizer("pso_pyswarms")(
        g, seed=5859, iters=3, n_particles=6,
        seed_with_routing_patterns=False,
    )
    res = opt.optimize(maxiter=9999)
    assert np.isfinite(res.fun)
    assert len(res.best_layers) == len(g.G.edges)


def test_run_optimization_routes_kwargs_to_pso() -> None:
    """End-to-end: ``api.run_optimization(optimizer='pso_pyswarms',
    optimizer_kwargs={'iters': 2, 'n_particles': 4})`` forwards the
    kwargs all the way to the PSO constructor.

    Trace-length assertion is the load-bearing kwargs-routing lock —
    silently dropping kwargs would default to ``iters=200,
    n_particles=50 = 10 000 evals`` and still produce a finite loss."""
    import json
    import tempfile
    from pathlib import Path

    from routing_py_rebuild.api import run_optimization

    iters, n_particles = 2, 4
    with tempfile.TemporaryDirectory() as td:
        res = run_optimization(
            k=8,
            optimizer="pso_pyswarms",
            maxiter=1000,  # ignored
            seed=5859,
            optimizer_kwargs={"iters": iters, "n_particles": n_particles},
            output_dir=str(Path(td)) + "/",
            plot=False,
            run_loss_analysis=False,
            collect_statistics=True,
        )
        assert np.isfinite(res["loss"])
        assert "report_path" in res and res["report_path"]
        with open(res["report_path"]) as fh:
            rep = json.load(fh)
        assert len(rep["trace"]) == iters * n_particles, (
            f"trace length {len(rep['trace'])} != iters * n_particles = "
            f"{iters * n_particles}; optimizer_kwargs may have been dropped"
        )
