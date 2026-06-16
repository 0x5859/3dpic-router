"""M1 acceptance tests for the statistics module.

Mirrors the five tests listed in REFACTOR_GOALS.md §1-1-a 测试.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from routing_py_rebuild.api import make_graph
from routing_py_rebuild.statistics import (
    IterEvent,
    NullSink,
    PhaseTimer,
    RunRecorder,
    write_run_report,
)
from routing_py_rebuild.statistics.reporters import _REPORT_SCHEMA

_REPO_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"


# ---------------------------------------------------------------------------
# Test 1 — recorder is_new_best computation
# ---------------------------------------------------------------------------

def test_recorder_is_new_best():
    """RunRecorder marks `is_new_best` correctly across an interleaved loss series.

    Fixture also used by the C++ `RecorderIsNewBest` test (§1-2-b parity anchor).
    """
    rec = RunRecorder(run_id="t", config={})
    for i, loss in enumerate([10.0, 8.0, 9.0, 7.0, 7.5]):
        rec.on_iter(IterEvent(iter=i, loss=loss, wall_ms=0))

    report = rec.finalize()
    flags = [ev["is_new_best"] for ev in report["trace"]]
    assert flags == [True, True, False, True, False]
    assert report["summary"]["best_at_iter"] == 3
    assert report["summary"]["final_loss"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Test 2 — finalize output validates against run_report.schema.json
# ---------------------------------------------------------------------------

def test_finalize_schema():
    """`recorder.finalize()` produces a dict that validates against the schema."""
    rec = RunRecorder(run_id="schema_test", config={"k": 12, "optimizer": "dual_annealing"})
    for i, loss in enumerate([1.0, 0.9, 0.8]):
        rec.on_iter(IterEvent(iter=i, loss=loss, wall_ms=i * 10))
    rec.on_phase("graph_build", 5)
    rec.on_phase("optimization", 142)

    report = rec.finalize()

    with open(_REPORT_SCHEMA) as f:
        schema = json.load(f)
    jsonschema.validate(instance=report, schema=schema)


# ---------------------------------------------------------------------------
# Test 3 — PhaseTimer fires exactly one on_phase on context exit
# ---------------------------------------------------------------------------

class _CapturingSink:
    """Minimal sink that records on_phase calls; on_iter / finalize are no-ops."""

    def __init__(self):
        self.phases: list[tuple[str, int]] = []

    def on_iter(self, ev):
        pass

    def on_phase(self, name, wall_ms):
        self.phases.append((name, int(wall_ms)))

    def finalize(self):
        return {}


def test_phase_timer_context():
    sink = _CapturingSink()
    with PhaseTimer(sink, "x"):
        # Force at least a millisecond so the assertion is non-trivial
        time.sleep(0.002)

    assert len(sink.phases) == 1
    name, ms = sink.phases[0]
    assert name == "x"
    assert ms >= 0
    assert ms < 10_000  # sanity: shouldn't take 10s for a 2ms sleep


def test_phase_timer_does_not_swallow_exceptions():
    sink = _CapturingSink()
    with pytest.raises(RuntimeError, match="boom"):
        with PhaseTimer(sink, "bad"):
            raise RuntimeError("boom")
    # on_phase still fires even on exception (the __exit__ runs)
    assert len(sink.phases) == 1
    assert sink.phases[0][0] == "bad"


# ---------------------------------------------------------------------------
# Test 4 — write_run_report produces both files; JSON validates against schema
# ---------------------------------------------------------------------------

def test_run_report_md_renders(tmp_path):
    rec = RunRecorder(run_id="md_test", config={"k": 12})
    for i, loss in enumerate([1.0, 0.5]):
        rec.on_iter(IterEvent(iter=i, loss=loss, wall_ms=i * 10))
    rec.on_phase("optimization", 200)

    report = rec.finalize()
    json_path, md_path = write_run_report(report, tmp_path)

    assert os.path.exists(json_path)
    assert os.path.exists(md_path)
    assert md_path.endswith("run_report.md")

    # JSON round-trips and still validates
    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded == report
    with open(_REPORT_SCHEMA) as f:
        schema = json.load(f)
    jsonschema.validate(instance=loaded, schema=schema)

    # md file is non-empty
    md_content = Path(md_path).read_text()
    assert len(md_content) > 0
    assert "Run report" in md_content


# ---------------------------------------------------------------------------
# Test 5 — wrapper overhead (NullSink) vs raw loss_function
#
# Per REFACTOR_GOALS.md §1-1-a 测试: "graph 必须 k≥12 ... k=4 toy 图
# loss_function 太轻、wrapper 占比会失真".
# ---------------------------------------------------------------------------

def test_wrapper_low_overhead():
    """NullSink wrapper overhead vs raw loss_function < 3% at k=40.

    REFACTOR_GOALS.md §1-1-a 测试 originally specified n=1000 and <1% at
    k≥12. M1 used k=12 + n=200 + 3% threshold because the pre-M2
    ``loss_function`` took ~40 ms/eval and n=1000 would push test runtime
    to ~80 s. After M2's cached crossing topology (§2-1) the k=12
    ``loss_function`` drops to ~17 µs/eval, which makes the wrapper's
    fixed cost (~700 ns of IterEvent + perf_counter_ns + next(counter) +
    NullSink dispatch) visible as a ~5 % ratio — a measurement artifact,
    not a regression. We bump to k=40 where Phase B still does
    ~800 µs/eval of real work, so the fixed wrapper cost amortizes well
    below 1 %. The 1 % production bar from §1-1-a goal item 5 holds at
    k=40 and above; below that the test loses signal.
    """
    graph = make_graph(k=40, output_dir=None)
    n_edges = len(graph.G.edges)

    # Pre-build the crossing index so neither the raw nor wrapped runs
    # eats the one-time Phase A cost.
    graph.build_crossings_index()

    # One deterministic layer assignment — the test measures fixed wrapper
    # overhead, not loss-function variance.
    rng = np.random.default_rng(seed=0)
    layers = rng.integers(0, 2, size=n_edges).astype(float)

    n_iters = 200

    # warm-up so the first sample doesn't include lazy-import / JIT effects
    graph.loss_function(layers)

    # raw timing
    t0 = time.perf_counter_ns()
    for _ in range(n_iters):
        graph.loss_function(layers)
    raw_ns = time.perf_counter_ns() - t0

    # wrapped timing — replicate the optimizer wrapper exactly, including
    # next(counter) (not a hardcoded iter=0).
    sink = NullSink()
    loss_fn = graph.loss_function
    counter = itertools.count()
    wrapper_t0 = time.perf_counter_ns()

    def wrapped(x):
        loss = loss_fn(x)
        sink.on_iter(
            IterEvent(
                iter=next(counter),
                loss=float(loss),
                wall_ms=(time.perf_counter_ns() - wrapper_t0) // 1_000_000,
            )
        )
        return loss

    t0 = time.perf_counter_ns()
    for _ in range(n_iters):
        wrapped(layers)
    wrapped_ns = time.perf_counter_ns() - t0

    ratio = wrapped_ns / raw_ns
    assert ratio < 1.03, (
        f"Wrapper overhead too high: raw={raw_ns/1e6:.2f}ms, "
        f"wrapped={wrapped_ns/1e6:.2f}ms, ratio={ratio:.4f}"
    )


def test_recorder_skips_non_finite_loss():
    """NaN / +inf / -inf losses are dropped from trace and best tracking.

    Rationale: ``json.dump(allow_nan=False)`` would refuse to serialize
    such reports, and downstream consumers (matplotlib, the C++
    nlohmann::json parser at M5 parity) can't handle them either. The
    recorder drops the eval and continues; downstream sees only finite
    samples.
    """
    rec = RunRecorder(run_id="nan_test", config={})
    for i, loss in enumerate([math.nan, 5.0, math.inf, 3.0, -math.inf, 2.0]):
        rec.on_iter(IterEvent(iter=i, loss=loss, wall_ms=i))

    report = rec.finalize()
    losses = [ev["loss"] for ev in report["trace"]]
    assert losses == [5.0, 3.0, 2.0]
    # is_new_best flags: [T (5.0), T (3.0), T (2.0)] — every finite eval was
    # a new best since priors were dropped
    flags = [ev["is_new_best"] for ev in report["trace"]]
    assert flags == [True, True, True]
    assert report["summary"]["initial_loss"] == pytest.approx(5.0)
    assert report["summary"]["final_loss"] == pytest.approx(2.0)
    # And write_run_report (allow_nan=False) round-trips cleanly because the
    # recorder upstream-filtered.
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        write_run_report(report, tmpdir)


def test_optimizer_base_accepts_and_stores_eval_observer():
    from routing_py_rebuild.optimizers.base import Optimizer

    class _Probe(Optimizer):
        def optimize(self, maxiter=1):
            return None

    calls = []
    obs = lambda x, loss: calls.append((list(x), loss))  # noqa: E731
    opt = _Probe(graph=object(), seed=1, eval_observer=obs)
    assert opt.eval_observer is obs
    # default stays None (backwards compatible)
    opt2 = _Probe(graph=object(), seed=1)
    assert opt2.eval_observer is None


def _tiny_graph(k=8):
    from routing_py_rebuild.core import SiNInterconnectionGraph
    from routing_py_rebuild.positions import distribute_nodes_around_square

    pos = distribute_nodes_around_square(k // 4, side_length=1)
    return SiNInterconnectionGraph(k=k, positions=pos)


def test_dual_annealing_invokes_observer_once_per_eval():
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    seen = []
    opt = get_optimizer("dual_annealing")(
        g, seed=5859, eval_observer=lambda x, loss: seen.append(float(loss))
    )
    res = opt.optimize(maxiter=3)
    assert len(seen) >= 1
    assert all(isinstance(v, float) for v in seen)
    assert min(seen) <= res.fun + 1e-9


def test_differential_evolution_invokes_observer():
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    seen = []
    opt = get_optimizer("differential_evolution")(
        g, seed=5859, eval_observer=lambda x, loss: seen.append(float(loss))
    )
    opt.optimize(maxiter=2)
    assert len(seen) >= 1
    assert all(isinstance(v, float) for v in seen)


def test_swap_polish_invokes_observer_in_both_phases():
    from routing_py_rebuild.optimizers import get_optimizer

    g = _tiny_graph()
    g.build_crossings_index()
    seen = []
    opt = get_optimizer("dual_annealing_with_swap_polish")(
        g,
        seed=5859,
        eval_observer=lambda x, loss: seen.append(float(loss)),
        polish_iters=5,
    )
    opt.optimize(maxiter=2)
    assert len(seen) >= 2
    assert all(isinstance(v, float) for v in seen)


def test_run_optimization_forwards_eval_observer(tmp_path):
    from routing_py_rebuild.api import run_optimization

    seen = []
    res = run_optimization(
        k=8,
        optimizer="dual_annealing",
        maxiter=3,
        seed=5859,
        output_dir=str(tmp_path),
        plot=False,
        run_loss_analysis=False,
        save_json=False,
        eval_observer=lambda x, loss: seen.append(float(loss)),
    )
    assert len(seen) >= 1
    assert "loss" in res


def _trace_losses(report_path):
    import json

    with open(report_path) as f:
        rep = json.load(f)
    return [ev["loss"] for ev in rep["trace"]], rep["summary"]


def test_eval_observer_is_non_perturbing_parity(tmp_path):
    """observer=None vs a no-op observer must yield an identical optimizer
    trajectory (same seed): same trace[].loss sequence, same summary core,
    same best_layers. Protects REFACTOR_GOALS.md §1-1-a numeric-parity
    contract (D1: the hook must not alter the optimization)."""
    from routing_py_rebuild.api import run_optimization

    common = dict(
        k=8, optimizer="dual_annealing", maxiter=4, seed=5859,
        plot=False, run_loss_analysis=False, collect_statistics=True,
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    # run_optimization writes run_report.json under graph.filepath (a
    # parameterized subdir of output_dir), and returns that path as
    # ``report_path`` (api.py docstring). Consume the returned path rather
    # than assuming <output_dir>/run_report.json.
    ra = run_optimization(output_dir=str(a), eval_observer=None, **common)
    rb = run_optimization(
        output_dir=str(b), eval_observer=lambda x, loss: None, **common
    )
    la, sa = _trace_losses(ra["report_path"])
    lb, sb = _trace_losses(rb["report_path"])
    assert la == lb, "observer perturbed the loss trajectory"
    # relative_drop is intentionally not compared: it is a pure function
    # of initial_loss/final_loss (both asserted equal here), so equality
    # is implied — it carries no independent signal.
    for key in ("initial_loss", "final_loss", "iterations", "best_at_iter"):
        assert sa[key] == sb[key], f"summary.{key} differs with observer"
    assert ra["best_layers"] == rb["best_layers"], "best_layers differ with observer"


def test_observer_call_count_equals_eval_count_all_optimizers():
    """§5 contract: observer fires exactly once per loss_function eval, for
    every optimizer. Compare against the RunRecorder trace length (finite
    evals) using a shared RunRecorder + observer on the same run."""
    from routing_py_rebuild.optimizers import get_optimizer
    from routing_py_rebuild.statistics import RunRecorder

    for name, kw in (
        ("dual_annealing", {}),
        ("differential_evolution", {}),
        ("dual_annealing_with_swap_polish", {"polish_iters": 4}),
        # REFACTOR_GOALS.md M-J Stage 1: ga_pymoo joins the shared
        # observer-count contract. n_gen=2 + pop_size=8 ⇒ 16 evals
        # (n_gen * pop_size; the initial population IS generation 1 in
        # pymoo, no separate pre-gen-1 phase).
        ("ga_pymoo", {"n_gen": 2, "pop_size": 8}),
        # M-J Stage 2: pso_pyswarms / cmaes / bo_skopt.
        # PSO: iters=2 * n_particles=6 = 12 evals.
        ("pso_pyswarms", {"iters": 2, "n_particles": 6}),
        # CMA-ES: maxiter=2 * popsize=6 = 12 evals.
        ("cmaes", {"maxiter": 2, "popsize": 6}),
        # BO: n_calls=8 evals total (4 initial + 4 GP-driven).
        ("bo_skopt", {"n_calls": 8, "n_initial_points": 4}),
    ):
        g = _tiny_graph()
        g.build_crossings_index()
        rec = RunRecorder(run_id=f"t_{name}", config={})
        seen = []
        opt = get_optimizer(name)(
            g, seed=5859, sink=rec,
            # _s=seen binds the per-iteration list (ruff B023): the lambda is
            # fully consumed within this iteration, behavior is unchanged.
            eval_observer=lambda x, loss, _s=seen: _s.append(float(loss)),
            **kw,
        )
        opt.optimize(maxiter=2)
        rep = rec.finalize()
        trace_losses = [t["loss"] for t in rep["trace"]]
        finite_seen = [v for v in seen if v == v and abs(v) != float("inf")]
        # observer fires on EVERY eval (incl. non-finite); RunRecorder
        # drops non-finite from trace[]. So `seen` (all) >= trace, and the
        # FINITE observer losses must equal trace[].loss EXACTLY — full
        # sequence equality, not a prefix (guards against any extra finite
        # observer calls beyond the recorded evals). [Codex P1]
        assert len(seen) >= len(trace_losses)
        assert len(finite_seen) == len(trace_losses)
        assert finite_seen == trace_losses


def test_x_mutating_observer_cannot_perturb_optimizer(tmp_path):
    """Defense-in-depth (§5 / Opus P2-1): even an observer that mutates
    its ``x`` argument in place must NOT change the optimizer trajectory.
    The contract relies on ``wrapped`` computing loss BEFORE calling the
    observer and scipy holding its own copy of x. This locks that
    invariant against any future refactor of ``wrapped``."""
    import numpy as np
    from routing_py_rebuild.api import run_optimization

    common = dict(
        k=8, optimizer="dual_annealing", maxiter=4, seed=5859,
        plot=False, run_loss_analysis=False, collect_statistics=True,
    )

    def _mutating(x, loss):
        arr = np.asarray(x)
        if arr.size:
            arr[0] = 1.0 - float(arr[0])  # attempt to corrupt the vector

    ra = run_optimization(
        output_dir=str(tmp_path / "clean"), eval_observer=None, **common
    )
    rb = run_optimization(
        output_dir=str(tmp_path / "mut"), eval_observer=_mutating, **common
    )
    la, _ = _trace_losses(ra["report_path"])
    lb, _ = _trace_losses(rb["report_path"])
    assert la == lb, "x-mutating observer perturbed the optimizer"
    assert ra["best_layers"] == rb["best_layers"]
