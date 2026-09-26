"""Optimizer abstract base + name-based registry.

Each concrete optimizer subclasses :class:`Optimizer`, registers itself via
``@register_optimizer("name")``, and implements :meth:`optimize`. Optimizers
hold a reference to a :class:`SiNInterconnectionGraph` and consume its
``loss_function``; they do not mutate the graph until :meth:`optimize`
returns, at which point the graph receives the best layer assignment.

Per REFACTOR_GOALS.md §1-1-a, optimizers also accept a ``sink`` (StatsSink)
and emit one ``IterEvent`` per ``loss_function`` evaluation through a
wrapper inside :meth:`optimize`. Default sink is :class:`NullSink` so
existing call-sites are unaffected.

Equal-budget comparison support (Stage H, 2026-05-27)
-----------------------------------------------------
The :class:`BudgetGuard` helper enables fair cross-optimizer comparison by
exposing two universal caps:

  * ``max_nfe``: stop after this many objective evaluations
  * ``wall_time_s``: stop after this many wall-clock seconds

Each wrapper polls ``guard.check_and_record(x, loss)`` from its wrapped
loss function. On exceeded budget, :class:`BudgetCapExceeded` is raised
to short-circuit out of the underlying library; the wrapper catches it
and returns the best-so-far state held by the guard. The native NFE-cap
kwarg of each library (scipy ``maxfun``, pymoo ``n_gen``, etc.) is
still set as a redundant safety net.

Default ``max_nfe=None`` / ``wall_time_s=None`` ⇒ both caps disabled,
i.e. byte-exact pre-Stage-H behavior.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ..statistics import NullSink, StatsSink


@dataclass
class OptimizationResult:
    """Lightweight container for optimizer output."""

    best_layers: list[int]
    fun: float
    raw: object | None = None  # underlying scipy result, optional


# --- Equal-budget comparison machinery (Stage H, 2026-05-27) -------------


class BudgetCapExceeded(Exception):
    """Raised by :class:`BudgetGuard` when either NFE or wall-time cap is
    exceeded. Wrappers catch this to short-circuit out of the underlying
    library and return the best-so-far state instead of letting the
    library keep running past budget."""


@dataclass
class BudgetGuard:
    """Shared NFE + wall-time budget enforcement for fair comparison.

    Polled from each wrapper's wrapped() closure on every loss eval:
      * tracks (best_x, best_loss) so the wrapper can return them on
        cap-exit instead of whatever library state was current,
      * counts NFE,
      * raises :class:`BudgetCapExceeded` once either cap fires.

    Both caps are optional; ``max_nfe=None`` AND ``wall_time_s=None``
    yields a no-op guard (pre-Stage-H parity).

    Caller contract:
      * Construct ONCE per :meth:`optimize` call, before the library
        starts.
      * Call :meth:`check_and_record(x, loss)` INSIDE the wrapped loss
        function, AFTER ``loss = loss_function(x)`` (so the eval already
        happened and the state is observable).
      * Catch :class:`BudgetCapExceeded` in the wrapper's outer try.
      * On catch, return ``OptimizationResult(best_layers=guard.best_layers,
        fun=guard.best_loss, raw=None)``.

    The guard's clock origin (``_t0_ns``) is set at construction time.
    Pass ``t0_ns`` explicitly to share the clock with the wrapper's
    PhaseTimer / IterEvent.wall_ms origin (Stage M-B contract).
    """

    max_nfe: int | None = None
    wall_time_s: float | None = None
    _t0_ns: int | None = None
    nfe: int = 0
    best_loss: float = float("inf")
    best_x: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self._t0_ns is None:
            self._t0_ns = time.perf_counter_ns()
        self._wall_ns_limit = (
            None
            if self.wall_time_s is None
            else int(self.wall_time_s * 1_000_000_000)
        )

    @property
    def active(self) -> bool:
        """True if either cap is set (otherwise guard is a no-op)."""
        return self.max_nfe is not None or self._wall_ns_limit is not None

    @property
    def best_layers(self) -> list[int]:
        """Best-so-far rounded to integer layer assignment."""
        return [int(round(v)) for v in self.best_x]

    def check_and_record(
        self, x: Sequence[float], loss: float
    ) -> None:
        """Record this eval and raise if either cap is now exceeded.

        Must be called from the wrapped loss function AFTER computing
        ``loss``. The (x, loss) pair is tracked as best-so-far iff
        ``loss < self.best_loss`` (strict <; ties keep the earlier
        winner). NFE counter and clock are checked AFTER recording so
        the just-finished eval is counted before we declare cap-hit.
        """
        if loss < self.best_loss:
            self.best_loss = float(loss)
            # Copy to a fresh list — scipy may reuse the underlying
            # buffer for the next eval, and a no-copy snapshot would
            # silently mutate the recorded best_x.
            self.best_x = list(map(float, x))
        self.nfe += 1
        if self.max_nfe is not None and self.nfe >= self.max_nfe:
            raise BudgetCapExceeded(
                f"NFE cap reached: nfe={self.nfe} >= max_nfe={self.max_nfe}"
            )
        if self._wall_ns_limit is not None:
            elapsed_ns = time.perf_counter_ns() - self._t0_ns
            if elapsed_ns >= self._wall_ns_limit:
                raise BudgetCapExceeded(
                    f"wall-time cap reached: "
                    f"{elapsed_ns / 1e9:.2f}s >= {self.wall_time_s:.2f}s"
                )


def observer_copy(x) -> np.ndarray:
    """The ``x`` an ``eval_observer`` receives: a private float copy.

    Optimizers keep the evaluated vector — scipy copies an accepted point
    into its own state only after the objective returns, and population
    methods pass rows of their population matrix — so an observer that
    wrote to the live vector would reach the optimizer's state and result.
    """
    return np.array(x, dtype=float)


_REGISTRY: dict[str, type[Optimizer]] = {}


def register_optimizer(name: str) -> Callable[[type[Optimizer]], type[Optimizer]]:
    """Decorator that registers an optimizer subclass under ``name``."""

    def _wrap(cls: type[Optimizer]) -> type[Optimizer]:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_optimizer(name: str) -> type[Optimizer]:
    """Lookup a registered optimizer class by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown optimizer '{name}'. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


class Optimizer:
    """Abstract base. Subclasses must implement :meth:`optimize`."""

    name: str = ""

    def __init__(
        self,
        graph,
        *,
        seed: int | None = None,
        sink: StatsSink | None = None,
        eval_observer: Callable[[Sequence[float], float], None] | None = None,
        **kwargs,
    ):
        self.graph = graph
        self.seed = seed
        self.sink: StatsSink = sink if sink is not None else NullSink()
        # REFACTOR_GOALS.md §1-1-a: optional, schema-neutral per-eval hook.
        # Called once per loss_function evaluation by each optimizer's
        # `wrapped`, at the same site/order as `sink.on_iter`. Default
        # None ⇒ numerically identical to pre-L2 (parity-tested).
        # CONTRACT: the observer MUST treat `x` as read-only (must not
        # mutate it). It is called only AFTER `loss = loss_function(x)`,
        # and every call site passes `observer_copy(x)` — a private copy,
        # made only when an observer is set (the default None path stays
        # copy-free, spec D1 low-overhead) — because scipy copies an
        # accepted point after the call, so writes to the live vector
        # would leak into its state. `test_x_mutating_observer_cannot_
        # perturb_optimizer` is the defense-in-depth guard.
        self.eval_observer = eval_observer
        self.kwargs = kwargs

    # --------------------------------------------------------------
    # Helpers shared by subclasses for building seed initial guesses
    # --------------------------------------------------------------
    def _seed_from_routing(self, takeaway: list[int], layer: int = 1) -> np.ndarray:
        """Run ``routing_method_1`` on the graph to build an initial vector."""
        nx.set_edge_attributes(self.graph.G, 0, "layer")
        self.graph.routing_method_1(takeaway=takeaway, layer=layer)
        seed = np.array([self.graph.G.edges[e]["layer"] for e in self.graph.G.edges])
        nx.set_edge_attributes(self.graph.G, 0, "layer")
        return seed

    def optimize(self, maxiter: int = 1000) -> OptimizationResult:
        raise NotImplementedError
