"""M-D acceptance tests: experiments/crossings.py geometric intralayer
crossing counter (spec §7, REFACTOR_GOALS.md §2-1/§2-3).

Three independent computations of the same integer must agree:
1. crossings.py O(X) counter (cached _crossing_pairs classification),
2. brute-force O(E²) oracle (shapely edge_crosses, no Phase A),
3. run_report-derived (Σ total_crossings_of_sub_G == Σ summary.layers[].crossings),
on both the Python and (when the binary is available) C++ backends.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from experiments.crossings import count_intralayer_geometric
from routing_py_rebuild.api import _extract_aggregate_stats
from routing_py_rebuild.core import SiNInterconnectionGraph
from routing_py_rebuild.positions import distribute_nodes

from tests.parity_harness import CPP_BINARY, _output_subdir, hermetic_env
from tests.test_crossings_oracle import _brute_force_pair_set


def _even_split(total: int, parts: int) -> list[int]:
    """Near-even positive split of ``total`` into ``parts`` buckets, each
    >= 1, summing exactly to ``total`` (the front buckets absorb the
    remainder). Used to derive minimal valid per-side counts for the
    multi-side generators so every shape yields exactly ``total``
    boundary nodes (M-D Task 4 adaptation note: M-A's
    ``distribute_nodes`` takes shape-specific kwargs, not a bare ``k``).
    """
    if total < parts:
        raise ValueError(
            f"cannot split {total} into {parts} buckets each >= 1"
        )
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _positions_for(shape: str, k: int) -> dict:
    """Return a positions dict with exactly ``k`` boundary nodes for
    ``shape``, using the M-A ``distribute_nodes`` keyword surface
    (verified against ``routing_py_rebuild/positions.py`` +
    ``tests/test_positions.py``).

    Per-shape minimal valid args summing to ``k``:
      * ``square``  : ``nodes_per_side = k // 4`` (4 equal sides;
        requires ``k % 4 == 0`` — the test ``k`` set {8,12,20} all are).
      * ``circle``  : ``k`` nodes equally spaced (any ``k >= 3``).
      * ``triangle``: near-even 3-split ``[n0,n1,n2]`` summing to ``k``.
      * ``polygon`` : 4 sides, ``k // 4`` per side (requires
        ``k % 4 == 0``); ``k`` total = ``n_sides * per_side``.
      * ``partial_rectangle``: near-even 3-split over
        ``{top,right,bottom}`` summing to ``k``.
    """
    if shape == "square":
        if k % 4 != 0:
            raise ValueError(f"square needs k % 4 == 0; got k={k}")
        return distribute_nodes(shape="square", nodes_per_side=k // 4)
    if shape == "circle":
        return distribute_nodes(shape="circle", k=k)
    if shape == "triangle":
        return distribute_nodes(
            shape="triangle", triangle_nodes_per_side=_even_split(k, 3)
        )
    if shape == "polygon":
        if k % 4 != 0:
            raise ValueError(f"polygon needs k % 4 == 0; got k={k}")
        return distribute_nodes(
            shape="polygon", polygon_n_sides=4, polygon_nodes_per_side=k // 4
        )
    if shape == "partial_rectangle":
        a, b, c = _even_split(k, 3)
        return distribute_nodes(
            shape="partial_rectangle",
            side_counts={"top": a, "right": b, "bottom": c},
        )
    raise ValueError(f"unsupported shape {shape!r}")


def _make_graph(
    shape: str, k: int, *, L: int = 3, perimeter_layer: int = 0
) -> SiNInterconnectionGraph:
    positions = _positions_for(shape, k)
    g = SiNInterconnectionGraph(
        k=k, positions=positions, L=L, perimeter_layer=perimeter_layer
    )
    g.build_crossings_index()
    return g


def test_wrong_length_raises_valueerror() -> None:
    g = _make_graph("square", 8)
    n = len(g._edge_list)
    with pytest.raises(ValueError, match=r"layers has shape .*expected"):
        count_intralayer_geometric(g, np.zeros(n + 1))


def test_returns_plain_int() -> None:
    g = _make_graph("square", 8)
    n = len(g._edge_list)
    out = count_intralayer_geometric(g, np.zeros(n, dtype=float))
    assert type(out) is int


def test_all_distinct_layers_zero_intralayer() -> None:
    # L large enough that a per-edge unique layer is representable; every
    # crossing pair is then cross-layer ⇒ geometric intralayer == 0.
    g = _make_graph("square", 12, L=200)
    n = len(g._edge_list)
    layers = np.arange(n, dtype=float)  # all distinct, all in [0, 200)
    assert count_intralayer_geometric(g, layers) == 0


def test_counter_does_not_recompute_geometry() -> None:
    """Spec §7 hard constraint: after build_crossings_index(), the counter
    must classify the cached pairs only — never call edge_crosses /
    shapely per eval. Poison edge_crosses post-build; the counter must
    still work (proving it touched no geometry)."""
    g = _make_graph("square", 12)

    def _poisoned(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("crossings.py recomputed geometry per eval")

    g.edge_crosses = _poisoned  # type: ignore[method-assign]
    n = len(g._edge_list)
    rng = np.random.default_rng(0)
    layers = rng.integers(0, g.L, size=n).astype(float)
    # Must not raise AssertionError.
    assert isinstance(count_intralayer_geometric(g, layers), int)


def test_build_index_is_idempotent_via_counter() -> None:
    g = _make_graph("circle", 10)
    pairs_id = id(g._crossing_pairs)
    n = len(g._edge_list)
    count_intralayer_geometric(g, np.zeros(n))
    count_intralayer_geometric(g, np.zeros(n))
    # The counter's internal build_crossings_index() call is a no-op once
    # populated (core.py:434 early return) — the cached array identity is
    # unchanged across calls.
    assert id(g._crossing_pairs) == pairs_id


def _pinned_layers(g: SiNInterconnectionGraph, layers: np.ndarray) -> np.ndarray:
    """Replicate core.py::loss_function's pinned derivation exactly
    (round → int64 → perimeter pin to graph.perimeter_layer)."""
    layers_int = np.round(np.asarray(layers, dtype=float)).astype(np.int64)
    pinned = layers_int.copy()
    pinned[g._perimeter_mask] = g.perimeter_layer
    return pinned


def _brute_force_intralayer(g: SiNInterconnectionGraph, layers: np.ndarray) -> int:
    """Independent O(E²) ground truth: shapely edge_crosses pair set
    (no Phase A) classified by the same pinned-layer + [0, L) gate."""
    pinned = _pinned_layers(g, layers)
    pairs = _brute_force_pair_set(g)  # set of (i, j) edge-index pairs, i<j
    total = 0
    for i, j in pairs:
        a = pinned[i]
        if a == pinned[j] and 0 <= a < g.L:
            total += 1
    return total


def _run_report_intralayer(
    shape: str,
    k: int,
    L: int,
    layers: np.ndarray,
    *,
    perimeter_layer: int = 0,
) -> int:
    """Independent production path: apply_optimization_result →
    analyze_loss → Σ total_crossings_of_sub_G, cross-checked against
    Σ _extract_aggregate_stats(...)['layers'][i]['crossings'].

    ``perimeter_layer`` defaults to 0 so existing call sites are
    unaffected (the constructor's own default — ``perimeter_layer ==
    edge_coupler_layer == 0`` — yields the identical graph). Threaded so
    the non-default-pin oracle stays a true independent production-path
    check: this path pins perimeter edges via the real
    ``apply_optimization_result``/``create_subgraphs`` machinery using
    ``graph.perimeter_layer``, never a literal 0.
    """
    g = SiNInterconnectionGraph(
        k=k,
        positions=_positions_for(shape, k),
        L=L,
        perimeter_layer=perimeter_layer,
    )
    g.apply_optimization_result(layers)
    g.analyze_loss()
    via_attr = sum(int(c) for c in g.total_crossings_of_sub_G)
    stats = _extract_aggregate_stats(g)
    via_summary = sum(int(d["crossings"]) for d in stats["layers"])
    assert via_attr == via_summary, (
        f"Python run_report self-inconsistency: total_crossings_of_sub_G "
        f"sum={via_attr} vs summary.layers sum={via_summary}"
    )
    return via_attr


_SHAPES = ["square", "circle", "triangle", "polygon", "partial_rectangle"]


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("k", [8, 12, 20])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_three_way_oracle_python(shape: str, k: int, seed: int) -> None:
    """Spec §7 acceptance: harness counter == brute-force O(E²) ==
    Σ run_report.summary.layers[].crossings, several shapes × small k."""
    L = 3
    g = _make_graph(shape, k, L=L)
    n = len(g._edge_list)
    rng = np.random.default_rng(seed)
    # In-range integer layers, as the clamped optimizer surface produces.
    layers = rng.integers(0, L, size=n).astype(float)

    harness = count_intralayer_geometric(g, layers)
    brute = _brute_force_intralayer(g, layers)
    report = _run_report_intralayer(shape, k, L, layers)

    assert harness == brute == report, (
        f"3-way mismatch [{shape} k={k} seed={seed}]: "
        f"harness={harness} brute={brute} run_report={report}"
    )


def test_out_of_range_layers_reconcile() -> None:
    """Lock the [0, L) gate (the M-D analogue of REFACTOR_GOALS.md §1-2
    design-tradeoff #4 / §1-4 P2-O5) with a DISCRIMINATING fixture: the
    gate must be the *deciding* factor for the asserted value, not a
    no-op that the assertion would pass with the gate deleted.

    Construction: pick ONE actual cached crossing pair (i, j) whose two
    edges are BOTH non-perimeter (so neither is layer-pinned and the
    pair's classification is governed purely by `layers[i]`/`layers[j]`),
    start from an all-in-[0, L) baseline (gate is a no-op for every other
    pair), then assign BOTH edges of that pair the SAME out-of-range
    value L+5. Now the pair is same-layer (`la == lb == L+5`) but
    out-of-range:
      * WITH the [0, L) gate it is dropped;
      * WITHOUT the gate it would be counted (`same` is True).
    So this fixture's value flips on the gate — exactly the property the
    prior version (two edges at *distinct* out-of-range values, already
    cross-layer and excluded by the `same` mask regardless of the gate)
    lacked.

    Positions/graph are built via the shared _make_graph / _positions_for
    helpers; _run_report_intralayer routes through the same _positions_for
    so the C++/Python convention stays identical. square k=12 L=2 is
    chosen because its cached _crossing_pairs is non-empty (495 pairs).
    """
    shape, k, L = "square", 12, 2
    g = _make_graph(shape, k, L=L)
    n = len(g._edge_list)

    # Deterministically pick the first cached crossing pair whose BOTH
    # edges are non-perimeter (|u-v| not in {1, k-1}). Perimeter edges are
    # layer-pinned regardless, so they cannot be used to exercise the gate.
    def _is_perim(idx: int) -> bool:
        u, v = g._edge_list[idx]
        return abs(u - v) == 1 or abs(u - v) == k - 1

    chosen: tuple[int, int] | None = None
    for row in range(g._crossing_pairs.shape[0]):
        ci = int(g._crossing_pairs[row, 0])
        cj = int(g._crossing_pairs[row, 1])
        if not _is_perim(ci) and not _is_perim(cj):
            chosen = (ci, cj)
            break
    assert chosen is not None, (
        "fixture invalid: no cached crossing pair with both edges "
        "non-perimeter — pick a different shape/k"
    )
    ei, ej = chosen

    # All-in-[0, L) baseline ⇒ the gate is a no-op for every pair EXCEPT
    # the one we push out of range below; that isolates the gate as the
    # sole cause of any count delta for this fixture.
    layers = np.array([i % L for i in range(n)], dtype=float)
    # Same out-of-range value on BOTH edges of the chosen crossing pair:
    # same-layer (would be counted by raw `same`) but out of [0, L)
    # (dropped by the gate). This is what makes the fixture discriminate.
    layers[ei] = L + 5
    layers[ej] = L + 5

    harness = count_intralayer_geometric(g, layers)
    brute = _brute_force_intralayer(g, layers)
    report = _run_report_intralayer(shape, k, L, layers)
    # All three independent paths must agree on the GATED value:
    # create_subgraphs buckets only [0, L) so run_report drops the pair;
    # _brute_force_intralayer applies the same `0 <= a < g.L` gate; the
    # counter applies its `in_range` gate.
    assert harness == brute == report, (
        f"out-of-range reconciliation broken [{shape}]: "
        f"harness={harness} brute={brute} run_report={report}"
    )

    # --- P2-O5-analogue contract lock: prove NON-VACUITY. ---
    # Reference == the counter's logic MINUS the [0, L) gate, i.e. the raw
    # same-layer count over the identical pinned _crossing_pairs (this is
    # exactly loss_function's ungated `same.sum()`, core.py:870). If the
    # gate were a no-op for this fixture this would equal `harness`; the
    # whole point of the new construction is that it does NOT. Asserting
    # STRICT inequality is what the old test could not do — it locks the
    # gate as load-bearing, the M-D analogue of the §1-2 #4 / §1-4 P2-O5
    # raw-value/no-clamp parity contract.
    pinned = _pinned_layers(g, layers)
    la = pinned[g._crossing_pairs[:, 0]]
    lb = pinned[g._crossing_pairs[:, 1]]
    ungated = int(np.count_nonzero(la == lb))
    assert ungated > harness, (
        "VACUOUS GATE TEST: removing the [0, L) gate did not change the "
        f"outcome for this fixture (ungated same-layer count={ungated}, "
        f"gated count={harness}); the fixture fails to discriminate the "
        "gate and would pass even with the gate deleted"
    )


@pytest.mark.parametrize("shape", ["triangle"])
@pytest.mark.parametrize("k", [12, 20])
def test_non_default_perimeter_layer_oracle(shape: str, k: int) -> None:
    """Non-default ``perimeter_layer`` coverage: with the perimeter ring
    pinned to a NON-zero layer, the three independent paths must still
    agree — and the fixture is chosen so this FAILS if ``crossings.py``
    were to hardcode the pin to 0 instead of reading
    ``graph.perimeter_layer``.

    Discrimination requires perimeter edges to actually participate in
    cached crossing pairs (otherwise the pin value is invisible to the
    count). Empirically only ``triangle`` satisfies this at small k —
    its outline has collinear in-side nodes whose ring chords cross
    interior chords (k=12 → 2 such pairs, k=20 → 17). The convex shapes
    (``circle``/``square``/``polygon``/``partial_rectangle``) have ZERO
    perimeter-involved crossing pairs at these sizes (a hull-edge chord
    of a convex point set crosses nothing), so they cannot discriminate
    a hardcoded-0 regression and are deliberately excluded here. (See
    REPORT note: this is a justified deviation from the suggested
    {triangle, circle} parametrization — circle would be a vacuous
    discrimination case.)

    ``_brute_force_intralayer``/``_pinned_layers`` pin to
    ``graph.perimeter_layer`` (not a literal 0) and
    ``_run_report_intralayer`` routes through the real
    ``apply_optimization_result``/``create_subgraphs`` with the same
    non-default pin, so both remain TRUE independent oracles under the
    non-default ``perimeter_layer``.
    """
    L, perimeter_layer, seed = 3, 1, 0
    g = _make_graph(shape, k, L=L, perimeter_layer=perimeter_layer)
    assert g.perimeter_layer == perimeter_layer
    n = len(g._edge_list)
    rng = np.random.default_rng(seed)
    layers = rng.integers(0, L, size=n).astype(float)

    harness = count_intralayer_geometric(g, layers)
    brute = _brute_force_intralayer(g, layers)
    report = _run_report_intralayer(
        shape, k, L, layers, perimeter_layer=perimeter_layer
    )
    assert harness == brute == report, (
        f"non-default perimeter_layer 3-way mismatch [{shape} k={k} "
        f"perimeter_layer={perimeter_layer}]: harness={harness} "
        f"brute={brute} run_report={report}"
    )

    # Anti-regression lock: a hardcoded-0 pin (the bug this guards
    # against — cf. the M2 helper `_phase_b_per_edge_counts` which
    # hardcodes 0 and only coincides for default perimeter_layer=0) would
    # produce a DIFFERENT count for this fixture. Assert the correct
    # (perimeter_layer-pinned) value differs from the hardcoded-0 value,
    # so this test genuinely fails if crossings.py regresses to a literal
    # 0. This recomputes the counter's own logic with the only change
    # being the pin target, isolating the pin as the discriminator.
    li = np.round(np.asarray(layers, dtype=float)).astype(np.int64)
    pin0 = li.copy()
    pin0[g._perimeter_mask] = 0  # the regression we guard against
    la0 = pin0[g._crossing_pairs[:, 0]]
    lb0 = pin0[g._crossing_pairs[:, 1]]
    hardcoded0 = int(
        np.count_nonzero((la0 == lb0) & (la0 >= 0) & (la0 < g.L))
    )
    assert harness != hardcoded0, (
        f"non-discriminating fixture [{shape} k={k}]: a hardcoded-0 pin "
        f"yields the SAME count ({hardcoded0}) as the correct "
        f"perimeter_layer={perimeter_layer} pin ({harness}); this test "
        "would not catch a crossings.py regression to a literal 0"
    )


def _cpp_intralayer_from_subgraphsdata(data: dict) -> int:
    """Derive the geometric intralayer total from a C++
    ``subgraphsdata.json`` payload, mirroring the C++ aggregate exactly
    (``main.cpp`` ~546-554: per-layer ``total_intra = Σ edge.crossings``,
    reported as ``total_intra / 2``; integer-divide **per layer** then
    sum, NOT a single divide of the global sum).

    The per-edge ``crossings`` key is written by ``io.cpp`` (~58,
    ``edge_array_v2``/``edge_array_legacy``); each ``Layer_i`` object is
    ``{"edges": [[u, v, {"crossings": int, ...}], ...]}`` per
    ``code/schema/subgraphsdata.schema.json`` (``patternProperties``
    ``^Layer_[0-9]+$`` → ``required: ["edges"]``).
    """
    total = 0
    for key, layer in data.items():
        if not key.startswith("Layer_"):
            continue
        layer_sum = 0
        for edge in layer["edges"]:
            # edge == [u, v, attrs]; attrs["crossings"] is per-edge int.
            layer_sum += int(edge[2]["crossings"])
        # Per-layer endpoint-stamped crossing sums are mathematically
        # EVEN: every same-layer crossing pair stamps +1 on each of its
        # two endpoint edges, so the per-layer total is 2 × (geometric
        # pair count). An odd sum means a C++ accounting bug — fail
        # loudly here rather than silently flooring it via `// 2`.
        assert layer_sum % 2 == 0, (
            f"C++ subgraphsdata {key}: Σ edge.crossings={layer_sum} is "
            f"ODD; per-layer endpoint-stamped crossing sums must be even "
            f"(2 × geometric pair count) — likely a C++ accounting bug"
        )
        total += layer_sum // 2
    return total


@pytest.mark.parametrize("shape", ["square", "circle", "triangle"])
def test_cross_backend_cpp_subgraphsdata(shape: str, tmp_path: Path) -> None:
    """Spec §7 cross-backend: for byte-identical positions + a fixed
    layer vector, the geometric intralayer total derived from the C++
    ``subgraphsdata.json`` (Σ_i ⌊Σ Layer_i edge.crossings / 2⌋, mirroring
    ``main.cpp`` ``total_intra/2``) must equal the Python harness counter
    EXACTLY (integer identity — no tolerance).

    Mechanism (the prior implementer's empirically-validated path —
    circle k=12 → C++-derived 256 == Python 256): drive ``Autowiring_CPP``
    with ``hermetic_env()`` + ``SINIC_POSITIONS_JSON`` (byte-identical
    geometry; ``main.cpp`` ~309-322 makes ``k = len(positions)`` and the
    output folder ``cl_…_tl_…_itl_…_nodes_{k}`` == ``_output_subdir``) +
    ``SINIC_FIXED_LAYERS_JSON`` (``{"layers": [...]}`` object —
    ``main.cpp`` ~423-427). ``SINIC_WRITE_RUN_REPORT`` is deliberately
    NOT set: it is mutually exclusive with ``SINIC_FIXED_LAYERS_JSON``
    (``main.cpp`` ~387-393; see
    ``test_parity.py::test_cpp_fixed_layers_run_report_mutual_exclusion``)
    so the run_report path is unavailable here — ``subgraphsdata.json``
    is the authoritative artifact instead.

    Skips cleanly (never fails) if ``Autowiring_CPP`` is not built, the
    subprocess returns nonzero, or the expected ``subgraphsdata.json`` is
    absent/unparseable. Per project decision M-D ships Python-only; this
    C++ leg is deferred until the C++ tree is clean (it is currently
    dirty under a separate concurrent task, so this test MUST NOT build
    the binary and MUST skip rather than fail on C++ unavailability).
    """
    if not CPP_BINARY.exists():
        pytest.skip(f"Autowiring_CPP not built at {CPP_BINARY}")

    k, L = 12, 2
    cl, tl, itl = 0.3, 0.5, 0.006
    g = _make_graph(shape, k, L=L)
    n = len(g._edge_list)
    rng = np.random.default_rng(3)
    layers = rng.integers(0, L, size=n).astype(float)

    positions = _positions_for(shape, k)
    pos_path = tmp_path / "positions.json"
    pos_path.write_text(
        json.dumps(
            {
                str(i): [float(positions[i][0]), float(positions[i][1])]
                for i in range(k)
            }
        )
    )
    fixed_path = tmp_path / "fixed_layers.json"
    # C++ requires an OBJECT {"layers": [...]}, not a bare list
    # (main.cpp ~423-427: rejects unless fl_j["layers"].is_array()).
    fixed_path.write_text(
        json.dumps({"layers": [int(round(v)) for v in layers]})
    )
    out_dir = tmp_path / "cpp_out"
    out_dir.mkdir()

    env = hermetic_env()
    env.update(
        {
            "SINIC_POSITIONS_JSON": str(pos_path),
            "SINIC_FIXED_LAYERS_JSON": str(fixed_path),
            "SINIC_SCHEMA_DIR": str(
                Path(__file__).resolve().parent.parent / "schema"
            ),
            "SINIC_VALIDATE_SCHEMA": "1",
            "OUTPUT_DIR": str(out_dir) + "/",
            "NUM_LAYERS": str(L),
            "CL_VALUES": str(cl),
            "TL_VALUES": str(tl),
            "ITL_VALUES": str(itl),
            "DUALSA_ITER": "1",  # ignored — fixed-layers path skips optimize
        }
    )
    proc = subprocess.run(
        [str(CPP_BINARY)], env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        pytest.skip(
            f"C++ subprocess returned {proc.returncode} (C++ tree is "
            f"deferred/dirty under a concurrent task; M-D ships "
            f"Python-only).\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )

    sub_path = (
        out_dir / _output_subdir(cl, tl, itl, k) / "subgraphsdata.json"
    )
    if not sub_path.exists():
        pytest.skip(
            f"C++ subgraphsdata.json absent at {sub_path} (C++ leg "
            f"deferred; M-D ships Python-only).\nstdout:\n{proc.stdout}"
        )
    try:
        data = json.loads(sub_path.read_text())
    except (OSError, ValueError) as exc:
        pytest.skip(
            f"C++ subgraphsdata.json unparseable at {sub_path}: {exc} "
            f"(C++ leg deferred; M-D ships Python-only)."
        )

    cpp_total = _cpp_intralayer_from_subgraphsdata(data)
    harness = count_intralayer_geometric(g, layers)
    assert cpp_total == harness, (
        f"cross-backend mismatch [{shape}]: C++ subgraphsdata "
        f"Σ_i⌊Σcrossings/2⌋={cpp_total} vs harness counter={harness}"
    )
