"""M4 acceptance tests for the rank-3 crosstalk tensor (REFACTOR_GOALS.md §2-2).

Covers the M4 rows of the §6 matrix:

- **T3 — 4-node analytical sanity**: unit square + 2 diagonals share-layer
  case; closed-form crosstalk = ``0.5 * loss_intralayer_crosstalk`` for
  each of the 2 victims; assert < 0.1 dB error.
- **T3 (extension) — interlayer variant**: diagonals on adjacent layers
  → ``loss_interlayer_crosstalk`` used.
- **T4 — crosstalk off ≡ pre-M4**: both coefficients None/0 → tensor
  values are all None; main-path ``loss_function`` is unchanged.
- **Hop / threshold pruning**: ``max_hops=0`` produces empty tensor;
  pushing threshold above signal level prunes everything.
- **Schema validation**: written ``subgraphsdata.json`` carries crosstalk
  and validates against ``code/schema/subgraphsdata.schema.json`` v2.0.
- **PlotData passthrough**: ``general_params["crosstalk"]`` is populated
  from a JSON written by the M4 path.
- **Heatmap plot**: ``plot_crosstalk_heatmap`` writes the expected file
  and consumes both view modes.
- **Non-adjacent layer pairs are NOT coupled** (§2-3 目标 D).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from routing_py_rebuild.api import make_graph, run_optimization
from routing_py_rebuild.crosstalk import compute_crosstalk_tensor
from routing_py_rebuild.plotting import PlotData, plot_crosstalk_heatmap


_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"
_SUBGRAPHS_SCHEMA = _SCHEMA_DIR / "subgraphsdata.schema.json"


def _unit_square_positions() -> dict[int, tuple[float, float]]:
    """Standard 4-node unit-square layout used by all 4-node analyses.

    Node indices follow the cyclic-convex convention used in the rest of
    the codebase: 0 = top-left, 1 = top-right, 2 = bottom-right,
    3 = bottom-left. The diagonals (0,2) and (1,3) cross at (0.5, 0.5),
    which is exactly the center — making the parametric intersection
    ``t = 0.5`` on both edges and the analytical crosstalk closed form.
    """
    return {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 0.0)}


def _build_4node_graph(
    *,
    L: int = 2,
    edge_coupler_layer: int = 0,
    perimeter_layer: int = 0,
    loss_intralayer_crosstalk: float | None = 1e-4,
    loss_interlayer_crosstalk: float | None = 1e-5,
    loss_crossing: float = 0.3,
    loss_taper: float = 0.0,
    loss_interlayercrossing: float = 0.0,
):
    """Make a 4-node graph with the unit-square layout."""
    return make_graph(
        k=4,
        positions=_unit_square_positions(),
        output_dir=None,
        L=L,
        edge_coupler_layer=edge_coupler_layer,
        perimeter_layer=perimeter_layer,
        loss_crossing=loss_crossing,
        loss_taper=loss_taper,
        loss_interlayercrossing=loss_interlayercrossing,
        loss_intralayer_crosstalk=loss_intralayer_crosstalk,
        loss_interlayer_crosstalk=loss_interlayer_crosstalk,
    )


# ---------------------------------------------------------------------------
# T3 — 4-node analytical sanity (intralayer)
# ---------------------------------------------------------------------------


def test_4node_intralayer_analytical_sanity() -> None:
    """REFACTOR_GOALS.md §2-2 验收第 3 条: 4 nodes on unit square corners,
    both diagonals on the same layer. Analytical expected crosstalk:

      - main path = diagonal (0,2): 1 crossing with (1,3) at center
        → leak total = 1.0 * loss_intralayer_crosstalk
        → split 50/50 → each of the two halves of (1,3) receives 0.5 * coef
        → arrivals: node 1 (one end of (1,3)), node 3 (other end)
        → dB = 10 * log10(0.5 * 1e-4) = −43.0103 dB

    Spec asks for ``< 0.1 dB``. We assert tighter (< 1e-6 dB) because the
    closed form is exact at this layout — no rounding/path-length
    approximations are involved.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    # All edges on layer 0 (the only available layer for L=2/ecl=0
    # baseline since the perimeter pin is also at 0). For diagonals to
    # cross intralayer, both diagonals must also be at layer 0 — which is
    # automatic since apply_optimization_result(0...) puts them there.
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))

    ct = compute_crosstalk_tensor(g)

    expected_db = 10.0 * math.log10(0.5 * 1e-4)

    values = ct["values"]
    # Source s=0, destination d=2 (the diagonal). Victims t=1 and t=3.
    for s, d, victims in [
        (0, 2, [1, 3]),
        (2, 0, [1, 3]),
        (1, 3, [0, 2]),
        (3, 1, [0, 2]),
    ]:
        for t in victims:
            v = values[s][d][t]
            assert v is not None, f"crosstalk[{s}][{d}][{t}] is None"
            assert abs(v - expected_db) < 1e-6, (
                f"crosstalk[{s}][{d}][{t}]={v:.6f} dB; expected "
                f"{expected_db:.6f} dB (±1e-6)"
            )


def test_4node_intralayer_perimeter_paths_have_no_crosstalk() -> None:
    """Perimeter edges (0,1), (1,2), (2,3), (0,3) don't cross any other
    edge → main paths along them never hit a junction → no leakage at
    all. The corresponding tensor slices are entirely None.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))

    ct = compute_crosstalk_tensor(g)
    values = ct["values"]

    for s, d in [(0, 1), (1, 2), (2, 3), (0, 3),
                 (1, 0), (2, 1), (3, 2), (3, 0)]:
        for t in range(4):
            assert values[s][d][t] is None, (
                f"perimeter path {s}->{d} should not leak to {t}; "
                f"got {values[s][d][t]}"
            )


# ---------------------------------------------------------------------------
# T3 (extension) — 4-node analytical sanity (interlayer)
# ---------------------------------------------------------------------------


def test_4node_interlayer_analytical_sanity() -> None:
    """Repeat the 4-node analysis with the two diagonals on **adjacent**
    layers — should now use ``loss_interlayer_crosstalk``.

    Setup: L=3, perimeter pinned to layer 1 (the coupler layer), then
    manually drop one diagonal into layer 0 and lift the other into
    layer 2 by writing directly onto ``graph.G``. The diagonals are
    layers 0 and 2 → |Δlayer| = 2, **NOT adjacent** → leak coefficient
    should be 0 per §2-3 目标 D. Confirm tensor is all None.

    Then redo with the two diagonals on layers 1 and 0 → |Δlayer| = 1
    → ``loss_interlayer_crosstalk = 1e-5`` should apply, expected dB
    is 10*log10(0.5 * 1e-5) = −53.0103.
    """
    # Non-adjacent (Δlayer=2): no coupling.
    g = _build_4node_graph(L=3, edge_coupler_layer=1, perimeter_layer=1)
    g.build_crossings_index()
    # Manually paint: diagonals on 0 and 2 (non-adjacent), perimeter on 1.
    diag_02_idx = g._edge_index[(0, 2)]
    diag_13_idx = g._edge_index[(1, 3)]
    layers = np.full(len(g._edge_list), 1, dtype=float)  # default = perimeter layer
    layers[diag_02_idx] = 0
    layers[diag_13_idx] = 2
    g.apply_optimization_result(layers)

    ct_nonadj = compute_crosstalk_tensor(g)
    # 0->2 has 1 crossing with diag (1,3) at layers 0 vs 2 → coef=0.
    assert ct_nonadj["values"][0][2][1] is None
    assert ct_nonadj["values"][0][2][3] is None

    # Adjacent (Δlayer=1): leak with loss_interlayer_crosstalk.
    layers[diag_02_idx] = 1  # diag (0,2) on layer 1
    layers[diag_13_idx] = 0  # diag (1,3) on layer 0  → adjacent.
    g.apply_optimization_result(layers)
    ct_adj = compute_crosstalk_tensor(g)
    expected_db = 10.0 * math.log10(0.5 * 1e-5)

    for s, d, victims in [(0, 2, [1, 3]), (2, 0, [1, 3])]:
        for t in victims:
            v = ct_adj["values"][s][d][t]
            assert v is not None, (
                f"interlayer crosstalk[{s}][{d}][{t}] is None; expected ~{expected_db}"
            )
            assert abs(v - expected_db) < 1e-6, (
                f"crosstalk[{s}][{d}][{t}]={v:.6f} dB; expected "
                f"{expected_db:.6f} dB"
            )


# ---------------------------------------------------------------------------
# T4 — crosstalk off ≡ pre-M4
# ---------------------------------------------------------------------------


def test_crosstalk_off_tensor_is_all_none() -> None:
    """``loss_intralayer_crosstalk = loss_interlayer_crosstalk = 0`` makes
    every entry None.  REFACTOR_GOALS.md §2-2 验收第 2 条 + T4.
    """
    g = _build_4node_graph(
        loss_intralayer_crosstalk=0.0,
        loss_interlayer_crosstalk=0.0,
    )
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    ct = compute_crosstalk_tensor(g)
    for s in range(4):
        for d in range(4):
            for t in range(4):
                assert ct["values"][s][d][t] is None


def test_crosstalk_coefficients_none_tensor_is_all_none() -> None:
    """``loss_intralayer_crosstalk = loss_interlayer_crosstalk = None``
    (the M3 / pre-M4 default for crosstalk-unaware callers) → tensor is
    still all None and the engine short-circuits without touching the
    geometric pass.
    """
    g = _build_4node_graph(
        loss_intralayer_crosstalk=None,
        loss_interlayer_crosstalk=None,
    )
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    ct = compute_crosstalk_tensor(g)
    for s in range(4):
        for d in range(4):
            for t in range(4):
                assert ct["values"][s][d][t] is None


def test_loss_function_independent_of_crosstalk_coefficients() -> None:
    """REFACTOR_GOALS.md §2-2 验收第 2 条 / T4: the loss model is
    unchanged when crosstalk coefficients are non-zero. The crosstalk
    engine is a pure analytical pass, never feeding back into the loss
    function. Confirm bit-exact loss equality between (coefs=None) and
    (coefs=1e-4 / 1e-5).
    """
    g_off = make_graph(
        k=12, output_dir=None, L=2, edge_coupler_layer=0,
        loss_intralayer_crosstalk=None, loss_interlayer_crosstalk=None,
    )
    g_on = make_graph(
        k=12, output_dir=None, L=2, edge_coupler_layer=0,
        loss_intralayer_crosstalk=1e-4, loss_interlayer_crosstalk=1e-5,
    )
    g_off.build_crossings_index()
    g_on.build_crossings_index()
    n_edges = len(g_off._edge_list)
    rng = np.random.default_rng(seed=2026)
    layers = rng.integers(0, 2, size=n_edges).astype(float)
    assert g_off.loss_function(layers) == g_on.loss_function(layers)


# ---------------------------------------------------------------------------
# Hop / threshold pruning
# ---------------------------------------------------------------------------


def test_max_hops_zero_produces_empty_tensor() -> None:
    """``max_hops=0`` disables the recursion entirely — even if
    coefficients are nonzero, no orthogonal branches are spawned, so the
    tensor is all None (main-path arrivals land on the diagonal slot at
    t=d which we always mask).
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    ct = compute_crosstalk_tensor(g, max_hops=0)
    for s in range(4):
        for d in range(4):
            for t in range(4):
                assert ct["values"][s][d][t] is None


def test_threshold_above_signal_prunes_everything() -> None:
    """If threshold_db is set above the only achievable signal level,
    every entry should be pruned to None. The 4-node intralayer setup
    delivers ~−43 dB; a threshold of −30 dB cuts them all.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    ct = compute_crosstalk_tensor(g, threshold_db=-30.0)
    for s in range(4):
        for d in range(4):
            for t in range(4):
                assert ct["values"][s][d][t] is None


def test_threshold_below_signal_keeps_signal() -> None:
    """Inverse of above: −60 dB threshold (the default) is well below
    the −43 dB signal, so all valid arrivals survive.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    ct = compute_crosstalk_tensor(g, threshold_db=-60.0)
    # We computed −43.01 in the sanity test; -60 dB threshold keeps it.
    assert ct["values"][0][2][1] is not None
    assert ct["values"][0][2][3] is not None


# ---------------------------------------------------------------------------
# Validation: loss_crossing >= 1 raises (model assumption)
# ---------------------------------------------------------------------------


def test_loss_crossing_at_or_above_one_raises() -> None:
    """The crosstalk model treats ``loss_crossing`` as a fractional power
    loss (T_main = 1 - loss_crossing). A value ≥ 1 produces
    non-physical transmission ≤ 0; the engine should raise rather than
    silently emit an all-None tensor.
    """
    g = _build_4node_graph(loss_crossing=1.0)
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    # opus-review P2-6: tightened matcher anchored on the specific
    # transmission-range guidance so a regression that changed the error
    # to a different ValueError about loss_crossing would still fail.
    with pytest.raises(ValueError, match="produces transmission"):
        compute_crosstalk_tensor(g)


# ---------------------------------------------------------------------------
# Negative max_hops / non-finite threshold
# ---------------------------------------------------------------------------


def test_negative_max_hops_raises() -> None:
    g = _build_4node_graph()
    # opus-review P2-6: anchor on the specific ">= 0" clause.
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_crosstalk_tensor(g, max_hops=-1)


def test_non_finite_threshold_db_raises() -> None:
    """opus-review P2-5: ``threshold_db = +inf`` / ``-inf`` slipped past
    the engine and only blew up at ``json.dump(..., allow_nan=False)``.
    M4 review fix raises locally at the engine entry.
    """
    g = _build_4node_graph()
    with pytest.raises(ValueError, match="threshold_db must be finite"):
        compute_crosstalk_tensor(g, threshold_db=float("inf"))
    with pytest.raises(ValueError, match="threshold_db must be finite"):
        compute_crosstalk_tensor(g, threshold_db=float("-inf"))
    with pytest.raises(ValueError, match="threshold_db must be finite"):
        compute_crosstalk_tensor(g, threshold_db=float("nan"))


# ---------------------------------------------------------------------------
# Tensor schema + JSON validation
# ---------------------------------------------------------------------------


def test_compute_crosstalk_payload_shape_and_metadata() -> None:
    """The returned payload must match the §3-1 / schema contract."""
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    ct = compute_crosstalk_tensor(g, max_hops=3, threshold_db=-60.0)

    assert ct["unit"] == "dB"
    assert ct["shape"] == [4, 4, 4]
    assert ct["coherence_model"] == "incoherent_v1"
    assert ct["polarization"] == "TE0_only"
    assert ct["symmetric"] is False
    assert ct["diagonal_convention"] == "NaN"
    assert ct["max_hops"] == 3
    assert ct["threshold_db"] == -60.0
    assert ct["loss_intralayer_crosstalk"] == 1e-4
    assert ct["loss_interlayer_crosstalk"] == 1e-5
    assert len(ct["values"]) == 4
    assert all(len(s) == 4 for s in ct["values"])
    assert all(len(d) == 4 for s in ct["values"] for d in s)


def test_written_json_carries_crosstalk_and_validates(tmp_path: Path) -> None:
    """End-to-end: ``run_optimization`` with non-zero coefficients writes
    a v2.0 ``subgraphsdata.json`` whose top-level ``crosstalk`` field
    passes the schema validator.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence wpl-experimental etc.
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            plot=False,
            run_loss_analysis=False,
            save_json=True,
        )

    json_path = res["json_path"]
    with open(json_path) as f:
        data = json.load(f)
    with open(_SUBGRAPHS_SCHEMA) as f:
        schema = json.load(f)
    jsonschema.validate(instance=data, schema=schema)

    assert data["schema_version"] == "2.0"
    assert "crosstalk" in data
    ct = data["crosstalk"]
    assert ct["unit"] == "dB"
    assert ct["shape"] == [12, 12, 12]
    assert ct["coherence_model"] == "incoherent_v1"
    assert ct["polarization"] == "TE0_only"
    # Values are 12*12*12 nested list; at least the structural shape
    # holds (real values vs None are runtime-dependent).
    assert len(ct["values"]) == 12
    assert all(len(s) == 12 for s in ct["values"])
    assert all(len(d) == 12 for s in ct["values"] for d in s)


def test_no_crosstalk_when_coefficients_zero(tmp_path: Path) -> None:
    """When both crosstalk coefficients are None (M3 default), the
    ``crosstalk`` key is absent from the JSON. v2.0 schema permits this
    (the field is optional even in v2.0; M3 files do not emit it).
    """
    res = run_optimization(
        k=12,
        optimizer="dual_annealing",
        maxiter=3,
        seed=42,
        output_dir=str(tmp_path),
        L=2,
        edge_coupler_layer=0,
        # loss_*_crosstalk default to None
        plot=False,
        run_loss_analysis=False,
        save_json=True,
    )
    with open(res["json_path"]) as f:
        data = json.load(f)
    assert "crosstalk" not in data
    assert res["crosstalk"] is None


def test_compute_crosstalk_false_skips_analysis(tmp_path: Path) -> None:
    """Explicit opt-out: ``compute_crosstalk=False`` skips the engine
    even when coefficients are non-zero. Useful for benchmark runs.
    """
    res = run_optimization(
        k=12,
        optimizer="dual_annealing",
        maxiter=3,
        seed=42,
        output_dir=str(tmp_path),
        L=2,
        edge_coupler_layer=0,
        loss_intralayer_crosstalk=1e-4,
        loss_interlayer_crosstalk=1e-5,
        compute_crosstalk=False,
        plot=False,
        run_loss_analysis=False,
        save_json=True,
    )
    assert res["crosstalk"] is None
    with open(res["json_path"]) as f:
        data = json.load(f)
    assert "crosstalk" not in data


# ---------------------------------------------------------------------------
# PlotData passthrough
# ---------------------------------------------------------------------------


def test_plotdata_from_json_surfaces_crosstalk(tmp_path: Path) -> None:
    """After an M4 write, ``PlotData.from_json`` exposes the tensor on
    ``general_params["crosstalk"]`` (opus-review-3 P2-A passthrough,
    now actually used in M4).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            plot=False,
            run_loss_analysis=False,
            save_json=True,
        )
    pd = PlotData.from_json(res["json_path"])
    ct = pd.general_params.get("crosstalk")
    assert ct is not None
    assert ct["unit"] == "dB"
    assert ct["shape"] == [12, 12, 12]


def test_plotdata_from_graph_carries_crosstalk_when_skip_json(
    tmp_path: Path,
) -> None:
    """When ``save_json=False`` but plotting/analysis is on,
    ``run_optimization`` builds ``PlotData`` via ``from_graph`` and must
    still surface the crosstalk tensor on the snapshot (otherwise the
    in-memory pipeline diverges from the JSON pipeline).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            plot=False,
            run_loss_analysis=True,  # triggers from_graph path
            save_json=False,
        )
    pd = res["plot_data"]
    assert pd is not None
    ct = pd.general_params.get("crosstalk")
    assert ct is not None
    assert ct["shape"] == [12, 12, 12]


# ---------------------------------------------------------------------------
# Heatmap plot smoke tests
# ---------------------------------------------------------------------------


def test_plot_crosstalk_heatmap_worst_over_d(tmp_path: Path) -> None:
    """Worst-over-d view writes a PDF whose existence we verify."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            plot=False,
            run_loss_analysis=False,
            save_json=True,
        )
    pd = PlotData.from_json(res["json_path"])
    out = plot_crosstalk_heatmap(pd, view="worst_over_d")
    assert Path(out).exists()
    assert Path(out).suffix == ".pdf"


def test_plot_crosstalk_heatmap_specific_d(tmp_path: Path) -> None:
    """``view='for_specific_d'`` requires ``d_node``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            plot=False,
            run_loss_analysis=False,
            save_json=True,
        )
    pd = PlotData.from_json(res["json_path"])
    out = plot_crosstalk_heatmap(pd, view="for_specific_d", d_node=5)
    assert Path(out).exists()


def test_plot_crosstalk_heatmap_specific_d_missing_node_raises(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            plot=False,
            run_loss_analysis=False,
            save_json=True,
        )
    pd = PlotData.from_json(res["json_path"])
    with pytest.raises(ValueError, match="d_node"):
        plot_crosstalk_heatmap(pd, view="for_specific_d")


def test_plot_crosstalk_heatmap_unknown_view_raises(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            plot=False,
            run_loss_analysis=False,
            save_json=True,
        )
    pd = PlotData.from_json(res["json_path"])
    with pytest.raises(ValueError, match="Unknown view"):
        plot_crosstalk_heatmap(pd, view="bogus")


def test_plot_crosstalk_heatmap_without_tensor_raises(tmp_path: Path) -> None:
    """A PlotData built from an M3-style JSON (no crosstalk) must refuse
    to render the heatmap with a clear error.
    """
    res = run_optimization(
        k=12,
        optimizer="dual_annealing",
        maxiter=3,
        seed=42,
        output_dir=str(tmp_path),
        L=2,
        edge_coupler_layer=0,
        # default: no crosstalk
        plot=False,
        run_loss_analysis=False,
        save_json=True,
    )
    pd = PlotData.from_json(res["json_path"])
    with pytest.raises(ValueError, match="no crosstalk tensor"):
        plot_crosstalk_heatmap(pd, view="worst_over_d")


# ---------------------------------------------------------------------------
# Run report contains the new crosstalk phase + config fields
# ---------------------------------------------------------------------------


def test_run_report_carries_crosstalk_phase_and_config(tmp_path: Path) -> None:
    """When ``collect_statistics=True`` the run report records the
    crosstalk analysis as a dedicated phase, and its config block
    mirrors the new crosstalk knobs (§3-2 + M4).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            crosstalk_max_hops=3,
            crosstalk_threshold_db=-60.0,
            plot=False,
            run_loss_analysis=False,
            save_json=True,
            collect_statistics=True,
        )
    report = json.loads(Path(res["report_path"]).read_text())

    timings = report["timings"]
    # Phase name is "crosstalk_analysis"; the recorder appends "_ms".
    assert "crosstalk_analysis_ms" in timings
    assert timings["crosstalk_analysis_ms"] >= 0

    config = report["config"]
    assert config["crosstalk_max_hops"] == 3
    assert config["crosstalk_threshold_db"] == -60.0
    assert config["crosstalk_include_in_loss"] is False


def test_run_report_no_crosstalk_phase_when_disabled(tmp_path: Path) -> None:
    """Without crosstalk coefficients, the crosstalk_analysis phase
    timing should be absent — we don't pay the PhaseTimer overhead at
    all when the engine is short-circuited.
    """
    res = run_optimization(
        k=12,
        optimizer="dual_annealing",
        maxiter=3,
        seed=42,
        output_dir=str(tmp_path),
        L=2,
        edge_coupler_layer=0,
        plot=False,
        run_loss_analysis=False,
        save_json=True,
        collect_statistics=True,
    )
    report = json.loads(Path(res["report_path"]).read_text())
    assert "crosstalk_analysis_ms" not in report["timings"]


# ---------------------------------------------------------------------------
# Dual-review fixes (2026-05-14): collect_statistics-only path + shape
# validation + determinism + mixed adjacency + from_graph parity
# ---------------------------------------------------------------------------


def test_collect_statistics_only_path_still_runs_crosstalk(
    tmp_path: Path,
) -> None:
    """opus + codex P1: ``collect_statistics=True`` + ``save_json=False``
    + ``plot=False`` + ``run_loss_analysis=False`` is a real caller mode
    (benchmark scripts that only want trace + timings). Pre-fix the
    crosstalk pass was silently dropped — same class of bug as the M3
    P0-1 aggregate-stats hole. Post-fix the third branch in
    ``run_optimization`` mirrors the other two: crosstalk is computed,
    ``crosstalk_analysis_ms`` is in the timings, and ``res["crosstalk"]``
    is populated.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_optimization(
            k=12,
            optimizer="dual_annealing",
            maxiter=3,
            seed=42,
            output_dir=str(tmp_path),
            L=2,
            edge_coupler_layer=0,
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            compute_crosstalk=True,
            plot=False,
            run_loss_analysis=False,
            save_json=False,
            collect_statistics=True,
        )
    assert res["crosstalk"] is not None, (
        "P1 regression: collect_statistics-only path dropped the crosstalk "
        "payload before the dual-review fix"
    )
    assert res["crosstalk"]["shape"] == [12, 12, 12]
    report = json.loads(Path(res["report_path"]).read_text())
    assert "crosstalk_analysis_ms" in report["timings"], (
        "P1 regression: collect_statistics-only path didn't emit the "
        "crosstalk_analysis phase timing"
    )


def test_writer_rejects_crosstalk_shape_mismatch(tmp_path: Path) -> None:
    """codex P1 / opus P2-1: JSON Schema doesn't enforce
    ``len(values) == shape[i]``. The writer must catch a tensor whose
    declared shape doesn't match its payload. Synthesize the bad case
    by feeding a hand-built ``crosstalk`` dict through
    ``save_subgraphs_to_json``.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    g.analyze_loss()

    bad_payload = {
        "unit": "dB",
        "shape": [4, 4, 4],  # lies — values is 3×3×3
        "coherence_model": "incoherent_v1",
        "polarization": "TE0_only",
        "symmetric": False,
        "diagonal_convention": "NaN",
        "max_hops": 3,
        "threshold_db": -60.0,
        "loss_intralayer_crosstalk": 1e-4,
        "loss_interlayer_crosstalk": 1e-5,
        "values": [[[None] * 3 for _ in range(3)] for _ in range(3)],
    }
    out = tmp_path / "bad_shape.json"
    with pytest.raises(ValueError, match="does not match shape"):
        g.save_subgraphs_to_json(
            str(out), skip_analysis=True, crosstalk=bad_payload
        )

    # And while we're here, verify a missing shape is also rejected.
    bad_payload2 = dict(bad_payload)
    bad_payload2.pop("shape")
    bad_payload2["values"] = [[[None] * 4 for _ in range(4)] for _ in range(4)]
    with pytest.raises(ValueError, match="shape must be a 3-element"):
        g.save_subgraphs_to_json(
            str(out), skip_analysis=True, crosstalk=bad_payload2
        )


def test_writer_accepts_valid_shape(tmp_path: Path) -> None:
    """Companion to the rejection test: a correctly-sized payload
    should pass the new validator and land on disk.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    g.analyze_loss()
    valid = compute_crosstalk_tensor(g)
    out = tmp_path / "good.json"
    g.save_subgraphs_to_json(str(out), skip_analysis=True, crosstalk=valid)
    assert out.exists()


def test_writer_rejects_inner_slab_or_leaf_row_mismatch(tmp_path: Path) -> None:
    """codex-rereview note: cover the two interior branches of
    ``_validate_crosstalk_payload_shape`` that didn't have dedicated
    tests — (a) inner slab is not a list / has wrong length, (b) leaf
    row is not a list / has wrong length.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    g.analyze_loss()

    out = tmp_path / "bad.json"

    base = {
        "unit": "dB",
        "shape": [4, 4, 4],
        "coherence_model": "incoherent_v1",
        "polarization": "TE0_only",
        "symmetric": False,
        "diagonal_convention": "NaN",
        "max_hops": 3,
        "threshold_db": -60.0,
        "loss_intralayer_crosstalk": 1e-4,
        "loss_interlayer_crosstalk": 1e-5,
    }

    # (a-1) Inner slab is not a list.
    bad_slab_type = dict(base)
    bad_slab_type["values"] = ["not a list"] + [
        [[None] * 4 for _ in range(4)] for _ in range(3)
    ]
    with pytest.raises(ValueError, match=r"values\[0\] length"):
        g.save_subgraphs_to_json(
            str(out), skip_analysis=True, crosstalk=bad_slab_type
        )

    # (a-2) Inner slab list with wrong length.
    bad_slab_len = dict(base)
    bad_slab_len["values"] = [
        [[None] * 4 for _ in range(3)]  # only 3 rows, expected 4
    ] + [[[None] * 4 for _ in range(4)] for _ in range(3)]
    with pytest.raises(ValueError, match=r"values\[0\] length"):
        g.save_subgraphs_to_json(
            str(out), skip_analysis=True, crosstalk=bad_slab_len
        )

    # (b-1) Leaf row is not a list.
    bad_leaf_type = dict(base)
    bad_leaf_type["values"] = [
        ["not a list"] + [[None] * 4 for _ in range(3)]
    ] + [[[None] * 4 for _ in range(4)] for _ in range(3)]
    with pytest.raises(ValueError, match=r"values\[0\]\[0\] length"):
        g.save_subgraphs_to_json(
            str(out), skip_analysis=True, crosstalk=bad_leaf_type
        )

    # (b-2) Leaf row with wrong length.
    bad_leaf_len = dict(base)
    bad_leaf_len["values"] = [
        [[None] * 3] + [[None] * 4 for _ in range(3)]  # first row 3 not 4
    ] + [[[None] * 4 for _ in range(4)] for _ in range(3)]
    with pytest.raises(ValueError, match=r"values\[0\]\[0\] length"):
        g.save_subgraphs_to_json(
            str(out), skip_analysis=True, crosstalk=bad_leaf_len
        )


def test_writer_rejects_bool_shape_entries(tmp_path: Path) -> None:
    """opus-rereview P2 nit: ``bool`` is a subclass of ``int`` in Python.
    A hand-built payload from an external (M5 C++) writer with
    ``shape = [True, True, True]`` (e.g. a typo or JSON deserialization
    artifact) would have silently passed before the explicit bool
    rejection landed. Cover the guard explicitly.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    g.analyze_loss()

    bad = {
        "unit": "dB",
        "shape": [True, True, True],  # bools, not ints
        "coherence_model": "incoherent_v1",
        "polarization": "TE0_only",
        "symmetric": False,
        "diagonal_convention": "NaN",
        "max_hops": 3,
        "threshold_db": -60.0,
        "loss_intralayer_crosstalk": 1e-4,
        "loss_interlayer_crosstalk": 1e-5,
        "values": [[[None]]],
    }
    out = tmp_path / "bool_shape.json"
    with pytest.raises(ValueError, match="shape must be a 3-element"):
        g.save_subgraphs_to_json(str(out), skip_analysis=True, crosstalk=bad)


def test_compute_crosstalk_tensor_is_deterministic() -> None:
    """opus P2-9: same graph state → identical tensor across repeated
    calls. Incoherent power addition is mathematically commutative; this
    test guards against any future iteration-order leakage from the
    DFS stack or per-edge sort ordering.
    """
    g = _build_4node_graph()
    g.build_crossings_index()
    n_edges = len(g._edge_list)
    g.apply_optimization_result(np.zeros(n_edges, dtype=float))
    ct1 = compute_crosstalk_tensor(g)
    ct2 = compute_crosstalk_tensor(g)
    # Exact equality on the values lists — no floating-point slack.
    assert ct1["values"] == ct2["values"]


def test_compute_crosstalk_tensor_l3_mixed_adjacency() -> None:
    """opus P2-9 / codex coverage gap: L=3 with a mixed intra+inter
    path tests both coefficients on the same (s,d) walk. Build a 4-node
    layout with one diagonal on layer 1 and the other on layer 0
    (adjacent) — only the interlayer coefficient should fire. Then
    redo with both diagonals on layer 1 — only intralayer fires. The
    two paths must produce different (and correct) dB values.
    """
    g = _build_4node_graph(L=3, edge_coupler_layer=1, perimeter_layer=1)
    g.build_crossings_index()

    diag_02 = g._edge_index[(0, 2)]
    diag_13 = g._edge_index[(1, 3)]
    layers = np.full(len(g._edge_list), 1, dtype=float)

    # Interlayer (Δ=1)
    layers[diag_02] = 1
    layers[diag_13] = 0
    g.apply_optimization_result(layers)
    ct_inter = compute_crosstalk_tensor(g)
    expected_inter = 10.0 * math.log10(0.5 * 1e-5)
    assert ct_inter["values"][0][2][1] is not None
    assert abs(ct_inter["values"][0][2][1] - expected_inter) < 1e-6

    # Intralayer (both on layer 1)
    layers[diag_02] = 1
    layers[diag_13] = 1
    g.apply_optimization_result(layers)
    ct_intra = compute_crosstalk_tensor(g)
    expected_intra = 10.0 * math.log10(0.5 * 1e-4)
    assert ct_intra["values"][0][2][1] is not None
    assert abs(ct_intra["values"][0][2][1] - expected_intra) < 1e-6

    # Sanity: the two regimes are distinct (~10 dB apart for default
    # 1e-4 vs 1e-5 coefficients).
    assert abs(ct_inter["values"][0][2][1] - ct_intra["values"][0][2][1]) > 5.0


def test_plotdata_from_graph_has_crosstalk_key_even_when_off(
    tmp_path: Path,
) -> None:
    """opus P2-2: ``from_graph`` and ``from_json`` should both expose
    ``params["crosstalk"]`` as a key (defaulting to None) so consumers
    can use plain attribute access without conditional branches. Pre-fix
    ``from_graph`` omitted the key entirely.
    """
    res = run_optimization(
        k=12,
        optimizer="dual_annealing",
        maxiter=3,
        seed=42,
        output_dir=str(tmp_path),
        L=2,
        edge_coupler_layer=0,
        # No crosstalk coefficients.
        plot=False,
        run_loss_analysis=True,
        save_json=False,
    )
    pd = res["plot_data"]
    assert pd is not None
    assert "crosstalk" in pd.general_params
    assert pd.general_params["crosstalk"] is None


def test_compute_crosstalk_skips_phase_a_when_short_circuited() -> None:
    """codex P2-5: when coefficients are zero or max_hops=0, the engine
    should not eagerly run Phase A. Verify by constructing a graph,
    calling ``compute_crosstalk_tensor`` with all-zero coefficients,
    and asserting the lazy Phase A index is still uninitialized.
    """
    g = _build_4node_graph(
        loss_intralayer_crosstalk=0.0,
        loss_interlayer_crosstalk=0.0,
    )
    # Don't call build_crossings_index() — leave Phase A lazy.
    assert g._crossing_pairs is None
    ct = compute_crosstalk_tensor(g)
    # Phase A is still not built — the early-exit short-circuited.
    assert g._crossing_pairs is None
    # And the tensor is empty as expected.
    assert all(v is None for s in ct["values"] for d in s for v in d)


def test_compute_crosstalk_skips_phase_a_when_max_hops_zero() -> None:
    """Same as above, for the ``max_hops=0`` early exit."""
    g = _build_4node_graph()  # has nonzero coefficients
    assert g._crossing_pairs is None
    ct = compute_crosstalk_tensor(g, max_hops=0)
    assert g._crossing_pairs is None
    assert all(v is None for s in ct["values"] for d in s for v in d)
