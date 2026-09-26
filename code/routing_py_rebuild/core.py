"""Core graph model for SiN photonic interconnection networks.

`SiNInterconnectionGraph` owns the complete graph, edge crossing detection,
loss accounting, layer subgraph decomposition and JSON serialization. It does
NOT perform optimization or plotting — those live in `optimizers/` and
`plotting/` respectively.

M3 (REFACTOR_GOALS.md §2-3) generalizes the graph to arbitrary layer count
``L`` with explicit ``edge_coupler_layer`` and ``perimeter_layer``. Loss
formula taper term becomes ``2 * |layer - edge_coupler_layer| * loss_taper``;
``waveguides_per_link`` is recorded on the graph and serialized into
``subgraphsdata.json`` v2.0 for downstream stats aggregation. Per-edge
``interlayercrossings_above`` / ``interlayercrossings_below`` counters are
populated alongside the legacy ``interlayercrossings`` so that v2.0 readers
have direction-resolved geometry. See §3-1 / §4 M3 附注.

Boundaries with the rest of the package:
  - Optimizers consume :meth:`loss_function` and end with
    :meth:`apply_optimization_result`.
  - Plotting consumes a :class:`plotting.PlotData` snapshot, built either
    from the self-contained JSON written by
    :meth:`save_subgraphs_to_json` or, as an escape hatch, from an
    already-analyzed graph via :meth:`plotting.PlotData.from_graph`. The
    plotting package never imports this module and never mutates a graph
    instance — the JSON file is the contract.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Iterable

import jsonschema
import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString


# REFACTOR_GOALS.md §3-3: write-time / read-time schema validation against
# ``code/schema/subgraphsdata.schema.json``. ``functools.lru_cache`` makes
# the first-load-wins memoization atomic at the function-call boundary so
# parallel pytest-xdist workers don't double-read the file
# (opus-review-3 P2-D).
from functools import lru_cache as _lru_cache

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"
_SUBGRAPHS_SCHEMA_PATH = _SCHEMA_DIR / "subgraphsdata.schema.json"


@_lru_cache(maxsize=1)
def _load_subgraphs_schema() -> dict:
    with open(_SUBGRAPHS_SCHEMA_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Default loss model, dB per event: a crossing between two links in the
# same layer (intralayer), one taper of a layer transition, and a crossing
# between links in different layers (interlayer). ``api.make_graph`` /
# ``api.run_optimization`` and the CLI take their defaults from here; the
# C++ harness (``include/sinic/graph.hpp``, ``src/main.cpp``) uses the
# same values.
# ---------------------------------------------------------------------------
DEFAULT_LOSS_CROSSING = 0.1
DEFAULT_LOSS_TAPER = 0.05
DEFAULT_LOSS_INTERLAYERCROSSING = 0.001


# ---------------------------------------------------------------------------
# M3 + post-M8 (2026-05-15): waveguides_per_link notice text — shared
# between __init__ and the v1.x→v2.0 loader fallback (PlotData) so the
# message stays consistent. REFACTOR_GOALS.md §4 M3 附注 "Warning 策略"
# + §7 Q-e "M8 后默认翻转 (2026-05-15)".
#
# Default flipped to wpl=1 so JSON output aligns with the paper's
# geometric crossing convention. The wpl=1 path no longer warns on
# every call; it emits a single informational notice per process via
# ``_emit_wpl_one_notice_once`` so users see the stats/loss-misalignment
# context once without stderr being spammed.
# ---------------------------------------------------------------------------
_WPL_ONE_INFO = (
    "waveguides_per_link=1 (default): stats use the geometric crossing "
    "convention (matches paper reporting). The per-edge loss formula keeps "
    "an implicit 2× from endpoint double-counting; pass "
    "waveguides_per_link=2 for the physical convention aligned with the "
    "loss model. See REFACTOR_GOALS.md §7 Q-e."
)
_WPL_EXPERIMENTAL_WARNING = (
    "waveguides_per_link not in {{1, 2}} is experimental: stats/loss "
    "semantics may diverge until §7 Q-e is decided. wpl={value}."
)

# opus-review-3 P2-E: physical photonic systems use ~1–8 waveguides per
# link; anything past 32 is almost certainly a user typo (e.g. confusing
# wpl with link bandwidth in Gbps). The cap rejects the value rather
# than emit an "experimental" warning that masks the typo.
_WPL_PHYSICAL_MAX = 32

# Phase A convexity detector: a boundary turn whose |sin| is at most this
# is a straight step. Side nodes computed by interpolation (triangle,
# polygon generators, user coordinates from trig) sit ~1e-17 (relative) off
# their line with either sign; without the tolerance those layouts missed
# the cyclic fast path and their crossing set depended on round-off. Mirrors
# ``COLLINEAR_TOL`` in ``routing_cpp_rebuild/src/crossings_cache.cpp``.
_COLLINEAR_TOL = 1e-9

# Process-wide one-time emit flag for the wpl=1 informational notice.
# Each module that validates wpl owns its own copy of this flag so that
# the "plotting must not import core" boundary (§1-3-c) stays intact;
# the test conftest resets both flags before each test for isolation.
_wpl_one_notice_emitted = False


def _emit_wpl_one_notice_once() -> None:
    """Emit the wpl=1 informational notice at most once per process.

    Uses ``warnings.warn`` (UserWarning) so existing ``pytest.warns``
    test mechanics keep working; the text is informational rather than
    alarming, and the one-time flag prevents stderr spam at the new
    default. Tests reset the flag via ``_reset_wpl_notice_for_tests``.
    """
    global _wpl_one_notice_emitted
    if _wpl_one_notice_emitted:
        return
    _wpl_one_notice_emitted = True
    warnings.warn(_WPL_ONE_INFO, stacklevel=4)


def _reset_wpl_notice_for_tests() -> None:
    """Reset the one-time flag. Test-only — invoked by ``conftest.py``."""
    global _wpl_one_notice_emitted
    _wpl_one_notice_emitted = False


def _validate_waveguides_per_link(wpl: int) -> int:
    """Raise on invalid wpl; emit info-once for wpl=1; warn for {3,4,...}.

    Mirrors the loader policy in REFACTOR_GOALS.md §4 M3 附注 "Warning
    策略" (post-2026-05-15 revision) so direct
    ``SiNInterconnectionGraph`` construction enforces the same contract
    as JSON load. wpl=1 is the documented v2.0 default since the
    M8-post flip; the notice is one-time per process to avoid spam.
    """
    if isinstance(wpl, bool) or not isinstance(wpl, (int, np.integer)):
        raise ValueError(
            f"waveguides_per_link must be a positive integer; "
            f"got {wpl!r} (type {type(wpl).__name__})."
        )
    wpl_int = int(wpl)
    if wpl_int <= 0:
        raise ValueError(
            f"waveguides_per_link must be >= 1; got {wpl_int}."
        )
    if wpl_int > _WPL_PHYSICAL_MAX:
        raise ValueError(
            f"waveguides_per_link={wpl_int} exceeds physical bound "
            f"{_WPL_PHYSICAL_MAX}; typical photonic interconnects use 1–8. "
            "If you really need a larger value, raise _WPL_PHYSICAL_MAX "
            "with documentation."
        )
    if wpl_int == 1:
        _emit_wpl_one_notice_once()
    elif wpl_int != 2:
        warnings.warn(
            _WPL_EXPERIMENTAL_WARNING.format(value=wpl_int), stacklevel=3
        )
    return wpl_int


# ---------------------------------------------------------------------------
# Exact 2-D orientation predicate (robust, fully vectorized).
# ---------------------------------------------------------------------------
#
# The Phase-A geometric fallback (:meth:`SiNInterconnectionGraph.
# _build_crossing_pairs_geometric`) decides crossings from the signs of four
# 2x2 orientation determinants. Computing those determinants with naive
# float64 arithmetic disagrees with shapely/GEOS's robust predicate on
# *near-degenerate* dense-boundary layouts (e.g. triangle / regular-polygon
# side points whose float64 representations are collinear only to within
# ~1e-17): the naive determinant rounds a tiny-but-nonzero value to exactly
# 0.0 for some triples (wrongly taking the collinear branch) and to a
# round-off-garbage nonzero for others (wrongly taking / skipping the proper
# branch). That violated the fallback's "bit-exact with ``edge_crosses``"
# contract and corrupted ``_crossing_pairs`` on triangle / polygon layouts.
# Those convex layouts now take the combinatorial fast path (their side
# runs count as straight within ``_COLLINEAR_TOL``); the exact predicate
# still decides every non-convex layout, near-collinear nodes included.
#
# Every float64 is an exact dyadic rational, so for finite, normal-range
# coordinates the sign of the determinant is exactly computable.
# ``_orient2d_sign`` is an adaptive predicate: a cheap float determinant
# with Shewchuk's static error filter resolves the vast majority of
# entries, and only the (geometry-intrinsic) in-band remainder is
# recomputed by ``_orient2d_sign_exact``, which evaluates
#   orient2d(a,b,c) = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)
# via Knuth/Dekker error-free transforms (TwoSum / TwoDiff / Dekker
# TwoProduct). Both stages are fully vectorized (no per-pair Python, no
# ``fractions.Fraction``).
#
# Domain: the predicate is sign-correct for finite, bounded, normal-range
# float64 coordinates -- the O(1) regime every ``distribute_nodes_*``
# generator produces (validated against exact-rational arithmetic over an
# adversarial suite incl. exact-collinear and ULP-perturbed triples). It
# is NOT robust for subnormal-magnitude or overflow-magnitude
# coordinates, which no layout generator emits (the prior naive code was
# likewise non-robust there -- this strictly improves the in-domain
# behavior, not a regression). The exact branch is one-shot Phase-A work
# outside any optimizer hot loop: it is *slower* than the old non-robust
# code on the rare dense-boundary fallback (k=160 ~14 s vs ~2 s) but
# bounded and well within the §5 budget, while the §2-1-budgeted
# cyclic-convex fast path (which serves the SiN square) is untouched.
_DEKKER_SPLITTER = 134217729.0  # 2**27 + 1


def _two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knuth TwoSum: ``a + b == x + y`` exactly, ``x = fl(a + b)``."""
    x = a + b
    bv = x - a
    y = (a - (x - bv)) + (b - bv)
    return x, y


def _two_diff(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TwoDiff: ``a - b == x + y`` exactly, ``x = fl(a - b)``."""
    x = a - b
    bv = a - x
    y = (a - (x + bv)) + (bv - b)
    return x, y


def _two_product(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dekker TwoProduct: ``a * b == x + y`` exactly, ``x = fl(a * b)``.

    Uses the 2**27+1 splitter (no FMA dependency — ``numpy`` exposes no
    vectorized fused-multiply-add, so the split form is used unconditionally).
    """
    x = a * b
    c = _DEKKER_SPLITTER * a
    ah = c - (c - a)
    al = a - ah
    d = _DEKKER_SPLITTER * b
    bh = d - (d - b)
    bl = b - bh
    y = al * bl - (((x - ah * bh) - al * bh) - ah * bl)
    return x, y


def _orient2d_sign_exact(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> np.ndarray:
    """Sign of ``orient2d(a, b, c)``, vectorized -- exact in-domain.

    Returns an ``int8`` array (broadcast over the inputs) with ``+1`` if
    ``c`` lies left of the directed line ``a -> b``, ``-1`` if right, ``0``
    iff ``a``, ``b``, ``c`` are exactly collinear. Sign-correct for finite,
    bounded, normal-range float64 coordinates (the O(1) SiNIC layout
    domain; products stay well inside the no-overflow range). Not robust
    for subnormal- or overflow-magnitude inputs, which no position
    generator produces.

    This is the slow exact branch; :func:`_orient2d_sign` is the adaptive
    front end that only routes the (rare) filter-uncertain entries here.
    """
    ax, ay, bx, by, cx, cy = np.broadcast_arrays(
        np.asarray(ax, np.float64),
        np.asarray(ay, np.float64),
        np.asarray(bx, np.float64),
        np.asarray(by, np.float64),
        np.asarray(cx, np.float64),
        np.asarray(cy, np.float64),
    )
    # Exact coordinate differences -> 2-term expansions (hi, lo).
    acx_h, acx_l = _two_diff(ax, cx)
    acy_h, acy_l = _two_diff(ay, cy)
    bcx_h, bcx_l = _two_diff(bx, cx)
    bcy_h, bcy_l = _two_diff(by, cy)

    def _prod2(
        uh: np.ndarray, ul: np.ndarray, vh: np.ndarray, vl: np.ndarray
    ) -> list[np.ndarray]:
        """``(uh+ul)*(vh+vl)`` as a float expansion -- sign-accurate, not
        a fully-compressed exact expansion.

        ``ul``/``vl`` are the round-off of an exact difference: each is
        below the ulp of its hi part, so the ``ul*vl`` cross term is
        order-eps**2 of the leading product and is added (uncompensated)
        last. This is therefore a fast sign-accurate evaluation, not a
        provably-exact Shewchuk expansion-sum: for the O(1) SiNIC
        coordinate domain ``|ul*vl|`` is many orders below the minimum
        nonzero exact determinant, so the dropped compensation cannot
        flip the sign carried by the Neumaier-compensated accumulation in
        :func:`_orient2d_sign_exact` (verified: 0 sign disagreements vs
        exact rational over the adversarial suite).
        """
        p_hh, e_hh = _two_product(uh, vh)
        p_hl, e_hl = _two_product(uh, vl)
        p_lh, e_lh = _two_product(ul, vh)
        p_ll = ul * vl
        s, e1 = _two_sum(e_hh, p_hl)
        s, e2 = _two_sum(s, p_lh)
        s, e3 = _two_sum(s, e_hl)
        s, e4 = _two_sum(s, e_lh)
        s, e5 = _two_sum(s, p_ll)
        return [p_hh, s, e1, e2, e3, e4, e5]

    pos_terms = _prod2(acx_h, acx_l, bcy_h, bcy_l)
    neg_terms = _prod2(acy_h, acy_l, bcx_h, bcx_l)

    # Exact value = sum(pos_terms) - sum(neg_terms). Neumaier-compensated
    # accumulation retains the exact sign.
    s = np.zeros(pos_terms[0].shape, np.float64)
    comp = np.zeros_like(s)
    for t in pos_terms:
        s, e = _two_sum(s, t)
        comp = comp + e
    for t in neg_terms:
        s, e = _two_sum(s, -t)
        comp = comp + e
    total = s + comp

    sign = np.zeros(total.shape, np.int8)
    sign[total > 0] = 1
    sign[total < 0] = -1
    return sign


# Shewchuk orient2d static error filter constant: the naive determinant
# ``det = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)`` has rounding error bounded by
# ``(3 + 16*eps)*eps * (|ac_x*bc_y| + |ac_y*bc_x|)``. Whenever ``|det|``
# exceeds that bound the float sign equals the exact sign, so only the
# (geometry-intrinsic, typically small) in-band remainder needs the exact
# branch -- which is why the robust predicate stays within the §5 timing
# budget even though the exact branch is slower than the old non-robust
# code on the rare dense-boundary fallback (the §2-1-budgeted fast path is
# untouched).
_ORIENT2D_FILTER_C = (3.0 + 16.0 * float(np.finfo(np.float64).eps)) * float(
    np.finfo(np.float64).eps
)


def _orient2d_sign(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> np.ndarray:
    """Robust sign of ``orient2d(a, b, c)``, vectorized (adaptive).

    Sign-identical to :func:`_orient2d_sign_exact` for finite, bounded,
    normal-range float64 coordinates (the O(1) SiNIC layout domain).
    A cheap float determinant resolves the vast majority of entries via
    Shewchuk's static error filter; only the entries inside the
    uncertainty band are recomputed with the exact error-free-transform
    predicate, on the compacted subset (no wasted exact work).

    Returns an ``int8`` array (broadcast over the inputs): ``+1`` if ``c``
    is left of the directed line ``a -> b``, ``-1`` if right, ``0`` iff
    exactly collinear.
    """
    ax, ay, bx, by, cx, cy = np.broadcast_arrays(
        np.asarray(ax, np.float64),
        np.asarray(ay, np.float64),
        np.asarray(bx, np.float64),
        np.asarray(by, np.float64),
        np.asarray(cx, np.float64),
        np.asarray(cy, np.float64),
    )
    acx = ax - cx
    acy = ay - cy
    bcx = bx - cx
    bcy = by - cy
    left = acx * bcy
    right = acy * bcx
    det = left - right
    errbound = _ORIENT2D_FILTER_C * (np.abs(left) + np.abs(right))

    sign = np.atleast_1d(
        np.where(
            det > errbound,
            np.int8(1),
            np.where(det < -errbound, np.int8(-1), np.int8(0)),
        ).astype(np.int8)
    )

    # Boolean-mask indexing (not np.nonzero) so the routine also works for
    # 0-d / scalar inputs; the production call site passes 1-d arrays.
    band = np.atleast_1d(np.abs(det) <= errbound)
    if band.any():
        ax1, ay1, bx1, by1, cx1, cy1 = (
            np.atleast_1d(v) for v in (ax, ay, bx, by, cx, cy)
        )
        sign[band] = _orient2d_sign_exact(
            ax1[band], ay1[band], bx1[band], by1[band], cx1[band], cy1[band]
        )
    return sign.reshape(np.shape(det))


class SiNInterconnectionGraph:
    """Model a SiN photonic complete graph with multi-layer routing.

    Parameters
    ----------
    k : int
        Total node count.
    positions : dict[int, (x, y)]
        Node positions; keys must cover ``range(k)``.
    filepath : str | None
        Output directory. If given, a per-config subdirectory is created.
    L : int, default 2
        Number of physical SiN waveguide layers.
    edge_coupler_layer : int, default 0
        Layer at which the fiber-to-chip edge coupler sits. Each cross-layer
        hop costs ``2 * loss_taper`` (in + out taper). Default 0 preserves
        bit-exact compatibility with the pre-M3 single-coupler-at-layer-0
        baseline; for L>=3 the recommended value is ``L // 2`` (centered)
        per REFACTOR_GOALS.md §2-3 目标 B (M3 keeps the legacy default to
        avoid changing numbers for run_optimization(k=...) one-arg
        invocations; CLI/users wanting the centered default pass it
        explicitly).
    perimeter_layer : int | None
        Layer that the perimeter ring (``|u-v| ∈ {1, k-1}``) is pinned to.
        Defaults to ``edge_coupler_layer``; spec §2-3-B 补充 recommends
        pinning to the coupler layer so perimeter edges pay zero taper.
    layer_pitch_um : float, default 1.2
        Physical layer spacing in μm. Serialized but not consumed by the
        loss model directly (M3); used downstream for crosstalk audits.
    waveguides_per_link : int, default 1
        Parallel waveguide count per logical edge. The default is 1
        (geometric crossing convention, matches paper reporting) since
        the post-M8 flip on 2026-05-15. M3 only uses this in the stats
        aggregator (``run_report.json.summary.crosslayer_crossings_total``);
        the per-edge loss formula does not parameterize on it yet — see
        §7 Q-e. Pass ``waveguides_per_link=2`` for the physical
        convention aligned with the loss model's implicit 2×. wpl=1
        emits a one-time informational notice per process; wpl ∈
        {3, 4, ..., 32} emits an experimental warning; non-positive /
        non-int / > 32 raises.
    loss_crossing, loss_taper, loss_interlayercrossing : float, default 0.1, 0.05, 0.001
        Per-event loss coefficients — intralayer crossing, taper, and
        interlayer crossing (dB-equivalent integer-summed; see §2-2 for
        unit conventions). Defaults: ``DEFAULT_LOSS_*``.
    loss_intralayer_crosstalk, loss_interlayer_crosstalk : float | None
        Crosstalk leakage coefficients (fractional power per crossing).
        Recorded on the graph for downstream consumers — the per-edge
        loss formula does NOT consume them (M4's
        ``crosstalk_include_in_loss`` is reserved for a future
        milestone; per §7 Q-e). The M4 post-optimization engine in
        :mod:`.crosstalk` reads these values to build the rank-3
        crosstalk tensor.
    coherence_model, polarization : str
        Physical-model labels; serialized verbatim into JSON metadata.
    """

    def __init__(
        self,
        k: int,
        positions: dict,
        filepath: str | None = None,
        *,
        L: int = 2,
        edge_coupler_layer: int = 0,
        perimeter_layer: int | None = None,
        layer_pitch_um: float = 1.2,
        waveguides_per_link: int = 1,
        loss_crossing: float = DEFAULT_LOSS_CROSSING,
        loss_taper: float = DEFAULT_LOSS_TAPER,
        loss_interlayercrossing: float = DEFAULT_LOSS_INTERLAYERCROSSING,
        loss_intralayer_crosstalk: float | None = None,
        loss_interlayer_crosstalk: float | None = None,
        coherence_model: str = "incoherent_v1",
        polarization: str = "TE0_only",
    ):
        self.k = k
        self.positions = positions

        L_int = int(L)
        if L_int < 1:
            raise ValueError(f"L must be >= 1; got {L_int}.")
        self.L = L_int

        ecl = int(edge_coupler_layer)
        if not (0 <= ecl < self.L):
            raise ValueError(
                f"edge_coupler_layer must be in [0, L); got {ecl} with L={self.L}."
            )
        self.edge_coupler_layer = ecl

        if perimeter_layer is None:
            perimeter_layer = ecl
        pl = int(perimeter_layer)
        if not (0 <= pl < self.L):
            raise ValueError(
                f"perimeter_layer must be in [0, L); got {pl} with L={self.L}."
            )
        self.perimeter_layer = pl

        if float(layer_pitch_um) <= 0:
            raise ValueError(
                f"layer_pitch_um must be > 0; got {layer_pitch_um}."
            )
        self.layer_pitch_um = float(layer_pitch_um)

        self.waveguides_per_link = _validate_waveguides_per_link(waveguides_per_link)

        self.loss_crossing = loss_crossing
        self.loss_taper = loss_taper
        self.loss_interlayercrossing = loss_interlayercrossing
        self.loss_intralayer_crosstalk = loss_intralayer_crosstalk
        self.loss_interlayer_crosstalk = loss_interlayer_crosstalk
        self.coherence_model = coherence_model
        self.polarization = polarization

        if filepath is not None:
            self.filepath = (
                f"{filepath}/cl_{self.loss_crossing:.2f}_"
                f"tl_{self.loss_taper:.2f}_itl_{self.loss_interlayercrossing:.3f}_nodes_{int(k)}/"
            )
        else:
            self.filepath = None

        self.G = nx.complete_graph(k)
        self.sub_G: list[nx.Graph] = []
        self.total_crossings_of_sub_G: list[int] = []
        self.edge_cross_counts_of_sub_G: list[dict] = []

        self._G_Planar = nx.complete_graph(k)
        self.loss_analysis: dict = {}

        for graph in (self.G, self._G_Planar):
            nx.set_edge_attributes(graph, 0, "layer")
            nx.set_edge_attributes(graph, 0, "crossings")
            nx.set_edge_attributes(graph, 0, "loss")
            nx.set_edge_attributes(graph, 0, "interlayercrossings")
            # M3 split fields — REFACTOR_GOALS.md §3-1 注 1. Per-edge
            # ``_above`` / ``_below`` counters; the planar baseline keeps
            # them at zero (single-layer reference graph).
            nx.set_edge_attributes(graph, 0, "interlayercrossings_above")
            nx.set_edge_attributes(graph, 0, "interlayercrossings_below")

        # REFACTOR_GOALS.md M2 (§2-1) Phase A index. Built lazily on the first
        # call to ``loss_function`` / ``analyze_loss`` / ``build_crossings_index``
        # so that constructing a graph without optimizing does not pay the
        # one-time O(E^2) geometric pass.
        self._crossing_pairs: np.ndarray | None = None
        self._edge_list: list[tuple[int, int]] | None = None
        self._edge_index: dict[tuple[int, int], int] | None = None
        self._perimeter_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Crossing detection
    # ------------------------------------------------------------------
    def edge_crosses(self, e1: tuple, e2: tuple) -> bool:
        """True if the two edges intersect properly (not merely touching)."""
        line1 = LineString([self.positions[e1[0]], self.positions[e1[1]]])
        line2 = LineString([self.positions[e2[0]], self.positions[e2[1]]])

        bool_result = shapely.intersects(line1, line2)
        e1_pts = [self.positions[e1[0]], self.positions[e1[1]]]
        e2_pts = [self.positions[e2[0]], self.positions[e2[1]]]

        for pt in e1_pts:
            if pt in e2_pts:
                bool_result = False
        if shapely.intersects(line1, shapely.Point(*e2_pts[0])) and shapely.intersects(
            line1, shapely.Point(*e2_pts[1])
        ):
            bool_result = False
        if shapely.intersects(line2, shapely.Point(*e1_pts[0])) and shapely.intersects(
            line2, shapely.Point(*e1_pts[1])
        ):
            bool_result = False
        return bool_result

    def count_crossings_with_detail(self, graph: nx.Graph) -> tuple[int, dict]:
        """Count crossings within ``graph`` and stamp the per-edge result."""
        edges = list(graph.edges())
        num_crosses = 0
        edge_cross_count = {edge: 0 for edge in edges}

        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if self.edge_crosses(edges[i], edges[j]):
                    num_crosses += 1
                    edge_cross_count[edges[i]] += 1
                    edge_cross_count[edges[j]] += 1

        nx.set_edge_attributes(graph, edge_cross_count, "crossings")
        return num_crosses, edge_cross_count

    def count_interlayercrossings(
        self,
        graph_lo: nx.Graph,
        graph_hi: nx.Graph,
        *,
        reset: bool = False,
    ) -> None:
        """Stamp per-edge inter-layer crossing counts on a pair of adjacent
        layer subgraphs.

        For each crossing event between an edge ``e ∈ graph_lo`` and
        ``f ∈ graph_hi`` (``graph_lo`` is at the *lower* layer index):

        - ``e.interlayercrossings_above += 1`` (e sees an event with the
          layer above it)
        - ``f.interlayercrossings_below += 1`` (f sees an event with the
          layer below it)
        - ``e.interlayercrossings += 1`` and ``f.interlayercrossings += 1``
          (legacy single-counter, kept for v1.x readers — sum of above +
          below per-edge; full-graph sum is 2 × geometric events).

        REFACTOR_GOALS.md §3-1 注 1 + §4 M3 附注.

        Parameters
        ----------
        reset : bool, default False
            When True, zero out ``interlayercrossings`` / ``_above`` /
            ``_below`` on both graphs before counting. False is the
            multi-pair-friendly default — ``create_subgraphs`` iterates
            adjacent pairs and each call accumulates into the running
            totals (middle layers receive events from both their lower
            and upper neighbours).
        """
        if reset:
            for graph in (graph_lo, graph_hi):
                nx.set_edge_attributes(graph, 0, "interlayercrossings")
                nx.set_edge_attributes(graph, 0, "interlayercrossings_above")
                nx.set_edge_attributes(graph, 0, "interlayercrossings_below")

        for edge_lo in graph_lo.edges():
            for edge_hi in graph_hi.edges():
                if self.edge_crosses(edge_lo, edge_hi):
                    graph_lo.edges[edge_lo]["interlayercrossings"] += 1
                    graph_hi.edges[edge_hi]["interlayercrossings"] += 1
                    graph_lo.edges[edge_lo]["interlayercrossings_above"] += 1
                    graph_hi.edges[edge_hi]["interlayercrossings_below"] += 1

    # ------------------------------------------------------------------
    # Phase A — cached crossing topology (REFACTOR_GOALS.md §2-1)
    # ------------------------------------------------------------------
    def build_crossings_index(self) -> None:
        """Public, idempotent entry point for Phase A.

        Builds and caches the K_k crossing-pair index used by Phase B's hot
        loop. After the first call, subsequent calls are a no-op.

        Surface this separately from ``__init__`` so callers (e.g. the API
        orchestrator) can time the precompute independently of plain graph
        construction — emitted as ``initial_crossing_count_ms`` in
        ``run_report.json`` per §3-2.
        """
        self._ensure_crossings_ready()

    def _ensure_crossings_ready(self) -> None:
        """Lazy Phase A build: one-time crossing-pair index that fills
        ``_crossing_pairs`` / ``_edge_list`` / ``_edge_index`` /
        ``_perimeter_mask`` and stamps the all-on-one-layer baseline
        ``_G_Planar.crossings`` (kept here so ``analyze_loss`` /
        ``save_subgraphs_to_json`` still see populated crossings without
        another geometric pass).

        Two-tier algorithm:

        1. Fast path — cyclic-convex node layout (e.g. all SiN cases from
           :func:`distribute_nodes`): two chords in K_n cross iff their
           endpoints alternate around the polygon. The check reduces to a
           purely integer XOR test on node indices, which holds bit-exact
           for proper crossings AND for collinear partial overlaps that
           :meth:`edge_crosses` flags as crossings (e.g. same-side edges
           on the SiN square boundary). This path is what hits the
           §2-1 ``≤ 5 s`` budget at k=160.
        2. Fallback — arbitrary positions: pure-numpy vectorized
           orientation test, with closed-form handling of T-junctions and
           fully-collinear partial overlap so the result exactly
           reproduces :meth:`edge_crosses` without spawning shapely calls.

        Both paths produce the same set of crossing pairs for SiN-style
        layouts; the oracle test in ``tests/test_crossings_oracle.py``
        validates parity against the brute-force shapely implementation
        for both convex and non-convex position dictionaries.
        """
        if self._crossing_pairs is not None:
            return

        edges = list(self.G.edges())
        n_edges = len(edges)
        edges_arr = (
            np.array(edges, dtype=np.int64) if n_edges else np.zeros((0, 2), dtype=np.int64)
        )

        if self._is_cyclic_convex_positions():
            cross_pairs = self._build_crossing_pairs_cyclic(edges_arr, n_edges)
        else:
            cross_pairs = self._build_crossing_pairs_geometric(edges, edges_arr, n_edges)

        # Stamp _G_Planar.crossings (every edge on the same layer baseline).
        counts = np.zeros(n_edges, dtype=np.int64)
        if cross_pairs.shape[0]:
            np.add.at(counts, cross_pairs.flatten(), 1)
        edge_to_count = {edges[i]: int(counts[i]) for i in range(n_edges)}
        nx.set_edge_attributes(self._G_Planar, edge_to_count, "crossings")

        mask = np.zeros(n_edges, dtype=bool)
        for idx, (u, v) in enumerate(edges):
            diff = abs(u - v)
            if diff == 1 or diff == self.k - 1:
                mask[idx] = True

        self._edge_list = edges
        self._edge_index = {e: i for i, e in enumerate(edges)}
        self._crossing_pairs = cross_pairs
        self._perimeter_mask = mask

        self.cal_loss_of_edge(self._G_Planar)

    # ------------------------------------------------------------------
    # Phase A — backends
    # ------------------------------------------------------------------
    def _is_cyclic_convex_positions(self) -> bool:
        """Detect whether ``positions[0..k-1]`` traces a *simple* convex
        polygon (winding number ±1).

        A consistent turn-sign check alone is **not enough** — a regular
        pentagram traversed in winding-2 order also has all turns of the
        same sign, but is self-intersecting and would corrupt the
        alternating-endpoints fast path. We additionally:

        - require all node positions to be pairwise distinct (coincident
          vertices produce zero edge vectors and break orientation tests
          downstream); and
        - require the **signed sum of turn angles** to equal ±2π within a
          tight tolerance, which is the topological signature of a simple
          (winding-1) convex polygon. Pentagram-like layouts have signed
          sum ±4π and so flunk this check.

        Nodes in a row along a straight side (square, rectangle, triangle,
        polygon layouts) make straight steps. A step counts as straight
        when ``|sin(turn)| <= _COLLINEAR_TOL`` — float round-off leaves
        interpolated side nodes ~1e-17 off their line with either sign —
        and it must go forward (a reversal is a spike, never convex).
        Such weakly convex layouts cross exactly like a strictly convex
        one with the same node order, which is what ``edge_crosses``
        gives on exactly collinear coordinates.

        These extra guards mean a non-cyclic-convex caller silently falls
        through to the geometric fallback rather than getting wrong
        crossing pairs from the alternating test. Verified by
        ``tests/test_crossings_oracle.py::test_phase_a_rejects_pentagram_fast_path``.
        """
        if self.k < 3:
            return True
        try:
            pts = np.array(
                [self.positions[i] for i in range(self.k)], dtype=np.float64
            )
        except KeyError:
            return False
        if pts.shape != (self.k, 2):
            return False
        # Reject duplicate positions — orient(P, P, R) is identically zero
        # and the alternating test relies on each vertex labeling a
        # geometrically distinct point.
        if len({(float(x), float(y)) for x, y in pts}) != self.k:
            return False

        nxt = np.roll(pts, -1, axis=0)
        nxt2 = np.roll(pts, -2, axis=0)
        e1 = nxt - pts
        e2 = nxt2 - nxt
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        dot = e1[:, 0] * e2[:, 0] + e1[:, 1] * e2[:, 1]

        straight = np.abs(cross) <= (
            _COLLINEAR_TOL * np.hypot(e1[:, 0], e1[:, 1]) * np.hypot(e2[:, 0], e2[:, 1])
        )
        if bool(np.any(straight & (dot <= 0))):
            return False
        nz = cross[~straight]
        if nz.size == 0:
            return False
        if not (bool(np.all(nz > 0)) or bool(np.all(nz < 0))):
            return False

        # Total signed exterior angle = 2π × winding number. For a simple
        # convex polygon, |Σ| = 2π; pentagram-style (winding 2) gives 4π.
        turn = np.arctan2(cross, dot)
        total = float(np.sum(turn))
        return abs(abs(total) - 2.0 * np.pi) < 1e-6

    def _build_crossing_pairs_cyclic(
        self, edges_arr: np.ndarray, n_edges: int
    ) -> np.ndarray:
        """Fast path: alternating-endpoints test on a convex polygon.

        For K_n with node indices laid out cyclically around a convex
        polygon (the SiN layout), two edges (u_i, v_i) and (u_j, v_j) cross
        iff exactly one of ``{u_j, v_j}`` lies strictly inside the open
        interval ``(u_i, v_i)`` AND the edges do not share a vertex. The
        check is purely combinatorial — no float arithmetic — and hits
        the §2-1 ``≤ 5 s`` budget at k=160 (E=12720) with room to spare.
        """
        if n_edges == 0:
            return np.zeros((0, 2), dtype=np.int64)
        u_arr = edges_arr[:, 0].astype(np.int64)
        v_arr = edges_arr[:, 1].astype(np.int64)

        # Process in row-chunks so the per-pair work is fully vectorized
        # without materializing the full O(E^2) bool matrix at k=160
        # (which would be ~8 GB even as bytes).
        chunk_rows = 256
        pieces: list[np.ndarray] = []
        for chunk_start in range(0, n_edges - 1, chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, n_edges)
            j_start = chunk_start + 1
            j_size = n_edges - j_start
            if j_size <= 0:
                continue

            u_i = u_arr[chunk_start:chunk_end, None]      # (cs, 1)
            v_i = v_arr[chunk_start:chunk_end, None]
            u_j = u_arr[j_start:][None, :]                # (1, j_size)
            v_j = v_arr[j_start:][None, :]

            uj_in = (u_j > u_i) & (u_j < v_i)
            vj_in = (v_j > u_i) & (v_j < v_i)
            shared = (u_j == u_i) | (u_j == v_i) | (v_j == u_i) | (v_j == v_i)
            # j > i mask (intra-chunk pairs where j <= i are duplicates of
            # an earlier chunk's row).
            i_idx = np.arange(chunk_start, chunk_end, dtype=np.int64)[:, None]
            j_idx = np.arange(j_start, n_edges, dtype=np.int64)[None, :]
            valid = j_idx > i_idx
            cross = (uj_in ^ vj_in) & ~shared & valid

            row_off, col_off = np.where(cross)
            if row_off.size:
                ii = (row_off.astype(np.int64) + chunk_start)
                jj = (col_off.astype(np.int64) + j_start)
                pieces.append(np.stack((ii, jj), axis=1))

        if not pieces:
            return np.zeros((0, 2), dtype=np.int64)
        return np.concatenate(pieces, axis=0)

    def _build_crossing_pairs_geometric(
        self,
        edges: list[tuple[int, int]],
        edges_arr: np.ndarray,
        n_edges: int,
    ) -> np.ndarray:
        """Fallback path: vectorized **exact** orientation + closed-form
        degenerate handling. Bit-exact with :meth:`edge_crosses` on arbitrary
        positions.

        Crossing decisions derive from the signs of four 2x2 orientation
        determinants. Those signs are computed with :func:`_orient2d_sign`
        (exact, via error-free transforms) rather than naive float64
        determinants — the latter disagreed with shapely/GEOS's robust
        predicate on near-degenerate dense-boundary layouts (triangle /
        regular-polygon side points collinear only to ~1e-17), which
        previously corrupted ``_crossing_pairs`` on those layouts. The
        cyclic-convex fast path (SiN square) is unaffected; it never reaches
        this method.
        """
        if n_edges == 0:
            return np.zeros((0, 2), dtype=np.int64)
        p1 = np.array([self.positions[u] for u, _ in edges], dtype=np.float64)
        p2 = np.array([self.positions[v] for _, v in edges], dtype=np.float64)

        # Per-i vectorized comparison: numpy keeps memory bounded
        # (O(n_edges) per iter) while still scanning every (i, j > i) pair.
        pairs_buf: list[tuple[int, int]] = []
        for i in range(n_edges - 1):
            a1 = p1[i]
            a2 = p2[i]
            i_u = int(edges_arr[i, 0])
            i_v = int(edges_arr[i, 1])

            b1 = p1[i + 1:]
            b2 = p2[i + 1:]
            j_u = edges_arr[i + 1:, 0]
            j_v = edges_arr[i + 1:, 1]

            ax = a2[0] - a1[0]
            ay = a2[1] - a1[1]
            bx = b2[:, 0] - b1[:, 0]
            by = b2[:, 1] - b1[:, 1]

            # Exact orientation signs (robust, vectorized). Naive float64
            # determinants disagree with shapely/GEOS on near-degenerate
            # dense-boundary layouts (triangle / polygon side points that are
            # collinear only to ~1e-17); the exact-sign predicate makes the
            # collinear-vs-skew and sidedness decisions bit-exact with
            # ``edge_crosses`` for every layout. ``s* ∈ {-1, 0, +1}``.
            n_bx = b1.shape[0]
            a1x = np.float64(a1[0])
            a1y = np.float64(a1[1])
            a2x = np.float64(a2[0])
            a2y = np.float64(a2[1])
            s1 = _orient2d_sign(a1x, a1y, a2x, a2y, b1[:, 0], b1[:, 1])
            s2 = _orient2d_sign(a1x, a1y, a2x, a2y, b2[:, 0], b2[:, 1])
            s3 = _orient2d_sign(
                b1[:, 0], b1[:, 1], b2[:, 0], b2[:, 1],
                np.full(n_bx, a1x), np.full(n_bx, a1y),
            )
            s4 = _orient2d_sign(
                b1[:, 0], b1[:, 1], b2[:, 0], b2[:, 1],
                np.full(n_bx, a2x), np.full(n_bx, a2y),
            )
            s1i = s1.astype(np.int64)
            s2i = s2.astype(np.int64)
            s3i = s3.astype(np.int64)
            s4i = s4.astype(np.int64)

            shared = (j_u == i_u) | (j_u == i_v) | (j_v == i_u) | (j_v == i_v)
            # Proper crossing: each segment strictly separates the other's
            # endpoints (a zero on either side is a degeneracy, handled by
            # the T-junction / collinear branches below).
            proper = (s1i * s2i < 0) & (s3i * s4i < 0)

            z1 = s1 == 0
            z2 = s2 == 0
            z3 = s3 == 0
            z4 = s4 == 0

            # T-junction at one endpoint of B (exactly one of {o1, o2} == 0).
            # Match ``edge_crosses``'s shapely semantics: if the zero point
            # also falls within the *bounding box* of A, it lies on segment
            # A (since the orient ≡ 0 already establishes collinearity with
            # the supporting line) and shapely.intersects returns True.
            amin_x = a1[0] if a1[0] <= a2[0] else a2[0]
            amax_x = a1[0] if a1[0] >= a2[0] else a2[0]
            amin_y = a1[1] if a1[1] <= a2[1] else a2[1]
            amax_y = a1[1] if a1[1] >= a2[1] else a2[1]
            b1_on_a_seg = (
                (b1[:, 0] >= amin_x)
                & (b1[:, 0] <= amax_x)
                & (b1[:, 1] >= amin_y)
                & (b1[:, 1] <= amax_y)
            )
            b2_on_a_seg = (
                (b2[:, 0] >= amin_x)
                & (b2[:, 0] <= amax_x)
                & (b2[:, 1] >= amin_y)
                & (b2[:, 1] <= amax_y)
            )

            bmin_x = np.minimum(b1[:, 0], b2[:, 0])
            bmax_x = np.maximum(b1[:, 0], b2[:, 0])
            bmin_y = np.minimum(b1[:, 1], b2[:, 1])
            bmax_y = np.maximum(b1[:, 1], b2[:, 1])
            a1_on_b_seg = (
                (a1[0] >= bmin_x)
                & (a1[0] <= bmax_x)
                & (a1[1] >= bmin_y)
                & (a1[1] <= bmax_y)
            )
            a2_on_b_seg = (
                (a2[0] >= bmin_x)
                & (a2[0] <= bmax_x)
                & (a2[1] >= bmin_y)
                & (a2[1] <= bmax_y)
            )

            t_b1_only = z1 & ~z2 & ~z3 & ~z4 & b1_on_a_seg
            t_b2_only = ~z1 & z2 & ~z3 & ~z4 & b2_on_a_seg
            t_a1_only = ~z1 & ~z2 & z3 & ~z4 & a1_on_b_seg
            t_a2_only = ~z1 & ~z2 & ~z3 & z4 & a2_on_b_seg

            # All-four-collinear: partial-overlap returns True; full
            # containment in either direction returns False (matches
            # ``edge_crosses``'s "both endpoints of e2 on line1" guard).
            all_coll = z1 & z2 & z3 & z4
            denom_a = ax * ax + ay * ay
            denom_b = bx * bx + by * by
            # Guard against zero-length A (shouldn't occur for K_n with
            # distinct node positions). For zero-length B, the pair is
            # also degenerate; mark as no-cross to match ``edge_crosses``.
            safe_a = denom_a if denom_a > 0 else 1.0
            t_b1 = ((b1[:, 0] - a1[0]) * ax + (b1[:, 1] - a1[1]) * ay) / safe_a
            t_b2 = ((b2[:, 0] - a1[0]) * ax + (b2[:, 1] - a1[1]) * ay) / safe_a
            safe_b = np.where(denom_b > 0, denom_b, 1.0)
            s_a1 = ((a1[0] - b1[:, 0]) * bx + (a1[1] - b1[:, 1]) * by) / safe_b
            s_a2 = ((a2[0] - b1[:, 0]) * bx + (a2[1] - b1[:, 1]) * by) / safe_b
            both_b_in_a = (t_b1 >= 0) & (t_b1 <= 1) & (t_b2 >= 0) & (t_b2 <= 1)
            both_a_in_b = (s_a1 >= 0) & (s_a1 <= 1) & (s_a2 >= 0) & (s_a2 <= 1)
            tb_min = np.minimum(t_b1, t_b2)
            tb_max = np.maximum(t_b1, t_b2)
            overlap_collinear = (tb_min < 1) & (tb_max > 0) & (denom_a > 0) & (denom_b > 0)

            collinear_cross = all_coll & overlap_collinear & ~both_b_in_a & ~both_a_in_b

            cross = (
                proper
                | t_b1_only
                | t_b2_only
                | t_a1_only
                | t_a2_only
                | collinear_cross
            ) & ~shared

            for k_off in np.flatnonzero(cross):
                pairs_buf.append((i, i + 1 + int(k_off)))

        if pairs_buf:
            return np.asarray(pairs_buf, dtype=np.int64)
        return np.zeros((0, 2), dtype=np.int64)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def cal_loss_of_edge(self, graph: nx.Graph) -> None:
        """Per-edge loss = 2*loss_crossing*crossings + 2*|layer-ecl|*loss_taper.

        M3 (§2-3 目标 B): taper term generalized to
        ``2 * |layer - edge_coupler_layer| * loss_taper`` so that the coupler
        layer pays zero taper regardless of which physical layer it sits on.
        For the legacy default ``edge_coupler_layer=0``, this reduces to the
        pre-M3 ``2 * layer * loss_taper`` (since ``layer >= 0``), preserving
        bit-exact compatibility — see REFACTOR_GOALS.md §2-3 验收第 1 条.
        """
        ecl = self.edge_coupler_layer
        for edge in graph.edges(data=True):
            crossings = edge[2].get("crossings", 0)
            layer = edge[2].get("layer", 0)
            taper_hops = abs(int(layer) - ecl)
            loss = self.loss_crossing * crossings * 2 + 2 * taper_hops * self.loss_taper
            graph.edges[edge[0], edge[1]]["loss"] = loss

    def update_interlayercrossings_loss(self, graph: nx.Graph) -> None:
        """Add the per-edge interlayer-crossing contribution onto ``loss``."""
        for edge in graph.edges(data=True):
            current_loss = edge[2].get("loss", 0)
            interlayer_crossings = edge[2].get("interlayercrossings", 0)
            graph.edges[edge[0], edge[1]]["loss"] = (
                current_loss + interlayer_crossings * self.loss_interlayercrossing
            )

    # ------------------------------------------------------------------
    # Layer management & k-base arithmetic (used by routing seeds)
    # ------------------------------------------------------------------
    def set_edge_layer(self, edge: tuple, layer: int) -> None:
        if self.G.has_edge(*edge):
            self.G.edges[edge]["layer"] = layer
        else:
            print(f"Edge {edge} not found in the graph.")

    def k_base_no_carry_add(self, *args, k: int) -> int:
        return sum(args) % k

    def k_base_no_carry_sub(self, *args, k: int) -> int:
        result = args[0]
        for i in range(1, len(args)):
            result = (result - args[i]) % k
        return result if result >= 0 else result + k

    def routing_method_1(self, takeaway: list[int] | None = None, layer: int = 1) -> None:
        """Stamp a k-base offset pattern onto edges (used as optimizer seed)."""
        if takeaway is None:
            takeaway = [2]
        for m in takeaway:
            for i in range(self.k):
                _p1 = self.k_base_no_carry_add(i, m, 1, k=self.k)
                _p2 = self.k_base_no_carry_sub(i, m, 1, k=self.k)
                self.set_edge_layer((i, _p1), layer)
                self.set_edge_layer((i, _p2), layer)

    # ------------------------------------------------------------------
    # Subgraph build (used by both optimizer's loss and analysis)
    # ------------------------------------------------------------------
    def create_subgraphs(self) -> None:
        """Split ``self.G`` into per-layer subgraphs and stamp losses.

        M3: produces ``self.sub_G`` with **L entries**, one per layer index
        ``0..L-1`` (including empty subgraphs for unpopulated layers). This
        keeps ``sub_G[i]`` equal to "the layer-i subgraph" rather than "the
        i-th occupied layer in sorted order", which the multi-layer
        analyze_loss / writer paths rely on (REFACTOR_GOALS.md §2-3 目标
        A). For L=2 with one layer empty (e.g. ``all1`` test pattern), the
        empty layer-0 entry is still present.

        Interlayer crossings are counted for **every adjacent pair**
        ``(i, i+1)`` per §2-3 目标 D (non-adjacent layer pairs are not
        modelled — physical evanescent coupling for layer distance ≥ 2 is
        below the −60 dB crosstalk floor), with the per-edge accounting of
        :meth:`count_interlayercrossings`.

        Crossings come from the cached Phase A index — the crossing relation
        :meth:`loss_function` scores — so the stamped per-edge ``crossings``
        / ``interlayercrossings*`` and losses always match what the
        optimizer saw for this assignment, on every layout, and no O(E²)
        geometry pass runs per layer. Wherever ``edge_crosses`` is exact
        (every layout except float round-off on collinear side runs; see
        :meth:`_is_cyclic_convex_positions`) the counts equal
        :meth:`count_crossings_with_detail` / :meth:`count_interlayercrossings`
        on the layer subgraphs.
        """
        self._ensure_crossings_ready()
        n_edges = len(self._edge_list)
        layer_of = np.array([self.G.edges[e]["layer"] for e in self._edge_list], dtype=np.int64)
        pairs = self._crossing_pairs
        la = layer_of[pairs[:, 0]]
        lb = layer_of[pairs[:, 1]]
        same = (la == lb) & (la >= 0) & (la < self.L)
        intra = np.bincount(pairs[same].ravel(), minlength=n_edges)
        pairs_in_layer = np.bincount(la[same], minlength=self.L)
        # Adjacent-layer events: the lower edge sees one "above", the upper
        # edge one "below"; the legacy counter is their sum.
        adj = (np.abs(la - lb) == 1) & (np.minimum(la, lb) >= 0) & (np.maximum(la, lb) < self.L)
        a_lower = la < lb
        above = np.bincount(np.where(a_lower, pairs[:, 0], pairs[:, 1])[adj], minlength=n_edges)
        below = np.bincount(np.where(a_lower, pairs[:, 1], pairs[:, 0])[adj], minlength=n_edges)

        self.sub_G = []
        self.total_crossings_of_sub_G = []
        self.edge_cross_counts_of_sub_G = []

        for layer in range(self.L):
            edges_in_layer = [
                (u, v) for u, v, d in self.G.edges(data=True) if d["layer"] == layer
            ]
            if edges_in_layer:
                subgraph = self.G.edge_subgraph(edges_in_layer)
            else:
                # Empty subgraph — keep the node set so plot / analysis
                # paths behave uniformly (``Graph.edges`` returns nothing,
                # ``Graph.nodes`` returns range(k)).
                subgraph = nx.Graph()
                subgraph.add_nodes_from(range(self.k))
            edge_cross_counts = {}
            for u, v in subgraph.edges():
                i = self._edge_index[(u, v) if (u, v) in self._edge_index else (v, u)]
                data = subgraph.edges[u, v]
                data["crossings"] = int(intra[i])
                data["interlayercrossings"] = int(above[i] + below[i])
                data["interlayercrossings_above"] = int(above[i])
                data["interlayercrossings_below"] = int(below[i])
                edge_cross_counts[(u, v)] = int(intra[i])
            self.cal_loss_of_edge(subgraph)
            self.sub_G.append(subgraph)
            self.total_crossings_of_sub_G.append(int(pairs_in_layer[layer]))
            self.edge_cross_counts_of_sub_G.append(edge_cross_counts)

        for sg in self.sub_G:
            self.update_interlayercrossings_loss(sg)

    # ------------------------------------------------------------------
    # Loss function consumed by optimizers
    # ------------------------------------------------------------------
    def loss_function(self, layers: Iterable[float]) -> float:
        """Phase B (REFACTOR_GOALS.md §2-1): mean per-edge loss without
        re-running geometry. Uses the cached ``_crossing_pairs`` index so
        the hot loop is O(X) integer classification.

        Edges connecting consecutive nodes (|u-v| in {1, k-1}) are pinned to
        ``self.perimeter_layer`` regardless of input — they are the perimeter
        ring. M3: perimeter layer is configurable (default = edge coupler
        layer); legacy code path was pinned to 0, which is preserved when
        ``perimeter_layer == 0``.

        Return value is bit-exact (modulo float summation order) with the
        pre-M2 ``count_crossings_with_detail``-based implementation for
        L=2 + ``edge_coupler_layer=0`` + ``perimeter_layer=0``; see
        ``tests/test_crossings_oracle.py`` and §2-3 验收第 1 条.

        Side effects: none. Unlike the pre-M2 implementation, the per-call
        layer assignment is NOT written back onto ``self.G``. The optimizer
        never reads ``self.G`` during the hot loop, and the final layer
        assignment is published via ``apply_optimization_result``.
        """
        self._ensure_crossings_ready()
        n_edges = len(self._edge_list)

        layers_int = np.round(np.asarray(layers, dtype=float)).astype(np.int64)
        if layers_int.shape != (n_edges,):
            raise ValueError(
                f"layers has shape {layers_int.shape}, expected ({n_edges},)"
            )

        pinned = layers_int.copy()
        pinned[self._perimeter_mask] = self.perimeter_layer

        pairs = self._crossing_pairs
        if pairs.shape[0]:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            la = pinned[i_idx]
            lb = pinned[j_idx]
            same = la == lb
            adj = np.abs(la - lb) == 1
            # Single concatenated ``bincount`` per mask — numpy's
            # unweighted C scatter dominates the concat copy by ~2x in
            # microbenchmarks, even with the 200 MB allocation at k=160.
            # The §2-1 ``≤ 10 ms`` per-call stretch goal is below the
            # bandwidth floor for the 26 M-crossing K_160 layout (~400 MB
            # of pair memory to scan); we hit ~300 ms here. §5's 10-min
            # total budget at 500 iter is still met with plenty of margin.
            if same.any():
                same_flat = np.concatenate((i_idx[same], j_idx[same]))
                intra = np.bincount(same_flat, minlength=n_edges).astype(np.int64)
            else:
                intra = np.zeros(n_edges, dtype=np.int64)
            if adj.any():
                adj_flat = np.concatenate((i_idx[adj], j_idx[adj]))
                inter = np.bincount(adj_flat, minlength=n_edges).astype(np.int64)
            else:
                inter = np.zeros(n_edges, dtype=np.int64)
        else:
            intra = np.zeros(n_edges, dtype=np.int64)
            inter = np.zeros(n_edges, dtype=np.int64)

        # M3: taper term uses |layer - edge_coupler_layer| per §2-3 目标 B.
        # For the default edge_coupler_layer=0 this matches the pre-M3
        # ``2 * pinned * loss_taper`` (since pinned >= 0), preserving the
        # §2-3 验收第 1 条 bit-exact L=2/ecl=0 baseline.
        taper_hops = np.abs(pinned - self.edge_coupler_layer)
        per_edge_loss = (
            2.0 * self.loss_crossing * intra
            + 2.0 * taper_hops * self.loss_taper
            + self.loss_interlayercrossing * inter
        )

        # M3 (REFACTOR_GOALS.md §2-3): aggregate the per-edge mean over the
        # subset of edges whose post-pin layer matches one of the input
        # layer values. For L=2 + ``edge_coupler_layer=0`` + ``perimeter_layer=0``
        # this is bit-equivalent to the pre-M2 iteration
        #
        #   ``for _layer in unique(layers_int): include sub_G[_layer]``
        #
        # because the legacy ``sub_G`` was indexed by post-pin layer **position**
        # in ``sorted(set(post_pin_layers))``, and for L=2 the post-pin set is
        # always ``{0, 1}`` (perimeter pin to 0 guarantees layer 0 occupancy),
        # so input ``_layer ∈ {0, 1}`` aligns with the sub_G position index. For
        # L>=3 the position-index trick silently dropped edges whose post-pin
        # layer exceeded ``len(sub_G)`` — see the in-line note in the M2 oracle
        # test. The pinned-value comparison below is the correct
        # generalization: select edges whose final layer is in the input
        # support set, regardless of which integer index that lands at.
        unique_in = np.unique(layers_int)
        selected = np.isin(pinned, unique_in)

        if selected.any():
            return float(np.mean(per_edge_loss[selected]))
        return 0.0

    def effective_layers(self, layers: Iterable[float]) -> np.ndarray:
        """Integer per-edge layers that ``loss_function(layers)`` scores.

        Rounds the (possibly continuous) optimizer vector the same way as
        :meth:`loss_function` / :meth:`apply_optimization_result` and pins
        the perimeter ring to ``perimeter_layer``. Indexed by
        ``list(self.G.edges())``. Read-only helper for progress tracking;
        not used by the optimizer hot loop.
        """
        self._ensure_crossings_ready()
        pinned = np.round(np.asarray(layers, dtype=float)).astype(np.int64)
        if pinned.shape != (len(self._edge_list),):
            raise ValueError(
                f"layers has shape {pinned.shape}, expected ({len(self._edge_list)},)"
            )
        pinned[self._perimeter_mask] = self.perimeter_layer
        return pinned

    def intralayer_crossing_counts(self, layers: np.ndarray) -> np.ndarray:
        """Per-edge same-layer crossing counts for an :meth:`effective_layers`
        vector, from the cached Phase A index (no geometry pass).

        Equals the ``crossings`` attribute that :meth:`create_subgraphs`
        stamps on each layer subgraph for the same assignment.
        """
        self._ensure_crossings_ready()
        n_edges = len(self._edge_list)
        pairs = self._crossing_pairs
        if not pairs.shape[0]:
            return np.zeros(n_edges, dtype=np.int64)
        layers = np.asarray(layers)
        same = layers[pairs[:, 0]] == layers[pairs[:, 1]]
        return np.bincount(pairs[same].ravel(), minlength=n_edges).astype(np.int64)

    def _apply_layer_assignment(self, layers: np.ndarray) -> None:
        """Stamp per-edge layers, pinning the perimeter ring to
        ``self.perimeter_layer``.

        M3: perimeter pin target generalized from the hardcoded 0 in
        pre-M3 to ``self.perimeter_layer`` (§2-3-B 补充). The default
        ``perimeter_layer = edge_coupler_layer`` lets a perimeter edge
        pay zero taper (no cross-layer hop), matching the physical
        intuition that the ring sits on the coupler layer.
        """
        for edge, layer in zip(self.G.edges, layers):
            if abs(edge[0] - edge[1]) == 1 or abs(edge[0] - edge[1]) == self.k - 1:
                self.set_edge_layer(edge, self.perimeter_layer)
            else:
                self.set_edge_layer(edge, int(layer))

    def apply_optimization_result(self, layers: Iterable[float]) -> None:
        """Apply a final layer assignment (post-optimization). Resets attrs."""
        nx.set_edge_attributes(self.G, 0, "layer")
        nx.set_edge_attributes(self.G, 0, "crossings")
        nx.set_edge_attributes(self.G, 0, "loss")
        nx.set_edge_attributes(self.G, 0, "interlayercrossings")
        nx.set_edge_attributes(self.G, 0, "interlayercrossings_above")
        nx.set_edge_attributes(self.G, 0, "interlayercrossings_below")
        self._apply_layer_assignment(np.round(np.asarray(layers)).astype(int))

    # ------------------------------------------------------------------
    # Loss analysis (used by plotting/loss_analysis)
    # ------------------------------------------------------------------
    def analyze_loss(self) -> None:
        """Compute mean / var / std / range of per-edge losses across layers.

        M3: aggregations are computed over **occupied** layers (subgraphs
        with at least one edge); empty layers contribute zero-length lists
        which numpy would otherwise warn-and-NaN on. ``sub_G`` itself is
        L entries long, the per-layer stats arrays are also L entries long
        (empty layers report ``nan`` so downstream consumers know the slot
        was unoccupied rather than zero-loss).
        """
        # ``_G_Planar.crossings``/``loss`` are stamped inside Phase A; ensure
        # the index exists before we read those attributes below.
        self._ensure_crossings_ready()

        self.sub_G = []
        self.total_crossings_of_sub_G = []
        self.edge_cross_counts_of_sub_G = []

        self.create_subgraphs()

        loss_subgraphs: list[list[float]] = []
        for sg in self.sub_G:
            loss_subgraphs.append(
                [sg.edges[edge]["loss"] for edge in sg.edges()]
            )

        # opus-review P1-7 / gpt-review P1-7: empty layers must emit
        # ``null`` rather than ``float('nan')`` so the JSON writer can use
        # ``allow_nan=False`` and downstream strict parsers (esp. the M5
        # C++ ``nlohmann::json``) do not reject the file. ``None``
        # round-trips as ``null`` in both directions.
        def _agg(fn, values):
            return float(fn(values)) if values else None

        avg_loss_subgraphs = [_agg(np.mean, sub) for sub in loss_subgraphs]
        var_loss_subgraphs = [_agg(np.var, sub) for sub in loss_subgraphs]
        std_loss_subgraphs = [_agg(np.std, sub) for sub in loss_subgraphs]
        range_loss_subgraphs = [_agg(np.ptp, sub) for sub in loss_subgraphs]

        flattened = [loss for sub in loss_subgraphs for loss in sub]

        loss_complete_graph = [self._G_Planar.edges[edge]["loss"] for edge in self._G_Planar.edges()]

        self.loss_analysis = {
            "avg_loss_subgraphs": avg_loss_subgraphs,
            "var_loss_subgraphs": var_loss_subgraphs,
            "std_loss_subgraphs": std_loss_subgraphs,
            "range_loss_subgraphs": range_loss_subgraphs,
            "avg_flattened_loss_subgraphs": _agg(np.mean, flattened),
            "var_flattened_loss_subgraphs": _agg(np.var, flattened),
            "std_flattened_loss_subgraphs": _agg(np.std, flattened),
            "range_flattened_loss_subgraphs": _agg(np.ptp, flattened),
            "avg_loss_completegraph": float(np.mean(loss_complete_graph)),
            "var_loss_completegraph": float(np.var(loss_complete_graph)),
            "std_loss_completegraph": float(np.std(loss_complete_graph)),
            "range_loss_completegraph": float(np.ptp(loss_complete_graph)),
        }

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_crosstalk_payload_shape(crosstalk: dict) -> None:
        """Cross-field length check for the crosstalk tensor (M4 dual-review).

        JSON Schema Draft 2020-12 cannot easily express
        ``len(values) == shape[i]`` constraints. We enforce them here
        so a buggy external writer is rejected at the JSON boundary
        rather than producing a file whose declared ``shape`` lies about
        the ``values`` payload. Catches the M5 C++ parity hazard
        flagged by both M4 reviewers.

        Raises
        ------
        ValueError
            If ``shape`` is missing / not a 3-list / contains non-int
            entries, or ``len(values) != shape[0]`` / any inner
            dimension mismatches.
        """
        shape = crosstalk.get("shape")
        values = crosstalk.get("values")
        # ``bool`` is a subclass of ``int`` in Python — accepting it would let
        # a hand-built payload (e.g. M5 C++ side) slip ``shape=[True, True,
        # True]`` past this guard. We defend explicitly because the validator
        # is the JSON boundary for non-Python writers (opus-review-2 P2 nit).
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 3
            or not all(
                isinstance(x, int) and not isinstance(x, bool) and x > 0
                for x in shape
            )
        ):
            raise ValueError(
                f"crosstalk.shape must be a 3-element list of positive ints; "
                f"got {shape!r}."
            )
        if not isinstance(values, list) or len(values) != shape[0]:
            raise ValueError(
                f"crosstalk.values outer length {len(values) if isinstance(values, list) else '<not a list>'} "
                f"does not match shape[0]={shape[0]}."
            )
        for s_idx, slab in enumerate(values):
            if not isinstance(slab, list) or len(slab) != shape[1]:
                raise ValueError(
                    f"crosstalk.values[{s_idx}] length "
                    f"{len(slab) if isinstance(slab, list) else '<not a list>'} "
                    f"does not match shape[1]={shape[1]}."
                )
            for d_idx, row in enumerate(slab):
                if not isinstance(row, list) or len(row) != shape[2]:
                    raise ValueError(
                        f"crosstalk.values[{s_idx}][{d_idx}] length "
                        f"{len(row) if isinstance(row, list) else '<not a list>'} "
                        f"does not match shape[2]={shape[2]}."
                    )

    def save_subgraphs_to_json(
        self,
        filename: str,
        *,
        skip_analysis: bool = False,
        crosstalk: dict | None = None,
        **kwargs,
    ) -> None:
        """Persist a self-contained snapshot to JSON.

        The file embeds everything plotting needs — positions, per-layer
        subgraphs (with per-edge ``crossings`` / ``loss`` / ``interlayercrossings``
        attrs and the M3 ``_above`` / ``_below`` split), the complete-graph
        baseline (``_G_Planar``), the ``loss_analysis`` dict, and all loss
        + multi-layer parameters — so downstream plotting can run purely
        off this file without touching the graph.

        ``analyze_loss`` is invoked first to ensure ``sub_G`` and
        ``loss_analysis`` are fresh, unless ``skip_analysis=True`` — used
        by the orchestrator after it has already run (and timed)
        ``analyze_loss`` separately, so this method's runtime is purely
        serialization (REFACTOR_GOALS.md §3-2 ``loss_analysis_ms`` vs
        ``json_write_ms`` split, landed in M2).

        Schema version: **v2.0** (M3) by default. Includes ``L``,
        ``edge_coupler_layer``, ``perimeter_layer``, ``layer_pitch_um``,
        ``waveguides_per_link``, plus per-edge ``_above`` / ``_below``
        counters. v1.x readers see ``interlayercrossings`` and ignore the
        new fields — backward compatible.

        M4 (§2-2): pass a pre-computed ``crosstalk`` payload to embed the
        rank-3 tensor under the top-level ``crosstalk`` key. The writer
        stays dumb — it just serializes whatever the orchestrator hands
        in — so the engine in :mod:`.crosstalk` can be tested
        independently of JSON I/O.
        """
        if not skip_analysis:
            self.analyze_loss()

        # M8 parity contract: any new General-Parameters key added below
        # OR via **kwargs MUST either (a) be mirrored on the C++ side in
        # `routing_cpp_rebuild/src/io.cpp::subgraphsdata_v1x_to_json` OR
        # (b) be added to `code/tests/parity_harness.py::KNOWN_GP_ASYMMETRIES`.
        # The parity test fails loudly on unknown drift; this comment
        # exists so the first thing a developer adding a GP field sees is
        # the parity boundary.
        sub_graph_data: dict = {
            "schema_version": "2.0",  # M3 ship; see code/schema/subgraphsdata.schema.json
            "General Parameters": {
                "k": int(self.k),
                "L": int(self.L),
                "edge_coupler_layer": int(self.edge_coupler_layer),
                "perimeter_layer": int(self.perimeter_layer),
                "layer_pitch_um": float(self.layer_pitch_um),
                "waveguides_per_link": int(self.waveguides_per_link),
                "Loss of Taper": self.loss_taper,
                "Loss of Crossing": self.loss_crossing,
                "Loss of Interlayer Crossing": self.loss_interlayercrossing,
                "Loss of Intralayer Crosstalk": self.loss_intralayer_crosstalk,
                "Loss of Interlayer Crosstalk": self.loss_interlayer_crosstalk,
                "coherence_model": self.coherence_model,
                "polarization": self.polarization,
            },
        }
        for key, value in kwargs.items():
            sub_graph_data["General Parameters"][key] = value

        sub_graph_data["positions"] = {
            str(int(idx)): [float(coord[0]), float(coord[1])]
            for idx, coord in self.positions.items()
        }

        for i, sub_G in enumerate(self.sub_G):
            sub_graph_data[f"Layer_{i}"] = {"edges": list(sub_G.edges(data=True))}

        sub_graph_data["complete_graph"] = list(self._G_Planar.edges(data=True))
        sub_graph_data["loss_analysis"] = self.loss_analysis

        # M4 (§2-2 / §3-1 兼容性策略 last bullet): embed the rank-3 crosstalk
        # tensor if the orchestrator computed one. ``None`` (the M3 default)
        # is skipped — schema keeps the key optional so v2.0 files that
        # pre-date the crosstalk engine still validate.
        if crosstalk is not None:
            # M4 dual-review (codex P1 / opus P2-1): JSON Schema Draft
            # 2020-12 cannot enforce ``len(values) == shape[i]`` cross-field,
            # so a buggy external writer (e.g. the M5 C++ side) could slip
            # a mismatched tensor past structural validation. Cross-check
            # here at the JSON boundary so any divergence raises before the
            # file lands on disk.
            self._validate_crosstalk_payload_shape(crosstalk)
            sub_graph_data["crosstalk"] = crosstalk

        # REFACTOR_GOALS.md §3-3: validate before writing so the caller sees
        # a ``ValidationError`` instead of a half-written file pair.
        jsonschema.validate(instance=sub_graph_data, schema=_load_subgraphs_schema())

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            # opus-review P1-7 / gpt-review P1-7: ``allow_nan=False`` forces
            # any stray ``float('nan')`` / ``inf`` to raise rather than emit
            # the non-RFC-8259 ``NaN``/``Infinity`` tokens that the M5 C++
            # ``nlohmann::json`` parser would reject. ``analyze_loss`` has
            # already normalised empty-layer aggregates to ``None``, so this
            # is defense-in-depth.
            json.dump(sub_graph_data, f, indent=4, allow_nan=False)
        print(f"Subgraphs saved to {filename}")
