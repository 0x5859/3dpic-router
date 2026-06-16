"""Regenerate the M8 parity fixtures under ``tests/golden/``.

Each fixture is a JSON spec describing a graph configuration plus a
fixed per-edge layer assignment for the K_k complete graph. The parity
harness (``tests/parity_harness.py``) feeds the spec to both Python and
C++ and compares their ``subgraphsdata.json`` outputs.

Run:

    uv run python code/tests/golden/generate_fixtures.py

The script is idempotent — fixture content depends only on the explicit
parameters below, so re-running produces byte-identical files.
"""

from __future__ import annotations

from pathlib import Path

# Path-import the harness sibling without a package install.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parity_harness import ParityFixture, write_fixture_spec

OUTPUT_DIR = Path(__file__).resolve().parent


def _striped_layers(num_edges: int, L: int) -> list[int]:
    """Alternating layer assignment so every L-stripe gets exercised."""
    return [i % L for i in range(num_edges)]


def _constant_layers(num_edges: int, layer: int) -> list[int]:
    return [layer] * num_edges


def _out_of_range_nonperimeter_layers(k: int, L: int) -> list[int]:
    """Striped base assignment with the first two **non-perimeter** edges
    pushed out of ``[0, L-1]`` (one over-range, one negative).

    Regression vehicle for the pre-existing ``apply_optimization_result``
    clamp parity bug. Python ``_apply_layer_assignment`` (core.py) and
    C++ ``loss_function`` (graph.cpp, post-M6-P1-4) stamp the raw rounded
    layer value with **no** clamp, but C++ ``apply_optimization_result``
    historically clamped non-perimeter values into ``[0, L-1]``. On the
    ``fixed_layers`` path that produced a silent Python↔C++ divergence in
    the ``subgraphsdata.json`` Layer maps + ``loss_analysis``
    (REFACTOR_GOALS.md §1-2-b; §1-2 design-tradeoff #4 rejected the same
    clamp in ``loss_function`` for exactly this reason).

    Perimeter edges (``|u-v| ∈ {1, k-1}``) are pinned to
    ``perimeter_layer`` on both sides regardless of input, so the
    out-of-range probes MUST land on non-perimeter edges to exercise the
    clamp. The two probes cover both removed clamp branches: ``lyr >= L``
    (over-range, was clamped to ``L-1``) and ``lyr < 0`` (negative, was
    clamped to ``0``). Edge enumeration matches the shared row-major
    ``u<v`` order both ends agree on (verified bit-exact by the existing
    M8 per-edge integer-count parity).
    """
    num_edges = k * (k - 1) // 2
    layers = [i % L for i in range(num_edges)]
    nonperim_idx: list[int] = []
    idx = 0
    for u in range(k):
        for v in range(u + 1, k):
            diff = abs(u - v)
            if diff != 1 and diff != (k - 1):
                nonperim_idx.append(idx)
            idx += 1
    if len(nonperim_idx) < 2:
        raise ValueError(
            f"k={k} has < 2 non-perimeter edges; cannot place both "
            f"out-of-range probes"
        )
    layers[nonperim_idx[0]] = L + 5   # over-range → was clamped to L-1
    layers[nonperim_idx[1]] = -1      # negative   → was clamped to 0
    return layers


def main() -> None:
    k = 12
    num_edges = k * (k - 1) // 2  # K_12 → 66 edges

    fixtures = [
        # --- F1: v2.0 wpl=1 (geometric convention) --------------------
        # Checks that the v2.0 writer correctly emits
        # convention='geometric' / wpl=1 and that both sides agree on
        # aggregate counts under that convention. Striped 0/1 assignment
        # guarantees Layer_0 and Layer_1 both non-empty so the
        # loss_analysis arrays exercise the multi-layer bucket logic.
        #
        # Note on the missing v1.x writer-parity fixture: §6 T6 lists
        # "legacy v1.x" as a fixture class, but the Python writer
        # (``core.py::save_subgraphs_to_json``) hard-codes
        # ``schema_version="2.0"`` since M3 — there is no Python-side
        # opt-in for v1.x output. The C++ side via
        # ``SINIC_SCHEMA_VERSION=1.0`` can produce a v1.x file, but
        # there is nothing to compare it against. v1.x is therefore a
        # **reader** parity concern, covered by the dedicated
        # ``test_legacy_v1x_reader_parity`` in ``test_parity.py``.
        ParityFixture(
            name="k12_L2_wpl1_v2",
            k=k, L=2, waveguides_per_link=1,
            edge_coupler_layer=0,
            perimeter_layer=0,
            schema_version="2.0",
            layers=_striped_layers(num_edges, 2),
        ),
        # --- F3: v2.0 wpl=2 (physical convention, M6 default) ---------
        ParityFixture(
            name="k12_L2_wpl2_v2",
            k=k, L=2, waveguides_per_link=2,
            edge_coupler_layer=0,
            perimeter_layer=0,
            schema_version="2.0",
            layers=_striped_layers(num_edges, 2),
        ),
        # --- F3b: out-of-range fixed-layer regression -----------------
        # Identical physical params to F3 (k12_L2_wpl2_v2) so the ONLY
        # changed variable is the layer vector: striped 0/1 with the
        # first two non-perimeter edges set to L+5 and -1. Pre-fix the
        # C++ `apply_optimization_result` clamp folded these into
        # [0, L-1] (so they entered Layer_1 / Layer_0) while Python /
        # C++ `loss_function` kept them raw (excluded from every
        # per-layer bucket). The parity comparator therefore flagged a
        # Layer_* edge-set + loss_analysis divergence. With the clamp
        # removed both ends drop the two probe edges from all per-layer
        # subgraphs identically (still present in `complete_graph`),
        # restoring §1-2-b parity. See REFACTOR_GOALS.md §1-2
        # design-tradeoff #4 (loss_function) + §1-4 M8 dual-review log
        # Opus P2-O5 RESOLVED (apply_optimization_result).
        ParityFixture(
            name="k12_L2_wpl2_oor_fixed",
            k=k, L=2, waveguides_per_link=2,
            edge_coupler_layer=0,
            perimeter_layer=0,
            schema_version="2.0",
            layers=_out_of_range_nonperimeter_layers(k, 2),
        ),
        # --- F4: multi-layer (Python M3 / C++ M6) ---------------------
        # L=3, ecl=1 (the "L//2" recommendation for L=3), wpl=2; striped
        # 0/1/2 assignment forces edges into every layer.
        ParityFixture(
            name="k12_L3_wpl2_v2",
            k=k, L=3, waveguides_per_link=2,
            edge_coupler_layer=1,
            perimeter_layer=1,
            schema_version="2.0",
            layers=_striped_layers(num_edges, 3),
        ),
        # --- F5: crosstalk, intralayer-only (Python M4 / C++ M7) ------
        # Same L=2 baseline as F3 but with crosstalk coefficients
        # enabled. The fixed layer assignment is all-zeros so the
        # crosstalk tensor exercises intralayer paths exclusively (the
        # 4-node analytical sanity from §2-2 验收第 3 条 — here scaled
        # to K_12 — gives a non-trivial set of non-null entries without
        # adjacent-layer perturbations).
        #
        # We pick coefficients that survive the threshold prune (-60
        # dB) at hop ≤ 3: intralayer = 1e-4 → −40 dB single-hop. Engine
        # short-circuit predicate (`any_nonzero_coef`) fires on either
        # side, so a `crosstalk` field IS emitted on both ends.
        ParityFixture(
            name="k12_L2_wpl2_xtalk",
            k=k, L=2, waveguides_per_link=2,
            edge_coupler_layer=0,
            perimeter_layer=0,
            schema_version="2.0",
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            compute_crosstalk=True,
            crosstalk_max_hops=3,
            crosstalk_threshold_db=-60.0,
            layers=_constant_layers(num_edges, 0),
        ),
        # --- F6: crosstalk, mixed inter+intralayer --------------------
        # M8 dual-review Opus P1-O5: the all-zeros F5 fixture never
        # exercises the `|Δlayer|=1` adjacency branch in the crosstalk
        # DFS (because all edges land on Layer_0). F6 uses the same
        # striped 0/1 assignment as F3 so both `intralayer` and
        # `interlayer` coefficient paths fire; the loss-formula
        # implicit 2× in v2.0 wpl=2 mode also stays exercised.
        ParityFixture(
            name="k12_L2_wpl2_xtalk_mixed",
            k=k, L=2, waveguides_per_link=2,
            edge_coupler_layer=0,
            perimeter_layer=0,
            schema_version="2.0",
            loss_intralayer_crosstalk=1e-4,
            loss_interlayer_crosstalk=1e-5,
            compute_crosstalk=True,
            crosstalk_max_hops=3,
            crosstalk_threshold_db=-60.0,
            layers=_striped_layers(num_edges, 2),
        ),
    ]

    for fixture in fixtures:
        out = OUTPUT_DIR / f"{fixture.name}.json"
        write_fixture_spec(fixture, out)
        print(f"wrote {out.relative_to(OUTPUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
