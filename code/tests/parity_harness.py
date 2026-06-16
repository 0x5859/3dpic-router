"""Python ↔ C++ parity harness (REFACTOR_GOALS.md §6 T6 / M8).

Drives both ``routing_py_rebuild.run_optimization`` and the C++
``Autowiring_CPP`` binary with an identical, *fixed* per-edge layer
assignment so we can verify that:

  1. Both writers produce JSON conforming to the **shared** schema files
     under ``code/schema/``.
  2. The structural field set is identical modulo a documented set of
     known cross-impl asymmetries (see ``KNOWN_GP_ASYMMETRIES``).
  3. Per-edge integer counts (``crossings``, ``interlayercrossings``,
     ``interlayercrossings_above`` / ``_below``) are **bit-exact** between
     the two implementations.
  4. Per-edge ``loss`` and aggregate ``loss_analysis`` floats fall within
     REFACTOR_GOALS.md §1-2-b tolerance (abs ``1e-8`` OR rel ``1e-10``).
  5. The optional ``crosstalk`` rank-3 tensor agrees within the same
     tolerance, with ``None`` ↔ ``null`` semantics preserved.

The harness is invoked by ``code/tests/test_parity.py`` (pytest); the
file is kept in a separate module so it can also be imported by ad-hoc
scripts and the fixture-builder helper.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Absolute path to the repo's `code/` directory regardless of CWD.
CODE_ROOT = Path(__file__).resolve().parent.parent
CPP_BUILD_DIR = CODE_ROOT / "routing_cpp_rebuild" / "build"
CPP_BINARY = CPP_BUILD_DIR / "Autowiring_CPP"
SCHEMA_DIR = CODE_ROOT / "schema"

# §1-2-b numerical tolerance: abs 1e-8 OR rel 1e-10 (either passes).
ABS_TOL = 1e-8
REL_TOL = 1e-10

# Orchestration env vars that any subprocess-spawning parity test must
# clear from the inherited shell environment before running the C++
# binary. A developer with one of these set in their shell would
# otherwise see a parity test fail for the wrong reason (e.g. the
# legacy writer firing, the binary entering load-and-exit early-exit
# path, or a crosstalk coefficient from `LOSS_INTRA_CROSSTALK` leaking
# into the General Parameters block on the C++ side while the Python
# side writes `None`).
#
# M8 R1 dual-review (Opus P1-O3 / Codex P1-G) sealed the SINIC_* subset
# in `run_cpp`. R2 dual-review (Opus R2-P1-1 / Codex R2-P1) caught the
# remaining gap: `LOSS_*` / `CROSSTALK_*` are only set conditionally
# by the fixture (when the corresponding field is non-None), so the
# inherited values leak through for non-crosstalk fixtures.
#
# **Update this set when adding any new C++ env var to `main.cpp`** —
# the parity contract is "every C++ env var must be either set by the
# fixture or cleared by the harness".
HERMETIC_ENV_CLEAR_LIST = (
    # Writer mode + orchestration.
    "SINIC_LEGACY_WRITER",
    "SINIC_KEEP_LAST_EVAL_STATE",
    "SINIC_WRITE_RUN_REPORT",
    "SINIC_LOAD_AND_EXIT_JSON",
    "SINIC_FIXED_LAYERS_JSON",
    "SINIC_POSITIONS_JSON",
    "SINIC_SCHEMA_VERSION",
    "SINIC_VALIDATE_SCHEMA",
    "SINIC_SEED",
    # Multi-layer (M6).
    "NUM_LAYERS",
    "EDGE_COUPLER_LAYER",
    "PERIMETER_LAYER",
    "WAVEGUIDES_PER_LINK",
    "LAYER_PITCH_UM",
    # Crosstalk (M7).
    "LOSS_INTRA_CROSSTALK",
    "LOSS_INTER_CROSSTALK",
    "COMPUTE_CROSSTALK",
    "CROSSTALK_MAX_HOPS",
    "CROSSTALK_THRESHOLD_DB",
    "CROSSTALK_INCLUDE_IN_LOSS",
    # Parameter scan (pre-M5).
    "NODES_MIN",
    "NODES_MAX",
    "DUALSA_ITER",
    "CL_VALUES",
    "TL_VALUES",
    "ITL_VALUES",
    "OUTPUT_DIR",
)


def hermetic_env() -> dict[str, str]:
    """Return `os.environ.copy()` with the orchestration-affecting C++
    env vars cleared. Use as the base for every subprocess that spawns
    the C++ binary so the test runs under a fixture-controlled env
    regardless of the developer's shell state.

    Path-like inherits (`PATH` / `DYLD_LIBRARY_PATH` / `HOME` / etc.)
    are intentionally NOT cleared — the binary still needs to locate
    shared libraries and load the schema dir.
    """
    env = os.environ.copy()
    for name in HERMETIC_ENV_CLEAR_LIST:
        env.pop(name, None)
    return env

# Known General-Parameters field-set asymmetries between Python and C++.
# Python writers stamp optimizer-result metadata; the C++ writer emits a
# legacy `Nodes` mirror of `k`. Neither side is wrong but the parity
# comparator strips both before checking key-set equality. The asymmetry
# is intentional and tracked here; if either side moves, this set is the
# single point to update.
KNOWN_GP_ASYMMETRIES = frozenset({
    # Python-only optimizer-result metadata (written via run_optimization
    # **kwargs → save_subgraphs_to_json). C++ does not pass these through
    # `save_subgraphsdata_v1x`. They are redundant with the new
    # run_report.json (M1) so dropping the parity check is safe.
    "optimizer", "maxiter", "seed", "best_loss",
    # C++-only legacy alias of `k`. Pre-M5 main.cpp emitted this; M5
    # kept the field for output/ fixture round-trip. The Python side
    # never wrote it.
    "Nodes",
})


@dataclass
class ParityFixture:
    """A single parity-test configuration.

    Attributes mirror the constructor arguments of both Python
    ``run_optimization`` and the C++ ``SINIC_*`` env-var contract. The
    ``layers`` field is the fixed per-edge layer assignment (length must
    equal ``k * (k - 1) / 2`` since both ends operate on the complete
    graph K_k).
    """

    name: str
    k: int
    L: int
    waveguides_per_link: int
    edge_coupler_layer: int = 0
    perimeter_layer: int | None = None
    layer_pitch_um: float = 1.2
    loss_crossing: float = 0.3
    loss_taper: float = 1.0
    loss_interlayercrossing: float = 0.006
    loss_intralayer_crosstalk: float | None = None
    loss_interlayer_crosstalk: float | None = None
    compute_crosstalk: bool = False
    crosstalk_max_hops: int = 3
    crosstalk_threshold_db: float = -60.0
    schema_version: str = "2.0"
    layers: list[int] = field(default_factory=list)

    def __post_init__(self):
        expected = self.k * (self.k - 1) // 2
        if len(self.layers) != expected:
            raise ValueError(
                f"fixture {self.name!r}: layers has {len(self.layers)} "
                f"entries, expected {expected} (k={self.k} complete graph)"
            )


def _output_subdir(loss_crossing: float, loss_taper: float,
                   loss_interlayercrossing: float, k: int) -> str:
    """Match the folder-name convention used by both Python's ``api.py``
    and C++ ``main.cpp``: ``cl_{cl:.2f}_tl_{tl:.2f}_itl_{itl:.3f}_nodes_{k}``.
    """
    return (
        f"cl_{loss_crossing:.2f}"
        f"_tl_{loss_taper:.2f}"
        f"_itl_{loss_interlayercrossing:.3f}"
        f"_nodes_{k}"
    )


def _shared_positions(k: int) -> dict:
    """Build positions matching the C++ ``distribute_nodes_around_square``
    contract (``side_length=10`` — main.cpp pins this literal). Python's
    own default uses ``side_length=1`` so we override here to avoid a
    10× linear-scale drift that would show up in the ``positions``
    field. Geometric crossings are scale-invariant for cyclic-convex
    SiN layouts so per-edge integer counts agree regardless, but the
    raw position values are part of the JSON contract and must match
    bit-exact in this parity test (REFACTOR_GOALS.md §1-2-b第3条).
    """
    from routing_py_rebuild.positions import distribute_nodes_around_square
    if k % 4 != 0:
        raise ValueError(f"k={k} must be divisible by 4 for square layout")
    return distribute_nodes_around_square(
        nodes_per_side=k // 4,
        side_length=10.0,
    )


def run_python(fixture: ParityFixture, output_dir: Path) -> Path:
    """Drive Python ``run_optimization`` with the fixture and return the
    path to the written ``subgraphsdata.json``.
    """
    # Imported lazily so the harness module is importable even if the
    # Python tree fails to load (e.g. missing optional deps).
    from routing_py_rebuild.api import run_optimization

    res = run_optimization(
        k=fixture.k,
        L=fixture.L,
        edge_coupler_layer=fixture.edge_coupler_layer,
        perimeter_layer=fixture.perimeter_layer,
        layer_pitch_um=fixture.layer_pitch_um,
        waveguides_per_link=fixture.waveguides_per_link,
        loss_crossing=fixture.loss_crossing,
        loss_taper=fixture.loss_taper,
        loss_interlayercrossing=fixture.loss_interlayercrossing,
        loss_intralayer_crosstalk=fixture.loss_intralayer_crosstalk,
        loss_interlayer_crosstalk=fixture.loss_interlayer_crosstalk,
        compute_crosstalk=fixture.compute_crosstalk,
        crosstalk_max_hops=fixture.crosstalk_max_hops,
        crosstalk_threshold_db=fixture.crosstalk_threshold_db,
        positions=_shared_positions(fixture.k),
        output_dir=str(output_dir) + "/",
        save_json=True,
        plot=False,
        run_loss_analysis=False,
        collect_statistics=False,
        fixed_layers=list(fixture.layers),
    )
    return Path(res["json_path"])


def run_cpp(fixture: ParityFixture, output_dir: Path,
            fixed_layers_path: Path) -> Path:
    """Drive the C++ binary with the fixture; returns the path to the
    written ``subgraphsdata.json``.
    """
    if not CPP_BINARY.exists():
        raise FileNotFoundError(
            f"C++ binary not found at {CPP_BINARY}. "
            f"Build it first: `cmake --build {CPP_BUILD_DIR}`."
        )
    # Write fixed-layers JSON to a path the binary can read.
    fixed_layers_path.parent.mkdir(parents=True, exist_ok=True)
    with fixed_layers_path.open("w") as f:
        json.dump({"layers": list(fixture.layers)}, f)

    nodes_per_side = fixture.k // 4
    if nodes_per_side * 4 != fixture.k:
        raise ValueError(
            f"fixture {fixture.name!r}: k={fixture.k} must be divisible "
            f"by 4 (C++ binary uses nodes_per_side scan)"
        )

    env = hermetic_env()
    env.update({
        "NODES_MIN": str(nodes_per_side),
        "NODES_MAX": str(nodes_per_side),
        "DUALSA_ITER": "1",  # ignored — SINIC_FIXED_LAYERS_JSON skips optimize
        "CL_VALUES": str(fixture.loss_crossing),
        "TL_VALUES": str(fixture.loss_taper),
        "ITL_VALUES": str(fixture.loss_interlayercrossing),
        "OUTPUT_DIR": str(output_dir) + "/",
        "NUM_LAYERS": str(fixture.L),
        "EDGE_COUPLER_LAYER": str(fixture.edge_coupler_layer),
        "PERIMETER_LAYER": str(
            fixture.perimeter_layer
            if fixture.perimeter_layer is not None
            else -1
        ),
        "WAVEGUIDES_PER_LINK": str(fixture.waveguides_per_link),
        "LAYER_PITCH_UM": str(fixture.layer_pitch_um),
        "SINIC_FIXED_LAYERS_JSON": str(fixed_layers_path),
        "SINIC_SEED": "0",
        "SINIC_VALIDATE_SCHEMA": "1",
        "SINIC_SCHEMA_VERSION": fixture.schema_version,
        "SINIC_SCHEMA_DIR": str(SCHEMA_DIR),
    })
    if fixture.loss_intralayer_crosstalk is not None:
        env["LOSS_INTRA_CROSSTALK"] = repr(fixture.loss_intralayer_crosstalk)
    if fixture.loss_interlayer_crosstalk is not None:
        env["LOSS_INTER_CROSSTALK"] = repr(fixture.loss_interlayer_crosstalk)
    env["COMPUTE_CROSSTALK"] = "1" if fixture.compute_crosstalk else "0"
    env["CROSSTALK_MAX_HOPS"] = str(fixture.crosstalk_max_hops)
    env["CROSSTALK_THRESHOLD_DB"] = repr(fixture.crosstalk_threshold_db)

    proc = subprocess.run(
        [str(CPP_BINARY)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"C++ binary failed (exit={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    json_path = output_dir / _output_subdir(
        fixture.loss_crossing,
        fixture.loss_taper,
        fixture.loss_interlayercrossing,
        fixture.k,
    ) / "subgraphsdata.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"C++ output JSON not found at {json_path}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json_path


@dataclass
class ParityReport:
    """Result of comparing a Python JSON to a C++ JSON.

    Empty ``differences`` ≡ parity passed. Each entry is a human-readable
    string suitable for ``pytest.fail`` output.
    """

    differences: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.differences

    def __bool__(self) -> bool:
        return self.passed

    def format(self) -> str:
        if self.passed:
            return "PARITY OK"
        return "PARITY FAILED:\n  - " + "\n  - ".join(self.differences)


def _close_enough(py_val: float, cpp_val: float,
                  abs_tol: float = ABS_TOL,
                  rel_tol: float = REL_TOL) -> bool:
    """REFACTOR_GOALS.md §1-2-b: pass if either absolute or relative
    tolerance is met. NaN ↔ NaN is treated as equal so the engine's
    diagonal-null convention round-trips cleanly.
    """
    if py_val is None and cpp_val is None:
        return True
    if py_val is None or cpp_val is None:
        return False
    if isinstance(py_val, float) and math.isnan(py_val):
        return isinstance(cpp_val, float) and math.isnan(cpp_val)
    if isinstance(cpp_val, float) and math.isnan(cpp_val):
        return False
    return math.isclose(
        py_val, cpp_val,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )


# Convention-normalization hook (§6 T6 "convention 不同时先归一化")
# intentionally NOT defined as a no-op function here: the current M8
# fixture matrix feeds the SAME `waveguides_per_link` to both ends so
# `crosslayer_crossings_convention` always matches. If a future PR
# introduces a fixture that pins Python `wpl=1` against C++ `wpl=2`
# (or vice versa), the right move is to extend `compare_json` with an
# explicit pre-comparison rescale rather than re-add a stub. Keeping
# the hook absent means a cross-convention regression raises a noisy
# diff instead of being silently rescaled — that's the documented
# §6 T6 stance.


def _gp_filter(gp: dict, *, side: str) -> dict:
    """Strip known asymmetric General-Parameters fields before comparison."""
    if side not in ("py", "cpp"):
        raise ValueError(f"side must be 'py' or 'cpp', got {side!r}")
    return {k: v for k, v in gp.items() if k not in KNOWN_GP_ASYMMETRIES}


def _compare_floats(label: str, py_val: Any, cpp_val: Any,
                    diffs: list[str]) -> None:
    if py_val is None and cpp_val is None:
        return
    if py_val is None or cpp_val is None:
        diffs.append(f"{label}: py={py_val!r} vs cpp={cpp_val!r}")
        return
    if isinstance(py_val, bool) or isinstance(cpp_val, bool):
        if py_val != cpp_val:
            diffs.append(f"{label}: py={py_val!r} vs cpp={cpp_val!r}")
        return
    if isinstance(py_val, (int,)) and isinstance(cpp_val, (int,)):
        if py_val != cpp_val:
            diffs.append(
                f"{label}: integer mismatch py={py_val} vs cpp={cpp_val}"
            )
        return
    if not _close_enough(float(py_val), float(cpp_val)):
        diffs.append(
            f"{label}: py={py_val!r} vs cpp={cpp_val!r} "
            f"(abs Δ={abs(float(py_val) - float(cpp_val)):.3e})"
        )


def _compare_edge_lists(label: str, py_edges: list, cpp_edges: list,
                        diffs: list[str]) -> None:
    if len(py_edges) != len(cpp_edges):
        diffs.append(
            f"{label}: edge count py={len(py_edges)} vs cpp={len(cpp_edges)}"
        )
        return

    def _key(e):
        return (int(e[0]), int(e[1]))

    py_map = {_key(e): e[2] for e in py_edges}
    cpp_map = {_key(e): e[2] for e in cpp_edges}
    if set(py_map) != set(cpp_map):
        missing_py = set(cpp_map) - set(py_map)
        missing_cpp = set(py_map) - set(cpp_map)
        diffs.append(
            f"{label}: edge sets differ: only-in-cpp={sorted(missing_py)[:3]} "
            f"only-in-py={sorted(missing_cpp)[:3]}"
        )
        return
    # M8 R2 dual-review (Opus R2-P2-1): per-edge integer keys are not in
    # the JSON schema's `required` list (only `edges` is on `Layer_*`),
    # so `if k in pa or k in ca` would silently pass if BOTH sides
    # dropped a key. Apply the same "must be present on both sides"
    # posture as the top-level R1 P1-B fix — per-edge integer counters
    # are part of the §1-2-b第2条 bit-exact contract.
    integer_keys = (
        "layer",
        "crossings",
        "interlayercrossings",
        "interlayercrossings_above",
        "interlayercrossings_below",
    )
    for ek in sorted(py_map):
        pa, ca = py_map[ek], cpp_map[ek]
        # Per-edge integer counts must be **bit-exact** AND **present**
        # on both sides (§1-2-b第2条 + R2 P2-1).
        for k in integer_keys:
            in_py = k in pa
            in_cpp = k in ca
            if in_py != in_cpp:
                diffs.append(
                    f"{label}/edge{ek}/{k}: presence mismatch "
                    f"py={in_py} cpp={in_cpp}"
                )
                continue
            if in_py and pa.get(k) != ca.get(k):
                diffs.append(
                    f"{label}/edge{ek}/{k}: py={pa.get(k)} vs cpp={ca.get(k)}"
                )
        # Per-edge loss is float — tolerance-compare.
        _compare_floats(
            f"{label}/edge{ek}/loss",
            pa.get("loss"),
            ca.get("loss"),
            diffs,
        )


def _compare_loss_analysis(py_la: dict, cpp_la: dict,
                           diffs: list[str]) -> None:
    if set(py_la.keys()) != set(cpp_la.keys()):
        diffs.append(
            f"loss_analysis keys differ: py={sorted(py_la.keys())} "
            f"cpp={sorted(cpp_la.keys())}"
        )
        return
    for k in sorted(py_la.keys()):
        py_v = py_la[k]
        cpp_v = cpp_la[k]
        if isinstance(py_v, list) and isinstance(cpp_v, list):
            if len(py_v) != len(cpp_v):
                diffs.append(
                    f"loss_analysis/{k}: array length py={len(py_v)} "
                    f"vs cpp={len(cpp_v)}"
                )
                continue
            for i, (pv, cv) in enumerate(zip(py_v, cpp_v, strict=True)):
                _compare_floats(
                    f"loss_analysis/{k}[{i}]", pv, cv, diffs
                )
        else:
            _compare_floats(f"loss_analysis/{k}", py_v, cpp_v, diffs)


def _compare_crosstalk(py_xt, cpp_xt, diffs: list[str]) -> None:
    if py_xt is None and cpp_xt is None:
        return
    if py_xt is None or cpp_xt is None:
        diffs.append(
            f"crosstalk presence mismatch: py={py_xt is not None} "
            f"cpp={cpp_xt is not None}"
        )
        return
    # M8 dual-review P1 (Codex P1-C3 / Opus P1-O4): the crosstalk
    # payload's own key-set must match before we compare values, so a
    # future divergence (e.g. one side dropping `unit`) doesn't slip
    # through as a `None == None` pseudo-equality. Mirrors the same
    # check applied to `loss_analysis` higher up.
    if set(py_xt.keys()) != set(cpp_xt.keys()):
        diffs.append(
            f"crosstalk keys differ: only-in-py="
            f"{sorted(set(py_xt.keys()) - set(cpp_xt.keys()))} "
            f"only-in-cpp="
            f"{sorted(set(cpp_xt.keys()) - set(py_xt.keys()))}"
        )
    # M7 §3-1 contract requires `unit` / `shape` / `values` to be
    # present on both sides (writer guarantees this). Flag missing
    # presence explicitly rather than letting `.get(...)` silently
    # default to None ≡ None.
    for required in ("unit", "shape", "values"):
        if required not in py_xt or required not in cpp_xt:
            diffs.append(
                f"crosstalk required key {required!r} missing: "
                f"py={required in py_xt} cpp={required in cpp_xt}"
            )
    for key in ("unit", "shape", "coherence_model", "polarization",
                "symmetric", "diagonal_convention"):
        if py_xt.get(key) != cpp_xt.get(key):
            diffs.append(
                f"crosstalk/{key}: py={py_xt.get(key)!r} vs "
                f"cpp={cpp_xt.get(key)!r}"
            )
    # max_hops / threshold_db are scalar metadata.
    _compare_floats(
        "crosstalk/threshold_db",
        py_xt.get("threshold_db"),
        cpp_xt.get("threshold_db"),
        diffs,
    )
    if py_xt.get("max_hops") != cpp_xt.get("max_hops"):
        diffs.append(
            f"crosstalk/max_hops: py={py_xt.get('max_hops')} vs "
            f"cpp={cpp_xt.get('max_hops')}"
        )
    # M8 P1-O4: compare the crosstalk-payload coefficient mirrors so a
    # future drift between `graph.loss_*_crosstalk` and the writer's
    # payload doesn't slip silently.
    for coef_key in ("loss_intralayer_crosstalk", "loss_interlayer_crosstalk"):
        _compare_floats(
            f"crosstalk/{coef_key}",
            py_xt.get(coef_key),
            cpp_xt.get(coef_key),
            diffs,
        )
    py_vals = py_xt.get("values")
    cpp_vals = cpp_xt.get("values")
    if py_vals is None or cpp_vals is None:
        diffs.append(
            f"crosstalk/values: presence mismatch py={py_vals is not None} "
            f"cpp={cpp_vals is not None}"
        )
        return
    if len(py_vals) != len(cpp_vals):
        diffs.append(
            f"crosstalk/values: top-level length py={len(py_vals)} "
            f"vs cpp={len(cpp_vals)}"
        )
        return
    for s, (py_slab, cpp_slab) in enumerate(
        zip(py_vals, cpp_vals, strict=True)
    ):
        if len(py_slab) != len(cpp_slab):
            diffs.append(
                f"crosstalk/values[{s}]: length py={len(py_slab)} "
                f"vs cpp={len(cpp_slab)}"
            )
            continue
        for d_idx, (py_row, cpp_row) in enumerate(
            zip(py_slab, cpp_slab, strict=True)
        ):
            if len(py_row) != len(cpp_row):
                diffs.append(
                    f"crosstalk/values[{s}][{d_idx}]: length py={len(py_row)} "
                    f"vs cpp={len(cpp_row)}"
                )
                continue
            for t_idx, (pv, cv) in enumerate(
                zip(py_row, cpp_row, strict=True)
            ):
                _compare_floats(
                    f"crosstalk/values[{s}][{d_idx}][{t_idx}]",
                    pv, cv, diffs,
                )


def compare_json(py_path: Path, cpp_path: Path) -> ParityReport:
    """Compare a Python-produced ``subgraphsdata.json`` against a C++
    one. Both sides are expected to have been driven from the **same**
    ``ParityFixture`` (identical fixed layer assignment, identical
    physical params).
    """
    with open(py_path) as f:
        py = json.load(f)
    with open(cpp_path) as f:
        cpp = json.load(f)

    diffs: list[str] = []

    # M8 dual-review P1 (Codex P1-C2): explicit presence check for the
    # `subgraphsdata.schema.json` required top-level fields. Without
    # this, a future regression that drops `complete_graph` on BOTH
    # sides would slip through the set-equality check (set(a) == set(b)
    # is true when both are missing the field). For v2.0 we also check
    # the v2.0-mandatory General Parameters fields per
    # `subgraphsdata.schema.json::allOf.if/then`.
    for required_top in ("General Parameters", "positions",
                         "complete_graph", "loss_analysis"):
        for side_name, side_data in (("py", py), ("cpp", cpp)):
            if required_top not in side_data:
                diffs.append(
                    f"required top-level key {required_top!r} missing "
                    f"on {side_name}"
                )

    # Top-level structural equality.
    if set(py.keys()) != set(cpp.keys()):
        diffs.append(
            f"top-level keys differ: only-in-py="
            f"{sorted(set(py.keys()) - set(cpp.keys()))} "
            f"only-in-cpp={sorted(set(cpp.keys()) - set(py.keys()))}"
        )

    # schema_version exact match.
    if py.get("schema_version") != cpp.get("schema_version"):
        diffs.append(
            f"schema_version: py={py.get('schema_version')!r} "
            f"vs cpp={cpp.get('schema_version')!r}"
        )

    # M8 dual-review P1: v2.0 General Parameters required-field
    # presence. Schema enforces these via allOf.if/then, but a writer
    # that drops them on BOTH sides would pass the strict-key equality
    # check below — explicit presence assertion seals that gap.
    if py.get("schema_version") == "2.0":
        for gp_required in ("k", "L", "edge_coupler_layer",
                            "perimeter_layer", "waveguides_per_link",
                            "layer_pitch_um"):
            for side_name, side_data in (("py", py), ("cpp", cpp)):
                gp = side_data.get("General Parameters", {})
                if gp_required not in gp:
                    diffs.append(
                        f"v2.0 General Parameters required key "
                        f"{gp_required!r} missing on {side_name}"
                    )

    # General Parameters — strip known asymmetries, then strict compare.
    py_gp = _gp_filter(py.get("General Parameters", {}), side="py")
    cpp_gp = _gp_filter(cpp.get("General Parameters", {}), side="cpp")
    if set(py_gp.keys()) != set(cpp_gp.keys()):
        diffs.append(
            f"General Parameters keys differ (after stripping "
            f"{sorted(KNOWN_GP_ASYMMETRIES)}): "
            f"py={sorted(py_gp.keys())} cpp={sorted(cpp_gp.keys())}"
        )
    for k in sorted(set(py_gp.keys()) & set(cpp_gp.keys())):
        if isinstance(py_gp[k], (int, str, bool)) or py_gp[k] is None:
            if py_gp[k] != cpp_gp[k]:
                diffs.append(
                    f"General Parameters/{k}: py={py_gp[k]!r} "
                    f"vs cpp={cpp_gp[k]!r}"
                )
        else:
            _compare_floats(
                f"General Parameters/{k}", py_gp[k], cpp_gp[k], diffs
            )

    # positions — both are k entries of [x, y]; must match exactly because
    # both sides derive from the same `distribute_nodes_around_square`.
    py_pos = py.get("positions", {})
    cpp_pos = cpp.get("positions", {})
    if set(py_pos.keys()) != set(cpp_pos.keys()):
        diffs.append(
            f"positions keys differ: only-py="
            f"{sorted(set(py_pos.keys()) - set(cpp_pos.keys()))[:3]} "
            f"only-cpp="
            f"{sorted(set(cpp_pos.keys()) - set(py_pos.keys()))[:3]}"
        )
    for k in sorted(set(py_pos.keys()) & set(cpp_pos.keys())):
        for axis_idx in range(2):
            _compare_floats(
                f"positions/{k}[{axis_idx}]",
                py_pos[k][axis_idx],
                cpp_pos[k][axis_idx],
                diffs,
            )

    # Layer_i edge lists.
    layer_keys_py = sorted(k for k in py if k.startswith("Layer_"))
    layer_keys_cpp = sorted(k for k in cpp if k.startswith("Layer_"))
    if layer_keys_py != layer_keys_cpp:
        diffs.append(
            f"Layer_* keys differ: py={layer_keys_py} vs cpp={layer_keys_cpp}"
        )
    for lk in sorted(set(layer_keys_py) & set(layer_keys_cpp)):
        _compare_edge_lists(lk, py[lk]["edges"], cpp[lk]["edges"], diffs)

    # complete_graph — list of [u, v, attrs] for all edges; depends on
    # `distribute_nodes_around_square` so positions agreement implies
    # complete_graph edge set is fixed; only attrs (crossings, etc.) vary
    # with layer assignment, but since `complete_graph` is the planar
    # reference (pre-layer-split) we expect bit-exact integer counts.
    _compare_edge_lists(
        "complete_graph",
        py.get("complete_graph", []),
        cpp.get("complete_graph", []),
        diffs,
    )

    # loss_analysis — schema-permitted variable shape; compare cells.
    _compare_loss_analysis(
        py.get("loss_analysis", {}),
        cpp.get("loss_analysis", {}),
        diffs,
    )

    # Optional crosstalk tensor.
    _compare_crosstalk(py.get("crosstalk"), cpp.get("crosstalk"), diffs)

    return ParityReport(differences=diffs)


def write_fixture_spec(fixture: ParityFixture, path: Path) -> None:
    """Persist a fixture's *spec* (config + layer assignment) to JSON.
    Useful so the parity test's expectations are reproducible from disk
    and reviewable in PRs.
    """
    payload = {
        "name": fixture.name,
        "k": fixture.k,
        "L": fixture.L,
        "waveguides_per_link": fixture.waveguides_per_link,
        "edge_coupler_layer": fixture.edge_coupler_layer,
        "perimeter_layer": fixture.perimeter_layer,
        "layer_pitch_um": fixture.layer_pitch_um,
        "loss_crossing": fixture.loss_crossing,
        "loss_taper": fixture.loss_taper,
        "loss_interlayercrossing": fixture.loss_interlayercrossing,
        "loss_intralayer_crosstalk": fixture.loss_intralayer_crosstalk,
        "loss_interlayer_crosstalk": fixture.loss_interlayer_crosstalk,
        "compute_crosstalk": fixture.compute_crosstalk,
        "crosstalk_max_hops": fixture.crosstalk_max_hops,
        "crosstalk_threshold_db": fixture.crosstalk_threshold_db,
        "schema_version": fixture.schema_version,
        "layers": list(fixture.layers),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_fixture_spec(path: Path) -> ParityFixture:
    with path.open() as f:
        spec = json.load(f)
    return ParityFixture(**spec)


__all__ = [
    "ABS_TOL", "REL_TOL", "KNOWN_GP_ASYMMETRIES",
    "HERMETIC_ENV_CLEAR_LIST", "hermetic_env",
    "ParityFixture", "ParityReport",
    "run_python", "run_cpp", "compare_json",
    "write_fixture_spec", "read_fixture_spec",
]
