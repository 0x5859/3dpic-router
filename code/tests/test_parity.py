"""Python ↔ C++ parity test (REFACTOR_GOALS.md M8 / §6 T6).

For each fixture under ``tests/golden/*.json`` we drive **both**
``routing_py_rebuild.run_optimization`` and the C++ ``Autowiring_CPP``
binary with the same fixed per-edge layer assignment, then compare the
two ``subgraphsdata.json`` outputs through the parity comparator in
``tests/parity_harness.py``.

Failure modes (each is a single ``pytest`` failure):

* Top-level JSON shape disagreement (key set / nesting).
* Per-edge integer count mismatch (``crossings`` /
  ``interlayercrossings_above`` etc.) — these are required to be
  bit-exact per §1-2-b第2条.
* Per-edge or aggregate float loss outside the §1-2-b tolerance
  (abs ``1e-8`` or rel ``1e-10``, either passes).
* Crosstalk tensor shape / per-cell drift outside tolerance.

Skip semantics: if the C++ binary is missing the entire module is
skipped with a hint to rebuild. Individual fixtures are not skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.parity_harness import (
    CPP_BINARY,
    ParityFixture,
    compare_json,
    hermetic_env,
    read_fixture_spec,
    run_cpp,
    run_python,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _collect_fixtures() -> list[ParityFixture]:
    fixtures: list[ParityFixture] = []
    for spec_path in sorted(GOLDEN_DIR.glob("*.json")):
        fixtures.append(read_fixture_spec(spec_path))
    if not fixtures:
        raise RuntimeError(
            f"No fixtures found under {GOLDEN_DIR}. Run "
            "`uv run python code/tests/golden/generate_fixtures.py` first."
        )
    return fixtures


_FIXTURES = _collect_fixtures()


@pytest.fixture(scope="session", autouse=True)
def _require_cpp_binary():
    if not CPP_BINARY.exists():
        pytest.skip(
            f"C++ binary not built at {CPP_BINARY}. "
            "Run `cmake --build code/routing_cpp_rebuild/build` "
            "before invoking the parity tests."
        )


@pytest.mark.parametrize(
    "fixture",
    _FIXTURES,
    ids=[f.name for f in _FIXTURES],
)
def test_python_cpp_parity(fixture: ParityFixture, tmp_path: Path) -> None:
    py_out = tmp_path / "py"
    cpp_out = tmp_path / "cpp"
    py_out.mkdir()
    cpp_out.mkdir()
    fixed_layers_path = tmp_path / "fixed_layers.json"

    py_json = run_python(fixture, py_out)
    cpp_json = run_cpp(fixture, cpp_out, fixed_layers_path)
    report = compare_json(py_json, cpp_json)
    if not report.passed:
        pytest.fail(
            f"Parity check failed for fixture {fixture.name!r}.\n"
            f"  py_json: {py_json}\n"
            f"  cpp_json: {cpp_json}\n"
            f"{report.format()}"
        )


def test_compute_crosstalk_zero_flag_short_circuits(tmp_path: Path) -> None:
    """REFACTOR_GOALS.md M7 deferred item: explicit
    ``COMPUTE_CROSSTALK=0`` must skip the crosstalk pass even when
    coefficients are non-zero. This was marked "delay to M8 because the
    orchestration logic is binary-only and needs a subprocess fixture"
    in the M7 design-tradeoffs section; the parity harness gives us
    exactly that.
    """
    import json

    # Use the existing crosstalk fixture but force the env flag off.
    fixture = next(f for f in _FIXTURES if f.compute_crosstalk)

    out = tmp_path / "cpp"
    out.mkdir()
    fixed_layers_path = tmp_path / "layers.json"
    fixed_layers_path.parent.mkdir(parents=True, exist_ok=True)
    with fixed_layers_path.open("w") as f:
        json.dump({"layers": list(fixture.layers)}, f)

    env = hermetic_env()
    nodes_per_side = fixture.k // 4
    env.update({
        "NODES_MIN": str(nodes_per_side),
        "NODES_MAX": str(nodes_per_side),
        "DUALSA_ITER": "1",
        "CL_VALUES": str(fixture.loss_crossing),
        "TL_VALUES": str(fixture.loss_taper),
        "ITL_VALUES": str(fixture.loss_interlayercrossing),
        "OUTPUT_DIR": str(out) + "/",
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
        # Coefficients ARE set — but COMPUTE_CROSSTALK=0 overrides.
        "LOSS_INTRA_CROSSTALK": repr(fixture.loss_intralayer_crosstalk),
        "LOSS_INTER_CROSSTALK": repr(fixture.loss_interlayer_crosstalk),
        "COMPUTE_CROSSTALK": "0",
        "CROSSTALK_MAX_HOPS": str(fixture.crosstalk_max_hops),
        "CROSSTALK_THRESHOLD_DB": repr(fixture.crosstalk_threshold_db),
    })

    import subprocess
    proc = subprocess.run(
        [str(CPP_BINARY)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"C++ binary failed:\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    folder = (
        f"cl_{fixture.loss_crossing:.2f}"
        f"_tl_{fixture.loss_taper:.2f}"
        f"_itl_{fixture.loss_interlayercrossing:.3f}"
        f"_nodes_{fixture.k}"
    )
    out_json = out / folder / "subgraphsdata.json"
    assert out_json.exists()
    with out_json.open() as f:
        data = json.load(f)
    # The whole point: explicit COMPUTE_CROSSTALK=0 ⇒ no top-level
    # ``crosstalk`` field, even though coefficients are set. Mirrors the
    # Python path (api.py `compute_crosstalk=False` short-circuit).
    assert "crosstalk" not in data, (
        "COMPUTE_CROSSTALK=0 must suppress the crosstalk field even "
        "when LOSS_INTRA/INTER_CROSSTALK are non-zero. Got "
        f"keys={sorted(data.keys())}"
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["k12_L2_wpl2_xtalk", "k12_L3_wpl2_v2", "k12_L2_wpl1_v2"],
)
def test_each_side_is_deterministic(fixture_name: str, tmp_path: Path) -> None:
    """Run the same fixture twice on each side and assert byte-identical
    JSON output. M8 dual-review (Codex P1-C5 / Opus P1-O7) sharpened
    the docstring on what this actually catches:

    **What this DOES catch**: any source of intra-implementation
    non-determinism — uninitialized memory, hash-map iteration order
    leaking into JSON serialization, std::random_device fallback when
    a seed should be deterministic, unsynchronized concurrent writes
    to per-edge state. The L=3 fixture exercises multi-layer
    create_subgraphs ordering + loss_analysis aggregation; the
    wpl=1 fixture exercises the convention='geometric' aggregate
    formula; the crosstalk fixture exercises the DFS path-tree.

    **What this does NOT catch**: the specific `std::sort` vs
    `std::stable_sort` hazard from M7 dual-review R2 codex P1. That
    hazard only manifests when `compute_crosstalk_tensor` sees
    ``t_self`` ties on `per-edge` crossings, which requires 3+
    collinear edges through a single point. Cyclic-convex K_k SiN
    layouts never produce that geometry under exact arithmetic; the
    in-tree C++ unit test ``compute_crosstalk_tensor is deterministic``
    similarly only verifies same-side reruns. Catching the stable_sort
    regression requires a property-based fixture with synthetic
    positions that force ties (deferred until a real geometry change
    motivates it).
    """
    fixture = next(f for f in _FIXTURES if f.name == fixture_name)

    py_out_a = tmp_path / "py_a"; py_out_a.mkdir()
    py_out_b = tmp_path / "py_b"; py_out_b.mkdir()
    cpp_out_a = tmp_path / "cpp_a"; cpp_out_a.mkdir()
    cpp_out_b = tmp_path / "cpp_b"; cpp_out_b.mkdir()

    py_a = run_python(fixture, py_out_a)
    py_b = run_python(fixture, py_out_b)
    cpp_a = run_cpp(fixture, cpp_out_a, tmp_path / "layers_a.json")
    cpp_b = run_cpp(fixture, cpp_out_b, tmp_path / "layers_b.json")

    assert py_a.read_bytes() == py_b.read_bytes(), (
        f"Python writer is non-deterministic for fixture {fixture_name!r}"
    )
    assert cpp_a.read_bytes() == cpp_b.read_bytes(), (
        f"C++ writer is non-deterministic for fixture {fixture_name!r}"
    )


def test_cpp_fixed_layers_run_report_mutual_exclusion(tmp_path: Path) -> None:
    """M8 dual-review P1-D (Codex P1-C4): the C++ binary must raise when
    ``SINIC_FIXED_LAYERS_JSON`` is set alongside ``SINIC_WRITE_RUN_REPORT``,
    symmetric with Python ``api.py::run_optimization``'s raise on
    ``fixed_layers + collect_statistics``. An empty-trace run_report
    would slip through schema validation but is semantically misleading.
    """
    import json
    import subprocess

    fixture = next(f for f in _FIXTURES if f.name == "k12_L2_wpl2_v2")

    out = tmp_path / "cpp_out"
    out.mkdir()
    fixed_layers_path = tmp_path / "layers.json"
    with fixed_layers_path.open("w") as f:
        json.dump({"layers": list(fixture.layers)}, f)

    env = hermetic_env()
    nodes_per_side = fixture.k // 4
    env.update({
        "NODES_MIN": str(nodes_per_side),
        "NODES_MAX": str(nodes_per_side),
        "DUALSA_ITER": "1",
        "CL_VALUES": str(fixture.loss_crossing),
        "TL_VALUES": str(fixture.loss_taper),
        "ITL_VALUES": str(fixture.loss_interlayercrossing),
        "OUTPUT_DIR": str(out) + "/",
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
        "SINIC_WRITE_RUN_REPORT": "1",
        "SINIC_SEED": "0",
        "SINIC_VALIDATE_SCHEMA": "1",
    })
    proc = subprocess.run(
        [str(CPP_BINARY)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, (
        "C++ binary should reject SINIC_FIXED_LAYERS_JSON + "
        "SINIC_WRITE_RUN_REPORT combo (empty-trace run_report is "
        "semantically misleading). Mirrors Python `run_optimization`'s "
        f"fixed_layers + collect_statistics raise.\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    combined = (proc.stdout + proc.stderr).lower()
    assert "mutually exclusive" in combined, (
        "C++ error message should explain the mutual-exclusion contract; "
        f"got:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_cpp_fixed_layers_keep_last_eval_state_mutual_exclusion(
    tmp_path: Path,
) -> None:
    """M8 R2 dual-review P2-3: the C++ mutual-exclusion guard covers
    BOTH ``SINIC_WRITE_RUN_REPORT`` (tested above) AND
    ``SINIC_KEEP_LAST_EVAL_STATE``. The R1 land covered only the
    former; this test seals the second branch.
    """
    import json
    import subprocess

    fixture = next(f for f in _FIXTURES if f.name == "k12_L2_wpl2_v2")
    out = tmp_path / "cpp_out"
    out.mkdir()
    fixed_layers_path = tmp_path / "layers.json"
    with fixed_layers_path.open("w") as f:
        json.dump({"layers": list(fixture.layers)}, f)

    env = hermetic_env()
    nodes_per_side = fixture.k // 4
    env.update({
        "NODES_MIN": str(nodes_per_side),
        "NODES_MAX": str(nodes_per_side),
        "DUALSA_ITER": "1",
        "CL_VALUES": str(fixture.loss_crossing),
        "TL_VALUES": str(fixture.loss_taper),
        "ITL_VALUES": str(fixture.loss_interlayercrossing),
        "OUTPUT_DIR": str(out) + "/",
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
        "SINIC_KEEP_LAST_EVAL_STATE": "1",
        "SINIC_SEED": "0",
        "SINIC_VALIDATE_SCHEMA": "1",
    })
    proc = subprocess.run(
        [str(CPP_BINARY)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, (
        "C++ binary should reject SINIC_FIXED_LAYERS_JSON + "
        "SINIC_KEEP_LAST_EVAL_STATE combo (fixed layers IS the desired "
        f"post-optimize state).\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    combined = (proc.stdout + proc.stderr).lower()
    assert "mutually exclusive" in combined, (
        f"C++ error message should mention 'mutually exclusive'; got:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_cpp_load_and_exit_failure_modes(tmp_path: Path) -> None:
    """M8 R2 dual-review P2-4: the ``SINIC_LOAD_AND_EXIT_JSON`` happy
    path is exercised by ``test_cpp_loads_python_json``; this test
    covers the negative paths so a future refactor that accidentally
    short-circuits the validation (e.g. ``validate=false``) regresses
    visibly.

    Two cases:
      * file-not-found → exit=1 + "load failed: " in stderr
      * malformed JSON → exit=1 + parse-error message
    """
    import subprocess

    # Case 1: file does not exist.
    env_missing = hermetic_env()
    env_missing["SINIC_LOAD_AND_EXIT_JSON"] = str(tmp_path / "does_not_exist.json")
    proc_missing = subprocess.run(
        [str(CPP_BINARY)],
        env=env_missing,
        capture_output=True,
        text=True,
    )
    assert proc_missing.returncode == 1, (
        f"missing-file LOAD_AND_EXIT should exit=1, got "
        f"{proc_missing.returncode}.\nstderr:\n{proc_missing.stderr}"
    )
    assert "load failed" in (proc_missing.stdout + proc_missing.stderr).lower(), (
        f"missing-file LOAD_AND_EXIT should emit 'load failed' diagnostic; "
        f"got:\nstdout:\n{proc_missing.stdout}\nstderr:\n{proc_missing.stderr}"
    )

    # Case 2: malformed JSON.
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not valid json at all}")
    env_malformed = hermetic_env()
    env_malformed["SINIC_LOAD_AND_EXIT_JSON"] = str(malformed)
    proc_malformed = subprocess.run(
        [str(CPP_BINARY)],
        env=env_malformed,
        capture_output=True,
        text=True,
    )
    assert proc_malformed.returncode == 1, (
        f"malformed-JSON LOAD_AND_EXIT should exit=1, got "
        f"{proc_malformed.returncode}.\nstderr:\n{proc_malformed.stderr}"
    )


def test_cpp_loads_python_json(tmp_path: Path) -> None:
    """Inverse of ``test_python_loads_cpp_json``: verifies that C++
    ``load_subgraphsdata`` accepts a Python-produced ``subgraphsdata.json``.

    M8 dual-review (Codex P1-C1 / Opus P1-O2): §1-2-b第3条 calls for
    bidirectional cross-load. The M5 self-roundtrip covered C++ → C++
    only; ``test_python_loads_cpp_json`` (added in M8) covered C++ →
    Python via PlotData; this test closes the remaining Python → C++
    direction.

    Mechanism: spawn the C++ binary with ``SINIC_LOAD_AND_EXIT_JSON``
    pointing at the Python JSON. The binary calls
    ``load_subgraphsdata(path, validate=true)`` (which includes
    schema validation against ``code/schema/subgraphsdata.schema.json``)
    and exits 0 on success / 1 on failure.
    """
    import subprocess

    # Pick the v2.0 wpl=2 fixture as the canonical sample.
    fixture = next(f for f in _FIXTURES if f.name == "k12_L2_wpl2_v2")
    py_out = tmp_path / "py"
    py_out.mkdir()
    py_json = run_python(fixture, py_out)

    env = hermetic_env()
    env["SINIC_LOAD_AND_EXIT_JSON"] = str(py_json)
    env["SINIC_SCHEMA_DIR"] = str(
        Path(__file__).resolve().parent.parent / "schema"
    )

    proc = subprocess.run(
        [str(CPP_BINARY)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"C++ load of Python JSON failed (exit={proc.returncode}).\n"
        f"py_json: {py_json}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    assert "schema_version=2.0" in proc.stdout, (
        f"C++ load did not report v2.0 schema_version. stdout:\n{proc.stdout}"
    )


def test_legacy_v1x_reader_parity(tmp_path: Path) -> None:
    """v1.x reader parity (REFACTOR_GOALS.md §6 T6 "legacy v1.x" leg).

    The v1.x writer-parity slot is intentionally absent — the Python
    writer hard-codes ``schema_version="2.0"`` since M3, so there is
    nothing on the Python side to compare a C++ v1.x output against.
    This test instead validates the **reader** invariant: a v1.x file
    produced by the C++ binary (via ``SINIC_SCHEMA_VERSION=1.0``) must
    load cleanly through Python's ``PlotData.from_json`` with the
    §3-1 兼容性策略 soft-fallback rules (missing ``_above``/``_below``
    treated as ``nan``; ``waveguides_per_link`` → 1;
    ``crosslayer_crossings_convention`` → ``geometric``).
    """
    import json
    from routing_py_rebuild.plotting import PlotData

    # Reuse the wpl=1 fixture's layer assignment but drive C++ in v1.x
    # mode. We can't go through ``run_cpp`` because that one pins
    # ``SINIC_SCHEMA_VERSION`` to the fixture's value; instead we call
    # the subprocess directly with an override.
    fixture = next(f for f in _FIXTURES if f.name == "k12_L2_wpl1_v2")
    out_dir = tmp_path / "cpp_v1x"
    out_dir.mkdir()
    fixed_layers_path = tmp_path / "layers.json"
    with fixed_layers_path.open("w") as f:
        json.dump({"layers": list(fixture.layers)}, f)

    env = hermetic_env()
    nodes_per_side = fixture.k // 4
    env.update({
        "NODES_MIN": str(nodes_per_side),
        "NODES_MAX": str(nodes_per_side),
        "DUALSA_ITER": "1",
        "CL_VALUES": str(fixture.loss_crossing),
        "TL_VALUES": str(fixture.loss_taper),
        "ITL_VALUES": str(fixture.loss_interlayercrossing),
        "OUTPUT_DIR": str(out_dir) + "/",
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
        "SINIC_SCHEMA_VERSION": "1.0",
    })

    import subprocess
    proc = subprocess.run(
        [str(CPP_BINARY)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"C++ v1.x binary failed:\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    folder = (
        f"cl_{fixture.loss_crossing:.2f}"
        f"_tl_{fixture.loss_taper:.2f}"
        f"_itl_{fixture.loss_interlayercrossing:.3f}"
        f"_nodes_{fixture.k}"
    )
    out_json = out_dir / folder / "subgraphsdata.json"
    assert out_json.exists()

    with out_json.open() as f:
        raw = json.load(f)
    assert raw["schema_version"] == "1.0", (
        f"expected v1.x schema_version='1.0', got {raw.get('schema_version')!r}"
    )

    # Python loads it via the soft-fallback path.
    pd = PlotData.from_json(str(out_json))
    assert pd.general_params["k"] == fixture.k
    # §3-1 兼容性策略: v1.x soft-defaults `waveguides_per_link` to 1.
    # M8 dual-review P2-O2: use strict subscript (not `.get(..., 1)`)
    # so we actually verify PlotData.from_json wrote the soft default
    # rather than vacuously matching a missing key against our own
    # default.
    assert pd.general_params["waveguides_per_link"] == 1
    # All edges accounted for after PlotData reconstruction.
    total_edges = sum(g.number_of_edges() for g in pd.layers)
    assert total_edges == fixture.k * (fixture.k - 1) // 2


def test_python_loads_cpp_json(tmp_path: Path) -> None:
    """``PlotData.from_json`` must successfully load a C++-produced
    v2.0 ``subgraphsdata.json`` (REFACTOR_GOALS.md §1-2-b第3条; M5
    validation deferred this to M8).
    """
    from routing_py_rebuild.plotting import PlotData

    fixture = next(f for f in _FIXTURES if f.name == "k12_L2_wpl2_v2")
    cpp_out = tmp_path / "cpp"
    cpp_out.mkdir()
    fixed_layers_path = tmp_path / "layers.json"
    cpp_json = run_cpp(fixture, cpp_out, fixed_layers_path)

    pd = PlotData.from_json(str(cpp_json))
    assert pd.general_params["k"] == fixture.k
    assert pd.general_params["L"] == fixture.L
    assert pd.general_params["waveguides_per_link"] == fixture.waveguides_per_link
    # Layer_0 / Layer_1 buckets must exist with edges populated.
    assert len(pd.layers) == fixture.L
    # `pd.layers` is a list of nx.Graph — use `number_of_edges()` rather
    # than `len()` (which returns node count). Sum across all layer
    # buckets recovers the K_k complete graph's edge count.
    total_edges = sum(g.number_of_edges() for g in pd.layers)
    assert total_edges == fixture.k * (fixture.k - 1) // 2, (
        f"PlotData reconstructed {total_edges} edges; expected "
        f"{fixture.k * (fixture.k - 1) // 2} (K_{fixture.k} complete graph)"
    )
