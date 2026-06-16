# M8 parity golden fixtures

Each `*.json` file in this directory is a *spec*, not a snapshot: it
describes the inputs to a parity test (graph parameters + fixed per-edge
layer assignment). The parity harness in `code/tests/parity_harness.py`
runs **both** Python (`routing_py_rebuild.run_optimization` with
`fixed_layers=...`) and C++ (`Autowiring_CPP` with
`SINIC_FIXED_LAYERS_JSON=...`) with the spec, then compares the two
produced `subgraphsdata.json` files in-process. No "expected output"
snapshot is committed — both sides re-derive it from the spec.

The fixture matrix mirrors REFACTOR_GOALS.md §6 T6:

| Fixture | k | L | wpl | schema_version | crosstalk | Purpose |
|---|---|---|---|---|---|---|
| `k12_L2_wpl1_v2.json` | 12 | 2 | 1 | `2.0` | off | v2.0 wpl=1 path (convention = `geometric`) |
| `k12_L2_wpl2_v2.json` | 12 | 2 | 2 | `2.0` | off | v2.0 wpl=2 path (M6 default, convention = `physical`) |
| `k12_L3_wpl2_v2.json` | 12 | 3 | 2 | `2.0` | off | multi-layer (Python M3 + C++ M6) |
| `k12_L2_wpl2_xtalk.json` | 12 | 2 | 2 | `2.0` | on, intralayer-only | crosstalk path (Python M4 + C++ M7) — all-zero layers, K_12 4-node sanity scaled |
| `k12_L2_wpl2_xtalk_mixed.json` | 12 | 2 | 2 | `2.0` | on, inter+intra | crosstalk path with striped 0/1 layers — exercises `|Δlayer|=1` adjacency branch (M8 R2 dual-review add per Opus P1-O5) |
| `k12_L2_wpl2_oor_fixed.json` | 12 | 2 | 2 | `2.0` | off | **regression**, not a T6 matrix leg: striped 0/1 with the first two non-perimeter edges set out of `[0, L-1]` (`L+5`, `-1`). Locks the `apply_optimization_result` no-clamp/raw-value Python↔C++ contract (REFACTOR_GOALS.md §1-4 M8 dual-review Opus **P2-O5 RESOLVED**; §1-2 design-tradeoff #4). Fails pre-fix, passes post-fix. |

**Note on v1.x:** §6 T6 lists "legacy v1.x" as a fixture class, but the
Python writer hard-codes ``schema_version="2.0"`` (since M3) — there is
no Python-side opt-in for v1.x output. The C++ side via
``SINIC_SCHEMA_VERSION=1.0`` can produce a v1.x file, but there is
nothing to writer-compare against. v1.x is therefore a **reader**
concern, covered by ``test_legacy_v1x_reader_parity`` in
``test_parity.py`` — that test asks: "do both sides accept a
representative legacy file without error and recover the same
per-layer structure".

Convention normalization (REFACTOR_GOALS.md §6 T6 "convention 不同时
先归一化"): each fixture's `waveguides_per_link` is identical on both
sides, so no rescaling is needed in the comparator today. If a future
fixture pins `wpl=1` on Python and `wpl=2` on C++ (or vice versa),
extend `compare_json` with an explicit pre-comparison rescale step
before the float comparison. M8 R2 dual-review (Opus P2-O1) removed
the original no-op stub function — the parity contract is "raise a
noisy diff on cross-convention regression, not silent rescale".

Layer assignments are simple deterministic patterns (constant, striped,
or hashed) — see `generate_fixtures.py` for the source of truth. The
goal is parity, not optimizer fidelity, so any layer vector that
exercises the per-edge crossings classifier is fine.
