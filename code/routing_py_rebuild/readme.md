# routing_py_rebuild

Refactor of `routing_py/Plots/SiN3DIC.py` into a library + CLI. Same physics,
same numerics; cleaner boundaries. Five layers, top to bottom:

```
            CLI (__main__.py)           argparse → api
                  │
            api.py                      run_optimization, make_graph
              │       │           │
   core (graph) │   optimizers/  │   plotting/ (PlotData → figures)
              │       │           │
                  positions.py, io_utils.py     helpers
```

The post-optimization plotting path is JSON-driven: the optimizer writes a
self-contained JSON snapshot; plotting consumes that JSON via
`PlotData.from_json` and never touches the graph model.

---

## File-tree overview

```
routing_py_rebuild/
├── __init__.py              Re-exports the public API
├── __main__.py              CLI: `python -m routing_py_rebuild {optimize|plot|report|replay}`
├── api.py                   make_graph, run_optimization (end-to-end orchestration)
├── core.py                  SiNInterconnectionGraph: graph + crossings + loss + JSON I/O
├── io_utils.py              .b16 binary readers + point-set transforms
├── positions.py             Node-position generators (square / rectangle)
├── progress.py              [P] ProgressTracker (eval_observer) for run_optimization(progress=...)
├── readme.md                this file
│
├── optimizers/              Pluggable layer-assignment strategies (registry-based)
│   ├── __init__.py
│   ├── base.py              Optimizer ABC + OptimizationResult + register/get
│   ├── dual_annealing.py    scipy.optimize.dual_annealing wrapper
│   ├── dual_annealing_with_swap_polish.py  [M3] DA + greedy layer-swap polish
│   ├── differential_evolution.py  scipy.optimize.differential_evolution wrapper
│   ├── ga_pymoo.py          [M-J Stage 1] pymoo Genetic Algorithm
│   ├── pso_pyswarms.py      [M-J Stage 2] pyswarms global-best PSO
│   ├── cmaes.py             [M-J Stage 2] Hansen CMA-ES (cma library)
│   └── bo_skopt.py          [M-J Stage 2] scikit-optimize Bayesian Optimization
│
├── statistics/              [M1] Optimizer trace + phase timings + run report
│   ├── __init__.py
│   ├── sink.py              IterEvent, StatsSink (Protocol), NullSink default
│   ├── recorder.py          RunRecorder: in-memory accumulator + is_new_best calc
│   ├── timer.py             PhaseTimer context manager
│   └── reporters.py         write_run_report → run_report.{json,md} (+ schema valid)
│
└── plotting/                Figure rendering — JSON-driven, no core imports
    ├── __init__.py
    ├── plot_data.py         PlotData dataclass — from_json / from_graph
    ├── _style.py            Shared matplotlib rcParams (Arial 7pt, 0.5pt lines)
    ├── _colormap.py         DEFAULT_COLORMAP (coolwarm) + NODE_FILL_COLOR + formula
    ├── _crossings.py        Standalone edge-crossing detection (utility)
    ├── layers.py            "visualize" style — the shipped layer plot
    ├── ring.py              XPU-style outer ring + complete-mesh diagram
    ├── layer_edges.py       Black-line single-layer subset plot
    ├── loss_analysis.py     Mean / variance / std / range bar charts
    ├── convergence.py       [M1] plot_convergence + plot_timings (run_report.json driven)
    ├── progress_figure.py   [P] ProgressData (optimization_history.json) + ProgressFigure
    ├── progress_live.py     [P] LiveProgressView — window / Jupyter / PNG-file live view
    ├── progress_replay.py   [P] replay → GIF + HTML player, interactive replay window
    └── from_json.py         JSON → PlotData → dispatch (convenience entry)
```

[P] tags mark the optimization-process visualization (`progress="live"` /
`"record"`, see "Watching the optimization process" below). Pair with
`code/schema/optimization_history.schema.json` and
`code/tests/test_progress.py`.

[M1] tags mark files added with the statistics milestone — REFACTOR_GOALS.md §1-1-a + §3-2 + §4. Pair with `code/schema/run_report.schema.json` for the report contract and `code/tests/test_statistics.py` for the acceptance suite (`uv run pytest code/tests/test_statistics.py`).

[M3] (2026-05-13, REFACTOR_GOALS.md §2-3 + §3-1 + §4 M3 附注) lands multi-layer
support, `subgraphsdata.json` schema_version `"2.0"`, the
`dual_annealing_with_swap_polish` optimizer, and end-to-end plumbing for
`L`, `edge_coupler_layer`, `perimeter_layer`, `layer_pitch_um`,
`waveguides_per_link`. New per-edge fields `interlayercrossings_above` /
`_below` are stamped alongside the legacy `interlayercrossings`; the
aggregate writer applies the
`crosslayer_crossings_total = waveguides_per_link * geometric_events`
formula and emits an explicit `crosslayer_crossings_convention` label
(`"physical"` / `"geometric"` / `"experimental"`). Schema validation runs
at both write and read time. Default `edge_coupler_layer=0` is preserved
for legacy bit-exact at L=2 (§2-3 验收第 1 条); for L≥3 the spec
recommendation is `L // 2`. Acceptance tests live in
`code/tests/test_multilayer.py`.

[M4] (2026-05-14, REFACTOR_GOALS.md §2-2) lands the rank-3 channel-to-channel
crosstalk tensor `crosstalk[s][d][t]` (k×k×k, dB). New module
`crosstalk.py` exposes `compute_crosstalk_tensor(graph, max_hops=3,
threshold_db=-60.0)`; the engine is DFS over a branch tree rooted at each
main path, treating each photonic crossing as `T_main = 1 -
loss_crossing` plus a symmetric 50/50 leak onto the orthogonal edge
weighted by `loss_intralayer_crosstalk` (same layer) or
`loss_interlayer_crosstalk` (`|Δlayer| = 1`; `|Δlayer| ≥ 2` is not
coupled per §2-3 目标 D). Per-path `visited` frozenset prevents loops;
hop limit + dB threshold prune. `run_optimization` gains
`compute_crosstalk` / `crosstalk_max_hops` / `crosstalk_threshold_db` /
`crosstalk_include_in_loss` (last is a reserved no-op flag). Output is
embedded under the optional top-level `crosstalk` field of
`subgraphsdata.json` (schema_version stays `"2.0"`; the field is
structurally required to carry `unit` / `shape` / `values` but the field
itself is optional so M3 fixtures remain valid). New plotting style
`plot_crosstalk_heatmap` consumes `plot_data.general_params["crosstalk"]`
with `view="worst_over_d"` / `view="for_specific_d"`. Crosstalk-off
(coefficients `None` / `0`) is bit-exact equivalent to M3:
`crosstalk_analysis_ms` phase is not emitted, no JSON field is written.
24 acceptance tests live in `code/tests/test_crosstalk.py` (4-node
analytical sanity at < 1e-6 dB closed-form error, hop/threshold pruning,
schema validation, PlotData passthrough, heatmap smoke, run report
integration).

[M-J Stage 1] (2026-05-20, REFACTOR_GOALS.md §1-5) adds `ga_pymoo` —
single-objective Genetic Algorithm via `pymoo` 0.6 — as the first
E4-expanded optimizer baseline for the paper reviewer response. The GA
operates on continuous `[0, L-1]^E` (same surface as DA / DE / swap-polish;
`apply_optimization_result` rounds back to int layers via banker's
rounding) with SBX crossover + polynomial mutation. **PM mutation
routing**: pymoo's `PM(prob=X)` is the *per-individual* gate (whether
the PM kernel is applied to a given offspring at all), NOT per-gene;
the per-gene rate is pymoo's `prob_var` (default
`min(0.5, 1/n_var)` ≈ Deb literature). We therefore route our user-facing
`mutation_prob` to `prob_var` and hard-pin `prob=1.0` so every offspring
receives the PM kernel and each gene mutates at the Deb rate. (M-J R1
P0 fix: the original routing to `prob` collapsed activity to ~1/n_var²
genes per individual ≈ 60× weaker than the Deb baseline.) Per-eval
IterEvent emission goes through pymoo's `ElementwiseProblem._evaluate`
for one-to-one parity with the scipy-wrapper sink semantics.
Optimizer-specific knobs (`n_gen`, `pop_size`, `crossover_prob`,
`crossover_eta`, `mutation_prob`, `mutation_eta`,
`seed_with_routing_patterns`) come from the experiments harness
`[optimizer_kwargs.ga_pymoo]` TOML table; unknown keys surface as
`TypeError`; `maxiter` is deliberately ignored (a GA generation is not
a scipy iter). Acceptance tests in `code/tests/test_ga_pymoo.py` (9 cases)
+ `code/experiments/tests/test_mj_optimizer_kwargs.py` (15 cases —
Stage 2 R1 fix added 1 shipped-kwargs smoke test) +
shipped config `code/experiments/configs/exp4b_optimizers_k12.toml`.

[M-J Stage 2] (2026-05-20, REFACTOR_GOALS.md §1-5) adds the remaining
three E4-expanded optimizer baselines: `pso_pyswarms` (pyswarms 1.3
global-best Particle Swarm Optimization), `cmaes` (Hansen Covariance
Matrix Adaptation Evolution Strategy via the `cma` 4.4 library), and
`bo_skopt` (scikit-optimize 0.10 Gaussian-Process Bayesian Optimization).
All three operate on the same continuous `[0, L-1]^E` surface as Stage
1 and earlier optimizers; `apply_optimization_result` rounds back to
int layers. Per-evaluation `IterEvent` emission matches the scipy /
pymoo wrappers exactly.

Stage 2 also introduces three optimizer-specific safety/correctness
treatments documented as REFACTOR_GOALS.md §1-5 Stage 2 design tradeoffs:
(a) **PSO numpy global RNG snapshot/restore** — pyswarms 1.3 has no
`seed` kwarg and reads the numpy global RNG; the optimizer snapshots
`np.random.get_state()` at entry, seeds, runs PSO, then restores in a
`finally` block — caller's RNG state is untouched. (b) **CMA-ES `ask()`
deep-copy perimeter** — `cma`'s `ask()` returns ndarrays that share
storage with the strategy's internal state; if an observer mutates
them, the subsequent `tell()` pairs a mutated solution with a stale
fitness ⇒ silent strategy drift. We copy each candidate before passing
to `loss_fn` / observer, then `tell()` with the original
(uncontaminated) solutions list. (c) **BO honest-baseline framing** —
`bo_skopt` runs at `n_calls=120` in E=66 dimensions (well beyond BO's
~20D comfort zone); the paper-reviewer-response intent is to document
where surrogate-model methods scale-limit on this routing instance,
NOT to tune BO to win — `ConvergenceWarning`s from skopt are
deliberately NOT silenced.

Acceptance tests in `code/tests/test_pso_pyswarms.py` (10 cases),
`code/tests/test_cmaes.py` (11 cases — Stage 2 R1 fix added
`test_cmaes_does_not_leak_numpy_global_rng`),
`code/tests/test_bo_skopt.py` (9 cases). The shipped
`exp4b_optimizers_k12.toml` now compares all 7 optimizers (3 scipy + GA
+ PSO + CMA-ES + BO) at k=12 over 10 seeds × 2 itl values. Tabu Search
remains deferred (REFACTOR_GOALS.md §7 Q-f open question — no standard
pip lib for the canonical Glover 1986 formulation; ~150–250 LOC custom
impl if reviewer round 2 requires).

---

## Usage

### Library

```python
from routing_py_rebuild import run_optimization, plot_from_json

# k=12 dual-annealing, seed 5859. Writes JSON + figures into output_dir.
res = run_optimization(
    k=12,
    optimizer="dual_annealing",
    maxiter=200,
    output_dir="/tmp/sin_run/",
)
print(res["loss"], res["json_path"])
# res keys: graph, best_layers, loss, json_path, plot_data, report_path

# M3 (REFACTOR_GOALS.md §2-3): multi-layer with centered coupler + the
# `dual_annealing_with_swap_polish` optimizer. Default
# `edge_coupler_layer=0` preserves bit-exact compatibility at L=2; for
# L>=3 the spec recommendation is `L // 2`, passed explicitly here.
#
# Post-2026-05-15 (§7 Q-e): default `waveguides_per_link=1` so JSON
# `crosslayer_crossings_total` matches the paper's geometric crossing
# count. Pass `waveguides_per_link=2` for the physical convention
# aligned with the loss model's implicit 2×.
res_L3 = run_optimization(
    k=12, L=3,
    optimizer="dual_annealing_with_swap_polish",
    edge_coupler_layer=1, perimeter_layer=1,
    optimizer_kwargs={"polish_iters": 500},
    # waveguides_per_link=1 by default (geometric / paper convention);
    # pass 2 explicitly if you want the physical convention.
    maxiter=100,
    output_dir="/tmp/sin_run_L3/",
    collect_statistics=True,
)
# run_report.json.summary.crosslayer_crossings_total carries the
# wpl-scaled count; crosslayer_crossings_convention says
# "geometric" (wpl=1, default) / "physical" (wpl=2) / "experimental" (3..32).

# Re-render later from saved JSON. The file is self-contained — positions,
# k, loss parameters, complete graph, loss_analysis all come from it.
plot_from_json(
    res["json_path"],
    style="visualize",
    out_dir="/tmp/sin_plot/",
)
```

### CLI

The project is uv-managed (see top-level `pyproject.toml`). Run from any
directory inside the repo — `uv run` resolves the project venv upward.

```bash
# Optimize end-to-end (k=12, dual annealing, seed 5859)
uv run python -m routing_py_rebuild optimize --k 12 --maxiter 200 \
    --output-dir /tmp/run

# Differential-evolution variant
uv run python -m routing_py_rebuild optimize --k 12 \
    --optimizer differential_evolution --maxiter 50 --output-dir /tmp/run_de

# Arbitrary node coordinates {"<node>": [x, y], ...}, numbered along the
# boundary; k comes from the file
uv run python -m routing_py_rebuild optimize --positions-json layout.json \
    --maxiter 200 --output-dir /tmp/run_custom

# Plot from a saved JSON (positions, k, loss params all come from the file)
uv run python -m routing_py_rebuild plot \
    --json /tmp/run/.../subgraphsdata.json \
    --style visualize --out-dir /tmp/plotout
```

> **maxiter defaults**: CLI / `run_optimization` set 500. The optimizer
> classes' `optimize()` methods default to 1000 (DA) and 100 (DE) when
> called directly.

### Watching the optimization process

`progress` selects how much of the optimization you see (default
`"off"`: only the final result, exactly as before):

```python
# live: redraw the current best routing + loss curve while optimizing
run_optimization(k=12, maxiter=200, output_dir="/tmp/run", progress="live")

# record: save every improvement, then replay the whole process
res = run_optimization(k=12, maxiter=200, output_dir="/tmp/run", progress="record")
res["progress"]   # {"mode": "record", "history": ..., "gif": ..., "html": ...}

# re-render / re-open a recorded run later
from routing_py_rebuild import render_progress_replay, show_progress_replay
render_progress_replay(res["progress"]["history"], formats=["gif"], fps=10)
show_progress_replay(res["progress"]["history"])   # window with slider + play
```

```bash
uv run python -m routing_py_rebuild optimize --k 12 --output-dir /tmp/run --progress live
uv run python -m routing_py_rebuild optimize --k 12 --output-dir /tmp/run --progress record \
    --progress-kwargs '{"formats": ["gif", "html"], "max_frames": 120}'
uv run python -m routing_py_rebuild replay --history /tmp/run/.../optimization_history.json --show
```

What each frame shows: one panel per layer, shaped like the node
layout's bounding box (height / width clamped to 0.4..1.5, so a 5 × 3
rectangle gets wide panels), with edges colored by per-edge same-layer
crossings (the `layers_combined.pdf` palette; edges that changed layer
since the previous frame are drawn heavier with a dark outline when only
a few changed); a colorbar; and the best loss so far
(black; the part still ahead in gray during a replay; the current point
in red) against the log-scaled evaluation count, over a light band of
all evaluated losses.

Figure style (publication rules, shared with `layers_combined.pdf`): one
7 pt font — Arial, else Helvetica / Liberation Sans / Arimo (see
`plotting/_style.py`) — no figure title, sentence-case axis labels with
units (`Evaluations`, `Mean edge loss (dB)`, `Crossings per edge`),
0.5 pt data lines, four black 0.5 pt spines, inward ticks, and on the
log axis minor ticks at 2..9 × 10^n with a light dashed / dotted grid.
The layout is fixed in centimetres (17.8 cm wide at k ≤ 12, growing with
√(k/12)); progress goes in a status line above the loss panel and the
run identity in a muted note under it.

- **live** — interactive matplotlib backend → a window, redrawn at most
  every `interval` seconds and never for more than ~20 % of the run time
  (`progress.LIVE_MAX_OVERHEAD`); at the end it stays open until closed
  (`show=False` to return immediately). Jupyter inline backend → one
  output updated in place. Headless (Agg) → `optimization_live.png` is
  rewritten atomically in the run directory. A display error disables
  the view with a warning; the optimization always continues.
- **record** — writes `optimization_history.json` (every frame; see the
  contract below), then `optimization_progress.gif` (one shared 256-color
  palette; the final routing holds 2.5 s), `optimization_progress.html`
  (self-contained player with lossless WebP frames: play / pause / step /
  slider / speed / loop, keyboard shortcuts) and a still of the final
  state — final routing plus the complete loss curve, without status
  line or run label — as `optimization_progress.pdf` (vector, TrueType
  text that stays editable) and `optimization_progress.png` (450 ppi).
  With a GUI backend the replay then opens in a window with a slider and
  a play button; in Jupyter the player is shown inline. Ctrl-C during the
  run still saves the improvements found so far.

`progress_kwargs` / `--progress-kwargs` (`progress.ProgressOptions`;
unknown keys raise `TypeError`): `interval` (0.5 s), `show` (True),
`formats` (`["gif", "html", "pdf", "png"]`, `[]` = history only),
`max_frames` (200 — the animation is capped, the JSON keeps every frame),
`fps` (None = automatic, 4–15), `dpi` (None = 120 for a window, 150 for
images; the PNG still is always 450 ppi).

Tracking is an `eval_observer`, called after each loss evaluation with
a private copy of the evaluated vector (`optimizers.base.observer_copy`;
scipy copies an accepted point only after the objective returns, so an
observer never sees the live one), so every mode yields the same trace
and result for a given seed
(`test_progress_modes_do_not_perturb_the_optimizer`). Record
mode adds well under 1 % to the optimizer wall time; live mode adds its
drawing time (bounded as above). With `collect_statistics=True` the
record outputs are timed as `progress_history_write_ms` and
`progress_replay_ms`.

#### `optimization_history.json` contract

Written next to `subgraphsdata.json`, validated against
`code/schema/optimization_history.schema.json` on write and read (plus
cross-field checks in `plotting/progress_figure.py::_validate_history`):

```
schema_version: "1.0"
k, L, edge_coupler_layer, perimeter_layer
positions:   {"0": [x, y], ...}
edges:       [[u, v], ...]            order of every per-edge array below
run:         {optimizer, maxiter, seed, loss_crossing, loss_taper, ...}
frames:      [{eval_index, wall_ms, loss, layers: [...], crossings: [...], final}, ...]
evaluations: {count, start: [...], min: [...], max: [...]}   log-spaced buckets
summary:     {evaluations, improvements, initial_loss, final_loss, wall_ms}
```

One frame per strict new best (in evaluation order); `layers` are the
effective per-edge layers (rounded, perimeter pinned) and `crossings`
the per-edge same-layer crossing counts. The last frame is flagged
`final` and equals the routing in `subgraphsdata.json`; if the
optimizer's returned result differs from its last new best, the result
is appended as an extra final frame (then `improvements` =
`len(frames) - 1`).

---

## Module-level overview

Top-level concerns, in dependency order:

### `core` — graph data model

`SiNInterconnectionGraph` owns the complete graph (`G`), the
complete-graph baseline at the base positions (`_G_Planar`, whose
crossings/loss are stamped **lazily** by the M2 Phase A index — the
first call to `loss_function`, `analyze_loss`, or the public
`build_crossings_index` triggers a single O(E²) geometric pass; before
that the per-edge attrs sit at zero), the per-layer subgraph
decomposition (`sub_G`), loss accounting, and JSON serialization.
Optimizers consume `loss_function(layers)`; everything else is
downstream of `analyze_loss()` and `save_subgraphs_to_json()`. The class
has no plot or optimizer logic inside.

### `positions` — node-position generators

Pure functions returning `dict[int, (x, y)]`. Two layouts ship — square
and rectangle, both placing nodes evenly along the sides. Used as the
input to `SiNInterconnectionGraph(positions=...)`.

### `io_utils` — `.b16` binary readers

Little-endian uint16 readers for the pre-computed `minimize{k}.b16`
point sets. Each `.b16` file packs a sequence of `(x, y)` integer
point sets; `transform_point_sets` turns the raw int chunks into a
list of `{node_index: (x, y)}` dicts. (The shipped `visualize` style
no longer consumes these after the 2026-05-15 rewrite; the readers
remain as a standalone utility.)

### `optimizers/` — pluggable strategies

A simple name-based registry (`register_optimizer("name")` decorator +
`get_optimizer("name")` lookup). Every optimizer subclasses `Optimizer`,
takes a graph instance + `seed` + optimizer-specific kwargs, and
implements `optimize(maxiter) → OptimizationResult`. Two strategies
ship: dual annealing and differential evolution, both thin scipy
wrappers. The optimizer never plots; it ends by calling
`graph.apply_optimization_result(best_layers)`.

### `plotting/` — JSON-driven figure rendering

Plotting is fully decoupled from the graph model. The contract is:

    optimizer  →  graph.save_subgraphs_to_json  →  PlotData.from_json  →  plot

Plot styles take a `PlotData` snapshot — never a `SiNInterconnectionGraph`
instance, never an import from `core`. A `from_graph` escape hatch exists
for tests; it is strictly read-only and requires the caller to have run
`analyze_loss()` first. Style functions are pluggable via a second
registry (`register_style("name")`).

### `api` — high-level orchestration

`run_optimization` wires graph creation → optimizer → JSON → plotting
into one call. When `save_json=True` (the default) plotting consumes
the JSON it just wrote; when `save_json=False` the orchestrator
explicitly calls `graph.analyze_loss()` first, then snapshots in
memory via `PlotData.from_graph`. Plotting itself is unchanged
between the two paths. `progress="live"` / `"record"` additionally
attaches a `progress.ProgressTracker` as the optimizer's
`eval_observer` (chained after any caller-supplied one).

### `__main__` — CLI

`optimize` calls `run_optimization` with argparse arguments. `plot`
loads a JSON and dispatches to a registered style. The plot subcommand
is minimal — JSON is self-contained, so it needs no `--k` / positions
arguments. `report` renders a `run_report.json`; `replay` renders /
opens a recorded `optimization_history.json`.

---

## Per-file reference

### Top level

#### `__init__.py`

Re-exports the public API. Importing `from routing_py_rebuild import ...`
gives you: `SiNInterconnectionGraph`, `run_optimization`, `make_graph`,
`PlotData`, `plot_from_json`, `visualize_layers`, `visualize_ring`,
`visualize_loss_analysis`, `plot_layer_edges`, `register_style`,
`list_styles`, `distribute_nodes`, `distribute_nodes_around_square`,
`distribute_nodes_around_rectangle`, `POSITIONS_12_NODES`, `Optimizer`,
`OptimizationResult`, `register_optimizer`, `get_optimizer`,
`DualAnnealingOptimizer`, `DifferentialEvolutionOptimizer`,
`DEFAULT_COLORMAP`, `NODE_FILL_COLOR`, `get_colormap`, `crossings_color`.

#### `__main__.py`

CLI entry: `python -m routing_py_rebuild ...`. Subparsers:

- `optimize` — flags `--k --positions-json
  --optimizer {dual_annealing,differential_evolution}
  --maxiter --seed --optimizer-kwargs --output-dir --loss-crossing
  --loss-taper --loss-interlayercrossing --plot-style {visualize}
  --plot-kwargs --no-plot --no-loss-analysis --no-json
  --progress {off,live,record} --progress-kwargs`. Hands everything off
  to `api.run_optimization`. `--positions-json PATH` loads arbitrary
  node coordinates with `positions.load_positions_json` (the C++
  `SINIC_POSITIONS_JSON` format, or a saved run's `positions` block);
  k is the file's node count (default 12 on a square otherwise), a
  different explicit `--k` exits with an error, and a node order that
  does not walk the boundary (`positions.perimeter_is_simple`) prints a
  warning.
- `plot` — flags `--json (required) --style {visualize} --out-dir
  --plot-kwargs --also-loss-analysis`. No `--k` or positions flags;
  everything comes from the JSON. Hands off to
  `plotting.plot_from_json`.
- `replay` — flags `--history (required; file or run directory)
  --out-dir --formats gif,html --fps --max-frames --dpi --show`.
  Hands off to `plotting.render_progress_replay` /
  `plotting.show_progress_replay`.

`--optimizer-kwargs` / `--plot-kwargs` accept a JSON dict string.

> **Default loss model** (CLI, `api`, and `core` alike): intralayer
> crossing 0.1 dB, taper 0.05 dB, interlayer crossing 0.001 dB per event
> — `core.DEFAULT_LOSS_CROSSING` / `DEFAULT_LOSS_TAPER` /
> `DEFAULT_LOSS_INTERLAYERCROSSING`; the C++ harness uses the same
> values. Earlier versions defaulted to 0.3 / 1.0 / 0.006 (0.3 / 0.05 /
> 0.006 in the CLI); pass those explicitly to reproduce an old run.

#### `api.py`

Two functions:

- `make_graph(*, k, positions=None, output_dir=None, loss_crossing=0.1,
  loss_taper=0.05, loss_interlayercrossing=0.001, shape="square")` —
  thin constructor that defaults to `distribute_nodes("square",
  nodes_per_side=k//4, side_length=1)` when `positions` is omitted.
- `run_optimization(*, k=12, optimizer="dual_annealing", maxiter=500,
  seed=5859, optimizer_kwargs=None, positions=None,
  output_dir="./assets/run/", loss_crossing=0.1, loss_taper=0.05,
  loss_interlayercrossing=0.001, plot=True, plot_style="visualize",
  plot_kwargs=None, save_json=True, run_loss_analysis=True, ...,
  progress="off", progress_kwargs=None)` —
  end-to-end orchestration. Returns
  `{graph, best_layers, loss, json_path, plot_data, report_path,
  crosstalk, progress}` (`progress` is None when `progress="off"`, else
  the mode plus the paths it wrote).

Plot path: when `save_json=True`, plotting consumes the JSON just written
(`PlotData.from_json`); when `save_json=False`, the orchestrator calls
`graph.analyze_loss()` itself and uses `PlotData.from_graph` — keeping
plotting code read-only on the graph regardless.

#### `core.py`

The data model. `SiNInterconnectionGraph(k, positions, filepath=None,
loss_crossing=0.1, loss_taper=0.05, loss_interlayercrossing=0.001)`
constructs the complete graph and a planar reference `_G_Planar`. M2
deferred the one-time O(E²) crossing stamp out of `__init__` — call
`build_crossings_index()` (or just call `loss_function` /
`analyze_loss`, which trigger it lazily) when you actually need it.

Method groups:

- **Crossing detection**: `edge_crosses(e1, e2)`,
  `count_crossings_with_detail(graph)`,
  `count_interlayercrossings(g1, g2)` (still shapely-based, used by
  `analyze_loss` for per-subgraph stamping; `loss_function`'s hot loop
  no longer touches them — see M2 cached topology below).
- **M2 Phase A — cached topology**: `build_crossings_index()` builds
  and caches the K_k crossing-pair index (`_crossing_pairs`,
  `_edge_list`, `_edge_index`, `_perimeter_mask`); idempotent. The
  fast path is a pure-integer alternating-endpoints test gated by
  `_is_cyclic_convex_positions`; non-convex layouts fall through to a
  vectorized orientation test with closed-form degenerate handling.
- **Loss**: `cal_loss_of_edge(graph)`,
  `update_interlayercrossings_loss(graph)`,
  `loss_function(layers)` (M2 rewrite: vectorized integer
  classification over `_crossing_pairs`; **no longer mutates
  `self.G`** — pre-M2 it did, M2 audit confirmed nothing reads `G`
  during the hot loop, and `apply_optimization_result` still
  publishes the final layer assignment to `G` post-optimize).
- **Layer management / seeding**: `set_edge_layer`, `k_base_no_carry_add`,
  `k_base_no_carry_sub`, `routing_method_1(takeaway, layer)` — k-base
  offset pattern used as initial guesses by both optimizers.
- **Subgraph build**: `create_subgraphs()` splits `self.G` by `layer`
  attr into `self.sub_G`, stamps crossings + losses, and computes
  inter-layer crossings between layer 0 and layer 1.
- **Optimization result application**: `_apply_layer_assignment(layers)`
  pins perimeter edges (`|u-v| ∈ {1, k-1}`) to layer 0 regardless of
  input — preserves the original physical-ring invariant.
  `apply_optimization_result(layers)` is the public version.
- **Progress helpers** (read-only, not on the hot path):
  `effective_layers(layers)` (round + perimeter pin — the routing
  `loss_function` scores) and `intralayer_crossing_counts(layers)`
  (per-edge same-layer crossings from the Phase A index; equals the
  `crossings` attr `create_subgraphs` stamps). Used by
  `progress.ProgressTracker`.
- **Loss analysis**: `analyze_loss()` populates `self.loss_analysis` —
  mean/var/std/range per layer plus the flattened and complete-graph
  variants. Resets `sub_G` / `total_crossings_of_sub_G` /
  `edge_cross_counts_of_sub_G` at entry, then calls `create_subgraphs()`,
  so the rebuilt state is always fresh. Invoked automatically from
  `save_subgraphs_to_json`, so the JSON snapshot is always consistent
  with the latest layer assignment.
- **JSON I/O**: `save_subgraphs_to_json(filename, **kwargs)` writes a
  self-contained snapshot (positions, per-layer subgraph edges with
  attributes, complete-graph edges, loss-analysis, all loss parameters,
  plus any `kwargs` recorded under `General Parameters` —
  `api.run_optimization` records `optimizer`, `maxiter`, `seed`, `best_loss`).

JSON schema (keys at the top level):

```
General Parameters: {k, Loss of Taper, Loss of Crossing,
                     Loss of Interlayer Crossing, plus kwargs}
positions:          {"0": [x, y], "1": [...], ...}
Layer_0:            {edges: [[u, v, {layer, crossings, loss, interlayercrossings}], ...]}
Layer_1:            {edges: [...]}
...
complete_graph:     [[u, v, {layer, crossings, loss, interlayercrossings}], ...]
loss_analysis:      {avg_loss_subgraphs, var_loss_subgraphs, std_loss_subgraphs,
                     range_loss_subgraphs, avg_flattened_loss_subgraphs, ...,
                     avg_loss_completegraph, ...}
```

#### `io_utils.py`

Three functions, all small:

- `read_and_interpret_b16_file(filename, points_per_set)` — read a `.b16`
  file as little-endian uint16, returning a list of chunks where each
  chunk is `2 * points_per_set` integers (interleaved x/y).
- `transform_point_sets(point_sets, k)` — convert raw int chunks to a
  list of `{node_index: (x, y)}` dicts.
- `print_point_sets(point_sets, points_per_set)` — debug printer.

#### `positions.py`

Pure functions returning `dict[int, (x, y)]`, with node indices in
boundary order (the perimeter ring i -> i + 1 is pinned to one layer):

- `distribute_nodes_around_square(nodes_per_side, side_length=4)` —
  even placement on a square; corners excluded.
- `distribute_nodes_around_rectangle(nodes_on_length, nodes_on_width,
  length=5, width=3)` — same idea for rectangles; `2 * (nodes_on_length
  + nodes_on_width) = k`.
- `distribute_nodes_around_circle`, `distribute_nodes_around_triangle`,
  `distribute_nodes_around_polygon`,
  `distribute_nodes_around_partial_rectangle` (2 or 3 sides).
- `distribute_nodes(shape, ...)` — dispatcher over `"square"`,
  `"rectangle"`, `"circle"`, `"triangle"`, `"polygon"`,
  `"partial_rectangle"`; raises `ValueError` otherwise.
- `load_positions_json(path)` — arbitrary coordinates from
  `{"<node>": [x, y], ...}` (canonical integer keys forming `0..k-1`,
  finite two-number values; same contract as the C++ loader). A file
  with a `positions` object (`subgraphsdata.json`,
  `optimization_history.json`) contributes that block. Raises
  `ValueError` naming the file.
- `perimeter_is_simple(positions)` — True when walking nodes `0..k-1`
  and back to 0 never crosses itself, i.e. the indices follow the
  boundary.

For nodes on a convex outline the crossing structure depends only on
the node order (two chords cross iff their ends alternate along the
boundary), so every convex layout with the same order optimizes
identically; coordinates matter once nodes sit inside the convex hull
of the others.

Plus one module-level constant, `POSITIONS_12_NODES`, kept for
backwards-compatible imports of the demo k=12 layout.

### `optimizers/`

#### `optimizers/__init__.py`

Re-exports `Optimizer`, `OptimizationResult`, `get_optimizer`,
`register_optimizer`, `DualAnnealingOptimizer`,
`DifferentialEvolutionOptimizer`.

#### `optimizers/base.py`

- `@dataclass class OptimizationResult` — `(best_layers: list[int], fun:
  float, raw: object | None = None)` — `raw` holds the underlying scipy
  result object when the optimizer wants to surface it; otherwise `None`.
- Registry: `register_optimizer("name")` decorator + `get_optimizer("name")`
  lookup. Strings are case-sensitive.
- `class Optimizer(graph, *, seed=None, **kwargs)` — ABC. Subclasses
  override `optimize(maxiter) → OptimizationResult`. Shared helper
  `_seed_from_routing(takeaway, layer=1)` runs `graph.routing_method_1`
  to build a k-base initial guess vector.

#### `optimizers/dual_annealing.py`

`DualAnnealingOptimizer` (registered as `"dual_annealing"`).
`optimize(maxiter=1000)` calls `scipy.optimize.dual_annealing` with:

- `bounds = [(0, 1)] * n_edges`
- `x0 = self._seed_from_routing(takeaway=[3, 5], layer=1)`
- `seed = self.seed`
- `**self.kwargs` forwarded to scipy

Result: rounds `result.x` to ints, calls `graph.apply_optimization_result`,
prints final loss, returns `OptimizationResult`.

#### `optimizers/differential_evolution.py`

`DifferentialEvolutionOptimizer` (registered as `"differential_evolution"`).
`optimize(maxiter=100)` calls `scipy.optimize.differential_evolution` with:

- `bounds = [(0, 1)] * n_edges`
- `popsize = self.kwargs.pop("popsize", self.graph.k)` (default = k)
- `init = init_population` — by default a `(popsize, n_edges)` random
  matrix whose first **up to** 4 rows (capped by `popsize`) are k-base
  seed patterns from `_seed_from_routing` with `takeaway` ∈
  `[[4,5], [3,4], [5], [3,5]]`. Pass any scipy-supported init string
  via `optimizer_kwargs` (e.g. `{"init": "sobol"}` / `{"init": "halton"}`)
  to opt out — user-supplied `init` wins.
- `DEFAULTS = {strategy: "rand2exp", mutation: (0.3, 0.8),
  recombination: 0.6}` — overridable through `optimizer_kwargs`.

**Seed-population fix vs original**: the original built the 4-pattern
matrix but never passed it to `differential_evolution`. The rebuild
passes `init=init_population` by default so the seed patterns are
actually used.

### `plotting/`

#### `plotting/__init__.py`

Re-exports the user-facing plot functions and the colormap symbols.
Importing `from routing_py_rebuild.plotting import ...` gives you
`PlotData`, `visualize_layers`, `register_style`, `list_styles`,
`visualize_ring`, `plot_layer_edges`, `visualize_loss_analysis`,
`plot_from_json`, and the color palette (`DEFAULT_COLORMAP`,
`NODE_FILL_COLOR`, `CMAP_LOWER/UPPER/SCALE`, `get_colormap`,
`crossings_color`).

#### `plotting/_style.py`

Shared matplotlib rcParams. `apply_rcparams()` sets Arial 7pt, 0.5pt
axis/edge lines, savefig dpi=500. Every public plot function calls it
once at the top so fonts/lines are consistent across styles. The font is
requested as a stack, `font_stack()` = `FONT_FAMILY` then
`FONT_FALLBACKS` (Helvetica, Liberation Sans, Arimo, DejaVu Sans), so a
machine without Arial renders a metric-compatible clone instead of
matplotlib's default (identical output where Arial exists). Module
constants `FONT_FAMILY`, `FONT_FALLBACKS`, `FONT_SIZE`, `LINE_WIDTH`,
`SAVEFIG_DPI` are mutable for global overrides before the first draw.
`style_axes(ax, grid=None)` / `style_colorbar(cbar, label=...)` apply
the plotting-box rules used by the progress figure (four black 0.5 pt
spines, inward ticks, log-axis minor ticks at 2..9 × 10^n, grid on by
default for log axes).

#### `plotting/_colormap.py`

Single source of truth for plot colors.

- `NODE_FILL_COLOR = "#AFE0E8"` — light-cyan node fill (layer plot and
  ring nodes).
- `CMAP_LOWER = 0.0`, `CMAP_UPPER = 0.95`, `CMAP_SCALE = 0.95` — the
  window applied on top of the caller's `Normalize`. Reserves the top
  5% of the colormap so a maximum-crossing edge does not land on the
  near-black extreme of coolwarm.
- `DEFAULT_COLORMAP` — matplotlib's standard `coolwarm` sampled at 256
  even stops on [0, 1] and wrapped as a `ListedColormap` named
  `sin_coolwarm`. Full range, no truncation. Mirrors
  `code/plot_olpapper/_colormap.py:DEFAULT_COLORMAP`.
- `get_colormap(name="coolwarm", n=256)` — `lru_cache`-d builder for
  named matplotlib colormaps resampled the same way.
- `crossings_color(value, norm, cmap=None)` — bundles the canonical
  formula `cmap(CMAP_LOWER + CMAP_SCALE * norm(value))` so the
  computation is defined in exactly one place.

#### `plotting/_crossings.py`

Standalone edge-crossing detection. Mirrors
`core.SiNInterconnectionGraph.edge_crosses` but takes `(edges,
positions)` directly so plotting code can compute crossings against
alternative positions without holding a graph instance. (No longer
used by the shipped `visualize` style after the 2026-05-15 rewrite;
kept as a standalone utility for downstream/custom styles.)

Functions: `edge_crosses(e1, e2, positions)`,
`count_crossings_with_detail(edges, positions) → (num_total, dict)`.

#### `plotting/plot_data.py`

`@dataclass PlotData` — the only argument every plot function takes.

Fields: `k`, `positions: dict[int, (x, y)]`, `layers: list[nx.Graph]`,
`complete_graph: nx.Graph`, `loss_analysis: dict`,
`loss_crossing: float`, `general_params: dict`, `source_path: str | None`.

`filepath` property — returns the source JSON's parent directory (or
the source path itself when it is an **existing** directory, checked
via `os.path.isdir`; a not-yet-created `graph.filepath` string from
`from_graph` therefore falls back to `os.path.dirname`). Used as the
default output directory.

Two constructors:

- `PlotData.from_json(json_path)` — production path. Raises `ValueError`
  if `positions` or `complete_graph` are missing (older JSON files).
- `PlotData.from_graph(graph)` — escape hatch for tests / in-memory use.
  **Strictly read-only**: raises `ValueError` if `sub_G` or
  `loss_analysis` are empty, so the caller is forced to run
  `graph.analyze_loss()` first. Snapshots fresh copies of the layer
  subgraphs and the planar reference.

#### `plotting/layers.py`

`visualize_layers(plot_data, *, style="visualize", **kwargs)` dispatches
into a style registry. `register_style("name")` is the decorator;
`list_styles()` lists registered names. Only one style ships:

`@register_style("visualize")` `_visualize(plot_data, *, out_dir=None)`
renders a **2-row** figure to `layers_combined.pdf`: row 0 has one
**square** edge-graph panel per subgraph (`L` layer subgraphs + 1
complete-graph baseline); row 1 has the matching per-graph
crossing-count bar charts.

Key behaviors:

- Edge colors driven by `crossings_color(value, norm, cmap)` — coolwarm
  with the top-5% reservation.
- Every node number is labelled; the label font shrinks with `k`
  (`_label_fontsize`). Graph panels are forced square via
  `set_box_aspect(1)`.
- An edge whose straight chord runs through another node (nodes in a
  row on one straight stretch of the boundary, on any layout) bows
  toward the interior (`arc3`, `_edge_bends` / `_draw_curved_edges`;
  bow `CURVE_FACTOR` × (nodes between + 1) / (nodes on that line),
  "through" meaning within `ON_LINE_TOL` × the layout span). The
  progress figure draws the same curves (`edge_polylines`). Endpoints
  stay anchored on the node:
  `_node_size(k)` is the single source for the node marker size and is
  passed to both `draw_networkx_nodes` and every `draw_networkx_edges`
  call so networkx's FancyArrowPatch shrink can't desync (otherwise
  arcs float clear of small nodes).
- Histogram x-axis and color norm derived from `layers +
  [complete_graph]`. x-ticks are equal-spaced integers
  (`MaxNLocator(integer=True)`); all bar charts share one bar width
  (`_layer_bar_width`, the layer rows' pooled median neighbour gap ×
  `BAR_WIDTH_FRAC`).
- Raises `ValueError` early if no output directory is resolvable
  (neither `out_dir` nor `plot_data.filepath`).

Tunable module constants: `CURVE_FACTOR`, `ON_LINE_TOL`, `XTICK_NBINS`,
`BAR_WIDTH_FRAC`, `HIST_H_FRAC`, and the figure-scale knobs.

> **2026-05-15 rewrite.** `_visualize` was wholesale-replaced for
> large-`k` readability. The old N-rows×2 layout and its
> `node_params` / `save_combined` / `show_loss_on_bar` / `panel_size`
> / `plot_minimize` / `minimize_path` kwargs, the `minimize{k}.b16`
> overlay row, the separate-file mode, and `DEFAULT_MINIMIZE_DIR`
> were **removed**. The `visualize_layers` / `plot_from_json` public
> entries and the `register_style` registry are unchanged. See
> REFACTOR_GOALS.md "plotting 布局改版".

#### `plotting/ring.py`

`visualize_ring(plot_data, *, num_nodes=None, use_current_positions=False,
layout="circle", node_label="{n} XPU\nnode", radius=1.0, out_path=None,
cmap=None)` draws an outer ring + all-to-all internal mesh + outward
bidirectional arrows. Edge color is driven by the complete-graph
crossing counts on `plot_data.complete_graph` (so edge colors here
reflect the same crossing density the layer plot uses, not the per-layer
local counts). Nodes filled with `NODE_FILL_COLOR`; labels in default
text color (black on light cyan).

#### `plotting/layer_edges.py`

`plot_layer_edges(plot_data, layer=0, nodes_subset=None, *,
out_path=None, node_params=None, panel_size=(4, 4, "cm"))` — black-line
single-layer view, optionally restricted to a `nodes_subset`. Nodes in
the subset are highlighted; the rest are faded grey for context.
Deliberately keeps black-on-white styling (not the package coolwarm
palette) — this is a categorical "show me one layer" plot, not a value
map.

#### `plotting/loss_analysis.py`

`visualize_loss_analysis(plot_data, *, out_path=None)` — four-panel bar
chart over `plot_data.loss_analysis`: mean / variance / std /
range, each panel showing per-layer + flattened + complete-graph
values. Panels use distinct pastel colors for visual differentiation —
again categorical, not coolwarm-mapped. Saves to
`{plot_data.filepath}/lossanalysis.pdf` by default.

#### `plotting/from_json.py`

`plot_from_json(json_path, *, style="visualize", out_dir=None,
also_loss_analysis=False, **style_kwargs) → PlotData` — the convenience
entry. Loads the JSON via `PlotData.from_json`, picks `out_dir` if
given (else falls back to the JSON's directory), runs `visualize_layers`
with the chosen style, and optionally runs `visualize_loss_analysis`.
Returns the `PlotData` for further use.

#### `progress.py` (top level)

`ProgressTracker(graph, *, mode, options, run, out_dir)` — the
`eval_observer` behind `run_optimization(progress=...)`. Per evaluation
it feeds a `LossEnvelope` (streaming min/max in log-spaced buckets,
bounded memory); per strict new best it snapshots
`graph.effective_layers(x)` + `graph.intralayer_crossing_counts(...)`
(read-only `core` helpers using the cached Phase A index) as a
`ProgressFrame`. `finish(best_layers, fun)` flags / appends the final
frame; `write_history()` / `write_replay()` / `present()` produce the
record outputs and the end-of-run display. Also `ProgressOptions`,
`PROGRESS_MODES`, `LIVE_MAX_OVERHEAD`, `chain_observers`.

#### `plotting/progress_figure.py`

`ProgressData` — geometry + frames + evaluation envelope, with
`from_json` / `write_json` (schema-validated) for
`optimization_history.json`. `ProgressFigure` — the frame renderer: one
`LineCollection` per layer panel updated in place (edge geometry from
`edge_polylines`, which reproduces the `visualize` style's arcs), a
colorbar with the `crossings_color` window, and the log-x loss panel.
Fixed centimetre layout so frames never jitter; `annotate=False` drops
the status line and run label (stills).

#### `plotting/progress_live.py`

`LiveProgressView` — picks `"window"` / `"notebook"` / `"file"` from the
matplotlib backend (`live_display_mode`), opens the figure before the
optimizer starts, then redraws on each `LiveState` the tracker pushes;
`hold()` blocks until the window is closed.

#### `plotting/progress_replay.py`

`render_progress_replay(history, *, out_dir, formats, fps, max_frames,
dpi)` renders each selected frame once, streaming it to the GIF (one
shared palette with the thin-line colors reserved) and/or the HTML
player (lossless-WebP data URIs + a small inline script); `"pdf"` /
`"png"` add the final-state still. `history` may be the JSON, its run
directory, or an `optimize --output-dir` holding a single run
(`load_progress_data`). `show_progress_replay(history, ...)` opens the same
frames in a matplotlib window with a `Slider`, a play/pause `Button` and
←/→/space/Home/End keys (prints a note and returns on non-GUI backends).
`select_frames` keeps first and final frames when capping.

---

## Architectural contracts

### Plotting decoupling

The plotting package never imports `core` and never mutates a
`SiNInterconnectionGraph` instance. Every plot style takes a `PlotData`
as its first argument. The JSON written by `save_subgraphs_to_json` is
the contract: it embeds everything plot styles can read off `PlotData`
(positions, per-layer subgraphs with `crossings`/`loss`/
`interlayercrossings` attrs, complete-graph baseline, `loss_analysis`,
loss parameters).

### Color palette

All value→color mappings in `layers.py` and `ring.py` route through
`crossings_color(value, norm, cmap=None)` from `_colormap.py`. The
formula `cmap(CMAP_LOWER + CMAP_SCALE * norm(value))` is defined in
exactly one place. Default colormap is full-range coolwarm
(`DEFAULT_COLORMAP`); default node fill is `NODE_FILL_COLOR =
"#AFE0E8"`. Two plot files deliberately do not use the coolwarm palette:
`layer_edges.py` (black-on-white categorical style) and
`loss_analysis.py` (pastel-per-panel categorical bars).

### Adding an optimizer

```python
from routing_py_rebuild.optimizers import (
    Optimizer, register_optimizer, OptimizationResult,
)

@register_optimizer("my_solver")
class MySolver(Optimizer):
    def optimize(self, maxiter=100):
        # call self.graph.loss_function(layers) to evaluate
        # ...
        self.graph.apply_optimization_result(best_layers)
        return OptimizationResult(best_layers=best_layers, fun=best_loss)
```

`run_optimization(optimizer="my_solver", ...)` then works. The CLI's
`--optimizer` choices list is hard-coded; extend it in `__main__.py` if
you want CLI access too.

### Adding a plot style

```python
from routing_py_rebuild.plotting import register_style

@register_style("my_style")
def my_style(plot_data, *, out_dir=None, **kwargs):
    # plot_data exposes: k, positions, layers, complete_graph,
    # loss_analysis, loss_crossing, filepath
    ...
```

`visualize_layers(plot_data, style="my_style")` then dispatches into it.
The CLI's `--style` choices are also hard-coded; extend in
`__main__.py` if needed.

---

## Notes on parity with the original

- **Layer-plot consolidation, then 2026-05-15 rewrite**: the original
  `visualize` / `visualize2` / ... / `visualize6` were first collapsed
  into a single `"visualize"` style. That style was then
  **wholesale-replaced** (2026-05-15) for large-`k` readability: a
  2-row layout (square edge-graph panels + bar charts), all node
  labels, node-anchored same-side arcs, equal-spaced integer histogram
  ticks. The `minimize{k}.b16` overlay row, separate-file mode,
  loss-bar annotations, and the `node_params` / `save_combined` /
  `panel_size` kwargs were dropped at that point. See REFACTOR_GOALS.md
  "plotting 布局改版".
- **Optimizers extracted**: `routing_method_dualsa` /
  `routing_method_differential_evolution` were removed from the graph
  class and reimplemented as `DualAnnealingOptimizer` /
  `DifferentialEvolutionOptimizer`. Strategy / popsize match the
  original; the rebuild *adds* deterministic default seeding (`seed=5859`
  forwarded to scipy), which the original did not pass.
- **Differential-evolution seed-population fix**: the original built a
  4-pattern initial population but never passed it to
  `differential_evolution(...)`, leaving scipy on its default `init`. The
  rebuild passes `init=init_population` (the 4 seed patterns + random
  fill) by default. To revert to the original implicit behavior, pass
  any scipy-supported init string via `optimizer_kwargs` (e.g.
  `{"init": "sobol"}` or `{"init": "halton"}`) — user-supplied `init`
  wins. "sobol" is **not** specifically the original's behavior; it's
  just one of the strings that opts out of the seed-population path.
- **JSON persistence is no longer a plot side-effect**: in the original
  `visualize` saved both the figure and a `subgraphsdata.json`. In the
  rebuild only `api.run_optimization` and an explicit call to
  `SiNInterconnectionGraph.save_subgraphs_to_json` write JSON;
  `visualize_layers(style="visualize")` only writes figures.
- **Plotting is JSON-driven and decoupled**: post-optimization plotting
  reads the JSON via `PlotData.from_json` and never touches the graph.
  `PlotData.from_graph` is an escape hatch — read-only, requires the
  caller to have run `analyze_loss()` first.
- **Color palette**: the layer-plot's colormap was `cividis` in the
  original `visualize4`; the rebuild uses `coolwarm` as the base palette
  (`DEFAULT_COLORMAP` — the colormap object itself is full-range over
  [0, 1], no truncation) and applies the [0, 0.95] reservation window
  at every value→color call via `crossings_color` (the same window
  `visualize6` already used inline). The ring view, formerly `plasma`,
  shares the same `coolwarm` palette so colors agree across views.
- **Perimeter pin preserved**: `loss_function`'s perimeter rule
  (`|u-v| ∈ {1, k-1}` → layer 0) is preserved.
- **Seed plumbed through**: `seed=5859` is the default and is forwarded
  to scipy.
