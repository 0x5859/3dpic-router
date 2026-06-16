```
██████╗ ██████╗ ██████╗ ██╗ ██████╗    ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██████╗
╚════██╗██╔══██╗██╔══██╗██║██╔════╝    ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝██╔══██╗
 █████╔╝██║  ██║██████╔╝██║██║         ██████╔╝██║   ██║██║   ██║   ██║   █████╗  ██████╔╝
 ╚═══██╗██║  ██║██╔═══╝ ██║██║         ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝  ██╔══██╗
██████╔╝██████╔╝██║     ██║╚██████╗    ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗██║  ██║
╚═════╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝    ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝
```

# 3D-PIC Router

*A layer-assignment router for 3D photonic integrated circuits.*

Routing optimization for a dual-layer SiN waveguide photonic interposer. Edges of a complete interconnect graph are assigned to waveguide layers so as to minimize an objective combining **crossing loss**, **taper loss**, and **interlayer-crossing loss**.

Two parallel, numerically-aligned implementations ship:

- **`code/routing_py_rebuild/`** — Python library + CLI (the reference implementation).
- **`code/routing_cpp_rebuild/`** — C++ implementation (Dual Annealing core, optional L-BFGS polish).

Both emit the same self-contained `subgraphsdata.json` snapshot, validated against the
JSON Schemas in `code/schema/`.

## Layout

| Path | What lives here |
|---|---|
| `code/routing_py_rebuild/` | Python core: `core.py` (graph + crossings + loss + JSON I/O), `api.py`, `crosstalk.py`, `positions.py`, `io_utils.py`, the `optimizers/` registry (dual annealing, differential evolution, GA, PSO, CMA-ES, BO, …), `statistics/` (run reports), and `plotting/` (JSON-driven figure rendering). |
| `code/routing_cpp_rebuild/` | C++ implementation. `src/` + `include/sinic/`, the vendored `external/dual-annealing/` GSA library, `tests/`, and a `build_and_run.sh` one-shot driver. |
| `code/schema/` | JSON Schemas for the data contract (`subgraphsdata.schema.json`, `run_report.schema.json`). Validated at write **and** read time by both implementations. |
| `code/tests/` | Python test suite — parity, crossings, crosstalk, multilayer, per-optimizer, and statistics — plus `golden/` reference fixtures. |
| `code/assets/MinimizedRectlinear/` | Pre-computed `minimize{k}.b16` point sets consumed by the `io_utils` readers. |

> **Path note:** `schema/` is a runtime dependency of both cores and is located relative
> to the `code/` tree (Python: sibling of `routing_py_rebuild/`; C++: walked up from the
> executable). Keep the `code/` directory layout intact.

## Python quickstart

The project is [uv](https://docs.astral.sh/uv/)-managed.

```bash
uv sync                                   # create .venv, install routing_py_rebuild + deps

# End-to-end optimize (k=12, dual annealing), writing JSON + figures to /tmp/run
uv run python -m routing_py_rebuild optimize --k 12 --maxiter 200 --output-dir /tmp/run

# Re-render figures later from a saved JSON snapshot (self-contained)
uv run python -m routing_py_rebuild plot \
    --json /tmp/run/.../subgraphsdata.json --style visualize --out-dir /tmp/plotout
```

As a library:

```python
from routing_py_rebuild import run_optimization, plot_from_json

res = run_optimization(k=12, optimizer="dual_annealing", maxiter=200,
                       output_dir="/tmp/sin_run/")
print(res["loss"], res["json_path"])
plot_from_json(res["json_path"], style="visualize", out_dir="/tmp/sin_plot/")
```

See `code/routing_py_rebuild/readme.md` for the full module-level reference (architecture,
the optimizer/plot-style registries, and the JSON contract).

## C++ quickstart

```bash
cd code/routing_cpp_rebuild
./build_and_run.sh                  # default: 40 nodes / 500 iters → ./output/
./build_and_run.sh -n 8 -i 200      # 32 nodes, 200 iters
./build_and_run.sh --help           # all options
```

Requires CMake ≥ 3.9 and a C++17 compiler. On macOS it links Accelerate (vecLib)
automatically; on Linux install `libopenblas-dev`. Build flags and runtime parameters are
documented in `code/routing_cpp_rebuild/README.md`. Third-party license terms for the
vendored Dual Annealing / L-BFGS / PCG libraries are in
`code/routing_cpp_rebuild/THIRD_PARTY_LICENSES.md`.

## Tests

```bash
# Python
uv run pytest code/tests

# C++ (built and run via CMake/CTest)
cd code/routing_cpp_rebuild
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBFGS_USE_BLAS=ON
cmake --build build -j && ctest --test-dir build --output-on-failure
```

## License & attribution

The original 3D-PIC Router code is released under the [MIT License](LICENSE). This
covers `code/routing_py_rebuild/`, `code/routing_cpp_rebuild/{src,include/sinic,tests}`,
`code/tests/`, `code/experiments/`, and `code/schema/`.

This project **bundles third-party code** that retains its own (permissive) licenses,
under `code/routing_cpp_rebuild/external/` and `code/routing_cpp_rebuild/include/nlohmann/`:

| Library | License | Copyright | Upstream |
|---|---|---|---|
| dual-annealing (GSA core) | BSD-3-Clause | Tom Westerhout | <https://github.com/twesterhout/dual-annealing> |
| lbfgs-cpp | BSD-3-Clause | Tom Westerhout | <https://github.com/twesterhout/lbfgs-cpp> |
| nlohmann/json | MIT | Niels Lohmann | <https://github.com/nlohmann/json> |
| Catch2 | BSL-1.0 | Two Blue Cubes Ltd | <https://github.com/catchorg/Catch2> |
| gsl-lite | MIT | M. Moene / Microsoft | <https://github.com/gsl-lite/gsl-lite> |
| pcg-cpp | Apache-2.0 OR MIT | Melissa O'Neill | <https://github.com/imneme/pcg-cpp> |

Full per-library terms, copyright holders, and in-tree file locations are documented in
[`code/routing_cpp_rebuild/THIRD_PARTY_LICENSES.md`](code/routing_cpp_rebuild/THIRD_PARTY_LICENSES.md).
Their license texts travel with the source under `external/` — **keep them intact** when
redistributing.

