# routing_cpp_rebuild

3D-SiNIC 互连路由优化（C++ 实现）。基于 **Dual Annealing**（外加可选的 L-BFGS 局部搜索）将一组节点之间的连边分配到两个层，最小化由 *crossing loss*、*taper loss*、*interlayer loss* 加权得到的目标函数。

本目录是原 `routing_cpp` 在 macOS / Apple Silicon (arm64) 上做最小适配后的可运行副本。

## 目录结构

```
routing_cpp_rebuild/
├── CMakeLists.txt           # 顶层 CMake
├── build_and_run.sh         # 一键 build + 运行（全部相对路径）
├── README.md
├── THIRD_PARTY_LICENSES.md  # 第三方库许可证（保留以满足分发要求）
├── src/
│   └── main.cpp             # 入口、SiNInterconnectionGraph、参数扫描
├── include/
│   └── nlohmann/json.hpp    # 单头文件 JSON 库
├── external/
│   └── dual-annealing/      # 外部 dual annealing 库（含 lbfgs-cpp、gsl-lite、pcg-cpp 等）
├── build/                   # （脚本生成）CMake 构建输出
└── output/                  # （程序生成）结果 JSON 写到这里
```

## 依赖

- **CMake** ≥ 3.9
- **C++17** 编译器
  - macOS：Apple clang
  - Linux：GCC 9+ 或 clang 9+
- **BLAS / CBLAS**
  - macOS：自动使用 Accelerate 框架（vecLib），无需安装
  - Linux：需要 `libopenblas-dev` 或 `libatlas-base-dev`，CMake 通过 `find_package(CBLAS)` 查找

## 构建与运行

### 一键脚本（推荐）

```bash
./build_and_run.sh                     # 默认：40 节点 / 500 iter
./build_and_run.sh -n 8 -i 200         # 32 节点、200 iter
./build_and_run.sh --help              # 列出全部选项
```

脚本动作：

1. `cd` 到自身所在目录，所有路径相对解析。
2. 解析 CLI 参数 → 导出为环境变量（C++ 端用 `std::getenv` 读取）。
3. 首次执行时用 `Release + LBFGS_USE_BLAS=ON` 配置 CMake，写入 `./build/`。
4. 增量构建 `Autowiring_CPP` 目标。
5. 运行 `./build/Autowiring_CPP`，结果写入 `./output/`（或 `--out` 指定的目录）。

二次运行会复用 `build/CMakeCache.txt`，只做增量编译。强制全量重配 → 删掉 `build/`。

### 运行时参数（不用改源码）

| CLI flag | 等价环境变量 | 默认 | 说明 |
|---|---|---|---|
| `-n, --nodes-per-side N` | `NODES_MIN`/`NODES_MAX` | `10` | 每边节点数，总节点 = `4 × N` |
| `--nodes-min N` / `--nodes-max N` | `NODES_MIN` / `NODES_MAX` | `10` / `10` | 扫描范围（min ≤ N ≤ max） |
| `-i, --iter N` | `DUALSA_ITER` | `500` | dual annealing 迭代数 |
| `--cl "a,b,c"` | `CL_VALUES` | `0.1` | crossing loss 取值列表 |
| `--tl "a,b,c"` | `TL_VALUES` | `0.05` | taper loss 取值列表 |
| `--itl "a,b,c"` | `ITL_VALUES` | `0` | interlayer loss 取值列表 |
| `--out DIR` | `OUTPUT_DIR` | `./output/` | 输出目录 |
| `--no-run` | — | — | 只 build 不运行 |

启动时控制台会打印一行 `Config: ...` 汇总当次使用的参数。

每个 `(cl, tl, itl, nodes_per_side)` 组合会写入一个独立子目录：

```
output/cl_<cl>_tl_<tl>_itl_<itl>_nodes_<4N>/subgraphsdata.json
```

例：`./build_and_run.sh --cl "0.05,0.1" --tl "0.05" --itl "0,0.001" -n 4` → 4 个目录、4 份 JSON。

### 手动 CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLBFGS_USE_BLAS=ON
cmake --build build -j
NODES_MIN=8 NODES_MAX=8 DUALSA_ITER=200 ./build/Autowiring_CPP   # 直接传环境变量
```

> ⚠️ 一定要带 `-DCMAKE_BUILD_TYPE=Release`。默认空构建类型相当于 `-O0`，对这种内部循环密集的算法会慢 10–50×。

## 输出格式

`subgraphsdata.json`：每层的边列表（节点对、坐标、损失等），以及 `genParam` 元数据（默认含 `Nodes`）。

控制台示例：

```
Optimal average loss by EXTERNAL Dual Annealing = 1.6709e+06
============ Loss Analysis ============
Layer 0 => count=410, avg=15.85, var=72.57, std=8.52, range=34.2
Layer 1 => count=370, avg=17.66, var=50.33, std=7.09, range=30.8
[Flattened layers] => count=780, avg=16.71, ...
[Planar Graph]     => avg=46.87, ...      ← 不分层的对照
Subgraphs saved to ./output/cl_0.10_tl_0.05_itl_0.000_nodes_40/subgraphsdata.json
```

注：求解过程中 `dual_annealing` 还会向 **stderr** 大量打印 `trace: updating best:` 日志（每次目标函数取得新最小值）。这些是诊断信息，不影响结果。需要静默：

```bash
./build/Autowiring_CPP 2>/dev/null
```

## 性能参考（M-系列 Apple Silicon，Release 构建）

- 40 节点、500 dual_annealing 迭代、L-BFGS 关闭：约 **3:35**（CPU 时间 ≈ wall × 0.98）
- 与 WSL/x86_64 Release 同档

## 相对原始 `routing_cpp` 的修改

为了在 macOS arm64 上跑通，对几处做了最小改动：

1. **`external/dual-annealing/third_party/lbfgs-cpp/CMakeLists.txt`**
   - 把 `target_compile_definitions(lbfgs INTERFACE LBFGS_USE_BLAS=1)` 改成 `PUBLIC`（原代码 bug：`INTERFACE` 不会作用到 `lbfgs.cpp` 自身）。
   - `APPLE` 分支跳过 `find_package(CBLAS)`，直接链接 Accelerate 框架并加上 vecLib 头文件搜索路径。Linux 分支保持原 `find_package(CBLAS)` 流程。

2. **`external/dual-annealing/third_party/lbfgs-cpp/src/lbfgs.cpp`**
   - `emplace_back_kernel_8` 函数与其调用块用 `#if defined(__x86_64__) || defined(_M_X64)` 包起来；arm64 走源码里已有的标量 fallback。
   - BLAS 路径在所有平台都可用，所以 `<immintrin.h>` 不再被包含。

3. **`external/dual-annealing/include/chain.hpp`**
   - `dual_annealing::minimize` 中：当 `local_search_parameters.max_iter == 0` 时跳过 L-BFGS `local_search`，避免每次新 best 都做一遍 lbfgs setup（在 zero-gradient objective 下纯属浪费）。

4. **`src/main.cpp`**
   - 移除 `/mnt/e/...` 硬编码的 WSL 头文件路径，改回 `#include <nlohmann/json.hpp>`；补 `<sstream>` / `<cstdlib>`。
   - `static_cast<std::int64_t>(maxiter)` → `static_cast<size_t>(maxiter)`，去掉 `param_t` 初始化的 narrowing warning。
   - `local_search_params.max_iter = 0`（显式关闭 L-BFGS 局部搜索；与 chain.hpp 配合后等价完全跳过）。
   - 引入 `env_int/env_str/env_list` 三个小辅助，把所有 `main()` 顶部硬编码参数（`nodes_per_side_min/max`、`dualsa_iter`、`cl/tl/itl_values`、`base_path`）改成可从环境变量覆盖；脚本端透过 CLI flag 转发，源码无需再改。

> 这些改动都是平台/编译适配，不改算法逻辑。在 x86_64 Linux/WSL 上同样可以构建（走 `find_package(CBLAS)` 分支）。
