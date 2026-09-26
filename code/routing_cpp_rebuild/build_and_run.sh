#!/usr/bin/env bash
# Build and run the SiNIC routing optimizer (dual annealing).
# All paths are relative to this script's location.
#
# Runtime parameters can be set via CLI flags or environment variables (flags win):
#
#   -n, --nodes-per-side N       Use N nodes per side (total = 4 * N).
#                                Sets both min and max. Default: 10  -> 40 total nodes.
#       --nodes-min N            Lower bound for the nodes-per-side sweep.
#       --nodes-max N            Upper bound for the nodes-per-side sweep.
#   -i, --iter N                 Dual-annealing iterations.   Default: 500.
#       --cl  "a,b,c"            Crossing-loss values.        Default: 0.1
#       --tl  "a,b,c"            Taper-loss values.           Default: 0.05
#       --itl "a,b,c"            Interlayer-loss values.      Default: 0.001
#       --out DIR                Output directory.            Default: ./output/
#       --no-run                 Build only, don't execute.
#   -h, --help                   Show this help and exit.
#
# Examples:
#   ./build_and_run.sh                                 # defaults: 40 nodes, 500 iter
#   ./build_and_run.sh -n 8 -i 200                     # 32 nodes, 200 iter
#   ./build_and_run.sh --nodes-min 6 --nodes-max 10    # sweep 24,28,32,36,40 nodes
#   ./build_and_run.sh --cl "0.05,0.1,0.2"             # parameter sweep over cl

set -euo pipefail

cd -- "$(dirname -- "$0")"

BUILD_DIR=build
TARGET=Autowiring_CPP
RUN=1

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }

# Parse CLI flags (override any inherited env vars).
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--nodes-per-side) NODES_MIN=$2; NODES_MAX=$2; shift 2 ;;
        --nodes-min)         NODES_MIN=$2;              shift 2 ;;
        --nodes-max)         NODES_MAX=$2;              shift 2 ;;
        -i|--iter)           DUALSA_ITER=$2;            shift 2 ;;
        --cl)                CL_VALUES=$2;              shift 2 ;;
        --tl)                TL_VALUES=$2;              shift 2 ;;
        --itl)               ITL_VALUES=$2;             shift 2 ;;
        --out)               OUTPUT_DIR=$2;             shift 2 ;;
        --no-run)            RUN=0;                     shift   ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Export so the C++ binary can read them via std::getenv. Unset names stay unset
# (the binary picks its hard-coded default in that case).
for v in NODES_MIN NODES_MAX DUALSA_ITER CL_VALUES TL_VALUES ITL_VALUES OUTPUT_DIR; do
    [ -n "${!v-}" ] && export "$v"
done

mkdir -p "$BUILD_DIR" "${OUTPUT_DIR:-output}"

# Configure once; reuse the existing cache on subsequent runs.
# LBFGS_USE_BLAS=ON: lbfgs.cpp then uses <cblas.h> (Accelerate on macOS) instead
# of x86-only <immintrin.h> -- REQUIRED on Apple Silicon (arm64); matches README.
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLBFGS_USE_BLAS=ON
fi

cmake --build "$BUILD_DIR" --target "$TARGET" -j

if [ "$RUN" -eq 0 ]; then
    echo "Build complete. Skipping run (--no-run)."
    exit 0
fi

echo
echo "==> Running ./$BUILD_DIR/$TARGET"
echo
"./$BUILD_DIR/$TARGET"
