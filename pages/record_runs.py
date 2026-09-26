"""Record the optimization runs that the replay page (pages/index.html) shows.

One run per boundary layout: k = 12 nodes, 2 layers (coupler on layer 0), dual
annealing, 200 iterations, seed 5859, and the default loss model. Each run
writes ``RUNS_DIR/<layout>/<run>/optimization_history.json``, which
build_page.py turns into the page.

Usage, from the repository root:
    uv run python pages/record_runs.py RUNS_DIR [LAYOUT ...]
"""
import argparse
import logging
import os
import warnings

from routing_py_rebuild.api import run_optimization
from routing_py_rebuild.positions import distribute_nodes, perimeter_is_simple

MAXITER = 200
SEED = 5859

# A 6 x 4 mm chip with 1 mm chamfered corners; 12 ports at uneven spacing,
# numbered counter-clockwise from the bottom-left (the boundary order).
CHIP_PORTS = {
    0: (1.4, 0.0), 1: (2.3, 0.0), 2: (3.9, 0.0),     # bottom edge
    3: (5.5, 0.5),                                    # lower-right chamfer
    4: (6.0, 1.6), 5: (6.0, 2.2),                     # right edge
    6: (4.6, 4.0), 7: (3.1, 4.0), 8: (2.5, 4.0), 9: (1.2, 4.0),  # top edge
    10: (0.4, 3.4),                                   # upper-left chamfer
    11: (0.0, 1.5),                                   # left edge
}

# The same chip with a 2 mm wide, 1.5 mm deep notch cut into the top edge:
# three ports sit on the notch walls and floor, inside the convex hull, so
# the crossing structure (and the optimization) differs from the convex runs.
NOTCHED_CHIP = {
    0: (1.0, 0.0), 1: (2.5, 0.0), 2: (4.2, 0.0), 3: (5.2, 0.0),   # bottom edge
    4: (6.0, 1.3), 5: (6.0, 2.8),                                 # right edge
    6: (5.0, 4.0),                                                # top edge, right of the notch
    7: (4.0, 3.2), 8: (3.0, 2.5), 9: (2.0, 3.2),                  # notch wall, floor, wall
    10: (1.0, 4.0),                                               # top edge, left of the notch
    11: (0.0, 2.0),                                               # left edge
}

LAYOUTS = {
    "square": distribute_nodes("square", nodes_per_side=3, side_length=1),
    "rectangle": distribute_nodes("rectangle", nodes_on_length=4, nodes_on_width=2,
                                  length=5, width=3),
    "triangle": distribute_nodes("triangle", triangle_nodes_per_side=4),
    "hexagon": distribute_nodes("polygon", polygon_n_sides=6, polygon_nodes_per_side=2),
    "circle": distribute_nodes("circle", k=12),
    "three_sides": distribute_nodes("partial_rectangle",
                                    side_counts={"top": 4, "right": 4, "bottom": 4}),
    "chip_ports": CHIP_PORTS,
    "notched_chip": NOTCHED_CHIP,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_dir", help="output directory, one subdirectory per layout")
    parser.add_argument("layouts", nargs="*", metavar="LAYOUT",
                        help=f"layouts to run (default: all of {', '.join(LAYOUTS)})")
    args = parser.parse_args()
    unknown = sorted(set(args.layouts) - set(LAYOUTS))
    if unknown:
        parser.error(f"unknown layout(s) {', '.join(unknown)}; choose from {', '.join(LAYOUTS)}")

    warnings.simplefilter("ignore")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    for key in args.layouts or LAYOUTS:
        pos = LAYOUTS[key]
        assert len(pos) == 12 and perimeter_is_simple(pos), key
        res = run_optimization(
            k=len(pos), positions=pos, optimizer="dual_annealing", maxiter=MAXITER, seed=SEED,
            output_dir=os.path.join(args.runs_dir, key), plot=False, run_loss_analysis=False,
            progress="record", progress_kwargs={"formats": []},
        )
        print(f"{key}: {res['loss']:.4f} dB -> {res['progress']['history']}", flush=True)


if __name__ == "__main__":
    main()
