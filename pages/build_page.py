"""Build the routing replay page (pages/index.html) from recorded runs.

The page draws every frame itself, as inline SVG, from the optimization
histories that record_runs.py writes. Geometry, colors, tick positions and the
evaluated-loss band come from the library (edge bends, the crossings color
map, MaxNLocator, the eval envelope), so the page shows what ProgressFigure
shows, as vectors, with the crossing counts set as page text. All layouts are
inlined; the page fetches nothing but its web font.

Usage, from the repository root:
    uv run python pages/build_page.py RUNS_DIR [--out pages/index.html]
"""
import argparse
import glob
import html
import json
import logging
import math
import os
import sys
import warnings

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.ticker import MaxNLocator
from routing_py_rebuild import core
from routing_py_rebuild.api import make_graph
from routing_py_rebuild.plotting._colormap import CMAP_LOWER, CMAP_SCALE, DEFAULT_COLORMAP
from routing_py_rebuild.plotting.layers import _edge_bends, _label_fontsize, _node_size
from routing_py_rebuild.plotting.progress_figure import ProgressData, _rolling, _y_limits
from routing_py_rebuild.plotting.progress_replay import auto_fps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from record_runs import LAYOUTS, MAXITER, SEED  # noqa: E402

DEFAULT = "square"
# Layouts with nodes inside the convex hull of the others; every other layout
# has the same crossing structure, which the page's note states.
NON_CONVEX = {"notched_chip"}
# Figure proportions to mimic (ProgressFigure at k=12, L=2: ~7.7 cm panels).
PANEL_PT = 7.7 / 2.54 * 72
LOSS = {
    "loss_crossing": core.DEFAULT_LOSS_CROSSING,
    "loss_taper": core.DEFAULT_LOSS_TAPER,
    "loss_interlayercrossing": core.DEFAULT_LOSS_INTERLAYERCROSSING,
}

# key: (picker label, one-line description)
INFO = {
    "square": ("Square", "Square, 3 nodes per side"),
    "rectangle": ("Rectangle", "5 × 3 rectangle, 4 + 2 nodes per side"),
    "triangle": ("Triangle", "Triangle, 4 nodes per side"),
    "hexagon": ("Hexagon", "Hexagon, 2 nodes per side"),
    "circle": ("Circle", "Circle, 12 nodes"),
    "three_sides": ("Three sides", "3 sides of a 5 × 3 rectangle, 4 nodes each"),
    "chip_ports": ("Chip ports", "Custom coordinates: 6 × 4 mm chip, chamfered corners"),
    "notched_chip": ("Notched chip", "Custom coordinates: notched 6 × 4 mm chip, non-convex"),
}


def pct(a: float, b: float) -> str:
    return f"{(b - a) / abs(a) * 100:+.1f}%".replace("-", "−")


def status(frame, index: int, total: int, initial: float) -> str:
    head = "Final result" if frame.final else f"Improvement {index + 1} / {total}"
    text = f"{head} · evaluation {frame.eval_index + 1:,} · {frame.loss:.4f} dB"
    return text + (f" ({pct(initial, frame.loss)})" if index or frame.final else "")


def fill_gaps(values: np.ndarray) -> np.ndarray:
    """Buckets without a finite loss (NaN) take their neighbour's value, so
    the band path stays connected."""
    out = np.asarray(values, dtype=float).copy()
    finite = np.flatnonzero(np.isfinite(out))
    if not len(finite):
        return out
    idx = np.maximum.accumulate(np.where(np.isfinite(out), np.arange(len(out)), -1))
    idx[idx < 0] = finite[0]
    return out[idx]


def r6(x: float) -> float:
    """Six significant digits: plenty for pixels, compact in JSON."""
    return float(f"{x:.6g}")


def glyph(positions: dict, closed: bool = True) -> str:
    """Tiny SVG of the node layout: the boundary walk and the nodes
    (``closed=False`` leaves a side without nodes open)."""
    pts = np.array([positions[n] for n in sorted(positions)], dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    w, h, pad = 56.0, 36.0, 4.0
    s = min((w - 2 * pad) / max(hi[0] - lo[0], 1e-9), (h - 2 * pad) / max(hi[1] - lo[1], 1e-9))
    ox = (w - s * (hi[0] - lo[0])) / 2
    oy = (h - s * (hi[1] - lo[1])) / 2
    xy = [(ox + s * (x - lo[0]), h - (oy + s * (y - lo[1]))) for x, y in pts]
    ring = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.1"/>' for x, y in xy)
    return (f'<svg class="glyph" viewBox="0 0 {w:.0f} {h:.0f}" aria-hidden="true">'
            f'<{"polygon" if closed else "polyline"} class="ring" points="{ring}"/>'
            f'<g class="nodes">{dots}</g></svg>')


def positions_json(positions: dict) -> str:
    body = ",\n".join(f'  "{n}": [{json.dumps(float(x))}, {json.dumps(float(y))}]'
                      for n, (x, y) in sorted(positions.items()))
    return "{\n" + body + "\n}"


def edge_paths(positions: dict, edges: list, span: float) -> list[str]:
    """SVG path per edge (y flipped): a line, or the arc3 quadratic Bezier
    that the layer plots draw for chords along the boundary."""
    q = 10 ** max(0, 4 - int(math.floor(math.log10(span))))  # ~1e-4 of the span

    def f(v: float) -> str:
        return f"{round(v * q) / q:g}"

    out = []
    for (u, v), rad in zip(edges, _edge_bends(positions, edges), strict=True):
        x0, y0 = positions[u]
        x2, y2 = positions[v]
        if rad == 0.0:
            out.append(f"M{f(x0)} {f(-y0)}L{f(x2)} {f(-y2)}")
            continue
        dx, dy = x2 - x0, y2 - y0
        cx = (x0 + x2) / 2 + rad * dy
        cy = (y0 + y2) / 2 - rad * dx
        out.append(f"M{f(x0)} {f(-y0)}Q{f(cx)} {f(-cy)} {f(x2)} {f(-y2)}")
    return out


def layout_entry(key: str, runs_dir: str) -> dict:
    """Everything the page needs for one layout, from its recorded run."""
    history = glob.glob(os.path.join(runs_dir, key, "*", "optimization_history.json"))
    assert len(history) == 1, (key, history)
    data = ProgressData.from_json(history[0])
    assert data.positions == {n: tuple(map(float, p)) for n, p in LAYOUTS[key].items()}, key
    assert all(data.run[name] == value for name, value in LOSS.items()), (key, data.run)
    assert (data.run["maxiter"], data.run["seed"]) == (MAXITER, SEED), (key, data.run)
    assert data.L <= 10, "layers are packed one digit per edge"

    # Geometry (SVG user units = layout units, y flipped).
    pos = data.positions
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    span = float(max(np.ptp(xs), np.ptp(ys), 1e-9))
    margin = 0.09 * span
    view = [float(xs.min() - margin), float(-(ys.max() + margin)),
            float(np.ptp(xs) + 2 * margin), float(np.ptp(ys) + 2 * margin)]
    unit = view[2] / PANEL_PT  # layout units per figure point
    node_r = math.sqrt(_node_size(data.k)) / 2 * unit
    font = _label_fontsize(data.k) * unit

    # Colors: the figure's crossings -> color map, fixed over the run.
    vmax = max(1, data.max_crossings())
    palette = [to_hex(DEFAULT_COLORMAP(CMAP_LOWER + CMAP_SCALE * min(c / vmax, 1.0)))
               for c in range(vmax + 1)]
    stops = [[r6(t), to_hex(DEFAULT_COLORMAP(CMAP_LOWER + CMAP_SCALE * t))]
             for t in np.linspace(0.0, 1.0, 17)]
    cticks = [int(t) for t in MaxNLocator(nbins=5, integer=True).tick_values(0, vmax)
              if 0 <= t <= vmax]

    # Interlayer crossings per frame (adjacent layers), from the crossing index.
    graph = make_graph(k=data.k, positions=pos)
    graph.build_crossings_index()
    assert [tuple(e) for e in graph._edge_list] == [tuple(e) for e in data.edges], key
    pairs = graph._crossing_pairs
    first, last = data.frames[0], data.frames[-1]
    improvements = len(data.frames) if data.improvements is None else data.improvements
    bx, by = data.best_curve()
    evaluations = max(data.evaluations, int(bx[-1]))

    frames = []
    for i, fr in enumerate(data.frames):
        la, lb = fr.layers[pairs[:, 0]], fr.layers[pairs[:, 1]]
        frames.append({
            "l": "".join(str(int(v)) for v in fr.layers),
            "c": [int(v) for v in fr.crossings],
            "x": fr.eval_index + 1, "y": r6(fr.loss), "f": int(fr.final),
            "i": int(np.count_nonzero(np.abs(la - lb) == 1)),
            "s": status(fr, i, improvements, float(by[0])),
        })
    per_layer = [int(last.crossings[last.layers == j].sum()) // 2 for j in range(data.L)]

    # Loss panel: same limits, ticks and band as ProgressFigure's replay.
    y0, y1 = _y_limits(by)
    yticks = [r6(t) for t in MaxNLocator(nbins=5).tick_values(y0, y1) if y0 <= t <= y1]
    ex, lo, hi = data.envelope.steps(evaluations) if data.envelope else ([], [], [])
    if len(ex):
        lo = fill_gaps(_rolling(np.asarray(lo), np.fmin))
        hi = fill_gaps(_rolling(np.asarray(hi), np.fmax))

    label, blurb = INFO[key]
    return {
        "key": key, "label": label, "blurb": blurb,
        "result": f"{first.loss:.4f} → {last.loss:.4f} dB ({pct(first.loss, last.loss)})",
        "loss": f"{first.loss:.4f} → {last.loss:.4f}", "change": pct(first.loss, last.loss),
        "glyph": glyph(pos, closed=key != "three_sides"),
        "positions": positions_json(pos),
        "pairs": int(len(pairs)),
        "final": {"per_layer": per_layer, "total": sum(per_layer), "inter": frames[-1]["i"]},
        "fig": {
            "L": data.L, "ecl": data.edge_coupler_layer,
            "view": [r6(v) for v in view], "r": r6(node_r), "font": r6(font),
            "nodes": [[r6(pos[n][0]), r6(-pos[n][1])] for n in sorted(pos)],
            "edges": [[int(u), int(v)] for u, v in data.edges],
            "paths": edge_paths(pos, data.edges, span),
            "palette": palette, "stops": stops, "cticks": cticks,
            "frames": frames, "fps": auto_fps(len(frames)),
            "curve": {
                "x0": 0.8, "x1": r6(max(evaluations, 2.0) * 1.05), "evals": evaluations,
                "y0": r6(y0), "y1": r6(y1), "yticks": yticks,
                "bx": [int(v) for v in bx], "by": [r6(v) for v in by],
                "ex": [r6(v) for v in ex], "lo": [r6(v) for v in lo], "hi": [r6(v) for v in hi],
            },
        },
    }


def render(index: list[dict]) -> str:
    by_key = {e["key"]: e for e in index}
    d = by_key[DEFAULT]
    esc = html.escape

    # The note states that the convex layouts share one crossing structure.
    convex = [e for e in index if e["key"] not in NON_CONVEX]
    assert len({e["pairs"] for e in convex}) == 1, "convex layouts differ in crossing pairs"
    assert len({json.dumps(e["fig"]["frames"]) for e in convex}) == 1, "convex runs differ"
    note = (f"With all links in one layer, every convex layout has {convex[0]['pairs']} "
            f"crossings and the notched chip {by_key['notched_chip']['pairs']}; "
            "the convex runs are identical.")

    picker = "\n".join(
        f'      <button type="button" role="radio" class="pick" data-key="{e["key"]}" '
        f'aria-checked="{"true" if e["key"] == DEFAULT else "false"}" '
        f'tabindex="{0 if e["key"] == DEFAULT else -1}">{e["glyph"]}'
        f'<span>{esc(e["label"])}</span></button>'
        for e in index)
    rows = "\n".join(
        f'          <tr data-key="{e["key"]}"{" class=current" if e["key"] == DEFAULT else ""}>'
        f'<th scope="row"><button type="button" class="rowpick" data-key="{e["key"]}">'
        f'{esc(e["label"])}</button></th><td>{e["final"]["total"]}</td>'
        f'<td>{e["loss"]}</td><td class="chg">{e["change"]}</td></tr>'
        for e in index)

    setup = (f"12 nodes, 2 layers · dual annealing, {MAXITER} iterations · default losses: "
             f'crossing {LOSS["loss_crossing"]:g} dB, taper {LOSS["loss_taper"]:g} dB, '
             f'interlayer crossing {LOSS["loss_interlayercrossing"]:g} dB')
    cli = ("python -m routing_py_rebuild optimize --positions-json layout.json \\\n"
           f"  --maxiter {MAXITER} --progress record")
    python = ("from routing_py_rebuild import load_positions_json, run_optimization\n\n"
              'pos = load_positions_json("layout.json")\n'
              f'run_optimization(k=len(pos), positions=pos, maxiter={MAXITER}, progress="record")')
    client = [{k: e[k] for k in ("key", "blurb", "result", "positions", "fig")} for e in index]

    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as fh:
        page = fh.read()
    for name, value in {
        "SETUP": esc(setup),
        "PICKER": picker,
        "BLURB": esc(d["blurb"]),
        "RESULT": esc(d["result"]),
        "ROWS": rows,
        "NOTE": esc(note),
        "POSITIONS": esc(d["positions"]),
        "CLI": esc(cli),
        "PYTHON": esc(python),
        "DEFAULT": DEFAULT,
        "INDEX_JSON": json.dumps(client, separators=(",", ":"), ensure_ascii=False)
                      .replace("</", "<\\/"),
    }.items():
        token = "{{" + name + "}}"
        assert token in page, token
        page = page.replace(token, value)
    assert "{{" not in page
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_dir", help="directory written by record_runs.py")
    parser.add_argument("--out", default=os.path.join(HERE, "index.html"),
                        help="page to write (default: pages/index.html)")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    index = []
    for key in LAYOUTS:
        entry = layout_entry(key, args.runs_dir)
        final = entry["final"]
        print(f"{key}: {len(entry['fig']['frames'])} frames, crossings {final['per_layer']} "
              f"(total {final['total']}, interlayer {final['inter']}), {entry['result']}",
              flush=True)
        index.append(entry)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render(index))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
