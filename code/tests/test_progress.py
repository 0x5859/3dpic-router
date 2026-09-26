"""Optimization-process visualization: ``run_optimization(progress=...)``.

Covers the three modes (``off`` / ``live`` / ``record``): the tracker
must never perturb the optimizer, the recorded history must end on the
exact routing written to ``subgraphsdata.json``, the history JSON contract
(schema + cross-field checks), the replay renderers, the live view's
headless fallback, and the CLI entry points.
"""
from __future__ import annotations

import itertools
import json
import math
import os

import jsonschema
import matplotlib.pyplot as plt
import numpy as np
import pypdf
import pytest
from matplotlib.text import Text
from PIL import Image, ImageSequence
from routing_py_rebuild.__main__ import main as cli_main
from routing_py_rebuild.api import make_graph, run_optimization
from routing_py_rebuild.plotting import _style
from routing_py_rebuild.plotting.layers import CURVE_FACTOR, _edge_bends
from routing_py_rebuild.plotting.progress_figure import (
    HISTORY_FILENAME,
    ProgressData,
    ProgressFigure,
    ProgressFrame,
    edge_polylines,
    frame_status,
)
from routing_py_rebuild.plotting.progress_live import (
    LiveProgressView,
    LiveState,
    live_display_mode,
    live_status,
)
from routing_py_rebuild.plotting.progress_replay import (
    _Replayer,
    load_progress_data,
    render_progress_replay,
    select_frames,
    show_progress_replay,
)
from routing_py_rebuild.positions import distribute_nodes
from routing_py_rebuild.progress import (
    LossEnvelope,
    ProgressOptions,
    ProgressTracker,
    normalize_progress_mode,
)

# Small + fast: a K_8 dual-annealing run does a few hundred evaluations
# with ~10 improvements. dpi 40 keeps the replay rendering quick.
_FAST = dict(k=8, optimizer="dual_annealing", maxiter=5, seed=5859,
             plot=False, run_loss_analysis=False)
_FAST_PROGRESS = {"dpi": 40, "fps": 20}


def _record(tmp_path, **overrides):
    kw = {**_FAST, **overrides}
    progress_kwargs = {**_FAST_PROGRESS, **kw.pop("progress_kwargs", {})}
    return run_optimization(output_dir=str(tmp_path), progress="record",
                            progress_kwargs=progress_kwargs, **kw)


def _layer_attrs(subgraphs_json: str) -> dict[tuple[int, int], dict]:
    with open(subgraphs_json) as f:
        data = json.load(f)
    out = {}
    for key, block in data.items():
        if key.startswith("Layer_"):
            for u, v, attr in block["edges"]:
                out[(min(u, v), max(u, v))] = attr
    return out


# ---------------------------------------------------------------------------
# Mode "off" + non-perturbation
# ---------------------------------------------------------------------------
def test_progress_off_by_default_writes_nothing(tmp_path):
    res = run_optimization(output_dir=str(tmp_path), save_json=False, **_FAST)
    assert res["progress"] is None
    produced = {name for _root, _dirs, files in os.walk(tmp_path) for name in files}
    assert not produced & {HISTORY_FILENAME, "optimization_progress.gif",
                           "optimization_progress.html", "optimization_live.png"}


@pytest.mark.parametrize("mode", ["live", "record"])
def test_progress_modes_do_not_perturb_the_optimizer(tmp_path, mode):
    """Same seed ⇒ identical per-eval loss trace, summary and result with
    and without tracking (the tracker is a read-only eval_observer)."""
    common = dict(**_FAST, collect_statistics=True, save_json=False)
    off = run_optimization(output_dir=str(tmp_path / "off"), **common)
    on = run_optimization(output_dir=str(tmp_path / mode), progress=mode,
                          progress_kwargs={**_FAST_PROGRESS, "formats": []}, **common)

    def trace(res):
        with open(res["report_path"]) as f:
            rep = json.load(f)
        return [ev["loss"] for ev in rep["trace"]], rep["summary"]

    (l_off, s_off), (l_on, s_on) = trace(off), trace(on)
    assert l_off == l_on
    for key in ("initial_loss", "final_loss", "iterations", "best_at_iter"):
        assert s_off[key] == s_on[key]
    assert off["best_layers"] == on["best_layers"]
    assert off["loss"] == on["loss"]


# ---------------------------------------------------------------------------
# Mode "record"
# ---------------------------------------------------------------------------
def test_record_history_tracks_every_improvement(tmp_path):
    seen: list[float] = []
    res = _record(tmp_path, eval_observer=lambda x, loss: seen.append(float(loss)),
                  progress_kwargs={"formats": []})
    data = ProgressData.from_json(res["progress"]["history"])

    # One frame per strict new best, in evaluation order.
    best, expected = math.inf, []
    for i, loss in enumerate(seen):
        if loss < best:
            best = loss
            expected.append((i, loss))
    got = [(f.eval_index, f.loss) for f in data.frames[: len(expected)]]
    assert got == expected
    assert data.improvements == len(expected)
    assert data.evaluations == len(seen)
    assert data.envelope.count == len(seen)
    assert all(b.loss < a.loss for a, b in zip(data.frames, data.frames[1:], strict=False))
    assert [f.final for f in data.frames] == [False] * (len(data.frames) - 1) + [True]
    assert data.frames[-1].loss == pytest.approx(res["loss"])


def test_record_final_frame_is_the_saved_routing(tmp_path):
    """The last frame must be exactly what subgraphsdata.json holds: same
    layer per edge (perimeter pinned) and same per-edge crossings."""
    res = _record(tmp_path, progress_kwargs={"formats": []})
    data = ProgressData.from_json(res["progress"]["history"])
    attrs = _layer_attrs(res["json_path"])
    final = data.frames[-1]
    assert final.final
    assert len(data.edges) == len(attrs)
    for (u, v), layer, crossings in zip(data.edges, final.layers, final.crossings, strict=True):
        saved = attrs[(min(u, v), max(u, v))]
        assert saved["layer"] == layer
        assert saved["crossings"] == crossings


def test_record_writes_gif_and_html_replay(tmp_path):
    res = _record(tmp_path, progress_kwargs={"formats": ["gif", "html"]})
    info = res["progress"]
    assert info["mode"] == "record"
    data = ProgressData.from_json(info["history"])
    n = len(select_frames(len(data.frames), ProgressOptions().max_frames))

    with Image.open(info["gif"]) as gif:
        assert gif.n_frames == n
        # The iterator re-seeks one image object: read each frame's info in turn.
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]
    assert durations[-1] > durations[1]  # the final routing holds longer

    with open(info["html"], encoding="utf-8") as f:
        page = f.read()
    payload = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert len(payload["frames"]) == len(payload["meta"]) == n
    assert all(f.startswith(("data:image/webp;base64,", "data:image/png;base64,"))
               for f in payload["frames"])
    assert payload["meta"][-1]["final"] is True
    assert payload["meta"][-1]["eval"] == data.frames[-1].eval_index + 1


def test_record_stills_are_publication_ready(tmp_path):
    """pdf / png: the final state as a still — 450 ppi PNG, PDF with
    TrueType (editable) text, and no status line / run label (a caption's
    job), per the figure style rules."""
    res = _record(tmp_path, progress_kwargs={"formats": ["pdf", "png"]})
    info = res["progress"]
    assert {"pdf", "png"} <= set(info) and "gif" not in info
    with Image.open(info["png"]) as png:
        assert png.info["dpi"][0] == pytest.approx(450, abs=1)
    reader = pypdf.PdfReader(info["pdf"])
    fonts = {f.get_object()["/Subtype"]
             for page in reader.pages
             for f in page["/Resources"]["/Font"].values()}
    assert fonts and "/Type3" not in fonts
    text = "".join(page.extract_text() for page in reader.pages)
    assert "Mean edge loss (dB)" in text and "Evaluations" in text
    assert "Final result" not in text and "seed" not in text


def test_record_statistics_time_the_progress_phases(tmp_path):
    res = _record(tmp_path, collect_statistics=True, plot=True,
                  progress_kwargs={"formats": ["gif"]})
    with open(res["report_path"]) as f:
        timings = json.load(f)["timings"]
    assert "progress_history_write_ms" in timings
    assert "progress_replay_ms" in timings
    # The history's wall time is the optimizer's, not "until the file was
    # written" (which would include the plotting that ran in between).
    data = ProgressData.from_json(res["progress"]["history"])
    assert data.wall_ms <= timings["optimization_wall_ms"] + 20
    assert timings["plotting_ms"] > 20


def test_record_respects_max_frames(tmp_path):
    res = _record(tmp_path, progress_kwargs={"formats": ["gif"], "max_frames": 3})
    data = ProgressData.from_json(res["progress"]["history"])
    assert len(data.frames) > 3  # the history keeps every frame ...
    with Image.open(res["progress"]["gif"]) as gif:
        assert gif.n_frames == 3  # ... the animation is capped


def test_interrupt_saves_partial_history(tmp_path, capsys):
    calls = []

    def stop_after_200(x, loss):
        calls.append(loss)
        if len(calls) == 200:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _record(tmp_path, maxiter=50, eval_observer=stop_after_200)
    paths = [os.path.join(root, HISTORY_FILENAME) for root, _d, files in os.walk(tmp_path)
             if HISTORY_FILENAME in files]
    assert len(paths) == 1
    data = ProgressData.from_json(paths[0])
    assert data.frames and not any(f.final for f in data.frames)
    assert data.evaluations == 199  # the interrupted eval never reached the tracker
    assert "interrupted" in capsys.readouterr().out


def test_tracker_appends_final_frame_when_result_differs():
    g = make_graph(k=8)
    tracker = ProgressTracker(g, mode="record")
    n = g.G.number_of_edges()
    tracker(np.zeros(n), 5.0)
    tracker(np.ones(n), 4.0)
    tracker(np.ones(n), 4.5)                   # not an improvement
    tracker(np.full(n, np.nan), float("nan"))  # non-finite: counted, ignored
    assert tracker.improvements == 2 and tracker.evaluations == 4

    other = np.zeros(n)
    other[: n // 2] = 1
    data = tracker.finish(other, 4.0)
    assert len(data.frames) == 3 and data.improvements == 2
    assert [f.final for f in data.frames] == [False, False, True]
    assert data.frames[-1].eval_index == 3
    np.testing.assert_array_equal(data.frames[-1].layers, g.effective_layers(other))

    same = ProgressTracker(g, mode="record")
    same(np.ones(n), 4.0)
    assert len(same.finish(np.ones(n), 4.0).frames) == 1


# ---------------------------------------------------------------------------
# Graph helpers used by the tracker
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,ecl", [(2, 0), (3, 1)])
def test_effective_layers_and_crossings_match_create_subgraphs(L, ecl):
    g = make_graph(k=12, L=L, edge_coupler_layer=ecl)
    rng = np.random.default_rng(7)
    x = rng.uniform(0, L - 1, size=g.G.number_of_edges())
    eff = g.effective_layers(x)
    counts = g.intralayer_crossing_counts(eff)

    g.apply_optimization_result(x)
    g.analyze_loss()
    for i, (u, v) in enumerate(g.G.edges()):
        assert g.G.edges[u, v]["layer"] == eff[i]
        assert g.sub_G[eff[i]].edges[u, v]["crossings"] == counts[i]
    perimeter = [i for i, (u, v) in enumerate(g.G.edges()) if abs(u - v) in (1, g.k - 1)]
    assert set(eff[perimeter]) == {g.perimeter_layer}


# ---------------------------------------------------------------------------
# History JSON contract
# ---------------------------------------------------------------------------
def test_history_round_trip(tmp_path):
    res = _record(tmp_path, progress_kwargs={"formats": []})
    path = res["progress"]["history"]
    data = ProgressData.from_json(path)
    copy = tmp_path / "copy.json"
    data.write_json(copy)
    with open(path) as a, open(copy) as b:
        assert json.load(a) == json.load(b)


@pytest.mark.parametrize("mutate, error", [
    (lambda d: d["frames"][0]["layers"].pop(), ValueError),
    (lambda d: d["frames"][0]["layers"].__setitem__(0, 7), ValueError),
    (lambda d: d["frames"][0].pop("crossings"), jsonschema.ValidationError),
    (lambda d: d.__setitem__("frames", []), jsonschema.ValidationError),
    (lambda d: d["evaluations"]["start"].reverse(), ValueError),
    (lambda d: d["evaluations"]["min"].pop(), ValueError),
    (lambda d: d.__setitem__("perimeter_layer", 5), ValueError),
    (lambda d: d["edges"].__setitem__(0, [0, 99]), ValueError),
    (lambda d: d.__setitem__("schema_version", "2.0"), jsonschema.ValidationError),
])
def test_history_validation_rejects_bad_files(tmp_path, mutate, error):
    res = _record(tmp_path, progress_kwargs={"formats": []})
    with open(res["progress"]["history"]) as f:
        payload = json.load(f)
    mutate(payload)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(error):
        ProgressData.from_json(bad)


def test_loss_envelope_log_buckets():
    q = 4
    env = LossEnvelope(per_octave=q)
    rng = np.random.default_rng(3)
    losses = rng.uniform(1, 2, size=1000)
    losses[[5, 500]] = [np.nan, np.inf]
    for v in losses:
        env.add(float(v))
    snap = env.snapshot()
    assert snap.count == len(losses)
    assert snap.start[0] == 1
    widths = np.diff(np.append(snap.start, snap.count + 1))
    assert np.all(widths[: 2 * q - 1] == 1)  # exact at the start ...
    assert widths[-2] >= 128                  # ... log-spaced later
    assert len(snap.start) < 40               # ~2q + q·log2(n / 2q)
    for j, s in enumerate(snap.start):
        e = snap.start[j + 1] if j + 1 < len(snap.start) else snap.count + 1
        block = losses[s - 1: e - 1]
        block = block[np.isfinite(block)]
        if block.size:
            assert snap.lo[j] == block.min() and snap.hi[j] == block.max()
        else:  # only the NaN evaluation (a width-1 bucket)
            assert np.isnan(snap.lo[j]) and np.isnan(snap.hi[j])


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------
def test_select_frames_keeps_first_and_final():
    assert select_frames(5, 10) == [0, 1, 2, 3, 4]
    picked = select_frames(1000, 50)
    assert picked[0] == 0 and picked[-1] == 999 and len(picked) <= 50
    assert picked == sorted(set(picked))


def test_load_progress_data_accepts_run_or_output_dir(tmp_path):
    res = _record(tmp_path / "out", progress_kwargs={"formats": []})
    history = res["progress"]["history"]
    run_dir = os.path.dirname(history)
    for path in (history, run_dir, tmp_path / "out"):  # file, run dir, --output-dir
        assert load_progress_data(path).frames
    other = tmp_path / "out" / "second_run"
    other.mkdir()
    (other / HISTORY_FILENAME).write_text(open(history).read())
    with pytest.raises(ValueError, match="several recorded runs"):
        load_progress_data(tmp_path / "out")


def test_render_replay_from_history_path(tmp_path):
    res = _record(tmp_path / "run", progress_kwargs={"formats": []})
    run_dir = os.path.dirname(res["progress"]["history"])
    paths = render_progress_replay(run_dir, out_dir=tmp_path / "out", formats=["html"],
                                   dpi=40)
    assert set(paths) == {"html"} and os.path.getsize(paths["html"]) > 0
    with pytest.raises(ValueError, match="unknown replay format"):
        render_progress_replay(run_dir, out_dir=tmp_path / "out", formats=["mp4"])


def test_show_replay_without_gui_returns(tmp_path, capsys):
    assert live_display_mode() == "file"  # test suite runs on Agg
    res = _record(tmp_path, progress_kwargs={"formats": []})
    show_progress_replay(res["progress"]["history"])
    assert "no GUI backend" in capsys.readouterr().out


def test_progress_figure_typography(tmp_path):
    """Figure style rules: one 7 pt font stack, no bold / figure title,
    sentence-case axis labels with units, four 0.5 pt spines, inward ticks."""
    res = _record(tmp_path, progress_kwargs={"formats": []})
    data = ProgressData.from_json(res["progress"]["history"])
    pf = ProgressFigure(data, dpi=40, context=True)
    _Replayer(data, [0, len(data.frames) - 1], pf).draw(1)
    pf.fig.canvas.draw()
    texts = [t for t in pf.fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    assert texts
    assert all(t.get_fontsize() <= _style.FONT_SIZE for t in texts)
    assert not any(t.get_fontweight() in ("bold", 700) for t in texts)
    assert pf.fig._suptitle is None
    lax = pf.loss_ax
    assert lax.get_xlabel() == "Evaluations"
    assert lax.get_ylabel() == "Mean edge loss (dB)"
    assert pf._cbar.ax.get_ylabel() == "Crossings per edge"
    assert all(sp.get_visible() and sp.get_linewidth() == _style.LINE_WIDTH
               for sp in lax.spines.values())
    assert lax.xaxis.get_tick_params(which="major")["direction"] == "in"
    assert pf.status_text.get_text().startswith("Final result, evaluation ")
    plt.close("all")


def test_style_axes_log_rules():
    assert _style.font_stack()[0] == _style.FONT_FAMILY == "Arial"
    assert "Liberation Sans" in _style.font_stack()  # metric-compatible fallback
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    _style.style_axes(ax)
    fig.canvas.draw()
    minor = ax.xaxis.get_minorticklocs()
    assert {2, 3, 9, 20, 90} <= {int(round(v)) for v in minor}
    assert ax.xaxis.get_tick_params(which="minor")["direction"] == "in"
    assert any(line.get_visible() for line in ax.get_xgridlines())  # log ⇒ grid on
    assert all(sp.get_visible() for sp in ax.spines.values())
    lin_fig, lin_ax = plt.subplots()
    _style.style_axes(lin_ax)
    lin_fig.canvas.draw()
    assert not any(line.get_visible() for line in lin_ax.get_xgridlines())
    plt.close("all")


def test_status_lines_are_sentence_case_with_units(tmp_path):
    res = _record(tmp_path, progress_kwargs={"formats": []})
    data = ProgressData.from_json(res["progress"]["history"])
    first, last = data.frames[0], data.frames[-1]
    kw = dict(total=len(data.frames), evaluations=data.evaluations, initial_loss=first.loss)
    assert frame_status(first, index=0, **kw).startswith("Improvement 1 of ")
    assert "dB (initial)" in frame_status(first, index=0, **kw)
    final = frame_status(last, index=len(data.frames) - 1, **kw)
    assert final.startswith("Final result, evaluation ") and "dB (\u2212" in final
    state = LiveState(frame=last, best_x=np.array([1.0, 2.0]),
                      best_y=np.array([first.loss, last.loss]), envelope=None,
                      evaluations=1234, elapsed_ms=1500, improvements=2, done=True)
    assert live_status(state).startswith("Finished, 1,234 evaluations, best loss ")


def test_edge_polylines_bow_same_side_pairs_inward():
    g = make_graph(k=12)
    edges = list(g.G.edges())
    polylines = edge_polylines(g.positions, edges)
    arcs = {e for e, p in zip(edges, polylines, strict=True) if len(p) > 2}
    # K_12 on a square with 3 nodes per side: one skip-one pair per side.
    assert arcs == {(0, 2), (3, 5), (6, 8), (9, 11)}
    for e, p in zip(edges, polylines, strict=True):
        if e in arcs:
            mid = p[len(p) // 2]
            assert 0.0 < mid[0] < 1.0 and 0.0 < mid[1] < 1.0  # inside the square


# Layouts built side by side, with the node count of each straight stretch
# of the boundary in index order (corners carry no node).
_LAYOUTS = {
    "square": (distribute_nodes("square", nodes_per_side=4), [4, 4, 4, 4]),
    "rectangle": (distribute_nodes("rectangle", nodes_on_length=4, nodes_on_width=2),
                  [4, 2, 4, 2]),
    "triangle": (distribute_nodes("triangle", triangle_nodes_per_side=[4, 3, 2]), [4, 3, 2]),
    "pentagon": (distribute_nodes("polygon", polygon_n_sides=5, polygon_nodes_per_side=3),
                 [3] * 5),
    "partial_rectangle": (distribute_nodes("partial_rectangle",
                                           side_counts={"top": 4, "right": 2, "bottom": 3}),
                          [4, 2, 3]),
    "circle": (distribute_nodes("circle", k=12), [1] * 12),
}


@pytest.mark.parametrize("name", list(_LAYOUTS))
def test_edge_bends_follow_straight_boundary_stretches(name):
    """Any layout: a chord bows exactly when other nodes sit between its
    ends on one straight stretch of the boundary, by CURVE_FACTOR x index
    gap / nodes on the stretch, toward the interior whichever way the edge
    is oriented."""
    positions, stretches = _LAYOUTS[name]
    pairs = list(itertools.combinations(range(len(positions)), 2))
    expected, start = {}, 0
    for n in stretches:
        for u, v in itertools.combinations(range(start, start + n), 2):
            if v - u >= 2:
                expected[(u, v)] = CURVE_FACTOR * (v - u) / n
        start += n
    bends = _edge_bends(positions, pairs)
    got = {e: abs(r) for e, r in zip(pairs, bends, strict=True) if r != 0.0}
    assert got.keys() == expected.keys()
    assert all(math.isclose(got[e], expected[e]) for e in expected)

    centroid = np.mean(list(positions.values()), axis=0)
    reverse = _edge_bends(positions, [(v, u) for u, v in pairs])
    for (u, v), rad, rad_rev in zip(pairs, bends, reverse, strict=True):
        p0, p2 = np.asarray(positions[u]), np.asarray(positions[v])
        mid, d = (p0 + p2) / 2, p2 - p0
        ctrl = mid + rad * np.array([d[1], -d[0]])
        ctrl_rev = mid - rad_rev * np.array([d[1], -d[0]])  # (dy, -dx) flips with the edge
        assert np.allclose(ctrl, ctrl_rev)
        if rad:
            assert (ctrl - mid) @ (centroid - mid) > 0  # bows inward

    # Rounded coordinates (e.g. typed into a positions JSON) still count
    # as on the stretch: the tolerance is a fraction of the layout span.
    rounded = {n: (round(x, 4), round(y, 4)) for n, (x, y) in positions.items()}
    assert [r != 0.0 for r in _edge_bends(rounded, pairs)] == [r != 0.0 for r in bends]


def _one_frame_data(positions, L=2) -> ProgressData:
    edges = list(itertools.combinations(range(len(positions)), 2))
    frame = ProgressFrame(eval_index=0, wall_ms=0, loss=1.0, final=True,
                          layers=np.arange(len(edges)) % L,
                          crossings=np.zeros(len(edges), dtype=int))
    return ProgressData(k=len(positions), L=L, edge_coupler_layer=0, perimeter_layer=0,
                        positions=positions, edges=edges, frames=[frame])


@pytest.mark.parametrize("positions, aspect", [
    (distribute_nodes("square", nodes_per_side=3), 1.0),
    # 5 x 3 bounding box plus a 9 % margin of the span (5) on every side
    (distribute_nodes("rectangle", nodes_on_length=4, nodes_on_width=2), 3.9 / 5.9),
    # a 12 x 1 strip is clamped to the widest allowed panel
    (distribute_nodes("rectangle", nodes_on_length=6, nodes_on_width=1, length=12, width=1),
     0.4),
], ids=["square", "rectangle", "strip"])
def test_routing_panels_take_the_layout_shape(positions, aspect):
    pf = ProgressFigure(_one_frame_data(positions), dpi=40)
    fig_w, fig_h = pf.fig.get_size_inches()
    for ax in pf.layer_axes:
        box = ax.get_position(original=True)
        assert math.isclose(box.height * fig_h / (box.width * fig_w), aspect, rel_tol=1e-9)
    cbar = pf.cax.get_position(original=True)
    panel = pf.layer_axes[0].get_position(original=True)
    assert panel.y0 < cbar.y0 and cbar.y1 < panel.y1  # colorbar within the panel row
    plt.close("all")


# ---------------------------------------------------------------------------
# Mode "live" (headless: the suite runs on the Agg backend)
# ---------------------------------------------------------------------------
def test_live_mode_rewrites_image_headless(tmp_path):
    res = run_optimization(output_dir=str(tmp_path), save_json=False, progress="live",
                           progress_kwargs={**_FAST_PROGRESS, "interval": 0.0}, **_FAST)
    path = res["progress"]["live_image"]
    assert os.path.basename(path) == "optimization_live.png"
    with Image.open(path) as im:
        assert im.size[0] > 100 and im.size[1] > 100
    assert not os.path.exists(path + ".tmp.png")


def test_live_view_error_never_aborts_the_run(tmp_path, monkeypatch):
    def boom(self, state):
        raise RuntimeError("display went away")

    monkeypatch.setattr(LiveProgressView, "update", boom)
    with pytest.warns(UserWarning, match="live progress view disabled"):
        res = run_optimization(output_dir=str(tmp_path), save_json=False, progress="live",
                               progress_kwargs=_FAST_PROGRESS, **_FAST)
    assert res["loss"] > 0


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
def test_progress_argument_validation(tmp_path):
    assert normalize_progress_mode(None) == "off"
    assert normalize_progress_mode("RECORD") == "record"
    with pytest.raises(ValueError, match="progress must be one of"):
        run_optimization(output_dir=str(tmp_path), progress="movie", **_FAST)
    with pytest.raises(TypeError, match="unknown progress_kwargs"):
        run_optimization(output_dir=str(tmp_path), progress="record",
                         progress_kwargs={"framerate": 3}, **_FAST)
    with pytest.raises(ValueError, match="unknown replay format"):
        ProgressOptions(formats=["mp4"])
    assert ProgressOptions(formats="gif, html").formats == ("gif", "html")
    with pytest.raises(ValueError, match="fixed_layers"):
        run_optimization(output_dir=str(tmp_path), progress="live",
                         fixed_layers=[0] * 28, **_FAST)
    with pytest.raises(ValueError, match="output_dir"):
        run_optimization(output_dir=None, save_json=False, progress="record", **_FAST)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_optimize_record_then_replay(tmp_path, capsys):
    cli_main([
        "optimize", "--k", "8", "--maxiter", "5", "--output-dir", str(tmp_path / "run"),
        "--no-plot", "--no-loss-analysis", "--progress", "record",
        "--progress-kwargs", json.dumps({"formats": ["html"], "dpi": 40}),
    ])
    out = capsys.readouterr().out
    history = next(line.split(": ", 1)[1] for line in out.splitlines()
                   if line.startswith("[progress] history: "))
    assert os.path.exists(history)
    assert os.path.exists(os.path.join(os.path.dirname(history), "optimization_progress.html"))

    cli_main(["replay", "--history", history, "--out-dir", str(tmp_path / "replay"),
              "--formats", "gif", "--max-frames", "4", "--dpi", "40"])
    gif = tmp_path / "replay" / "optimization_progress.gif"
    with Image.open(gif) as im:
        assert im.n_frames <= 4


# An irregular 8-port outline, numbered counter-clockwise along the boundary.
_OUTLINE = {0: (0.0, 0.0), 1: (1.2, 0.0), 2: (3.0, 0.0), 3: (4.0, 1.1),
            4: (4.0, 2.5), 5: (2.6, 3.4), 6: (0.9, 3.4), 7: (0.0, 1.7)}


def test_cli_optimize_positions_json(tmp_path, capsys):
    """Arbitrary coordinates from a file reach the saved routing and the
    recorded history; k comes from the file."""
    path = tmp_path / "outline.json"
    path.write_text(json.dumps({str(n): list(p) for n, p in _OUTLINE.items()}))
    cli_main([
        "optimize", "--positions-json", str(path), "--maxiter", "5",
        "--output-dir", str(tmp_path / "run"), "--no-plot", "--no-loss-analysis",
        "--progress", "record", "--progress-kwargs", json.dumps({"formats": []}),
    ])
    captured = capsys.readouterr()
    assert "crosses itself" not in captured.err
    history = next(line.split(": ", 1)[1] for line in captured.out.splitlines()
                   if line.startswith("[progress] history: "))
    data = ProgressData.from_json(history)
    assert data.k == len(_OUTLINE) and data.positions == _OUTLINE
    saved = next(line.split(": ", 1)[1] for line in captured.out.splitlines()
                 if line.startswith("JSON: "))
    with open(saved) as f:
        assert json.load(f)["positions"] == {str(n): list(p) for n, p in _OUTLINE.items()}


def test_cli_positions_json_checks_k_and_node_order(tmp_path, capsys):
    path = tmp_path / "outline.json"
    path.write_text(json.dumps({str(n): list(p) for n, p in _OUTLINE.items()}))
    with pytest.raises(SystemExit, match="--k 12 does not match the 8 nodes"):
        cli_main(["optimize", "--positions-json", str(path), "--k", "12"])
    path.write_text(json.dumps({"0": [0, 0], "1": [1, 0], "3": [1, 1]}))
    with pytest.raises(SystemExit, match="contiguous set"):
        cli_main(["optimize", "--positions-json", str(path)])
    # Swap two nodes: the index walk now crosses itself -> warned, still runs.
    swapped = dict(_OUTLINE)
    swapped[1], swapped[5] = _OUTLINE[5], _OUTLINE[1]
    path.write_text(json.dumps({str(n): list(p) for n, p in swapped.items()}))
    cli_main(["optimize", "--positions-json", str(path), "--maxiter", "1",
              "--output-dir", str(tmp_path / "run"), "--no-plot", "--no-loss-analysis",
              "--no-json"])
    assert "crosses itself" in capsys.readouterr().err
