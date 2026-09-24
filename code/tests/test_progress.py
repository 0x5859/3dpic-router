"""Optimization-process visualization: ``run_optimization(progress=...)``.

Covers the three modes (``off`` / ``live`` / ``record``): the tracker
must never perturb the optimizer, the recorded history must end on the
exact routing written to ``subgraphsdata.json``, the history JSON contract
(schema + cross-field checks), the replay renderers, the live view's
headless fallback, and the CLI entry points.
"""
from __future__ import annotations

import json
import math
import os

import jsonschema
import numpy as np
import pytest
from PIL import Image, ImageSequence
from routing_py_rebuild.__main__ import main as cli_main
from routing_py_rebuild.api import make_graph, run_optimization
from routing_py_rebuild.plotting.progress_figure import (
    HISTORY_FILENAME,
    ProgressData,
    edge_polylines,
)
from routing_py_rebuild.plotting.progress_live import LiveProgressView, live_display_mode
from routing_py_rebuild.plotting.progress_replay import (
    render_progress_replay,
    select_frames,
    show_progress_replay,
)
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
    res = _record(tmp_path)
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
    assert payload["meta"][-1]["final"] is True
    assert payload["meta"][-1]["eval"] == data.frames[-1].eval_index + 1


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
