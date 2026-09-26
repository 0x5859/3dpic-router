"""Replay a recorded optimization (``optimization_history.json``).

* :func:`render_progress_replay` — every (or, above ``max_frames``, an
  evenly spaced subset of the) recorded improvements, rendered once each
  and written as an animated GIF and/or a self-contained HTML player
  (play / pause / step / scrub / speed, keyboard shortcuts; opens in any
  browser, no server or extra files). ``"pdf"`` / ``"png"`` add a still of
  the final state — final routing plus the complete loss curve — as a
  vector PDF (editable text) and a 450 ppi PNG for papers and slides.
* :func:`show_progress_replay` — the same frames in an interactive
  matplotlib window with a slider and a play button (needs a GUI
  backend).

The last frame is always the final routing — the one written to
``subgraphsdata.json`` — and holds longer in the GIF.
"""
from __future__ import annotations

import base64
import html
import io
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from ._colormap import NODE_FILL_COLOR
from ._style import FONT_SIZE, GRID_COLOR
from .progress_figure import (
    CONTEXT_COLOR,
    ENVELOPE_ALPHA,
    ENVELOPE_COLOR,
    FRAME_DPI,
    HISTORY_FILENAME,
    LIVE_WINDOW_DPI,
    MARKER_COLOR,
    MOVED_OUTLINE,
    MUTED,
    STILL_DPI,
    CurveView,
    ProgressData,
    ProgressFigure,
    _y_limits,
    frame_status,
)

REPLAY_FORMATS = ("gif", "html", "pdf", "png")
REPLAY_BASENAME = "optimization_progress"
_FIRST_HOLD_MS = 800
_LAST_HOLD_MS = 2500
_CM = 1 / 2.54


def load_progress_data(data: ProgressData | str | os.PathLike) -> ProgressData:
    """Return ``data`` as :class:`ProgressData`, loading it from an
    ``optimization_history.json`` path, the run directory holding one, or
    an ``optimize --output-dir`` whose single run subdirectory holds one."""
    if isinstance(data, ProgressData):
        return data
    path = Path(data)
    if path.is_dir() and not (path / HISTORY_FILENAME).exists():
        found = sorted(path.glob(f"*/{HISTORY_FILENAME}"))
        if len(found) > 1:
            raise ValueError(
                f"{path} holds several recorded runs; pass one of: "
                + ", ".join(str(f.parent) for f in found)
            )
        if found:
            path = found[0].parent
    if path.is_dir():
        path = path / HISTORY_FILENAME
    return ProgressData.from_json(path)


def select_frames(n: int, max_frames: int | None) -> list[int]:
    """Frame indices to animate: all of them, or ``max_frames`` evenly
    spaced ones that always include the first and the last (final)."""
    if n <= 0:
        return []
    if max_frames is None or n <= max_frames:
        return list(range(n))
    idx = np.unique(np.round(np.linspace(0, n - 1, max(2, int(max_frames)))).astype(int))
    return [int(i) for i in idx]


def auto_fps(n_frames: int) -> float:
    """4 fps for short histories, rising to 15 fps so that even the default
    200-frame cap plays in about 13 s."""
    return float(min(15.0, max(4.0, n_frames / 12.0)))


class _Replayer:
    """Draws frame ``pos`` of the selected subset onto a ProgressFigure."""

    def __init__(self, data: ProgressData, idx: list[int], pf: ProgressFigure):
        self.data, self.idx, self.pf = data, idx, pf
        pf.set_color_limit(data.max_crossings())
        self.best_x, self.best_y = data.best_curve()
        self.y_range = _y_limits(self.best_y)
        self.evaluations = max(data.evaluations, int(self.best_x[-1]))
        self.initial = float(self.best_y[0])
        self.improvements = (
            len(data.frames) if data.improvements is None else data.improvements
        )

    def curve(self, i: int) -> CurveView:
        frame = self.data.frames[i]
        return CurveView(
            best_x=self.best_x[: i + 1],
            best_y=self.best_y[: i + 1],
            x_now=self.evaluations if frame.final else frame.eval_index + 1,
            marker=(frame.eval_index + 1, frame.loss),
            envelope=self.data.envelope,
            x_max=self.evaluations,
            y_range=self.y_range,
        )

    def draw(self, pos: int) -> None:
        i = self.idx[pos]
        frame = self.data.frames[i]
        prev = self.data.frames[self.idx[pos - 1]] if pos > 0 else None
        status = frame_status(
            frame, index=i, total=self.improvements,
            evaluations=self.evaluations, initial_loss=self.initial,
        )
        self.pf.draw(frame, prev, self.curve(i), status)

    def meta(self, pos: int) -> dict:
        f = self.data.frames[self.idx[pos]]
        return {"eval": f.eval_index + 1, "loss": f.loss, "ms": f.wall_ms, "final": f.final}


def _rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _key_colors() -> list[tuple[int, int, int]]:
    """Colors of thin strokes and text that a population-based quantizer
    would merge away (a 0.5 pt line covers few pixels), plus their
    anti-aliasing blends toward the backgrounds they are drawn on."""
    white = np.ones(3)
    black = np.zeros(3)
    envelope = white + ENVELOPE_ALPHA * (np.array(to_rgb(ENVELOPE_COLOR)) - white)
    colors = [white, envelope, np.array(to_rgb(MUTED)), np.array(to_rgb(MARKER_COLOR)),
              np.array(to_rgb(MOVED_OUTLINE)), np.array(to_rgb(NODE_FILL_COLOR)),
              white + 0.5 * (np.array(to_rgb(GRID_COLOR)) - white)]
    for base in (np.array(to_rgb(CONTEXT_COLOR)), black):
        for t in (1.0, 0.75, 0.5, 0.3):  # 0.5 pt lines over white and over the band
            colors += [white + t * (base - white), envelope + t * (base - envelope)]
    rgb = {tuple(int(round(255 * c)) for c in col) for col in colors}
    return sorted(rgb)


def _gif_palette(samples: list[np.ndarray]):
    """One 256-color palette for every GIF frame (no flicker between
    frames): median cut over a few sample frames, with :func:`_key_colors`
    reserved so thin lines keep their true color."""
    from PIL import Image

    keys = _key_colors()
    mosaic = np.concatenate([s[::2, ::2] for s in samples], axis=0)
    adaptive = Image.fromarray(mosaic).quantize(
        colors=256 - len(keys), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette = adaptive.getpalette()[: 3 * (256 - len(keys))]
    palette += [c for rgb in keys for c in rgb]
    image = Image.new("P", (1, 1))
    image.putpalette(palette)
    return image


def _encode_frame(rgb: np.ndarray) -> tuple[str, str]:
    """``(mime, base64)`` for the HTML player: lossless WebP (exact colors
    at about the size of a 256-color PNG), or RGB PNG without WebP."""
    from PIL import Image, features

    buf = io.BytesIO()
    if features.check("webp"):
        Image.fromarray(rgb).save(buf, format="WEBP", lossless=True, method=4)
        mime = "image/webp"
    else:
        Image.fromarray(rgb).save(buf, format="PNG")
        mime = "image/png"
    return mime, base64.b64encode(buf.getvalue()).decode("ascii")


@contextmanager
def _quiet_font_subsetting():
    """Embedding TrueType text makes fontTools log every subsetting step at
    INFO, which leaks to stderr when a dependency (e.g. pyswarms) has set
    the root logger to INFO."""
    logger = logging.getLogger("fontTools")
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)


def _write_stills(data: ProgressData, out_dir: str | os.PathLike,
                  formats: tuple[str, ...]) -> dict[str, str]:
    """Final routing + complete loss curve as PDF (vector, TrueType text)
    and / or a ``STILL_DPI`` PNG. No status line or run label — for a
    paper the caption carries those."""
    pf = ProgressFigure(data, annotate=False)
    last = len(data.frames) - 1
    rp = _Replayer(data, [last], pf)
    pf.draw(data.frames[last], None, rp.curve(last), status="")
    out: dict[str, str] = {}
    if "pdf" in formats:
        path = os.path.join(out_dir, f"{REPLAY_BASENAME}.pdf")
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}), _quiet_font_subsetting():
            pf.fig.savefig(path)
        out["pdf"] = path
    if "png" in formats:
        path = os.path.join(out_dir, f"{REPLAY_BASENAME}.png")
        pf.fig.savefig(path, dpi=STILL_DPI)
        out["png"] = path
    return out


def render_progress_replay(
    data: ProgressData | str | os.PathLike,
    *,
    out_dir: str | os.PathLike | None = None,
    formats: tuple[str, ...] | list[str] = REPLAY_FORMATS,
    fps: float | None = None,
    max_frames: int | None = 200,
    dpi: float | None = None,
) -> dict[str, str]:
    """Render the recorded process to ``optimization_progress.*``.

    Parameters
    ----------
    data : ProgressData | path
        The history, or a path to ``optimization_history.json`` (or its
        directory).
    out_dir : path | None
        Output directory; defaults to the history file's directory.
    formats : iterable of {"gif", "html", "pdf", "png"}
        ``gif`` / ``html``: the animation; ``pdf`` / ``png``: a still of
        the final state (PNG at ``STILL_DPI``).
    fps : float | None
        Frames per second; None = :func:`auto_fps`.
    max_frames : int | None
        Cap on animation frames (evenly spaced, first and final kept).
    dpi : float | None
        Animation frame resolution; None = ``FRAME_DPI``.

    Returns
    -------
    dict
        ``{format: path}`` for the formats written.
    """
    from PIL import Image  # matplotlib dependency

    data = load_progress_data(data)
    formats = tuple(str(f).lower() for f in formats)
    unknown = sorted(set(formats) - set(REPLAY_FORMATS))
    if unknown:
        raise ValueError(f"unknown replay format(s) {unknown}; choose from {REPLAY_FORMATS}.")
    if not formats:
        return {}
    if not data.frames:
        raise ValueError("history has no frames to replay.")
    if out_dir is None:
        if data.source_path is None:
            raise ValueError("render_progress_replay: pass out_dir= for in-memory data.")
        out_dir = os.path.dirname(os.path.abspath(data.source_path))
    os.makedirs(out_dir, exist_ok=True)

    out: dict[str, str] = {}
    if "gif" in formats or "html" in formats:
        idx = select_frames(len(data.frames), max_frames)
        fps = float(fps) if fps else auto_fps(len(idx))
        pf = ProgressFigure(data, dpi=dpi or FRAME_DPI, context=True)
        rp = _Replayer(data, idx, pf)

        def rendered():
            for pos in range(len(idx)):
                rp.draw(pos)
                yield _rgb(pf.fig)

        # Frames are rendered once and streamed: the GIF writer pulls them
        # through a generator and the HTML keeps only compressed bytes.
        html_frames: list[tuple[str, str]] = []

        def encoded(palette):
            for rgb in rendered():
                if "html" in formats:
                    html_frames.append(_encode_frame(rgb))
                if palette is not None:
                    yield Image.fromarray(rgb).quantize(palette=palette, dither=Image.Dither.NONE)

        if "gif" in formats:
            samples = []
            for pos in sorted({0, len(idx) // 3, 2 * len(idx) // 3, len(idx) - 1}):
                rp.draw(pos)
                samples.append(_rgb(pf.fig))
            frames = encoded(_gif_palette(samples))
            step_ms = int(round(1000.0 / fps))
            durations = [step_ms] * len(idx)
            durations[0] = max(step_ms, _FIRST_HOLD_MS)
            durations[-1] = max(step_ms, _LAST_HOLD_MS)
            gif_path = os.path.join(out_dir, f"{REPLAY_BASENAME}.gif")
            next(frames).save(gif_path, save_all=True, append_images=frames,
                              duration=durations, loop=0)
            out["gif"] = gif_path
        else:
            for _ in encoded(None):
                pass
        if "html" in formats:
            html_path = os.path.join(out_dir, f"{REPLAY_BASENAME}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(replay_html(html_frames, [rp.meta(p) for p in range(len(idx))],
                                    run_label=data.run_label(), fps=fps))
            out["html"] = html_path
    out.update(_write_stills(data, out_dir, formats))
    return out


def replay_html(frames: list[tuple[str, str]], meta: list[dict], *, run_label: str,
                fps: float) -> str:
    """Self-contained HTML player for ``(mime, base64)`` frames + per-frame ``meta``."""
    payload = json.dumps({
        "frames": [f"data:{mime};base64,{b64}" for mime, b64 in frames],
        "meta": meta,
        "fps": fps,
    }).replace("</", "<\\/")
    return (_HTML_TEMPLATE.replace("__RUN__", html.escape(run_label))
            .replace("__PAYLOAD__", payload))


def show_progress_replay(
    data: ProgressData | str | os.PathLike,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    dpi: float | None = None,
    autoplay: bool = True,
) -> None:
    """Interactive replay window: slider + play/pause, ←/→/space keys.

    Blocks until the window is closed. Needs an interactive matplotlib
    backend; with a non-interactive one (e.g. Agg) it prints a note and
    returns — use :func:`render_progress_replay` files instead.
    """
    from matplotlib.widgets import Button, Slider

    from .progress_live import live_display_mode

    if live_display_mode() != "window":
        print(f"[progress] no GUI backend ({plt.get_backend()}); open the "
              "optimization_progress.gif / .html replay files instead.")
        return
    data = load_progress_data(data)
    if not data.frames:
        raise ValueError("history has no frames to replay.")
    idx = select_frames(len(data.frames), max_frames)
    fps = float(fps) if fps else auto_fps(len(idx))
    pf = ProgressFigure(data, dpi=dpi or LIVE_WINDOW_DPI, pyplot=True, context=True,
                        footer_cm=1.1)
    fig = pf.fig
    manager = fig.canvas.manager
    if manager is not None:
        manager.set_window_title(f"3D-PIC router — optimization replay ({data.run_label()})")
    rp = _Replayer(data, idx, pf)
    last = len(idx) - 1

    w_cm, h_cm = (v / _CM for v in fig.get_size_inches())
    btn_ax = fig.add_axes((0.3 / w_cm, 0.25 / h_cm, 1.6 / w_cm, 0.6 / h_cm))
    sld_ax = fig.add_axes((3.0 / w_cm, 0.25 / h_cm, 1 - 4.4 / w_cm, 0.6 / h_cm))
    button = Button(btn_ax, "Pause" if autoplay else "Play")
    button.label.set_fontsize(FONT_SIZE)
    slider = Slider(sld_ax, "Frame", 1, max(last + 1, 2), valinit=1, valstep=1, valfmt="%d")
    slider.label.set_fontsize(FONT_SIZE)
    slider.valtext.set_fontsize(FONT_SIZE)
    state = {"playing": autoplay}
    timer = fig.canvas.new_timer(interval=int(round(1000.0 / fps)))

    def on_slide(val):
        rp.draw(min(int(val) - 1, last))
        fig.canvas.draw_idle()

    def set_playing(flag: bool) -> None:
        state["playing"] = flag
        button.label.set_text("Pause" if flag else "Play")
        (timer.start if flag else timer.stop)()
        fig.canvas.draw_idle()

    def on_tick():
        pos = int(slider.val) - 1
        if pos >= last:
            set_playing(False)
        else:
            slider.set_val(pos + 2)

    def on_click(_event):
        if not state["playing"] and int(slider.val) - 1 >= last:
            slider.set_val(1)  # replay from the start
        set_playing(not state["playing"])

    def on_key(event):
        pos = int(slider.val) - 1
        if event.key == " ":
            on_click(None)
        elif event.key in ("right", "left"):
            set_playing(False)
            slider.set_val(min(last, max(0, pos + (1 if event.key == "right" else -1))) + 1)
        elif event.key in ("home", "end"):
            set_playing(False)
            slider.set_val(1 if event.key == "home" else last + 1)

    slider.on_changed(on_slide)
    button.on_clicked(on_click)
    timer.add_callback(on_tick)
    fig.canvas.mpl_connect("key_press_event", on_key)
    rp.draw(0)
    fig._progress_widgets = (slider, button, timer)  # keep references alive
    if autoplay and last > 0:
        timer.start()
    plt.show()


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Optimization replay — __RUN__</title>
<style>
  :root { --bg: #f6f7f9; --fg: #1f2328; --muted: #5f6670; --line: #d0d4da; --accent: #9b2629; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #16181c; --fg: #e6e8eb; --muted: #9aa1ab; --line: #3a3f47; --accent: #e07a7c; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.45 Arial, Helvetica, "Liberation Sans", Arimo, sans-serif; }
  main { max-width: 1500px; margin: 0 auto; padding: 16px; }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px;
           margin: 0 0 12px; }
  h1 { font-size: 16px; font-weight: 600; margin: 0; }
  .run { color: var(--muted); }
  .stage { background: #fff; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .stage img { display: block; margin: 0 auto; max-width: 100%; height: auto;
               max-height: calc(100vh - 170px); }
  .bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 12px; }
  button, select { font: inherit; color: var(--fg); background: transparent;
                   border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px;
                   min-height: 32px; cursor: pointer; }
  button:hover, select:hover { border-color: var(--muted); }
  button:focus-visible, select:focus-visible, input:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px; }
  #play { min-width: 76px; }
  input[type=range] { flex: 1 1 260px; accent-color: var(--accent); min-height: 32px; }
  label { color: var(--muted); display: inline-flex; align-items: center; gap: 6px; }
  #info { color: var(--muted); margin-top: 8px; font-variant-numeric: tabular-nums; }
  .keys { color: var(--muted); font-size: 12px; margin-top: 4px; }
</style>
</head>
<body>
<main>
  <header><h1>Optimization replay</h1><span class="run">__RUN__</span></header>
  <div class="stage"><img id="frame" alt="Routing at the selected optimization step"></div>
  <div class="bar">
    <button id="first" title="First frame (Home)" aria-label="First frame">&#x23EE;</button>
    <button id="prev" title="Previous frame (&#x2190;)" aria-label="Previous">&#x25C0;</button>
    <button id="play" title="Play / pause (Space)">Play</button>
    <button id="next" title="Next frame (&#x2192;)" aria-label="Next">&#x25B6;</button>
    <button id="last" title="Final result (End)" aria-label="Final result">&#x23ED;</button>
    <input id="slider" type="range" min="0" value="0" aria-label="Frame">
    <label>Speed <select id="speed">
      <option value="0.25">0.25&times;</option><option value="0.5">0.5&times;</option>
      <option value="1" selected>1&times;</option><option value="2">2&times;</option>
      <option value="4">4&times;</option></select></label>
    <label><input id="loop" type="checkbox"> Loop</label>
  </div>
  <div id="info"></div>
  <div class="keys">Space: play / pause &middot; &#x2190; &#x2192;: step &middot;
    Home / End: first / final</div>
</main>
<script>
const DATA = __PAYLOAD__;
const N = DATA.frames.length;
const img = document.getElementById("frame");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("play");
const info = document.getElementById("info");
const speed = document.getElementById("speed");
const loop = document.getElementById("loop");
slider.max = N - 1;
let pos = 0, timer = null;
const fmt = new Intl.NumberFormat("en-US");

function show(i) {
  pos = Math.max(0, Math.min(N - 1, i));
  img.src = DATA.frames[pos];
  slider.value = pos;
  const m = DATA.meta[pos];
  info.textContent = `Frame ${pos + 1} of ${N}` + (m.final ? " (final result)" : "") +
    `, evaluation ${fmt.format(m.eval)}, loss ${m.loss.toPrecision(5)} dB` +
    `, ${(m.ms / 1000).toFixed(2)} s`;
}
function delay() {
  const base = 1000 / (DATA.fps * parseFloat(speed.value));
  return pos === N - 1 ? Math.max(base, 1500) : base;
}
function tick() {
  if (pos < N - 1) { show(pos + 1); timer = setTimeout(tick, delay()); }
  else if (loop.checked) { show(0); timer = setTimeout(tick, delay()); }
  else { stop(); }
}
function play() {
  if (pos >= N - 1) show(0);
  playBtn.textContent = "Pause";
  timer = setTimeout(tick, delay());
}
function stop() { clearTimeout(timer); timer = null; playBtn.textContent = "Play"; }
function toggle() { timer ? stop() : play(); }

playBtn.onclick = toggle;
document.getElementById("first").onclick = () => { stop(); show(0); };
document.getElementById("last").onclick = () => { stop(); show(N - 1); };
document.getElementById("prev").onclick = () => { stop(); show(pos - 1); };
document.getElementById("next").onclick = () => { stop(); show(pos + 1); };
slider.oninput = () => { stop(); show(parseInt(slider.value, 10)); };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); toggle(); }
  else if (e.key === "ArrowRight") { stop(); show(pos + 1); }
  else if (e.key === "ArrowLeft") { stop(); show(pos - 1); }
  else if (e.key === "Home") { stop(); show(0); }
  else if (e.key === "End") { stop(); show(N - 1); }
});
show(0);
</script>
</body>
</html>
"""


__all__ = [
    "REPLAY_BASENAME",
    "REPLAY_FORMATS",
    "auto_fps",
    "load_progress_data",
    "render_progress_replay",
    "replay_html",
    "select_frames",
    "show_progress_replay",
]
