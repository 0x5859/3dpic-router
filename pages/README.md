# Routing replay page

`index.html` is the page served on GitHub Pages at <https://0x5859.github.io/3dpic-router/>.
It replays one optimization per boundary layout (k = 12, 2 layers, dual annealing,
200 iterations, seed 5859, default loss model) and shows the crossings in each layer as the
optimizer improves the routing. The browser draws every frame as SVG from run data inlined in
the page, so the file is self-contained; it fetches only its web font. `#<layout>` in the
address opens a layout directly, for example `#notched_chip`.

| File | Role |
|---|---|
| `record_runs.py` | The layouts, and one recorded run per layout (`--progress record`). |
| `build_page.py` | Turns the recorded histories into `index.html`, using the library's geometry, color map and axis ticks. |
| `template.html` | The page: markup, styles and the player script. |
| `index.html` | The built page. Do not edit it by hand. |

## Rebuild

From the repository root (the runs take about half a minute):

```bash
uv run python pages/record_runs.py /tmp/replay-runs
uv run python pages/build_page.py /tmp/replay-runs     # writes pages/index.html
```

## Publish

GitHub Pages serves the root of the `gh-pages` branch, which holds only `index.html` and
`.nojekyll`. To publish a new build:

```bash
git fetch origin gh-pages
git worktree add /tmp/gh-pages gh-pages
cp pages/index.html /tmp/gh-pages/
git -C /tmp/gh-pages commit -am "Update the routing replay page"
git -C /tmp/gh-pages push origin gh-pages
git worktree remove /tmp/gh-pages
```
