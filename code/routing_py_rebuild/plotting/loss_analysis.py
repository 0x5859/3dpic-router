"""Bar charts of mean / variance / std / range across layers.

Renders four panels (Average / Variance / Std / Range) in a 2×2 grid so
the figure stays compact and the per-panel x-axis labels sit at a
readable size next to the rest of the package's plots.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from ._style import apply_rcparams


def visualize_loss_analysis(plot_data, *, out_path: str | None = None):
    """Plot loss-statistic bars for layers + flattened + complete graph.

    Reads ``plot_data.loss_analysis`` (already computed by
    :meth:`core.SiNInterconnectionGraph.analyze_loss` before JSON write).
    Saves an editable vector ``lossanalysis.pdf`` under
    ``plot_data.filepath`` unless ``out_path`` is given.
    """
    apply_rcparams()
    loss_analysis = plot_data.loss_analysis
    if not loss_analysis:
        print("No loss analysis data available. Re-run the optimizer to regenerate JSON.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)

    panels = [
        ("avg_loss_subgraphs", "avg_flattened_loss_subgraphs", "avg_loss_completegraph",
         "Average Loss", "Avg loss", "#F8C9C2"),
        ("var_loss_subgraphs", "var_flattened_loss_subgraphs", "var_loss_completegraph",
         "Variance of Loss", "Variance", "#BDDDE9"),
        ("std_loss_subgraphs", "std_flattened_loss_subgraphs", "std_loss_completegraph",
         "Standard Deviation of Loss", "Standard Deviation", "#DBDDEF"),
        ("range_loss_subgraphs", "range_flattened_loss_subgraphs", "range_loss_completegraph",
         "Range of Loss", "Range", "#C5E2BB"),
    ]

    for ax, (k_layers, k_flat, k_complete, title, ylabel, color) in zip(axes.flat, panels):
        values = list(loss_analysis[k_layers])
        values.append(loss_analysis[k_flat])
        values.append(loss_analysis[k_complete])

        bars = ax.bar(range(len(values)), values, color=color)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(
            [f"Layer {i}" for i in range(len(values) - 2)] + ["Flattened", "Complete"],
            rotation=45, ha="right",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom")

    if out_path is None:
        base = plot_data.filepath or "."
        out_path = os.path.join(base, "lossanalysis.pdf")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[visualize_loss_analysis] saved to {out_path}")
