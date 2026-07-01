"""Turn the benchmark JSON into charts (and, in Phase 6, the full report)."""

from __future__ import annotations

import json
from pathlib import Path


def _label(row: dict) -> str:
    return f"{row['backend']}\n{row['precision']}/{row['device']}"


def render_charts(results_json: Path, out_path: Path) -> Path:
    """Render size / throughput / perplexity bar charts to a single PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    rows = json.loads(results_json.read_text())
    labels = [_label(r) for r in rows]

    panels = [
        ("On-disk size (MB)", [r.get("size_mb") for r in rows], "#4C72B0"),
        ("Throughput (tok/s)", [r.get("tokens_per_second") for r in rows], "#55A868"),
        ("Perplexity (lower=better)", [r.get("perplexity") for r in rows], "#C44E52"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, values, color) in zip(axes, panels, strict=False):
        xs = list(range(len(labels)))
        vals = [v if v is not None else 0 for v in values]
        bars = ax.bar(xs, vals, color=color)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        for bar, original in zip(bars, values, strict=False):
            text = "n/a" if original is None else f"{original:g}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("EdgeLLM benchmark (real runs)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
