"""Aggregate every submitted card into a leaderboard.

The interesting question is not "which machine is fastest" — that is just a list
of who owns the newest CPU. It is **"does quantization actually pay off, and where"**.
So the leaderboard is organised around the per-precision speedup distribution
across machines, with the raw per-machine rows underneath.

Cards flagged as not comparable by :mod:`edgellm.validate` are listed but kept out
of the aggregate statistics, so a run with a different token budget cannot quietly
move the headline numbers.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from edgellm.validate import validate_card

BASELINE = "fp32"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _speedups(rows: list[dict]) -> dict[str, float]:
    by_precision = {r["precision"]: r for r in rows}
    base = by_precision.get(BASELINE)
    if not base or not base.get("tokens_per_second"):
        return {}
    return {
        r["precision"]: r["tokens_per_second"] / base["tokens_per_second"]
        for r in rows
        if r["precision"] != BASELINE and r.get("tokens_per_second")
    }


def _ppl_deltas(rows: list[dict]) -> dict[str, float]:
    by_precision = {r["precision"]: r for r in rows}
    base = by_precision.get(BASELINE)
    if not base or not base.get("perplexity"):
        return {}
    return {
        r["precision"]: (r["perplexity"] / base["perplexity"] - 1) * 100
        for r in rows
        if r["precision"] != BASELINE and r.get("perplexity")
    }


def build_leaderboard(card_paths: list[Path]) -> dict[str, Any]:
    """Load, validate and aggregate cards into a JSON-serializable leaderboard."""
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    for path in sorted(card_paths):
        report = validate_card(path)
        if not report.ok:
            rejected.append({"file": path.name, "reason": "; ".join(report.errors)})
            continue

        card = json.loads(path.read_text())
        machine = card["machine"]
        entries.append(
            {
                "file": path.name,
                "comparable": report.comparable,
                "model_id": card["model_id"],
                "created_utc": card["created_utc"],
                "submitted_by": card.get("submitted_by", ""),
                "notes": card.get("notes", ""),
                "cpu": machine["cpu"],
                "arch": machine["arch"],
                "os": machine["os"],
                "physical_cores": machine["physical_cores"],
                "ram_gb": machine["ram_gb"],
                "provider": machine["provider"],
                "threads": machine["intra_op_threads"],
                "thread_policy": machine.get("thread_policy", "unknown"),
                "onnxruntime": machine["onnxruntime"],
                "rows": card["rows"],
                "speedups": _speedups(card["rows"]),
                "ppl_deltas": _ppl_deltas(card["rows"]),
            }
        )

    aggregate = _aggregate(e for e in entries if e["comparable"])
    return {
        "entries": sorted(entries, key=lambda e: (e["model_id"], e["cpu"])),
        "aggregate": aggregate,
        "rejected": rejected,
        "machine_count": len({e["cpu"] for e in entries}),
        "model_count": len({e["model_id"] for e in entries}),
    }


def _aggregate(entries) -> dict[str, Any]:
    """Per-(model, precision) speedup and quality-cost distribution across machines."""
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for entry in entries:
        for precision, speedup in entry["speedups"].items():
            bucket = buckets.setdefault((entry["model_id"], precision), {"speedup": [], "ppl": []})
            bucket["speedup"].append(speedup)
        for precision, delta in entry["ppl_deltas"].items():
            bucket = buckets.setdefault((entry["model_id"], precision), {"speedup": [], "ppl": []})
            bucket["ppl"].append(delta)

    out = []
    for (model_id, precision), values in sorted(buckets.items()):
        speedups = values["speedup"]
        ppls = values["ppl"]
        out.append(
            {
                "model_id": model_id,
                "precision": precision,
                "machines": len(speedups),
                "speedup_median": round(statistics.median(speedups), 3) if speedups else None,
                "speedup_min": round(min(speedups), 3) if speedups else None,
                "speedup_max": round(max(speedups), 3) if speedups else None,
                "slower_than_baseline": sum(1 for s in speedups if s < 0.95),
                "ppl_delta_median_pct": round(statistics.median(ppls), 2) if ppls else None,
            }
        )
    return {"by_model_precision": out}


def render_leaderboard_markdown(board: dict[str, Any]) -> str:
    """The committed LEADERBOARD.md."""
    lines = [
        "# Leaderboard",
        "",
        "<!-- Generated by `quantcost leaderboard`. Do not edit by hand:",
        "     every number here is derived from a card in results/community/. -->",
        "",
        f"**{_plural(board['machine_count'], 'machine')} · "
        f"{_plural(board['model_count'], 'model')} · "
        f"{_plural(len(board['entries']), 'card')} submitted**",
        "",
        "## Does quantization pay off?",
        "",
        "Speedup is throughput relative to that machine's own fp32 baseline, so the",
        "column compares quantization choices rather than hardware budgets.",
        "",
    ]

    aggregate = board["aggregate"]["by_model_precision"]
    if aggregate:
        lines += [
            "| Model | Precision | Machines | Median speedup | Range | "
            "Slower than fp32 | Median PPL change |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in aggregate:
            span = (
                f"{row['speedup_min']:.2f}x – {row['speedup_max']:.2f}x"
                if row["speedup_min"] is not None
                else "—"
            )
            median = f"{row['speedup_median']:.2f}x" if row["speedup_median"] else "—"
            ppl = (
                f"{row['ppl_delta_median_pct']:+.1f}%"
                if row["ppl_delta_median_pct"] is not None
                else "—"
            )
            slower = f"{row['slower_than_baseline']}/{row['machines']}" if row["machines"] else "—"
            lines.append(
                f"| `{row['model_id']}` | {row['precision']} | {row['machines']} | "
                f"{median} | {span} | {slower} | {ppl} |"
            )
    else:
        lines.append("_No comparable cards yet._")

    lines += ["", "## Per-machine results", ""]
    for entry in board["entries"]:
        flag = "" if entry["comparable"] else " _(not ranked — non-default settings)_"
        credit = f" — submitted by {entry['submitted_by']}" if entry["submitted_by"] else ""
        lines += [
            f"### {entry['cpu']} · {entry['os']} ({entry['arch']}){flag}",
            "",
            f"`{entry['model_id']}` · {entry['physical_cores']} physical cores · "
            f"{entry['ram_gb']:.0f} GB RAM · "
            f"{entry['threads']} threads ({entry['thread_policy']}) · "
            f"onnxruntime {entry['onnxruntime']} · {entry['provider']}{credit}",
            "",
            "| Precision | Size (MB) | tok/s | Speedup | Peak RAM (MB) | Perplexity |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in entry["rows"]:
            speed = entry["speedups"].get(row["precision"])
            speed_text = (
                "baseline" if row["precision"] == BASELINE else (f"{speed:.2f}x" if speed else "—")
            )
            ppl = f"{row['perplexity']:.2f}" if row.get("perplexity") else "—"
            lines.append(
                f"| {row['precision']} | {row['size_mb']:.0f} | {row['tokens_per_second']:.2f} | "
                f"{speed_text} | {row['peak_ram_mb']:.0f} | {ppl} |"
            )
        if entry["notes"]:
            lines += ["", f"> {entry['notes']}"]
        lines.append("")

    if board["rejected"]:
        lines += ["## Rejected cards", ""]
        for item in board["rejected"]:
            lines.append(f"- `{item['file']}`: {item['reason']}")
        lines.append("")

    return "\n".join(lines)
