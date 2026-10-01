"""Turn a result card into the thing a person actually wanted: a tradeoff table.

Raw per-precision numbers are not the answer to "should I quantize this?". The
answer is *relative*: how much faster, how much smaller, how much worse. So every
view here is anchored to a baseline precision and reports deltas, with the
absolute numbers kept alongside so nothing is hidden.

The verdict line exists because the honest answer is frequently "don't" — INT4 on
a CPU without native 4-bit kernels routinely dequantizes to fp32 inside the GEMM
and comes out slower than the baseline it was supposed to beat. A tool that
prints numbers but will not say that is making the reader do the work.
"""

from __future__ import annotations

from edgellm.card import PrecisionRow, ResultCard

BASELINE = "fp32"


def _fmt(value: float | None, spec: str = ".1f", dash: str = "—") -> str:
    return dash if value is None else format(value, spec)


def _signed(value: float, spec: str = ".1f") -> str:
    """Format with an explicit sign, so a quality *improvement* reads as negative."""
    return format(value, f"+{spec}")


def tradeoff_table(card: ResultCard, *, baseline: str = BASELINE) -> str:
    """A Markdown table of absolute numbers plus deltas against the baseline."""
    by_precision = {r.precision: r for r in card.rows}
    base = by_precision.get(baseline)

    header = (
        "| Precision | Size (MB) | Throughput (tok/s) | Speedup | "
        "Peak RAM (MB) | Perplexity | PPL change |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = []
    for row in card.rows:
        speedup = "baseline" if row.precision == baseline else "—"
        ppl_delta = "baseline" if row.precision == baseline else "—"

        if base and row.precision != baseline:
            if base.tokens_per_second > 0:
                ratio = row.tokens_per_second / base.tokens_per_second
                speedup = f"{ratio:.2f}x"
            if base.perplexity and row.perplexity:
                pct = (row.perplexity / base.perplexity - 1) * 100
                ppl_delta = f"{_signed(pct)}%"

        lines.append(
            f"| {row.precision} | {_fmt(row.size_mb, '.0f')} | "
            f"{_fmt(row.tokens_per_second, '.2f')} | {speedup} | "
            f"{_fmt(row.peak_ram_mb, '.0f')} | {_fmt(row.perplexity, '.2f')} | {ppl_delta} |"
        )
    return header + "\n".join(lines) + "\n"


def unstable_rows(card: ResultCard, *, ratio: float = 0.15) -> list[str]:
    """Precisions whose run-to-run spread is too wide for the number to be trusted."""
    return [
        row.precision
        for row in card.rows
        if row.latency_s_mean > 0 and row.latency_s_std / row.latency_s_mean > ratio
    ]


def verdict(card: ResultCard, *, baseline: str = BASELINE) -> list[str]:
    """Plain-language findings. Says 'not worth it' when the numbers say so."""
    by_precision = {r.precision: r for r in card.rows}
    base = by_precision.get(baseline)
    if base is None:
        return [f"No {baseline} baseline was measured, so nothing can be compared against it."]

    findings: list[str] = []
    candidates = [r for r in card.rows if r.precision != baseline]

    def shrink(row: PrecisionRow) -> float:
        return base.size_mb / row.size_mb if row.size_mb else 0.0

    for row in candidates:
        ratio = row.tokens_per_second / base.tokens_per_second if base.tokens_per_second else 0.0
        size_x = shrink(row)
        bits = [f"{size_x:.1f}x smaller"] if size_x > 1.05 else []

        if ratio >= 1.15:
            bits.append(f"{ratio:.2f}x faster")
        elif ratio <= 0.9:
            bits.append(f"**{1 / ratio:.2f}x slower**")
        else:
            bits.append("about the same speed")

        if base.perplexity and row.perplexity:
            pct = (row.perplexity / base.perplexity - 1) * 100
            if pct > 10:
                bits.append(f"and {pct:.0f}% worse perplexity")
            elif pct > 2:
                bits.append(f"for {pct:.0f}% worse perplexity")
            else:
                bits.append("at essentially no quality cost")

        findings.append(f"- **{row.precision}**: " + ", ".join(bits) + ".")

    base_tps = base.tokens_per_second
    faster = [r for r in candidates if base_tps and r.tokens_per_second > base_tps * 1.15]
    slower = [r for r in candidates if base_tps and r.tokens_per_second < base_tps * 0.9]

    if slower:
        names = ", ".join(r.precision for r in slower)
        verb = "is" if len(slower) == 1 else "are"
        findings.append(
            f"- On this machine **{names} {verb} slower than {baseline}** despite being smaller. "
            "That is the normal outcome when ONNX Runtime has no native kernel for the "
            "quantized format and dequantizes inside the matmul: you pay the unpacking "
            "cost on every token and keep none of the arithmetic savings. Quantize here "
            "to fit in memory, not to go faster."
        )
    if faster:
        best = max(faster, key=lambda r: r.tokens_per_second)
        findings.append(
            f"- Best throughput on this machine is **{best.precision}** at "
            f"{best.tokens_per_second:.1f} tok/s."
        )
    return findings


def console_report(card: ResultCard, errors: dict[str, str] | None = None) -> str:
    """The full human-facing report printed after a run."""
    machine = card.machine
    measured = card.rows[0].measured_runs if card.rows else 0
    out = [
        "",
        f"  model    {card.model_id} @ {card.revision}",
        f"  machine  {machine.cpu} ({machine.arch}), {machine.physical_cores} physical cores, "
        f"{machine.ram_gb:.0f} GB RAM",
        f"  software {machine.os} {machine.os_release} · Python {machine.python} · "
        f"onnxruntime {machine.onnxruntime} · {machine.provider}",
        f"  settings {machine.intra_op_threads} threads · {card.gen_tokens} generated tokens · "
        f"{card.warmup_runs} warmup + {measured} measured runs"
        + (f" · {card.eval_windows} perplexity windows" if card.eval_windows else ""),
        "",
        tradeoff_table(card),
    ]
    findings = verdict(card)
    if findings:
        out += ["What this means on this machine:", ""] + findings + [""]

    if unstable := unstable_rows(card):
        out += [
            "Measurement warning:",
            "",
            f"- {', '.join(unstable)} varied by more than 15% between runs, so these "
            "numbers are noisy. Close other applications and re-run before submitting — "
            "a busy machine makes the comparison between precisions meaningless.",
            "",
        ]
    if errors:
        out += ["Skipped:"] + [f"- {p}: {why}" for p, why in errors.items()] + [""]
    return "\n".join(out)
