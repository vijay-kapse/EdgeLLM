"""Validate a submitted result card.

What this can and cannot do, stated plainly, because a leaderboard that overclaims
its own rigour is worse than one that is honest about being trust-based:

**It can** reject cards that are malformed, internally inconsistent, scored
against a different corpus, or run with settings that make them incomparable to
everyone else's — and it can flag numbers that are physically implausible.

**It cannot** prove a number came from real hardware. Nothing short of attested
execution could, and this is a benchmark run on strangers' laptops. The defence
is that the inputs are pinned (model revision, artifact bytes, corpus hash,
prompt hash, token budget), so a fabricated card has to be *internally
consistent* across every one of those to pass — and a reviewer can re-run the
exact configuration from the card itself. Cards are reviewed, not trusted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from edgellm.card import CARD_SCHEMA_VERSION
from edgellm.eval_lite import EVAL_CORPUS_SHA256
from edgellm.hub import PRECISION_FILES
from edgellm.sweep import prompt_sha256

#: A card whose settings differ from these is still valid, but is not comparable
#: with the main leaderboard, so it is listed separately rather than ranked.
COMPARABLE_GEN_TOKENS = 32
COMPARABLE_EVAL_WINDOWS = 8

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "model_id",
    "revision",
    "created_utc",
    "tool_version",
    "machine",
    "rows",
    "eval_windows",
    "eval_corpus_sha256",
    "warmup_runs",
    "gen_tokens",
    "prompt_sha256",
}

REQUIRED_MACHINE = {
    "cpu",
    "arch",
    "physical_cores",
    "logical_cores",
    "ram_gb",
    "os",
    "os_release",
    "python",
    "onnxruntime",
    "provider",
    "intra_op_threads",
}

REQUIRED_ROW = {
    "precision",
    "size_mb",
    "latency_s_mean",
    "latency_s_std",
    "tokens_per_second",
    "peak_ram_mb",
    "perplexity",
    "generated_tokens",
    "measured_runs",
}

#: Loose physical bounds. Deliberately wide — the point is to catch a typo or a
#: fabricated order of magnitude, not to second-guess unusual but real hardware.
MAX_PLAUSIBLE_TOK_S = 100_000.0
MAX_PLAUSIBLE_PERPLEXITY = 1e6


@dataclass
class ValidationReport:
    path: Path
    errors: list[str]
    warnings: list[str]
    comparable: bool

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        flag = "" if self.comparable else "  (not comparable — will be listed, not ranked)"
        lines = [f"[{status}] {self.path.name}{flag}"]
        lines += [f"    error:   {e}" for e in self.errors]
        lines += [f"    warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_card(path: Path) -> ValidationReport:
    """Check one card file. Never raises for bad content — it reports."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        card = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationReport(path, [f"unreadable JSON: {exc}"], [], False)

    if not isinstance(card, dict):
        return ValidationReport(path, ["top level must be a JSON object"], [], False)

    missing = REQUIRED_TOP_LEVEL - set(card)
    if missing:
        errors.append(f"missing field(s): {', '.join(sorted(missing))}")

    if card.get("schema_version") != CARD_SCHEMA_VERSION:
        errors.append(
            f"schema_version is {card.get('schema_version')!r}, expected {CARD_SCHEMA_VERSION}. "
            "Re-run with the current edge-llm-bench."
        )

    machine = card.get("machine")
    if not isinstance(machine, dict):
        errors.append("'machine' must be an object")
    else:
        missing_machine = REQUIRED_MACHINE - set(machine)
        if missing_machine:
            errors.append(f"machine missing: {', '.join(sorted(missing_machine))}")
        if machine.get("intra_op_threads") in (0, None):
            errors.append(
                "machine.intra_op_threads is 0/null — thread count must be pinned, "
                "or throughput is not comparable. Re-run with the current version."
            )
        for leak in ("hostname", "user", "username", "mac", "ip", "serial"):
            if leak in machine:
                errors.append(f"machine.{leak} must not be submitted (identifying information)")

    rows = card.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("'rows' must be a non-empty list")
        rows = []

    seen: set[str] = set()
    scores_perplexity = False
    for index, row in enumerate(rows):
        where = f"rows[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where} must be an object")
            continue

        missing_row = REQUIRED_ROW - set(row)
        if missing_row:
            errors.append(f"{where} missing: {', '.join(sorted(missing_row))}")
            continue

        precision = row["precision"]
        if precision not in PRECISION_FILES:
            errors.append(f"{where}: unknown precision {precision!r}")
        if precision in seen:
            errors.append(f"{where}: duplicate precision {precision!r}")
        seen.add(precision)

        for field in ("size_mb", "latency_s_mean", "tokens_per_second", "peak_ram_mb"):
            value = row[field]
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"{where}.{field} must be a positive number, got {value!r}")

        if isinstance(row.get("tokens_per_second"), (int, float)):
            if row["tokens_per_second"] > MAX_PLAUSIBLE_TOK_S:
                errors.append(
                    f"{where}.tokens_per_second = {row['tokens_per_second']} is not physically "
                    f"plausible (cap {MAX_PLAUSIBLE_TOK_S:.0f})"
                )

        # Throughput must agree with latency and the token count it claims.
        lat, tps, gen = row["latency_s_mean"], row["tokens_per_second"], row["generated_tokens"]
        if all(isinstance(v, (int, float)) and v > 0 for v in (lat, tps, gen)):
            implied = gen / lat
            if abs(implied - tps) / max(implied, tps) > 0.02:
                errors.append(
                    f"{where}: tokens_per_second ({tps}) disagrees with "
                    f"generated_tokens/latency_s_mean ({implied:.2f}). "
                    "These are derived from the same measurement and must match."
                )

        if row.get("generated_tokens") != card.get("gen_tokens"):
            errors.append(
                f"{where}.generated_tokens ({row.get('generated_tokens')}) does not match "
                f"the card's gen_tokens ({card.get('gen_tokens')})"
            )

        if row.get("perplexity") is not None:
            scores_perplexity = True
            ppl = row["perplexity"]
            if not isinstance(ppl, (int, float)) or ppl <= 1.0:
                errors.append(f"{where}.perplexity must be > 1.0, got {ppl!r}")
            elif ppl > MAX_PLAUSIBLE_PERPLEXITY:
                errors.append(f"{where}.perplexity = {ppl} is implausible")

        if isinstance(row.get("latency_s_std"), (int, float)) and isinstance(lat, (int, float)):
            if lat > 0 and row["latency_s_std"] > lat * 0.5:
                warnings.append(
                    f"{where}: latency varied by more than 50% of the mean — the machine was "
                    "probably busy. Consider re-running on an idle system."
                )

    if scores_perplexity and card.get("eval_corpus_sha256") != EVAL_CORPUS_SHA256:
        errors.append(
            f"eval_corpus_sha256 {card.get('eval_corpus_sha256')!r} does not match the bundled "
            f"corpus {EVAL_CORPUS_SHA256!r}. Perplexity scored against different text is not "
            "comparable."
        )

    if card.get("prompt_sha256") != prompt_sha256():
        errors.append(
            "prompt_sha256 does not match the pinned benchmark prompt. Latency depends on "
            "prompt length, so a different prompt is not comparable."
        )

    if "fp32" not in seen and rows:
        warnings.append("no fp32 baseline in this card, so speedups cannot be computed for it")

    comparable = (
        not errors
        and card.get("gen_tokens") == COMPARABLE_GEN_TOKENS
        and (not scores_perplexity or card.get("eval_windows") == COMPARABLE_EVAL_WINDOWS)
    )
    if not comparable and not errors:
        warnings.append(
            f"settings differ from the comparable defaults "
            f"(gen_tokens={COMPARABLE_GEN_TOKENS}, eval_windows={COMPARABLE_EVAL_WINDOWS}); "
            "this card will be listed but not ranked"
        )

    return ValidationReport(path, errors, warnings, comparable)


def validate_all(paths: list[Path]) -> list[ValidationReport]:
    return [validate_card(p) for p in sorted(paths)]


def _main(argv: list[str] | None = None) -> int:
    """``python -m edgellm.validate results/community/*.json`` — the CI entry point."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m edgellm.validate")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures as well.")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])

    reports = validate_all(files)
    for report in reports:
        print(report.render())

    failed = [r for r in reports if not r.ok]
    warned = [r for r in reports if r.warnings]
    print(
        f"\n{len(reports) - len(failed)}/{len(reports)} card(s) passed"
        + (f", {len(warned)} with warnings" if warned else "")
    )
    if failed:
        return 1
    return 1 if (args.strict and warned) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
