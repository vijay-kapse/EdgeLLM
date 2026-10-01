"""Measure every requested precision and assemble a result card.

**Each precision is measured in a fresh subprocess.** That is not incidental.
Peak RSS is a per-process high-water mark, and an ONNX Runtime session does not
hand all of its arenas back when it is dropped, so measuring several precisions
in one process reports the *union* of their footprints and attributes it to
whichever ran last. Isolating each one is the only way the peak-RAM column means
what it says. It also contains failures: a precision that OOMs or has no
published artifact is recorded as an error row instead of ending the sweep.

Run ``python -m edgellm.sweep --model ... --precision int8`` to execute the
single-precision worker directly; it prints one JSON object on stdout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from edgellm import __version__
from edgellm.card import Fingerprint, PrecisionRow, ResultCard, fingerprint, now_utc_iso
from edgellm.config import GenerationConfig
from edgellm.eval_lite import DEFAULT_EVAL_WINDOWS, EVAL_CORPUS_SHA256, evaluate_perplexity
from edgellm.hub import DEFAULT_PRECISIONS, ModelResolutionError, resolve

log = logging.getLogger(__name__)

#: The fixed prompt every run decodes from. Changing it changes the latency
#: numbers (prefill cost scales with its length), so its hash goes in the card.
BENCH_PROMPT = "Explain what neural network quantization is, in one short paragraph."

DEFAULT_WARMUP_RUNS = 2
DEFAULT_MEASURED_RUNS = 5
DEFAULT_GEN_TOKENS = 32

#: Above this ratio of latency std-dev to mean, the machine was too busy for the
#: number to mean much, and the report says so instead of quietly reporting it.
UNSTABLE_LATENCY_RATIO = 0.15


def warm_file_cache(path: Path, *, chunk_bytes: int = 8 << 20) -> None:
    """Read ``path`` once so the OS page cache is warm before anything is timed.

    ONNX Runtime memory-maps weights and faults them in lazily, so the first
    pass over a freshly downloaded multi-hundred-megabyte model pays disk I/O
    *and* competes with the writeback still flushing the download. Measured
    cold, an fp32 baseline came in at 17 tok/s; warm, the same machine and
    build measured 63-72 tok/s. That is a 4x error on the number every other
    row is compared against, so the cache is warmed deliberately rather than
    left to whatever state the run happens to start in.
    """
    with open(path, "rb") as handle:
        while handle.read(chunk_bytes):
            pass


def prompt_sha256(prompt: str = BENCH_PROMPT) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass
class SweepSettings:
    """Everything that affects the numbers, in one place, so it can be recorded."""

    model_id: str
    precisions: tuple[str, ...] = DEFAULT_PRECISIONS
    revision: str = "main"
    provider: str = "CPUExecutionProvider"
    intra_op_threads: int | None = None
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    measured_runs: int = DEFAULT_MEASURED_RUNS
    gen_tokens: int = DEFAULT_GEN_TOKENS
    eval_windows: int = DEFAULT_EVAL_WINDOWS
    skip_perplexity: bool = False


# --------------------------------------------------------------------- worker


def measure_one(precision: str, settings: SweepSettings) -> dict:
    """Measure a single precision in this process. Returns a serializable dict."""
    from edgellm.benchmark import _PeakRSSSampler
    from edgellm.ort_lite import OnnxLiteRunner

    resolved = resolve(
        settings.model_id,
        precision,
        revision=settings.revision,
    )
    warm_file_cache(resolved.onnx_path)
    runner = OnnxLiteRunner(
        str(resolved.onnx_path),
        str(resolved.tokenizer_path),
        provider=settings.provider,
        name=f"ort-{precision}",
        intra_op_threads=settings.intra_op_threads,
    )

    generation = GenerationConfig(max_new_tokens=settings.gen_tokens, do_sample=False)

    # Perplexity runs inside the RSS window on purpose: a full 512-token forward
    # pass is usually the memory high-water mark of the whole run, and hiding it
    # would understate what the model actually needs.
    perplexity = None
    with _PeakRSSSampler() as sampler:
        for _ in range(settings.warmup_runs):
            runner.generate(BENCH_PROMPT, generation)

        latencies, generated = [], 0
        for _ in range(settings.measured_runs):
            result = runner.generate(BENCH_PROMPT, generation)
            latencies.append(result.latency_s)
            generated = result.generated_tokens

        if not settings.skip_perplexity:
            perplexity = evaluate_perplexity(
                runner, runner.tokenizer, windows=settings.eval_windows
            ).perplexity

    latency_mean = statistics.fmean(latencies)

    # Throughput is total tokens over total time, i.e. derived from latency_s_mean,
    # not the mean of each run's own tok/s. Averaging per-run rates computes a
    # different statistic (mean of ratios, not ratio of means) and the two disagree
    # by a few percent, which would make the two columns of every card inconsistent.
    row = PrecisionRow(
        precision=precision,
        size_mb=resolved.size_mb,
        latency_s_mean=round(latency_mean, 4),
        latency_s_std=round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
        tokens_per_second=round(generated / latency_mean, 2),
        peak_ram_mb=round(sampler.peak_mb, 1),
        perplexity=perplexity,
        generated_tokens=generated,
        measured_runs=settings.measured_runs,
    )
    runner.close()
    return {"row": row.as_dict(), "threads": runner.intra_op_threads}


# ---------------------------------------------------------------- orchestrator


def _worker_command(precision: str, settings: SweepSettings) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "edgellm.sweep",
        "--model",
        settings.model_id,
        "--precision",
        precision,
        "--revision",
        settings.revision,
        "--provider",
        settings.provider,
        "--warmup-runs",
        str(settings.warmup_runs),
        "--measured-runs",
        str(settings.measured_runs),
        "--gen-tokens",
        str(settings.gen_tokens),
        "--eval-windows",
        str(settings.eval_windows),
    ]
    if settings.intra_op_threads is not None:
        cmd += ["--intra-op-threads", str(settings.intra_op_threads)]
    if settings.skip_perplexity:
        cmd.append("--skip-perplexity")
    return cmd


def run_sweep(
    settings: SweepSettings,
    *,
    on_progress=None,
    isolate: bool = True,
) -> tuple[ResultCard, dict[str, str]]:
    """Measure every precision and build a card.

    Returns ``(card, errors)`` where ``errors`` maps a precision to why it was
    skipped. A partial card is a valid result: "this model publishes no q4" is
    ordinary, and the rows that did run are still worth reporting.
    """
    rows: list[PrecisionRow] = []
    errors: dict[str, str] = {}
    threads_used: int | None = settings.intra_op_threads

    for precision in settings.precisions:
        if on_progress:
            on_progress(precision, "start")
        try:
            if isolate:
                completed = subprocess.run(
                    _worker_command(precision, settings),
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if completed.returncode != 0:
                    tail = (completed.stderr or "").strip().splitlines()
                    errors[precision] = tail[-1] if tail else f"exit {completed.returncode}"
                    if on_progress:
                        on_progress(precision, "fail")
                    continue
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
            else:
                payload = measure_one(precision, settings)

            rows.append(PrecisionRow(**payload["row"]))
            threads_used = payload.get("threads", threads_used)
            if on_progress:
                on_progress(precision, "ok")

        except ModelResolutionError as exc:
            errors[precision] = str(exc)
            if on_progress:
                on_progress(precision, "fail")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            errors[precision] = f"{type(exc).__name__}: {exc}"
            if on_progress:
                on_progress(precision, "fail")

    machine: Fingerprint = fingerprint(
        provider=settings.provider,
        intra_op_threads=threads_used or 0,
    )
    card = ResultCard(
        schema_version=1,
        model_id=settings.model_id,
        revision=settings.revision,
        created_utc=now_utc_iso(),
        tool_version=__version__,
        machine=machine,
        rows=rows,
        eval_windows=0 if settings.skip_perplexity else settings.eval_windows,
        eval_corpus_sha256="" if settings.skip_perplexity else EVAL_CORPUS_SHA256,
        warmup_runs=settings.warmup_runs,
        gen_tokens=settings.gen_tokens,
        prompt_sha256=prompt_sha256(),
    )
    return card, errors


def save_card(card: ResultCard, directory: Path) -> Path:
    """Write the card as pretty JSON under ``directory``, named for the machine."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / card.filename
    path.write_text(json.dumps(card.as_dict(), indent=2, sort_keys=True) + "\n")
    return path


def _main(argv: list[str] | None = None) -> int:
    """Single-precision worker: measure one precision, print one JSON object."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m edgellm.sweep")
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--intra-op-threads", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--measured-runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--gen-tokens", type=int, default=DEFAULT_GEN_TOKENS)
    parser.add_argument("--eval-windows", type=int, default=DEFAULT_EVAL_WINDOWS)
    parser.add_argument("--skip-perplexity", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = SweepSettings(
        model_id=args.model,
        precisions=(args.precision,),
        revision=args.revision,
        provider=args.provider,
        intra_op_threads=args.intra_op_threads,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        gen_tokens=args.gen_tokens,
        eval_windows=args.eval_windows,
        skip_perplexity=args.skip_perplexity,
    )
    print(json.dumps(measure_one(args.precision, settings)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
