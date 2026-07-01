"""Benchmark harness: latency, throughput, peak RAM, on-disk size, perplexity.

Every metric here is measured on the real machine. Nothing is estimated or
scaled. The harness is backend-agnostic: it drives any :class:`InferenceRunner`
and computes perplexity from any callable that returns logits, so PyTorch and
ONNX Runtime (FP32 or quantized) are measured identically.

Perplexity method (documented so results are reproducible): the eval text is a
slice of WikiText-2, tokenized once and split into non-overlapping windows of
``eval_max_length`` tokens. For each window we compute the summed
next-token cross-entropy; perplexity is ``exp(total_nll / total_tokens)``.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch

from edgellm.config import BenchmarkConfig, GenerationConfig
from edgellm.runners import InferenceRunner


@dataclass
class BenchmarkResult:
    """All measured numbers for one (backend, precision) combination."""

    backend: str
    precision: str
    device: str
    model_id: str
    size_mb: float | None
    latency_s_mean: float
    latency_s_std: float
    tokens_per_second: float
    peak_ram_mb: float
    perplexity: float | None
    generated_tokens: int
    measured_runs: int

    def as_dict(self) -> dict:
        return asdict(self)


class _PeakRSSSampler:
    """Context manager sampling process RSS on a background thread to find its peak."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._proc.memory_info().rss)
            time.sleep(self.interval_s)

    def __enter__(self) -> _PeakRSSSampler:
        self.peak_bytes = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)


def measure_size_mb(path: Path, patterns: tuple[str, ...]) -> float:
    """Sum the on-disk size (MB) of files under ``path`` matching any glob pattern."""
    total = 0
    for pattern in patterns:
        for f in path.glob(pattern):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 * 1024)


class PerplexityEvaluator:
    """Compute perplexity on a WikiText slice from any logits-producing callable."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def _load_text(self) -> str:
        from datasets import load_dataset

        ds = load_dataset(self.config.eval_dataset, self.config.eval_config, split="test")
        lines = [t for t in ds["text"] if t.strip()]
        return "\n\n".join(lines[: self.config.eval_num_samples])

    def evaluate(
        self,
        forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        tokenizer,
    ) -> float:
        """``forward_fn(input_ids, attention_mask) -> logits`` (all torch tensors)."""
        text = self._load_text()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        max_len = self.config.eval_max_length

        total_nll = torch.zeros((), dtype=torch.float64)
        total_tokens = 0
        for start in range(0, input_ids.shape[1], max_len):
            window = input_ids[:, start : start + max_len]
            if window.shape[1] < 2:
                continue
            attn = torch.ones_like(window)
            logits = forward_fn(window, attn).float()
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = window[:, 1:].reshape(-1).to(shift_logits.device)
            nll = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="sum")
            total_nll += nll.double().cpu()
            total_tokens += int(shift_labels.numel())

        return float(torch.exp(total_nll / total_tokens))


class BenchmarkHarness:
    """Drive a runner through warmup + measured runs and collect all metrics."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def _bench_generation_config(self, base: GenerationConfig) -> GenerationConfig:
        """Deterministic, fixed-length decoding so runs are comparable."""
        n = self.config.gen_tokens
        return GenerationConfig(
            max_new_tokens=n,
            min_new_tokens=n,  # force full length -> stable token count across backends
            do_sample=False,
            seed=base.seed,
        )

    def run(
        self,
        runner: InferenceRunner,
        *,
        precision: str,
        device: str,
        model_id: str,
        generation: GenerationConfig,
        size_mb: float | None = None,
        perplexity: float | None = None,
    ) -> BenchmarkResult:
        gen = self._bench_generation_config(generation)
        prompt = self.config.prompt

        for _ in range(self.config.warmup_runs):
            runner.generate(prompt, gen)

        latencies: list[float] = []
        throughputs: list[float] = []
        generated_tokens = 0
        with _PeakRSSSampler() as sampler:
            for _ in range(self.config.measured_runs):
                result = runner.generate(prompt, gen)
                latencies.append(result.latency_s)
                throughputs.append(result.tokens_per_second)
                generated_tokens = result.generated_tokens

        return BenchmarkResult(
            backend=runner.name,
            precision=precision,
            device=device,
            model_id=model_id,
            size_mb=round(size_mb, 2) if size_mb is not None else None,
            latency_s_mean=round(statistics.fmean(latencies), 4),
            latency_s_std=round(statistics.pstdev(latencies), 4),
            tokens_per_second=round(statistics.fmean(throughputs), 2),
            peak_ram_mb=round(sampler.peak_mb, 1),
            perplexity=round(perplexity, 3) if perplexity is not None else None,
            generated_tokens=generated_tokens,
            measured_runs=self.config.measured_runs,
        )


def save_results(results: list[BenchmarkResult], path: Path) -> None:
    """Merge results into a JSON file keyed by (backend, precision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if path.exists():
        for row in json.loads(path.read_text()):
            existing[f"{row['backend']}|{row['precision']}"] = row
    for r in results:
        existing[f"{r.backend}|{r.precision}"] = r.as_dict()
    path.write_text(json.dumps(list(existing.values()), indent=2) + "\n")


def render_markdown(results_json: Path) -> str:
    """Render the JSON results as a Markdown table (real numbers only)."""
    rows = json.loads(results_json.read_text()) if results_json.exists() else []
    header = (
        "| Backend | Precision | Device | Size (MB) | Latency (s) | "
        "Throughput (tok/s) | Peak RAM (MB) | Perplexity |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = []
    for r in rows:
        size = f"{r['size_mb']}" if r.get("size_mb") is not None else "—"
        ppl = f"{r['perplexity']}" if r.get("perplexity") is not None else "—"
        lat = f"{r['latency_s_mean']} ± {r['latency_s_std']}"
        lines.append(
            f"| {r['backend']} | {r['precision']} | {r['device']} | {size} | "
            f"{lat} | {r['tokens_per_second']} | {r['peak_ram_mb']} | {ppl} |"
        )
    return header + "\n".join(lines) + "\n"
