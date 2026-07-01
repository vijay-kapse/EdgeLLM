"""Tests for benchmark IO and helpers (no model download required)."""

from __future__ import annotations

from pathlib import Path

from edgellm.benchmark import (
    BenchmarkResult,
    measure_size_mb,
    render_markdown,
    save_results,
)


def _fake_result(backend: str, precision: str) -> BenchmarkResult:
    return BenchmarkResult(
        backend=backend,
        precision=precision,
        device="cpu",
        model_id="dummy/model",
        size_mb=123.45,
        latency_s_mean=1.0,
        latency_s_std=0.1,
        tokens_per_second=64.0,
        peak_ram_mb=512.0,
        perplexity=18.5,
        generated_tokens=64,
        measured_runs=5,
    )


def test_save_merges_by_backend_precision(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    save_results([_fake_result("pytorch", "fp32")], path)
    save_results([_fake_result("ort-cpu", "int8")], path)
    # Re-saving the same key overwrites rather than duplicating.
    save_results([_fake_result("pytorch", "fp32")], path)

    md = render_markdown(path)
    assert md.count("| pytorch | fp32 |") == 1
    assert "| ort-cpu | int8 |" in md


def test_measure_size_mb(tmp_path: Path) -> None:
    (tmp_path / "a.onnx").write_bytes(b"\0" * (1024 * 1024))
    (tmp_path / "b.txt").write_bytes(b"\0" * (1024 * 1024))
    size = measure_size_mb(tmp_path, ("*.onnx",))
    assert 0.99 < size < 1.01  # only the .onnx file counts
