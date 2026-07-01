"""The ``edgellm`` command-line interface (Typer)."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import typer

from edgellm import __version__
from edgellm.config import Config

app = typer.Typer(
    add_completion=False,
    help="EdgeLLM: quantize a small LLM and run it on-device with honest benchmarks.",
)

DEFAULT_CONFIG = Path("configs/default.yaml")


def _load_config(config_path: Path) -> Config:
    if config_path.exists():
        return Config.from_yaml(config_path)
    typer.echo(f"[warn] config '{config_path}' not found; using built-in defaults.", err=True)
    return Config()


@app.command()
def version() -> None:
    """Print the EdgeLLM version."""
    typer.echo(__version__)


@app.command()
def info(
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to YAML config."),
) -> None:
    """Show the resolved model + device configuration for this machine."""
    from edgellm.models import ModelLoader

    cfg = _load_config(config_path)
    loader = ModelLoader(cfg.model)
    typer.echo(f"model:   {cfg.model.id}")
    typer.echo(f"dtype:   {cfg.model.dtype}")
    typer.echo(f"device:  {loader.resolve_device()} (requested: {cfg.model.device})")
    typer.echo(f"backends: {', '.join(cfg.backends)}")


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to generate from."),
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to YAML config."),
    max_new_tokens: int | None = typer.Option(None, help="Override max new tokens."),
    temperature: float | None = typer.Option(None, help="Override sampling temperature."),
    seed: int | None = typer.Option(None, help="Override random seed."),
    greedy: bool = typer.Option(False, "--greedy", help="Disable sampling (deterministic)."),
) -> None:
    """Generate text from PROMPT with the PyTorch backend and print timing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from edgellm.models import ModelLoader
    from edgellm.runners import PyTorchRunner

    cfg = _load_config(config_path)
    if max_new_tokens is not None:
        cfg.generation.max_new_tokens = max_new_tokens
    if temperature is not None:
        cfg.generation.temperature = temperature
    if seed is not None:
        cfg.generation.seed = seed
    if greedy:
        cfg.generation.do_sample = False

    loaded = ModelLoader(cfg.model).load()
    runner = PyTorchRunner(loaded)
    result = runner.generate(prompt, cfg.generation)

    typer.echo("\n=== output ===")
    typer.echo(result.text.strip())
    typer.echo("\n=== stats ===")
    typer.echo(f"backend:          {result.backend}")
    typer.echo(f"device:           {loaded.device}")
    typer.echo(f"prompt tokens:    {result.prompt_tokens}")
    typer.echo(f"generated tokens: {result.generated_tokens}")
    typer.echo(f"latency:          {result.latency_s:.3f} s")
    typer.echo(f"throughput:       {result.tokens_per_second:.2f} tok/s")


@app.command()
def export(
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to YAML config."),
    force: bool = typer.Option(False, "--force", help="Re-export even if it already exists."),
) -> None:
    """Export the configured model to ONNX (FP32) via Optimum."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from edgellm.export import ONNXExporter

    cfg = _load_config(config_path)
    out_dir = ONNXExporter(cfg.model, cfg.export).export(force=force)
    typer.echo(f"ONNX export: {out_dir}")


@app.command()
def encode(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to tokenize."),
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to YAML config."),
) -> None:
    """Print space-separated token ids for PROMPT (feeds the C++ harness)."""
    from transformers import AutoTokenizer

    from edgellm.runners import encode_prompt

    cfg = _load_config(config_path)
    tok = AutoTokenizer.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    ids = encode_prompt(tok, prompt)["input_ids"][0].tolist()
    typer.echo(" ".join(str(i) for i in ids))


@app.command()
def decode(
    ids: str = typer.Option(..., "--ids", help="Space/comma-separated token ids to decode."),
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to YAML config."),
) -> None:
    """Decode token ids back to text (e.g. the C++ harness's GENERATED_IDS)."""
    from transformers import AutoTokenizer

    cfg = _load_config(config_path)
    tok = AutoTokenizer.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    id_list = [int(x) for x in ids.replace(",", " ").split()]
    typer.echo(tok.decode(id_list, skip_special_tokens=True))


@app.command()
def report(
    results_dir: Path = typer.Option(Path("results"), "--results", help="Results directory."),
) -> None:
    """Regenerate the Markdown table + bar charts from benchmarks.json."""
    from edgellm.benchmark import render_markdown
    from edgellm.report import render_charts

    json_path = results_dir / "benchmarks.json"
    if not json_path.exists():
        typer.echo(f"no results at {json_path}; run 'edgellm benchmark' first.", err=True)
        raise typer.Exit(1)
    (results_dir / "benchmark.md").write_text(render_markdown(json_path))
    chart = render_charts(json_path, results_dir / "benchmark_chart.png")
    typer.echo(f"wrote {results_dir / 'benchmark.md'} and {chart}")


@app.command()
def quantize(
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to YAML config."),
) -> None:
    """Produce INT8 (ONNX Runtime) and INT4 (block-wise) quantized artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from edgellm.export import ONNXExporter
    from edgellm.quantize import Quantizer

    cfg = _load_config(config_path)
    fp32_dir = ONNXExporter(cfg.model, cfg.export).export()
    q = Quantizer(cfg.quantize)
    base = fp32_dir.parent / fp32_dir.name.replace("-fp32", "")
    int8_dir = q.ort_dynamic_int8(fp32_dir, Path(f"{base}-int8-dynamic"))
    int4_dir = q.ort_int4(fp32_dir, Path(f"{base}-int4"))
    typer.echo(f"INT8: {int8_dir}")
    typer.echo(f"INT4: {int4_dir}")
    typer.echo("\nINT4 backend availability on this machine:")
    for backend, state in q.int4_availability().items():
        typer.echo(f"  {backend}: {state}")


@app.command()
def benchmark(
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to YAML config."),
    skip_ppl: bool = typer.Option(False, "--skip-ppl", help="Skip perplexity (faster)."),
    results_dir: Path = typer.Option(Path("results"), "--results", help="Output directory."),
    only: str = typer.Option(
        "pt-fp32,ort-fp32,ort-int8,ort-int4,pt-int8",
        "--only",
        help="Comma list of backends to benchmark.",
    ),
) -> None:
    """Benchmark FP32/INT8/INT4 across PyTorch + ONNX Runtime and write real numbers."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from edgellm.benchmark import (
        BenchmarkHarness,
        PerplexityEvaluator,
        measure_size_mb,
        render_markdown,
        save_results,
    )
    from edgellm.export import ONNXExporter
    from edgellm.models import ModelLoader
    from edgellm.quantize import Quantizer
    from edgellm.runners import ORTRunner, PyTorchRunner

    cfg = _load_config(config_path)
    selected = {s.strip() for s in only.split(",") if s.strip()}
    harness = BenchmarkHarness(cfg.benchmark)
    ppl_eval = None if skip_ppl else PerplexityEvaluator(cfg.benchmark)
    json_path = results_dir / "benchmarks.json"

    loaded = ModelLoader(cfg.model).load()  # FP32 on the best local device (mps/cuda/cpu)
    fp32_dir = ONNXExporter(cfg.model, cfg.export).export()
    quantizer = Quantizer(cfg.quantize)
    base = fp32_dir.parent / fp32_dir.name.replace("-fp32", "")

    def bench_ort(model_dir: Path, precision: str, name: str):
        runner = ORTRunner(str(model_dir), loaded.tokenizer, name=name)
        ppl = _ppl_ort(runner, ppl_eval, loaded.tokenizer) if ppl_eval else None
        size = measure_size_mb(model_dir, ("*.onnx", "*.onnx_data"))
        return harness.run(
            runner,
            precision=precision,
            device="cpu",
            model_id=cfg.model.id,
            generation=cfg.generation,
            size_mb=size,
            perplexity=ppl,
        )

    def bench_pt_fp32():
        ppl = (
            _ppl_torch(loaded.model, loaded.device, ppl_eval, loaded.tokenizer)
            if ppl_eval
            else None
        )
        return harness.run(
            PyTorchRunner(loaded),
            precision="fp32",
            device=loaded.device,
            model_id=cfg.model.id,
            generation=cfg.generation,
            size_mb=_pytorch_size_mb(cfg.model.id, cfg.model.revision),
            perplexity=ppl,
        )

    def bench_pt_int8():
        cpu_loaded = ModelLoader(replace(cfg.model, device="cpu")).load()
        qmodel = Quantizer.pytorch_dynamic_int8(cpu_loaded.model)
        cpu_loaded = replace(cpu_loaded, model=qmodel, device="cpu")
        ppl = _ppl_torch(qmodel, "cpu", ppl_eval, cpu_loaded.tokenizer) if ppl_eval else None
        return harness.run(
            PyTorchRunner(cpu_loaded),
            precision="int8",
            device="cpu",
            model_id=cfg.model.id,
            generation=cfg.generation,
            size_mb=None,
            perplexity=ppl,
        )

    # (key, human label, thunk). Each runs guarded so one failure can't discard the rest.
    steps = [
        ("pt-fp32", "pytorch fp32", bench_pt_fp32),
        ("ort-fp32", "ort-cpu fp32", lambda: bench_ort(fp32_dir, "fp32", "ort-cpu")),
        (
            "ort-int8",
            "ort-cpu int8",
            lambda: bench_ort(
                quantizer.ort_dynamic_int8(fp32_dir, Path(f"{base}-int8-dynamic")),
                "int8",
                "ort-cpu-int8",
            ),
        ),
        (
            "ort-int4",
            "ort-cpu int4",
            lambda: bench_ort(
                quantizer.ort_int4(fp32_dir, Path(f"{base}-int4")), "int4", "ort-cpu-int4"
            ),
        ),
        ("pt-int8", "pytorch int8 (cpu)", bench_pt_int8),
    ]

    for key, label, thunk in steps:
        if key not in selected:
            continue
        typer.echo(f"[bench] {label}...")
        try:
            result = thunk()
        except Exception as exc:  # noqa: BLE001 - one backend must not sink the others
            typer.echo(f"[bench] {label} FAILED: {type(exc).__name__}: {exc}", err=True)
            continue
        save_results([result], json_path)  # persist incrementally

    md = render_markdown(json_path)
    (results_dir / "benchmark.md").write_text(md)
    typer.echo("\n=== results ===")
    typer.echo(md)
    typer.echo(f"wrote {json_path} and {results_dir / 'benchmark.md'}")


def _ppl_torch(model, device: str, ppl_eval, tokenizer) -> float:
    import torch

    def forward(ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out = model(input_ids=ids.to(device), attention_mask=attn.to(device))
        return out.logits.detach().to("cpu")

    return ppl_eval.evaluate(forward, tokenizer)


def _ppl_ort(runner, ppl_eval, tokenizer) -> float:
    import torch

    def forward(ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        out = runner.model(input_ids=ids, attention_mask=attn)
        return out.logits.detach().to("cpu")

    return ppl_eval.evaluate(forward, tokenizer)


def _pytorch_size_mb(model_id: str, revision: str) -> float | None:
    """On-disk size (MB) of the model weights in the local HF cache."""
    from huggingface_hub import snapshot_download

    from edgellm.benchmark import measure_size_mb

    try:
        snap = Path(
            snapshot_download(
                model_id, revision=revision, allow_patterns=["*.safetensors", "*.bin"]
            )
        )
    except Exception:
        return None
    size = measure_size_mb(snap, ("*.safetensors", "*.bin"))
    return size or None


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
