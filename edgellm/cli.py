"""The ``edgellm`` command-line interface (Typer)."""

from __future__ import annotations

import logging
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
def benchmark(
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to YAML config."),
    skip_ppl: bool = typer.Option(False, "--skip-ppl", help="Skip perplexity (faster)."),
    results_dir: Path = typer.Option(Path("results"), "--results", help="Output directory."),
) -> None:
    """Benchmark the FP32 baseline (PyTorch + ONNX Runtime) and write real numbers."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch

    from edgellm.benchmark import (
        BenchmarkHarness,
        PerplexityEvaluator,
        measure_size_mb,
        render_markdown,
        save_results,
    )
    from edgellm.export import ONNXExporter
    from edgellm.models import ModelLoader
    from edgellm.runners import ORTRunner, PyTorchRunner

    cfg = _load_config(config_path)
    harness = BenchmarkHarness(cfg.benchmark)
    ppl_eval = None if skip_ppl else PerplexityEvaluator(cfg.benchmark)
    results = []

    # --- PyTorch FP32 baseline ---
    loaded = ModelLoader(cfg.model).load()
    pt_ppl = None
    if ppl_eval is not None:
        typer.echo("[ppl] scoring PyTorch model...")

        def torch_forward(ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                out = loaded.model(
                    input_ids=ids.to(loaded.device), attention_mask=attn.to(loaded.device)
                )
            return out.logits.detach().to("cpu")

        pt_ppl = ppl_eval.evaluate(torch_forward, loaded.tokenizer)

    pt_size = _pytorch_size_mb(cfg.model.id, cfg.model.revision)
    typer.echo("[bench] PyTorch FP32...")
    results.append(
        harness.run(
            PyTorchRunner(loaded),
            precision="fp32",
            device=loaded.device,
            model_id=cfg.model.id,
            generation=cfg.generation,
            size_mb=pt_size,
            perplexity=pt_ppl,
        )
    )

    # --- ONNX Runtime FP32 baseline ---
    out_dir = ONNXExporter(cfg.model, cfg.export).export()
    ort_runner = ORTRunner(str(out_dir), loaded.tokenizer, name="ort-cpu")
    ort_ppl = None
    if ppl_eval is not None:
        typer.echo("[ppl] scoring ONNX Runtime model...")

        def ort_forward(ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
            out = ort_runner.model(input_ids=ids, attention_mask=attn)
            return out.logits.detach().to("cpu")

        ort_ppl = ppl_eval.evaluate(ort_forward, loaded.tokenizer)

    ort_size = measure_size_mb(out_dir, ("*.onnx", "*.onnx_data"))
    typer.echo("[bench] ONNX Runtime FP32 (CPU)...")
    results.append(
        harness.run(
            ort_runner,
            precision="fp32",
            device="cpu",
            model_id=cfg.model.id,
            generation=cfg.generation,
            size_mb=ort_size,
            perplexity=ort_ppl,
        )
    )

    json_path = results_dir / "benchmarks.json"
    save_results(results, json_path)
    md = render_markdown(json_path)
    (results_dir / "benchmark.md").write_text(md)

    typer.echo("\n=== results ===")
    typer.echo(md)
    typer.echo(f"wrote {json_path} and {results_dir / 'benchmark.md'}")


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
