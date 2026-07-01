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


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
