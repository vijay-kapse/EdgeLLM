"""The ``run`` / ``submit`` / ``leaderboard`` commands — the lightweight path.

Registered onto the main Typer app by :mod:`edgellm.cli`. Kept in its own module
because this is the path a contributor touches, and it should be readable without
scrolling past the export/quantize authoring commands.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from edgellm.eval_lite import DEFAULT_EVAL_WINDOWS
from edgellm.hub import DEFAULT_MODEL, DEFAULT_PRECISIONS
from edgellm.sweep import (
    DEFAULT_GEN_TOKENS,
    DEFAULT_MEASURED_RUNS,
    DEFAULT_WARMUP_RUNS,
)

COMMUNITY_DIR = Path("results/community")


def _parse_precisions(value: str) -> tuple[str, ...]:
    items = tuple(p.strip() for p in value.split(",") if p.strip())
    if not items:
        raise typer.BadParameter("Give at least one precision, e.g. --precisions fp32,int8,q4")
    return items


def register(app: typer.Typer) -> None:
    """Attach the lightweight commands to ``app``."""

    @app.command()
    def run(
        model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Hub model id with ONNX."),
        precisions: str = typer.Option(
            ",".join(DEFAULT_PRECISIONS), "--precisions", help="Comma-separated precision list."
        ),
        revision: str = typer.Option("main", "--revision", help="Hub revision to pin."),
        provider: str = typer.Option(
            "CPUExecutionProvider", "--provider", help="ONNX Runtime execution provider."
        ),
        threads: int | None = typer.Option(
            None, "--threads", help="intra-op threads (default: physical core count)."
        ),
        gen_tokens: int = typer.Option(
            DEFAULT_GEN_TOKENS, "--gen-tokens", help="Tokens to generate per run."
        ),
        measured_runs: int = typer.Option(
            DEFAULT_MEASURED_RUNS, "--measured-runs", help="Timed runs per precision."
        ),
        warmup_runs: int = typer.Option(
            DEFAULT_WARMUP_RUNS, "--warmup-runs", help="Untimed warmup runs."
        ),
        eval_windows: int = typer.Option(
            DEFAULT_EVAL_WINDOWS, "--eval-windows", help="512-token perplexity windows."
        ),
        skip_perplexity: bool = typer.Option(
            False, "--skip-perplexity", help="Measure speed and size only (much faster)."
        ),
        out: Path = typer.Option(
            COMMUNITY_DIR, "--out", help="Directory to write the result card into."
        ),
        no_isolate: bool = typer.Option(
            False,
            "--no-isolate",
            help="Measure all precisions in one process (faster, but peak-RAM becomes unreliable).",
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    ) -> None:
        """Benchmark pre-quantized ONNX precisions of MODEL on this machine.

        Downloads already-quantized artifacts from the Hub — nothing is quantized
        locally, so this needs no PyTorch and no GPU.
        """
        from edgellm.render import console_report
        from edgellm.sweep import SweepSettings, run_sweep, save_card

        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
        )

        settings = SweepSettings(
            model_id=model,
            precisions=_parse_precisions(precisions),
            revision=revision,
            provider=provider,
            intra_op_threads=threads,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            gen_tokens=gen_tokens,
            eval_windows=eval_windows,
            skip_perplexity=skip_perplexity,
        )

        typer.echo(f"Benchmarking {model} on this machine ({len(settings.precisions)} precisions).")
        typer.echo("First run downloads the ONNX artifacts; later runs reuse the HF cache.\n")

        def progress(precision: str, state: str) -> None:
            if state == "start":
                typer.echo(f"  {precision:>6} ... ", nl=False)
            elif state == "ok":
                typer.echo("done")
            else:
                typer.echo("skipped")

        card, errors = run_sweep(settings, on_progress=progress, isolate=not no_isolate)

        if not card.rows:
            typer.echo("\nNo precision could be measured. Reasons:", err=True)
            for precision, why in errors.items():
                typer.echo(f"  {precision}: {why}", err=True)
            raise typer.Exit(1)

        typer.echo(console_report(card, errors))

        path = save_card(card, out)
        typer.echo(f"Result card: {path}")
        typer.echo("Share it on the public leaderboard with:  edge-llm-bench submit")

    @app.command()
    def models(
        model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Hub model id to inspect."),
        revision: str = typer.Option("main", "--revision"),
    ) -> None:
        """List which ONNX precisions a Hub model actually publishes."""
        from edgellm.hub import available_precisions

        found = available_precisions(model, revision=revision)
        if not found:
            typer.echo(
                f"{model} publishes no ONNX files under onnx/. Try a model from "
                "https://huggingface.co/onnx-community",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"{model} publishes: {', '.join(found)}")

    @app.command()
    def leaderboard(
        community: Path = typer.Option(COMMUNITY_DIR, "--dir", help="Directory of result cards."),
        out: Path = typer.Option(
            Path("results/LEADERBOARD.md"), "--out", help="Markdown file to write."
        ),
        json_out: Path | None = typer.Option(
            Path("site/leaderboard.json"), "--json-out", help="JSON for the site to render."
        ),
    ) -> None:
        """Rebuild the leaderboard from every card in the community directory."""
        from edgellm.leaderboard import build_leaderboard, render_leaderboard_markdown

        cards = sorted(community.glob("*.json"))
        if not cards:
            typer.echo(f"No result cards found in {community}.", err=True)
            raise typer.Exit(1)

        board = build_leaderboard(cards)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_leaderboard_markdown(board))
        typer.echo(
            f"Wrote {out} from {len(cards)} card(s), {len(board['entries'])} machine-model rows."
        )

        if json_out is not None:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(board, indent=2) + "\n")
            typer.echo(f"Wrote {json_out}")

    @app.command()
    def submit(
        card: Path | None = typer.Option(
            None, "--card", help="Card to submit (default: the newest in results/community)."
        ),
        community: Path = typer.Option(COMMUNITY_DIR, "--dir"),
        name: str = typer.Option("", "--name", help="Credit line for the leaderboard (optional)."),
        notes: str = typer.Option("", "--notes", help="Anything unusual about this machine."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print, do not push."),
    ) -> None:
        """Open a pull request adding your result card to the public leaderboard."""
        from edgellm.submit import submit_card

        target = card
        if target is None:
            cards = sorted(community.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if not cards:
                typer.echo(
                    f"No result card in {community}. Run `edge-llm-bench run` first.", err=True
                )
                raise typer.Exit(1)
            target = cards[-1]

        submit_card(target, name=name, notes=notes, dry_run=dry_run, echo=typer.echo)
