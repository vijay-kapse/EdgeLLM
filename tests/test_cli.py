"""Smoke tests for the CLI wiring that do not download a model."""

from __future__ import annotations

from typer.testing import CliRunner

from edgellm import __version__
from edgellm.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("generate", "info", "version"):
        assert command in result.stdout
