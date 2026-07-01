"""Tests for the typed config loader (no heavy ML deps required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgellm.config import Config, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_sane() -> None:
    cfg = Config()
    assert cfg.model.id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.generation.max_new_tokens > 0
    assert "pytorch" in cfg.backends


def test_load_default_yaml() -> None:
    cfg = Config.from_yaml(REPO_ROOT / "configs" / "default.yaml")
    assert isinstance(cfg.model, ModelConfig)
    assert cfg.model.id
    assert cfg.quantize.int4.method in {"gptq", "awq", "gguf"}
    assert cfg.benchmark.measured_runs >= 1


def test_unknown_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  not_a_field: 1\n")
    with pytest.raises(ValueError, match="Unknown config key"):
        Config.from_yaml(bad)


def test_nested_override(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("model:\n  id: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n  device: cpu\n")
    cfg = Config.from_yaml(cfg_file)
    assert cfg.model.id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert cfg.model.device == "cpu"
    # Untouched fields keep their defaults.
    assert cfg.model.dtype == "float32"
