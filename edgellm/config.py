"""Typed configuration objects loaded from YAML.

The whole project is driven by a single :class:`Config` tree so that every phase
(loading, export, quantization, benchmarking) reads its settings from one place
and the CLI can override any field.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    """Which model to load and how to place it on hardware."""

    id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    dtype: str = "float32"
    device: str = "auto"
    trust_remote_code: bool = False


@dataclass
class GenerationConfig:
    """Text-generation decoding parameters."""

    max_new_tokens: int = 128
    min_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    seed: int = 0


@dataclass
class BenchmarkConfig:
    """Settings for the benchmark harness (used from Phase 2 onward)."""

    warmup_runs: int = 2
    measured_runs: int = 5
    gen_tokens: int = 64
    prompt: str = "Explain what neural network quantization is, in one short paragraph."
    eval_dataset: str = "wikitext"
    eval_config: str = "wikitext-2-raw-v1"
    eval_num_samples: int = 32
    eval_max_length: int = 512


@dataclass
class ExportConfig:
    """ONNX export settings (used from Phase 2 onward)."""

    opset: int = 17
    output_dir: str = "artifacts/onnx"


@dataclass
class Int8Config:
    scheme: str = "dynamic"
    per_channel: bool = True


@dataclass
class Int4Config:
    method: str = "gptq"
    group_size: int = 128


@dataclass
class QuantizeConfig:
    """Quantization settings (used from Phase 3 onward)."""

    int8: Int8Config = field(default_factory=Int8Config)
    int4: Int4Config = field(default_factory=Int4Config)


@dataclass
class Config:
    """Root configuration tree."""

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    backends: list[str] = field(default_factory=lambda: ["pytorch", "ort-cpu"])

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a :class:`Config` from a YAML file, filling in defaults."""
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
        return _from_dict(cls, raw)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Unknown keys raise, so typos in the YAML fail loudly instead of being ignored.
    """
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    # Resolve annotations to real types (needed because `from __future__ import
    # annotations` turns dataclass field types into strings).
    hints = get_type_hints(cls)
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        field_type = hints[key]
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _from_dict(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)
