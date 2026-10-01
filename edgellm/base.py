"""Backend-agnostic interfaces, shared by the torch and the torch-free paths.

These used to live in :mod:`edgellm.runners`, which imports ``torch`` at module
level. The lightweight benchmark path (ONNX Runtime + NumPy, no torch, no
transformers) needs the same interface without paying for that import, so the
parts that carry no torch dependency live here and ``runners`` re-exports them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from edgellm.config import GenerationConfig


@dataclass
class GenerationResult:
    """The text plus the timing/throughput numbers for one generation call."""

    backend: str
    prompt: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_s: float
    tokens_per_second: float

    def as_dict(self) -> dict:
        return asdict(self)


class InferenceRunner(ABC):
    """Common interface for all inference backends."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, generation: GenerationConfig) -> GenerationResult:
        """Generate a completion for ``prompt`` and report timing."""

    def close(self) -> None:
        """Release whatever the backend holds (sessions, memory). Optional."""
        return None
