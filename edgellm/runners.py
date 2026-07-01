"""Inference backends.

Every backend implements the same :class:`InferenceRunner` interface so the
benchmark harness can treat PyTorch, ONNX Runtime (CPU/GPU) and the Qualcomm QNN
NPU path identically. Phase 1 ships the PyTorch backend; later phases add the
ONNX Runtime and QNN runners.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import torch

from edgellm.config import GenerationConfig
from edgellm.models import LoadedModel


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


class PyTorchRunner(InferenceRunner):
    """Eager PyTorch generation — the correctness/quality reference backend."""

    name = "pytorch"

    def __init__(self, loaded: LoadedModel) -> None:
        self.loaded = loaded

    def _build_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        """Apply the chat template for instruct models, else tokenize raw text.

        Returns a dict of tensors (``input_ids`` + ``attention_mask``) already
        placed on the model's device.
        """
        tokenizer = self.loaded.tokenizer
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.loaded.device) for k, v in inputs.items()}

    @torch.inference_mode()
    def generate(self, prompt: str, generation: GenerationConfig) -> GenerationResult:
        torch.manual_seed(generation.seed)
        tokenizer = self.loaded.tokenizer
        inputs = self._build_inputs(prompt)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        start = time.perf_counter()
        output_ids = self.loaded.model.generate(
            **inputs,
            max_new_tokens=generation.max_new_tokens,
            do_sample=generation.do_sample,
            temperature=generation.temperature,
            top_p=generation.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        if self.loaded.device == "mps":
            torch.mps.synchronize()
        elif self.loaded.device == "cuda":
            torch.cuda.synchronize()
        latency_s = time.perf_counter() - start

        new_ids = output_ids[0, prompt_tokens:]
        generated_tokens = int(new_ids.shape[-1])
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        tokens_per_second = generated_tokens / latency_s if latency_s > 0 else 0.0

        return GenerationResult(
            backend=self.name,
            prompt=prompt,
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_s=latency_s,
            tokens_per_second=tokens_per_second,
        )
