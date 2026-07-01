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


def encode_prompt(tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    """Turn a prompt into model inputs (CPU tensors, not yet placed on a device).

    Uses the chat template for instruct models; falls back to plain tokenization
    for base models. Shared by every backend so all runners see identical inputs.
    """
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    return dict(tokenizer(prompt, return_tensors="pt"))


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
        """Encode ``prompt`` and place the tensors on the model's device."""
        inputs = encode_prompt(self.loaded.tokenizer, prompt)
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
            min_new_tokens=generation.min_new_tokens,
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


class ORTRunner(InferenceRunner):
    """ONNX Runtime backend (loads an exported/quantized ONNX model directory).

    Used for the FP32 ONNX baseline (Phase 2) and every quantized ONNX model
    (Phase 3). The execution provider selects the hardware: ``CPUExecutionProvider``
    for CPU, ``CoreMLExecutionProvider`` on Apple, ``QNNExecutionProvider`` for the
    Qualcomm NPU (Phase 5).
    """

    def __init__(
        self,
        model_dir: str,
        tokenizer,
        provider: str = "CPUExecutionProvider",
        name: str = "ort-cpu",
    ) -> None:
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForCausalLM

        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"Execution provider '{provider}' is not available in this ONNX "
                f"Runtime build (have: {available}). For the Qualcomm NPU, install "
                "onnxruntime-qnn on a Snapdragon host, or use aihub/run_on_snapdragon.py."
            )
        self.name = name
        self.tokenizer = tokenizer
        self.model = ORTModelForCausalLM.from_pretrained(model_dir, provider=provider)


class QNNRunner(ORTRunner):
    """ONNX Runtime QNN Execution Provider runner (Qualcomm NPU).

    Requires an ``onnxruntime-qnn`` build on a Snapdragon host; on other platforms
    construction raises a clear error. For laptop-side profiling of real Snapdragon
    hardware without a device in hand, use ``aihub/run_on_snapdragon.py`` instead.
    """

    def __init__(self, model_dir: str, tokenizer, name: str = "ort-qnn") -> None:
        super().__init__(model_dir, tokenizer, provider="QNNExecutionProvider", name=name)

    def generate(self, prompt: str, generation: GenerationConfig) -> GenerationResult:
        torch.manual_seed(generation.seed)
        inputs = encode_prompt(self.tokenizer, prompt)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        start = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=generation.max_new_tokens,
            min_new_tokens=generation.min_new_tokens,
            do_sample=generation.do_sample,
            temperature=generation.temperature,
            top_p=generation.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        latency_s = time.perf_counter() - start

        new_ids = output_ids[0, prompt_tokens:]
        generated_tokens = int(new_ids.shape[-1])
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
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
