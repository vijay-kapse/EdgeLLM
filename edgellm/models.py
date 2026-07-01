"""Model + tokenizer loading.

:class:`ModelLoader` wraps Hugging Face ``transformers`` so the rest of the code
never touches ``AutoModelForCausalLM`` directly and device/dtype resolution lives
in exactly one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from edgellm.config import ModelConfig

logger = logging.getLogger(__name__)

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class LoadedModel:
    """A loaded model bundled with its tokenizer and resolved placement."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str
    dtype: torch.dtype


class ModelLoader:
    """Load a causal-LM + tokenizer from a :class:`ModelConfig`."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def resolve_device(self) -> str:
        """Turn ``device: auto`` into a concrete device string for this machine."""
        requested = self.config.device
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def resolve_dtype(self) -> torch.dtype:
        try:
            return _DTYPES[self.config.dtype]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported dtype '{self.config.dtype}'. Choose one of {list(_DTYPES)}."
            ) from exc

    def load(self) -> LoadedModel:
        """Load the model and tokenizer and move them onto the resolved device."""
        device = self.resolve_device()
        dtype = self.resolve_dtype()
        logger.info("Loading %s (dtype=%s, device=%s)", self.config.id, self.config.dtype, device)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.id,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            # Small decoder-only models often ship without a pad token.
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.id,
            revision=self.config.revision,
            dtype=dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        model.to(device)
        model.eval()

        return LoadedModel(model=model, tokenizer=tokenizer, device=device, dtype=dtype)
