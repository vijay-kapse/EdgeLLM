"""Resolve a (model, precision) pair to a local ONNX file, downloading if needed.

The default benchmark path deliberately consumes **pre-quantized ONNX already
published on the Hugging Face Hub** rather than quantizing locally. That is what
keeps the install light: a contributor needs ``onnxruntime`` and NumPy, not
PyTorch, ``transformers``, ``optimum`` and a GPU. It also keeps results
comparable, because every machine scores the *same bytes* of the same artifact
instead of its own locally-produced quantization.

The naming convention here is the one used across the ``onnx-community`` and
``transformers.js`` model repos, so thousands of existing models work unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Canonical precision label -> filename stem inside the repo's ``onnx/`` folder.
#: Keys are what the user types; values are the Hub convention.
PRECISION_FILES: dict[str, str] = {
    "fp32": "model",
    "fp16": "model_fp16",
    "int8": "model_int8",
    "uint8": "model_uint8",
    "q4": "model_q4",
    "q4f16": "model_q4f16",
    "bnb4": "model_bnb4",
}

#: What `--precisions` expands to when the user does not say. fp32 is the honest
#: baseline every other number is relative to; int8 and q4 are the two choices a
#: person actually weighs when shipping to a CPU.
DEFAULT_PRECISIONS: tuple[str, ...] = ("fp32", "int8", "q4")

#: Small, widely-mirrored instruct models that carry the full precision set.
#: The default is the 135M: a complete three-precision sweep stays under a GB of
#: download and finishes in minutes on a laptop CPU with no accelerator.
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


class ModelResolutionError(RuntimeError):
    """A model/precision combination is not available on the Hub."""


@dataclass(frozen=True)
class ResolvedModel:
    """A downloaded ONNX artifact, ready to hand to ONNX Runtime."""

    model_id: str
    precision: str
    onnx_path: Path
    size_mb: float
    tokenizer_path: Path

    @property
    def filename(self) -> str:
        return PRECISION_FILES[self.precision] + ".onnx"


def available_precisions(model_id: str, *, revision: str = "main") -> list[str]:
    """Return the precision labels this repo actually publishes, in canonical order."""
    from huggingface_hub import HfApi

    files = set(HfApi().list_repo_files(model_id, revision=revision))
    return [p for p, stem in PRECISION_FILES.items() if f"onnx/{stem}.onnx" in files]


def resolve(
    model_id: str,
    precision: str,
    *,
    revision: str = "main",
    cache_dir: str | None = None,
) -> ResolvedModel:
    """Download (or reuse from cache) one precision of ``model_id``.

    Raises :class:`ModelResolutionError` with the list of precisions the repo does
    publish, rather than letting a 404 surface, because "this model has no int4"
    is an ordinary situation a contributor needs to act on.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    if precision not in PRECISION_FILES:
        raise ModelResolutionError(
            f"Unknown precision '{precision}'. Known: {', '.join(PRECISION_FILES)}"
        )

    stem = PRECISION_FILES[precision]
    kwargs = {"repo_id": model_id, "revision": revision, "cache_dir": cache_dir}

    try:
        onnx_path = Path(hf_hub_download(filename=f"onnx/{stem}.onnx", **kwargs))
    except EntryNotFoundError as exc:
        have = available_precisions(model_id, revision=revision)
        raise ModelResolutionError(
            f"{model_id} does not publish '{precision}' "
            f"(looked for onnx/{stem}.onnx). Available: {', '.join(have) or 'none'}"
        ) from exc

    total_bytes = onnx_path.stat().st_size

    # Models over the 2 GB protobuf limit keep their weights in a sibling
    # ``.onnx_data`` file. ONNX Runtime needs it next to the graph, and it is most
    # of the real on-disk footprint, so it counts toward the reported size.
    try:
        data_path = Path(hf_hub_download(filename=f"onnx/{stem}.onnx_data", **kwargs))
        total_bytes += data_path.stat().st_size
        log.debug("external weights: %s", data_path)
    except EntryNotFoundError:
        pass

    tokenizer_path = Path(hf_hub_download(filename="tokenizer.json", **kwargs))

    return ResolvedModel(
        model_id=model_id,
        precision=precision,
        onnx_path=onnx_path,
        size_mb=round(total_bytes / (1024 * 1024), 2),
        tokenizer_path=tokenizer_path,
    )
