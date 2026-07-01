"""ONNX export via Optimum.

:class:`ONNXExporter` converts a Hugging Face causal-LM into an ONNX model
directory that ONNX Runtime (and the C++ harness in Phase 4) can load. The
exported directory also carries the tokenizer so downstream runners are
self-contained.
"""

from __future__ import annotations

import logging
from pathlib import Path

from edgellm.config import ExportConfig, ModelConfig

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export a causal-LM to ONNX with Optimum's ONNX Runtime integration."""

    def __init__(self, model_config: ModelConfig, export_config: ExportConfig) -> None:
        self.model_config = model_config
        self.export_config = export_config

    def output_dir(self) -> Path:
        """Directory the FP32 ONNX export is written to (``<out>/<model>-fp32``)."""
        safe_name = self.model_config.id.replace("/", "__")
        return Path(self.export_config.output_dir) / f"{safe_name}-fp32"

    def export(self, force: bool = False) -> Path:
        """Export the model to ONNX and return the output directory.

        If the export already exists and ``force`` is False, the existing export
        is reused (export is slow and deterministic, so this keeps iteration fast).
        """
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        out_dir = self.output_dir()
        model_file = out_dir / "model.onnx"
        if model_file.exists() and not force:
            logger.info("Reusing existing ONNX export at %s", out_dir)
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting %s to ONNX at %s", self.model_config.id, out_dir)

        model = ORTModelForCausalLM.from_pretrained(
            self.model_config.id,
            revision=self.model_config.revision,
            export=True,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        model.save_pretrained(out_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.id,
            revision=self.model_config.revision,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        tokenizer.save_pretrained(out_dir)

        logger.info("ONNX export complete: %s", out_dir)
        return out_dir
