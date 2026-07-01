"""Quantization: PyTorch INT8 and ONNX Runtime INT8 / INT4.

Plain-English summary of what happens here (the README expands on this):

* **INT8 (8-bit)** stores each weight as an 8-bit integer plus a floating-point
  *scale*, cutting size ~4x vs FP32. **Dynamic** quantization keeps activations
  in float and quantizes weights only; per-token activation scales are computed
  on the fly at inference (no calibration data needed) — the robust default for
  transformers.
* **INT4 (4-bit)** packs weights into 4 bits using *block-wise* quantization:
  each contiguous block of ``block_size`` weights shares one scale (and, when
  asymmetric, a zero-point). Smaller blocks = better accuracy, more overhead.

On CUDA hardware, GPTQ/AWQ give higher-quality INT4; they are unavailable on
this Apple-Silicon machine (both require CUDA), so the INT4 path here uses ONNX
Runtime's block-wise weight-only quantizer, which runs on CPU. See
:meth:`Quantizer.int4_availability` for the honest state of each backend.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from edgellm.config import QuantizeConfig

logger = logging.getLogger(__name__)

# Non-weight files copied alongside a quantized model.onnx so an ORT runner can
# load the directory standalone (tokenizer, config, chat template, ...).
_SIDE_CAR_GLOBS = (
    "*.json",
    "*.txt",
    "*.jinja",
    "*.model",
)


def _copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in _SIDE_CAR_GLOBS:
        for f in src_dir.glob(pattern):
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)


class Quantizer:
    """Produce quantized model artifacts from an exported FP32 ONNX directory."""

    def __init__(self, config: QuantizeConfig) -> None:
        self.config = config

    # --- ONNX Runtime INT8 (dynamic, weight-only) ---
    def ort_dynamic_int8(self, fp32_dir: Path, out_dir: Path) -> Path:
        """Dynamically quantize ONNX weights to INT8. Returns the output dir."""
        from onnxruntime.quantization import QuantType, quantize_dynamic

        out_dir.mkdir(parents=True, exist_ok=True)
        model_out = out_dir / "model.onnx"
        logger.info("ORT dynamic INT8: %s -> %s", fp32_dir, model_out)
        quantize_dynamic(
            model_input=str(fp32_dir / "model.onnx"),
            model_output=str(model_out),
            weight_type=QuantType.QInt8,
            per_channel=self.config.int8.per_channel,
        )
        _copy_sidecars(fp32_dir, out_dir)
        return out_dir

    # --- ONNX Runtime INT4 (block-wise, weight-only) ---
    def ort_int4(self, fp32_dir: Path, out_dir: Path) -> Path:
        """Block-wise weight-only INT4 quantization (native ORT quantizer).

        The model is passed *by path* so ONNX Runtime streams the >2GB external
        data instead of requiring it all in a single in-memory ModelProto.
        """
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        out_dir.mkdir(parents=True, exist_ok=True)
        model_out = out_dir / "model.onnx"
        logger.info("ORT INT4 (block_size=%d): %s", self.config.int4.group_size, model_out)

        quantizer = MatMulNBitsQuantizer(
            str(fp32_dir / "model.onnx"),
            block_size=self.config.int4.group_size,
            is_symmetric=True,
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(model_out), use_external_data_format=False)
        _copy_sidecars(fp32_dir, out_dir)
        return out_dir

    # --- PyTorch INT8 (dynamic) ---
    @staticmethod
    def pytorch_dynamic_int8(model):
        """Dynamically quantize ``nn.Linear`` layers of a torch model to INT8 (CPU).

        Selects the ``qnnpack`` engine, which is the INT8 kernel backend available
        on ARM (Apple Silicon); x86 would use ``fbgemm``.
        """
        import torch
        from torch.ao.quantization import quantize_dynamic

        engines = torch.backends.quantized.supported_engines
        if "qnnpack" in engines:
            torch.backends.quantized.engine = "qnnpack"
        elif "fbgemm" in engines:
            torch.backends.quantized.engine = "fbgemm"

        model = model.to("cpu").eval()
        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    @staticmethod
    def int4_availability() -> dict[str, str]:
        """Report which INT4 backends can actually run on this machine."""
        import importlib.util

        import torch

        cuda = torch.cuda.is_available()
        return {
            "gptq (auto-gptq)": (
                "available"
                if (cuda and importlib.util.find_spec("auto_gptq"))
                else "unavailable (needs CUDA)"
            ),
            "awq (autoawq)": (
                "available"
                if (cuda and importlib.util.find_spec("awq"))
                else "unavailable (needs CUDA)"
            ),
            "onnxruntime block-wise INT4": "available (CPU)",
        }
